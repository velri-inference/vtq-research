"""Fused runtime for the offline trellis-distortion teacher. No predictor in the file."""
import argparse
from contextlib import ExitStack,contextmanager
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice
from src import vtq
from src.exp09_group_order import run_model_ab


@triton.jit
def _predict(X,B,FAC,STEP,O,U,NB:tl.constexpr,S0:tl.constexpr,S1:tl.constexpr,SC:tl.constexpr,
             STATIC:tl.constexpr,SHRINK:tl.constexpr,ROUND:tl.constexpr):
    pid=tl.program_id(0);j=tl.arange(0,256);r=pid//NB; group=pid%NB
    x=tl.load(X+r*S0+(group*256+j)*S1); rms=tl.sqrt(tl.sum(x*x,0)/256.)
    if STATIC>0:
        factor=STATIC
    else:
        z=x/tl.maximum(rms,1.e-20);s=tl.sqrt(tl.sum(tl.reshape(z*z,(16,16)),1)/16.)
        zone=(s>.82).to(tl.int32)+(s>1.).to(tl.int32)+(s>1.18).to(tl.int32)
        bins=(tl.minimum((tl.abs(z)*16).to(tl.int32),63)+64*tl.reshape(tl.broadcast_to(zone[:,None],(16,16)),(256,)))
        subbin=256+tl.minimum((s*16).to(tl.int32),31)
        best=float('inf');which=0
        for c in tl.static_range(7):
            row=B+c*289
            val=tl.sum(tl.load(row+bins),0)/256.+tl.sum(tl.load(row+subbin),0)/16.+tl.load(row+288)
            which=tl.where(val<best,c,which);best=tl.minimum(best,val)
        factor=1.+SHRINK*(tl.load(FAC+which)-1.)
    target=tl.maximum(rms,1.e-10)*SC*factor
    if ROUND:
        step=tl.load(STEP+r);u=tl.maximum(1.,tl.minimum(255.,libdevice.rint(tl.div_rn(target,step))))
        tl.store(U+r,u);tl.store(O+r,tl.maximum(1.e-10,step*u))
    else:tl.store(O+pid,target)


def experiment(path):
    raw=json.loads(Path(path).read_text());mult=torch.tensor(raw['mult'],device='cuda')
    bank={sp:torch.tensor(v['weights'],device='cuda').T.contiguous() for sp,v in raw['specs'].items()}
    @contextmanager
    def ctx(mode):
        if mode=='natural':yield;return
        def call(x,sc,sp,step=None):
            nb=x.shape[1]//256; r=x.shape[0]; out=torch.empty((r,nb) if step is None else (r,),device=x.device);u=out if step is None else torch.empty_like(out)
            static=raw['specs'][sp]['static_factor'] if mode=='static' else -1.
            _predict[(r*nb,)](x,bank[sp],mult,step if step is not None else out,out,u,nb,*x.stride(),float(sc),static,.5 if mode=='adaptive-half' else 1.,step is not None,enable_fp_fusion=False)
            return out if step is None else (u,out)
        with ExitStack() as st:
            st.enter_context(patch.object(vtq,'_initial_tile_scales',lambda w,sc,sp:call(w,sc,sp)))
            st.enter_context(patch.object(vtq,'_initial_block_u8',lambda w,sc,step,sp:call(w,sc,sp,step)))
            yield
    return ctx


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--model',required=True);ap.add_argument('--mode',choices=['discover','confirm'],required=True);ap.add_argument('--text',type=Path,default=Path('data/wikitext2_test.raw'));ap.add_argument('--output',type=Path,required=True);ap.add_argument('--trained',type=Path,default=Path('results/exp16_scale_teacher.json'));ap.add_argument('--specs',nargs='+',default=['VTQ_200','VTQ_300','VTQ_400']);ap.add_argument('--seeds',nargs='+',type=int,default=[1000,2000]);ap.add_argument('--winner');a=ap.parse_args();modes=['natural','static','adaptive','adaptive-half']
    if a.mode=='confirm':
        if a.winner not in modes:ap.error('freeze winner first')
        modes=list(dict.fromkeys(['natural',a.winner]))
    run_model_ab(a,modes,experiment(a.trained),provenance=dict(predictor=hashlib.sha256(a.trained.read_bytes()).hexdigest()))


if __name__=='__main__':main()
