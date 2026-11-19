"""Offline shaping experiment, not a VTQ-compatible table file.

Keep the rate/trellis/2-bit group header of VTQ_300. Instead of four scaled
copies of one constellation, learn four int8-compatible 32-entry magnitude
banks under the joint gain/state Bellman search. A retrained tied-bank control
separates the effect of more Lloyd iterations from the effect of untying.
"""
import argparse
import ctypes
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
from unittest.mock import patch

import numpy as np
import torch

from src import tcq_proto as tcq, vtq
from src.vtq_format import load_vtq_formats, reconstruct_lut_from_hw_vtq


@contextmanager
def banked_library(source=None):
    source = Path(source) if source else Path(__file__).parent/'cuda'/'tcq_viterbi.cu'
    if b'VTQ_GAIN_SHAPES' not in source.read_bytes():
        raise ValueError('missing banked kernel specialization')
    # The .so is private to the experiment. Never swap a canonical .so on disk.
    key=hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    so=Path(tempfile.gettempdir())/f'vtq_gain_shapes_{key}.so'
    if not so.exists():
        tmp=so.with_suffix('.tmp.so')
        subprocess.run(['/usr/local/cuda/bin/nvcc','-O3','-arch=sm_120','-DVTQ_GAIN_SHAPES','-shared','-Xcompiler','-fPIC','-o',str(tmp),str(source)],check=True)
        tmp.replace(so)
    orig=tcq._tcq_lib(); lib=ctypes.CDLL(str(so))
    for name in ('tcq_viterbi','tcq_ldlq_block256'):
        fn=getattr(lib,name); fn.argtypes=getattr(orig,name).argtypes; fn.restype=ctypes.c_int
    with patch.object(tcq,'_TCQ_LIB',lib):
        yield


def expand_tables(banks):
    banks=np.asarray(banks)
    assert banks.shape==(4,32) and banks.min()>=0 and banks.max()<=127
    return torch.tensor(np.stack([reconstruct_lut_from_hw_vtq(31,dict(type='u8',qmax=127,tab_div=64,tab_u8=b.tolist())) for b in banks]),device='cuda')


def sample(n,seed,mixture=False):
    gen=torch.Generator(device='cuda').manual_seed(seed)
    x=torch.randn(n,16,16,device='cuda',generator=gen)
    if mixture:
        # Heterogeneous 16-channel variance; RMS-normalize each physical row block.
        x.mul_(torch.exp(.28*torch.randn(n,16,1,device='cuda',generator=gen)))
    x=x.reshape(n,256); return x/x.square().mean(1,keepdim=True).sqrt()


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',type=Path); ap.add_argument('--steps',type=int,default=32)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args(); torch.backends.cuda.matmul.allow_tf32=False
    f=load_vtq_formats(menu=['VTQ_300'])['VTQ_300']; base=np.array(f.hw_codebook['tab_u8'],dtype=np.int32)
    banks0=np.tile(base,(4,1)); w=torch.ones(256,device='cuda'); gain=torch.tensor(f.gain_tab,device='cuda')
    ids,sign,_=vtq._bucket_mapping_vtq(31); ids=torch.tensor(ids,device='cuda'); sign=torch.tensor(sign,device='cuda')
    def score(x,lut,scale):
        return vtq._eval_vtq_cand_on_blocks(x,w,lut.flatten(),gain,scale,16,4,0)
    val=sample(4096,9621); mixedval=sample(4096,9622,True); sanity=sample(128,2714)
    orig=score(sanity,torch.tensor(f.lut,device='cuda'),f.scale)
    report=dict(gpu=torch.cuda.get_device_name(),spec='VTQ_300',seed_train=7200,steps=args.steps,
                baseline=dict(scale=f.scale,banks=banks0.tolist()),experiments={})
    def save():
        tmp=args.output.with_suffix('.tmp'); tmp.write_text(json.dumps(report,indent=2)+'\n');tmp.replace(args.output)
    with banked_library(args.source):
        chk=score(sanity,expand_tables(banks0),f.scale)
        assert all(torch.equal(a,b) for a,b in zip(orig[1:],chk[1:])), 'repeated banks differ from VTQ'
        for mode in ('tied','untied','initial-warp','mixed-train'):
            tied=mode=='tied'; bank=banks0.copy(); scale=f.scale
            if mode=='initial-warp':
                bank=np.clip(np.rint(64*(bank/64)**np.array([.85,.95,1.05,1.15])[:,None]),0,127).astype(np.int32)
            best=None
            for it in range(args.steps+1):
                tab=expand_tables(bank)
                if it%4==0 or it==args.steps:
                    for ratio in (.98,1.,1.02):
                        sc=scale*ratio; valerr=score(val,tab,sc)[0]; mixerr=score(mixedval,tab,sc)[0]
                        metric=valerr if mode!='mixed-train' else .5*(valerr+mixerr)
                        if best is None or metric<best['metric']:
                            best=dict(it=it,metric=metric,mse=valerr,mix_mse=mixerr,banks=bank.tolist(),scale=sc)
                    scale=best['scale'] if best['it']==it else scale
                if it==args.steps: break
                x=sample(4096,7200+it,mode=='mixed-train' and it%2==1)
                _,d,gi,code,_=score(x,tab,scale)
                g=gi.long().repeat_interleave(16,1); bid=ids[code.long()] + (0 if tied else 32*g)
                coeff=sign[code.long()]*gain[g]*d
                num=torch.zeros(32 if tied else 128,device='cuda',dtype=torch.float64)
                den=torch.zeros_like(num)
                num.scatter_add_(0,bid.flatten(),(x*coeff).double().flatten()); den.scatter_add_(0,bid.flatten(),coeff.double().square().flatten())
                old=bank[:1] if tied else bank
                target=(64*num/den.clamp_min(1e-12)).cpu().numpy().reshape(old.shape)
                ok=den.cpu().numpy().reshape(old.shape)>2
                updated=np.rint(.15*old+.85*np.where(ok,target,old)).clip(0,127).astype(np.int32)
                bank=np.tile(updated,(4,1)) if tied else updated
            report['experiments'][mode]=best;save();print(mode,'mse',best['mse'],'mixed',best['mix_mse'],flush=True)
        # Fresh test sets: no scale, stopping iteration or model selection on these.
        for name,rec in [('baseline',report['baseline']),*report['experiments'].items()]:
            tab=expand_tables(rec['banks']); sc=rec['scale']
            rec['heldout']={key:float(np.mean([score(sample(4096,seed,mix),tab,sc)[0] for seed in range(15000,15004)])) for key,mix in [('gauss',False),('mixture',True)]}
            print(name,'test',rec['heldout'],flush=True)
        save()


if __name__=='__main__': main()
