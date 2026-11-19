"""Matched-bit LRT1: fund rank factors by shrinking 2-bit gains to 1 bit.
Use --mode discover on GPT-2, fix a winner before opening confirmation splits.
"""
import argparse
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import torch
from src import lrt1_format as lr, vtq
from src.exp09_group_order import run_model_ab, vtq_roundtrip
from src.vtq_format import load_vtq_formats


def experiment():
    f0=load_vtq_formats(menu=['VTQ_300'])['VTQ_300'];fmt=lr.make_format(f0.hw_codebook['tab_u8'],lr.GAINS)
    gains=torch.tensor(fmt.gain_tab,device='cuda');meta={}
    @contextmanager
    def ctx(name):
        meta.clear();meta.update(params=0,charged_bytes=0,vtq_payload_cap=0,ranks=[])
        if name=='natural':yield meta;return
        if name not in ('h1-only','lr-euclidean','lr-hessian','pre-euclidean','pre-hessian'):raise ValueError(name)
        original_get,original_quant=vtq._get_spec_gpu_vtq,vtq.quantize_matrix_vtq
        def get(sp,dev):
            assert sp=='VTQ_300';cfg,tab,_=original_get(sp,dev);return cfg,tab,gains
        def quant(w,h,*a,**kw):
            r=0 if name=='h1-only' else lr.budget_rank(*w.shape)
            wsrc=kw.get('W_target'); wsrc=w if wsrc is None else wsrc
            seed=kw.get('seed',1337)+812
            if name.startswith('pre-'):
                # Quantize the residual of the *serialized int8* low-rank path,
                # not the residual of an ideal float/SVD approximation.
                abc=lr.fit_residual(wsrc,h,None,r,seed,weighted=(name=='pre-hessian'))
                aa,bb,ss=abc
                rest=torch.addmm(wsrc.float(),aa.float()*ss[None,:],bb.float(),alpha=-1.) if r else wsrc
                qd,q,_=original_quant(w,h,*a,**dict(kw,W_target=rest)); del rest
            else:
                qd,q,_=original_quant(w,h,*a,**kw)
                abc=lr.fit_residual(wsrc,h,q,r,seed,weighted=(name=='lr-hessian'))
            vtq.PACK['matrix']['lr']=abc
            return qd,lr.apply_residual(q,*abc),float('nan') # base proxy loss does not describe corrected W
        with patch.object(vtq,'_get_spec_gpu_vtq',get),patch.object(vtq,'quantize_matrix_vtq',quant):yield meta
    def reload(path,p,base,method):
        m,k=p['d_row'].numel(),p['suh'].numel();cap=(m//16)*(k//256)*1616+2*(m+k);meta['params']+=m*k;meta['vtq_payload_cap']+=cap
        if method=='natural':
            q=vtq_roundtrip(path,p,base,method);meta['charged_bytes']+=path.stat().st_size;return q
        blob=lr.serialize(p,fmt); assert len(blob)<=cap
        meta['charged_bytes']+=len(blob);meta['ranks'].append(p['lr'][2].numel());return lr.decode(blob,'cuda')
    return ctx,reload


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--model',required=True);ap.add_argument('--mode',choices=['discover','confirm'],required=True)
    ap.add_argument('--text',type=Path,default=Path('data/wikitext2_test.raw'));ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--seeds',nargs='+',type=int,default=[1000,2000]);ap.add_argument('--winner');ap.add_argument('--pre',action='store_true');a=ap.parse_args();a.specs=['VTQ_300']
    modes=['natural','lr-hessian','pre-euclidean','pre-hessian'] if a.pre else ['natural','h1-only','lr-euclidean','lr-hessian']
    if a.mode=='confirm':
        if a.winner not in modes:ap.error('fix the winner before confirmation')
        modes=list(dict.fromkeys(['natural',a.winner]))
    ctx,reload=experiment();run_model_ab(a,modes,ctx,reload,dict(format='LRT1',gain_u8=lr.GAINS.tolist(),h_damping=.1,power_passes=3,oversample=8))


if __name__=='__main__':main()
