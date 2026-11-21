"""Held-out matrix/model A/B, P2T1 vs VTQ_300. Same row/block scales, gains,
rotation, block16 outer feedback and FP16 refit. Only trellis/branch metric differ.
Cartesian control uses the original scalar levels at both coordinates of a pair.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from src import vtq,p2t_format
from src.exp09_group_order import run_model_ab,vtq_roundtrip
from src.exp12_pair_trellis import PairSearch,cartesian,dp_check
from src.vtq_format import load_vtq_formats


def block_encoder(enc,book,gains,pair_metric):
    def encode(w,d,dw,U,Ui,tab,g,nl,unc,frac=0,return_code=False):
        m,k=w.shape
        assert k==256 and m%16==0 and frac==0 and nl==16 and unc==4
        rec=torch.empty_like(w)
        if return_code:
            gi=torch.empty((m,16),dtype=torch.uint8,device=w.device);c=torch.empty((m,16,2),dtype=torch.uint8,device=w.device);u=torch.empty((m,256),dtype=torch.uint8,device=w.device)
        sub=Ui.view(16,16,16,16); ids=torch.arange(16,device=w.device); v=sub[ids,:,ids,:].contiguous()
        if pair_metric:
            h=v@v.transpose(1,2); e=2*torch.arange(8,device=w.device)
            metric=torch.stack([h[:,e,e],h[:,e,e+1],h[:,e+1,e+1]],dim=-1)
        else:
            dw=dw.reshape(16,8,2);metric=torch.stack([dw[:,:,0],torch.zeros_like(dw[:,:,0]),dw[:,:,1]],dim=-1)
        for j in range(16):
            s=16*j; x=w[:,s:s+16]
            raw,r=enc.encode((x/d[:,None]).reshape(-1,256),book,gains,metric[j],return_rec=True);r=r.view(m,16)
            rec[:,s:s+16]=r
            if return_code:
                raw=raw.view(m,8);gids=raw>>6;gi[:,j]=gids[:,0]
                cc=(raw&3).view(m,2,4);c[:,j]=cc[:,:,0]|(cc[:,:,1]<<2)|(cc[:,:,2]<<4)|(cc[:,:,3]<<6)
                u[:,s:s+16]=((raw>>2)&15).repeat_interleave(2,dim=1)
            if s+16<k:
                w[:,s+16:].sub_(((x-d[:,None]*r)@v[j])@U[s:s+16,s+16:])
        return (rec,gi,c,u) if return_code else (rec,None)
    return encode


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--model',required=True);ap.add_argument('--mode',choices=['discover','confirm'],required=True)
    ap.add_argument('--text',type=Path,default=Path('data/wikitext2_test.raw'));ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--trained',type=Path,default=Path('results/exp12_pair_trellis.json'));ap.add_argument('--seeds',nargs='+',type=int,default=[1000,2000]);ap.add_argument('--winner')
    args=ap.parse_args();args.specs=['VTQ_300'];exp=json.loads(args.trained.read_text()); chosen=min(exp['runs'],key=lambda v:v['dev'])
    S=chosen['states']; fmt=load_vtq_formats(menu=args.specs)['VTQ_300'];tab0=cartesian(fmt);tab1=np.fromfile(args.trained.parent/chosen['book'],dtype=np.int8).reshape(256,16,2)
    modes=['natural','pair-base','pair-base-H','pair-trained','pair-trained-H']
    if args.mode=='confirm':
        if args.winner not in modes:ap.error('freeze candidate before confirmation')
        modes=list(dict.fromkeys(['natural',args.winner]))
    gains=torch.tensor(fmt.gain_tab,device='cuda'); search=PairSearch(S,chunk=128)
    def config(name): return (tab1,chosen['scale']) if 'trained' in name else (tab0,fmt.scale)
    @contextmanager
    def ctx(name):
        if name=='natural':yield;return
        rec,sc=config(name);book=torch.tensor(rec,device='cuda').float()/64
        orig=vtq._get_spec_gpu_vtq
        def get(sp,dev):
            cfg,tab,g=orig(sp,dev);return dict(cfg,scale=sc),tab,g
        with patch.object(vtq,'ldlq_vtq_block256_cuda',block_encoder(search,book,gains,name.endswith('-H'))),patch.object(vtq,'_get_spec_gpu_vtq',get):yield
    def reload(path,p,fmt,method):
        if method=='natural':return vtq_roundtrip(path,p,fmt,method)
        return p2t_format.decode(p2t_format.serialize(p,config(method)[0],fmt.gain_u8,S),'cuda')
    # Also check the fused reconstructed vector against the independent state oracle.
    dp_check(search,fmt,torch.tensor(tab1,device='cuda').float()/64,gains)
    run_model_ab(args,modes,ctx,reload,dict(format='P2T1',states=S,book_sha256=hashlib.sha256(tab1).hexdigest(),
        workspace_bytes=sum(v.element_size()*v.numel() for v in search.ws),extra_model_book_bytes_vs_vtq=8192-32))


if __name__=='__main__':main()
