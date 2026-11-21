"""Encoder-only sign selection on a deterministic small weight panel. The signs
are already carried by FP16 suh, so there are no extra bits or decode ops. A
panel Viterbi with refitted step predicts distortion for multiple pre-rotations;
only one full causal LDLQ is run. No eval contexts are used for this choice.
"""
import argparse
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch
import torch
from src import vtq
from src.exp09_group_order import run_model_ab

OFF=(0,29009,63823,124019)

@torch.inference_mode()
def pick(w,h,sp,seed,rb,rows,metric):
    assert isinstance(sp,str);m,k=w.shape;dev=w.device;n=min(24,m);ids=((torch.arange(n,device=dev)*m+m//(2*n))//n).clamp_max(m-1);x=w[ids].float();rw=rows[ids].float() if rows is not None else torch.ones(n,device=dev);rw=rw.clamp_min(rw.mean()*.1);rw=rw/rw.mean().clamp_min(1.e-12)
    col=(x.square()*rw[:,None]).mean(0).sqrt().clamp_min(1.e-6);hd=h.diagonal().clamp_min(1.e-8).sqrt();mag=((col/col.mean())**.5*(hd/hd.mean())**(-.25)).clamp(.25,4.).half().float();signs=torch.stack([vtq.make_rademacher_signs(k,seed+off,dev) for off in OFF]);suh=mag[None,:]*signs
    y=vtq.apply_vtq_rot((x[None,:,:]/suh[:,None,:]).reshape(-1,k),rot_block=rb,transpose=False).reshape(len(OFF),n,k//256,256);cfg,tab,gains=vtq._get_spec_gpu_vtq(sp,dev);rms=y.square().mean(-1).sqrt().clamp_min(1.e-10)*cfg['scale'];step=((rms.amax(-1,keepdim=True)*128./224.).clamp_min(1.e-7).half().float()/128.).clamp_min(1.e-12);d=step*(rms/step).round().clamp(1,255)
    rec,*_=vtq.viterbi_vtq_cuda((y/d[...,None]).reshape(-1,256),torch.ones(256,device=dev),tab,gains,cfg['nl'],1<<(cfg['K']-1),frac=cfg['frac']);rec=rec.reshape_as(y);opt=(rec*y).sum(-1)/rec.square().sum(-1).clamp_min(1.e-12);fin=(opt/step).round().clamp(1,255)*step
    e=(y-fin[...,None]*rec).reshape(-1,k);e=vtq.apply_vtq_rot(e,rot_block=rb,transpose=True).reshape(len(OFF),n,k)*suh[:,None,:];cost=(e.square() if metric=='panel-mse' else (e.reshape(-1,k)@h).reshape_as(e)*e).sum(-1);v=(cost*rw).mean(1);return int(v.argmin())


@contextmanager
def context(method):
    if method=='natural':yield;return
    quant=vtq.quantize_matrix_vtq;choices=[];stats={}
    def quantized(w,h,sp,*a,**kw):
        seed=kw.get('seed',1337);target=kw.get('W_target');i=1 if method=='fixed-offset' else pick(w if target is None else target,h,sp,seed,kw.get('rot_block',-1),kw.get('row_weights'),method);choices.append(i);kw['seed']=seed+OFF[i];return quant(w,h,sp,*a,**kw)
    with patch.object(vtq,'quantize_matrix_vtq',quantized):yield stats
    stats['offset_hist']=[choices.count(i) for i in range(len(OFF))]


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--model',required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--text',type=Path,default=Path('data/wikitext2_test.raw'));ap.add_argument('--specs',nargs='+',default=['VTQ_200','VTQ_300','VTQ_400']);ap.add_argument('--seeds',type=int,nargs='+',default=[17000,18000]);ap.add_argument('--mode',choices=['discover','confirm'],required=True);ap.add_argument('--winner',choices=['natural','panel-mse','panel-H']);a=ap.parse_args();methods=['natural','fixed-offset','panel-mse','panel-H']
    if a.mode=='confirm':
        if a.winner is None:ap.error('freeze method first');
        methods=list(dict.fromkeys(['natural','fixed-offset',a.winner]))
    run_model_ab(a,methods,context,provenance={'offsets':OFF,'sampled_rows':24,'full_encodes_per_matrix':1})

if __name__=='__main__':main()
