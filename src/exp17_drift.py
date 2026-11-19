"""Structured GPTAQ: estimate channelwise drift + a restricted linear residual.
No new fields in a .vtq. H stays full for LDLQ. Only the map from quantized to
teacher activations is constrained, reducing cross-covariance memory/solves.
Discovery uses calibration rows 65--96, NOT the two final dialogue traces.
"""
import argparse
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import torch
from src import vtq
from src.vtq_v2 import DualCovAccumulator
from src.exp11_llama_transfer import run_transfer

METHODS=['natural','diag-down','diag-all','prior-down','pca64-down','pca64-all','alpha1-down']
ALPHA=.5  # the underlying Llama comparison uses this same full-GPTAQ alpha in both arms


class StructuredCov(DualCovAccumulator):
    def __init__(self,need_cross,mode):
        super().__init__(need_cross); self.mode=mode;self.cd=None;self.pc=None;self.sc=None

    @torch.no_grad()
    def add(self,xq,xr=None):
        if self.mode in ('prior','alpha1') or not self.need_cross or xr is None:
            return super().add(xq,xr)
        super().add(xq,None)  # retain the IDENTICAL full Hessian used by baseline LDLQ
        x=xq.reshape(-1,xq.shape[-1]).float(); e=xr.reshape_as(x).float()-x
        if self.mode=='pca':
            if self.pc is None:
                # Sample a few hundred tokens in the first calibration batch.
                # The orthonormal columns span dominant input directions. No
                # K x K eigen-decomposition and no reference/evaluation tokens.
                z=x[torch.linspace(0,len(x)-1,min(len(x),384),device=x.device).long()];r=min(64,*z.shape)
                val,vec=torch.linalg.eigh(z@z.T); del val
                self.pc=torch.linalg.qr(z.T@vec[:,-r:],mode='reduced')[0].contiguous()
            c=((x@self.pc).T@e).double()
            self.sc=c if self.sc is None else self.sc.add_(c)
        cd=e.mul_(x).sum(dim=0).double();self.cd=cd if self.cd is None else self.cd.add_(cd)

    @torch.no_grad()
    def get(self,gptaq_damp=.2):
        if not self.need_cross:return super().get(gptaq_damp)
        if self.mode=='alpha1':
            h,b=super().get(gptaq_damp);return h,b/ALPHA
        xdiag=self.sum_xx.diagonal();mean=xdiag.mean().clamp_min(1e-12)
        cd=self.sum_qc.diagonal() if self.mode=='prior' else self.cd
        b=(cd/(xdiag+.01*mean).clamp_min(1e-20)).clamp(-.5,.5).float()
        if self.mode=='prior':
            # Ridge on the deviation from diag(b), not toward a zero map.
            original=self.sum_qc.diagonal().clone()
            self.sum_qc.diagonal().add_(gptaq_damp*mean*b.double())
            try:h,B=super().get(gptaq_damp)
            finally:self.sum_qc.diagonal().copy_(original)
            B.diagonal().add_((1/ALPHA-1)*b);return h,B
        h,_=super().get(gptaq_damp)  # sum_qc is None => no dense drift solve
        if self.mode=='diag':return h,b/ALPHA
        pc=self.pc;pc_h=pc.T@h; gram=pc_h@pc;gram.diagonal().add_(gptaq_damp*h.diagonal().mean().clamp_min(1e-8))
        rhs=(self.sc/self.n_tokens).float().sub_(pc_h*b[None,:]); right=torch.linalg.solve(gram,rhs)
        return h,(b,pc,right)


def projected_target(w,B,alpha):
    if isinstance(B,tuple):
        diag,p,r=B
        return torch.addmm(w*(1+diag),w@r.T,p.T,alpha=alpha)
    return w*(1+alpha*B) if B.ndim==1 else w+alpha*(w@B.T)


@contextmanager
def drift(method):
    if method=='natural': yield;return
    if method not in METHODS:raise ValueError(method)
    i=0
    def create(need_cross):
        nonlocal i
        stage=('qkv','o','up','down')[i%4];i+=1
        if method.endswith('-down') and stage!='down':return DualCovAccumulator(need_cross)
        mode=method.split('-')[0];mode='pca' if mode=='pca64' else mode
        return StructuredCov(need_cross,mode)
    with patch.object(vtq,'DualCovAccumulator',create),patch.object(vtq,'_drift_target',projected_target):yield
    assert i and i%4==0, 'unexpected encoder stage creation order'


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--model',required=True);ap.add_argument('--data',type=Path,default=Path('data'));ap.add_argument('--fisher',default='fisher_1b.pt');ap.add_argument('--output',type=Path,required=True);ap.add_argument('--cal-seqs',type=int,default=64);ap.add_argument('--offsets',nargs='+',type=int,default=[0,1000]);ap.add_argument('--spec',default='VTQ_300');ap.add_argument('--mode',choices=['discover','confirm'],required=True);ap.add_argument('--winner',choices=METHODS);a=ap.parse_args()
    methods=METHODS
    if a.mode=='confirm':
        if a.winner is None:ap.error('freeze winner before confirmation')
        methods=list(dict.fromkeys(['natural',a.winner]))
    run_transfer(a,methods,drift,a.spec,discovery=a.mode=='discover',provenance=dict(experiment='structured GPTAQ',alpha=ALPHA,rank=64,diag_clip=.5,diag_damp=.01))


if __name__=='__main__':main()
