"""Output-space GPTAQ for down_proj: use the residual branch as another teacher.

Fit Delta^T=(H+lambda I)^-1 E[z_q^T ((y_r-y_q)+s*(mid_r-mid_q))].
The same alpha multiplies all corrections. With s=0 this equals ordinary GPTAQ
in exact arithmetic; bf16 MLP outputs include rounding. C and the solves have
M RHS instead of K. Full H and downstream VTQ are untouched. Cache the already
computed reference residual update; no second teacher MLP forward is needed.
"""
import argparse
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from unittest.mock import patch

import torch
from src import vtq
from src.vtq_v2 import DualCovAccumulator
from src.exp11_llama_transfer import run_transfer

SKIP={'output-only':0., 'skip-quarter':.25, 'skip-half':.5, 'skip-full':1.}


class OutputCov(DualCovAccumulator):
    def __init__(self,s):
        super().__init__(True); self.s=s

    @torch.no_grad()
    def add_outputs(self,z,yq,yr,cq,cr):
        super().add(z,None)
        err=yr.float()-yq.float()
        if self.s: err.add_(cr.float()-cq.float(),alpha=self.s)
        c=z.reshape(-1,z.shape[-1]).float().T @ err.reshape(-1,err.shape[-1]);c=c.double()
        self.sum_qc=c if self.sum_qc is None else self.sum_qc.add_(c)

    def get(self,gptaq_damp):
        h,delta=super().get(gptaq_damp); return h,delta.T


@contextmanager
def context(method):
    if method=='natural':yield;return
    with patch.object(vtq,'quantize_llama_model_vtq',partial(vtq.quantize_llama_model_vtq,down_output_fit=partial(OutputCov,SKIP[method]))):yield


@torch.inference_mode()
def oracle():
    # Three batches, rectangular features, exact independent normal equation.
    g=torch.Generator(device='cuda').manual_seed(19)
    z,zr,mid,midr,w=[torch.randn(shape,device='cuda',dtype=torch.float64,generator=g) for shape in [(3,96,64),(3,96,64),(3,96,16),(3,96,16),(16,64)]]
    y=z@w.T;yr=zr@w.T;x=z.flatten(0,1);h=x.T@x/len(x);reg=h+.3*h.diagonal().mean()*torch.eye(64,device='cuda');
    for s in [0.,1.]:
        a=OutputCov(s)
        for batch in zip(z,y,yr,mid,midr):a.add_outputs(*batch)
        hh,dd=a.get(.3);target=(yr-y+s*(midr-mid)).flatten(0,1);expected=torch.linalg.solve(reg,x.T@target/len(x)).T
        torch.testing.assert_close(hh.double(),h,rtol=2e-5,atol=2e-6);torch.testing.assert_close(dd.double(),expected,rtol=8e-5,atol=6e-6)
    print('output-space analytic oracle passed',flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--model',required=True);ap.add_argument('--data',type=Path,default=Path('data'));ap.add_argument('--fisher',default='fisher_1b.pt');ap.add_argument('--output',type=Path,required=True);ap.add_argument('--spec',default='VTQ_300');ap.add_argument('--cal-seqs',type=int,default=64);ap.add_argument('--offsets',type=int,nargs='+',default=[0,1000]);ap.add_argument('--mode',choices=['discover','confirm'],required=True);ap.add_argument('--winner',choices=['natural',*SKIP]);a=ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32=False;oracle();methods=['natural',*SKIP]
    if a.mode=='confirm':
        if a.winner is None:ap.error('freeze winner first')
        methods=list(dict.fromkeys(['natural','output-only',a.winner]))
    run_transfer(a,methods,context,a.spec,discovery=a.mode=='discover',provenance=dict(experiment='output residual fit',coefficients=SKIP,alpha=.5))


if __name__=='__main__':main()
