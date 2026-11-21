"""One-pass VTQ scale predictor. Offline labels are actual trellis distortions,
not scalar lookup MSE. Learn expected loss *differences* on normalized tiles;
test by rerunning one Viterbi with independently predicted per-row scales.
Constants in the resulting predictor are encoder-only; no change in .vtq.
"""
import argparse
import json
from pathlib import Path
import time

import torch
from src import vtq

MULT=(.84,.90,.95,1.,1.05,1.10,1.18)
F=4*64+32+1


def features(x):
    x=x.float().reshape(-1,16,16);r=x.square().mean((1,2),keepdim=True).sqrt().clamp_min(1e-20);z=x/r
    sub=z.square().mean(2).sqrt();zone=(sub>.82).long()+(sub>1.0).long()+(sub>1.18).long()
    bins=((z.abs()*16).long().clamp_max(63)+64*zone[:,:,None]).flatten(1)
    h=torch.zeros(len(x),F,device=x.device);h.scatter_add_(1,bins,torch.full(bins.shape,1/256.,device=x.device))
    j=(sub*16).long().clamp_max(31)+256;h.scatter_add_(1,j,torch.full(j.shape,1/16.,device=x.device));h[:,-1]=1.
    return h


def tiles(n,seed,distribution):
    gen=torch.Generator(device='cuda').manual_seed(seed);dev='cuda'
    def rand(*sz):return torch.rand(sz,device=dev,generator=gen)
    def normal(*sz):return torch.randn(sz,device=dev,generator=gen)
    x=normal(n,16,16)
    if distribution!='gauss':
        kind=(torch.arange(n,device=dev)%8)[:,None,None]
        x=torch.where(kind==3,(rand(n,16,16)*2-1)*3**.5,x)
        x=torch.where(kind==4,x.sign()*x.abs().pow(1.35),x)
        x=torch.where(kind==5,x*(1+1.7*(rand(n,16,16)<.02)),x)
        x=torch.where(kind>=6,x*torch.exp(.35*normal(n,16,1)),x)
    x=x.reshape(n,256);return x/x.square().mean(1,keepdim=True).sqrt()


def teacher(x,cfg,tab,gains,fac):
    sc=cfg['scale'];step=sc/208;u0=(208*fac).round().clamp(1,255);d0=(step*u0)[:,None]
    rec,*_=vtq.viterbi_vtq_cuda(x/d0,torch.ones(256,device=x.device),tab,gains,cfg['nl'],1<<(cfg['K']-1),frac=cfg['frac'])
    d=(x*rec).sum(1)/rec.square().sum(1).clamp_min(1e-12);dr=step*(d/step).round().clamp(1,255)
    return (x-dr[:,None]*rec).square().mean(1)


def fit(x,y,ridge):
    # Unit-variance ridge; bin columns need not be independent. Store the
    # affine map in the histogram's native coordinates for the fused kernel.
    x=x.double(); y=y.double()-y[:,3:4].double();mu=x[:,:-1].mean(0);sd=x[:,:-1].std(0).clamp_min(.002);z=(x[:,:-1]-mu)/sd
    ym=y.mean(0);a=z.T@z;a.diagonal().add_(len(x)*ridge);b=torch.linalg.solve(a,z.T@(y-ym))/sd[:,None]
    return torch.cat([b,(ym-mu@b)[None,:]]).float()


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--ntrain',type=int,default=65536);ap.add_argument('--specs',nargs='+',default=['VTQ_200','VTQ_300','VTQ_400']);a=ap.parse_args()
    vtq.load_and_register_vtq();torch.backends.cuda.matmul.allow_tf32=False;mult=torch.tensor(MULT,device='cuda');report=dict(ntrain=a.ntrain,mult=MULT,features=F,specs={})
    for sp in a.specs:
        start=time.perf_counter();cfg,tab,g=vtq._get_spec_gpu_vtq(sp,torch.device('cuda')); xx=[];yy=[]
        for i in range((a.ntrain+1023)//1024):
            x=tiles(1024,10001+i,'mix');xx.append(features(x));yy.append(torch.stack([teacher(x,cfg,tab,g,f.expand(len(x))) for f in mult],1))
        xx=torch.cat(xx); yy=torch.cat(yy); static=int(yy.mean(0).argmin()); models=[fit(xx,yy,r) for r in (.01,.1,1.)]
        def assess(seed,n,distribution,weights):
            scores=[];shares=torch.zeros(len(MULT),device='cuda')
            for i in range(n//1024):
                x=tiles(1024,seed+i,distribution); j=(features(x)@weights).argmin(1); shares+=torch.bincount(j,minlength=len(MULT))
                scores.append(torch.stack([teacher(x,cfg,tab,g,torch.ones(len(x),device='cuda')),teacher(x,cfg,tab,g,mult[static].expand(len(x))),teacher(x,cfg,tab,g,mult[j])],1).mean(0))
            v=torch.stack(scores).mean(0).tolist();return dict(baseline=v[0],static=v[1],adaptive=v[2],rel_pct=100*(v[2]/v[0]-1),shares=(shares/shares.sum()).tolist())
        dev=[assess(23000,8192,'mix',b) for b in models];idx=min(range(len(dev)),key=lambda j:dev[j]['adaptive']);b=models[idx]
        tests={ds:assess(39000,16384,ds,b) for ds in ('gauss','mix')}
        report['specs'][sp]=dict(static_factor=MULT[static],dev=dev,ridge=(.01,.1,1.)[idx],weights=b.cpu().tolist(),test=tests,sec=time.perf_counter()-start)
        tmp=a.output.with_suffix('.tmp');tmp.write_text(json.dumps(report)+'\n');tmp.replace(a.output);print(sp,'static',MULT[static],{ds:t['rel_pct'] for ds,t in tests.items()},flush=True)


if __name__=='__main__':main()
