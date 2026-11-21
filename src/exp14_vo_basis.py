"""Lossless-in-real-arithmetic V/O basis changes, with no basis in the file.
For each KV head use V' = T V and W_o' = W_o T^{-1} for *every* associated
query head. Attention probabilities and RoPE are untouched. Not a standalone
output rotation/inverse at inference. Compare to fixed Hadamard and PCA controls.
"""
import argparse
from contextlib import contextmanager
from pathlib import Path

import torch

from src.exp09_group_order import run_model_ab

MODES=['natural','vo-hadamard','vo-pca','vo-balance','vo-balance-mix','vo-weight-mix']


def eig(a):
    e,v=torch.linalg.eigh((a+a.transpose(1,2))*.5)
    j=v.abs().argmax(1,keepdim=True); s=v.gather(1,j).sign(); v=v*torch.where(s==0,1.,s)
    return e.clamp_min(1e-8),v


def transforms(mode,wv,wo,cov,nkv,repeats,seed):
    d=wv.shape[0]//nkv; dev=wv.device; m,k=wo.shape
    assert d&(d-1)==0 and k==nkv*repeats*d
    ht=torch.ones(1,1,device=dev)
    while ht.shape[0]<d: ht=torch.cat([torch.cat([ht,ht],1),torch.cat([ht,-ht],1)],0)
    rng=torch.Generator(device=dev).manual_seed(seed)
    signs=2*torch.randint(0,2,(nkv,1,d),generator=rng,device=dev)-1
    mix=ht.unsqueeze(0)*(d**-.5)*signs
    if mode=='vo-hadamard':return mix,mix.transpose(1,2).contiguous()
    v=wv.reshape(nkv,d,-1); o=wo.reshape(m,nkv,repeats,d).permute(1,0,2,3).reshape(nkv,-1,d)
    cv=v@v.transpose(1,2) if mode=='vo-weight-mix' else cov
    go=o.transpose(1,2)@o;eye=torch.eye(d,device=dev)
    def norm(a):return a/a.diagonal(dim1=1,dim2=2).mean(1).clamp_min(1e-12)[:,None,None]+.05*eye
    cv,go=norm(cv),norm(go)
    e,p=eig(cv)
    if mode=='vo-pca':return p.transpose(1,2).contiguous(),p
    cs=(p*e.sqrt()[:,None,:])@p.transpose(1,2);ci=(p*e.rsqrt()[:,None,:])@p.transpose(1,2)
    ev,rot=eig(cs@go@cs); scale=ev.pow(.25)
    t=scale[:,:,None]*(rot.transpose(1,2)@ci); inv=(cs@rot)/scale[:,None,:]
    if mode in ('vo-balance-mix','vo-weight-mix'):return mix@t,inv@mix.transpose(1,2)
    if mode!='vo-balance':raise ValueError(mode)
    return t,inv


def convert_o(wo,inv,repeats):
    n,d,_=inv.shape;m=wo.shape[0]
    o=wo.reshape(m,n,repeats,d).permute(1,0,2,3).reshape(n,-1,d)
    return (o@inv).reshape(n,m,repeats,d).permute(1,0,2,3).reshape_as(wo).contiguous()


def convert_h(h,t,repeats):
    tt=t.repeat_interleave(repeats,dim=0); nh,d,_=tt.shape;k=h.shape[0]
    y=(tt@h.reshape(nh,d,k)).reshape(k,k)
    return (tt@y.T.reshape(nh,d,k)).reshape(k,k).T.contiguous()


def setup(model,weights,hess,transpose):
    if transpose: raise ValueError('use a bias-free Llama/Qwen-style GQA model')
    for layer in model.model.layers:
        assert layer.self_attn.v_proj.bias is None and layer.self_attn.o_proj.bias is None
    nkv=model.config.num_key_value_heads;repeats=model.config.num_attention_heads//nkv
    memo={}
    @contextmanager
    def ctx(mode):
        memo.clear()
        try:yield
        finally:memo.clear()
    def adapt(mode,i,w,h):
        if mode=='natural' or i%7 not in (2,3):return w,h
        li=i//7
        if i%7==2:
            o=weights[i+1].to(device=w.device,dtype=torch.float32);d=w.shape[0]//nkv;nh=nkv*repeats
            if mode in ('vo-weight-mix','vo-hadamard'): cov=None
            else:
                blocks=hess[i+1].reshape(nh,d,nh,d); ids=torch.arange(nh)
                cov=blocks[ids,:,ids,:].reshape(nkv,repeats,d,d).mean(1).to(w.device)
            t,inv=transforms(mode,w,o,cov,nkv,repeats,9217+li);memo[li]=(t,inv)
            return (t@w.reshape(nkv,d,-1)).reshape_as(w).contiguous(),h
        t,inv=memo.pop(li);return convert_o(w,inv,repeats),convert_h(h,t,repeats)
    return ctx,adapt


def selftest():
    # Use different query-head contexts (GQA), not a replicated common value.
    torch.manual_seed(43);dev='cuda';n,r,d=2,3,16;k=n*r*d
    v=torch.randn(n*d,64,device=dev);o=torch.randn(64,k,device=dev);x=torch.randn(48,64,device=dev)
    attn=torch.randn(n,r,11,len(x),device=dev).softmax(-1); vals=(x@v.T).reshape(len(x),n,d).permute(1,0,2)
    def output(vv,oo):
        a=attn@vv[:,None];z=a.permute(2,0,1,3).reshape(11,k);return z@oo.T,z
    target,z=output(vals,o);h=z.T@z/len(z);cov=torch.stack([h[j*d:(j+1)*d,j*d:(j+1)*d] for j in range(n*r)]).reshape(n,r,d,d).mean(1)
    for mode in MODES[1:]:
        t,inv=transforms(mode,v,o,cov,n,r,1);rot_v=t@v.reshape(n,d,-1); y,z1=output((x@rot_v.reshape(n*d,-1).T).reshape(len(x),n,d).permute(1,0,2),convert_o(o,inv,r))
        torch.testing.assert_close(y,target,rtol=4e-5,atol=3e-4)
        torch.testing.assert_close(convert_h(h,t,r),z1.T@z1/len(z),rtol=4e-5,atol=1e-4)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--model',required=True);ap.add_argument('--mode',choices=['discover','confirm'],required=True);ap.add_argument('--text',type=Path,default=Path('data/wikitext2_test.raw'));ap.add_argument('--output',type=Path,required=True);ap.add_argument('--seeds',nargs='+',type=int,default=[1000,2000]);ap.add_argument('--specs',nargs='+',default=['VTQ_300']);ap.add_argument('--winner');a=ap.parse_args();modes=MODES
    if a.mode=='confirm':
        if a.winner not in MODES:ap.error('freeze candidate before confirmation')
        modes=list(dict.fromkeys(['natural',a.winner]))
    torch.backends.cuda.matmul.allow_tf32=False; selftest()
    run_model_ab(a,modes,setup=setup,provenance=dict(experiment='V/O gauge',regularization=.05))


if __name__=='__main__':main()
