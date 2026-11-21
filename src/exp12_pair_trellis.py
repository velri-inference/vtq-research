"""P2T: 2 coded bits and a 4-bit *2-D vector* index per pair of weights.
At 2048 states the last 11 scalar bits determine BOTH 1024-state VTQ labels.
Thus a Cartesian-product 2-D lookup table represents the VTQ_300 code space
before training, unlike a naked VQ. Do not compare this research CUDA kernel's
speed to a production decoder. Screen shaping on held-out tiles first.
"""
import argparse
import ctypes
import hashlib
import json
import math
from pathlib import Path
import subprocess
import tempfile
import time

import numpy as np
import torch

from src import vtq
from src.vtq_format import galois_lb_vtq, load_vtq_formats


def library():
    src=Path(__file__).parent/'cuda'/'p2t.cu'; key=hashlib.sha256(src.read_bytes()).hexdigest()[:14]
    so=Path(tempfile.gettempdir())/f'p2t_{key}.so'
    if not so.exists():
        tmp=so.with_suffix('.tmp.so'); subprocess.run(['/usr/local/cuda/bin/nvcc','-O3','-arch=sm_120','-shared','-Xcompiler','-fPIC','-o',str(tmp),str(src)],check=True);tmp.replace(so)
    lib=ctypes.CDLL(str(so));lib.p2t_encode.argtypes=[ctypes.c_void_p]*13+[ctypes.c_int]*3+[ctypes.c_void_p];lib.p2t_encode.restype=ctypes.c_int;return lib


def ring_view(x):
    # Involution: a ring is 16 successive rows of one 16-column group.
    return x.reshape(-1,16,16,16).transpose(1,2).reshape(-1,256).contiguous()


def pair_labels(states):
    s=np.arange(states,dtype=np.uint16)
    return ((16*galois_lb_vtq(s>>1,16).astype(np.int32))+galois_lb_vtq(s,16)).astype(np.uint8)


def cartesian(fmt):
    lut=fmt.lut.reshape(16,4)
    lx,ly=np.divmod(np.arange(256),16); u,v=np.divmod(np.arange(16),4)
    return np.rint(64*np.stack([lut[lx[:,None],u],lut[ly[:,None],v]],axis=-1)).astype(np.int8)


class PairSearch:
    def __init__(self,states=2048,chunk=128):
        assert states in (256,1024,2048) and 0<chunk<=4096
        self.S=states; self.chunk=chunk; self.lib=library();self.labels=torch.tensor(pair_labels(states),device='cuda')
        def alloc(dtype,*shape):return torch.empty(shape,device='cuda',dtype=dtype)
        self.ws=[alloc(torch.float32,chunk,128,4,256),alloc(torch.uint8,chunk,128,4,256),alloc(torch.uint8,chunk,128,4,states//4),alloc(torch.uint8,chunk,16,states),alloc(torch.int32,chunk),alloc(torch.int32,chunk)]

    def encode(self,x,book,gains,metric=None,return_rec=False):
        assert x.is_cuda and x.dtype==book.dtype==gains.dtype==torch.float32 and x.shape[1:]==(256,) and book.shape==(256,16,2) and gains.shape==(4,)
        x=x.contiguous(); book=book.contiguous(); gains=gains.contiguous();out=torch.empty((len(x),128),device=x.device,dtype=torch.uint8)
        if metric is not None:
            assert metric.is_cuda and metric.dtype==torch.float32 and metric.shape==(8,3)
            metric=metric.contiguous()
        rec=torch.empty_like(x) if return_rec else None
        for i in range(0,len(x),self.chunk):
            xx=x[i:i+self.chunk]; yy=out[i:i+self.chunk]
            args=[xx,book,gains,self.labels,*self.ws,yy]
            rc=self.lib.p2t_encode(*[a.data_ptr() for a in args],metric.data_ptr() if metric is not None else None,
                                   rec[i:].data_ptr() if rec is not None else None,len(xx),self.S,256,torch.cuda.current_stream().cuda_stream)
            if rc: raise RuntimeError(f'P2T CUDA error {rc}')
        return (out,rec) if return_rec else out

    def indices(self,raw):
        s=raw.long()&3; state=s.clone()
        for lag in range(1, math.ceil(math.log2(self.S)/2)):
            state|=torch.roll(s,lag,1)<<(2*lag)
        return self.labels[(state&(self.S-1))].long(),((raw>>2)&15).long(),(raw>>6).long(),state&(self.S-1)


def pack_ring(raw):
    raw=np.asarray(raw,np.uint8).reshape(-1,128);g=raw>>6
    assert np.all(g.reshape(-1,16,8)==g[:,::8,None])
    def pack(x,bits):
        lanes=8//bits
        return ((x.reshape(len(x),-1,lanes).astype(np.uint32)<<(bits*np.arange(lanes,dtype=np.uint32))).sum(axis=2)).astype(np.uint8)
    return np.concatenate([pack(raw&3,2),pack((raw>>2)&15,4),pack(g[:,::8],2)],axis=1)


def unpack_ring(buf):
    b=np.asarray(buf,np.uint8).reshape(-1,100)
    def unpack(x,bits): return ((x[:,:,None]>>(bits*np.arange(8//bits,dtype=np.uint8)))&((1<<bits)-1)).reshape(len(b),-1)
    return unpack(b[:,:32],2)|(unpack(b[:,32:96],4)<<2)|(np.repeat(unpack(b[:,96:],2),8,axis=1)<<6)


def dp_check(enc,fmt,book,gains):
    """Torch Bellman oracle with the seam state read only from packed symbols."""
    x=torch.randn((2,256),device='cuda',generator=torch.Generator(device='cuda').manual_seed(41));raw=enc.encode(x,book,gains)
    recovered=unpack_ring(pack_ring(raw.cpu().numpy()));assert np.array_equal(recovered,raw.cpu().numpy())
    l,u,g,states=enc.indices(torch.from_numpy(recovered).cuda()); rec=gains[g,None]*book[l,u]
    emitted=(x.view(2,128,2)-rec).square().sum((1,2)); start=states[:,63]; n=torch.arange(2,device='cuda');S=enc.S
    d=torch.full((2,4,S),float('inf'),device='cuda'); d[n,:,start]=0
    choices=(gains[:,None,None,None]*book).view(1,4,256,16,2); labels=enc.labels.long()
    for t in range(128):
        p=(64+t)%128
        if t and not t%8: d=d.amin(1,keepdim=True).expand(-1,4,-1)
        branch=(x[:,2*p:2*p+2].view(2,1,1,1,2)-choices).square().sum(-1).amin(-1)
        d=d.reshape(2,4,4,S//4).amin(2).repeat_interleave(4,dim=2)+branch[:,:,labels]
    optimum=d[n,:,start].amin(1); torch.testing.assert_close(emitted,optimum,rtol=2e-5,atol=1e-5)
    return float((emitted-optimum).abs().max())


def sample(n,seed):
    x=torch.randn(n,256,device='cuda',generator=torch.Generator(device='cuda').manual_seed(seed));return x/x.square().mean(1,keepdim=True).sqrt()


def objective(enc,x,book,gains,scale,assign=False):
    rings=ring_view(x);raw=enc.encode(rings/scale,book,gains);label,idx,g,_=enc.indices(raw)
    rec=ring_view((gains[g][...,None]*book[label,idx]).reshape(-1,256))
    step=scale/208.; u=((x*rec).sum(1)/(rec*rec).sum(1).clamp_min(1e-12)/step).round().clamp(1,255); d=(step*u)[:,None]
    err=float((x-d*rec).square().mean())
    if not assign: return err
    dg=ring_view(d.expand_as(x)).reshape(-1,128,2)*gains[g][...,None]
    return err,rings.reshape(-1,128,2),dg,label*16+idx


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--steps',type=int,default=32);ap.add_argument('--states',nargs='+',type=int,default=[2048,1024,256]);ap.add_argument('--train',type=int,default=2048);ap.add_argument('--val',type=int,default=2048);args=ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32=False; fmt=load_vtq_formats(menu=['VTQ_300'])['VTQ_300']; base=cartesian(fmt);gains=torch.tensor(fmt.gain_tab,device='cuda');val=sample(args.val,99121)
    test=[sample(args.val,seed) for seed in range(8910,8914)]; scalar=torch.tensor(fmt.lut,device='cuda');ones=torch.ones(256,device='cuda')
    def control(x):return vtq._eval_vtq_cand_on_blocks(x,ones,scalar,gains,fmt.scale,16,4,0)[0]
    report=dict(gpu=torch.cuda.get_device_name(),format='P2T',bits_per_256=808,steps=args.steps,train=args.train,val=args.val,scalar=dict(dev=control(val),test=float(np.mean([control(x) for x in test]))),runs=[])
    def save():
        tmp=args.output.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2)+'\n');tmp.replace(args.output)
    base_t=torch.tensor(base,device='cuda').float()
    for S in args.states:
        enc=PairSearch(S); oracle=dp_check(enc,fmt,base_t/64,gains)
        for init in ('cartesian','stagger','spin'):
            tick=time.perf_counter(); cc=base_t.clone(); rng=np.random.default_rng(27);sc=fmt.scale
            if init=='spin':
                theta=torch.tensor(rng.uniform(-.4,.4,256),device='cuda',dtype=torch.float32)[:,None];theta=(theta+theta.flip(0))*.5
                a,b=cc.unbind(-1);cc=torch.stack((theta.cos()*a-theta.sin()*b,theta.sin()*a+theta.cos()*b),-1)
            elif init=='stagger':
                j=torch.arange(16,device='cuda');cc[:,:,0]+=(((j%4)&1)*2-1)*5;cc[:,:,1]+=((((j//4)&1)*2-1)*5)
            best=None;history=[]
            for i in range(args.steps+1):
                tab=cc.round().clamp(-127,127)/64
                if i%4==0 or i==args.steps:
                    e=objective(enc,val,tab,gains,sc);history.append([i,e]);
                    if best is None or e<best[0]:best=(e,tab.clone(),sc,i)
                if i==args.steps:break
                x=sample(args.train,67000+i); _,t,d,ids=objective(enc,x,tab,gains,sc,True)
                # Tie central inversion: (l,u) -> (255-l,15-u) represents -v.
                rev=4095-ids;fold=torch.minimum(ids,rev).flatten(); side=torch.where(ids<rev,1.,-1.)[...,None]
                num=torch.zeros(2048,2,device='cuda',dtype=torch.float64);den=torch.zeros_like(num)
                num.index_add_(0,fold,(side*t*d*64).reshape(-1,2).double()); den.index_add_(0,fold,d.square().reshape(-1,2).double())
                prior=base_t.reshape(-1,2)[:2048];estimate=((num+8*prior)/(den+8)).float()
                update=torch.cat([estimate,-estimate.flip(0)]).view(256,16,2)
                cc=cc*.45 + update*.55
            e,tab,sc,i=best
            # Select scale on dev only.
            for mul in (.94,.97,.985,1.015,1.03,1.06):
                ee=objective(enc,val,tab,gains,fmt.scale*mul)
                if ee<e:e,sc=ee,fmt.scale*mul
            te=float(np.mean([objective(enc,x,tab,gains,sc) for x in test])); stem=args.output.stem+f'_{S}_{init}.bin'
            (args.output.parent/stem).write_bytes((tab*64).round().to(torch.int8).cpu().numpy().tobytes())
            report['runs'].append(dict(states=S,init=init,dev=e,test=te,scale=sc,best_iter=i,history=history,book=stem,oracle_error=oracle,sec=time.perf_counter()-tick));save();print(S,init,e,te,'ratio',te/report['scalar']['test'],flush=True)
        del enc


if __name__=='__main__':main()
