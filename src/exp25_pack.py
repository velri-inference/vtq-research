"""Experimental host serializer used only when comparing writer cost. Compile
src/native/vtq_pack_tiles.cpp on the test host and pass the .so explicitly.
The candidate never changes code search, table selection, scales, or decoding.
"""
import argparse
import ctypes as ct
from pathlib import Path
import hashlib,json,time
import numpy as np
from src import vtq_format as vf
from src.vtq_v2_format import HAM_CW8


class NativePack:
    def __init__(self,lib):
        self.so=ct.CDLL(str(Path(lib).resolve()));self.fn=self.so.vtq_pack
        self.fn.argtypes=[ct.c_void_p]*6+[ct.c_int]*6+[ct.c_size_t];self.fn.restype=ct.c_int
        self.lut=np.ascontiguousarray(vf.CW8_TO_IDX,dtype=np.int16);assert self.lut.shape==(256,)

    def __call__(self,c,u,g,s,fmt):
        c,u,g,s=[np.ascontiguousarray(x,dtype=np.uint8) for x in (c,u,g,s)]
        m,n=c.shape[:2];sh=(m,n)
        if (m<1 or n<1 or m>=2**31 or n>=2**31 or c.shape!=sh+(16,2) or u.shape!=sh+(256,) or g.shape!=sh+(16,) or s.shape!=sh):raise ValueError('invalid code plane shape')
        if vf.FMT_META[fmt.fmt_code]!=(fmt.K,fmt.gosset,fmt.nl,fmt.h,fmt.st_bytes):raise ValueError('invalid native format')
        out=np.empty(((m+15)//16)*n*fmt.st_bytes,np.uint8);args=[v.ctypes.data for v in (c,u,g,s,self.lut,out)]
        err=self.fn(*args,m,n,fmt.K,fmt.gosset,fmt.h,fmt.st_bytes,out.size)
        if err:raise ValueError(f'invalid trellis pack request or constrained codeword, status {err}')
        return out


def data(m,k,fmt,seed):
    rng=np.random.default_rng(seed);sh=(m,k//256);u=rng.integers(0,fmt.U,sh+(256,),dtype=np.uint8)
    if fmt.gosset:
        if fmt.gosset==1:
            cw=HAM_CW8[rng.integers(0,16,sh+(32,))];b=np.unpackbits(cw,axis=-1)
        else:
            b=rng.integers(0,2,sh+(64,4),dtype=np.uint8)
            if fmt.gosset==2:b[...,3]=b[...,0]^b[...,1]^b[...,2]
            else:b[...,1:]=b[...,:1]
            b=b.reshape(u.shape)
        u=(u&254)|b
    return rng.integers(0,256,sh+(16,2),dtype=np.uint8),u,rng.integers(0,1<<fmt.h,sh+(16,),dtype=np.uint8),rng.integers(0,256,sh,dtype=np.uint8)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--lib',required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();pack=NativePack(a.lib);fmts=vf.load_vtq_formats();rep={'formats':{},'timings':[]}
    for index,(name,fmt) in enumerate(fmts.items()):
        for m,k in [(1,256),(17,768),(64,2048)]:
            inputs=data(m,k,fmt,793+index);b=vf.pack_supertiles_vtq(*inputs,fmt);c=pack(*inputs,fmt);assert np.array_equal(b,c),(name,m,k)
        if fmt.gosset:
            inputs[1].flat[0]^=1
            for fn in (pack,vf.pack_supertiles_vtq):
                try:fn(*inputs,fmt)
                except ValueError:pass
                else:raise AssertionError('invalid constrained input accepted')
        rep['formats'][name]=len(c)
    for m,k in [(16,256),(128,2048),(768,768),(2048,8192),(3072,8192),(5120,17408)]:
        fmt=fmts['VTQ_300'];inputs=data(m,k,fmt,712);row=dict(m=m,k=k,runs=[])
        for i in range(4):
            sha={};tm={}
            for which in (['numpy','native'] if i%2==0 else ['native','numpy']):
                fn=pack if which=='native' else vf.pack_supertiles_vtq;t=time.perf_counter();res=fn(*inputs,fmt);tm[which]=time.perf_counter()-t;sha[which]=hashlib.sha256(res).hexdigest();del res
            assert sha['numpy']==sha['native'];row['runs'].append(tm)
        print(m,k,tm,flush=True);rep['timings'].append(row)
    a.output.write_text(json.dumps(rep,indent=2)+'\n')


if __name__=='__main__':main()
