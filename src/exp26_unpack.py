"""Test Cu-free CPU materialization of W_rot in the file loader. Neither the
CUDA nor AVX2 inference matvec is replaced. Native operates on the same bytes
and must match the vectorized decoder, including partial tiles and every format.
"""
import argparse,ctypes as ct,json,time
from pathlib import Path
import numpy as np
from src import vtq_format as vf
from src.exp25_pack import data
from src.vtq_v2_format import HAM_CW8


class NativeUnpack:
    def __init__(self,lib):
        self.so=ct.CDLL(str(Path(lib).resolve()));self.fn=self.so.vtq_unpack;self.fn.argtypes=[ct.c_void_p,ct.c_size_t,ct.c_void_p,ct.c_size_t]+[ct.c_void_p]*5+[ct.c_int]*6;self.fn.restype=ct.c_int;self.labels={nl:np.ascontiguousarray(vf.galois_lb_vtq(np.arange(1024),nl),dtype=np.uint8) for nl in (4,8,16)};self.ham=np.ascontiguousarray(HAM_CW8,dtype=np.uint8)
    def __call__(self,raw,M,K,fmt,d_row=None,*,gain_luts=None):
        if gain_luts is not None:raise ValueError('VTQ only: no gain-conditioned table support')
        if not 0<M<2**31 or not 0<K<2**31 or K%256 or vf.FMT_META[fmt.fmt_code]!=(fmt.K,fmt.gosset,fmt.nl,fmt.h,fmt.st_bytes):raise ValueError('invalid VTQ geometry')
        buf=np.frombuffer(raw,dtype=np.uint8);dr=np.ones(M,np.float32) if d_row is None else np.ascontiguousarray(np.asarray(d_row,dtype=np.float16),dtype=np.float32).reshape(M);tab=np.ascontiguousarray(fmt.lut,dtype=np.float32);g=np.ascontiguousarray(fmt.gain_tab,dtype=np.float32)
        if tab.size!=fmt.nl*fmt.U or g.size!=1<<fmt.h:raise ValueError('invalid lookup size')
        if buf.nbytes<((M+15)//16)*(K//256)*fmt.st_bytes:raise ValueError('truncated supertiles')
        out=np.empty((M,K),np.float32);args=[v.ctypes.data for v in (tab,g,self.labels[fmt.nl],self.ham,dr)];err=self.fn(buf.ctypes.data,buf.nbytes,out.ctypes.data,out.size,*args,M,K//256,fmt.K,fmt.gosset,fmt.h,fmt.st_bytes)
        if err:raise ValueError(f'unpack status {err}')
        return out


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--lib',required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();dec=NativeUnpack(a.lib);fs=vf.load_vtq_formats();r=dict(formats=[],times=[]);rng=np.random.default_rng(1026)
    for name,fmt in fs.items():
        for n,k in [(1,256),(15,512),(17,768),(48,1024)]:
            inp=data(n,k,fmt,172+n);raw=vf.pack_supertiles_vtq(*inp,fmt);dr=rng.normal(size=n).astype(np.float16)
            for d in [None,dr]:
                x=vf.decode_supertiles_vtq(raw,n,k,fmt,d);y=dec(raw,n,k,fmt,d);assert np.array_equal(x.view('u4'),y.view('u4')),(name,n,np.abs(x-y).max())
            try:dec(raw[:-1],n,k,fmt,dr)
            except ValueError:pass
            else:raise AssertionError('truncation accepted')
        r['formats'].append(name)
    for n,k in [(128,2048),(768,768),(2048,8192),(5120,17408)]:
        fmt=fs['VTQ_300'];raw=vf.pack_supertiles_vtq(*data(n,k,fmt,184),fmt);dr=rng.normal(size=n).astype(np.float16);row=dict(m=n,k=k,runs=[])
        for i in range(4):
            times={};ref=None
            for mode in (['numpy','native'] if i%2==0 else ['native','numpy']):
                f=dec if mode=='native' else vf.decode_supertiles_vtq;t=time.perf_counter();x=f(raw,n,k,fmt,dr);times[mode]=time.perf_counter()-t
                if ref is not None:assert np.array_equal(x.view('u4'),ref.view('u4'))
                else:ref=x
            del x,ref;row['runs'].append(times)
        r['times'].append(row);print(n,k,times,flush=True)
    a.output.write_text(json.dumps(r,indent=2)+'\n')


if __name__=='__main__':main()
