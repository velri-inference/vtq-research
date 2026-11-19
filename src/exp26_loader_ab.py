"""Measure two complete Python loader calls on identical actual .vtq files.
Never label file materialization (CPU unpack + H2D + inverse rotation) as a
benchmark of inference matvec. Keep just the largest actual sample per model
for comparable isolated-process RSS tests of both writer and reader.
"""
import argparse,hashlib,json,shutil,time
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from src import vtq_format as vf
from src.exp25_pack import NativePack
from src.exp26_unpack import NativeUnpack


class LoaderAB:
    def __init__(self,sample):
        self.sample=Path(sample);self.sample.mkdir(parents=True,exist_ok=True);self.big=0;self.native=NativeUnpack('/content/libexp26_unpack.so');self.base=vf.decode_supertiles_vtq
    def check(self,path,p,fmt,i):
        m,k=p['u_all'].shape[:2];k*=256
        if m*k>self.big:
            # Not in either timer. ndarray files avoid a second large in-memory archive.
            for key in ['c_all','u_all','g_sub','u_sb','d_row','suh']:
                x=p[key];arr=x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x);np.save(self.sample/(key+'.npy'),arr,allow_pickle=False)
            shutil.copyfile(path,self.sample/'w.vtq');(self.sample/'info.json').write_text(json.dumps(dict(m=m,k=k,fmt=fmt.name)));self.big=m*k
        times={};peaks={};hashes={};order=['native','numpy'] if i%2==0 else ['numpy','native']
        for j,mode in enumerate(order):
            torch.cuda.synchronize();base=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats();t=time.perf_counter()
            with patch.object(vf,'decode_supertiles_vtq',self.native if mode=='native' else self.base):
                rd=vf.VtqFile(path)
                try:out=rd.decode_matrix('w',device='cuda')
                finally:rd.mm.close();rd.f.close()
            torch.cuda.synchronize();times[mode]=time.perf_counter()-t;peaks[mode]=torch.cuda.max_memory_allocated()-base
            hashes[mode]=hashlib.sha256(memoryview(out.contiguous().cpu().numpy()).cast('B')).hexdigest()
            if j==0:del out
        assert hashes['numpy']==hashes['native'],(fmt.name,i,hashes)
        return out,dict(times=times,extra_gpu_bytes=peaks,output_sha256=hashes['native'])


def rss():
    return {k:int(v.split()[0])*1024 for line in Path('/proc/self/status').read_text().splitlines() if ':' in line for k,v in [line.split(':',1)] if k in ('VmRSS','VmHWM')}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--sample',type=Path,required=True);ap.add_argument('--action',choices=['pack','unpack'],required=True);ap.add_argument('--mode',choices=['numpy','native'],required=True);a=ap.parse_args();info=json.loads((a.sample/'info.json').read_text());fmt=vf.load_vtq_formats(menu=[info['fmt']])[info['fmt']];size=fmt.st_bytes*((info['m']+15)//16)*(info['k']//256)
    if a.action=='pack':
        inputs=[np.load(a.sample/(key+'.npy'),allow_pickle=False) for key in ['c_all','u_all','g_sub','u_sb']];fn=vf.pack_supertiles_vtq if a.mode=='numpy' else NativePack('/content/libexp25pack.so');call=lambda:fn(*inputs,fmt)
    else:
        rd=vf.VtqFile(a.sample/'w.vtq');slices=rd.tensor('w.vtq.slices').tolist();assert len(slices)==1 and slices[0][:2]==[0,info['m']];dr=rd.tensor('w.vtq.d_row').numpy();raw=bytes(rd.raw('w.vtq.data')[:size]);rd.mm.close();rd.f.close();fn=vf.decode_supertiles_vtq if a.mode=='numpy' else NativeUnpack('/content/libexp26_unpack.so');call=lambda:fn(raw,info['m'],info['k'],fmt,dr)
    # Reset only this worker's RSS high-water mark, after inputs are resident.
    Path('/proc/self/clear_refs').write_text('5');before=rss();t=time.perf_counter();out=call();sec=time.perf_counter()-t;after=rss();sha=hashlib.sha256(memoryview(out).cast('B')).hexdigest();print(json.dumps(dict(action=a.action,mode=a.mode,sec=sec,before=before,after=after,extra_peak=after['VmHWM']-before['VmRSS'],sha=sha,**info)))


if __name__=='__main__':main()
