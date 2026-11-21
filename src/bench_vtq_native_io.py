"""Verify the public reader on a directory of actual exp26 samples: each is the
largest packed matrix retained for that particular model. This supplements,
not replaces, the all-matrix materialization test. Warm orders alternate.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path
import torch
from src.vtq_format import VtqFile
from src.vtq_native import NativeVtqCodec


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--library',required=True,type=Path);ap.add_argument('--samples',required=True,type=Path);ap.add_argument('--output',required=True,type=Path);a=ap.parse_args();codec=NativeVtqCodec(a.library);families=['llama1','llama3','llama8','qwen8','qwen35','gemma4','qwen27','moe30','gpt2','qwen'];report={}
    for family in families:
        path=a.samples/family/'w.vtq';rows=[];expected=None
        for trial in range(3):
            order=['numpy','native'] if trial%2==0 else ['native','numpy'];tm={}
            for name in order:
                torch.cuda.synchronize();t=time.perf_counter();rd=VtqFile(path,codec=codec if name=='native' else None)
                try:w=rd.decode_matrix('w',device='cuda')
                finally:rd.mm.close();rd.f.close()
                torch.cuda.synchronize();tm[name]=time.perf_counter()-t;shape=list(w.shape);sha=hashlib.sha256(memoryview(w.cpu().contiguous().numpy()).cast('B')).hexdigest();del w
                if expected is None:expected=sha
                assert sha==expected,(family,trial,name)
            rows.append(dict(order=order,times=tm))
        report[family]=dict(shape=shape,bytes=path.stat().st_size,sha256_fp32=expected,runs=rows);tmp=a.output.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2)+'\n');tmp.replace(a.output);print(family,rows[-1]['times'],flush=True)

if __name__=='__main__':main()
