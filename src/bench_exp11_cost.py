"""Matched warm PACK time/peak & reconstruction hashes, no quality ranking on synthetic data."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import statistics
import time

import torch
from src import vtq
from src.exp11_fast_schedule import schedule


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output',type=Path,required=True); args=ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32=False; vtq.load_and_register_vtq(); torch.manual_seed(647)
    methods=['natural','act-desc','fast-order','fused-only','fast-fused']; report={'gpu':torch.cuda.get_device_name(),'spec':'VTQ_200','cases':[]}
    for m,k in [(768,3072),(3072,1024),(8192,3072),(3072,8192)]:
        w=torch.randn(m,k,device='cuda')*.02; x=torch.randn(1024,k,device='cuda')*torch.rand(k,device='cuda').add_(.2)
        h=x.T@x/len(x); del x
        case={'shape':[m,k],'runs':[],'digest':{}}; report['cases'].append(case)
        rng=random.Random(785); perm=methods.copy(); rng.shuffle(perm)
        for rep in range(6):
            for mode in (methods if not rep else perm[rep-1:]+perm[:rep-1]):
                with schedule(mode):
                    vtq.PACK={}; torch.cuda.synchronize(); base=torch.cuda.memory_allocated(); torch.cuda.reset_peak_memory_stats(); start=time.perf_counter()
                    _,q,loss=vtq.quantize_matrix_vtq(w,h,'VTQ_200',seed=1200,u_sb_gecd_sweeps=0)
                    torch.cuda.synchronize(); sec=time.perf_counter()-start; peak=torch.cuda.max_memory_allocated()-base
                    if not rep:
                        dig=hashlib.sha256(); dig.update(q.contiguous().cpu().numpy()); p=vtq.PACK['matrix']
                        for key in ('d_row','suh','u_sb','g_sub','c_all','u_all'): dig.update(p[key].contiguous().cpu().numpy())
                        case['digest'][mode]=dig.hexdigest(); del p
                    else:
                        case['runs'].append(dict(rep=rep,mode=mode,ms=1000*sec,peak=peak,loss=loss))
                    vtq.PACK=None; del q
        print(m,k,[(v,round(statistics.median(r['ms'] for r in case['runs'] if r['mode']==v),2),max(r['peak'] for r in case['runs'] if r['mode']==v)) for v in methods],flush=True)
        args.output.write_text(json.dumps(report,indent=2)+'\n'); del w,h


if __name__=='__main__': main()
