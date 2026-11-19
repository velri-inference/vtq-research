"""Frozen 2-bit candidate vs deployed natural order on the project's independent dialogue traces.

Uses identical calibration tokens, cached Fisher row weights, GPTAQ, and no
polish/DP in both arms. Every linear matrix is written, reloaded, compared
bitwise, then the reloaded weights actually drive the rest of the model.
"""
import argparse
import json
from pathlib import Path
import tempfile
import time
from unittest.mock import patch

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM

from src import vtq
from src.eval_kld import (evaluate_model_on_cached_trace, load_eval_trace,
                          paired_bootstrap_diff, precompute_ref_trace_data)
from src.exp09_group_order import vtq_roundtrip
from src.exp11_fast_schedule import schedule
from src.run_1b_liqk_v2_experiments import restore_clean_model_weights, save_clean_model_weights
from src.vtq_format import load_vtq_formats


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model',required=True); ap.add_argument('--data',type=Path,default=Path('data'))
    ap.add_argument('--fisher',default='fisher_1b.pt'); ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--cal-seqs',type=int,default=64); ap.add_argument('--offsets',nargs='+',type=int,default=[0,1000,2000]); args=ap.parse_args()
    run_transfer(args, ('natural','fast-fused'), schedule, 'VTQ_200')


@torch.inference_mode()
def run_transfer(args, methods, context, sp, *, discovery=False, provenance=None):
    torch.backends.cuda.matmul.allow_tf32=False; dev=torch.device('cuda')
    m=AutoModelForCausalLM.from_pretrained(args.model,dtype=torch.bfloat16,attn_implementation='sdpa').eval().to(dev)
    clean=save_clean_model_weights(m); vtq.load_and_register_vtq(); fmt=load_vtq_formats(menu=[sp])[sp]
    all_cal=load_file(str(args.data/'cal_trace.safetensors'))['input_ids']; h=all_cal[:args.cal_seqs]
    fisher=torch.load(args.data/args.fisher,map_location='cpu',weights_only=True)
    if discovery:
        seqs=[dict(id=f'cal_holdout_{i}', token_ids=row.tolist(),score_spans=[[1,len(row)]])
              for i,row in enumerate(all_cal[args.cal_seqs:args.cal_seqs+32,:512])]
        assert len(seqs)==32, 'held-out calibration must not overlap fitting tokens'
        ref={'cal_holdout':precompute_ref_trace_data(m,seqs,dev)}
    else:
        ref={ds:precompute_ref_trace_data(m,load_eval_trace(args.data/file),dev) for ds,file in [('nem','nemotron_trace.json'),('mul','multilingual_trace.json')]}
    rep=dict(model=args.model,spec=sp,cal_shape=list(h.shape),offsets=args.offsets,gpu=torch.cuda.get_device_name(),torch=torch.__version__,runs=[])
    if provenance is not None:rep.update(provenance=provenance,methods=list(methods),discovery=discovery)
    if args.output.exists():
        old=json.loads(args.output.read_text()); assert all(old[k]==v for k,v in rep.items() if k!='runs'); rep=old
    def save():
        tmp=args.output.with_suffix('.tmp'); tmp.write_text(json.dumps(rep,indent=2)+'\n'); tmp.replace(args.output)
    quant=vtq.quantize_matrix_vtq
    with tempfile.TemporaryDirectory() as td:
        path=Path(td)/'m.vtq'
        for it,off in enumerate(args.offsets):
            for method in methods[it%len(methods):]+methods[:it%len(methods)]:
                if any(r['method']==method and r['offset']==off for r in rep['runs']): continue
                restore_clean_model_weights(m,clean); vtq.PACK={}; secs=0.; io=0.; count=0; params=0; payload=0
                def checked(w,H,*a,**kw):
                    nonlocal secs,io,count,params,payload
                    kw['seed']+=off
                    torch.cuda.synchronize(); t=time.perf_counter(); qd,q,loss=quant(w,H,*a,**kw); torch.cuda.synchronize(); secs+=time.perf_counter()-t
                    t=time.perf_counter(); p=vtq.PACK.pop('matrix'); back=vtq_roundtrip(path,p,fmt,method); assert torch.equal(back,q), (method,off,count)
                    torch.cuda.synchronize(); io+=time.perf_counter()-t; count+=1; params+=w.numel(); payload+=fmt.st_bytes*(w.shape[0]//16)*(w.shape[1]//256)+2*sum(w.shape)
                    return qd,back,loss
                torch.cuda.synchronize(); base=torch.cuda.memory_allocated(); torch.cuda.reset_peak_memory_stats(); start=time.perf_counter()
                with context(method),patch.object(vtq,'quantize_matrix_vtq',checked):
                    vtq.quantize_llama_model_vtq(m,h,sp,fisher_out=fisher,batch_size=4,gptaq_alpha=.5,d_sb_gecd_sweeps=0,polish_steps=0)
                torch.cuda.synchronize(); wall=time.perf_counter()-start; peak=torch.cuda.max_memory_allocated()-base;vtq.PACK=None
                r=dict(offset=off,method=method,matrices=count,linear_bpw=8*payload/params,matrix_encode_s=secs,
                       roundtrip_s=io,model_encode_s=wall-io,wall_s=wall,peak_extra_incl_roundtrip=peak)
                r.update({ds:evaluate_model_on_cached_trace(m,rr,dev) for ds,rr in ref.items()});rep['runs'].append(r);save()
                print(off,method,count,r['linear_bpw'],secs,wall-io,peak,{ds:r[ds]['mean_kld'] for ds in ref},flush=True)
    rep['paired']={}
    for ds in ref:
        def av(method):
            rows=[r[ds] for r in rep['runs'] if r['method']==method]
            counts=rows[0]['per_seq_tok_counts']; assert all(r['per_seq_tok_counts']==counts for r in rows)
            sums=np.mean([r['per_seq_kld_sums'] for r in rows],axis=0)
            return dict(mean_kld=float(sums.sum()/sum(counts)),per_seq_kld_sums=sums,per_seq_tok_counts=counts)
        ctrl=av('natural');pairs={}
        for method in methods:
            if method=='natural':continue
            pair=paired_bootstrap_diff(av(method),ctrl,n_boot=4000);pair['rel_pct']=100*pair['mean_diff']/ctrl['mean_kld'];pairs[method]=pair
        rep['paired'][ds]=pairs if provenance is not None else next(iter(pairs.values()))
    if discovery:rep['winner']=min(methods,key=lambda mm:np.mean([r['cal_holdout']['mean_kld'] for r in rep['runs'] if r['method']==mm]))
    save()


if __name__=='__main__': main()
