"""Cheap output calibration in a GPTAQ encode: use already live final teacher
and student hidden streams. Only the existing BF16 final RMSNorm weight may
change. Not a claim to improve reasoning or greedy ranking through temperature.
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
from src.eval_kld import evaluate_model_on_cached_trace,load_eval_trace,paired_bootstrap_diff,precompute_ref_trace_data
from src.exp09_group_order import vtq_roundtrip
from src.run_1b_liqk_v2_experiments import restore_clean_model_weights,save_clean_model_weights
from src.vtq_format import VtqWriter,VtqFile,load_vtq_formats

MODES=('natural','temperature','feature-ls','feature-temperature')


def sample_pairs(model,cal_q,cal_r):
    # Use 128 spaced positions per sequence for diagonal moments; only 8 of
    # those for the scalar logit fit. No evaluation or held-out rows here.
    norm=model.model.norm; num=None;den=None; qs=[];rs=[]
    for j,(q,r) in enumerate(zip(cal_q,cal_r)):
        stride=max(1,q.shape[1]//128); pos=torch.arange(1+(j*13)%(max(stride-1,1)),q.shape[1],stride,device=q.device)[:128]
        hq=norm(q[:,pos]);hr=norm(r[:,pos]);qf=hq.flatten(0,1).float();rf=hr.flatten(0,1).float()
        a=(qf*rf).sum(0);b=qf.square().sum(0);num=a if num is None else num+a;den=b if den is None else den+b
        qs.append(hq[:,::16].flatten(0,1));rs.append(hr[:,::16].flatten(0,1))
    ridge=.01*den.mean().clamp_min(1e-12)
    return ((num+ridge)/(den+ridge)).clamp(.65,1.35),torch.cat(qs),torch.cat(rs)


def temperature(model,q,r,c):
    # One damped Newton step in inverse temperature for forward KL(ref || q).
    # Stream logit blocks; don't allocate an all-calibration logits tensor.
    grad=hess=0.
    for qq,rr in zip(q.split(32),r.split(32)):
        logits=model.lm_head(torch.cat([(qq.float()*c).to(qq.dtype),rr])).float();u,v=logits.chunk(2)
        p=u.softmax(-1);ref=v.softmax(-1);mean=(p*u).sum(-1)
        grad=grad+(mean-(ref*u).sum(-1)).sum();hess=hess+(p*(u-mean[:,None]).square()).sum()
    return float((1.-grad/hess.clamp_min(1e-12)).clamp(.70,1.30))


def fit_modes(model,q,r,modes):
    gamma=model.model.norm.weight.detach().clone(); dev=gamma.device
    out={'natural':gamma}; seconds={'natural':0.}; vals={};torch.cuda.synchronize();t=time.perf_counter()
    c,qs,rs=sample_pairs(model,q,r);torch.cuda.synchronize();common=time.perf_counter()-t
    for mode in modes:
        if mode=='natural':continue
        start=time.perf_counter(); cc=c if mode.startswith('feature') else torch.ones_like(c)
        s=temperature(model,qs,rs,cc) if 'temperature' in mode else 1.
        out[mode]=(gamma.float()*(cc*s)).to(gamma.dtype);torch.cuda.synchronize();seconds[mode]=common+time.perf_counter()-start
        vals[mode]=dict(inverse_temperature=s,mean_factor=float((cc*s).mean()),min_factor=float((cc*s).min()),max_factor=float((cc*s).max()))
    return out,seconds,vals


def reload_norm(path,gamma,device):
    wr=VtqWriter(path,[],reserve=2048);wr.add_tensor('model.norm.weight',gamma);wr.finalize({'quantization':{'formats':[]}})
    rd=VtqFile(path)
    try:g=rd.tensor('model.norm.weight').to(device)
    finally:rd.mm.close();rd.f.close()
    assert torch.equal(gamma,g)
    return g


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--model',required=True);ap.add_argument('--data',type=Path,default=Path('data'));ap.add_argument('--fisher',default='fisher_1b.pt');ap.add_argument('--output',type=Path,required=True);ap.add_argument('--specs',nargs='+',default=['VTQ_200','VTQ_300','VTQ_400']);ap.add_argument('--offsets',type=int,nargs='+',default=[0,1000]);ap.add_argument('--mode',choices=['discover','confirm'],required=True);ap.add_argument('--winner',choices=MODES);a=ap.parse_args()
    modes=MODES
    if a.mode=='confirm':
        if a.winner is None:ap.error('freeze winner before confirmation')
        modes=tuple(dict.fromkeys(['natural',a.winner]))
    torch.backends.cuda.matmul.allow_tf32=False;dev=torch.device('cuda');model=AutoModelForCausalLM.from_pretrained(a.model,dtype=torch.bfloat16,attn_implementation='sdpa').eval().to(dev)
    clean=save_clean_model_weights(model);tokens=load_file(str(a.data/'cal_trace.safetensors'))['input_ids'];cal=tokens[:64]
    fisher=torch.load(a.data/a.fisher,map_location='cpu',weights_only=True);vtq.load_and_register_vtq();quant=vtq.quantize_matrix_vtq
    if a.mode=='discover':
        seqs=[dict(id=f'cal_holdout_{i}',token_ids=row.tolist(),score_spans=[[1,len(row)]]) for i,row in enumerate(tokens[64:96,:512])];assert len(seqs)==32
        refs={'cal_holdout':precompute_ref_trace_data(model,seqs,dev)}
    else:refs={ds:precompute_ref_trace_data(model,load_eval_trace(a.data/f),dev) for ds,f in [('nem','nemotron_trace.json'),('mul','multilingual_trace.json')]}
    rep=dict(model=a.model,mode=a.mode,methods=modes,specs=a.specs,offsets=a.offsets,gpu=torch.cuda.get_device_name(),runs=[])
    def save():
        tmp=a.output.with_suffix('.tmp');tmp.write_text(json.dumps(rep,indent=2)+'\n');tmp.replace(a.output)
    with tempfile.TemporaryDirectory() as td:
        path=Path(td)/'w.vtq'
        for sp in a.specs:
            fmt=load_vtq_formats(menu=[sp])[sp]
            for off in a.offsets:
                restore_clean_model_weights(model,clean);vtq.PACK={};norms={};times={};vals={};count=0;io=0.
                def callback(m,q,r):
                    gs,ts,vs=fit_modes(m,q,r,modes);norms.update(gs);times.update(ts);vals.update(vs)
                def checked(w,h,*args,**kw):
                    nonlocal count,io
                    kw['seed']+=off;qd,q,loss=quant(w,h,*args,**kw);torch.cuda.synchronize();t=time.perf_counter()
                    back=vtq_roundtrip(path,vtq.PACK.pop('matrix'),fmt,'natural');assert torch.equal(q,back);torch.cuda.synchronize();io+=time.perf_counter()-t;count+=1
                    return qd,back,loss
                with patch.object(vtq,'quantize_matrix_vtq',checked):
                    info=vtq.quantize_llama_model_vtq(model,cal,sp,fisher_out=fisher,gptaq_alpha=.5,batch_size=4,d_sb_gecd_sweeps=0,polish_steps=0,final_calibration=callback)
                vtq.PACK=None
                for mode in modes:
                    model.model.norm.weight.copy_(reload_norm(path,norms[mode],dev))
                    r=dict(method=mode,spec=sp,offset=off,matrices=count,fit_s=times[mode],fit=vals.get(mode,{}),norm_bytes=norms[mode].numel()*norms[mode].element_size(),all_mode_encode_s_excl_io=info['quant_sec']-io)
                    r.update({ds:evaluate_model_on_cached_trace(model,ref,dev) for ds,ref in refs.items()});rep['runs'].append(r);save();print(sp,off,mode,times[mode],vals.get(mode,{}),{ds:r[ds]['mean_kld'] for ds in refs},flush=True)
    rep['paired']={};rep['winners']={}
    for sp in a.specs:
        by={}
        for ds in refs:
            def av(m):
                rows=[r[ds] for r in rep['runs'] if r['method']==m and r['spec']==sp];per=np.mean([r['per_seq_kld_sums'] for r in rows],axis=0);counts=rows[0]['per_seq_tok_counts'];return dict(mean_kld=float(per.sum()/sum(counts)),per_seq_kld_sums=per,per_seq_tok_counts=counts)
            ctrl=av('natural');by[ds]={}
            for mode in modes:
                cand=av(mode);d=paired_bootstrap_diff(cand,ctrl,n_boot=4000);d['rel_pct']=100*d['mean_diff']/ctrl['mean_kld'];by[ds][mode]=d
        rep['paired'][sp]=by
        if a.mode=='discover':rep['winners'][sp]=min(modes,key=lambda m:by['cal_holdout'][m]['mean_diff'])
    save()


if __name__=='__main__':main()
