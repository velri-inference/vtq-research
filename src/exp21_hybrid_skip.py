"""Frozen VTQ residual correction on real hybrid text backbones. Stock HF forward
owns DeltaNet, PLE, SDPA masks, shared KV and the dialogue scoring context.
Natural = no GPTAQ (hybrid baseline). Only the down projection target differs.
All eligible block matrices, including the token mixer and PLE, are quantized.
"""
import argparse
import hashlib
from contextlib import ExitStack
import json
from pathlib import Path
import tempfile,time
from unittest.mock import patch

import torch
import transformers
from safetensors.torch import load_file
from src import vtq
from src.eval_kld import get_scored_positions_and_targets,load_eval_trace,paired_bootstrap_diff
from src.exp09_group_order import vtq_roundtrip
from src.exp19_skip_fit import OutputCov
from src.hybrid_quant import quantizable,linear_groups
from src.streaming_quant import _seed,trace_metrics
from src.vtq_format import load_vtq_formats


class Stop(Exception):pass


def mid_module(layer):
    return layer.pre_feedforward_layernorm if hasattr(layer,'pre_feedforward_layernorm') else layer.post_attention_layernorm


def teacher(tm,cal):
    refs=[{} for _ in tm.layers]
    with ExitStack() as st:
        for l,b in enumerate(tm.layers):
            def pre(m,a,i=l):refs[i]['mid']=a[0].detach().cpu()
            def post(m,a,y,i=l):refs[i]['out']=y.detach().cpu()
            st.callback(mid_module(b).register_forward_pre_hook(pre).remove);st.callback(b.mlp.down_proj.register_forward_hook(post).remove)
        tm(input_ids=cal,use_cache=False)
    assert all(set(row)=={'mid','out'} for row in refs);return refs


def traces(tm,seqs):
    return [[tm(input_ids=ids,use_cache=False).last_hidden_state[0,pos].cpu() for ids,pos,_ in rows] for rows in seqs]


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--model',required=True);ap.add_argument('--data',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--offset',type=int,default=2000);ap.add_argument('--cal-rows',type=int,default=16);ap.add_argument('--cal-length',type=int,default=512);ap.add_argument('--spec',default='VTQ_300');ap.add_argument('--norm-pair',action='store_true',help='confirm the frozen post-norm variant against natural only');ap.add_argument('--postnorm-hessian',action='store_true',help='exp22: weight down_proj calibration by per-token postnorm sensitivity, without a teacher');ap.add_argument('--single-forward',action='store_true',help='exp23: full-calibration single batch, no prefix replays; check all file hashes');ap.add_argument('--compact-wide',action='store_true',help='exp24: compact wide-Hessian workspace in the single-forward arm');ap.add_argument('--writer-ab',action='store_true',help='exp25: time both serializers on identical native-sequential VTQ codes');ap.add_argument('--energy-tilt',type=float,default=None,help='exp29: compare a preselected row-energy tilt with the unchanged VTQ');a=ap.parse_args()
    if a.compact_wide and not a.single_forward:ap.error('--compact-wide requires --single-forward')
    torch.backends.cuda.matmul.allow_tf32=False;dev=torch.device('cuda');vtq.load_and_register_vtq();fmt=load_vtq_formats(menu=[a.spec])[a.spec];cfg=transformers.AutoConfig.from_pretrained(a.model);cls=getattr(transformers,cfg.architectures[0])
    # A global 'sdpa' is propagated to the unused Gemma audio tower, which rejects it.
    # Keep the evaluated text backend identical in the BF16 teacher and every arm.
    attn={'':'eager','audio_config':'eager','vision_config':'eager','text_config':'sdpa'} if cfg.model_type=='gemma4' else 'sdpa'
    model=cls.from_pretrained(a.model,dtype=torch.bfloat16,attn_implementation=attn).eval().to(dev);tm=model.model.language_model
    assert tm.config._attn_implementation=='sdpa',tm.config._attn_implementation
    types=list(cfg.text_config.layer_types);assert len(set(types))>=2 and len(types)==len(tm.layers)
    data=a.data;cal=load_file(str(data/'cal_trace.safetensors'))['input_ids'][:a.cal_rows,:a.cal_length].contiguous().to(dev);fisher=torch.load(data/'fisher.pt',map_location='cpu',weights_only=True);sets=['nemotron','multilingual'];seqs=[]
    for ds in sets:
        rows=[]
        for sq in load_eval_trace(data/f'{ds}_trace.json'):
            p,t=get_scored_positions_and_targets(sq)
            if p:rows.append((torch.tensor([sq['token_ids'][:8192]],device=dev),p,t))
        seqs.append(rows)
    refh=traces(tm,seqs);refs=[];teacher_s=0.
    if not (a.postnorm_hessian or a.single_forward or a.writer_ab or a.energy_tilt is not None):
        torch.cuda.synchronize();t=time.perf_counter();refs=teacher(tm,cal);torch.cuda.synchronize();teacher_s=time.perf_counter()-t
    clean={(l,n):None if a.writer_ab else m.weight.cpu().clone() for l,b in enumerate(tm.layers) for n,m in b.named_modules() if quantizable(n,m)}
    rep=dict(model=a.model,types=types,cal_shape=list(cal.shape),offset=a.offset,spec=a.spec,alpha_down=0. if (a.postnorm_hessian or a.single_forward or a.writer_ab or a.energy_tilt is not None) else .5,other_gptaq=0,teacher_s=teacher_s,teacher_cpu_bytes=sum(t.numel()*t.element_size() for row in refs for t in row.values()),runs={})
    def save():
        tmp=a.output.with_suffix('.tmp');tmp.write_text(json.dumps(rep)+'\n');tmp.replace(a.output)
    def logits(h):
        v=model.lm_head(h);cap=getattr(cfg.text_config,'final_logit_softcapping',None);return torch.tanh(v/cap)*cap if cap else v
    modes=['natural','output-only','skip-full'];soft=hasattr(tm.layers[0],'post_feedforward_layernorm')
    if soft:modes.append('skip-tangent')  # separate predeclared norm-aware variant; report literal skip-full too
    if a.norm_pair:
        if not soft:raise ValueError('post-MLP normalization required')
        modes=['natural','skip-tangent']
    order=modes
    if a.postnorm_hessian:
        assert soft and not a.norm_pair
        modes=['natural','postnorm-H','shuffled-H'];j=(a.offset//1000-5)%3;order=modes[j:]+modes[:j]
        rep['hessian_protocol']='inverse token mean square before post-MLP RMSNorm, relative clamp [1/4,4], renormalized; control rolls by half sequence'
    if a.single_forward:
        from src.exp23_once import single_call_layer
        assert not a.postnorm_hessian and not a.norm_pair
        modes=['natural','single-forward'];order=modes if a.offset%2000==0 else modes[::-1]
        if a.compact_wide:rep['compact_wide_workspace']=dict(min_k=8192,stripe=1024,overwrite_triangular_rhs=True)
    if a.energy_tilt is not None:
        assert a.energy_tilt in (-.5,.5,1.) and not (a.norm_pair or a.postnorm_hessian or a.single_forward or a.writer_ab)
        modes=['natural','energy-tilt'];order=modes if (a.offset//1000)%2 else modes[::-1];rep['frozen_energy_tilt']=a.energy_tilt
    rt=vtq_roundtrip
    if a.writer_ab:
        from src.exp25_writer_ab import PairedWriter
        assert not (a.norm_pair or a.postnorm_hessian or a.single_forward)
        modes=order=['natural'];rt=PairedWriter('/content/libexp25pack.so')
    for method in order:
        drift=method in ('output-only','skip-full','skip-tangent')
        if not a.writer_ab:
            for (l,n),w in clean.items():tm.layers[l].get_submodule(n).weight.copy_(w)
        done=set();stats={'bytes':0,'params':0,'file_bytes':0,'io':0.,'bf16_unexecuted':[]};vtq.PACK={}
        with tempfile.TemporaryDirectory() as td, ExitStack() as handlers:
            path=Path(td)/'matrix.vtq'
            def encode(l,n,mod,H,target=None):
                w=mod.weight;key=(l,n);assert key not in done and key in fisher,(key,'Fisher missing or duplicate');done.add(key)
                _,q,_=vtq.quantize_matrix_vtq(w,H,a.spec,seed=_seed(l,n)+a.offset,damp=.30 if n=='mlp.down_proj' else .20,row_weights=fisher[key].to(dev),W_target=target,u_sb_gecd_sweeps=0,compact_wide_workspace=(a.compact_wide and method=='single-forward'),**({'row_energy_tilt':a.energy_tilt} if method=='energy-tilt' else {}))
                torch.cuda.synchronize();start=time.perf_counter();back=rt(path,vtq.PACK.pop('matrix'),fmt,method);assert torch.equal(q,back)
                if a.single_forward:stats.setdefault('sha256',{})[f'{l}/{n}']=hashlib.sha256(path.read_bytes()).hexdigest()
                torch.cuda.synchronize();stats['io']+=time.perf_counter()-start;stats['file_bytes']+=path.stat().st_size;stats['bytes']+=fmt.st_bytes*(w.shape[0]//16)*(w.shape[1]//256)+2*sum(w.shape);stats['params']+=w.numel();w.copy_(back)
            def wrap(layer,l,orig):
                def forward(*args,**kw):
                    def run():return orig(*args,**kw)
                    if method=='single-forward':return single_call_layer(layer,run,lambda n,h:encode(l,n,layer.get_submodule(n),h))
                    token_scale=None
                    if method in ('postnorm-H','shuffled-H'):
                        def norm_scale(m,inputs):
                            nonlocal token_scale
                            # Use the existing group discovery forward, not a second pass.
                            h=inputs[0];g=(torch.linalg.vector_norm(h,dim=-1,keepdim=True,dtype=torch.float32).square()/h.shape[-1]+m.eps).reciprocal()
                            g=(g/g.mean()).clamp_(.25,4.);token_scale=(g/g.mean()).sqrt_()
                            if method=='shuffled-H':token_scale=token_scale.roll(token_scale.shape[1]//2,dims=1)
                        hh=layer.post_feedforward_layernorm.register_forward_pre_hook(norm_scale)
                        try:groups=linear_groups(layer,run)
                        finally:hh.remove()
                        assert token_scale is not None
                    else:groups=linear_groups(layer,run)
                    mods=dict(layer.named_modules())
                    for group in groups:
                        nm=group[0];down=nm=='mlp.down_proj';cache={}
                        def grab(m,v):cache['z']=v[0];raise Stop()
                        def mid(m,v):cache['mid']=v[0]
                        with ExitStack() as capture:
                            if down and drift:capture.callback(mid_module(layer).register_forward_pre_hook(mid).remove)
                            capture.callback(mods[nm].register_forward_pre_hook(grab).remove)
                            try:run()
                            except Stop:pass
                        z=cache['z'];target=None;acc=OutputCov(0.)
                        if down and drift:
                            assert group==['mlp.down_proj'];yq=mods[nm](z);yr=refs[l]['out'].to(dev);qm=cache['mid'];rm=refs[l]['mid'].to(dev);err=yr.float()-yq.float()
                            if method!='output-only':
                                d=rm.float()-qm.float()
                                if method=='skip-tangent':
                                    # Pull the desired post-RMSNorm skip change back into the
                                    # tangent plane at yq. Damped inverse for tiny norm gains.
                                    norm=layer.post_feedforward_layernorm;w=norm.weight.float();rms=(yq.float().square().mean(-1,keepdim=True)+norm.eps).sqrt();unit=yq.float()/rms
                                    d=d*(w/(w.square()+.01*w.square().mean()))*rms;d=d-unit*(unit*d).mean(-1,keepdim=True)
                                err.add_(d)
                            x=z.reshape(-1,z.shape[-1]).float();acc.add(z,None);acc.sum_qc=(x.T@err.reshape(-1,err.shape[-1])).double();H,delta=acc.get(.30);target=mods[nm].weight.float()+.5*delta
                        else:
                            if down and token_scale is not None:z=z.float()*token_scale
                            acc.add(z,None);H,_=vtq.DualCovAccumulator.get(acc,.20)
                        del acc
                        for n in group:encode(l,n,mods[n],H,target)
                    if l%8==0:print(method,l,types[l],len(done),flush=True)
                    return run()
                return forward
            for l,b in enumerate(tm.layers):handlers.enter_context(patch.object(b,'forward',wrap(b,l,b.forward)))
            torch.cuda.synchronize();base=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats();t=time.perf_counter();tm(input_ids=cal,use_cache=False);torch.cuda.synchronize();secs=time.perf_counter()-t;peak=torch.cuda.max_memory_allocated()-base
        vtq.PACK=None;assert sum(n=='mlp.down_proj' for _,n in done)==len(types);miss=set(clean)-done
        # Some shared-KV layers declare key/value modules but never execute them.
        stats['bf16_unexecuted']=sorted(miss);stats.update(matrices=len(done),quantized_bpw=8*stats['bytes']/stats['params'],encode_s=secs-stats['io']+(teacher_s if drift else 0),peak_extra_student_incl_roundtrip=peak)
        hs=traces(tm,seqs)
        for ds,hh,rr,rows in zip(sets,hs,refh,seqs):stats[ds]=trace_metrics(model,rr,hh,[r[2] for r in rows],dev,logits_fn=logits,chunk=256)
        rep['runs'][method]=stats;save();print(method,len(done),stats['quantized_bpw'],[stats[s]['mean_kld'] for s in sets],flush=True)
    ctrl=rep['runs']['natural'];rep['paired']={}
    if a.writer_ab:
        rep['writer']=rt.report();assert rep['writer']['matrices']==ctrl['matrices']
    if a.single_forward:
        other=rep['runs']['single-forward']['sha256'];rep['sha_mismatches']=[k for k in sorted(ctrl['sha256'].keys()|other.keys()) if ctrl['sha256'].get(k)!=other.get(k)]
    for m in modes[1:]:
        v=rep['runs'][m];assert (v['file_bytes'],v['bytes'],v['bf16_unexecuted'])==(ctrl['file_bytes'],ctrl['bytes'],ctrl['bf16_unexecuted']);rep['paired'][m]={}
        for ds in sets:
            p=paired_bootstrap_diff(v[ds],ctrl[ds],n_boot=4000);p['rel_pct']=100*p['mean_diff']/ctrl[ds]['mean_kld'];rep['paired'][m][ds]=p
    save()


if __name__=='__main__':main()
