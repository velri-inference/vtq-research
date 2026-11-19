"""Independent full-file builds (one backend per fresh process). Timed from
quantization through serialization of every unchanged source tensor and fsync.
Hash after timer. This measures export, not an inference backend or quality gain.
"""
import argparse,contextlib,hashlib,json,os,shutil,tempfile,time
from pathlib import Path
from unittest.mock import patch
import torch,transformers
from safetensors.torch import load_file
from src import vtq,vtq_format as vf,streaming_quant as sq
from src.exp25_pack import NativePack
from src.hybrid_quant import linear_groups
from src.exp21_hybrid_skip import Stop

DENSE={'llama1','llama3','llama8','qwen8'}
HYBRID={'qwen35','gemma4','qwen27'}
N={'llama1':112,'llama3':196,'llama8':224,'qwen8':252,'qwen35':200,'gemma4':342,'qwen27':400,'moe30':12480}


def rss():
    return {k:int(v.split()[0])*1024 for ln in Path('/proc/self/status').read_text().splitlines() if ':' in ln for k,v in [ln.split(':',1)] if k in ('VmRSS','VmHWM')}


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--family',choices=sorted(N),required=True);ap.add_argument('--backend',choices=['numpy','native'],required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--offset',type=int,default=13000);ap.add_argument('--stream-unchanged',action='store_true');ap.add_argument('--overlap-unchanged',action='store_true',help='plan source suffix, fill concurrently and release streamed source shards');a=ap.parse_args();f=a.family;root=Path('/content/trellis-night');inv=json.loads((root/'results/exp25_model_inventory.json').read_text());model=Path(inv[f]['path']);spec='VTQ_300';dev=torch.device('cuda');torch.backends.cuda.matmul.allow_tf32=False;vtq.load_and_register_vtq();fmt=vf.load_vtq_formats(menu=[spec])[spec];store=sq.TensorStore(model);prefix='model.language_model' if store.has('model.language_model.embed_tokens.weight') else 'model';assert store.has(prefix+'.embed_tokens.weight')
    data=root/'data';data=data/({'qwen8':'qwen3','moe30':'qwen3','qwen27':'qwen38','qwen35':'qwen35','gemma4':'gemma4'}.get(f,''));fp=data/({'llama1':'fisher_1b.pt','llama3':'fisher_3b.pt','llama8':'fisher_llama8b.pt','moe30':'fisher_30b.pt'}.get(f,'fisher.pt'));fisher=torch.load(fp,map_location='cpu',weights_only=True);cal=load_file(str(data/'cal_trace.safetensors'))['input_ids'][:64];sd=a.offset
    if f in HYBRID:
        cal=cal[:16,:512].contiguous().to(dev);cfg=transformers.AutoConfig.from_pretrained(model);cls=getattr(transformers,cfg.architectures[0]);attn={'':'eager','audio_config':'eager','vision_config':'eager','text_config':'sdpa'} if f=='gemma4' else 'sdpa';net=cls.from_pretrained(model,dtype=torch.bfloat16,attn_implementation=attn).eval().to(dev);tm=net.model.language_model
    elif f in DENSE:net=transformers.AutoModelForCausalLM.from_pretrained(model,dtype=torch.bfloat16,attn_implementation='sdpa').eval().to(dev)
    else:net,store,cfg=sq.build_skeleton(model,dev);sq.FISHER=fisher
    base_pack=vf.pack_supertiles_vtq if a.backend=='numpy' else NativePack('/content/libexp25pack.so');orig=vtq.quantize_matrix_vtq;vtq.PACK={};torch.cuda.manual_seed(417);w=torch.randn(128,256,device=dev);orig(w,torch.eye(256,device=dev),spec,u_sb_gecd_sweeps=0);probe=vtq.PACK.pop('matrix');cp=[probe[k].cpu().numpy() for k in ('c_all','u_all','g_sub','u_sb')];base_pack(*cp,fmt);del w,probe,cp;torch.cuda.synchronize()
    files={v for v in store.files.values()};need=sum(p.stat().st_size for p in files);assert shutil.disk_usage(root).free>0.75*need+2e9,'not enough space for complete export';seen=set();skip=set();wsec=0.;count_unchanged=0;unc_bytes=0
    with tempfile.TemporaryDirectory(dir='/content') as td,patch.object(vf,'pack_supertiles_vtq',base_pack),contextlib.ExitStack() as bg:
        path=Path(td)/'model.vtq';Path('/proc/self/clear_refs').write_text('5');before=rss();base=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats();t0=time.perf_counter();wr=vf.VtqWriter(path,[fmt],reserve=(16<<20)-16)
        finish=None
        if a.overlap_unchanged:
            from src.exp31_parallel_copy import begin_copy
            assert not a.stream_unchanged and len(fisher)==N[f]
            layers=tm.layers if f in HYBRID else net.model.layers
            finish=begin_copy(bg,wr,path,f,store,prefix,layers,fisher,fmt)
        def emit(i,n):
            nonlocal wsec
            key=f'{prefix}.layers.{i}.{n}';assert key not in seen;seen.add(key);start=time.perf_counter();wr.add_matrix(key,vtq.PACK.pop('matrix'),verify=False);torch.cuda.synchronize();wsec+=time.perf_counter()-start
            if f=='moe30' and n.startswith('mlp.experts.'):
                _,_,e,kind=n.split('.');fused=f'{prefix}.layers.{i}.mlp.experts.{kind}'
                if store.has(fused):skip.add(fused)
                elif kind=='gate_up_proj':skip.update(key.rsplit('.',1)[0]+'.'+p+'.weight' for p in ('gate_proj','up_proj'))
                else:skip.add(key+'.weight')
            else:skip.add(key+'.weight')
        if f in DENSE:
            subs=['self_attn.q_proj','self_attn.k_proj','self_attn.v_proj','self_attn.o_proj','mlp.gate_proj','mlp.up_proj','mlp.down_proj']
            def enc(w,h,*args,**kw):
                i,j=divmod(len(seen),7);kw['seed']+=sd;r=orig(w,h,*args,**kw);emit(i,subs[j]);return r
            with patch.object(vtq,'quantize_matrix_vtq',enc):vtq.quantize_llama_model_vtq(net,cal,spec,fisher_out=fisher,batch_size=4,gptaq_alpha=None,d_sb_gecd_sweeps=0,polish_steps=0)
        elif f in HYBRID:
            def block(b,i,fn):
                def wrapped(*args,**kwargs):
                    run=lambda:fn(*args,**kwargs)
                    for names in linear_groups(b,run):
                        z=[]
                        def capture(_,ins):z.append(ins[0]);raise Stop()
                        hook=b.get_submodule(names[0]).register_forward_pre_hook(capture)
                        try:run()
                        except Stop:pass
                        finally:hook.remove()
                        acc=vtq.DualCovAccumulator(False);acc.add(z.pop());h,_=acc.get();del acc
                        for n in names:
                            w=b.get_submodule(n).weight;_,q,_=orig(w,h,spec,row_weights=fisher[i,n].to(dev),seed=sq._seed(i,n)+sd,damp=.3 if n=='mlp.down_proj' else .2,u_sb_gecd_sweeps=0);emit(i,n);w.copy_(q);del q
                        del h
                    return run()
                return wrapped
            with contextlib.ExitStack() as st:
                for i,b in enumerate(tm.layers):st.enter_context(patch.object(b,'forward',block(b,i,b.forward)))
                tm(input_ids=cal,use_cache=False)
        else:
            bs=4;hs=[net.model.embed_tokens(z.to(dev)) for z in cal.split(bs)];pos=net.model.rotary_emb(hs[0][:1],torch.arange(cal.shape[1],device=dev)[None,:])
            def enc(w,h,sp,seed,damp,full):
                i=sq._CUR_LAYER[0];_,q,_=orig(w,h,sp,seed=seed+sd,damp=damp,row_weights=fisher[(i,full)].to(dev),u_sb_gecd_sweeps=0);emit(i,full);return q.to(w.dtype)
            with patch.object(sq,'_quantize_linear_weight',enc):
                for i,b in enumerate(net.model.layers):
                    sq.load_module_weights(b,f'{prefix}.layers.{i}',store,dev)
                    if a.overlap_unchanged:
                        torch.cuda.synchronize();store._handles.clear()
                    sq._CUR_LAYER[0]=i;sq.quantize_layer_v2(b,hs,pos,spec,i);net.model.layers[i]=b.to('meta')
        torch.cuda.synchronize();quant_write_s=time.perf_counter()-t0;assert len(seen)==N[f] and skip.issubset(set(store.keys())),(len(seen),list(skip-set(store.keys()))[:3]);qpeak=torch.cuda.max_memory_allocated()-base
        unchanged=sorted(set(store.keys())-skip)
        if finish is not None:
            count_unchanged,unc_bytes=finish(skip)
        elif a.stream_unchanged:
            from src.vtq_source_copy import FrozenTensorCopier
            with FrozenTensorCopier() as copier:
                for key in unchanged:unc_bytes+=copier.add(wr,key,store.files[key]);count_unchanged+=1
        else:
            for key in unchanged:
                tensor=store.get(key,'cpu');wr.add_tensor(key,tensor);count_unchanged+=1;unc_bytes+=tensor.numel()*tensor.element_size();del tensor
        wr.finalize({'source_revision':inv[f]['sha'],'quantization':{'formats':[vf.format_json_vtq(fmt)],'linear_matrices':len(seen),'unchanged_tensors':count_unchanged,'rot_block':-1}})
        with path.open('rb') as stream:os.fsync(stream.fileno())
        export_s=time.perf_counter()-t0;end=rss();size=path.stat().st_size
        with path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
    r=dict(family=f,spec=spec,backend=a.backend,stream_unchanged=a.stream_unchanged,overlap_unchanged=a.overlap_unchanged,source_revision=inv[f]['sha'],seed_offset=sd,cal_shape=list(cal.shape),matrices=len(seen),source_keys=len(store.files),replaced_source_keys=len(skip),unchanged_tensors=count_unchanged,unchanged_tensor_bytes=unc_bytes,file_bytes=size,sha256=digest,matrix_writer_s=wsec,quant_plus_matrix_writes_s=quant_write_s,complete_export_fsync_s=export_s,peak_extra_cuda_bytes=qpeak,rss_before=before,rss_after=end,peak_extra_rss=end['VmHWM']-before['VmRSS']);a.output.write_text(json.dumps(r,indent=2)+'\n');print(r,flush=True)


if __name__=='__main__':main()
