"""Primary dense-model writer gate: production scalar-rate GPTAQ recipe, sequential
calibration and model-specific Fisher. Both serializers see each actual code.
"""
import argparse,json,tempfile,time
from pathlib import Path
from unittest.mock import patch
import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM
from src import vtq,liqk_v2 as lv2
from src.eval_kld import get_scored_positions_and_targets,load_eval_trace
from src.streaming_quant import trace_metrics
from src.exp25_writer_ab import PairedWriter
from src.vtq_format import load_vtq_formats


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',required=True);ap.add_argument('--data',type=Path,required=True);ap.add_argument('--fisher',required=True);ap.add_argument('--spec',default='VTQ_300');ap.add_argument('--output',required=True,type=Path);a=ap.parse_args();dev=torch.device('cuda');torch.backends.cuda.matmul.allow_tf32=False
    m=AutoModelForCausalLM.from_pretrained(a.model,dtype=torch.bfloat16,attn_implementation='sdpa').eval().to(dev);cal=load_file(str(a.data/'cal_trace.safetensors'))['input_ids'][:64];vtq.load_and_register_vtq();fp=a.data/a.fisher
    if not fp.exists():
        fisher=lv2.compute_exact_row_fisher_sensitivities(m,cal,n_Menu_seqs=8,seq_len=1024);torch.save({k:v.cpu() for k,v in fisher.items()},fp);del fisher
    fisher=torch.load(fp,map_location='cpu',weights_only=True);scores={};seqs={}
    def features(rows):return [m.model(input_ids=ids,use_cache=False).last_hidden_state[0,pos].cpu() for ids,pos,t in rows]
    with torch.inference_mode():
        for ds in ['nemotron','multilingual']:
            rows=[]
            for seq in load_eval_trace(a.data/(ds+'_trace.json')):
                pos,t=get_scored_positions_and_targets(seq)
                if pos:rows.append((torch.tensor([seq['token_ids'][:8192]],device=dev),pos,t))
            seqs[ds]=rows;scores[ds]=features(rows)
        orig=vtq.quantize_matrix_vtq;fmt=load_vtq_formats(menu=[a.spec])[a.spec];pw=PairedWriter('/content/libexp25pack.so');vtq.PACK={};overhead=0.
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'w.vtq'
            def encode(*args,**kw):
                nonlocal overhead
                qd,q,loss=orig(*args,**kw);torch.cuda.synchronize();t=time.perf_counter();back=pw(path,vtq.PACK.pop('matrix'),fmt,'natural');assert torch.equal(q,back);torch.cuda.synchronize();overhead+=time.perf_counter()-t;return qd,back,loss
            torch.cuda.synchronize();base=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats();t=time.perf_counter()
            with patch.object(vtq,'quantize_matrix_vtq',encode):vtq.quantize_llama_model_vtq(m,cal,a.spec,fisher_out=fisher,batch_size=4,gptaq_alpha=None,d_sb_gecd_sweeps=0,polish_steps=0)
            torch.cuda.synchronize();t=time.perf_counter()-t;peak=torch.cuda.max_memory_allocated()-base;vtq.PACK=None
        assert len(pw.rows)==7*len(m.model.layers);r=dict(model=a.model,spec=a.spec,model_encode_without_writer_s=t-overhead,peak_extra_incl_roundtrip=peak,writer=pw.report(),cal_shape=list(cal.shape))
        for ds,rows in seqs.items():r[ds]=trace_metrics(m,scores[ds],features(rows),[t for _,_,t in rows],dev,logits_fn=m.lm_head,chunk=256)
        a.output.write_text(json.dumps(r)+'\n');print(a.spec,len(pw.rows),r['writer']['writer_s'],r['model_encode_without_writer_s'],flush=True)


if __name__=='__main__':main()
