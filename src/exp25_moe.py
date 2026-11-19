"""12,480 component-file gate on the real 48-layer 128-expert MoE;
use the repository's layer-streaming routed calibration and both dialogue traces.
"""
import argparse,json,sys,tempfile,time
from pathlib import Path
from unittest.mock import patch
import torch
from src import vtq,streaming_quant as sq
from src.exp25_writer_ab import PairedWriter
from src.vtq_format import load_vtq_formats


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--model',required=True);ap.add_argument('--data',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--spec',default='VTQ_300');a=ap.parse_args();torch.backends.cuda.matmul.allow_tf32=False;vtq.load_and_register_vtq();fmt=load_vtq_formats(menu=[a.spec])[a.spec];pw=PairedWriter('/content/libexp25pack.so');vtq.PACK={};overhead=0.;counts={}
    with tempfile.TemporaryDirectory() as td:
        path=Path(td)/'matrix.vtq';res=Path(td)/'model.json'
        def encode(w,h,spec,seed,damp,full):
            nonlocal overhead
            assert spec==a.spec;key=(sq._CUR_LAYER[0],full);assert key not in counts;counts[key]=1;_,q,_=vtq.quantize_matrix_vtq(w,h,spec,seed=seed+13000,damp=damp,row_weights=sq._row_w(full,w.device),u_sb_gecd_sweeps=0);torch.cuda.synchronize();t=time.perf_counter();back=pw(path,vtq.PACK.pop('matrix'),fmt,'natural');assert torch.equal(q,back);pw.rows[-1]['name']=f'{key[0]}/{full}';torch.cuda.synchronize();overhead+=time.perf_counter()-t;return back.to(w.dtype)
        with patch.object(sq,'_quantize_linear_weight',encode),patch.object(sys,'argv',['streaming','--model_dir',a.model,'--data_dir',str(a.data),'--methods','v2:'+a.spec,'--fisher',str(a.data/'fisher_30b.pt'),'--cal_rows','64','--out',str(res)]):sq.main()
        vtq.PACK=None;r=json.loads(res.read_text());assert r['n_layers']==48 and len(counts)==48*(4+128*2);r['writer']=pw.report();r['common_quant_s']=r['methods']['v2:'+a.spec]['quant_sec']-overhead;r['cal_rows']=64;r['routed_expert_matrices']=sum(n.startswith('mlp.experts.') for _,n in counts);a.output.write_text(json.dumps(r)+'\n');print(r['writer']['writer_s'],r['routed_expert_matrices'],flush=True)


if __name__=='__main__':main()
