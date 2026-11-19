"""Model-level TSB1 ablation. Read actual TSB1/VTQ bytes for every evaluation."""
import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

from src import tsb1_format, vtq
from src.exp09_group_order import run_model_ab, vtq_roundtrip
from src.exp10_gain_shapes import banked_library, expand_tables


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model',required=True); ap.add_argument('--mode',choices=['discover','confirm'],required=True)
    ap.add_argument('--text',type=Path,default=Path('data/wikitext2_test.raw')); ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--trained',type=Path,default=Path('results/exp10_gain_shapes.json')); ap.add_argument('--source',type=Path)
    ap.add_argument('--seeds',nargs='+',type=int,default=[1000,2000]); ap.add_argument('--winner')
    args=ap.parse_args(); args.specs=['VTQ_300']; raw=json.loads(args.trained.read_text()); base=raw['baseline']
    runs=raw['experiments']; modes=['natural','untied','mixed-train','scale-untied','scale-mixed-train']
    if args.mode=='confirm':
        if args.winner not in modes: ap.error('freeze a valid winner before confirmation')
        modes=list(dict.fromkeys(['natural',args.winner]))
    def record(method):
        if method.startswith('scale-'):
            return dict(banks=base['banks'],scale=runs[method[6:]]['scale'])
        return runs[method]
    @contextmanager
    def ctx(method):
        if method=='natural': yield; return
        rec=record(method); lut=expand_tables(rec['banks']).flatten()
        get=vtq._get_spec_gpu_vtq
        def tables(name,dev):
            assert name=='VTQ_300'; cfg,_,g=get(name,dev)
            return dict(cfg,scale=rec['scale']),lut,g
        with banked_library(args.source),patch.object(vtq,'_get_spec_gpu_vtq',tables): yield
    def reload(path,packed,fmt,method):
        if method=='natural': return vtq_roundtrip(path,packed,fmt,method)
        # Deliberately decode without referencing 'rec', VTQ_CONFIG or cached CUDA tables.
        blob=tsb1_format.encode_bytes(packed,record(method)['banks'],fmt)
        return tsb1_format.decode_bytes(blob,device='cuda')
    run_model_ab(args,modes,ctx,reload,dict(format='TSB1',table_sha256=hashlib.sha256(args.trained.read_bytes()).hexdigest()))


if __name__=='__main__': main()
