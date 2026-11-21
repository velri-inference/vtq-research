"""Bit-identity gate for a single-use GPTAQ accumulator and teacher MLP reuse.
All component files, not only decoded values, must match across variants.
Hashing is in the excluded roundtrip interval of the native model harness.
"""
import argparse
from contextlib import ExitStack,contextmanager
from functools import partial
import hashlib
import json
from pathlib import Path
from unittest.mock import patch
import torch
from src import vtq
from src import exp11_llama_transfer as native
from src.vtq_stream_cov import ConsumableCov


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--model',required=True);ap.add_argument('--data',type=Path,default=Path('data'));ap.add_argument('--fisher',default='fisher_1b.pt');ap.add_argument('--output',type=Path,required=True);ap.add_argument('--cal-seqs',type=int,default=64);ap.add_argument('--offsets',type=int,nargs='+',default=[2000,3000,4000]);ap.add_argument('--spec',default='VTQ_300');ap.add_argument('--pair',action='store_true',help='compare integrated compact_gptaq path to reference');a=ap.parse_args();methods=['natural','both'] if a.pair else ['natural','reuse','consume','both']
    if a.output.exists():raise RuntimeError('fresh output required for byte identity test')
    digests={m:[] for m in methods};base_rt=native.vtq_roundtrip
    @contextmanager
    def experiment(mode):
        blocks=[]
        def checked(path,packed,fmt,method):
            back=base_rt(path,packed,fmt,method);blocks.append(hashlib.sha256(path.read_bytes()).hexdigest());return back
        with ExitStack() as st:
            st.enter_context(patch.object(native,'vtq_roundtrip',checked))
            if mode=='consume':st.enter_context(patch.object(vtq,'DualCovAccumulator',ConsumableCov))
            if mode in ('reuse','both'):st.enter_context(patch.object(vtq,'quantize_llama_model_vtq',partial(vtq.quantize_llama_model_vtq,compact_gptaq=mode=='both',reuse_reference_mlp=mode=='reuse')))
            yield
        assert blocks;digests[mode].append(blocks)
    native.run_transfer(a,methods,experiment,a.spec,discovery=True,provenance=dict(experiment='exact streaming 2x2',hash='sha256 of each matrix file incl header'))
    r=json.loads(a.output.read_text());base=digests['natural'];assert all(len(base)==len(digests[m])==len(a.offsets) for m in methods)
    r['exact']={m:all(v==w for v,w in zip(base,digests[m])) for m in methods};r['file_sha256']=digests
    tmp=a.output.with_suffix('.tmp');tmp.write_text(json.dumps(r)+'\n');tmp.replace(a.output)
    print('byte equality:',r['exact'],flush=True)


if __name__=='__main__':main()
