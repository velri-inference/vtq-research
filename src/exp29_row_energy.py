"""Model KLD A/B of calibrating per-neuron output RMS after the frozen-code
least-squares refit. Only existing fp16 d_row changes. Negative tilt is a
directional control; end-to-end quality, not output variance itself, decides.
"""
import argparse
from pathlib import Path
from contextlib import contextmanager
from functools import partial
from unittest.mock import patch
from src import vtq
from src.exp09_group_order import run_model_ab

TILT={'natural':0.,'half':.5,'full':1.,'away':-.5}

@contextmanager
def ctx(mode):
    if mode=='natural':yield;return
    with patch.object(vtq,'quantize_matrix_vtq',partial(vtq.quantize_matrix_vtq,row_energy_tilt=TILT[mode])):yield


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--model',required=True);ap.add_argument('--text',type=Path,default=Path('data/wikitext2_test.raw'));ap.add_argument('--mode',choices=['discover','confirm'],required=True);ap.add_argument('--winner',choices=list(TILT));ap.add_argument('--specs',nargs='+',default=['VTQ_200','VTQ_300','VTQ_400']);ap.add_argument('--seeds',nargs='+',type=int,default=[21000,22000]);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();methods=list(TILT)
    if a.mode=='confirm':
        if not a.winner:ap.error('frozen winner required')
        methods=list(dict.fromkeys(['natural',a.winner]))
    run_model_ab(a,methods,ctx,provenance=dict(tilt=TILT,rms_ratio_clamp=[.8,1.25],format_unchanged=True))

if __name__=='__main__':main()
