"""Pair dialogues across methods after averaging KLD sums across encoder seeds."""
import argparse
import json
from pathlib import Path

import numpy as np
from src.eval_kld import paired_bootstrap_diff


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--files',nargs='+',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--offsets',nargs=3,type=int,default=[2000,3000,4000]);a=ap.parse_args()
    rows=[json.loads(p.read_text()) for p in a.files];assert sorted(r['offset'] for r in rows)==sorted(a.offsets) and len(set(a.offsets))==3
    base=rows[0];assert all(all(r[k]==base[k] for k in ('model','types','spec','alpha_down','other_gptaq','cal_shape')) for r in rows)
    modes=sorted(set.intersection(*(set(r['runs']) for r in rows)));assert 'natural' in modes and len(modes)>1
    out={'inputs':[p.name for p in a.files],'model':base['model'],'spec':base['spec'],'offsets':[r['offset'] for r in rows],'paired':{}}
    for ds in ('nemotron','multilingual'):
        counts=np.array(base['runs']['natural'][ds]['per_seq_tok_counts']);assert all(np.array_equal(r['runs'][m][ds]['per_seq_tok_counts'],counts) for r in rows for m in modes)
        def avg(m):
            t=np.mean([r['runs'][m][ds]['per_seq_kld_sums'] for r in rows],0);return dict(mean_kld=float(t.sum()/counts.sum()),per_seq_kld_sums=t,per_seq_tok_counts=counts)
        ctrl=avg('natural');out['paired'][ds]={}
        for m in modes:
            if m=='natural':continue
            v=avg(m);p=paired_bootstrap_diff(v,ctrl,n_boot=4000);p.update(rel_pct=100*p['mean_diff']/ctrl['mean_kld'],baseline=ctrl['mean_kld'],candidate=v['mean_kld']);out['paired'][ds][m]=p
    a.output.write_text(json.dumps(out,indent=2)+'\n');print(out['paired'],flush=True)


if __name__=='__main__':main()
