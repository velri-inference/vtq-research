"""Encoder-only ablation: schedule physical groups in the *first* LDLQ pass.

Experiment-local wrappers; neither the VTQ file format nor its decoder is patched.
Run discovery then transfer with --mode confirm --winner <frozen mode>.
"""
import argparse
from contextlib import ExitStack, contextmanager
import json
from pathlib import Path
import tempfile
import time
from unittest.mock import patch

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src import vtq
from src.vtq_format import VtqFile, VtqWriter, format_json_vtq, load_vtq_formats


MODES = ('natural', 'reverse', 'act-desc', 'act-asc', 'tile-desc', 'inner-desc', 'energy-desc', 'leverage-desc')


@contextmanager
def order_first_pass(mode, *, lowmem=False):
    if mode == 'natural':
        yield
        return
    if mode not in MODES:
        raise ValueError(mode)
    orig_suh, orig_rows = vtq.compute_suh_vtq, vtq._quantize_rows_vtq_ldlq
    inverse = {}

    def prepare(*a, **kw):
        w, h, suh = orig_suh(*a, **kw)
        k = w.shape[-1]
        n = k // 256
        diag = h.diagonal().clamp_min(1e-12)
        score = diag
        if mode == 'energy-desc':
            score = score * w.norm(dim=0).square()
        elif mode == 'leverage-desc':
            # Cheap estimate of coupling to columns outside a group, using a
            # fixed strided sample. No randomized validation-dependent ordering.
            anchors = torch.arange(0, k, max(1, k//64), device=w.device)
            score = ((h[:, anchors].square() / diag[anchors]).mean(dim=1) / diag)
        score = score.view(n, 16, 16).mean(dim=-1)
        descending = mode != 'act-asc'
        tile_ids = torch.arange(n, device=w.device)
        groups = torch.arange(16, device=w.device).expand(n, -1)
        if mode == 'reverse':
            tile_ids, groups = tile_ids.flip(0), groups.flip(1)
        else:
            if mode != 'inner-desc':
                tile_ids = score.mean(dim=1).argsort(descending=descending, stable=True)
            if mode != 'tile-desc':
                groups = score[tile_ids].argsort(dim=1, descending=descending, stable=True)
        order16 = (tile_ids[:, None]*16 + groups).flatten()
        cols = (16*order16[:, None] + torch.arange(16, device=w.device)).flatten()
        inverse.update(tile=tile_ids.argsort(), group=order16.argsort(), col=cols.argsort(), rows_left=w.shape[0])
        if lowmem:
            from src.exp11_fast_schedule import gather_square
            return w.index_select(1, cols), gather_square(h, cols), suh
        return w[:, cols], h[cols[:, None], cols], suh

    def encode_rows(*a, **kw):
        out = orig_rows(*a, **kw)
        r, k = out['G_nat'].shape
        def gather(x, idx):
            return x.index_select(1, idx) if lowmem else x[:, idx]
        out['G_nat'] = gather(out['G_nat'], inverse['col'])
        out['u_sb'] = gather(out['u_sb'], inverse['tile'])
        for key, width in (('g_sub', 1), ('c_all', 2), ('u_all', 16)):
            if out[key] is not None:
                x = out[key]
                out[key] = gather(x.reshape(r, k//16, width), inverse['group']).reshape_as(x)
        inverse['rows_left'] -= r
        if inverse['rows_left'] == 0:
            inverse.clear()  # the final FP16 refit needs no permutation buffers
        return out

    with ExitStack() as stack:
        stack.enter_context(patch.object(vtq, 'compute_suh_vtq', prepare))
        stack.enter_context(patch.object(vtq, '_quantize_rows_vtq_ldlq', encode_rows))
        yield


def paired_ci(candidate, control):
    d = np.asarray(candidate) - np.asarray(control)
    rng = np.random.default_rng(934)
    boot = d[rng.integers(0, len(d), (4000, len(d)))].mean(axis=1)
    return dict(mean=float(d.mean()), ci95=np.quantile(boot, [.025,.975]).tolist(),
                rel_pct=float(100*d.mean()/np.mean(control)))


def matrix_modules(model):
    if model.config.model_type == 'gpt2':
        return ([m for l in model.transformer.h for m in
                 (l.attn.c_attn,l.attn.c_proj,l.mlp.c_fc,l.mlp.c_proj)], True)
    return ([m for l in model.model.layers for m in
             (l.self_attn.q_proj,l.self_attn.k_proj,l.self_attn.v_proj,l.self_attn.o_proj,
              l.mlp.gate_proj,l.mlp.up_proj,l.mlp.down_proj)], False)


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', required=True)
    ap.add_argument('--text', type=Path, default=Path('data/wikitext2_test.raw'))
    ap.add_argument('--mode', choices=('discover','confirm'), required=True)
    ap.add_argument('--winner', choices=MODES)
    ap.add_argument('--specs', nargs='+', default=['VTQ_300'])
    ap.add_argument('--seeds', nargs='+', type=int, default=[1000,2000])
    ap.add_argument('--output', type=Path, required=True)
    args=ap.parse_args()
    if args.mode=='confirm' and args.winner is None:
        ap.error('--winner must be fixed before opening confirmation data')
    modes=MODES if args.mode=='discover' else list(dict.fromkeys(['natural',args.winner]))
    run_model_ab(args, modes)


def vtq_roundtrip(path, packed, fmt, method):
    wr=VtqWriter(path,[fmt],reserve=4096)
    wr.add_matrix('w',packed,verify=False); wr.finalize({'quantization':{'formats':[format_json_vtq(fmt)]}})
    rd=VtqFile(path)
    try: return rd.decode_matrix('w',device='cuda')
    finally: rd.mm.close(); rd.f.close()


@torch.inference_mode()
def run_model_ab(args, modes, context=order_first_pass, roundtrip=vtq_roundtrip, provenance=None, setup=None):
    torch.backends.cuda.matmul.allow_tf32=False
    model=AutoModelForCausalLM.from_pretrained(args.model,dtype=torch.bfloat16).eval().cuda()
    tok=AutoTokenizer.from_pretrained(args.model)
    ids=torch.tensor(tok.encode(args.text.read_text(),add_special_tokens=False,verbose=False))
    def windows(t,n):
        return torch.stack([t[j:j+192] for j in torch.linspace(0,len(t)-192,n).long()]).cuda()
    a,b,c=ids.tensor_split(3)
    cal=windows(a,32); val=windows(b,32) if args.mode=='discover' else windows(c,64)
    mods,transpose=matrix_modules(model)
    weights=[(m.weight.T if transpose else m.weight).contiguous().cpu() for m in mods]
    assert all(w.shape[0]%16==0 and w.shape[1]%256==0 for w in weights)
    hess={}
    def hook(i,x):
        x=x.reshape(-1,x.shape[-1]).float(); hess[i]=((x.T@x)/len(x)).cpu()
    hooks=[m.register_forward_pre_hook(lambda m,a,i=i:hook(i,a[0])) for i,m in enumerate(mods)]
    (model.transformer if transpose else model.model)(input_ids=cal,use_cache=False)
    for hk in hooks: hk.remove()
    adapt = None
    if setup is not None:
        context, adapt = setup(model, weights, hess, transpose)
    def logp(x): return model(input_ids=x,use_cache=False).logits.float().log_softmax(-1)
    refs=[logp(x).cpu() for x in val.split(2)]
    vtq.load_and_register_vtq()
    report=dict(model=args.model,mode=args.mode,specs=args.specs,seeds=args.seeds,methods=list(modes),
                gpu=torch.cuda.get_device_name(),tokens=len(val)*val.shape[1],matrices=len(mods),runs=[])
    if provenance is not None: report['provenance']=provenance
    def save():
        p=args.output.with_suffix('.tmp'); p.write_text(json.dumps(report,indent=2)+'\n'); p.replace(args.output)
    if args.output.exists():
        old=json.loads(args.output.read_text()); assert all(old[k]==v for k,v in report.items() if k!='runs'); report=old
    with tempfile.TemporaryDirectory() as td:
        path=Path(td)/'matrix.vtq'
        for spec in args.specs:
            fmt=load_vtq_formats(menu=[spec])[spec]
            for seed in args.seeds:
                for method in modes:
                    if any((r['spec'],r['seed'],r['method'])==(spec,seed,method) for r in report['runs']): continue
                    sec,peak=0.,0
                    vtq.PACK={}
                    with context(method) as extra:
                        for i,m in enumerate(mods):
                            w,h=weights[i].cuda().float(),hess[i].cuda()
                            torch.cuda.synchronize(); base=torch.cuda.memory_allocated(); torch.cuda.reset_peak_memory_stats(); t=time.perf_counter()
                            if adapt is not None: w,h=adapt(method,i,w,h)
                            _,q,_=vtq.quantize_matrix_vtq(w,h,spec,seed=seed+i,u_sb_gecd_sweeps=0)
                            torch.cuda.synchronize(); sec+=time.perf_counter()-t; peak=max(peak,torch.cuda.max_memory_allocated()-base)
                            packed=vtq.PACK.pop('matrix'); decoded=roundtrip(path,packed,fmt,method)
                            assert torch.equal(decoded,q), (method,i,'roundtrip')
                            m.weight.copy_(decoded.T if transpose else decoded); del q,packed,decoded,w,h
                    kl=[]
                    for xx,rr in zip(val.split(2),refs):
                        rr=rr.cuda(); kl.extend((rr.exp()*(rr-logp(xx))).sum(-1).mean(-1).tolist())
                    report['runs'].append(dict(method=method,spec=spec,seed=seed,per_context=kl,kld=float(np.mean(kl)),
                                               quant_s=sec,peak_bytes=peak,**(extra or {}))); save()
                    print(method,spec,seed,'kld',np.mean(kl),'s',sec,flush=True)
    vtq.PACK=None
    # Average over encoder seeds within each context; seeds are not independent texts.
    for spec in args.specs:
        avg={m:np.mean([r['per_context'] for r in report['runs'] if r['method']==m and r['spec']==spec],axis=0) for m in modes}
        report.setdefault('paired',{})[spec]={m:paired_ci(v,avg['natural']) for m,v in avg.items()}
    if args.mode=='discover':
        report['winner']=min(modes,key=lambda m:np.mean([r['kld'] for r in report['runs'] if r['method']==m]))
    save()


if __name__=='__main__': main()
