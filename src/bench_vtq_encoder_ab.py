"""Paired VTQ encoder ablation on the 48 Conv1D matrices of a local GPT-2 small.

git show 98d8ef1:src/vtq.py > src/vtq_baseline.py
python -m src.bench_vtq_encoder_ab --model /path/to/gpt2 --spec VTQ_300
Uses the first half of WikiText-2 test for calibration and only the second half
for evaluation. Embeddings, biases, norms and tied output head remain BF16.
"""
import argparse
from functools import partial
import hashlib
import json
from pathlib import Path
import statistics
import time

import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from src import vtq, vtq_baseline


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', type=Path, required=True)
    ap.add_argument('--text', type=Path, default=Path('data/wikitext2_test.raw'))
    ap.add_argument('--spec', default='VTQ_300')
    ap.add_argument('--repeats', type=int, default=4)
    ap.add_argument('--pack', action='store_true', help='measure encoder with packed codes requested')
    ap.add_argument('--output', type=Path)
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    model = GPT2LMHeadModel.from_pretrained(args.model, dtype=torch.bfloat16).cuda().eval()
    tok = GPT2TokenizerFast.from_pretrained(args.model)
    tokens = torch.tensor(tok.encode(args.text.read_text(), verbose=False))
    # 16 disjoint 128-token contexts, spread across each half.
    cal, val = [torch.stack([t[i:i + 128] for i in torch.linspace(0, len(t) - 128, 16).long()]).cuda()
                for t in tokens.chunk(2)]
    mods = [mod for layer in model.transformer.h for mod in
            (layer.attn.c_attn, layer.attn.c_proj, layer.mlp.c_fc, layer.mlp.c_proj)]
    orig = [m.weight.detach().T.contiguous().cpu() for m in mods]
    hess = {}

    def capture(i, mod, inputs):
        x = inputs[0].reshape(-1, inputs[0].shape[-1]).float()
        hess[i] = ((x.T @ x) / len(x)).cpu()

    hooks = [m.register_forward_pre_hook(partial(capture, i)) for i, m in enumerate(mods)]
    model.transformer(cal, use_cache=False)
    for hook in hooks:
        hook.remove()

    def logprobs(x):
        return model(x, use_cache=False).logits.float().log_softmax(-1)

    ref = [logprobs(x).cpu() for x in val.split(2)]
    vtq.load_and_register_vtq()
    vtq_baseline.load_and_register_vtq()
    report = dict(model=model.config.model_type, gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                  spec=args.spec, pack=args.pack, warmup_rep=0, contexts=len(val), tokens_per_context=val.shape[1], matrices=len(mods), runs=[])
    for rep in range(args.repeats):
        for enc in ((vtq_baseline, vtq) if rep % 2 == 0 else (vtq, vtq_baseline)):
            seconds, peak = 0., 0
            enc.PACK = {} if args.pack else None
            digest = hashlib.sha256()
            for i, m in enumerate(mods):
                w, h = orig[i].cuda().float(), hess[i].cuda()
                torch.cuda.synchronize()
                base = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                start = time.perf_counter()
                _, q, _ = enc.quantize_matrix_vtq(w, h, args.spec, seed=1000 + i, u_sb_gecd_sweeps=0)
                torch.cuda.synchronize()
                seconds += time.perf_counter() - start
                peak = max(peak, torch.cuda.max_memory_allocated() - base)
                m.weight.copy_(q.T)
                digest.update(q.contiguous().cpu().numpy())
                if args.pack:
                    packed = enc.PACK.pop('matrix')
                    for key in ('d_row', 'suh', 'u_sb', 'g_sub', 'c_all', 'u_all'):
                        digest.update(packed[key].contiguous().cpu().numpy())
                    del packed
                del q, w, h
            enc.PACK = None
            kl = []
            for x, r in zip(val.split(2), ref):
                r = r.cuda()
                kl.extend((r.exp() * (r - logprobs(x))).sum(-1).mean(-1).tolist())
            label = enc.__name__.split('.')[-1]
            run = dict(rep=rep, encoder=label, encoder_s=seconds, peak_extra_bytes=peak,
                       weights_and_codes_sha256=digest.hexdigest(), kld=statistics.mean(kl), kld_by_context=kl)
            report['runs'].append(run)
            print(f'{rep=} {label=} KLD={run["kld"]:.6f} {seconds=:.3f} {peak=}', flush=True)
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
