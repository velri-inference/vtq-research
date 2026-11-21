"""Q4V1 vs production VTQ_300: equal tile+scale payload, measured headers separate.

python -m src.bench_q4v --model /path/to/gpt2-small --output results/q4v_g4_gpt2.json
"""

import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import time

import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from src import q4v, vtq
from src.vtq_format import VtqFile, VtqWriter, format_json_vtq, load_vtq_formats


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--text", type=Path, default=Path("data/wikitext2_test.raw"))
    ap.add_argument("--book", type=Path, default=Path("results/q4v_book.bin"))
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    model = (
        GPT2LMHeadModel.from_pretrained(args.model, dtype=torch.bfloat16).cuda().eval()
    )
    ids = torch.tensor(
        GPT2TokenizerFast.from_pretrained(args.model).encode(
            args.text.read_text(), verbose=False
        )
    )
    cal, val = [
        torch.stack(
            [t[i : i + 128] for i in torch.linspace(0, len(t) - 128, 16).long()]
        ).cuda()
        for t in ids.chunk(2)
    ]
    mods = [
        m
        for layer in model.transformer.h
        for m in (
            layer.attn.c_attn,
            layer.attn.c_proj,
            layer.mlp.c_fc,
            layer.mlp.c_proj,
        )
    ]
    orig = [m.weight.detach().T.contiguous().cpu() for m in mods]
    hess = {}

    def capture(i, x):
        x = x.reshape(-1, x.shape[-1]).float()
        hess[i] = (x.T @ x / len(x)).cpu()

    hooks = [
        m.register_forward_pre_hook(lambda m, a, i=i: capture(i, a[0]))
        for i, m in enumerate(mods)
    ]
    model.transformer(cal, use_cache=False)
    for hook in hooks:
        hook.remove()

    def logp(x):
        return model(x, use_cache=False).logits.float().log_softmax(-1)

    refs = [logp(x).cpu() for x in val.split(2)]
    book = np.frombuffer(args.book.read_bytes(), np.uint8).reshape(256, 4)
    enc = q4v.Encoder(book)
    vtq.load_and_register_vtq()
    fmt = load_vtq_formats(menu=["VTQ_300"])["VTQ_300"]
    params = sum(w.numel() for w in orig)
    payload = sum((w.numel() // 256) * 101 + 2 * sum(w.shape) for w in orig)
    report = dict(
        gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,
        model="gpt2-small",
        spec="VTQ_300",
        book_sha256=hashlib.sha256(book).hexdigest(),
        params=params,
        matrices=len(mods),
        contexts=16,
        tokens_per_context=128,
        payload_bytes=payload,
        payload_bpw=payload * 8 / params,
        runs=[],
    )
    # rep=0 warms the workspaces/JIT and actually serializes/reloads every matrix.
    # Two warm runs alternate ordering. Encoder timings exclude host I/O and decode.
    with TemporaryDirectory() as td:
        tmp = Path(td) / "matrix.vtq"
        for rep in range(3):
            methods = ("VTQ", "Q4V", "Q4V-H")
            for method in methods[rep:] + methods[:rep]:
                vtq.PACK = {} if method == "VTQ" else None
                sec, dec_sec, peak, files, norm, err = 0.0, 0.0, 0, 0, 0.0, 0.0
                for i, m in enumerate(mods):
                    w = orig[i].float().cuda()
                    h = (
                        hess[i].diagonal().contiguous() if method == "Q4V" else hess[i]
                    ).cuda()
                    torch.cuda.synchronize()
                    base = torch.cuda.memory_allocated()
                    torch.cuda.reset_peak_memory_stats()
                    t = time.perf_counter()
                    if method == "VTQ":
                        _, recon, _ = vtq.quantize_matrix_vtq(
                            w, h, "VTQ_300", seed=1000 + i, u_sb_gecd_sweeps=0
                        )
                        packed = vtq.PACK.pop("matrix")
                    elif method == "Q4V":
                        packed = enc.encode(w, h, seed=1000 + i)
                    else:
                        packed = enc.encode_hessian(w, h, seed=1000 + i)
                    torch.cuda.synchronize()
                    sec += time.perf_counter() - t
                    peak = max(peak, torch.cuda.max_memory_allocated() - base)
                    if rep == 0:
                        if method == "VTQ":
                            wr = VtqWriter(tmp, [fmt], reserve=4096)
                            wr.add_matrix("w", packed, verify=False)
                            wr.finalize(
                                {"quantization": {"formats": [format_json_vtq(fmt)]}}
                            )
                            files += tmp.stat().st_size
                            t = time.perf_counter()
                            reader = VtqFile(tmp)
                            q = reader.decode_matrix("w", device="cuda")
                            reader.mm.close()
                            reader.f.close()
                        else:
                            blob = packed.serialize(book)
                            files += len(blob)
                            t = time.perf_counter()
                            q = q4v.decode(blob, "cuda")
                        torch.cuda.synchronize()
                        dec_sec += time.perf_counter() - t
                        if method == "VTQ":
                            assert torch.equal(q, recon)
                        m.weight.copy_(q.T)
                        err += float((w - q).square().sum())
                        norm += float(w.square().sum())
                        del q
                    if method == "VTQ":
                        del recon
                    del packed, w, h
                run = dict(method=method, rep=rep, encode_s=sec, peak_extra_bytes=peak)
                if rep == 0:
                    kl = []
                    for x, r in zip(val.split(2), refs):
                        r = r.cuda()
                        kl.extend((r.exp() * (r - logp(x))).sum(-1).mean(-1).tolist())
                    run.update(
                        kld_by_context=kl,
                        kld=float(np.mean(kl)),
                        rel_mse=err / norm,
                        file_bytes=files,
                        file_bpw=8 * files / params,
                        reference_decode_s=dec_sec,
                    )
                report["runs"].append(run)
                args.output.write_text(json.dumps(report, indent=2) + "\n")
                print(rep, method, sec, peak, run.get("kld"), flush=True)
    vtq.PACK = None


if __name__ == "__main__":
    main()
