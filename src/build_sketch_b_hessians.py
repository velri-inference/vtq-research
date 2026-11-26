#!/usr/bin/env python3
"""
Architecture-General Streaming Sketch-B Output Hessian Builder (`src/build_sketch_b_hessians.py`).

Builds high-rank, multi-domain YAQA Sketch-B left Kronecker output Hessians:
    G_l = E_seq[ (grad_W) (grad_W)^T ] = E_seq[ (g^T x) (g^T x)^T ]
using `StreamingCausalEvaluator` (supports Dense Llama, Gemma-4 26B MoE, and Qwen3.6 35B
Hybrid Gated DeltaNet + MoE with low peak VRAM).

Targets saved per layer `l` into `<out_dir>/{l}_{short}_hout.pt` (YAQA lower-triangular `flatH`):
  - `self_attn.o_proj`          -> `{l}_o_hout.pt`
  - `linear_attn.out_proj`      -> `{l}_out_hout.pt`
  - `mlp.down_proj`             -> `{l}_down_hout.pt`
  - `mlp.shared_expert.down_proj` -> `{l}_shared_down_hout.pt`
  - `experts.*.down_proj` (MoE) -> `{l}_moe_down_hout.pt` (exact summed Sketch-B across all active experts!)

Uses exact matrix re-association `g^T (x x^T) g == (g^T x)(g^T x)^T` when `T < K` (and block-diagonal
per-sequence masking for batched MoE experts), avoiding cross-sequence Wishart contamination while
running ~5-20x faster than naive `[M, K]` outer products.
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import src.run_paper_benchmarks as rpb
from src.vtq_gptq2d import _short_yaqa_name, sym_to_flat

DENSE_SKETCH_TARGETS = (
    "self_attn.o_proj",
    "linear_attn.out_proj",
    "mlp.down_proj",
    "mlp.shared_expert.down_proj",
)


def _eff_rank(G: torch.Tensor) -> float:
    tr = float(torch.diagonal(G).sum())
    fn2 = float((G * G).sum())
    return (tr * tr) / max(fn2, 1e-30)


def _stream_hf_texts(repo: str, name: Optional[str], split: str, n_docs: int, text_keys=("text", "content", "code")) -> List[str]:
    from datasets import load_dataset

    kw = {"split": split, "streaming": True}
    ds = load_dataset(repo, name, **kw) if name is not None else load_dataset(repo, **kw)
    out: List[str] = []
    for row in ds:
        txt = None
        for k in text_keys:
            if k in row and isinstance(row[k], str) and len(row[k].strip()) >= 128:
                txt = row[k]
                break
        if txt is not None:
            out.append(txt)
            if len(out) >= n_docs:
                break
    return out


def _collect_token_chunks(
    tok,
    texts: List[str],
    n_seqs: int,
    seq_len: int,
) -> List[List[int]]:
    buf: List[int] = []
    chunks: List[List[int]] = []
    eos = tok.eos_token_id
    for t in texts:
        ids = tok.encode(t, add_special_tokens=False)
        if not ids:
            continue
        buf.extend(ids)
        if eos is not None:
            buf.append(int(eos))
        while len(buf) >= seq_len:
            chunks.append(buf[:seq_len])
            buf = buf[seq_len:]
            if len(chunks) >= n_seqs:
                return chunks
    return chunks


def build_multidomain_sketch_cal(
    model_dir: Path,
    cache_path: Path,
    *,
    n_seqs: int = 512,
    seq_len: int = 512,
    seed: int = 20261125,
) -> Tuple[torch.Tensor, dict]:
    """
    Streams an open, ungated multi-domain calibration set (50% English web, ~31% multilingual web
    across 8 scripts/languages, ~19% code) strictly disjoint from all paper eval traces.
    """
    meta_path = cache_path.with_suffix(".json")
    if cache_path.exists() and meta_path.exists():
        cal = load_file(str(cache_path))["input_ids"]
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if tuple(cal.shape) == (n_seqs, seq_len):
            print(f"[sketch_b] Loaded cached multi-domain calibration {tuple(cal.shape)} from {cache_path}", flush=True)
            return cal, meta

    print(f"[sketch_b] Streaming open multi-domain calibration ({n_seqs}x{seq_len}) for {model_dir.name}...", flush=True)
    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)

    n_en = n_seqs // 2                         # 256
    n_mul = (n_seqs * 5) // 16                 # 160 (20 per language x 8 languages)
    n_code = n_seqs - n_en - n_mul             # 96

    provenance: Dict[str, int] = {}
    all_chunks: List[List[int]] = []

    # 1) English web (FineWeb-Edu -> FineWeb -> C4)
    en_chunks: List[List[int]] = []
    for repo, cfg in [
        ("HuggingFaceFW/fineweb-edu", "sample-10BT"),
        ("HuggingFaceFW/fineweb", "sample-10BT"),
        ("allenai/c4", "en"),
    ]:
        try:
            texts = _stream_hf_texts(repo, cfg, "train", n_docs=max(600, n_en * 3))
            en_chunks = _collect_token_chunks(tok, texts, n_en, seq_len)
            if len(en_chunks) >= n_en:
                provenance[f"{repo}:{cfg}"] = len(en_chunks)
                break
        except Exception as e:
            print(f"  [cal fallback] {repo}:{cfg} failed: {e}", flush=True)
    all_chunks.extend(en_chunks)

    # 2) Multilingual web across 8 languages (FineWeb-2 -> Wikipedia)
    langs = [
        ("rus_Cyrl", "ru"),
        ("cmn_Hani", "zh"),
        ("deu_Latn", "de"),
        ("fra_Latn", "fr"),
        ("spa_Latn", "es"),
        ("jpn_Jpan", "ja"),
        ("kor_Hang", "ko"),
        ("arb_Arab", "ar"),
    ]
    per_lang = max(1, n_mul // len(langs))
    mul_chunks: List[List[int]] = []
    for fw2_cfg, wiki_code in langs:
        got: List[List[int]] = []
        for repo, cfg in [
            ("HuggingFaceFW/fineweb-2", fw2_cfg),
            ("wikimedia/wikipedia", f"20231101.{wiki_code}"),
        ]:
            try:
                texts = _stream_hf_texts(repo, cfg, "train", n_docs=max(120, per_lang * 4))
                got = _collect_token_chunks(tok, texts, per_lang, seq_len)
                if len(got) >= per_lang:
                    provenance[f"{repo}:{cfg}"] = len(got)
                    break
            except Exception as e:
                print(f"  [cal fallback] {repo}:{cfg} failed: {e}", flush=True)
        mul_chunks.extend(got)
    all_chunks.extend(mul_chunks)

    # 3) Code (ungated smollm-corpus/python-edu -> fineweb-edu fallback)
    code_chunks: List[List[int]] = []
    for repo, cfg in [
        ("HuggingFaceTB/smollm-corpus", "python-edu"),
        ("HuggingFaceFW/fineweb-edu", "sample-10BT"),
    ]:
        try:
            texts = _stream_hf_texts(repo, cfg, "train", n_docs=max(500, n_code * 5))
            got = _collect_token_chunks(tok, texts, n_code, seq_len)
            if got:
                provenance[f"{repo}:{cfg}_code"] = len(got)
                code_chunks.extend(got[:n_code])
                break
        except Exception as e:
            print(f"  [cal fallback] {repo}:{cfg} failed: {e}", flush=True)
    all_chunks.extend(code_chunks)

    # Top up if any language came slightly short
    if len(all_chunks) < n_seqs:
        need = n_seqs - len(all_chunks)
        texts = _stream_hf_texts("HuggingFaceFW/fineweb-edu", "sample-10BT", "train", n_docs=max(600, need * 4))
        extra = _collect_token_chunks(tok, texts, need, seq_len)
        provenance["fineweb_edu_topup"] = len(extra)
        all_chunks.extend(extra)

    all_chunks = all_chunks[:n_seqs]
    assert len(all_chunks) == n_seqs, f"Expected {n_seqs} chunks, got {len(all_chunks)}"
    rng = random.Random(seed)
    rng.shuffle(all_chunks)

    cal = torch.tensor(all_chunks, dtype=torch.long)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    save_file({"input_ids": cal}, str(cache_path))
    meta = {
        "n_seqs": n_seqs,
        "seq_len": seq_len,
        "total_tokens": n_seqs * seq_len,
        "seed": seed,
        "provenance": provenance,
        "build_sec": round(time.perf_counter() - t0, 2),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[sketch_b] Cached {tuple(cal.shape)} multi-domain tokens in {meta['build_sec']:.1f}s: {provenance}", flush=True)
    return cal, meta


def _accum_dense_sketch_b(G_acc: torch.Tensor, x_b: torch.Tensor, g_b: torch.Tensor, trace_norm: bool = True):
    """Zero-sync exact per-sequence Sketch-B accumulation `sum_b (g_b^T x_b)(g_b^T x_b)^T / tr(.)`."""
    B, T, K = x_b.shape
    for b in range(B):
        xb = x_b[b]
        gb = g_b[b]
        if T < K:
            Gb = (gb.T @ ((xb @ xb.T) @ gb)).double()
        else:
            gw = gb.T @ xb
            Gb = (gw @ gw.T).double()
        if trace_norm:
            tr = Gb.diagonal().sum()
            G_acc.add_(Gb * torch.where(tr > 1e-20, 1.0 / tr.clamp(min=1e-20), torch.zeros_like(tr)))
        else:
            G_acc.add_(Gb)


def _accum_moe_expert_sketch_b_per_seq(
    G_seq_moe: torch.Tensor,  # [B_max, M, M] float64 per-sequence accumulator
    mid_e: torch.Tensor,
    g_e: torch.Tensor,
    seq_ids_e: torch.Tensor,
    B: int,
):
    """Zero-sync exact per-sequence Sketch-B accumulation for one routed MoE expert into `G_seq_moe[b]`."""
    n_tok = mid_e.shape[0]
    if n_tok == 0:
        return
    same_seq = seq_ids_e.unsqueeze(1) == seq_ids_e.unsqueeze(0)
    K_e = (mid_e @ mid_e.T).masked_fill_(~same_seq, 0.0)
    tmp_e = K_e @ g_e  # [n_tok, M], block-isolated per sequence
    for b in range(B):
        mb = (seq_ids_e == b).to(g_e.dtype).unsqueeze(1)
        G_seq_moe[b].add_(((g_e * mb).T @ tmp_e).double())


def _move_shared_kv_to_dev(kwargs: dict, dev: torch.device) -> dict:
    st = kwargs.get("shared_kv_states")
    if not isinstance(st, dict):
        return kwargs
    new_st = {}
    for k, v in st.items():
        if isinstance(v, tuple):
            new_st[k] = tuple(t.to(dev) if torch.is_tensor(t) else t for t in v)
        elif torch.is_tensor(v):
            new_st[k] = v.to(dev)
        else:
            new_st[k] = v
    return dict(kwargs, shared_kv_states=new_st)


def build_streaming_sketch_b_hessians(
    ev: rpb.StreamingCausalEvaluator,
    cal_ids: torch.Tensor,
    out_dir: Path,
    *,
    batch_size: int = 4,
    chunk_seqs: int = 128,
    include_moe: bool = True,
    seed_base: int = 5000,
) -> dict:
    """
    Computes Sketch-B output Hessians across all layers of `ev` (Dense, Gemma-4 MoE, Qwen3.6 Hybrid MoE)
    using exact unbiased `n_mc=1` inverse-CDF Fisher sampling per sequence.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "sketch_b_meta.json"
    n_seqs, seq_len = int(cal_ids.shape[0]), int(cal_ids.shape[1])

    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("n_seqs") == n_seqs and meta.get("seq_len") == seq_len and meta.get("include_moe") == include_moe:
            print(f"[sketch_b] Reusing existing Sketch-B Hessians in {out_dir} ({meta.get('wall_sec')}s)", flush=True)
            return meta

    dev = ev.device
    ev.restore_clean_embed_and_head()
    t0 = time.perf_counter()

    # Prepend BOS token to every sequence if the architecture uses BOS (critical on Gemma-4 / Llama
    # to avoid massive pos=0 gradient spikes on mid-document chunks).
    bos_id = getattr(ev.config, "bos_token_id", None)
    if isinstance(bos_id, (list, tuple)):
        bos_id = bos_id[0] if bos_id else None
    if bos_id is not None:
        bos_col = torch.full((n_seqs, 1), int(bos_id), dtype=cal_ids.dtype)
        cal_ids = torch.cat([bos_col, cal_ids[:, :-1]], dim=1).contiguous()
        print(f"[sketch_b] Prepended BOS token id={bos_id} to all {n_seqs} sequences", flush=True)

    # Discover targets per layer using meta modules (zero weight load!)
    layer_dense_targets: List[List[Tuple[str, str, int]]] = []
    layer_has_moe: List[Tuple[bool, int]] = []
    accum_cpu: Dict[Tuple[int, str], torch.Tensor] = {}

    for l_idx in range(ev.n_layers):
        lay = ev.layers[l_idx]
        dt: List[Tuple[str, str, int]] = []
        for name, mod in lay.named_modules():
            if isinstance(mod, nn.Linear) and any(name == t or name.endswith("." + t) for t in DENSE_SKETCH_TARGETS):
                sn = _short_yaqa_name(name) or name
                dt.append((name, sn, int(mod.out_features)))
                accum_cpu[(l_idx, sn)] = torch.zeros((int(mod.out_features), int(mod.out_features)), dtype=torch.float64, device="cpu")
        layer_dense_targets.append(dt)
        ex_obj, _ = rpb.get_moe_experts_and_prefix(lay)
        if include_moe and ex_obj is not None:
            m_dim = int(ex_obj.down_proj.shape[1])
            layer_has_moe.append((True, m_dim))
            accum_cpu[(l_idx, "moe_down")] = torch.zeros((m_dim, m_dim), dtype=torch.float64, device="cpu")
        else:
            layer_has_moe.append((False, 0))

    print(
        f"\n=== Building Streaming Sketch-B Output Hessians: {len(accum_cpu)} matrices | "
        f"{n_seqs}x{seq_len} (B={batch_size}, chunk={chunk_seqs}, include_moe={include_moe}) ===",
        flush=True,
    )

    n_chunks = (n_seqs + chunk_seqs - 1) // chunk_seqs
    for c_idx in range(n_chunks):
        s0 = c_idx * chunk_seqs
        s1 = min(n_seqs, s0 + chunk_seqs)
        t_ch = time.perf_counter()

        # Group sequences in this chunk into batches of `batch_size`
        batches = [cal_ids[i : min(i + batch_size, s1)] for i in range(s0, s1, batch_size)]
        n_b = len(batches)
        hs_cpu: List[torch.Tensor] = []
        ctxs_per_b: List[List[tuple]] = []
        for b_ids in batches:
            h0, ctxs = ev.capture_input_and_ctx(b_ids)
            hs_cpu.append(h0)
            ctxs_per_b.append(ctxs)

        # 1) Forward pass across layers, storing per-layer inputs on CPU
        layer_inputs: List[List[torch.Tensor]] = []
        for l_idx in range(ev.n_layers):
            layer_inputs.append(hs_cpu)
            lay = ev.load_layer_bf16(l_idx)
            for p in lay.parameters():
                p.requires_grad_(False)
            with torch.no_grad():
                hs_cpu = [ev.call_layer(lay, hs_cpu[b], ctxs_per_b[b][l_idx]) for b in range(n_b)]
            ev.release_layer(l_idx)

        # 2) Final lm_head unbiased inverse-CDF Fisher loss (n_mc=1 per sequence, per-sequence logits for low VRAM)
        gs_cpu: List[torch.Tensor] = []
        seq_cursor = s0
        for b in range(n_b):
            h_batch = hs_cpu[b]  # [B_cur, T, H]
            B_cur = int(h_batch.shape[0])
            g_parts: List[torch.Tensor] = []
            for j in range(B_cur):
                h_one = h_batch[j : j + 1].to(dev).detach().requires_grad_(True)
                with torch.enable_grad():
                    logits = ev.logits(h_one).reshape(-1, ev.config.vocab_size)
                    with torch.no_grad():
                        probs = F.softmax(logits.float(), dim=-1)
                        u = torch.rand((probs.shape[0], 1), generator=torch.Generator().manual_seed(seed_base + seq_cursor)).to(dev)
                        cdf = probs.cumsum(dim=-1)
                        y = torch.searchsorted(cdf, (u * cdf[:, -1:]).contiguous()).clamp(max=probs.shape[-1] - 1).squeeze(1)
                        del probs, cdf, u
                    loss = F.cross_entropy(logits.float(), y, reduction="sum")
                    g_one = torch.autograd.grad(loss, h_one)[0].detach().cpu()
                    g_parts.append(g_one)
                del logits, loss, h_one, y
                seq_cursor += 1
            gs_cpu.append(torch.cat(g_parts, dim=0))
        del hs_cpu
        torch.cuda.empty_cache()

        # 3) Reverse-order layer-by-layer backprop + exact per-sequence Sketch-B accumulation
        for l_idx in reversed(range(ev.n_layers)):
            lay = ev.load_layer_bf16(l_idx)
            for p in lay.parameters():
                p.requires_grad_(False)

            dt = layer_dense_targets[l_idx]
            has_moe, moe_m = layer_has_moe[l_idx]
            G_gpu: Dict[str, torch.Tensor] = {
                sn: torch.zeros((m_dim, m_dim), dtype=torch.float64, device=dev) for _, sn, m_dim in dt
            }
            if has_moe:
                G_gpu["moe_down"] = torch.zeros((moe_m, moe_m), dtype=torch.float64, device=dev)
                G_seq_moe = torch.zeros((batch_size, moe_m, moe_m), dtype=torch.float64, device=dev)
            else:
                G_seq_moe = None

            cur_B = [batch_size]
            saved_x: Dict[str, Optional[torch.Tensor]] = {sn: None for _, sn, _ in dt}
            hooks = []
            for sub_nm, sn, _ in dt:
                def _make_fwd(sn_key: str):
                    def _fwd(m, inp, out):
                        if out.requires_grad:
                            B_c = cur_B[0]
                            saved_x[sn_key] = inp[0].detach().float().reshape(B_c, seq_len, -1)

                            def _bwd(gr, sk=sn_key, bc=B_c):
                                x_b = saved_x[sk]
                                saved_x[sk] = None
                                g_b = gr.detach().float().reshape(bc, seq_len, -1)
                                _accum_dense_sketch_b(G_gpu[sk], x_b, g_b, trace_norm=True)
                                del x_b, g_b

                            out.register_hook(_bwd)

                    return _fwd

                hooks.append(lay.get_submodule(sub_nm).register_forward_hook(_make_fwd(sn)))

            ex_obj, _ = rpb.get_moe_experts_and_prefix(lay)
            patched_moe = False
            if has_moe and ex_obj is not None:
                patched_moe = True

                def _ex_fwd(hidden_states, top_k_index, top_k_weights, _ex=ex_obj):
                    out = torch.zeros_like(hidden_states)
                    B_c = cur_B[0]
                    with torch.no_grad():
                        mask = F.one_hot(top_k_index, num_classes=_ex.num_experts).permute(2, 1, 0)
                        hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()
                    for e in hit[:, 0].tolist():
                        pos_i, tok_i = torch.where(mask[e])
                        gu = F.linear(hidden_states[tok_i], _ex.gate_up_proj[e])
                        gate_t, up_t = gu.chunk(2, dim=-1)
                        mid_e = _ex.act_fn(gate_t) * up_t
                        dn = F.linear(mid_e, _ex.down_proj[e])
                        if dn.requires_grad:
                            mid_saved = mid_e.detach().float()
                            seq_ids_e = torch.div(tok_i, seq_len, rounding_mode="floor")

                            def _moe_bwd(gr, ms=mid_saved, si=seq_ids_e, bc=B_c):
                                _accum_moe_expert_sketch_b_per_seq(G_seq_moe, ms, gr.detach().float(), si, bc)

                            dn.register_hook(_moe_bwd)
                        out.index_add_(0, tok_i, (dn * top_k_weights[tok_i, pos_i, None]).to(out.dtype))
                    return out

                ex_obj.forward = _ex_fwd

            for b in range(n_b):
                x = layer_inputs[l_idx][b].to(dev).detach().requires_grad_(True)
                B_c = int(x.shape[0])
                cur_B[0] = B_c
                if G_seq_moe is not None:
                    G_seq_moe[:B_c].zero_()
                rest, kwargs = ctxs_per_b[b][l_idx]
                kwargs = _move_shared_kv_to_dev(kwargs, dev)
                g_out = gs_cpu[b].to(dev)
                with torch.enable_grad():
                    out = lay(x, *rest, **kwargs)
                    out = out[0] if isinstance(out, tuple) else out
                    gs_cpu[b] = torch.autograd.grad(out, x, grad_outputs=g_out)[0].detach().cpu()
                if G_seq_moe is not None:
                    trs = G_seq_moe[:B_c].diagonal(dim1=1, dim2=2).sum(dim=1)  # [B_c]
                    scales = torch.where(trs > 1e-20, 1.0 / trs.clamp(min=1e-20), torch.zeros_like(trs))
                    G_gpu["moe_down"].add_((G_seq_moe[:B_c] * scales.view(B_c, 1, 1)).sum(dim=0))
                del x, out, g_out

            if G_seq_moe is not None:
                del G_seq_moe

            for hk in hooks:
                hk.remove()
            if patched_moe and ex_obj is not None:
                del ex_obj.forward

            for sn, G_t in G_gpu.items():
                accum_cpu[(l_idx, sn)].add_(G_t.cpu())
            del G_gpu
            ev.release_layer(l_idx)
            layer_inputs[l_idx] = []
            torch.cuda.empty_cache()

        del gs_cpu, layer_inputs, ctxs_per_b
        gc.collect()
        torch.cuda.empty_cache()
        print(f"  [sketch_b] chunk {c_idx + 1}/{n_chunks} ({s1}/{n_seqs} seqs) done in {time.perf_counter() - t_ch:.1f}s (total +{time.perf_counter() - t0:.1f}s)", flush=True)

    # Save flatH lower-triangular `.pt` files and log effective rank diagnostics
    er_by_kind: Dict[str, List[float]] = {}
    for (l_idx, sn), G_sum in sorted(accum_cpu.items()):
        Gm = G_sum / float(max(n_seqs, 1))
        Gm = (Gm + Gm.T).mul_(0.5)
        er = _eff_rank(Gm)
        er_by_kind.setdefault(sn, []).append(er)
        torch.save(sym_to_flat(Gm.float()), out_dir / f"{l_idx}_{sn}_hout.pt")

    del accum_cpu
    gc.collect()
    wall_sec = round(time.perf_counter() - t0, 2)
    er_summary = {
        k: {"count": len(v), "mean_eff_rank": round(sum(v) / len(v), 1), "min": round(min(v), 1), "max": round(max(v), 1)}
        for k, v in er_by_kind.items()
    }
    meta = {
        "model_dir": str(ev.model_dir),
        "gpu": torch.cuda.get_device_name(0),
        "n_seqs": n_seqs,
        "seq_len": seq_len,
        "batch_size": batch_size,
        "chunk_seqs": chunk_seqs,
        "include_moe": include_moe,
        "wall_sec": wall_sec,
        "eff_rank_summary": er_summary,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"=== Saved Sketch-B Hessians to {out_dir} in {wall_sec:.1f}s ===", flush=True)
    for k, s in er_summary.items():
        print(f"  [{k:12s}] n={s['count']:2d} | eff_rank mean={s['mean_eff_rank']:.1f} (min={s['min']:.1f}, max={s['max']:.1f})", flush=True)
    return meta


def main():
    ap = argparse.ArgumentParser(description="Build streaming multi-domain Sketch-B output Hessians (Dense + MoE).")
    ap.add_argument("--model", choices=sorted(rpb.MODEL_SPECS), required=True)
    ap.add_argument("--models-root", type=Path, required=True)
    ap.add_argument("--work-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--n-seqs", type=int, default=512)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--chunk-seqs", type=int, default=128)
    ap.add_argument("--no-moe", action="store_true")
    args = ap.parse_args()

    spec = rpb.MODEL_SPECS[args.model]
    model_dir = args.models_root / spec["local_name"]
    m_work = args.work_dir / args.model
    m_work.mkdir(parents=True, exist_ok=True)

    dev = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False

    cal_cache = m_work / f"sketch_b_multidomain_{args.n_seqs}x{args.seq_len}.safetensors"
    cal_ids, cal_meta = build_multidomain_sketch_cal(model_dir, cal_cache, n_seqs=args.n_seqs, seq_len=args.seq_len)

    ev = rpb.StreamingCausalEvaluator(model_dir, dev)
    meta = build_streaming_sketch_b_hessians(
        ev,
        cal_ids,
        args.out_dir,
        batch_size=args.batch_size,
        chunk_seqs=args.chunk_seqs,
        include_moe=(not args.no_moe),
    )
    meta["calibration"] = cal_meta
    (args.out_dir / "sketch_b_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
