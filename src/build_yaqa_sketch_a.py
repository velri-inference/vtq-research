#!/usr/bin/env python3
"""
Architecture-General Single-Pass YAQA Sketch-A Builder (`src/build_yaqa_sketch_a.py`).

Builds BOTH KL-weighted Kronecker factors in ONE streaming forward+backward pass (`Sketch-A Iter-1`,
validated in `exp35` as strictly superior in effective rank, cosine accuracy, and speed vs both
Sketch-A Iter-2 and Sketch-B):
  - Output Hessian `G_out = E_t[ ||x_t||^2 g_t g_t^T ]` saved as `{l}_{short}_hout.pt`
    (for `q_proj, k_proj, v_proj, o_proj, out_proj, down_proj, shared_expert.down_proj`, and MoE `moe_down`)
  - Input  Hessian `H_in  = E_t[ ||g_t||^2 x_t x_t^T ]` saved as `{l}_{short}_hin.pt`
    (for all dense quantizable projections: `q, k, v, o, out, gate, up, down, shared_down`)

Supports Dense Llama, Gemma-4 MoE, and Qwen3.6 Hybrid Gated DeltaNet + MoE via `StreamingCausalEvaluator`.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import src.build_sketch_b_hessians as sb
import src.run_paper_benchmarks as rpb
from src.vtq_gptq2d import _short_yaqa_name, sym_to_flat

HOUT_DENSE_TARGETS = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "linear_attn.out_proj",
    "mlp.down_proj",
    "mlp.shared_expert.down_proj",
)


def _eff_rank(M: torch.Tensor) -> float:
    d = M.diagonal()
    return float(d.sum()) ** 2 / max(float((M * M).sum()), 1e-30)


def build_streaming_sketch_a_hessians(
    ev: rpb.StreamingCausalEvaluator,
    cal_ids: torch.Tensor,
    out_dir: Path,
    *,
    batch_size: int = 4,
    chunk_seqs: int = 128,
    save_hin: bool = True,
    include_moe: bool = True,
    trace_norm: bool = False,
    seed_base: int = 7000,
) -> dict:
    """
    Single-pass streaming YAQA Sketch-A (`Iter-1`) builder for `ev`.
    Stores GPU accumulators in float64 per chunk and merges into float32 CPU buffers across chunks.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "sketch_a_meta.json"
    n_seqs, seq_len = int(cal_ids.shape[0]), int(cal_ids.shape[1])

    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if (
            meta.get("n_seqs") == n_seqs
            and meta.get("seq_len") == seq_len
            and meta.get("save_hin") == save_hin
            and meta.get("include_moe") == include_moe
            and meta.get("trace_norm") == trace_norm
        ):
            print(f"[sketch_a] Reusing existing Sketch-A Hessians in {out_dir} ({meta.get('wall_sec')}s)", flush=True)
            return meta

    dev = ev.device
    ev.restore_clean_embed_and_head()
    t0 = time.perf_counter()

    bos_id = getattr(ev.config, "bos_token_id", None)
    if isinstance(bos_id, (list, tuple)):
        bos_id = bos_id[0] if bos_id else None
    if bos_id is not None:
        bos_col = torch.full((n_seqs, 1), int(bos_id), dtype=cal_ids.dtype)
        cal_ids = torch.cat([bos_col, cal_ids[:, :-1]], dim=1).contiguous()
        print(f"[sketch_a] Prepended BOS token id={bos_id} to all {n_seqs} sequences", flush=True)

    layer_dense_in: List[List[Tuple[str, str, int]]] = []
    layer_dense_out: List[List[Tuple[str, str, int]]] = []
    layer_has_moe: List[Tuple[bool, int]] = []
    accum_in: Dict[Tuple[int, str], torch.Tensor] = {}
    accum_out: Dict[Tuple[int, str], torch.Tensor] = {}

    for l_idx in range(ev.n_layers):
        lay = ev.layers[l_idx]
        din: List[Tuple[str, str, int]] = []
        dout: List[Tuple[str, str, int]] = []
        for name, mod in lay.named_modules():
            if not (isinstance(mod, nn.Linear) and rpb.is_quantizable_linear(name, mod)):
                continue
            sn = _short_yaqa_name(name) or name
            if save_hin:
                din.append((name, sn, int(mod.in_features)))
                accum_in[(l_idx, sn)] = torch.zeros((int(mod.in_features), int(mod.in_features)), dtype=torch.float32, device="cpu")
            if any(name == t or name.endswith("." + t) for t in HOUT_DENSE_TARGETS):
                dout.append((name, sn, int(mod.out_features)))
                accum_out[(l_idx, sn)] = torch.zeros((int(mod.out_features), int(mod.out_features)), dtype=torch.float32, device="cpu")
        layer_dense_in.append(din)
        layer_dense_out.append(dout)
        ex_obj, _ = rpb.get_moe_experts_and_prefix(lay)
        if include_moe and ex_obj is not None:
            m_dim = int(ex_obj.down_proj.shape[1])
            layer_has_moe.append((True, m_dim))
            accum_out[(l_idx, "moe_down")] = torch.zeros((m_dim, m_dim), dtype=torch.float32, device="cpu")
        else:
            layer_has_moe.append((False, 0))

    print(
        f"\n=== Building Streaming YAQA Sketch-A (Iter-1): {len(accum_in)} H_in + {len(accum_out)} H_out | "
        f"{n_seqs}x{seq_len} (B={batch_size}, chunk={chunk_seqs}, trace_norm={trace_norm}) ===",
        flush=True,
    )

    n_chunks = (n_seqs + chunk_seqs - 1) // chunk_seqs
    for c_idx in range(n_chunks):
        s0 = c_idx * chunk_seqs
        s1 = min(n_seqs, s0 + chunk_seqs)
        t_ch = time.perf_counter()

        batches = [cal_ids[i : min(i + batch_size, s1)] for i in range(s0, s1, batch_size)]
        n_b = len(batches)
        hs_cpu, ctxs_per_b = [], []
        for b_ids in batches:
            h0, ctxs = ev.capture_input_and_ctx(b_ids)
            hs_cpu.append(h0)
            ctxs_per_b.append(ctxs)

        layer_inputs = []
        for l_idx in range(ev.n_layers):
            layer_inputs.append(hs_cpu)
            lay = ev.load_layer_bf16(l_idx)
            for p in lay.parameters():
                p.requires_grad_(False)
            with torch.no_grad():
                hs_cpu = [ev.call_layer(lay, hs_cpu[b], ctxs_per_b[b][l_idx]) for b in range(n_b)]
            ev.release_layer(l_idx)

        gs_cpu = []
        seq_cursor = s0
        for b in range(n_b):
            h_batch = hs_cpu[b]
            g_parts = []
            for j in range(int(h_batch.shape[0])):
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
                    g_parts.append(torch.autograd.grad(loss, h_one)[0].detach().cpu())
                del logits, loss, h_one, y
                seq_cursor += 1
            gs_cpu.append(torch.cat(g_parts, dim=0))
        del hs_cpu
        torch.cuda.empty_cache()

        for l_idx in reversed(range(ev.n_layers)):
            lay = ev.load_layer_bf16(l_idx)
            for p in lay.parameters():
                p.requires_grad_(False)

            in_map = {nm: (sn, k_dim) for nm, sn, k_dim in layer_dense_in[l_idx]}
            out_map = {nm: (sn, m_dim) for nm, sn, m_dim in layer_dense_out[l_idx]}
            all_names = sorted(set(in_map) | set(out_map))

            H_gpu: Dict[str, torch.Tensor] = {
                sn: torch.zeros((k_dim, k_dim), dtype=torch.float64, device=dev) for _, sn, k_dim in layer_dense_in[l_idx]
            }
            G_gpu: Dict[str, torch.Tensor] = {
                sn: torch.zeros((m_dim, m_dim), dtype=torch.float64, device=dev) for _, sn, m_dim in layer_dense_out[l_idx]
            }
            has_moe, moe_m = layer_has_moe[l_idx]
            if has_moe:
                G_gpu["moe_down"] = torch.zeros((moe_m, moe_m), dtype=torch.float64, device=dev)
                G_seq_moe = torch.zeros((batch_size, moe_m, moe_m), dtype=torch.float64, device=dev) if trace_norm else None
            else:
                G_seq_moe = None

            cur_B = [batch_size]
            saved_x: Dict[str, Optional[torch.Tensor]] = {}
            hooks = []

            for sub_nm in all_names:
                w_in = sub_nm in in_map
                w_out = sub_nm in out_map
                sn_in = in_map[sub_nm][0] if w_in else None
                sn_out = out_map[sub_nm][0] if w_out else None

                def _mk(key_nm: str, _wi: bool, _wo: bool, _si: Optional[str], _so: Optional[str]):
                    def _fwd(m, inp, out):
                        if out.requires_grad:
                            bc = cur_B[0]
                            saved_x[key_nm] = inp[0].detach().float().reshape(bc, seq_len, -1)

                            def _bwd(gr, kn=key_nm, B_c=bc):
                                x_b = saved_x[kn]
                                saved_x[kn] = None
                                g_b = gr.detach().float().reshape(B_c, seq_len, -1)
                                if not trace_norm:
                                    x2 = x_b.reshape(-1, x_b.shape[-1])
                                    g2 = g_b.reshape(-1, g_b.shape[-1])
                                    if _wo and _so is not None:
                                        wx = (x2 * x2).sum(dim=-1)
                                        G_gpu[_so].add_(((g2 * wx.unsqueeze(1)).T @ g2).double())
                                    if _wi and _si is not None:
                                        wg = (g2 * g2).sum(dim=-1)
                                        H_gpu[_si].add_(((x2 * wg.unsqueeze(1)).T @ x2).double())
                                else:
                                    for j in range(B_c):
                                        xj, gj = x_b[j], g_b[j]
                                        if _wo and _so is not None:
                                            wx = (xj * xj).sum(dim=-1)
                                            Gb = ((gj * wx.unsqueeze(1)).T @ gj).double()
                                            tr = Gb.diagonal().sum()
                                            G_gpu[_so].add_(Gb * torch.where(tr > 1e-20, 1.0 / tr.clamp(min=1e-20), torch.zeros_like(tr)))
                                        if _wi and _si is not None:
                                            wg = (gj * gj).sum(dim=-1)
                                            Hb = ((xj * wg.unsqueeze(1)).T @ xj).double()
                                            tr = Hb.diagonal().sum()
                                            H_gpu[_si].add_(Hb * torch.where(tr > 1e-20, 1.0 / tr.clamp(min=1e-20), torch.zeros_like(tr)))
                                del x_b, g_b

                            out.register_hook(_bwd)

                    return _fwd

                hooks.append(lay.get_submodule(sub_nm).register_forward_hook(_mk(sub_nm, w_in, w_out, sn_in, sn_out)))

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
                            wx_e = (mid_e.detach().float() ** 2).sum(dim=-1)
                            seq_ids_e = torch.div(tok_i, seq_len, rounding_mode="floor") if trace_norm else None

                            def _moe_bwd(gr, wxe=wx_e, si=seq_ids_e, bc=B_c):
                                ge = gr.detach().float()
                                if not trace_norm:
                                    G_gpu["moe_down"].add_(((ge * wxe.unsqueeze(1)).T @ ge).double())
                                else:
                                    for j in range(bc):
                                        mj = (si == j).to(ge.dtype)
                                        G_seq_moe[j].add_(((ge * (wxe * mj).unsqueeze(1)).T @ ge).double())

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
                kwargs = sb._move_shared_kv_to_dev(kwargs, dev)
                g_out = gs_cpu[b].to(dev)
                with torch.enable_grad():
                    out = lay(x, *rest, **kwargs)
                    out = out[0] if isinstance(out, tuple) else out
                    gs_cpu[b] = torch.autograd.grad(out, x, grad_outputs=g_out)[0].detach().cpu()
                if G_seq_moe is not None:
                    trs = G_seq_moe[:B_c].diagonal(dim1=1, dim2=2).sum(dim=1)
                    scales = torch.where(trs > 1e-20, 1.0 / trs.clamp(min=1e-20), torch.zeros_like(trs))
                    G_gpu["moe_down"].add_((G_seq_moe[:B_c] * scales.view(B_c, 1, 1)).sum(dim=0))
                del x, out, g_out

            for hk in hooks:
                hk.remove()
            if patched_moe and ex_obj is not None:
                del ex_obj.forward
            if G_seq_moe is not None:
                del G_seq_moe

            for sn, t in G_gpu.items():
                accum_out[(l_idx, sn)].add_(t.float().cpu())
            for sn, t in H_gpu.items():
                accum_in[(l_idx, sn)].add_(t.float().cpu())
            del G_gpu, H_gpu
            ev.release_layer(l_idx)
            layer_inputs[l_idx] = []
            torch.cuda.empty_cache()

        del gs_cpu, layer_inputs, ctxs_per_b
        gc.collect()
        torch.cuda.empty_cache()
        print(f"  [sketch_a] chunk {c_idx + 1}/{n_chunks} ({s1}/{n_seqs} seqs) done in {time.perf_counter() - t_ch:.1f}s (total +{time.perf_counter() - t0:.1f}s)", flush=True)

    denom = float(max(n_seqs if trace_norm else n_seqs * seq_len, 1))
    er_out: Dict[str, List[float]] = {}
    for (l_idx, sn), M in sorted(accum_out.items()):
        Gm = 0.5 * (M + M.T) / denom
        er_out.setdefault(sn, []).append(_eff_rank(Gm))
        torch.save(sym_to_flat(Gm.float()), out_dir / f"{l_idx}_{sn}_hout.pt")

    er_in: Dict[str, List[float]] = {}
    for (l_idx, sn), M in sorted(accum_in.items()):
        Hm = 0.5 * (M + M.T) / denom
        er_in.setdefault(sn, []).append(_eff_rank(Hm))
        torch.save(sym_to_flat(Hm.float()), out_dir / f"{l_idx}_{sn}_hin.pt")

    del accum_in, accum_out
    gc.collect()
    wall_sec = round(time.perf_counter() - t0, 2)

    def _sum_er(d):
        return {k: {"count": len(v), "mean_eff_rank": round(sum(v) / len(v), 1), "min": round(min(v), 1), "max": round(max(v), 1)} for k, v in d.items()}

    meta = {
        "model_dir": str(ev.model_dir),
        "gpu": torch.cuda.get_device_name(0),
        "n_seqs": n_seqs,
        "seq_len": seq_len,
        "batch_size": batch_size,
        "chunk_seqs": chunk_seqs,
        "save_hin": save_hin,
        "include_moe": include_moe,
        "trace_norm": trace_norm,
        "wall_sec": wall_sec,
        "eff_rank_hout": _sum_er(er_out),
        "eff_rank_hin": _sum_er(er_in),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"=== Saved Sketch-A Hessians to {out_dir} in {wall_sec:.1f}s ===", flush=True)
    for k, s in meta["eff_rank_hout"].items():
        print(f"  [hout {k:11s}] n={s['count']:2d} | eff_rank mean={s['mean_eff_rank']:.1f} (min={s['min']:.1f}, max={s['max']:.1f})", flush=True)
    for k, s in meta["eff_rank_hin"].items():
        print(f"  [hin  {k:11s}] n={s['count']:2d} | eff_rank mean={s['mean_eff_rank']:.1f} (min={s['min']:.1f}, max={s['max']:.1f})", flush=True)
    return meta


def main():
    ap = argparse.ArgumentParser(description="Build single-pass multi-domain YAQA Sketch-A (H_in + H_out) Hessians.")
    ap.add_argument("--model", choices=sorted(rpb.MODEL_SPECS), required=True)
    ap.add_argument("--models-root", type=Path, required=True)
    ap.add_argument("--work-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--n-seqs", type=int, default=256)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--chunk-seqs", type=int, default=128)
    ap.add_argument("--no-hin", action="store_true")
    ap.add_argument("--no-moe", action="store_true")
    ap.add_argument("--trace-norm", action="store_true", help="Per-sequence trace normalization (recommended on Gemma-4).")
    args = ap.parse_args()

    spec = rpb.MODEL_SPECS[args.model]
    model_dir = args.models_root / spec["local_name"]
    m_work = args.work_dir / args.model
    m_work.mkdir(parents=True, exist_ok=True)

    dev = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False

    cal_cache = m_work / f"sketch_a_multidomain_{args.n_seqs}x{args.seq_len}.safetensors"
    cal_ids, cal_meta = sb.build_multidomain_sketch_cal(model_dir, cal_cache, n_seqs=args.n_seqs, seq_len=args.seq_len)

    ev = rpb.StreamingCausalEvaluator(model_dir, dev)
    meta = build_streaming_sketch_a_hessians(
        ev,
        cal_ids,
        args.out_dir,
        batch_size=args.batch_size,
        chunk_seqs=args.chunk_seqs,
        save_hin=(not args.no_hin),
        include_moe=(not args.no_moe),
        trace_norm=args.trace_norm,
    )
    meta["calibration"] = cal_meta
    (args.out_dir / "sketch_a_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
