"""
Comprehensive VTQ Multi-Model, Multi-Architecture & Hardware Speed Suite (`src/run_vtq_benchmarks.py`):
  - `--suite 3b`, `1b`, `3b_1b`:
      Full equal-bpw sweep (`1.75, 2.0, 2.10, 2.5, 3.0, 3.4375, 4.0, 4.5, 5.0, 6.0 bpw`) on `Llama-3.2-3B-Instruct`
      and `Llama-3.2-1B-Instruct` (`VTQ`, `VTQ-NoGPTAQ`, `VTQ-Polish`, `VTQ-Selective`), paired bootstrap (`B=2000`)
      against `VTQ2`, `VTQ2-Polish`, `VTQ v1` (`TCQPQ`), and `EXL3`; full `.vtq` binary container write/reload
      verification (`VtqWriter` -> `VtqFile`); and domain-neutral Wikipedia calibration + external `wikitext2` audit.
  - `--suite qwen3_8b`:
      Full equal-bpw sweep (`2.5, 3.0, 3.4375, 4.0, 4.5, 5.0 bpw`) on `Qwen3-8B` vs `VTQ2`, `VTQ2-Polish`, `TCQPQ`, `EXL3`.
  - `--suite llama8b`:
      Full equal-bpw sweep (`2.5, 3.0, 3.4375, 4.0, 4.5 bpw`) on `Llama-3.1-8B-Instruct` (`K = 14336 = 2048 * 7`, `Outer-DCT`).
  - `--suite qwen35`, `gemma4`, `hybrid`, `qwen27`:
      Architecture-agnostic `TextModel` + `VTQ` DP evaluation (`suh`-only `16 * K_in` overhead) on `Qwen3.5-4B`
      (Gated DeltaNet), `gemma-4-E4B-it` (PLE + shared KV + `VtqRing` PLE table), and `Qwen3.8-27B`.
  - `--suite moe_30b`:
      Streaming `VTQ` DP evaluation on `Qwen3-30B-A3B-Instruct-2507` (`48` layers, `128` experts) at `3.0, 3.5, 4.5 bpw`.
  - `--suite speed`:
      Controlled 3-generation (`VTQ v1`, `VTQ v2`, `VTQ`) CPU AVX2 (`-mno-avx512f`) + GPU Blackwell (`sm_120`) benchmark.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM

import src.equal_bpw as eqb
import src.hybrid_quant as hqm
import src.liqk_v2 as lv2
import src.streaming_quant as sqm
import src.vtq as vtq
import src.vtq_format as vtqf
from src.eval_kld import (
    evaluate_model_on_cached_trace,
    get_scored_positions_and_targets,
    load_eval_trace,
    paired_bootstrap_diff,
    precompute_ref_trace_data,
)
from src.run_1b_liqk_v2_experiments import restore_clean_model_weights, save_clean_model_weights
from src.run_vtq2_benchmarks import build_wikitext_eval_and_neutral_cal, compute_hybrid_fisher_offload

ROOT = Path(__file__).resolve().parent.parent


def _shares_pct(model_or_layers, assignment: dict) -> dict:
    sh: Dict[str, int] = {}
    layers = model_or_layers.model.layers if hasattr(model_or_layers, "model") else model_or_layers
    for (l_idx, sub), slices in assignment.items():
        mod = layers[l_idx].get_submodule(sub)
        K = mod.in_features if hasattr(mod, "in_features") else mod.weight.shape[1]
        for r0, r1, sp in slices:
            sh[sp] = sh.get(sp, 0) + (r1 - r0) * K
    tot = max(sum(sh.values()), 1)
    return {k: round(100.0 * v / tot, 2) for k, v in sorted(sh.items())}


def _pair_boot(res_a: dict, res_b: dict, datasets=("nemotron", "multilingual")) -> dict:
    out = {}
    for ds in datasets:
        if ds in res_a and ds in res_b and "per_seq_kld_sums" in res_a[ds] and "per_seq_kld_sums" in res_b[ds]:
            d = paired_bootstrap_diff(res_a[ds], res_b[ds])
            base_m = float(res_b[ds]["mean_kld"])
            d["rel_pct"] = round(100.0 * d["mean_diff"] / max(base_m, 1e-12), 2)
            out[ds] = d
    return out


def _fmt_tgt_label(t) -> str:
    if isinstance(t, str):
        return t
    fv = float(t)
    return f"{fv:.1f}" if abs(round(fv, 1) - fv) < 1e-6 else f"{fv:.4f}".rstrip("0").rstrip(".")


def overhead_bpw_vtq(tmod: hqm.TextModel, n_layers: int) -> float:
    """Exact `d_row[M] + suh[K]` header overhead (`16 * (M_r + K_in)` bits per quantized matrix; `u_sb[16]` is inside `st_bytes`)."""
    return hqm.overhead_bpw(tmod, n_layers)


# =================================================================================================
# 1. Suite `3b`, `1b`, `3b_1b`: Full Bitrate Sweep + `.vtq` Container Test + Neutral Audit
# =================================================================================================

def run_dense_sweep_model(tag: str, model_name: str, polish_steps: int, polish_lr: float, seed_json: Optional[Path] = None):
    dev = torch.device("cuda")
    vtq.load_and_register_vtq()
    model_dir = ROOT / "models" / model_name
    out_json = ROOT / "results" / f"vtq_{tag}_sweep.json"
    results = (
        json.loads(out_json.read_text(encoding="utf-8"))
        if out_json.exists()
        else {"model": model_name, "runs": {}, "comparisons": {}}
    )

    # Import cached baselines (`EXL3-*`, `TCQPQ-*`, `VTQ2-*`) from `vtq2_{tag}_sweep.json` & `equal_bpw_{tag}.json`
    for src_p in (ROOT / "results" / f"vtq2_{tag}_sweep.json", ROOT / "results" / f"equal_bpw_{tag}.json"):
        if src_p.exists():
            src_runs = json.loads(src_p.read_text(encoding="utf-8")).get("runs", {})
            for k, val in src_runs.items():
                if k.startswith(("EXL3-", "TCQPQ-", "VTQ2-")) and k not in results["runs"]:
                    results["runs"][k] = val

    if seed_json is not None and seed_json.exists():
        seed_runs = json.loads(seed_json.read_text(encoding="utf-8"))
        for k, val in seed_runs.items():
            if k.startswith("VTQ-") and k not in results["runs"]:
                results["runs"][k] = val

    print(f"\n=== Dense Equal-BPW VTQ Sweep on `{model_name}` (`{tag}`) ===", flush=True)
    model = AutoModelForCausalLM.from_pretrained(str(model_dir), dtype=torch.bfloat16).to(dev).eval()
    clean = save_clean_model_weights(model)
    cal64 = load_file(str(ROOT / "data" / "cal_trace.safetensors"))["input_ids"][:64]

    nem_ref = precompute_ref_trace_data(model, load_eval_trace(ROOT / "data" / "nemotron_trace.json"), dev)
    mul_ref = precompute_ref_trace_data(model, load_eval_trace(ROOT / "data" / "multilingual_trace.json"), dev)
    ref_hidden = lv2.precompute_ref_final_hidden_states(model, cal64, n_seqs=64, seq_len=512, batch_size=4 if tag == "1b" else 2)

    fisher_path = ROOT / "data" / f"fisher_{tag}.pt"
    if fisher_path.exists():
        fisher = torch.load(str(fisher_path), map_location="cpu", weights_only=True)
    else:
        fisher = lv2.compute_exact_row_fisher_sensitivities(model, cal64, n_Menu_seqs=8, seq_len=1024)
        torch.save(fisher, str(fisher_path))

    t0 = time.perf_counter()
    prof_vtq = vtq.build_vtq_fisher_slice_profile(
        model, cal64, candidate_specs=list(vtq.VTQ_MENU), batch_size=8, fisher_out=fisher, rot_block=-1, rot_mode="outer"
    )
    prof_vtq_sec = time.perf_counter() - t0
    print(f"  Built VTQ Fisher slice profile ({len(prof_vtq['slices'])} slices) in {prof_vtq_sec:.1f}s", flush=True)

    def save_eval(label: str, info: dict):
        entry = {
            "info": info,
            "nemotron": evaluate_model_on_cached_trace(model, nem_ref, dev),
            "multilingual": evaluate_model_on_cached_trace(model, mul_ref, dev),
        }
        results["runs"][label] = entry
        out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(
            f"  [{label:26s}] bpw={info['linear_bpw']:.4f} t={info.get('quant_sec', 0):.1f}s | "
            f"nem={entry['nemotron']['mean_kld']:.5f} mul={entry['multilingual']['mean_kld']:.5f} | "
            f"top1={entry['nemotron']['top1_match']*100:.2f}%/{entry['multilingual']['top1_match']*100:.2f}%",
            flush=True,
        )

    vtq_targets = (1.75, 2.0, 2.10, 2.5, 3.0, 3.4375, 4.0, 4.5, 5.0, 6.0)
    nogptaq_targets = (2.5, 3.0, 3.4375, 4.0, 4.5) if tag == "3b" else (3.0, 4.0)
    polish_targets = (2.0, 2.10, 2.5, 3.0, 3.4375, 4.0, 4.5) if tag == "3b" else (2.5, 3.0, 3.4375, 4.0)

    for target in vtq_targets:
        cands = vtq.vtq_candidates_for_target(target)
        asgn3 = lv2.solve_fisher_slice_dp(prof_vtq, target_bpw=target, candidate_specs=cands)

        if target in nogptaq_targets:
            lbl_ng = f"VTQ-{target:.4f}-NoGPTAQ"
            if lbl_ng not in results["runs"]:
                restore_clean_model_weights(model, clean)
                q_ng = vtq.quantize_llama_model_vtq(
                    model, cal64, asgn3, mlp_perms=prof_vtq["mlp_perms"], fisher_out=prof_vtq["fisher_out"],
                    batch_size=4, gptaq_alpha=0.0, rot_block=-1, rot_mode="outer", d_sb_gecd_sweeps=None, polish_steps=0
                )
                save_eval(
                    lbl_ng,
                    {
                        "linear_bpw": q_ng["linear_bpw"],
                        "quant_sec": q_ng["quant_sec"],
                        "profile_sec": prof_vtq_sec,
                        "target_bpw": target,
                        "shares_pct": _shares_pct(model, asgn3),
                    },
                )

        lbl = f"VTQ-{target:.4f}"
        lbl_pol = f"VTQ-{target:.4f}-Polish"
        need_main = lbl not in results["runs"]
        need_pol = (target in polish_targets) and (lbl_pol not in results["runs"])
        if need_main or need_pol:
            restore_clean_model_weights(model, clean)
            q = vtq.quantize_llama_model_vtq(
                model, cal64, asgn3, mlp_perms=prof_vtq["mlp_perms"], fisher_out=prof_vtq["fisher_out"],
                batch_size=4, gptaq_alpha=0.50, rot_block=-1, rot_mode="outer", d_sb_gecd_sweeps=None, polish_steps=0
            )
            info = {
                "linear_bpw": q["linear_bpw"],
                "quant_sec": q["quant_sec"],
                "profile_sec": prof_vtq_sec,
                "target_bpw": target,
                "shares_pct": _shares_pct(model, asgn3),
            }
            if need_main:
                save_eval(lbl, info)
            if need_pol:
                with torch.enable_grad():
                    pol_s = vtq.polish_model_scales_kld_checkpointed(
                        model, cal64, ref_hidden, n_steps=polish_steps, lr=polish_lr, batch_size=2, reg_wd=0.02, ce_weight=0.0
                    )
                save_eval(lbl_pol, dict(info, polish_sec=pol_s))

    # Selective Block-256 FWHT (`rot_mode='selective'`) check on 3B @ 3.0000 bpw:
    if tag == "3b" and "VTQ-3.0000-Selective" not in results["runs"]:
        t0_s = time.perf_counter()
        prof_sel = vtq.build_vtq_fisher_slice_profile(
            model, cal64, batch_size=8, fisher_out=fisher, rot_block=-1, rot_mode="selective"
        )
        p_sel_sec = time.perf_counter() - t0_s
        asgn_sel = lv2.solve_fisher_slice_dp(prof_sel, target_bpw=3.0, candidate_specs=vtq.vtq_candidates_for_target(3.0))
        restore_clean_model_weights(model, clean)
        q_sel = vtq.quantize_llama_model_vtq(
            model, cal64, asgn_sel, mlp_perms=prof_sel["mlp_perms"], fisher_out=prof_sel["fisher_out"],
            batch_size=4, gptaq_alpha=0.50, rot_block=-1, rot_mode="selective", d_sb_gecd_sweeps=None, polish_steps=0
        )
        save_eval(
            "VTQ-3.0000-Selective",
            {
                "linear_bpw": q_sel["linear_bpw"],
                "quant_sec": q_sel["quant_sec"],
                "profile_sec": p_sel_sec,
                "target_bpw": 3.0,
                "shares_pct": _shares_pct(model, asgn_sel),
            },
        )

    # Full `.vtq` container write + reload verification on 3B @ 3.0000 bpw across all 196 matrices!
    if tag == "3b" and "vtq_file_verification" not in results:
        print("\n  === Packing Full `Llama-3.2-3B-Instruct @ 3.0000 bpw` into `.vtq` & Reloading ===", flush=True)
        restore_clean_model_weights(model, clean)
        asgn_pack = lv2.solve_fisher_slice_dp(prof_vtq, target_bpw=3.0, candidate_specs=vtq.vtq_candidates_for_target(3.0))
        fmts_all = list(vtqf.load_vtq_formats().values())
        with tempfile.TemporaryDirectory() as td:
            vtq_path = Path(td) / "llama3_2_3b_vtq_3bpw.vtq"
            writer = vtqf.VtqWriter(vtq_path, fmts_all, reserve=(8 << 20) - 16)
            orig_qm = vtq.quantize_matrix_vtq
            mat_counter = [0]
            packed_names = []

            def _pack_wrap(W, H, r_specs, **kw):
                vtq.PACK = {}
                out = orig_qm(W, H, r_specs, **kw)
                m_idx = mat_counter[0]
                mat_counter[0] += 1
                l_i, s_i = m_idx // 7, m_idx % 7
                p_name = f"model.layers.{l_i}.{lv2.LINEAR_SUBMODULES[s_i]}"
                writer.add_matrix(p_name, vtq.PACK["matrix"], verify=True)
                packed_names.append((l_i, lv2.LINEAR_SUBMODULES[s_i], p_name))
                vtq.PACK = None
                return out

            vtq.quantize_matrix_vtq = _pack_wrap
            try:
                vtq.quantize_llama_model_vtq(
                    model, cal64, asgn_pack, mlp_perms=prof_vtq["mlp_perms"], fisher_out=prof_vtq["fisher_out"],
                    batch_size=4, gptaq_alpha=0.50, rot_block=-1, rot_mode="outer", d_sb_gecd_sweeps=None, polish_steps=0
                )
            finally:
                vtq.quantize_matrix_vtq = orig_qm

            writer.finalize(
                {
                    "format": "vtq",
                    "quantization": {
                        "quant_method": "vtq",
                        "version": 1,
                        "rot_block": -1,
                        "bits_per_weight": 3.0,
                        "formats": [vtqf.format_json_vtq(f) for f in fmts_all],
                    },
                }
            )
            nem_mem = evaluate_model_on_cached_trace(model, nem_ref, dev)["mean_kld"]
            mul_mem = evaluate_model_on_cached_trace(model, mul_ref, dev)["mean_kld"]

            for l_i, sub_nm, _ in packed_names:
                model.model.layers[l_i].get_submodule(sub_nm).weight.data.zero_()
            vf = vtqf.VtqFile(vtq_path)
            for l_i, sub_nm, p_name in packed_names:
                w_dec = vf.decode_matrix(p_name, device=dev).to(torch.bfloat16)
                model.model.layers[l_i].get_submodule(sub_nm).weight.data.copy_(w_dec)
            nem_file = evaluate_model_on_cached_trace(model, nem_ref, dev)["mean_kld"]
            mul_file = evaluate_model_on_cached_trace(model, mul_ref, dev)["mean_kld"]
            file_sz = vtq_path.stat().st_size
            print(
                f"  [PASS] `.vtq` 3B file ({file_sz / 1e9:.3f} GB, {len(packed_names)} matrices): "
                f"mem={nem_mem:.6f}/{mul_mem:.6f} vs file={nem_file:.6f}/{mul_file:.6f}",
                flush=True,
            )
            assert abs(nem_mem - nem_file) < 1e-6 and abs(mul_mem - mul_file) < 1e-6
            results["vtq_file_verification"] = {
                "quantized_matrices": len(packed_names),
                "matrix_container_bytes": file_sz,
                "nem_kld_mem": nem_mem,
                "nem_kld_file": nem_file,
                "mul_kld_mem": mul_mem,
                "mul_kld_file": mul_file,
            }

    # Paired bootstrap (`B=2000`) against `VTQ2`, `VTQ2-Polish`, `TCQPQ`, and `EXL3`
    comps = {}
    for target in vtq_targets:
        t = f"{target:.4f}"
        for cand in (f"VTQ-{t}", f"VTQ-{t}-NoGPTAQ", f"VTQ-{t}-Polish", f"VTQ-{t}-Selective"):
            if cand not in results["runs"]:
                continue
            for ref in (f"VTQ2-{t}", f"VTQ2-{t}-Polish", f"TCQPQ-{t}", f"EXL3-{t}", f"VTQ-{t}-NoGPTAQ"):
                if ref in results["runs"] and cand != ref:
                    comps[f"{cand} vs {ref}"] = _pair_boot(results["runs"][cand], results["runs"][ref])
                    d = comps[f"{cand} vs {ref}"]
                    print(
                        f"  {cand} vs {ref}: nem {d['nemotron']['mean_diff']:+.5f} ({d['nemotron']['rel_pct']:+.1f}%, p={d['nemotron']['p_a_better']:.4f}) | "
                        f"mul {d['multilingual']['mean_diff']:+.5f} ({d['multilingual']['rel_pct']:+.1f}%, p={d['multilingual']['p_a_better']:.4f})",
                        flush=True,
                    )
    results["comparisons"] = comps
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")

    # Domain-neutral Wikipedia calibration + external `wikitext2` trace audit on 3B
    if tag == "3b":
        audit_out = ROOT / "results" / "vtq_3b_neutral_audit.json"
        if not audit_out.exists():
            print("\n=== Domain-Neutral Wikipedia Calibration & External `wikitext2` Trace Audit (3B VTQ) ===", flush=True)
            restore_clean_model_weights(model, clean)
            wt_seqs, cal_neutral = build_wikitext_eval_and_neutral_cal(model_dir, n_cal_rows=64, seq_len=2048)
            wt_ref = precompute_ref_trace_data(model, wt_seqs, dev)
            fisher_neu = lv2.compute_exact_row_fisher_sensitivities(model, cal_neutral, n_Menu_seqs=8, seq_len=1024)
            prof_v3_neu = vtq.build_vtq_fisher_slice_profile(model, cal_neutral, batch_size=8, fisher_out=fisher_neu, rot_block=-1)

            neu_runs = {}
            v2_audit_p = ROOT / "results" / "vtq2_3b_neutral_audit.json"
            if v2_audit_p.exists():
                neu_runs.update(json.loads(v2_audit_p.read_text(encoding="utf-8")).get("runs", {}))

            for target in (3.0, 4.0):
                t = f"{target:.4f}"
                a3 = lv2.solve_fisher_slice_dp(prof_v3_neu, target_bpw=target, candidate_specs=vtq.vtq_candidates_for_target(target))
                restore_clean_model_weights(model, clean)
                q3_ng = vtq.quantize_llama_model_vtq(
                    model, cal_neutral, a3, mlp_perms=prof_v3_neu["mlp_perms"],
                    fisher_out=prof_v3_neu["fisher_out"], batch_size=4, gptaq_alpha=0.0
                )
                neu_runs[f"VTQ-NoGPTAQ-NeuCal-{t}"] = {
                    "info": {"linear_bpw": q3_ng["linear_bpw"], "quant_sec": q3_ng["quant_sec"]},
                    "nemotron": evaluate_model_on_cached_trace(model, nem_ref, dev),
                    "multilingual": evaluate_model_on_cached_trace(model, mul_ref, dev),
                    "wikitext2": evaluate_model_on_cached_trace(model, wt_ref, dev),
                }
                restore_clean_model_weights(model, clean)
                q3_g = vtq.quantize_llama_model_vtq(
                    model, cal_neutral, a3, mlp_perms=prof_v3_neu["mlp_perms"],
                    fisher_out=prof_v3_neu["fisher_out"], batch_size=4, gptaq_alpha=0.50
                )
                neu_runs[f"VTQ-NeuCal-{t}"] = {
                    "info": {"linear_bpw": q3_g["linear_bpw"], "quant_sec": q3_g["quant_sec"]},
                    "nemotron": evaluate_model_on_cached_trace(model, nem_ref, dev),
                    "multilingual": evaluate_model_on_cached_trace(model, mul_ref, dev),
                    "wikitext2": evaluate_model_on_cached_trace(model, wt_ref, dev),
                }
                for k_nm in (f"VTQ-NoGPTAQ-NeuCal-{t}", f"VTQ-NeuCal-{t}"):
                    r_e = neu_runs[k_nm]
                    print(
                        f"  [{k_nm:24s}] nem={r_e['nemotron']['mean_kld']:.5f} mul={r_e['multilingual']['mean_kld']:.5f} "
                        f"wiki2={r_e['wikitext2']['mean_kld']:.5f}",
                        flush=True,
                    )

            neu_comps = {}
            for target in (3.0, 4.0):
                t = f"{target:.4f}"
                for cand in (f"VTQ-NeuCal-{t}", f"VTQ-NoGPTAQ-NeuCal-{t}"):
                    for ref in (f"VTQ2-NeuCal-{t}", f"VTQ2-NoGPTAQ-NeuCal-{t}", f"TCQPQ-NeuCal-{t}"):
                        if cand in neu_runs and ref in neu_runs:
                            neu_comps[f"{cand} vs {ref}"] = _pair_boot(
                                neu_runs[cand], neu_runs[ref], ("nemotron", "multilingual", "wikitext2")
                            )
            audit_out.write_text(json.dumps({"runs": neu_runs, "comparisons": neu_comps}, indent=2), encoding="utf-8")

    del model, clean, nem_ref, mul_ref, ref_hidden
    torch.cuda.empty_cache()


# =================================================================================================
# 2. Suite `qwen3_8b`: Full Equal-BPW Sweep on `Qwen3-8B` vs `VTQ2`, `TCQPQ`, `EXL3`
# =================================================================================================

def run_suite_qwen3_8b():
    dev = torch.device("cuda")
    vtq.load_and_register_vtq()
    model_dir = ROOT / "models" / "Qwen3-8B"
    data_dir = ROOT / "data" / "qwen3"
    out_json = ROOT / "results" / "vtq_qwen3_8b.json"
    results = (
        json.loads(out_json.read_text(encoding="utf-8"))
        if out_json.exists()
        else {"model": "Qwen3-8B", "runs": {}, "comparisons": {}}
    )

    v2_p = ROOT / "results" / "vtq2_qwen3_8b.json"
    if v2_p.exists():
        for k, val in json.loads(v2_p.read_text(encoding="utf-8")).get("runs", {}).items():
            results["runs"].setdefault(k, val)

    print("\n=== Full Equal-BPW VTQ Sweep on `Qwen3-8B` (`2.5 .. 5.0 bpw`) ===", flush=True)
    model = AutoModelForCausalLM.from_pretrained(str(model_dir), dtype=torch.bfloat16).to(dev).eval()
    clean = save_clean_model_weights(model)
    cal64 = load_file(str(data_dir / "cal_trace.safetensors"))["input_ids"][:64]
    fisher = torch.load(str(data_dir / "fisher.pt"), map_location="cpu", weights_only=True)

    nem_ref = precompute_ref_trace_data(model, load_eval_trace(data_dir / "nemotron_trace.json"), dev)
    mul_ref = precompute_ref_trace_data(model, load_eval_trace(data_dir / "multilingual_trace.json"), dev)

    t0 = time.perf_counter()
    prof_vtq = vtq.build_vtq_fisher_slice_profile(model, cal64, batch_size=4, fisher_out=fisher, rot_block=-1)
    prof_vtq_sec = time.perf_counter() - t0
    print(f"  Built Qwen3-8B VTQ Fisher slice profile ({len(prof_vtq['slices'])} slices) in {prof_vtq_sec:.1f}s", flush=True)

    def save_eval(label: str, info: dict):
        entry = {
            "info": info,
            "nemotron": evaluate_model_on_cached_trace(model, nem_ref, dev),
            "multilingual": evaluate_model_on_cached_trace(model, mul_ref, dev),
        }
        results["runs"][label] = entry
        out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(
            f"  [{label:24s}] bpw={info['linear_bpw']:.4f} t={info.get('quant_sec', 0):.1f}s | "
            f"nem={entry['nemotron']['mean_kld']:.5f} mul={entry['multilingual']['mean_kld']:.5f}",
            flush=True,
        )

    ref_hidden = lv2.precompute_ref_final_hidden_states(model, cal64, n_seqs=64, seq_len=512, batch_size=4)
    targets = (2.5, 3.0, 3.4375, 4.0, 4.5, 5.0)
    for target in targets:
        asgn3 = lv2.solve_fisher_slice_dp(prof_vtq, target_bpw=target, candidate_specs=vtq.vtq_candidates_for_target(target))
        lbl3 = f"VTQ-{target:.4f}"
        lbl3_pol = f"VTQ-{target:.4f}-Polish"
        need_m = lbl3 not in results["runs"]
        need_p = (target in (2.5, 3.0, 3.4375)) and (lbl3_pol not in results["runs"])
        if need_m or need_p:
            restore_clean_model_weights(model, clean)
            q3 = vtq.quantize_llama_model_vtq(
                model, cal64, asgn3, mlp_perms=prof_vtq["mlp_perms"],
                fisher_out=prof_vtq["fisher_out"], batch_size=4, gptaq_alpha=None
            )
            inf3 = {
                "linear_bpw": q3["linear_bpw"],
                "quant_sec": q3["quant_sec"],
                "profile_sec": prof_vtq_sec,
                "target_bpw": target,
                "shares_pct": _shares_pct(model, asgn3),
            }
            if need_m:
                save_eval(lbl3, inf3)
            if need_p:
                with torch.enable_grad():
                    pol_s = vtq.polish_model_scales_kld_checkpointed(
                        model, cal64, ref_hidden, n_steps=120, lr=4e-4, batch_size=2, reg_wd=0.02, ce_weight=0.0
                    )
                save_eval(lbl3_pol, dict(inf3, polish_sec=pol_s))

    comps = {}
    for target in targets:
        t = f"{target:.4f}"
        for cand in (f"VTQ-{t}", f"VTQ-{t}-Polish"):
            if cand not in results["runs"]:
                continue
            for ref in (f"VTQ2-{t}", f"VTQ2-{t}-Polish", f"TCQPQ-{t}", f"EXL3-{t}", f"VTQ-{t}"):
                if ref in results["runs"] and cand != ref:
                    comps[f"{cand} vs {ref}"] = _pair_boot(results["runs"][cand], results["runs"][ref])
                    d = comps[f"{cand} vs {ref}"]
                    print(
                        f"  {cand} vs {ref}: nem {d['nemotron']['mean_diff']:+.5f} ({d['nemotron']['rel_pct']:+.1f}%, p={d['nemotron']['p_a_better']:.4f}) | "
                        f"mul {d['multilingual']['mean_diff']:+.5f} ({d['multilingual']['rel_pct']:+.1f}%, p={d['multilingual']['p_a_better']:.4f})",
                        flush=True,
                    )
    results["comparisons"] = comps
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    del model, clean, nem_ref, mul_ref, ref_hidden
    torch.cuda.empty_cache()


# =================================================================================================
# 3. Suite `llama8b`: `Llama-3.1-8B-Instruct` (`down_proj` K = 14336 = 2048 * 7, `Outer-DCT`)
# =================================================================================================

def run_suite_llama8b():
    dev = torch.device("cuda")
    vtq.load_and_register_vtq()
    m_dir = ROOT / "models" / "Llama-3.1-8B-Instruct"
    out_j = ROOT / "results" / "vtq_llama31_8b.json"
    res8 = (
        json.loads(out_j.read_text(encoding="utf-8"))
        if out_j.exists()
        else {"model": "Llama-3.1-8B-Instruct", "runs": {}, "comparisons": {}}
    )
    v2_l8 = ROOT / "results" / "vtq2_llama31_8b.json"
    if v2_l8.exists():
        for k, val in json.loads(v2_l8.read_text(encoding="utf-8")).get("runs", {}).items():
            res8["runs"].setdefault(k, val)

    print("\n=== Full Equal-BPW VTQ Sweep on `Llama-3.1-8B-Instruct` (`2.5 .. 4.5 bpw`, `K=14336` `Outer-DCT`) ===", flush=True)
    m8 = AutoModelForCausalLM.from_pretrained(str(m_dir), dtype=torch.bfloat16).to(dev).eval()
    cl8 = save_clean_model_weights(m8)
    cal64 = load_file(str(ROOT / "data" / "cal_trace.safetensors"))["input_ids"][:64]
    nem_r = precompute_ref_trace_data(m8, load_eval_trace(ROOT / "data" / "nemotron_trace.json"), dev)
    mul_r = precompute_ref_trace_data(m8, load_eval_trace(ROOT / "data" / "multilingual_trace.json"), dev)

    f8_path = ROOT / "data" / "fisher_llama8b.pt"
    if f8_path.exists():
        f8 = torch.load(str(f8_path), map_location="cpu", weights_only=True)
    else:
        f8 = lv2.compute_exact_row_fisher_sensitivities(m8, cal64, n_Menu_seqs=8, seq_len=1024)
        torch.save(f8, str(f8_path))

    t0 = time.perf_counter()
    p_v3 = vtq.build_vtq_fisher_slice_profile(m8, cal64, batch_size=4, fisher_out=f8, rot_block=-1)
    p_v3_sec = time.perf_counter() - t0
    ref_h8 = lv2.precompute_ref_final_hidden_states(m8, cal64, n_seqs=64, seq_len=512, batch_size=2)

    for tgt in (2.5, 3.0, 3.4375, 4.0, 4.5):
        t_s = f"{tgt:.4f}"
        a3 = lv2.solve_fisher_slice_dp(p_v3, target_bpw=tgt, candidate_specs=vtq.vtq_candidates_for_target(tgt))

        if tgt == 3.0 and f"VTQ-{t_s}-NoGPTAQ" not in res8["runs"]:
            restore_clean_model_weights(m8, cl8)
            q_ng = vtq.quantize_llama_model_vtq(
                m8, cal64, a3, mlp_perms=p_v3["mlp_perms"], fisher_out=p_v3["fisher_out"], batch_size=4, gptaq_alpha=0.0
            )
            res8["runs"][f"VTQ-{t_s}-NoGPTAQ"] = {
                "info": dict(q_ng, profile_sec=p_v3_sec, shares_pct=_shares_pct(m8, a3)),
                "nemotron": evaluate_model_on_cached_trace(m8, nem_r, dev),
                "multilingual": evaluate_model_on_cached_trace(m8, mul_r, dev),
            }
            out_j.write_text(json.dumps(res8, indent=2), encoding="utf-8")

        need_m = f"VTQ-{t_s}" not in res8["runs"]
        need_p = (tgt in (2.5, 3.0, 3.4375)) and (f"VTQ-{t_s}-Polish" not in res8["runs"])
        if need_m or need_p:
            restore_clean_model_weights(m8, cl8)
            q3 = vtq.quantize_llama_model_vtq(
                m8, cal64, a3, mlp_perms=p_v3["mlp_perms"], fisher_out=p_v3["fisher_out"], batch_size=4, gptaq_alpha=0.50
            )
            inf8 = dict(q3, profile_sec=p_v3_sec, shares_pct=_shares_pct(m8, a3))
            if need_m:
                res8["runs"][f"VTQ-{t_s}"] = {
                    "info": inf8,
                    "nemotron": evaluate_model_on_cached_trace(m8, nem_r, dev),
                    "multilingual": evaluate_model_on_cached_trace(m8, mul_r, dev),
                }
                print(
                    f"  [Llama8B VTQ-{t_s}       ] nem={res8['runs'][f'VTQ-{t_s}']['nemotron']['mean_kld']:.5f} "
                    f"mul={res8['runs'][f'VTQ-{t_s}']['multilingual']['mean_kld']:.5f}",
                    flush=True,
                )
            if need_p:
                with torch.enable_grad():
                    ps = vtq.polish_model_scales_kld_checkpointed(m8, cal64, ref_h8, n_steps=100, lr=4e-4, batch_size=2, reg_wd=0.02, ce_weight=0.0)
                res8["runs"][f"VTQ-{t_s}-Polish"] = {
                    "info": dict(inf8, polish_sec=ps),
                    "nemotron": evaluate_model_on_cached_trace(m8, nem_r, dev),
                    "multilingual": evaluate_model_on_cached_trace(m8, mul_r, dev),
                }
                print(
                    f"  [Llama8B VTQ-{t_s}-Polish] nem={res8['runs'][f'VTQ-{t_s}-Polish']['nemotron']['mean_kld']:.5f} "
                    f"mul={res8['runs'][f'VTQ-{t_s}-Polish']['multilingual']['mean_kld']:.5f}",
                    flush=True,
                )
        for cand in (f"VTQ-{t_s}", f"VTQ-{t_s}-NoGPTAQ", f"VTQ-{t_s}-Polish"):
            if cand in res8["runs"]:
                for rk in (f"VTQ2-{t_s}", f"VTQ2-{t_s}-Polish", f"TCQPQ-{t_s}", f"EXL3-{t_s}", f"VTQ-{t_s}"):
                    if rk in res8["runs"] and cand != rk:
                        res8["comparisons"][f"{cand} vs {rk}"] = _pair_boot(res8["runs"][cand], res8["runs"][rk])
                        d = res8["comparisons"][f"{cand} vs {rk}"]
                        print(
                            f"  {cand} vs {rk}: nem {d['nemotron']['mean_diff']:+.5f} ({d['nemotron']['rel_pct']:+.1f}%, p={d['nemotron']['p_a_better']:.4f}) | "
                            f"mul {d['multilingual']['mean_diff']:+.5f} ({d['multilingual']['rel_pct']:+.1f}%, p={d['multilingual']['p_a_better']:.4f})",
                            flush=True,
                        )
        out_j.write_text(json.dumps(res8, indent=2), encoding="utf-8")
    del m8, cl8, nem_r, mul_r, ref_h8
    torch.cuda.empty_cache()


# =================================================================================================
# 4. Architecture-Agnostic Hybrid `VTQ` Runner (`Qwen3.5-4B` DeltaNet, `Gemma-4-E4B` PLE, `Qwen3.8-27B`)
# =================================================================================================

@torch.no_grad()
def run_hybrid_vtq_eval(
    model_dir: Path,
    data_dir: Path,
    fisher_path: Path,
    targets: tuple,
    out_json: Path,
    cached_baseline_jsons: Tuple[Path, ...] = (),
    extra_ref_pairs: Optional[Dict[str, List[str]]] = None,
):
    print(f"\n=== Evaluating Hybrid Model `{model_dir.name}` (`TextModel` + `VTQ` DP) ===", flush=True)
    dev = torch.device("cuda")
    vtq.install_vtq_hooks()
    results = (
        json.loads(out_json.read_text(encoding="utf-8"))
        if out_json.exists()
        else {"model": str(model_dir), "methods": {}, "comparisons": {}}
    )
    for cb_p in cached_baseline_jsons:
        if cb_p is not None and cb_p.exists():
            base_m = json.loads(cb_p.read_text(encoding="utf-8")).get("methods", {})
            for k, val in base_m.items():
                if k not in results["methods"]:
                    results["methods"][k] = {kk: vv for kk, vv in val.items() if kk != "tensors"}

    tmod = hqm.TextModel(model_dir, dev)
    n_layers = len(tmod.layers)
    sqm.FISHER = torch.load(str(fisher_path), map_location="cpu", weights_only=True)

    traces = {}
    for ds in ("nemotron", "multilingual"):
        items = []
        for sq_item in load_eval_trace(data_dir / f"{ds}_trace.json"):
            pos, tg = get_scored_positions_and_targets(sq_item)
            if pos:
                items.append((torch.tensor([sq_item["token_ids"][:8192]]), pos, tg))
        traces[ds] = items

    cal_tokens = load_file(str(data_dir / "cal_trace.safetensors"))["input_ids"][:64]
    cal_ids = [cal_tokens[i : i + 4].to(dev) for i in range(0, 64, 4)]
    emb = {ds: [tmod.capture(ids.to(dev), 0)[0].cpu() for ids, _, _ in items] for ds, items in traces.items()}
    cal0 = [tmod.capture(ids, 0)[0] for ids in cal_ids]
    over3 = overhead_bpw_vtq(tmod, n_layers)

    need_any_v3 = [t for t in targets if f"dp:VTQ:{_fmt_tgt_label(t)}" not in results["methods"]]
    prof_vtq = None
    prof_vtq_sec = 0.0
    if need_any_v3:
        os.environ["VTQ_PROFILE"] = "1"
        t0 = time.perf_counter()
        prof_vtq = hqm.build_profile(tmod, cal_ids[:4], n_layers)
        prof_vtq_sec = time.perf_counter() - t0
        os.environ["VTQ_PROFILE"] = "0"
        print(f"  Built hybrid VTQ profile ({len(prof_vtq['slices'])} slices) in {prof_vtq_sec:.1f}s (suh over={over3:.5f} bpw)", flush=True)

    active = {}
    for t in targets:
        t_lbl = _fmt_tgt_label(t)
        k_v3 = f"dp:VTQ:{t_lbl}"
        if k_v3 not in results["methods"] and prof_vtq is not None:
            eff_tgt = float(t) - over3
            asgn3 = lv2.solve_fisher_slice_dp(
                prof_vtq, target_bpw=eff_tgt, candidate_specs=vtq.vtq_candidates_for_target(eff_tgt)
            )
            active[k_v3] = {
                "menu": "VTQ",
                "asgn": asgn3,
                "gptaq": 0.0,
                "quant_sec": 0.0,
                "bits": 0.0,
                "params": 0,
                "q_bits": 0.0,
                "q_params": 0,
                "streams": {ds: [e.clone() for e in v] for ds, v in emb.items()},
                "states": {ds: [{} for _ in v] for ds, v in emb.items()},
                "cal": [c.clone() for c in cal0],
                "cal_states": [{} for _ in cal0],
            }

    if active:
        ref = {"streams": emb, "states": {ds: [{} for _ in v] for ds, v in emb.items()}}
        cal_ref = [c.clone() for c in cal0]
        cal_ref_states = [{} for _ in cal0]

        for l_idx in range(n_layers):
            layer_ref = tmod.load_layer(l_idx)
            cal_ctx = [tmod.capture(ids, l_idx)[1:] for ids in cal_ids]
            qlayers = {}
            sqm._CUR_LAYER[0] = l_idx

            for m_name, st in active.items():
                t1 = time.perf_counter()
                lq = copy.deepcopy(layer_ref)
                mods_q = dict(lq.named_modules())
                mods_r = dict(layer_ref.named_modules())
                groups = hqm.linear_groups(lq, lambda: hqm.call_layer(lq, st["cal"][0], cal_ctx[0], dict(st["cal_states"][0])))
                use_g = st["gptaq"] > 0.0

                for g in groups:
                    acc = vtq.DualCovAccumulator(need_cross=use_g)
                    buf_q, buf_r = [None], [None]

                    def _gq(m, a):
                        buf_q[0] = a[0]
                        raise hqm._Stop

                    def _gr(m, a):
                        buf_r[0] = a[0]
                        raise hqm._Stop

                    hq_h = mods_q[g[0]].register_forward_pre_hook(_gq)
                    hr_h = mods_r[g[0]].register_forward_pre_hook(_gr) if use_g else None
                    for i in range(len(st["cal"])):
                        try:
                            hqm.call_layer(lq, st["cal"][i], cal_ctx[i], dict(st["cal_states"][i]))
                        except hqm._Stop:
                            pass
                        if use_g:
                            try:
                                hqm.call_layer(layer_ref, cal_ref[i], cal_ctx[i], dict(cal_ref_states[i]))
                            except hqm._Stop:
                                pass
                        acc.add(buf_q[0], buf_r[0])
                        buf_q[0] = buf_r[0] = None
                    hq_h.remove()
                    if hr_h is not None:
                        hr_h.remove()

                    d_base = 0.20 * (1.5 if g[0].endswith("down_proj") else 1.0)
                    H, B = acc.get(gptaq_damp=d_base)
                    del acc
                    for sub_n in g:
                        mod = mods_q[sub_n]
                        W = mod.weight.data.float()
                        r_sp = st["asgn"][(l_idx, sub_n)]
                        rw = sqm._row_w(sub_n, dev)
                        sd = sqm._seed(l_idx, sub_n)
                        dm = 1.5 if sub_n.endswith("down_proj") else 1.0
                        W_tgt = (W + st["gptaq"] * (W @ B.T)) if (use_g and B is not None) else None
                        _, w_hat, _ = vtq.quantize_matrix_vtq(W, H, r_sp, seed=sd, damp=0.20 * dm, row_weights=rw, W_target=W_tgt)
                        mod.weight.data.copy_(w_hat.to(mod.weight.dtype))
                    del H, B

                done = {n for g in groups for n in g}
                b = p = qb = qp = 0
                for sub_n, mod in lq.named_modules():
                    if isinstance(mod, nn.Linear) and not sub_n.endswith(("mlp.gate", "router.proj", "shared_expert_gate")):
                        M_r, K_in = mod.weight.shape
                        if sub_n in done:
                            bi = sum((r1 - r0) * K_in * lv2.LIQK_V2_SPECS[sp].bpw for r0, r1, sp in st["asgn"][(l_idx, sub_n)]) + 16 * (M_r + K_in)
                            qb += bi
                            qp += M_r * K_in
                        else:
                            bi = 16 * M_r * K_in
                        b += bi
                        p += M_r * K_in

                for i in range(len(st["cal"])):
                    st["cal"][i] = hqm.call_layer(lq, st["cal"][i], cal_ctx[i], st["cal_states"][i])
                torch.cuda.synchronize()
                st["quant_sec"] += time.perf_counter() - t1
                st["bits"] += b
                st["params"] += p
                st["q_bits"] += qb
                st["q_params"] += qp
                qlayers[m_name] = lq

            for i in range(len(cal_ref)):
                cal_ref[i] = hqm.call_layer(layer_ref, cal_ref[i], cal_ctx[i], cal_ref_states[i])
            del cal_ctx

            for ds, items in traces.items():
                for i, (ids, _, _) in enumerate(items):
                    _, *ctx = tmod.capture(ids.to(dev), l_idx)
                    ctx = tuple(ctx)
                    for streams, states, lay in [(ref["streams"], ref["states"], layer_ref)] + [
                        (st["streams"], st["states"], qlayers[nm]) for nm, st in active.items()
                    ]:
                        streams[ds][i] = hqm.call_layer(lay, streams[ds][i].to(dev), ctx, states[ds][i]).cpu()
            del qlayers
            tmod.release_layer(l_idx)
            torch.cuda.empty_cache()
            if (l_idx + 1) % 6 == 0 or l_idx == n_layers - 1:
                print(f"    [{model_dir.name}] completed layer {l_idx + 1}/{n_layers}", flush=True)

        for m_name, st in active.items():
            entry = {
                "kind": "dp",
                "quant_sec": st["quant_sec"],
                "profile_sec": prof_vtq_sec,
                "linear_bpw": st["bits"] / max(st["params"], 1),
                "q_bpw": st["q_bits"] / max(st["q_params"], 1),
                "shares_pct": _shares_pct(tmod.layers, st["asgn"]),
            }
            for ds, items in traces.items():
                ref_f = [h[0, pos] for h, (_, pos, _) in zip(ref["streams"][ds], items)]
                q_f = [h[0, pos] for h, (_, pos, _) in zip(st["streams"][ds], items)]
                entry[ds] = sqm.trace_metrics(tmod.model, ref_f, q_f, [tg for _, _, tg in items], dev, logits_fn=tmod.logits)
            results["methods"][m_name] = entry
            print(
                f"  [{m_name:24s}] q_bpw={entry['q_bpw']:.4f} t={entry['quant_sec']:.1f}s | "
                f"nem={entry['nemotron']['mean_kld']:.5f} mul={entry['multilingual']['mean_kld']:.5f}",
                flush=True,
            )

    comps = {}
    for t in targets:
        t_lbl = _fmt_tgt_label(t)
        cand = f"dp:VTQ:{t_lbl}"
        if cand not in results["methods"]:
            continue
        ref_candidates = [
            f"dp:VTQ2:{t_lbl}",
            f"dp:VTQ2:{float(t):g}",
            f"dp:TCQPQ:{t_lbl}",
            f"dp:TCQPQ:{float(t):g}",
            f"exl3q:{t_lbl}",
            f"exl3q:{float(t):g}",
        ]
        if extra_ref_pairs and t_lbl in extra_ref_pairs:
            ref_candidates.extend(extra_ref_pairs[t_lbl])
        seen_r = set()
        for ref_k in ref_candidates:
            if ref_k in seen_r:
                continue
            seen_r.add(ref_k)
            if ref_k in results["methods"] and cand != ref_k:
                comps[f"{cand} vs {ref_k}"] = _pair_boot(results["methods"][cand], results["methods"][ref_k])
                d = comps[f"{cand} vs {ref_k}"]
                print(
                    f"  {cand} vs {ref_k}: nem {d['nemotron']['mean_diff']:+.5f} ({d['nemotron']['rel_pct']:+.1f}%, p={d['nemotron']['p_a_better']:.4f}) | "
                    f"mul {d['multilingual']['mean_diff']:+.5f} ({d['multilingual']['rel_pct']:+.1f}%, p={d['multilingual']['p_a_better']:.4f})",
                    flush=True,
                )
    results["comparisons"] = comps
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")


@torch.no_grad()
def run_gemma4_ple_vtq_rings(model_dir: Path, data_dir: Path, out_json: Path):
    print("\n=== Evaluating `VtqRing` on Gemma 4 E4B PLE (`embed_tokens_per_layer`) ===", flush=True)
    dev = torch.device("cuda")
    vtq.load_and_register_vtq()
    res = json.loads(out_json.read_text(encoding="utf-8")) if out_json.exists() else {"methods": {}}
    for bp in (ROOT / "results" / "vtq2_gemma4_ple.json", ROOT / "results" / "g4_ple_gemma4.json"):
        if bp.exists():
            raw_ple = json.loads(bp.read_text(encoding="utf-8"))
            for k, val in raw_ple.get("methods", raw_ple).items():
                res["methods"].setdefault(k, val)

    def _move_lay(m: nn.Module, target_dev):
        m.to(target_dev)
        for sub in m.modules():
            if hasattr(sub, "embed_scale") and isinstance(sub.embed_scale, torch.Tensor):
                sub.embed_scale = sub.embed_scale.to(target_dev)

    tmod = hqm.TextModel(model_dir, dev)
    n_layers = len(tmod.layers)
    gpu_resident = n_layers if torch.cuda.get_device_properties(dev).total_memory >= (28 << 30) else max(0, n_layers - 14)
    for l_i in range(n_layers):
        lay = tmod.load_layer(l_i)
        if l_i >= gpu_resident:
            _move_lay(lay, "cpu")
            lay.register_forward_pre_hook(lambda m, a: (_move_lay(m, dev), None)[1])
            lay.register_forward_hook(lambda m, a, o: (_move_lay(m, "cpu"), None)[1])
    tmod.tm.layers = nn.ModuleList(tmod.layers)
    if getattr(tmod, "store", None) is not None:
        tmod.store._handles.clear()
        tmod.store = None
    gc.collect()
    torch.cuda.empty_cache()

    ple_mod = tmod.tm.embed_tokens_per_layer
    W_orig = ple_mod.weight.data

    traces = {}
    for ds in ("nemotron", "multilingual"):
        items = []
        for sq_item in load_eval_trace(data_dir / f"{ds}_trace.json"):
            pos, tg = get_scored_positions_and_targets(sq_item)
            if pos:
                items.append((torch.tensor([sq_item["token_ids"][:8192]]), pos, tg))
        traces[ds] = items

    variants = [
        ("VTQ_200:rot", "VTQ_200", 256),
        ("VTQ_250:rot", "VTQ_250", 256),
        ("VTQ_300:rot", "VTQ_300", 256),
        ("VTQ_400:rot", "VTQ_400", 256),
    ]
    todo = [v for v in variants if v[0] not in res["methods"]]
    if todo:
        w_sq_sum = sum(
            float(W_orig[r : r + 16384].float().square().sum().item())
            for r in range(0, W_orig.shape[0], 16384)
        )
        ref_h = {}
        ple_mod.weight.data = W_orig
        for ds, items in traces.items():
            ref_h[ds] = [tmod.tm(input_ids=ids.to(dev), use_cache=False).last_hidden_state[0, pos].cpu() for ids, pos, _ in items]
        torch.cuda.empty_cache()

        for label, sp_nm, rlen in todo:
            t0 = time.perf_counter()
            wq, bpv = vtq.quantize_rings_vtq(W_orig, sp_nm, ring_len=rlen, rot=True)
            dt = time.perf_counter() - t0
            err_sq_sum = sum(
                float((W_orig[r : r + 16384].float() - wq[r : r + 16384].float()).square().sum().item())
                for r in range(0, W_orig.shape[0], 16384)
            )
            rel_mse = err_sq_sum / max(w_sq_sum, 1e-12)
            torch.cuda.empty_cache()
            print(f"  Quantized PLE `{label}` ({bpv:.4f} bpv) in {dt:.1f}s | rel_mse={rel_mse:.4e}", flush=True)

            ple_mod.weight.data = wq
            entry = {
                "bits_per_value": bpv,
                "size_gb": W_orig.numel() * bpv / 8.0 / (2**30),
                "quant_sec": dt,
                "rel_mse": rel_mse,
            }
            for ds, items in traces.items():
                q_f = [tmod.tm(input_ids=ids.to(dev), use_cache=False).last_hidden_state[0, pos].cpu() for ids, pos, _ in items]
                entry[ds] = sqm.trace_metrics(tmod.model, ref_h[ds], q_f, [tg for _, _, tg in items], dev, logits_fn=tmod.logits)
                del q_f
            res["methods"][label] = entry
            ple_mod.weight.data = W_orig
            del wq
            gc.collect()
            torch.cuda.empty_cache()
            out_json.write_text(json.dumps(res, indent=2), encoding="utf-8")
            print(
                f"  [PLE {label:14s}] bpv={bpv:.4f} | nem={entry['nemotron']['mean_kld']:.6f} mul={entry['multilingual']['mean_kld']:.6f}",
                flush=True,
            )
        del ref_h

    comps = {}
    ple_pairs = [
        ("VTQ_200:rot", "VTQ200:rot"),
        ("VTQ_200:rot", "TCQ2X"),
        ("VTQ_250:rot", "VTQ250:rot"),
        ("VTQ_300:rot", "VTQ300:rot"),
        ("VTQ_300:rot", "TCQ3X:rot"),
        ("VTQ_400:rot", "VTQ400:rot"),
        ("VTQ_400:rot", "TCQ4X:rot"),
    ]
    for a_k, b_k in ple_pairs:
        if a_k in res["methods"] and b_k in res["methods"] and "per_seq_kld_sums" in res["methods"][b_k].get("nemotron", {}):
            comps[f"{a_k} vs {b_k}"] = _pair_boot(res["methods"][a_k], res["methods"][b_k])
            d = comps[f"{a_k} vs {b_k}"]
            print(
                f"  {a_k} vs {b_k}: nem {d['nemotron']['mean_diff']:+.6f} ({d['nemotron']['rel_pct']:+.1f}%, p={d['nemotron']['p_a_better']:.4f}) | "
                f"mul {d['multilingual']['mean_diff']:+.6f} ({d['multilingual']['rel_pct']:+.1f}%, p={d['multilingual']['p_a_better']:.4f})",
                flush=True,
            )
    res["comparisons"] = comps
    ple_mod.weight.data = W_orig
    out_json.write_text(json.dumps(res, indent=2), encoding="utf-8")
    del tmod
    gc.collect()
    torch.cuda.empty_cache()


# =================================================================================================
# 5. Suite `moe_30b`: Streaming `Qwen3-30B-A3B-Instruct-2507` (`48` MoE layers x `128` experts)
# =================================================================================================

def run_suite_moe_30b():
    vtq.install_vtq_hooks()
    model_dir = ROOT / "models" / "Qwen3-30B-A3B-Instruct-2507"
    data_dir = ROOT / "data" / "qwen3"
    fisher_path = data_dir / "fisher_30b.pt"
    out_json = ROOT / "results" / "vtq_qwen3_30b.json"

    if not fisher_path.exists():
        print(f"\n=== Computing layer-offloaded Fisher for {model_dir.name} -> {fisher_path} ===", flush=True)
        subprocess.run(
            [
                sys.executable, "-u", "-m", "src.fisher_offload",
                "--model_dir", str(model_dir),
                "--data_dir", str(data_dir),
                "--out", str(fisher_path),
            ],
            check=True,
        )

    existing = json.loads(out_json.read_text(encoding="utf-8")) if out_json.exists() else {"methods": {}, "comparisons": {}}
    todo_m = [f"dp:VTQ:{t}" for t in ("3.0", "3.5", "4.5") if f"dp:VTQ:{t}" not in existing.get("methods", {})]
    if todo_m:
        tmp_out = ROOT / "results" / "_tmp_vtq_30b.json"
        cmd = [
            sys.executable,
            "-u",
            "-c",
            (
                "import os, sys; os.environ['VTQ_PROFILE']='1'; "
                "import src.vtq as vtq; vtq.install_vtq_hooks(); "
                "import src.streaming_quant as sq; "
                "_orig_h = sq.TensorStore._handle; "
                "sq.TensorStore._handle = lambda self, n: (self._handles.clear() if len(self._handles) > 2 and self.files[n] not in self._handles else None, _orig_h(self, n))[1]; "
                f"sys.argv = ['streaming_quant', '--model_dir', {str(model_dir)!r}, '--data_dir', {str(data_dir)!r}, "
                f"'--fisher', {str(fisher_path)!r}, '--methods', {', '.join(repr(x) for x in todo_m)}, "
                f"'--out', {str(tmp_out)!r}]; sq.main()"
            ),
        ]
        subprocess.run(cmd, check=True)
        res = json.loads(tmp_out.read_text(encoding="utf-8"))
        tmp_out.unlink(missing_ok=True)
        for k, val in existing.get("methods", {}).items():
            res["methods"].setdefault(k, val)
    else:
        res = existing

    for bp in (ROOT / "results" / "vtq2_qwen3_30b.json", ROOT / "results" / "g4_qwen3_30b_dp.json"):
        if bp.exists():
            for k, val in json.loads(bp.read_text(encoding="utf-8")).get("methods", {}).items():
                res["methods"].setdefault(k, val)

    comps = {}
    for t_str, exl_k in [("3.0", "exl3:exl3_out/qwen3_30b_3.0"), ("3.5", None), ("4.5", "exl3:exl3_out/qwen3_30b_4.5")]:
        cand = f"dp:VTQ:{t_str}"
        if cand in res["methods"]:
            for rk in [f"dp:VTQ2:{t_str}", f"dp:TCQPQ:{t_str}", exl_k]:
                if rk and rk in res["methods"]:
                    comps[f"{cand} vs {rk}"] = _pair_boot(res["methods"][cand], res["methods"][rk])
                    d = comps[f"{cand} vs {rk}"]
                    print(
                        f"  {cand} vs {rk}: nem {d['nemotron']['mean_diff']:+.5f} ({d['nemotron']['rel_pct']:+.1f}%, p={d['nemotron']['p_a_better']:.4f}) | "
                        f"mul {d['multilingual']['mean_diff']:+.5f} ({d['multilingual']['rel_pct']:+.1f}%, p={d['multilingual']['p_a_better']:.4f})",
                        flush=True,
                    )
    res["comparisons"] = comps
    out_json.write_text(json.dumps(res, indent=2), encoding="utf-8")


# =================================================================================================
# 6. Suite `speed`: Controlled 3-Generation (`VTQ v1`, `VTQ v2`, `VTQ`) CPU AVX2 + GPU Benchmark
# =================================================================================================

def run_suite_speed(out_name: str = "vtq_decode_speed.json"):
    out_j = ROOT / "results" / out_name
    maj, minr = torch.cuda.get_device_capability() if torch.cuda.is_available() else (12, 0)
    arch = f"sm_{maj}{minr}"
    bins = [
        "/tmp/_v1_cpu_bench",
        "/tmp/_v2_cpu_bench",
        "/tmp/_v3_cpu_bench",
        "/tmp/_v1_gpu_bench",
        "/tmp/_v2_gpu_bench",
        "/tmp/_v3_gpu_bench",
    ]
    try:
        subprocess.run(["gcc", "-O3", "-mavx2", "-mfma", "-mno-avx512f", str(ROOT / "src/cpu/tcq_avx2_bench.c"), "-lm", "-o", bins[0]], check=True)
        subprocess.run(["gcc", "-O3", "-mavx2", "-mfma", "-mno-avx512f", str(ROOT / "src/cpu/vtq2_avx2_bench.c"), "-lm", "-o", bins[1]], check=True)
        subprocess.run(["gcc", "-O3", "-mavx2", "-mfma", "-mno-avx512f", str(ROOT / "src/cpu/vtq_avx2_bench.c"), "-lm", "-o", bins[2]], check=True)

        subprocess.run(["nvcc", "-O3", "-std=c++17", f"-arch={arch}", str(ROOT / "src/cuda/ltq_matvec_bench.cu"), "-o", bins[3]], check=True)
        r_v2 = subprocess.run(["nvcc", "-O3", "-std=c++17", f"-arch={arch}", str(ROOT / "src/cuda/vtq2_matvec_bench.cu"), "-o", bins[4]], capture_output=True, text=True)
        if r_v2.returncode != 0:
            subprocess.run(
                ["nvcc", "-O3", "-std=c++17", "-arch=compute_89", str(ROOT / "src/cuda/vtq2_matvec_bench.cu"), "-o", bins[4]],
                check=True,
            )
        subprocess.run(["nvcc", "-O3", "-std=c++17", f"-arch={arch}", str(ROOT / "src/cuda/vtq_matvec_bench.cu"), "-o", bins[5]], check=True)

        res = {
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
            "gpu_arch": arch,
        }
        for key, bpath in [
            ("cpu_avx2_vtq1_raw", bins[0]),
            ("cpu_avx2_vtq2_raw", bins[1]),
            ("cpu_avx2_vtq_raw", bins[2]),
            ("gpu_vtq1_raw", bins[3]),
            ("gpu_vtq2_raw", bins[4]),
            ("gpu_vtq_raw", bins[5]),
        ]:
            out_txt = subprocess.run([bpath], capture_output=True, text=True, check=True).stdout
            print(out_txt, flush=True)
            res[key] = out_txt
        out_j.write_text(json.dumps(res, indent=2), encoding="utf-8")
        print(f"=== Saved 3-generation speed results to {out_j} ===", flush=True)
    finally:
        for bpath in bins:
            Path(bpath).unlink(missing_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--suite",
        choices=["3b", "1b", "3b_1b", "qwen3_8b", "qwen35", "gemma4", "hybrid", "llama8b", "qwen27", "moe_30b", "speed"],
        required=True,
    )
    ap.add_argument("--seed-json", type=Path, default=None)
    ap.add_argument("--speed-out", type=str, default="vtq_decode_speed.json")
    args = ap.parse_args()

    if args.suite == "speed":
        run_suite_speed(out_name=args.speed_out)
        return
    if args.suite == "qwen27":
        f27 = ROOT / "data" / "qwen38" / "fisher.pt"
        compute_hybrid_fisher_offload(ROOT / "models" / "Qwen3.8-27B", ROOT / "data" / "qwen38", f27)
        run_hybrid_vtq_eval(
            ROOT / "models" / "Qwen3.8-27B",
            ROOT / "data" / "qwen38",
            f27,
            ("2.9409", "3.0043", "3.4794", "3.5038"),
            ROOT / "results" / "vtq_qwen38_27b.json",
            cached_baseline_jsons=(ROOT / "results" / "vtq2_qwen38_27b.json", ROOT / "results" / "q27_compare.json"),
            extra_ref_pairs={
                "2.9409": ["gguf:/content/m/gguf/Qwen3.8-27B-GSQ-RCO-IQ3_XXS.gguf"],
                "3.0043": ["exl3:/content/m/exl3_SC_3.00bpw_H4"],
                "3.4794": ["gguf:/content/m/gguf/Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf"],
                "3.5038": ["exl3:/content/m/exl3_3.50bpw"],
            },
        )
        return
    if args.suite in ("3b", "3b_1b"):
        run_dense_sweep_model("3b", "Llama-3.2-3B-Instruct", polish_steps=80, polish_lr=4e-4, seed_json=args.seed_json)
    if args.suite in ("1b", "3b_1b"):
        run_dense_sweep_model("1b", "Llama-3.2-1B-Instruct", polish_steps=100, polish_lr=5e-4)
    if args.suite == "qwen3_8b":
        run_suite_qwen3_8b()
    if args.suite in ("qwen35", "hybrid"):
        run_hybrid_vtq_eval(
            ROOT / "models" / "Qwen3.5-4B",
            ROOT / "data" / "qwen35",
            ROOT / "data" / "qwen35" / "fisher.pt",
            (2.5, 3.0, 3.5, 4.0, 4.5),
            ROOT / "results" / "vtq_qwen35_4b.json",
            cached_baseline_jsons=(ROOT / "results" / "vtq2_qwen35_4b.json", ROOT / "results" / "g4_hybrid_qwen35.json"),
        )
    if args.suite in ("gemma4", "hybrid"):
        run_hybrid_vtq_eval(
            ROOT / "models" / "gemma-4-E4B-it",
            ROOT / "data" / "gemma4",
            ROOT / "data" / "gemma4" / "fisher.pt",
            (2.5, 3.0, 3.5, 4.0, 4.5),
            ROOT / "results" / "vtq_gemma4_e4b.json",
            cached_baseline_jsons=(ROOT / "results" / "vtq2_gemma4_e4b.json", ROOT / "results" / "g4_hybrid_gemma4.json"),
        )
        run_gemma4_ple_vtq_rings(
            ROOT / "models" / "gemma-4-E4B-it",
            ROOT / "data" / "gemma4",
            ROOT / "results" / "vtq_gemma4_ple.json",
        )
    if args.suite == "llama8b":
        run_suite_llama8b()
    if args.suite == "moe_30b":
        run_suite_moe_30b()


if __name__ == "__main__":
    main()
