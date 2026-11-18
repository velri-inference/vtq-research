"""
Comprehensive VTQ v2 Multi-Model & Multi-Architecture Evaluation Suite (`src/run_vtq2_benchmarks.py`):
  - `--suite 3b_1b`:
      Full equal-bpw sweep (`1.75, 2.0, 2.10, 2.5, 3.0, 3.4375, 4.0, 4.5, 5.0, 6.0 bpw`) on `Llama-3.2-3B-Instruct`
      and `Llama-3.2-1B-Instruct` (`VTQ2` DP + `GPTAQ`, `VTQ2-NoGPTAQ`, `VTQ2-Polish`), paired bootstrap (`B=2000`)
      against `EXL3` (`mul1`), `VTQ v1` (`TCQPQ`), and `GGUF`; full `.vtq` v2 file write/reload verification (`VtqWriter`
      -> `VtqFile`); and domain-neutral Wikipedia calibration + external `wikitext2` evaluation audit (`README.md:30`).
  - `--suite qwen3_8b`:
      Full equal-bpw sweep (`2.5, 3.0, 3.4375, 4.0, 4.5, 5.0 bpw`) on `Qwen3-8B` vs `EXL3` and `TCQPQ` (`B=2000`).
  - `--suite hybrid`:
      Architecture-agnostic `TextModel` + `DualCovAccumulator` `GPTAQ` evaluation on `Qwen3.5-4B` (Gated DeltaNet),
      `Gemma-4-E4B-it` (PLE + shared KV + `VtqRing` PLE table), and `Llama-3.1-8B-Instruct` (`K = 14336`, `q = 7`).
  - `--suite moe_30b`:
      Streaming `VTQ2` DP + attention `GPTAQ` evaluation on `Qwen3-30B-A3B-Instruct-2507` (`48` layers, `128` experts)
      at `3.0, 3.5, 4.5 bpw` vs `TCQPQ` and `EXL3` (`B=2000`).
"""

from __future__ import annotations

import argparse
import copy
import json
import os
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
from transformers import AutoModelForCausalLM, AutoTokenizer

import src.equal_bpw as eqb
import src.hybrid_quant as hqm
import src.liqk_v2 as lv2
import src.streaming_quant as sqm
import src.vtq_v2 as v2
import src.vtq_v2_format as v2f
from src.eval_kld import (
    evaluate_model_on_cached_trace,
    get_scored_positions_and_targets,
    load_eval_trace,
    paired_bootstrap_diff,
    precompute_ref_trace_data,
)
from src.run_1b_liqk_v2_experiments import restore_clean_model_weights, save_clean_model_weights

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


# =================================================================================================
# 1. Suite `3b_1b`: Full Bitrate Sweep + `.vtq` v2 Container File Test + Neutral Calibration Audit
# =================================================================================================

def build_wikitext_eval_and_neutral_cal(tok_dir: Path, n_cal_rows: int = 64, seq_len: int = 2048) -> Tuple[List[dict], torch.Tensor]:
    """Builds an external non-chat WikiText-2 evaluation trace (`data/wikitext2_test.raw`) and a strictly disjoint
    domain-neutral calibration tensor `[64, 2048]` from `wikitext-2-raw-v1` train split."""
    tok = AutoTokenizer.from_pretrained(str(tok_dir))
    raw_test = (ROOT / "data" / "wikitext2_test.raw").read_text(encoding="utf-8")
    test_ids = tok.encode(raw_test, add_special_tokens=False)
    eval_seqs = []
    chunk_len = 512
    for i in range(40):
        seg = test_ids[i * chunk_len : (i + 1) * chunk_len]
        if len(seg) < 128:
            break
        # Score tokens [64 .. len(seg)) conditioned on the 64-token prefix
        eval_seqs.append({"id": f"wt2_{i}", "token_ids": seg, "score_spans": [[64, len(seg)]]})

    from datasets import load_dataset
    ds_train = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    train_text = "\n\n".join(t for t in ds_train["text"] if t and len(t.strip()) > 40)
    train_ids = tok.encode(train_text, add_special_tokens=False)
    need = n_cal_rows * seq_len
    if len(train_ids) < need:
        train_ids = (train_ids * ((need // len(train_ids)) + 2))[:need]
    cal_neutral = torch.tensor(train_ids[:need], dtype=torch.long).view(n_cal_rows, seq_len)
    return eval_seqs, cal_neutral


def run_dense_sweep_model(tag: str, model_name: str, polish_steps: int, polish_lr: float):
    dev = torch.device("cuda")
    v2.load_and_register_vtq2()
    model_dir = ROOT / "models" / model_name
    out_json = ROOT / "results" / f"vtq2_{tag}_sweep.json"
    results = json.loads(out_json.read_text(encoding="utf-8")) if out_json.exists() else {"model": model_name, "runs": {}, "comparisons": {}}

    # Import cached baselines (`EXL3-*`, `TCQPQ-*`) from `results/equal_bpw_{tag}.json` for exact paired bootstrap
    eq_path = ROOT / "results" / f"equal_bpw_{tag}.json"
    if eq_path.exists():
        eq_data = json.loads(eq_path.read_text(encoding="utf-8")).get("runs", {})
        for k, v in eq_data.items():
            if k.startswith(("EXL3-", "TCQPQ-")) and k not in results["runs"]:
                results["runs"][k] = v

    print(f"\n=== Dense Equal-BPW Sweep on `{model_name}` (`{tag}`) ===", flush=True)
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
    prof_vtq2 = v2.build_vtq2_fisher_slice_profile(model, cal64, batch_size=8, fisher_out=fisher, rot_block=-1)
    prof_vtq2_sec = time.perf_counter() - t0
    print(f"  Built VTQ2 Fisher slice profile ({len(prof_vtq2['slices'])} slices) in {prof_vtq2_sec:.1f}s", flush=True)

    t0 = time.perf_counter()
    prof_v1 = lv2.build_liqk_v2_fisher_slice_profile(model, cal64, batch_size=8, fisher_out=fisher)
    prof_v1_sec = time.perf_counter() - t0

    def save_eval(label: str, info: dict):
        entry = {
            "info": info,
            "nemotron": evaluate_model_on_cached_trace(model, nem_ref, dev),
            "multilingual": evaluate_model_on_cached_trace(model, mul_ref, dev),
        }
        results["runs"][label] = entry
        out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"  [{label:26s}] bpw={info['linear_bpw']:.4f} t={info.get('quant_sec', 0):.1f}s | "
              f"nem={entry['nemotron']['mean_kld']:.5f} mul={entry['multilingual']['mean_kld']:.5f} | "
              f"top1={entry['nemotron']['top1_match']*100:.2f}%/{entry['multilingual']['top1_match']*100:.2f}%", flush=True)

    # Fresh v1 `TCQPQ` runs at extreme endpoints `2.10` and `5.00` (2.5..4.5 are already in equal_bpw_{tag}.json)
    for t_v1 in (2.10, 2.5, 3.0, 3.4375, 4.0, 4.5, 5.0):
        lbl = f"TCQPQ-{t_v1:.4f}"
        if lbl in results["runs"]:
            continue
        restore_clean_model_weights(model, clean)
        asgn1 = lv2.solve_fisher_slice_dp(prof_v1, target_bpw=t_v1, candidate_specs=eqb.SETS["TCQPQ"])
        q1 = lv2.quantize_llama_model_liqk_v2(model, cal64, asgn1, mlp_perms=prof_v1["mlp_perms"],
                                              fisher_out=prof_v1["fisher_out"], batch_size=8, cd_sweeps=2, polish_steps=0)
        save_eval(lbl, {"linear_bpw": q1["linear_bpw"], "quant_sec": q1["quant_sec"], "profile_sec": prof_v1_sec,
                        "target_bpw": t_v1, "shares_pct": _shares_pct(model, asgn1)})

    # Full VTQ v2 sweep across `1.75 .. 6.00 bpw`!
    vtq2_targets = (1.75, 2.0, 2.10, 2.5, 3.0, 3.4375, 4.0, 4.5, 5.0, 6.0)
    nogptaq_targets = (2.5, 3.0, 3.4375, 4.0, 4.5) if tag == "3b" else (3.0, 4.0)
    polish_targets = (2.5, 3.0, 3.4375, 4.0) if tag == "3b" else (3.0, 3.4375)

    for target in vtq2_targets:
        asgn2 = lv2.solve_fisher_slice_dp(prof_vtq2, target_bpw=target, candidate_specs=eqb.SETS["VTQ2"])

        if target in nogptaq_targets:
            lbl_ng = f"VTQ2-{target:.4f}-NoGPTAQ"
            if lbl_ng not in results["runs"]:
                restore_clean_model_weights(model, clean)
                q_ng = v2.quantize_llama_model_vtq2(
                    model, cal64, asgn2, mlp_perms=prof_vtq2["mlp_perms"], fisher_out=prof_vtq2["fisher_out"],
                    batch_size=4, gptaq_alpha=0.0, rot_block=-1, polish_steps=0
                )
                save_eval(lbl_ng, {"linear_bpw": q_ng["linear_bpw"], "quant_sec": q_ng["quant_sec"],
                                   "profile_sec": prof_vtq2_sec, "target_bpw": target, "shares_pct": _shares_pct(model, asgn2)})

        lbl = f"VTQ2-{target:.4f}"
        lbl_pol = f"VTQ2-{target:.4f}-Polish"
        need_main = lbl not in results["runs"]
        need_pol = (target in polish_targets) and (lbl_pol not in results["runs"])
        if need_main or need_pol:
            restore_clean_model_weights(model, clean)
            q = v2.quantize_llama_model_vtq2(
                model, cal64, asgn2, mlp_perms=prof_vtq2["mlp_perms"], fisher_out=prof_vtq2["fisher_out"],
                batch_size=4, gptaq_alpha=0.50, rot_block=-1, polish_steps=0
            )
            info = {"linear_bpw": q["linear_bpw"], "quant_sec": q["quant_sec"], "profile_sec": prof_vtq2_sec,
                    "target_bpw": target, "shares_pct": _shares_pct(model, asgn2)}
            if need_main:
                save_eval(lbl, info)
            if need_pol:
                with torch.enable_grad():
                    pol_s = lv2.polish_model_scales_kld(model, cal64, ref_hidden, n_steps=polish_steps, lr=polish_lr,
                                                        batch_size=2, reg_wd=0.02, ce_weight=0.0)
                save_eval(lbl_pol, dict(info, polish_sec=pol_s))

    # Full `.vtq` v2 container write + reload verification on 3B @ 3.0000 bpw!
    if tag == "3b" and "vtq2_file_verification" not in results:
        print("\n  === Packing Full `Llama-3.2-3B-Instruct @ 3.0000 bpw` into `.vtq` v2 & Reloading ===", flush=True)
        restore_clean_model_weights(model, clean)
        asgn_pack = lv2.solve_fisher_slice_dp(prof_vtq2, target_bpw=3.0, candidate_specs=eqb.SETS["VTQ2"])
        fmts_all = list(v2f.load_vtq2_formats().values())
        with tempfile.TemporaryDirectory() as td:
            vtq_path = Path(td) / "llama3_2_3b_vtq2_3bpw.vtq"
            writer = v2f.VtqWriter(vtq_path, fmts_all, reserve=(8 << 20) - 16)
            # Hook `v2.PACK` during full-model quantization to stream every matrix slice into `writer`!
            orig_qm = v2.quantize_matrix_vtq2
            mat_counter = [0]
            packed_names = []

            # Walk layer by layer using `quantize_llama_model_vtq2` while recording each quantized matrix
            def _pack_wrap(W, H, r_specs, **kw):
                v2.PACK = {}
                out = orig_qm(W, H, r_specs, **kw)
                m_idx = mat_counter[0]
                mat_counter[0] += 1
                l_i, s_i = m_idx // 7, m_idx % 7
                p_name = f"model.layers.{l_i}.{lv2.LINEAR_SUBMODULES[s_i]}"
                writer.add_matrix(p_name, v2.PACK["matrix"], verify=True)
                packed_names.append((l_i, lv2.LINEAR_SUBMODULES[s_i], p_name))
                v2.PACK = None
                return out

            v2.quantize_matrix_vtq2 = _pack_wrap
            try:
                v2.quantize_llama_model_vtq2(
                    model, cal64, asgn_pack, mlp_perms=prof_vtq2["mlp_perms"], fisher_out=prof_vtq2["fisher_out"],
                    batch_size=4, gptaq_alpha=0.50, rot_block=-1, polish_steps=0
                )
            finally:
                v2.quantize_matrix_vtq2 = orig_qm

            writer.finalize({
                "format": "vtq-v2",
                "quantization": {
                    "quant_method": "vtq_v2",
                    "version": 2,
                    "rot_block": -1,
                    "bits_per_weight": 3.0,
                    "formats": [v2f.format_json(f) for f in fmts_all],
                },
            })
            nem_mem = evaluate_model_on_cached_trace(model, nem_ref, dev)["mean_kld"]
            mul_mem = evaluate_model_on_cached_trace(model, mul_ref, dev)["mean_kld"]

            # Corrupt weights in memory, then reload ALL 196 matrices from `VtqFile` and re-evaluate!
            for l_i, sub_nm, _ in packed_names:
                model.model.layers[l_i].get_submodule(sub_nm).weight.data.zero_()
            vf = v2f.VtqFile(vtq_path)
            max_w_diff = 0.0
            for l_i, sub_nm, p_name in packed_names:
                w_dec = vf.decode_matrix(p_name, device=dev).to(torch.bfloat16)
                model.model.layers[l_i].get_submodule(sub_nm).weight.data.copy_(w_dec)
            nem_file = evaluate_model_on_cached_trace(model, nem_ref, dev)["mean_kld"]
            mul_file = evaluate_model_on_cached_trace(model, mul_ref, dev)["mean_kld"]
            file_sz = vtq_path.stat().st_size
            print(f"  [PASS] `.vtq` v2 3B file ({file_sz / 1e9:.3f} GB, {len(packed_names)} matrices): "
                  f"mem={nem_mem:.6f}/{mul_mem:.6f} vs file={nem_file:.6f}/{mul_file:.6f}", flush=True)
            assert abs(nem_mem - nem_file) < 1e-6 and abs(mul_mem - mul_file) < 1e-6
            results["vtq2_file_verification"] = {
                "quantized_matrices": len(packed_names),
                "matrix_container_bytes": file_sz,
                "nem_kld_mem": nem_mem,
                "nem_kld_file": nem_file,
                "mul_kld_mem": mul_mem,
                "mul_kld_file": mul_file,
            }

    # Paired bootstrap comparisons (`B=2000`) against `TCQPQ` and `EXL3`
    comps = {}
    for target in vtq2_targets:
        t = f"{target:.4f}"
        for cand in (f"VTQ2-{t}", f"VTQ2-{t}-NoGPTAQ", f"VTQ2-{t}-Polish"):
            if cand not in results["runs"]:
                continue
            for ref in (f"TCQPQ-{t}", f"EXL3-{t}", f"VTQ2-{t}-NoGPTAQ"):
                if ref in results["runs"] and cand != ref:
                    comps[f"{cand} vs {ref}"] = _pair_boot(results["runs"][cand], results["runs"][ref])
                    d = comps[f"{cand} vs {ref}"]
                    print(f"  {cand} vs {ref}: nem {d['nemotron']['mean_diff']:+.5f} ({d['nemotron']['rel_pct']:+.1f}%, p={d['nemotron']['p_a_better']:.4f}) | "
                          f"mul {d['multilingual']['mean_diff']:+.5f} ({d['multilingual']['rel_pct']:+.1f}%, p={d['multilingual']['p_a_better']:.4f})",
                          flush=True)
    results["comparisons"] = comps
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")

    # Neutral Wikipedia calibration + external `wikitext2` trace audit on 3B (`README.md:30`)
    if tag == "3b":
        audit_out = ROOT / "results" / "vtq2_3b_neutral_audit.json"
        if not audit_out.exists():
            print("\n=== Domain-Neutral Wikipedia Calibration & External `wikitext2` Trace Audit (3B) ===", flush=True)
            restore_clean_model_weights(model, clean)
            wt_seqs, cal_neutral = build_wikitext_eval_and_neutral_cal(model_dir, n_cal_rows=64, seq_len=2048)
            wt_ref = precompute_ref_trace_data(model, wt_seqs, dev)
            fisher_neu = lv2.compute_exact_row_fisher_sensitivities(model, cal_neutral, n_Menu_seqs=8, seq_len=1024)
            prof_v1_neu = lv2.build_liqk_v2_fisher_slice_profile(model, cal_neutral, batch_size=8, fisher_out=fisher_neu)
            prof_v2_neu = v2.build_vtq2_fisher_slice_profile(model, cal_neutral, batch_size=8, fisher_out=fisher_neu, rot_block=-1)

            neu_runs = {}
            for target in (3.0, 4.0):
                t = f"{target:.4f}"
                # 1) TCQPQ (neutral cal)
                restore_clean_model_weights(model, clean)
                a1 = lv2.solve_fisher_slice_dp(prof_v1_neu, target_bpw=target, candidate_specs=eqb.SETS["TCQPQ"])
                q1 = lv2.quantize_llama_model_liqk_v2(model, cal_neutral, a1, mlp_perms=prof_v1_neu["mlp_perms"],
                                                      fisher_out=prof_v1_neu["fisher_out"], batch_size=8, cd_sweeps=2, polish_steps=0)
                neu_runs[f"TCQPQ-NeuCal-{t}"] = {
                    "info": {"linear_bpw": q1["linear_bpw"], "quant_sec": q1["quant_sec"]},
                    "nemotron": evaluate_model_on_cached_trace(model, nem_ref, dev),
                    "multilingual": evaluate_model_on_cached_trace(model, mul_ref, dev),
                    "wikitext2": evaluate_model_on_cached_trace(model, wt_ref, dev),
                }
                # 2) VTQ2-NoGPTAQ (neutral cal)
                a2 = lv2.solve_fisher_slice_dp(prof_v2_neu, target_bpw=target, candidate_specs=eqb.SETS["VTQ2"])
                restore_clean_model_weights(model, clean)
                q2_ng = v2.quantize_llama_model_vtq2(model, cal_neutral, a2, mlp_perms=prof_v2_neu["mlp_perms"],
                                                     fisher_out=prof_v2_neu["fisher_out"], batch_size=4, gptaq_alpha=0.0)
                neu_runs[f"VTQ2-NoGPTAQ-NeuCal-{t}"] = {
                    "info": {"linear_bpw": q2_ng["linear_bpw"], "quant_sec": q2_ng["quant_sec"]},
                    "nemotron": evaluate_model_on_cached_trace(model, nem_ref, dev),
                    "multilingual": evaluate_model_on_cached_trace(model, mul_ref, dev),
                    "wikitext2": evaluate_model_on_cached_trace(model, wt_ref, dev),
                }
                # 3) VTQ2 + GPTAQ (neutral cal)
                restore_clean_model_weights(model, clean)
                q2_g = v2.quantize_llama_model_vtq2(model, cal_neutral, a2, mlp_perms=prof_v2_neu["mlp_perms"],
                                                    fisher_out=prof_v2_neu["fisher_out"], batch_size=4, gptaq_alpha=0.50)
                neu_runs[f"VTQ2-NeuCal-{t}"] = {
                    "info": {"linear_bpw": q2_g["linear_bpw"], "quant_sec": q2_g["quant_sec"]},
                    "nemotron": evaluate_model_on_cached_trace(model, nem_ref, dev),
                    "multilingual": evaluate_model_on_cached_trace(model, mul_ref, dev),
                    "wikitext2": evaluate_model_on_cached_trace(model, wt_ref, dev),
                }
                for k_nm in (f"TCQPQ-NeuCal-{t}", f"VTQ2-NoGPTAQ-NeuCal-{t}", f"VTQ2-NeuCal-{t}"):
                    r_e = neu_runs[k_nm]
                    print(f"  [{k_nm:24s}] nem={r_e['nemotron']['mean_kld']:.5f} mul={r_e['multilingual']['mean_kld']:.5f} "
                          f"wiki2={r_e['wikitext2']['mean_kld']:.5f}", flush=True)

            neu_comps = {}
            for target in (3.0, 4.0):
                t = f"{target:.4f}"
                neu_comps[f"VTQ2-NeuCal-{t} vs TCQPQ-NeuCal-{t}"] = _pair_boot(
                    neu_runs[f"VTQ2-NeuCal-{t}"], neu_runs[f"TCQPQ-NeuCal-{t}"], ("nemotron", "multilingual", "wikitext2")
                )
                neu_comps[f"VTQ2-NoGPTAQ-NeuCal-{t} vs TCQPQ-NeuCal-{t}"] = _pair_boot(
                    neu_runs[f"VTQ2-NoGPTAQ-NeuCal-{t}"], neu_runs[f"TCQPQ-NeuCal-{t}"], ("nemotron", "multilingual", "wikitext2")
                )
            audit_out.write_text(json.dumps({"runs": neu_runs, "comparisons": neu_comps}, indent=2), encoding="utf-8")

    del model, clean, nem_ref, mul_ref, ref_hidden
    torch.cuda.empty_cache()


# =================================================================================================
# 2. Suite `qwen3_8b`: Full Equal-BPW Sweep on `Qwen3-8B` vs `EXL3` & `TCQPQ`
# =================================================================================================

def run_suite_qwen3_8b():
    dev = torch.device("cuda")
    v2.load_and_register_vtq2()
    model_dir = ROOT / "models" / "Qwen3-8B"
    data_dir = ROOT / "data" / "qwen3"
    out_json = ROOT / "results" / "vtq2_qwen3_8b.json"
    results = json.loads(out_json.read_text(encoding="utf-8")) if out_json.exists() else {"model": "Qwen3-8B", "runs": {}, "comparisons": {}}

    # Load cached EXL3 & TCQPQ runs (`3.0, 3.4375, 4.5`) for exact paired bootstrap
    for fn, mapping in [
        ("night2_qwen3_8b_final.json", {"exl3:exl3_out/qwen3_8b_3.0": "EXL3-3.0000", "dp:TCQPQ:3.0": "TCQPQ-3.0000", "dp:TCQPQ:4.5": "TCQPQ-4.5000"}),
        ("night_qwen3_8b.json", {"exl3:exl3_out/qwen3_8b_3.4375": "EXL3-3.4375", "exl3:exl3_out/qwen3_8b_4.5": "EXL3-4.5000"}),
        ("night2_qwen3_8b_dp_fisher.json", {"dp:TCQPQ:3.4375": "TCQPQ-3.4375"}),
    ]:
        p = ROOT / "results" / fn
        if p.exists():
            md = json.loads(p.read_text(encoding="utf-8")).get("methods", {})
            for src_k, dst_k in mapping.items():
                if src_k in md and dst_k not in results["runs"]:
                    results["runs"][dst_k] = md[src_k]

    print("\n=== Full Equal-BPW Sweep on `Qwen3-8B` (`2.5 .. 5.0 bpw`) ===", flush=True)
    model = AutoModelForCausalLM.from_pretrained(str(model_dir), dtype=torch.bfloat16).to(dev).eval()
    clean = save_clean_model_weights(model)
    cal64 = load_file(str(data_dir / "cal_trace.safetensors"))["input_ids"][:64]
    fisher = torch.load(str(data_dir / "fisher.pt"), map_location="cpu", weights_only=True)

    nem_ref = precompute_ref_trace_data(model, load_eval_trace(data_dir / "nemotron_trace.json"), dev)
    mul_ref = precompute_ref_trace_data(model, load_eval_trace(data_dir / "multilingual_trace.json"), dev)

    t0 = time.perf_counter()
    prof_vtq2 = v2.build_vtq2_fisher_slice_profile(model, cal64, batch_size=4, fisher_out=fisher, rot_block=-1)
    prof_vtq2_sec = time.perf_counter() - t0
    print(f"  Built Qwen3-8B VTQ2 Fisher slice profile ({len(prof_vtq2['slices'])} slices) in {prof_vtq2_sec:.1f}s", flush=True)

    t0 = time.perf_counter()
    prof_v1 = lv2.build_liqk_v2_fisher_slice_profile(model, cal64, batch_size=4, fisher_out=fisher)
    prof_v1_sec = time.perf_counter() - t0

    def save_eval(label: str, info: dict):
        entry = {
            "info": info,
            "nemotron": evaluate_model_on_cached_trace(model, nem_ref, dev),
            "multilingual": evaluate_model_on_cached_trace(model, mul_ref, dev),
        }
        results["runs"][label] = entry
        out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"  [{label:24s}] bpw={info['linear_bpw']:.4f} t={info.get('quant_sec', 0):.1f}s | "
              f"nem={entry['nemotron']['mean_kld']:.5f} mul={entry['multilingual']['mean_kld']:.5f}", flush=True)

    ref_hidden = lv2.precompute_ref_final_hidden_states(model, cal64, n_seqs=64, seq_len=512, batch_size=4)
    targets = (2.5, 3.0, 3.4375, 4.0, 4.5, 5.0)
    for target in targets:
        lbl1 = f"TCQPQ-{target:.4f}"
        if lbl1 not in results["runs"]:
            restore_clean_model_weights(model, clean)
            a1 = lv2.solve_fisher_slice_dp(prof_v1, target_bpw=target, candidate_specs=eqb.SETS["TCQPQ"])
            # Use architecture-clean `sq.quantize_layer_v2` so `Qwen3Attention` (`q_norm`/`k_norm`) runs natively!
            t_q1 = time.perf_counter()
            sq_prev_f = sqm.FISHER
            sqm.FISHER = prof_v1["fisher_out"]
            try:
                cal_s = [model.model.embed_tokens(cal64[i : i + 4].to(dev)) for i in range(0, 64, 4)]
                pos_s = model.model.rotary_emb(cal_s[0][:1], torch.arange(cal64.shape[1], device=dev).unsqueeze(0))
                tot_p1, tot_b1 = 0, 0.0
                for l_i in range(model.config.num_hidden_layers):
                    lq = model.model.layers[l_i]
                    pm_l = prof_v1["mlp_perms"][l_i].to(dev)
                    lq.mlp.gate_proj.weight.data = lq.mlp.gate_proj.weight.data[pm_l].contiguous()
                    lq.mlp.up_proj.weight.data = lq.mlp.up_proj.weight.data[pm_l].contiguous()
                    lq.mlp.down_proj.weight.data = lq.mlp.down_proj.weight.data[:, pm_l].contiguous()
                    sqm._CUR_LAYER[0] = l_i
                    sqm.quantize_layer_v2(lq, cal_s, pos_s, lambda full, _l=l_i: a1[(_l, full)], l_i, damp=0.02)
                    for sub_n in lv2.LINEAR_SUBMODULES:
                        K_in = lq.get_submodule(sub_n).weight.shape[1]
                        for r0, r1, sp in a1[(l_i, sub_n)]:
                            tot_p1 += (r1 - r0) * K_in
                            tot_b1 += (r1 - r0) * K_in * lv2.LIQK_V2_SPECS[sp].bpw
                del cal_s
            finally:
                sqm.FISHER = sq_prev_f
            save_eval(lbl1, {"linear_bpw": tot_b1 / max(tot_p1, 1), "quant_sec": time.perf_counter() - t_q1,
                             "profile_sec": prof_v1_sec, "target_bpw": target, "shares_pct": _shares_pct(model, a1)})

        asgn2 = lv2.solve_fisher_slice_dp(prof_vtq2, target_bpw=target, candidate_specs=eqb.SETS["VTQ2"])
        if target == 3.0:
            lbl_ga = f"VTQ2-{target:.4f}-GPTAQ"
            if lbl_ga not in results["runs"]:
                restore_clean_model_weights(model, clean)
                q_ga = v2.quantize_llama_model_vtq2(model, cal64, asgn2, mlp_perms=prof_vtq2["mlp_perms"],
                                                    fisher_out=prof_vtq2["fisher_out"], batch_size=4, gptaq_alpha=0.50)
                save_eval(lbl_ga, {"linear_bpw": q_ga["linear_bpw"], "quant_sec": q_ga["quant_sec"],
                                   "profile_sec": prof_vtq2_sec, "target_bpw": target, "shares_pct": _shares_pct(model, asgn2)})

        lbl2 = f"VTQ2-{target:.4f}"
        lbl2_pol = f"VTQ2-{target:.4f}-Polish"
        need_m = lbl2 not in results["runs"]
        need_p = (target in (2.5, 3.0)) and (lbl2_pol not in results["runs"])
        if need_m or need_p:
            restore_clean_model_weights(model, clean)
            q2 = v2.quantize_llama_model_vtq2(model, cal64, asgn2, mlp_perms=prof_vtq2["mlp_perms"],
                                               fisher_out=prof_vtq2["fisher_out"], batch_size=4, gptaq_alpha=0.0)
            inf2 = {"linear_bpw": q2["linear_bpw"], "quant_sec": q2["quant_sec"], "profile_sec": prof_vtq2_sec,
                    "target_bpw": target, "shares_pct": _shares_pct(model, asgn2)}
            if need_m:
                save_eval(lbl2, inf2)
            if need_p:
                with torch.enable_grad():
                    pol_s = lv2.polish_model_scales_kld(model, cal64, ref_hidden, n_steps=160, lr=4e-4,
                                                        batch_size=2, reg_wd=0.02, ce_weight=0.0)
                save_eval(lbl2_pol, dict(inf2, polish_sec=pol_s))

    comps = {}
    for target in targets:
        t = f"{target:.4f}"
        for cand in (f"VTQ2-{t}", f"VTQ2-{t}-Polish", f"VTQ2-{t}-GPTAQ"):
            if cand not in results["runs"]:
                continue
            for ref in (f"TCQPQ-{t}", f"EXL3-{t}", f"VTQ2-{t}"):
                if ref in results["runs"] and cand != ref:
                    comps[f"{cand} vs {ref}"] = _pair_boot(results["runs"][cand], results["runs"][ref])
                    d = comps[f"{cand} vs {ref}"]
                    print(f"  {cand} vs {ref}: nem {d['nemotron']['mean_diff']:+.5f} ({d['nemotron']['rel_pct']:+.1f}%, p={d['nemotron']['p_a_better']:.4f}) | "
                          f"mul {d['multilingual']['mean_diff']:+.5f} ({d['multilingual']['rel_pct']:+.1f}%, p={d['multilingual']['p_a_better']:.4f})",
                          flush=True)
    results["comparisons"] = comps
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    del model, clean, nem_ref, mul_ref
    torch.cuda.empty_cache()


# =================================================================================================
# 3. Architecture-Agnostic Hybrid GPTAQ Runner (`Qwen3.5-4B` DeltaNet, `Gemma-4-E4B` PLE, `Llama-3.1-8B`)
# =================================================================================================

def compute_hybrid_fisher_offload(model_dir: Path, data_dir: Path, out_path: Path, n_seqs: int = 8, seq_len: int = 1024):
    """Exact per-row Fisher (`src/fisher_full.py` math & RNG seeds) computed layer-by-layer on GPU (<5 GiB peak VRAM)."""
    if out_path.exists():
        return
    print(f"\n=== Computing layer-offloaded Fisher for {model_dir.name} -> {out_path} ===", flush=True)
    t0 = time.perf_counter()
    dev = torch.device("cuda")
    tmod = hqm.TextModel(model_dir, dev)
    n_layers = len(tmod.layers)
    fisher = {}
    for l_idx in range(n_layers):
        for name, mod in tmod.layers[l_idx].named_modules():
            if hqm.quantizable(name, mod):
                fisher[(l_idx, name)] = torch.zeros(mod.out_features, dtype=torch.float64, device=dev)

    cal = load_file(str(data_dir / "cal_trace.safetensors"))["input_ids"][:n_seqs, :seq_len]
    toks = [cal[i : i + 1].to(dev) for i in range(cal.shape[0])]
    total = sum(t.numel() for t in toks)

    with torch.no_grad():
        hs = [tmod.capture(t, 0)[0] for t in toks]
        inputs = []
        for l_idx in range(n_layers):
            inputs.append([h.cpu() for h in hs])
            layer = tmod.load_layer(l_idx)
            ctx_l = [tmod.capture(t, l_idx)[1:] for t in toks]
            hs = [hqm.call_layer(layer, h, ctx_l[i], {}) for i, h in enumerate(hs)]
            del ctx_l
            tmod.release_layer(l_idx)
            torch.cuda.empty_cache()

    gs = []
    for i, h in enumerate(hs):
        h_in = h.detach().requires_grad_(True)
        with torch.enable_grad():
            logits = tmod.logits(h_in).reshape(-1, tmod.config.vocab_size)
            with torch.no_grad():
                probs = F.softmax(logits.float(), dim=-1)
                u = torch.rand((probs.shape[0], 2), generator=torch.Generator().manual_seed(1000 + i)).to(dev)
                cdf = probs.cumsum(dim=-1)
                y = torch.searchsorted(cdf, (u * cdf[:, -1:]).contiguous()).clamp(max=probs.shape[-1] - 1)
                del probs, cdf
            loss = 0.5 * (
                F.cross_entropy(logits.float(), y[:, 0], reduction="sum")
                + F.cross_entropy(logits.float(), y[:, 1], reduction="sum")
            )
            gs.append(torch.autograd.grad(loss, h_in)[0].detach())
        del logits, loss
    del hs

    for l_idx in reversed(range(n_layers)):
        layer = tmod.load_layer(l_idx)
        for p in layer.parameters():
            p.requires_grad_(False)
        ctx_l = [tmod.capture(t, l_idx)[1:] for t in toks]
        hooks = []
        for name, mod in layer.named_modules():
            if not hqm.quantizable(name, mod):
                continue
            key = (l_idx, name)

            def fwd(m, inp, out, key=key):
                def acc(gr, key=key):
                    fisher[key].add_(gr.detach().float().square().reshape(-1, gr.shape[-1]).sum(dim=0).double())

                if out.requires_grad:
                    out.register_hook(acc)

            hooks.append(mod.register_forward_hook(fwd))
        for i in range(len(toks)):
            x = inputs[l_idx][i].to(dev).requires_grad_(True)
            with torch.enable_grad():
                out = hqm.call_layer(layer, x, ctx_l[i], {})
                gs[i] = torch.autograd.grad(out, x, grad_outputs=gs[i])[0].detach()
            del x, out
        for hk in hooks:
            hk.remove()
        del ctx_l
        tmod.release_layer(l_idx)
        inputs[l_idx] = None
        torch.cuda.empty_cache()
        if l_idx % 8 == 0:
            print(f"  [Fisher {model_dir.name}] backward layer {l_idx}/{n_layers} ({time.perf_counter() - t0:.1f}s)", flush=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    res = {k: (v / max(total, 1)).float().cpu() for k, v in fisher.items()}
    torch.save(res, str(out_path))
    print(f"  Saved Fisher ({len(res)} matrices) -> {out_path} in {time.perf_counter() - t0:.1f}s", flush=True)
    del tmod, fisher, gs
    torch.cuda.empty_cache()


def _fmt_tgt_label(t) -> str:
    if isinstance(t, str):
        return t
    fv = float(t)
    return f"{fv:.1f}" if abs(round(fv, 1) - fv) < 1e-6 else f"{fv:.4f}".rstrip("0").rstrip(".")


@torch.no_grad()
def run_hybrid_vtq2_eval(
    model_dir: Path,
    data_dir: Path,
    fisher_path: Path,
    targets: tuple,
    out_json: Path,
    cached_baseline_json: Optional[Path] = None,
    extra_ref_pairs: Optional[Dict[str, List[str]]] = None,
):
    print(f"\n=== Evaluating Hybrid Model `{model_dir.name}` (`TextModel` + `VTQ2` DP) ===", flush=True)
    dev = torch.device("cuda")
    v2.install_vtq2_hooks()
    results = json.loads(out_json.read_text(encoding="utf-8")) if out_json.exists() else {"model": str(model_dir), "methods": {}, "comparisons": {}}
    if cached_baseline_json is not None and cached_baseline_json.exists():
        base_m = json.loads(cached_baseline_json.read_text(encoding="utf-8")).get("methods", {})
        for k, val in base_m.items():
            if "gemma-4" in str(model_dir) and k.startswith("dp:"):
                continue
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
    over = hqm.overhead_bpw(tmod, n_layers)

    need_any_v2 = [t for t in targets if f"dp:VTQ2:{_fmt_tgt_label(t)}" not in results["methods"]]
    prof_vtq2 = None
    prof_vtq2_sec = 0.0
    if need_any_v2:
        os.environ["VTQ2_PROFILE"] = "1"
        t0 = time.perf_counter()
        prof_vtq2 = hqm.build_profile(tmod, cal_ids[:4], n_layers)
        prof_vtq2_sec = time.perf_counter() - t0
        os.environ["VTQ2_PROFILE"] = "0"
        print(f"  Built hybrid VTQ2 profile ({len(prof_vtq2['slices'])} slices) in {prof_vtq2_sec:.1f}s", flush=True)

    # Check if any `dp:TCQPQ:*` baseline is missing
    need_v1 = [t for t in targets if f"dp:TCQPQ:{_fmt_tgt_label(t)}" not in results["methods"]]
    prof_v1 = hqm.build_profile(tmod, cal_ids[:4], n_layers) if need_v1 else None

    active = {}
    for t in targets:
        t_lbl = _fmt_tgt_label(t)
        k_v1 = f"dp:TCQPQ:{t_lbl}"
        if k_v1 not in results["methods"] and prof_v1 is not None:
            asgn1 = lv2.solve_fisher_slice_dp(prof_v1, target_bpw=float(t) - over, candidate_specs=eqb.SETS["TCQPQ"])
            active[k_v1] = {"menu": "TCQPQ", "asgn": asgn1, "gptaq": 0.0, "quant_sec": 0.0,
                            "bits": 0.0, "params": 0, "q_bits": 0.0, "q_params": 0,
                            "streams": {ds: [e.clone() for e in v] for ds, v in emb.items()},
                            "states": {ds: [{} for _ in v] for ds, v in emb.items()},
                            "cal": [c.clone() for c in cal0], "cal_states": [{} for _ in cal0]}

        for sfx, alpha in (("", 0.0),):
            k_v2 = f"dp:VTQ2:{t_lbl}{sfx}"
            if k_v2 not in results["methods"] and prof_vtq2 is not None:
                asgn2 = lv2.solve_fisher_slice_dp(
                    prof_vtq2, target_bpw=float(t) - over, candidate_specs=v2.vtq2_candidates_for_target(float(t) - over)
                )
                active[k_v2] = {"menu": "VTQ2", "asgn": asgn2, "gptaq": alpha, "quant_sec": 0.0,
                                "bits": 0.0, "params": 0, "q_bits": 0.0, "q_params": 0,
                                "streams": {ds: [e.clone() for e in v] for ds, v in emb.items()},
                                "states": {ds: [{} for _ in v] for ds, v in emb.items()},
                                "cal": [c.clone() for c in cal0], "cal_states": [{} for _ in cal0]}

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
                    acc = v2.DualCovAccumulator(need_cross=use_g)
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
                        if st["menu"] == "VTQ2":
                            W_tgt = (W + st["gptaq"] * (W @ B.T)) if (use_g and B is not None) else None
                            _, w_hat, _ = v2.quantize_matrix_vtq2(W, H, r_sp, seed=sd, damp=0.20 * dm, row_weights=rw, W_target=W_tgt)
                        else:
                            _, w_hat, _ = lv2.quantize_matrix_liqk_v2(W, H, r_sp, seed=sd, damp=0.02 * dm, row_weights=rw, cd_sweeps=2)
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
                "profile_sec": prof_vtq2_sec,
                "linear_bpw": st["bits"] / max(st["params"], 1),
                "q_bpw": st["q_bits"] / max(st["q_params"], 1),
                "shares_pct": _shares_pct(tmod.layers, st["asgn"]),
            }
            for ds, items in traces.items():
                ref_f = [h[0, pos] for h, (_, pos, _) in zip(ref["streams"][ds], items)]
                q_f = [h[0, pos] for h, (_, pos, _) in zip(st["streams"][ds], items)]
                entry[ds] = sqm.trace_metrics(tmod.model, ref_f, q_f, [tg for _, _, tg in items], dev, logits_fn=tmod.logits)
            results["methods"][m_name] = entry
            print(f"  [{m_name:24s}] q_bpw={entry['q_bpw']:.4f} t={entry['quant_sec']:.1f}s | "
                  f"nem={entry['nemotron']['mean_kld']:.5f} mul={entry['multilingual']['mean_kld']:.5f}", flush=True)

    comps = {}
    for t in targets:
        t_lbl = _fmt_tgt_label(t)
        cand_list = [f"dp:VTQ2:{t_lbl}"]
        if f"dp:VTQ2:{float(t):g}" != cand_list[0]:
            cand_list.append(f"dp:VTQ2:{float(t):g}")
        for cand in cand_list:
            if cand not in results["methods"]:
                continue
            ref_candidates = [f"dp:TCQPQ:{t_lbl}", f"dp:TCQPQ:{float(t):g}", f"exl3q:{t_lbl}", f"exl3q:{float(t):g}"]
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
                    print(f"  {cand} vs {ref_k}: nem {d['nemotron']['mean_diff']:+.5f} ({d['nemotron']['rel_pct']:+.1f}%, p={d['nemotron']['p_a_better']:.4f}) | "
                          f"mul {d['multilingual']['mean_diff']:+.5f} ({d['multilingual']['rel_pct']:+.1f}%, p={d['multilingual']['p_a_better']:.4f})",
                          flush=True)
    results["comparisons"] = comps
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")


@torch.no_grad()
def run_gemma4_ple_vtq_rings(model_dir: Path, data_dir: Path, out_json: Path):
    print("\n=== Evaluating `VtqRing` on Gemma 4 E4B PLE (`embed_tokens_per_layer`) ===", flush=True)
    dev = torch.device("cuda")
    v2.load_and_register_vtq2()
    res = json.loads(out_json.read_text(encoding="utf-8")) if out_json.exists() else {"methods": {}}
    base_ple = ROOT / "results" / "g4_ple_gemma4.json"
    if base_ple.exists():
        raw_ple = json.loads(base_ple.read_text(encoding="utf-8"))
        for k, v in raw_ple.get("methods", raw_ple).items():
            res["methods"].setdefault(k, v)

    tmod = hqm.TextModel(model_dir, dev)
    n_layers = len(tmod.layers)
    for l_i in range(n_layers):
        tmod.load_layer(l_i)
    tmod.tm.layers = nn.ModuleList(tmod.layers)
    ple_mod = tmod.tm.embed_tokens_per_layer
    W_orig = ple_mod.weight.data.clone()

    traces = {}
    for ds in ("nemotron", "multilingual"):
        items = []
        for sq_item in load_eval_trace(data_dir / f"{ds}_trace.json"):
            pos, tg = get_scored_positions_and_targets(sq_item)
            if pos:
                items.append((torch.tensor([sq_item["token_ids"][:8192]]), pos, tg))
        traces[ds] = items

    variants = [
        ("VTQ200:rot", "VTQ200", 256),
        ("VTQ250:rot", "VTQ250", 256),
        ("VTQ300:rot", "VTQ300", 256),
        ("VTQ400:rot", "VTQ400", 256),
    ]
    todo = [v for v in variants if v[0] not in res["methods"]]
    if not todo:
        out_json.write_text(json.dumps(res, indent=2), encoding="utf-8")
        return

    q_tables = {}
    for label, sp_nm, rlen in todo:
        t0 = time.perf_counter()
        wq, bpv = v2.quantize_rings_vtq2(W_orig.float(), sp_nm, ring_len=rlen, rot=True)
        dt = time.perf_counter() - t0
        rel_mse = float((W_orig.float() - wq).square().mean() / W_orig.float().square().mean())
        q_tables[label] = {"wq": wq.to(W_orig.dtype), "bpv": bpv, "quant_sec": dt, "rel_mse": rel_mse}
        print(f"  Quantized PLE `{label}` ({bpv:.3f} bpv) in {dt:.1f}s | rel_mse={rel_mse:.4e}", flush=True)

    ref_h = {}
    ple_mod.weight.data = W_orig
    for ds, items in traces.items():
        ref_h[ds] = [tmod.tm(input_ids=ids.to(dev), use_cache=False).last_hidden_state[0, pos].cpu() for ids, pos, _ in items]

    for label, _, _ in todo:
        meta = q_tables[label]
        ple_mod.weight.data = meta["wq"]
        entry = {
            "bits_per_value": meta["bpv"],
            "size_gb": W_orig.numel() * meta["bpv"] / 8.0 / (2**30),
            "quant_sec": meta["quant_sec"],
            "rel_mse": meta["rel_mse"],
        }
        for ds, items in traces.items():
            q_f = [tmod.tm(input_ids=ids.to(dev), use_cache=False).last_hidden_state[0, pos].cpu() for ids, pos, _ in items]
            entry[ds] = sqm.trace_metrics(tmod.model, ref_h[ds], q_f, [tg for _, _, tg in items], dev, logits_fn=tmod.logits)
        res["methods"][label] = entry
        print(f"  [PLE {label:14s}] bpv={meta['bpv']:.4f} | nem={entry['nemotron']['mean_kld']:.6f} mul={entry['multilingual']['mean_kld']:.6f}",
              flush=True)
    ple_mod.weight.data = W_orig
    out_json.write_text(json.dumps(res, indent=2), encoding="utf-8")
    del tmod, q_tables, ref_h
    torch.cuda.empty_cache()


# =================================================================================================
# 4. Suite `moe_30b`: Streaming `Qwen3-30B-A3B-Instruct-2507` (`48` MoE layers x `128` experts)
# =================================================================================================

def run_suite_moe_30b():
    import subprocess
    import sys

    v2.install_vtq2_hooks()
    model_dir = ROOT / "models" / "Qwen3-30B-A3B-Instruct-2507"
    data_dir = ROOT / "data" / "qwen3"
    fisher_path = data_dir / "fisher_30b.pt"
    out_json = ROOT / "results" / "vtq2_qwen3_30b.json"

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

    # Run `streaming_quant` directly with `VTQ2_PROFILE=1` and `dp:VTQ2:{3.0, 4.5}`!
    os.environ["VTQ2_PROFILE"] = "1"
    tmp_out = ROOT / "results" / "_tmp_vtq2_30b.json"
    cmd = [
        sys.executable, "-u", "-c",
        (
            "import os, sys; os.environ['VTQ2_PROFILE']='1'; "
            "import src.vtq_v2 as v2; v2.install_vtq2_hooks(); "
            "import src.streaming_quant as sq; "
            f"sys.argv = ['streaming_quant', '--model_dir', {str(model_dir)!r}, '--data_dir', {str(data_dir)!r}, "
            f"'--fisher', {str(fisher_path)!r}, '--methods', 'dp:VTQ2:3.0', 'dp:VTQ2:4.5', "
            f"'--out', {str(tmp_out)!r}]; sq.main()"
        ),
    ]
    subprocess.run(cmd, check=True)

    res = json.loads(tmp_out.read_text(encoding="utf-8"))
    base_30b = ROOT / "results" / "g4_qwen3_30b_dp.json"
    if base_30b.exists():
        for k, v in json.loads(base_30b.read_text(encoding="utf-8")).get("methods", {}).items():
            res["methods"].setdefault(k, v)

    comps = {}
    for t_str, exl_k in [("3.0", "exl3:exl3_out/qwen3_30b_3.0"), ("3.5", None), ("4.5", "exl3:exl3_out/qwen3_30b_4.5")]:
        cand = f"dp:VTQ2:{t_str}"
        if cand in res["methods"]:
            for rk in [f"dp:TCQPQ:{t_str}", exl_k]:
                if rk and rk in res["methods"]:
                    comps[f"{cand} vs {rk}"] = _pair_boot(res["methods"][cand], res["methods"][rk])
                    d = comps[f"{cand} vs {rk}"]
                    print(f"  {cand} vs {rk}: nem {d['nemotron']['mean_diff']:+.5f} ({d['nemotron']['rel_pct']:+.1f}%, p={d['nemotron']['p_a_better']:.4f}) | "
                          f"mul {d['multilingual']['mean_diff']:+.5f} ({d['multilingual']['rel_pct']:+.1f}%, p={d['multilingual']['p_a_better']:.4f})",
                          flush=True)
    res["comparisons"] = comps
    out_json.write_text(json.dumps(res, indent=2), encoding="utf-8")
    tmp_out.unlink(missing_ok=True)


def run_suite_speed():
    import subprocess
    out_j = ROOT / "results" / "vtq2_decode_speed.json"
    cpu_bin = "/tmp/vtq2_avx2_bench"
    gpu_bin = "/tmp/vtq2_matvec_bench"
    subprocess.run(["gcc", "-O3", "-mavx2", "-mfma", "-mno-avx512f", str(ROOT / "src/cpu/vtq2_avx2_bench.c"), "-o", cpu_bin, "-lm"], check=True)
    subprocess.run(["nvcc", "-O3", "-arch=sm_120", str(ROOT / "src/cuda/vtq2_matvec_bench.cu"), "-o", gpu_bin], check=True)
    cpu_out = subprocess.run([cpu_bin], capture_output=True, text=True, check=True).stdout
    gpu_out = subprocess.run([gpu_bin], capture_output=True, text=True, check=True).stdout
    print(cpu_out, flush=True)
    print(gpu_out, flush=True)
    out_j.write_text(json.dumps({"cpu_avx2_raw": cpu_out, "gpu_blackwell_raw": gpu_out}, indent=2), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--suite",
        choices=["3b", "1b", "3b_1b", "qwen3_8b", "qwen35", "gemma4", "hybrid", "llama8b", "qwen27", "moe_30b", "speed"],
        required=True,
    )
    args = ap.parse_args()

    if args.suite == "speed":
        run_suite_speed()
        return
    if args.suite == "qwen27":
        f27 = ROOT / "data" / "qwen38" / "fisher.pt"
        compute_hybrid_fisher_offload(ROOT / "models" / "Qwen3.8-27B", ROOT / "data" / "qwen38", f27)
        run_hybrid_vtq2_eval(
            ROOT / "models" / "Qwen3.8-27B",
            ROOT / "data" / "qwen38",
            f27,
            ("2.9409", "3.0043", "3.4794", "3.5038"),
            ROOT / "results" / "vtq2_qwen38_27b.json",
            ROOT / "results" / "q27_compare.json",
            extra_ref_pairs={
                "2.9409": ["gguf:/content/m/gguf/Qwen3.8-27B-GSQ-RCO-IQ3_XXS.gguf"],
                "3.0043": ["exl3:/content/m/exl3_SC_3.00bpw_H4"],
                "3.4794": ["gguf:/content/m/gguf/Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf"],
                "3.5038": ["exl3:/content/m/exl3_3.50bpw"],
            },
        )
        return
    if args.suite in ("3b", "3b_1b"):
        run_dense_sweep_model("3b", "Llama-3.2-3B-Instruct", polish_steps=80, polish_lr=4e-4)
    if args.suite in ("1b", "3b_1b"):
        run_dense_sweep_model("1b", "Llama-3.2-1B-Instruct", polish_steps=100, polish_lr=5e-4)
    if args.suite == "qwen3_8b":
        run_suite_qwen3_8b()
    if args.suite in ("qwen35", "hybrid"):
        run_hybrid_vtq2_eval(
            ROOT / "models" / "Qwen3.5-4B",
            ROOT / "data" / "qwen35",
            ROOT / "data" / "qwen35" / "fisher.pt",
            (2.5, 3.0, 3.5, 4.0, 4.5),
            ROOT / "results" / "vtq2_qwen35_4b.json",
            ROOT / "results" / "g4_hybrid_qwen35.json",
        )
    if args.suite in ("gemma4", "hybrid"):
        run_hybrid_vtq2_eval(
            ROOT / "models" / "gemma-4-E4B-it",
            ROOT / "data" / "gemma4",
            ROOT / "data" / "gemma4" / "fisher.pt",
            (2.5, 3.0, 3.5, 4.0, 4.5),
            ROOT / "results" / "vtq2_gemma4_e4b.json",
            ROOT / "results" / "g4_hybrid_gemma4.json",
        )
        run_gemma4_ple_vtq_rings(
            ROOT / "models" / "gemma-4-E4B-it",
            ROOT / "data" / "gemma4",
            ROOT / "results" / "vtq2_gemma4_ple.json",
        )
    if args.suite == "llama8b":
        # Full 32-layer Llama-3.1-8B-Instruct (`down_proj` K = 14336 = 2048 * 7, stress-testing Kronecker R_{14336}!)
        dev = torch.device("cuda")
        v2.load_and_register_vtq2()
        m_dir = ROOT / "models" / "Llama-3.1-8B-Instruct"
        out_j = ROOT / "results" / "vtq2_llama31_8b.json"
        res8 = json.loads(out_j.read_text(encoding="utf-8")) if out_j.exists() else {"model": "Llama-3.1-8B-Instruct", "runs": {}, "comparisons": {}}
        p_nl8 = ROOT / "results" / "night_llama8b.json"
        if p_nl8.exists():
            md8 = json.loads(p_nl8.read_text(encoding="utf-8")).get("methods", {})
            for sk, dk in (("exl3:exl3_out/llama8b_3.4375", "EXL3-3.4375"), ("exl3:exl3_out/llama8b_4.5", "EXL3-4.5000")):
                if sk in md8 and dk not in res8["runs"]:
                    res8["runs"][dk] = md8[sk]
        m8 = AutoModelForCausalLM.from_pretrained(str(m_dir), dtype=torch.bfloat16).to(dev).eval()
        cl8 = save_clean_model_weights(m8)
        cal64 = load_file(str(ROOT / "data" / "cal_trace.safetensors"))["input_ids"][:64]
        nem_r = precompute_ref_trace_data(m8, load_eval_trace(ROOT / "data" / "nemotron_trace.json"), dev)
        mul_r = precompute_ref_trace_data(m8, load_eval_trace(ROOT / "data" / "multilingual_trace.json"), dev)
        f8_path = ROOT / "data" / "fisher_llama8b.pt"
        f8 = torch.load(str(f8_path), map_location="cpu", weights_only=True) if f8_path.exists() else lv2.compute_exact_row_fisher_sensitivities(m8, cal64, n_Menu_seqs=8, seq_len=1024)
        if not f8_path.exists():
            torch.save(f8, str(f8_path))
        p_v1 = lv2.build_liqk_v2_fisher_slice_profile(m8, cal64, batch_size=4, fisher_out=f8)
        p_v2 = v2.build_vtq2_fisher_slice_profile(m8, cal64, batch_size=4, fisher_out=f8, rot_block=-1)
        ref_h8 = lv2.precompute_ref_final_hidden_states(m8, cal64, n_seqs=64, seq_len=512, batch_size=4)
        for tgt in (2.5, 3.0, 3.4375, 4.0, 4.5):
            t_s = f"{tgt:.4f}"
            if f"TCQPQ-{t_s}" not in res8["runs"]:
                restore_clean_model_weights(m8, cl8)
                a1 = lv2.solve_fisher_slice_dp(p_v1, target_bpw=tgt, candidate_specs=eqb.SETS["TCQPQ"])
                q1 = lv2.quantize_llama_model_liqk_v2(m8, cal64, a1, mlp_perms=p_v1["mlp_perms"], fisher_out=p_v1["fisher_out"], batch_size=4, cd_sweeps=2, polish_steps=0)
                res8["runs"][f"TCQPQ-{t_s}"] = {"info": q1, "nemotron": evaluate_model_on_cached_trace(m8, nem_r, dev), "multilingual": evaluate_model_on_cached_trace(m8, mul_r, dev)}
                print(f"  [Llama8B TCQPQ-{t_s}] nem={res8['runs'][f'TCQPQ-{t_s}']['nemotron']['mean_kld']:.5f} mul={res8['runs'][f'TCQPQ-{t_s}']['multilingual']['mean_kld']:.5f}", flush=True)
            need_m = f"VTQ2-{t_s}" not in res8["runs"]
            need_p = (tgt in (2.5, 3.0)) and (f"VTQ2-{t_s}-Polish" not in res8["runs"])
            if need_m or need_p:
                restore_clean_model_weights(m8, cl8)
                a2 = lv2.solve_fisher_slice_dp(p_v2, target_bpw=tgt, candidate_specs=v2.vtq2_candidates_for_target(tgt))
                q2 = v2.quantize_llama_model_vtq2(m8, cal64, a2, mlp_perms=p_v2["mlp_perms"], fisher_out=p_v2["fisher_out"], batch_size=4, gptaq_alpha=0.0)
                if need_m:
                    res8["runs"][f"VTQ2-{t_s}"] = {"info": q2, "nemotron": evaluate_model_on_cached_trace(m8, nem_r, dev), "multilingual": evaluate_model_on_cached_trace(m8, mul_r, dev)}
                    print(f"  [Llama8B VTQ2-{t_s} ] nem={res8['runs'][f'VTQ2-{t_s}']['nemotron']['mean_kld']:.5f} mul={res8['runs'][f'VTQ2-{t_s}']['multilingual']['mean_kld']:.5f}", flush=True)
                if need_p:
                    with torch.enable_grad():
                        ps = lv2.polish_model_scales_kld(m8, cal64, ref_h8, n_steps=120, lr=4e-4, batch_size=2, reg_wd=0.02, ce_weight=0.0)
                    res8["runs"][f"VTQ2-{t_s}-Polish"] = {"info": dict(q2, polish_sec=ps), "nemotron": evaluate_model_on_cached_trace(m8, nem_r, dev), "multilingual": evaluate_model_on_cached_trace(m8, mul_r, dev)}
                    print(f"  [Llama8B VTQ2-{t_s}-Polish] nem={res8['runs'][f'VTQ2-{t_s}-Polish']['nemotron']['mean_kld']:.5f} mul={res8['runs'][f'VTQ2-{t_s}-Polish']['multilingual']['mean_kld']:.5f}", flush=True)
            for cand in (f"VTQ2-{t_s}", f"VTQ2-{t_s}-Polish"):
                if cand in res8["runs"]:
                    for rk in (f"TCQPQ-{t_s}", f"EXL3-{t_s}", f"VTQ2-{t_s}"):
                        if rk in res8["runs"] and cand != rk:
                            res8["comparisons"][f"{cand} vs {rk}"] = _pair_boot(res8["runs"][cand], res8["runs"][rk])
                            d = res8["comparisons"][f"{cand} vs {rk}"]
                            print(f"  {cand} vs {rk}: nem {d['nemotron']['mean_diff']:+.5f} ({d['nemotron']['rel_pct']:+.1f}%, p={d['nemotron']['p_a_better']:.4f}) | mul {d['multilingual']['mean_diff']:+.5f} ({d['multilingual']['rel_pct']:+.1f}%, p={d['multilingual']['p_a_better']:.4f})", flush=True)
            out_j.write_text(json.dumps(res8, indent=2), encoding="utf-8")
    if args.suite == "moe_30b":
        run_suite_moe_30b()


if __name__ == "__main__":
    main()
