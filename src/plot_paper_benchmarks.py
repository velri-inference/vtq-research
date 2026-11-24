#!/usr/bin/env python3
"""
Reproducible dark-themed figure generator (`python -m src.plot_paper_benchmarks`):
Reads `results/paper_{llama31_8b,gemma4_26b,qwen36_35b}.json` and writes all `.svg` + `.png`
figures into `docs/figures/`:
  1. `<model>_kld_ppl.{svg,png}` : 2x3 Mean KLD + ΔPPL vs Total BPW
  2. `<model>_kld.{svg,png}`     : 1x3 Mean KLD vs Total BPW
  3. `paper_quant_time.{svg,png}`: Quantization creation time & speedup across all 3 models
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

BG_OUTER = "#0f1117"
BG_AX = "#181c25"
GRID_COL = "#282e3d"
SPINE_COL = "#343b4f"
TXT_MAIN = "#f3f4f6"
TXT_SUB = "#9ca3af"

COL_VTQ = "#34d399"
COL_EXL3 = "#fbbf24"
COL_GGUF = "#f87171"

SERIES_SPEC = [
    ("VTQ (.vtq v2)", COL_VTQ, "o", ["VTQ:2.5", "VTQ:3.5", "VTQ:4.5", "VTQ:6.0"], 3),
    ("EXL3 (mul1)", COL_EXL3, "s", ["EXL3:2.5", "EXL3:3.5", "EXL3:4.5", "EXL3:6.0"], 2),
    ("GGUF (imatrix)", COL_GGUF, "D", ["GGUF:IQ2_S", "GGUF:IQ3_M", "GGUF:Q4_K_S", "GGUF:Q5_K_M", "GGUF:Q6_K"], 1),
]

PANELS = [
    ("wikitext2", "WikiText-2 (held-out test)"),
    ("nemotron", "Nemotron (agentic & reasoning)"),
    ("multilingual", "Multilingual (10 languages)"),
]

MODELS = [
    ("llama31_8b", "Llama-3.1-8B-Instruct", "Llama-3.1-8B (Dense)"),
    ("gemma4_26b", "gemma-4-26B-A4B-it", "Gemma-4-26B (128E MoE)"),
    ("qwen36_35b", "Qwen3.6-35B-A3B", "Qwen3.6-35B (256E Hybrid)"),
]


def _apply_dark_rc():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.facecolor": BG_AX,
        "figure.facecolor": BG_OUTER,
        "axes.edgecolor": SPINE_COL,
        "axes.labelcolor": TXT_SUB,
        "xtick.color": TXT_SUB,
        "ytick.color": TXT_SUB,
        "text.color": TXT_MAIN,
    })


def render_pareto_charts(json_path: Path, fig_dir: Path, tag: str, title_model: str):
    _apply_dark_rc()
    ms = json.loads(json_path.read_text(encoding="utf-8"))["methods"]
    all_bpws = [ms[k]["total_bpw"] for _, _, _, keys, _ in SERIES_SPEC for k in keys if k in ms]
    x_min = min(2.32, min(all_bpws) - 0.15)
    x_max = max(6.75, max(all_bpws) + 0.18)

    # --- 1. 2x3 KLD + ΔPPL chart ---
    fig, axes = plt.subplots(2, 3, figsize=(16.2, 8.6), dpi=160, sharex="col")
    fig.patch.set_facecolor(BG_OUTER)
    fig.suptitle(
        f"{title_model} — End-to-End KLD & Perplexity vs Total BPW (Shared WikiText-2 Calibration)",
        fontsize=13.5, fontweight="bold", color=TXT_MAIN, y=0.975,
    )

    for col, (ds_key, ds_title) in enumerate(PANELS):
        ax_k = axes[0, col]
        ax_k.set_yscale("log")
        for label, color, marker, keys, zord in SERIES_SPEC:
            pts = [(ms[k]["total_bpw"], ms[k][ds_key]["mean_kld"]) for k in keys if k in ms]
            if pts:
                xs, ys = zip(*pts)
                ax_k.plot(
                    xs, ys, label=label, color=color, marker=marker, markersize=6.2,
                    markeredgecolor=BG_OUTER, markeredgewidth=1.1, linewidth=2.3, zorder=zord + 2,
                )
        ax_k.set_title(ds_title, fontsize=11, fontweight="bold", color=TXT_MAIN, pad=7)
        if col == 0:
            ax_k.set_ylabel("Mean KLD vs BF16 (log, ↓ better)", fontsize=9.8, color=TXT_SUB)
            leg = ax_k.legend(loc="upper right", frameon=True, facecolor="#12151e", edgecolor=SPINE_COL, fontsize=9.2, labelcolor=TXT_MAIN)
            leg.set_zorder(10)
        ax_k.grid(True, which="major", linestyle="--", linewidth=0.7, color=GRID_COL, alpha=0.9)
        ax_k.grid(True, which="minor", linestyle=":", linewidth=0.5, color=GRID_COL, alpha=0.55)
        ax_k.set_xlim(x_min, x_max)

        ax_p = axes[1, col]
        ref_ppl = next(ms[k][ds_key]["ppl_ref"] for _, _, _, keys, _ in SERIES_SPEC for k in keys if k in ms)
        dppls = [ms[k][ds_key]["delta_ppl"] for _, _, _, keys, _ in SERIES_SPEC for k in keys if k in ms]
        all_pos = min(dppls) > 0.0015
        if all_pos:
            ax_p.set_yscale("log")
            y_label = "ΔPerplexity = PPL_q − PPL_BF16 (log, ↓ better)"
        else:
            ax_p.set_yscale("symlog", linthresh=0.05, linscale=0.75)
            ax_p.axhline(0.0, color="#9ca3af", linestyle="--", linewidth=1.1, alpha=0.8, zorder=1)
            y_lo, y_hi = min(dppls), max(dppls)
            cand_ticks = [-2.0, -1.0, -0.3, -0.1, -0.05, 0.0, 0.05, 0.1, 0.3, 1.0, 2.0]
            vis_ticks = [t for t in cand_ticks if (y_lo * 1.25 - 0.02) <= t <= (y_hi * 1.25 + 0.02)]
            if 0.0 not in vis_ticks:
                vis_ticks = sorted(vis_ticks + [0.0])
            ax_p.set_yticks(vis_ticks)
            ax_p.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: "0" if abs(v) < 1e-9 else f"{v:+.2g}"))
            y_label = "ΔPerplexity = PPL_q − PPL_BF16 (0 = BF16)"

        for label, color, marker, keys, zord in SERIES_SPEC:
            pts = [(ms[k]["total_bpw"], ms[k][ds_key]["delta_ppl"]) for k in keys if k in ms]
            if pts:
                xs, ys = zip(*pts)
                ax_p.plot(
                    xs, ys, label=label, color=color, marker=marker, markersize=6.2,
                    markeredgecolor=BG_OUTER, markeredgewidth=1.1, linewidth=2.3, zorder=zord + 2,
                )
        ax_p.set_title(f"{ds_title} — ΔPPL (BF16 = {ref_ppl:.3f})", fontsize=10.5, fontweight="bold", color=TXT_MAIN, pad=7)
        if col == 0:
            ax_p.set_ylabel(y_label, fontsize=9.6, color=TXT_SUB)
        ax_p.set_xlabel("Total Model Bitrate (total_bpw)", fontsize=9.8, color=TXT_SUB)
        ax_p.xaxis.set_major_locator(mticker.MultipleLocator(0.5))
        ax_p.grid(True, which="major", linestyle="--", linewidth=0.7, color=GRID_COL, alpha=0.9)
        ax_p.grid(True, which="minor", linestyle=":", linewidth=0.5, color=GRID_COL, alpha=0.55)
        ax_p.set_xlim(x_min, x_max)

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.945))
    out_kp = fig_dir / f"{tag}_kld_ppl"
    fig.savefig(out_kp.with_suffix(".svg"), facecolor=BG_OUTER)
    fig.savefig(out_kp.with_suffix(".png"), facecolor=BG_OUTER, dpi=160)
    plt.close(fig)

    # --- 2. 1x3 KLD-only chart ---
    fig2, axes2 = plt.subplots(1, 3, figsize=(16.2, 4.7), dpi=160)
    fig2.patch.set_facecolor(BG_OUTER)
    fig2.suptitle(
        f"{title_model} — End-to-End Mean KLD vs Total BPW (Shared WikiText-2 Calibration)",
        fontsize=13.2, fontweight="bold", color=TXT_MAIN, y=0.97,
    )
    for col, (ds_key, ds_title) in enumerate(PANELS):
        ax = axes2[col]
        ax.set_yscale("log")
        for label, color, marker, keys, zord in SERIES_SPEC:
            pts = [(ms[k]["total_bpw"], ms[k][ds_key]["mean_kld"]) for k in keys if k in ms]
            if pts:
                xs, ys = zip(*pts)
                ax.plot(
                    xs, ys, label=label, color=color, marker=marker, markersize=6.2,
                    markeredgecolor=BG_OUTER, markeredgewidth=1.1, linewidth=2.3, zorder=zord + 2,
                )
        ax.set_title(ds_title, fontsize=11, fontweight="bold", color=TXT_MAIN, pad=7)
        if col == 0:
            ax.set_ylabel("Mean KLD vs BF16 (log, ↓ better)", fontsize=9.8, color=TXT_SUB)
            leg = ax.legend(loc="upper right", frameon=True, facecolor="#12151e", edgecolor=SPINE_COL, fontsize=9.2, labelcolor=TXT_MAIN)
            leg.set_zorder(10)
        ax.set_xlabel("Total Model Bitrate (total_bpw)", fontsize=9.8, color=TXT_SUB)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(0.5))
        ax.grid(True, which="major", linestyle="--", linewidth=0.7, color=GRID_COL, alpha=0.9)
        ax.grid(True, which="minor", linestyle=":", linewidth=0.5, color=GRID_COL, alpha=0.55)
        ax.set_xlim(x_min, x_max)

    fig2.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    out_k = fig_dir / f"{tag}_kld"
    fig2.savefig(out_k.with_suffix(".svg"), facecolor=BG_OUTER)
    fig2.savefig(out_k.with_suffix(".png"), facecolor=BG_OUTER, dpi=160)
    plt.close(fig2)
    print(f"Saved {out_kp.name}.{{svg,png}} and {out_k.name}.{{svg,png}}")


def render_quant_time_chart(results_dir: Path, fig_dir: Path):
    _apply_dark_rc()
    loaded = []
    for tag, full_nm, short_nm in MODELS:
        p = results_dir / f"paper_{tag}.json"
        if p.exists():
            loaded.append((tag, full_nm, short_nm, json.loads(p.read_text(encoding="utf-8"))))
    if not loaded:
        return

    gpu_name = loaded[0][3].get("gpu_name", "Single GPU")
    bpw_targets = [2.5, 3.5, 4.5, 6.0]
    gguf_match = {2.5: "GGUF:IQ2_S", 3.5: "GGUF:IQ3_M", 4.5: "GGUF:Q4_K_S", 6.0: "GGUF:Q6_K"}

    fig, axes = plt.subplots(2, 3, figsize=(16.4, 8.8), dpi=160)
    fig.patch.set_facecolor(BG_OUTER)
    fig.suptitle(
        f"Quantization Creation Time & Speedup vs EXL3 on {gpu_name}",
        fontsize=13.6, fontweight="bold", color=TXT_MAIN, y=0.975,
    )

    x = np.arange(len(bpw_targets))
    w = 0.255

    # --- Top Row: Per-bitrate quantization time (minutes) + shared one-time calibration cap ---
    for col, (tag, full_nm, short_nm, d) in enumerate(loaded):
        ax = axes[0, col]
        ms = d["methods"]
        vtq_prep_m = float(d.get("vtq_shared_profile_fisher_sec", 0.0)) / 60.0
        gguf_im_m = float(d.get("gguf_imatrix_sec", 0.0)) / 60.0

        vtq_q = [ms[f"VTQ:{b:.1f}"]["quant_sec"] / 60.0 for b in bpw_targets]
        exl_q = [ms[f"EXL3:{b:.1f}"]["quant_sec"] / 60.0 for b in bpw_targets]
        ggf_q = [ms[gguf_match[b]]["quant_sec"] / 60.0 for b in bpw_targets]

        ax.bar(x - w, vtq_q, width=w, color=COL_VTQ, edgecolor=BG_OUTER, linewidth=0.9, zorder=3, label="VTQ quant")
        ax.bar(
            x - w, [vtq_prep_m] * 4, bottom=vtq_q, width=w,
            color=COL_VTQ, alpha=0.34, edgecolor=COL_VTQ, linewidth=0.8, hatch="///", zorder=3,
            label="VTQ shared Fisher+profile (1×)",
        )

        ax.bar(x, exl_q, width=w, color=COL_EXL3, edgecolor=BG_OUTER, linewidth=0.9, zorder=3, label="EXL3 convert.py")

        ax.bar(x + w, ggf_q, width=w, color=COL_GGUF, edgecolor=BG_OUTER, linewidth=0.9, zorder=3, label="GGUF llama-quantize")
        ax.bar(
            x + w, [gguf_im_m] * 4, bottom=ggf_q, width=w,
            color=COL_GGUF, alpha=0.34, edgecolor=COL_GGUF, linewidth=0.8, hatch="///", zorder=3,
            label="GGUF shared imatrix (1×)",
        )

        y_mult = 1.44 if col == 0 else 1.24
        y_top = max(max(exl_q), max(v + vtq_prep_m for v in vtq_q), max(g + gguf_im_m for g in ggf_q)) * y_mult
        ax.set_ylim(0, y_top)

        for i in range(len(bpw_targets)):
            sp = exl_q[i] / max(vtq_q[i], 1e-6)
            vtq_tot_i = vtq_q[i] + vtq_prep_m
            ax.text(
                x[i] - w, vtq_tot_i + y_top * 0.018,
                f"{sp:.1f}×", ha="center", va="bottom",
                fontsize=8.3, fontweight="bold", color=COL_VTQ,
            )
            ax.text(
                x[i], exl_q[i] + y_top * 0.015,
                f"{exl_q[i]:.1f}m", ha="center", va="bottom",
                fontsize=7.9, color=COL_EXL3,
            )

        ax.set_title(f"{full_nm} — Per-Bitrate Time", fontsize=11, fontweight="bold", color=TXT_MAIN, pad=7)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{b:.1f} bpw" for b in bpw_targets], fontsize=9.2)
        if col == 0:
            ax.set_ylabel("Wall-Clock Time (minutes, ↓ faster)", fontsize=9.8, color=TXT_SUB)
            handles = [
                mpatches.Patch(facecolor=COL_VTQ, edgecolor=BG_OUTER, label="VTQ (.vtq v2) quant"),
                mpatches.Patch(facecolor=COL_VTQ, alpha=0.38, edgecolor=COL_VTQ, hatch="///", label="+ shared Fisher/profile (1×)"),
                mpatches.Patch(facecolor=COL_EXL3, edgecolor=BG_OUTER, label="EXL3 (mul1) convert.py"),
                mpatches.Patch(facecolor=COL_GGUF, edgecolor=BG_OUTER, label="GGUF llama-quantize"),
                mpatches.Patch(facecolor=COL_GGUF, alpha=0.38, edgecolor=COL_GGUF, hatch="///", label="+ shared llama-imatrix (1×)"),
            ]
            leg = ax.legend(handles=handles, loc="upper right", frameon=True, facecolor="#12151e", edgecolor=SPINE_COL, fontsize=8.0, labelcolor=TXT_MAIN)
            leg.set_zorder(10)
        ax.grid(True, axis="y", linestyle="--", linewidth=0.7, color=GRID_COL, alpha=0.9)

    # --- Bottom-Left (0): Full 4-Bitrate Suite Total Time (2.5 + 3.5 + 4.5 + 6.0 + 1x shared prep) ---
    ax_tot = axes[1, 0]
    xm = np.arange(len(loaded))
    wm = 0.25
    vtq_suite = [
        (float(d.get("vtq_shared_profile_fisher_sec", 0.0)) + sum(d["methods"][f"VTQ:{b:.1f}"]["quant_sec"] for b in bpw_targets)) / 60.0
        for _, _, _, d in loaded
    ]
    exl_suite = [
        sum(d["methods"][f"EXL3:{b:.1f}"]["quant_sec"] for b in bpw_targets) / 60.0
        for _, _, _, d in loaded
    ]
    ggf_suite = [
        (float(d.get("gguf_imatrix_sec", 0.0)) + sum(d["methods"][gguf_match[b]]["quant_sec"] for b in bpw_targets)) / 60.0
        for _, _, _, d in loaded
    ]
    ax_tot.bar(xm - wm, vtq_suite, width=wm, color=COL_VTQ, edgecolor=BG_OUTER, zorder=3, label="VTQ (all 4 bitrates + 1× prep)")
    ax_tot.bar(xm, exl_suite, width=wm, color=COL_EXL3, edgecolor=BG_OUTER, zorder=3, label="EXL3 (all 4 bitrates)")
    ax_tot.bar(xm + wm, ggf_suite, width=wm, color=COL_GGUF, edgecolor=BG_OUTER, zorder=3, label="GGUF (4 presets + 1× imatrix)")
    ymax_s = max(exl_suite) * 1.20
    ax_tot.set_ylim(0, ymax_s)
    for i in range(len(loaded)):
        ax_tot.text(xm[i] - wm, vtq_suite[i] + ymax_s * 0.018, f"{vtq_suite[i]:.0f}m\n({exl_suite[i]/vtq_suite[i]:.1f}×)", ha="center", va="bottom", fontsize=8.1, fontweight="bold", color=COL_VTQ)
        ax_tot.text(xm[i], exl_suite[i] + ymax_s * 0.018, f"{exl_suite[i]:.0f}m", ha="center", va="bottom", fontsize=8.1, color=COL_EXL3)
        ax_tot.text(xm[i] + wm, ggf_suite[i] + ymax_s * 0.018, f"{ggf_suite[i]:.0f}m", ha="center", va="bottom", fontsize=8.1, color=COL_GGUF)
    ax_tot.set_xticks(xm)
    ax_tot.set_xticklabels([s.replace(" (", "\n(") for _, _, s, _ in loaded], fontsize=8.8)
    ax_tot.set_ylabel("Total Time for 4 Quants (minutes, ↓ faster)", fontsize=9.6, color=TXT_SUB)
    ax_tot.set_title("Full 4-Bitrate Suite Creation Time", fontsize=11, fontweight="bold", color=TXT_MAIN, pad=7)
    ax_tot.grid(True, axis="y", linestyle="--", linewidth=0.7, color=GRID_COL, alpha=0.9)

    # --- Bottom-Center (1): VTQ Speedup Factor over EXL3 across Bitrates & Architectures ---
    ax_sp = axes[1, 1]
    model_cols = ["#60a5fa", "#c084fc", "#34d399"]
    model_mk = ["o", "s", "D"]
    for idx_m, (tag, full_nm, short_nm, d) in enumerate(loaded):
        ms = d["methods"]
        sps_pure = [ms[f"EXL3:{b:.1f}"]["quant_sec"] / ms[f"VTQ:{b:.1f}"]["quant_sec"] for b in bpw_targets]
        ax_sp.plot(
            bpw_targets, sps_pure, label=short_nm, color=model_cols[idx_m], marker=model_mk[idx_m],
            markersize=6.5, markeredgecolor=BG_OUTER, markeredgewidth=1.1, linewidth=2.4, zorder=4,
        )
    ax_sp.axhline(1.0, color="#9ca3af", linestyle="--", linewidth=1.1, alpha=0.8, zorder=1)
    ax_sp.set_ylim(0.7, 5.6)
    ax_sp.set_xticks(bpw_targets)
    ax_sp.set_xticklabels([f"{b:.1f}" for b in bpw_targets])
    ax_sp.set_xlabel("Target Bitrate (total_bpw)", fontsize=9.8, color=TXT_SUB)
    ax_sp.set_ylabel("Speedup = T_EXL3 / T_VTQ (× faster, ↑ better)", fontsize=9.6, color=TXT_SUB)
    ax_sp.set_title("VTQ Quantization Speedup vs EXL3", fontsize=11, fontweight="bold", color=TXT_MAIN, pad=7)
    leg_sp = ax_sp.legend(loc="upper right", frameon=True, facecolor="#12151e", edgecolor=SPINE_COL, fontsize=8.8, labelcolor=TXT_MAIN)
    leg_sp.set_zorder(10)
    ax_sp.grid(True, linestyle="--", linewidth=0.7, color=GRID_COL, alpha=0.9)

    # --- Bottom-Right (2): Quality vs Quantization Time @ 2.5 & 3.5 bpw (3-Trace Mean KLD vs Standalone Time) ---
    ax_qt = axes[1, 2]
    ax_qt.set_yscale("log")
    # Plot Qwen3.6-35B and Gemma-4-26B (solid vs dashed) across the 4 bitrates as Time vs Mean KLD!
    for _, full_nm, short_nm, d in loaded[1:]:
        ms = d["methods"]
        ls = "-" if "Qwen" in short_nm else "--"
        al = 1.0 if "Qwen" in short_nm else 0.72
        vtq_p = float(d.get("vtq_shared_profile_fisher_sec", 0.0)) / 60.0
        ggf_p = float(d.get("gguf_imatrix_sec", 0.0)) / 60.0
        for fmt_lbl, col_c, mk_c, keys_l, add_p in [
            ("VTQ", COL_VTQ, "o", ["VTQ:2.5", "VTQ:3.5", "VTQ:4.5", "VTQ:6.0"], vtq_p),
            ("EXL3", COL_EXL3, "s", ["EXL3:2.5", "EXL3:3.5", "EXL3:4.5", "EXL3:6.0"], 0.0),
            ("GGUF", COL_GGUF, "D", ["GGUF:IQ2_S", "GGUF:IQ3_M", "GGUF:Q4_K_S", "GGUF:Q6_K"], ggf_p),
        ]:
            ts = [(ms[k]["quant_sec"] / 60.0) + add_p for k in keys_l]
            ks = [
                (ms[k]["nemotron"]["mean_kld"] + ms[k]["multilingual"]["mean_kld"] + ms[k]["wikitext2"]["mean_kld"]) / 3.0
                for k in keys_l
            ]
            lbl = f"{fmt_lbl} ({short_nm.split()[0]})"
            ax_qt.plot(
                ts, ks, label=lbl, color=col_c, linestyle=ls, alpha=al, marker=mk_c,
                markersize=5.8, markeredgecolor=BG_OUTER, markeredgewidth=1.0, linewidth=2.1, zorder=3,
            )
    ax_qt.set_ylim(0.0038, 1.35)
    ax_qt.set_xlabel("Standalone Creation Time incl. 1× Prep (minutes, ↓ faster)", fontsize=9.5, color=TXT_SUB)
    ax_qt.set_ylabel("3-Trace Mean KLD vs BF16 (log, ↓ better)", fontsize=9.6, color=TXT_SUB)
    ax_qt.set_title("Quality vs Creation Time (26B & 35B MoE)", fontsize=11, fontweight="bold", color=TXT_MAIN, pad=7)
    leg_qt = ax_qt.legend(loc="upper right", ncol=2, frameon=True, facecolor="#12151e", edgecolor=SPINE_COL, fontsize=7.6, labelcolor=TXT_MAIN)
    leg_qt.set_zorder(10)
    ax_qt.grid(True, which="major", linestyle="--", linewidth=0.7, color=GRID_COL, alpha=0.9)
    ax_qt.grid(True, which="minor", linestyle=":", linewidth=0.5, color=GRID_COL, alpha=0.55)

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.945))
    out_t = fig_dir / "paper_quant_time"
    fig.savefig(out_t.with_suffix(".svg"), facecolor=BG_OUTER)
    fig.savefig(out_t.with_suffix(".png"), facecolor=BG_OUTER, dpi=160)
    plt.close(fig)
    print(f"Saved {out_t.name}.{{svg,png}}")


def main():
    ap = argparse.ArgumentParser(description="Generate all dark-themed SVG/PNG benchmark figures from results/paper_*.json")
    ap.add_argument("--results-dir", type=Path, default=ROOT / "results")
    ap.add_argument("--fig-dir", type=Path, default=ROOT / "docs" / "figures")
    args = ap.parse_args()
    args.fig_dir.mkdir(parents=True, exist_ok=True)

    for tag, full_nm, _ in MODELS:
        jp = args.results_dir / f"paper_{tag}.json"
        if jp.exists():
            render_pareto_charts(jp, args.fig_dir, tag, full_nm)
    render_quant_time_chart(args.results_dir, args.fig_dir)


if __name__ == "__main__":
    main()
