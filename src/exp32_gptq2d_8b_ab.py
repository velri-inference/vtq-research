#!/usr/bin/env python3
"""
exp32: Controlled GPTQ-2D (full output Hessian G vs diag(G)) A/B test on Llama-3.1-8B-Instruct with VTQ.

Tests whether propagating rounding error across output rows via GPTQ-2D (arXiv:2607.27042v2)
improves VTQ's 16x16 tail-biting trellis without changing the .vtq format or decoder:
  - Phase 0: Numerical self-test (ldl_of_inverse equivalence, 1D L=I parity with tcq_ldlq_block256,
             2-sided proxy drop check).
  - Phase 1: Single-matrix controlled A/B on layers {0, 15, 31} mlp.down_proj (4096 x 14336)
             at VTQ_200 and VTQ_300: identical trellis, scales, input factor H, and W_target,
             swapping ONLY diag(G) (L=I) vs full G (L=ldl_of_inverse(G, size=16)), plus
             Schur-weighted trellis, (16, 256) live-u_sb Algorithm 6, 2-sided refit, and prod VTQ.
             Evaluates both 1D/2D quadratic proxy losses AND full-model KLD on WikiText-2,
             Nemotron, and Multilingual traces (paired bootstrap 95% CI).
  - Phase 2: Full 32-layer Llama-3.1-8B-Instruct end-to-end A/B at VTQ_200 (streaming GPTAQ alpha=0.5):
             compares prod VTQ, fixed-scale 1D control on all 64 (o_proj + down_proj),
             fixed-scale GPTQ-2D on all 64 (o_proj + down_proj), and GPTQ-2D + 2-sided refit
             on all 3 evaluation traces with paired bootstrap 95% CIs.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import src.liqk_v2 as lv2
import src.streaming_quant as sq
import src.vtq as vtq
from src.eval_kld import (
    evaluate_model_on_cached_trace,
    load_eval_trace,
    paired_bootstrap_diff,
    precompute_ref_trace_data,
)
from src.tcq_proto import viterbi_ub_cuda
from src.vtq_format import SUPER_ROWS, TILE, U_SB_UNIT, apply_vtq_rot
from src.vtq_v2 import DualCovAccumulator

HESS_REPO = "relaxml/Llama-3.1-8B-Instruct-Hessians-2Sided"


# =================================================================================================
# 1. YAQA Hessian Loader & Exact Block-LDL-of-Inverse (`size=16` / `size=256`)
# =================================================================================================

def flat_to_sym(v: torch.Tensor, n: int) -> torch.Tensor:
    """Reconstructs symmetric [n, n] matrix from lower-triangular flatH vector (relaxml / YAQA format)."""
    if v.ndim == 2 and v.shape == (n, n):
        return (v + v.T).mul_(0.5)
    a = torch.zeros((n, n), dtype=v.dtype, device=v.device)
    idxs = torch.tril_indices(n, n, device=v.device)
    a[idxs[0], idxs[1]] = v
    a[idxs[1], idxs[0]] = v
    return a


@torch.no_grad()
def load_yaqa_hessian(
    layer_idx: int,
    short_name: str,
    side: str,
    n: int,
    cache_dir: Path,
    device: torch.device,
    sigma_reg: float = 0.01,
) -> Tuple[torch.Tensor, float]:
    """Downloads & unpacks `{layer_idx}_{short_name}_h{side}.pt` into normalized damped float64 [n, n]."""
    fname = f"{layer_idx}_{short_name}_h{side}.pt"
    local_path = hf_hub_download(repo_id=HESS_REPO, filename=fname, local_dir=str(cache_dir))
    raw = torch.load(local_path, map_location="cpu", weights_only=True)
    if isinstance(raw, dict):
        if "flatH" in raw:
            v = raw["flatH"]
        elif "H" in raw:
            v = raw["H"]
        else:
            raise KeyError(f"Unexpected dict keys in {fname}: {list(raw.keys())}")
    else:
        v = raw
    h = flat_to_sym(v.to(device=device, dtype=torch.float64), n)
    h.div_(h.diagonal().mean().clamp(min=1e-12))
    # Match YAQA regularization: add sigma_reg until SPD
    fsr = 0.0
    step = float(sigma_reg)
    while True:
        h.diagonal().add_(step)
        fsr += step
        _, info = torch.linalg.cholesky_ex(h)
        if int(info.item()) == 0:
            break
        if fsr > 10.0:
            raise RuntimeError(f"Failed to make {fname} SPD even at sigma_reg={fsr}")
    return h, fsr


@torch.no_grad()
def ldl_of_inverse_with_prec(M: torch.Tensor, size: int = 16, use_fp64: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes unit block-lower-triangular L (`M^{-1} = L Lambda L^T` with `size x size` identity diagonal blocks),
    `prec_diag = diag(Lambda^{-1})`, and `scalar_dw = diag(Um)^2` (exact VTQ `dw_diag`) in one pass.
    Matches `gptq_2d_reference.ldl_of_inverse(M, size=size)`.
    """
    work_dt = torch.float64 if use_fp64 else torch.float32
    Mw = M.to(work_dt)
    n = Mw.shape[0]
    assert n % size == 0, f"Dimension {n} not divisible by super-entry size {size}"
    k = n // size
    # Reverse-order Cholesky: M = Um @ Um.T (Um upper triangular)
    Um = torch.linalg.cholesky(Mw.flip(dims=(0, 1))).flip(dims=(0, 1))
    del Mw
    scalar_dw = (Um.diagonal() ** 2).clamp(min=1e-12)
    eye = torch.eye(n, dtype=work_dt, device=M.device)
    L_raw = torch.linalg.solve_triangular(Um, eye, upper=True).T.contiguous()  # lower triangular = Um^{-T}
    del eye
    # j-th diagonal block of L_raw is D_j = Um_jj^{-T}, so D_j^{-1} = Um_jj^T and Lambda_j^{-1} = Um_jj @ Um_jj^T
    Um_bb = Um.reshape(k, size, k, size).diagonal(dim1=0, dim2=2).permute(2, 0, 1).contiguous()  # [k, size, size]
    del Um
    prec_diag = (Um_bb * Um_bb).sum(dim=-1).reshape(n)
    # Right-multiply each block-column j of L_raw by Um_bb[j].T via batched matmul [k, n, size] @ [k, size, size]
    L = torch.bmm(L_raw.reshape(n, k, size).transpose(0, 1), Um_bb.transpose(1, 2)).transpose(0, 1).reshape(n, n).contiguous()
    del L_raw, Um_bb
    # Enforce exact identity diagonal blocks and exact zero upper triangle
    L.tril_()
    L_blk = L.view(k, size, k, size)
    idx = torch.arange(k, device=M.device)
    L_blk[idx, :, idx, :] = torch.eye(size, dtype=work_dt, device=M.device)
    return L.float().contiguous(), prec_diag.float().contiguous(), scalar_dw.float().contiguous()


@torch.no_grad()
def proxy_losses(W_src: torch.Tensor, W_hat: torch.Tensor, H: torch.Tensor, G: torch.Tensor, use_fp64: bool = False) -> dict:
    """Computes exact 1-sided (I), diag(G)-weighted, and full 2-sided (G, H) relative quadratic losses."""
    dt = torch.float64 if use_fp64 else torch.float32
    dW = (W_src - W_hat).to(dt)
    Ww = W_src.to(dt)
    Hw = H.to(dt)
    Gw = G.to(dt)
    g_diag = Gw.diagonal()

    dWH = dW @ Hw
    WH = Ww @ Hw

    num_1d = float((dWH * dW).sum(dtype=torch.float64).item())
    den_1d = float((WH * Ww).sum(dtype=torch.float64).clamp(min=1e-12).item())

    row_d = (dWH * dW).sum(dim=1, dtype=torch.float64)
    row_w = (WH * Ww).sum(dim=1, dtype=torch.float64)
    num_dg = float((g_diag.double() * row_d).sum().item())
    den_dg = float((g_diag.double() * row_w).sum().clamp(min=1e-12).item())

    num_2d = float(((Gw @ dW) * dWH).sum(dtype=torch.float64).item())
    den_2d = float(((Gw @ Ww) * WH).sum(dtype=torch.float64).clamp(min=1e-12).item())

    return {
        "rel_1d": num_1d / den_1d,
        "rel_diagG": num_dg / den_dg,
        "rel_2d": num_2d / den_2d,
        "abs_2d": num_2d,
        "abs_diagG": num_dg,
    }


# =================================================================================================
# 2. VTQ Super-Entry GPTQ-2D Quantizer (16x16 Blocked Triton & 16x256 Nested Algorithm 6)
# =================================================================================================

def _import_gptq2d(gptq2d_dir: Path):
    if str(gptq2d_dir) not in sys.path:
        sys.path.insert(0, str(gptq2d_dir))
    import gptq_2d
    import gptq_2d_reference
    return gptq_2d, gptq_2d_reference


@torch.no_grad()
def _make_vtq_super16_callback(spec_name: str, dev: torch.device):
    cfg, tab, gains = vtq._get_spec_gpu_vtq(spec_name, dev)
    K = int(cfg["K"])
    U = 1 << (K - 1)
    nl = int(cfg["nl"])
    frac = int(cfg["frac"])
    pp = vtq._FRAC_TO_TCQ_PP[frac]
    m_extra = vtq._NL_TO_M_EXTRA[nl]
    G_cnt = int(gains.numel())
    n_cand = 4 if G_cnt > 4 else 0
    tb_win = int(os.environ.get("TCQ_TB_WIN", "32"))

    def vtq_q(y_flat: torch.Tensor, select, scale_flat: torch.Tensor, wts_flat: torch.Tensor, *, out: torch.Tensor):
        y_sub = select(y_flat)       # [d, 16, 16]
        sc_sub = select(scale_flat)  # [d, 16, 16]
        wt_sub = select(wts_flat)    # [d, 16, 16]
        x_tiles = (y_sub / sc_sub).reshape(-1, 256).contiguous()
        w_tiles = wt_sub.reshape(-1, 256).contiguous()
        rec_t = viterbi_ub_cuda(
            x_tiles, tab, K, 10, 359, gains, sub=16, wts=w_tiles,
            n_cand=n_cand, return_code=False, tail_biting=True, tb_window=tb_win,
            m_extra=m_extra, pair_parity=pp,
        )
        torch.mul(rec_t.view_as(y_sub), sc_sub, out=out)

    return vtq_q, cfg


@torch.no_grad()
def _refit_2sided(
    W_src: torch.Tensor,
    G_nat_all: torch.Tensor,
    d_row_all: torch.Tensor,
    u_sb_all: torch.Tensor,
    suh_fp16: torch.Tensor,
    H_f32: torch.Tensor,
    G_f32: torch.Tensor,
    rot_block: int,
    n_sweeps: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Monotone alternating closed-form `(d_row -> fp16, suh -> fp16)` refit under the EXACT
    two-sided objective `tr((W_src - W_hat)^T G (W_src - W_hat) H)`.
    """
    M, Kin = W_src.shape
    nblocks = Kin // TILE

    def recon(d_r, s_h):
        d_eff = d_r.float().unsqueeze(1) * (u_sb_all.float() * float(U_SB_UNIT))
        W_rot_h = (G_nat_all.view(M, nblocks, TILE) * d_eff.unsqueeze(-1)).reshape(M, Kin)
        U_base = apply_vtq_rot(W_rot_h, rot_block=rot_block, transpose=True)
        return U_base, U_base * s_h.float().unsqueeze(0)

    def loss_2d(W_h):
        dW = (W_src - W_h).to(torch.float64)
        return float(((G_f32.double() @ dW) * (dW @ H_f32.double())).sum().item())

    _, W_hat_eff = recon(d_row_all, suh_fp16)
    l_cur = loss_2d(W_hat_eff)
    HW = W_src @ H_f32

    for _ in range(n_sweeps):
        # 1) Joint optimal row scale multipliers c_row solving (G * (Q H Q^T)) c = diag(G W H Q^T)
        HQ = W_hat_eff @ H_f32
        A_row = (G_f32 * (HQ @ W_hat_eff.T)).double()
        b_row = (G_f32 * (HW @ W_hat_eff.T)).sum(dim=1).double()
        reg_r = 1e-3 * A_row.diagonal().mean().clamp(min=1e-12)
        A_row.diagonal().add_(reg_r)
        try:
            c_row = torch.linalg.solve(A_row, b_row).float()
        except RuntimeError:
            c_row = torch.ones((M,), dtype=torch.float32, device=W_src.device)
        c_row = torch.where(torch.isfinite(c_row) & (c_row > 0.5) & (c_row < 2.0), c_row, torch.ones_like(c_row))
        d_row_cand = (d_row_all.float() * c_row).to(torch.float16)

        # Check row step acceptance
        U_scaled, W_hat_r = recon(d_row_cand, suh_fp16)
        l_r = loss_2d(W_hat_r)
        if l_r < l_cur:
            d_row_all = d_row_cand
            W_hat_eff = W_hat_r
            l_cur = l_r
        else:
            U_scaled, _ = recon(d_row_all, suh_fp16)

        # 2) Joint optimal column scales suh solving ((U^T G U) * H) s = diag((U^T G W) H)
        U_w = G_f32 @ U_scaled
        A_in = (U_w.T @ U_scaled) * H_f32
        b_in = (U_w * HW).sum(dim=0)
        del U_w
        A_in.diagonal().add_(1e-3 * A_in.diagonal().mean().clamp(min=1e-12))
        try:
            suh_try = torch.linalg.solve(A_in.double(), b_in.double()).float().to(torch.float16)
        except RuntimeError:
            break
        del A_in
        valid_s = torch.isfinite(suh_try.float()) & (suh_try.float().abs() > 0.05) & (suh_try.float().abs() < 10.0)
        suh_try = torch.where(valid_s, suh_try, suh_fp16)
        W_hat_cand = U_scaled * suh_try.float().unsqueeze(0)
        l_new = loss_2d(W_hat_cand)
        if l_new < l_cur:
            suh_fp16 = suh_try
            W_hat_eff = W_hat_cand
            l_cur = l_new

    return d_row_all, suh_fp16, W_hat_eff


@torch.no_grad()
def _alg6_blk256_vtq(
    W_rot: torch.Tensor,
    L_G: torch.Tensor,
    U_256: torch.Tensor,
    U_blks: torch.Tensor,
    U_blks_inv: torch.Tensor,
    dw_diag: torch.Tensor,
    d_row_f32: torch.Tensor,
    step_row: torch.Tensor,
    spec_name: str,
    use_refit_u8: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Exact Algorithm 6 (nested block-column-major GPTQ-2D, Appendix B.1) over (16, 256) super-entries!
    Inside each [16, 256] super-entry, calls VTQ's native `ldlq_vtq_block256_cuda` (+ `_refit_block_u8`),
    then pushes exact error down via L_G (blocks of 16 rows) and right via U_256 (blocks of 256 cols).
    """
    dev = W_rot.device
    M, Kin = W_rot.shape
    nblocks = Kin // TILE
    mbar = M // SUPER_ROWS
    cfg, tab, gains = vtq._get_spec_gpu_vtq(spec_name, dev)
    K = int(cfg["K"])
    U_cnt = 1 << (K - 1)
    nl = int(cfg["nl"])
    frac = int(cfg["frac"])
    cfg_scale = float(cfg.get("scale", 1.0))

    Y = W_rot.clone()
    G_nat_all = torch.empty_like(W_rot)
    u_sb_all = torch.empty((M, nblocks), dtype=torch.uint8, device=dev)

    for jb in range(nblocks):
        c0, c1 = jb * TILE, (jb + 1) * TILE
        y_col_orig = Y[:, c0:c1].clone()
        y_col = y_col_orig.clone()
        U_blk = U_blks[jb]
        U_blk_inv = U_blks_inv[jb]
        dw_blk = dw_diag[jb]

        for ib in range(mbar):
            r0, r1 = ib * SUPER_ROWS, (ib + 1) * SUPER_ROWS
            y_tile = y_col[r0:r1].clone()
            u_init_f, d_init = vtq._initial_block_u8(y_tile, cfg_scale, step_row[r0:r1], spec_name)
            G_blk, _ = vtq.ldlq_vtq_block256_cuda(
                y_tile.clone(), d_init, dw_blk, U_blk, U_blk_inv, tab, gains, nl, U_cnt, frac=frac, return_code=False
            )
            if use_refit_u8:
                A_c = y_tile @ U_blk_inv
                B_c = G_blk @ U_blk_inv
                u_fin_f, d_fin = vtq._refit_block_u8(A_c, B_c, step_row[r0:r1], u_init_f, d_init)
            else:
                u_fin_f, d_fin = u_init_f, d_init
            z_tile = d_fin.unsqueeze(1) * G_blk
            G_nat_all[r0:r1, c0:c1] = G_blk
            u_sb_all[r0:r1, jb] = u_fin_f.to(torch.uint8)
            e_tile = z_tile - y_tile
            y_col[r0:r1] = z_tile
            if r1 < M:
                y_col[r1:] += L_G[r1:, r0:r1] @ e_tile

        # Algorithm 6 Line 10: propagate full column correction (y_col - y_col_orig) = L_G @ E_col right via U_256
        if c1 < Kin:
            Y[:, c1:] += (y_col - y_col_orig) @ U_256[c0:c1, c1:]

    return G_nat_all, u_sb_all


@torch.no_grad()
def quantize_matrix_vtq_mode(
    W: torch.Tensor,
    H: torch.Tensor,
    spec_name: str,
    mode: str,
    *,
    G_out: Optional[torch.Tensor] = None,
    L_G_cached: Optional[torch.Tensor] = None,
    row_prec_cached: Optional[torch.Tensor] = None,
    seed: int = 1337,
    damp: float = 0.20,
    row_weights: Optional[torch.Tensor] = None,
    rot_block: int = -1,
    W_target: Optional[torch.Tensor] = None,
    gptq2d_mod=None,
    use_triton: bool = True,
    reverse_order: bool = False,
) -> Tuple[torch.Tensor, float]:
    """
    Runs one of the controlled modes on matrix W [M, Kin] and returns (W_hat [M, Kin], elapsed_sec).
    Modes:
      - `vtq_prod`         : unmodified production `vtq.quantize_matrix_vtq(..., u_sb_gecd_sweeps=1)`
      - `vtq_prod_nogs`    : unmodified `vtq.quantize_matrix_vtq(..., u_sb_gecd_sweeps=0)` with refit=1
      - `static_1d`        : GPTQ-2D engine with L=I, static u_sb, w=dw_diag, no refit/GS
      - `static_1d_w`      : GPTQ-2D engine with L=I, static u_sb, w=diag(G)*d^2*col_prec, no refit/GS
      - `static_2d`        : GPTQ-2D engine with L=L_G, static u_sb, w=dw_diag, no refit/GS  [pure Dot A/B]
      - `static_2d_w`      : GPTQ-2D engine with L=L_G, static u_sb, w=row_prec*d^2*col_prec, no refit/GS
      - `static_1d_refit1d`: `static_1d` + 1-sided (d_row, suh) refit (matching prod VTQ refit)
      - `static_2d_refit2d`: `static_2d` + exact 2-sided G-weighted (d_row, suh) refit
      - `static_2d_w_refit2d`: `static_2d_w` + exact 2-sided G-weighted (d_row, suh) refit
      - `blk256_1d`        : Algorithm 6 (16, 256) with L=I + live `_initial_block_u8` & `_refit_block_u8`
      - `blk256_2d`        : Algorithm 6 (16, 256) with L=L_G + live `_initial_block_u8` & `_refit_block_u8`
      - `blk256_2d_refit2d`: `blk256_2d` + exact 2-sided G-weighted (d_row, suh) refit
    """
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    if mode == "vtq_prod":
        _, w_hat, _ = vtq.quantize_matrix_vtq(
            W, H, spec_name, seed=seed, damp=damp, row_weights=row_weights,
            rot_block=rot_block, W_target=W_target, u_sb_gecd_sweeps=1,
        )
        torch.cuda.synchronize()
        return w_hat, time.perf_counter() - t0

    if mode == "vtq_prod_nogs":
        _, w_hat, _ = vtq.quantize_matrix_vtq(
            W, H, spec_name, seed=seed, damp=damp, row_weights=row_weights,
            rot_block=rot_block, W_target=W_target, u_sb_gecd_sweeps=0,
        )
        torch.cuda.synchronize()
        return w_hat, time.perf_counter() - t0

    dev = W.device
    W_src = (W_target if W_target is not None else W).float()
    H_f32 = H.float()
    M, Kin = W_src.shape
    assert M % SUPER_ROWS == 0 and Kin % TILE == 0
    nblocks = Kin // TILE

    # Identical VTQ suh + Outer-DCT preconditioning
    W_rot, H_rot, suh = vtq.compute_suh_vtq(W_src, H_f32, seed=seed, row_weights=row_weights, rot_block=rot_block)
    dmean = H_rot.diagonal().mean().clamp(min=1e-8)
    H_reg = H_rot.clone()
    H_reg.diagonal().add_(damp * dmean)

    # Identical initial d_row & static u_sb
    cfg = vtq.VTQ_CONFIG[spec_name]
    cfg_scale = float(cfg.get("scale", 1.0))
    rms_rb = vtq._initial_tile_scales(W_rot, cfg_scale, spec_name)  # [M, nblocks]
    d_row_fp16 = (rms_rb.max(dim=1).values * (128.0 / 224.0)).clamp(min=1e-7).to(torch.float16)
    d_row_f32 = d_row_fp16.float().clamp(min=1e-8)
    step_row = (d_row_f32 * float(U_SB_UNIT)).clamp(min=1e-12)
    u_sb_static_f = torch.round(rms_rb / step_row.unsqueeze(1)).clamp(1.0, 255.0)
    d_eff_static = (step_row.unsqueeze(1) * u_sb_static_f).clamp(min=1e-10)  # [M, nblocks]

    is_2d = ("2d" in mode)
    if is_2d:
        assert G_out is not None or L_G_cached is not None
        if L_G_cached is not None and not reverse_order:
            L_G, row_prec = L_G_cached, row_prec_cached
        else:
            G_in = G_out
            if reverse_order:
                # Reverse coarse 16-row block order (preserving intra-tile row order for VTQ trellis)
                mbar = M // SUPER_ROWS
                idx_r = (torch.arange(mbar - 1, -1, -1, device=dev).unsqueeze(1) * SUPER_ROWS +
                         torch.arange(SUPER_ROWS, device=dev).unsqueeze(0)).reshape(-1)
                G_in = G_out[idx_r][:, idx_r]
            L_G, row_prec, _ = ldl_of_inverse_with_prec(G_in, size=SUPER_ROWS)
    else:
        L_G = torch.eye(M, dtype=torch.float32, device=dev)
        row_prec = G_out.diagonal().float().clamp(min=1e-8) if G_out is not None else torch.ones(M, device=dev)

    if mode.startswith("blk256_"):
        # (16, 256) super-entries via Algorithm 6 + native tcq_ldlq_block256
        assert not reverse_order, "reverse_order only supported for 16x16 modes"
        H_flip = H_reg.flip(0, 1)
        L_flip = torch.linalg.cholesky(H_flip)
        U_inv_flip = torch.linalg.solve_triangular(L_flip, torch.eye(Kin, dtype=L_flip.dtype, device=dev), upper=False)
        U_nat = U_inv_flip.flip(0, 1)
        eye256 = torch.eye(TILE, dtype=torch.float32, device=dev).unsqueeze(0).expand(nblocks, -1, -1)
        U_blks = torch.stack([U_nat[b * TILE:(b + 1) * TILE, b * TILE:(b + 1) * TILE] for b in range(nblocks)], dim=0)
        U_blks_inv = torch.linalg.solve_triangular(U_blks, eye256, upper=True)
        dw_diag = (1.0 / (torch.diagonal(U_nat) ** 2).clamp(min=1e-12)).view(nblocks, TILE).contiguous()
        U_256, _, _ = ldl_of_inverse_with_prec(H_reg, size=TILE)
        U_256 = U_256.T.contiguous()

        G_nat_all, u_sb_all = _alg6_blk256_vtq(
            W_rot, L_G, U_256, U_blks, U_blks_inv, dw_diag, d_row_f32, step_row, spec_name, use_refit_u8=True
        )
    else:
        # (16, 16) super-entries via IST-DASLab GPTQ-2D blocked Triton engine
        assert gptq2d_mod is not None
        if reverse_order:
            mbar, nbar = M // SUPER_ROWS, Kin // 16
            idx_r = (torch.arange(mbar - 1, -1, -1, device=dev).unsqueeze(1) * SUPER_ROWS +
                     torch.arange(SUPER_ROWS, device=dev).unsqueeze(0)).reshape(-1)
            idx_c = (torch.arange(nbar - 1, -1, -1, device=dev).unsqueeze(1) * 16 +
                     torch.arange(16, device=dev).unsqueeze(0)).reshape(-1)
            W_run = W_rot[idx_r][:, idx_c].contiguous()
            H_run = H_reg[idx_c][:, idx_c].contiguous()
            d_eff_full = d_eff_static.repeat_interleave(TILE, dim=1)[idx_r][:, idx_c].contiguous()
        else:
            W_run = W_rot
            H_run = H_reg
            d_eff_full = d_eff_static.repeat_interleave(TILE, dim=1).contiguous()

        L_H, col_prec, dw_col = ldl_of_inverse_with_prec(H_run, size=16, use_fp64=False)
        U_16 = L_H.T.contiguous()
        del L_H

        # Match exact VTQ dw_diag = 1 / diag(U_nat)^2 for pure `static_1d` / `static_2d`,
        # and exact Schur precision `row_prec * d^2 * col_prec` for `_w` variants.
        if "_w" in mode:
            rp = row_prec / row_prec.mean().clamp(min=1e-12)
            cp = col_prec / col_prec.mean().clamp(min=1e-12)
            wts_2d = (rp.unsqueeze(1) * (d_eff_full * d_eff_full) * cp.unsqueeze(0)).clamp(min=1e-12)
            wts_2d = wts_2d / wts_2d.mean().clamp(min=1e-12)
        else:
            wts_2d = dw_col.unsqueeze(0).expand(M, Kin).contiguous()
        del H_rot, H_reg, H_run

        vtq_q, _ = _make_vtq_super16_callback(spec_name, dev)
        sweep_fn = gptq2d_mod.gptq_2d_super_entry if use_triton else gptq2d_mod.gptq_2d_super_entry_torch
        kw_blk = dict(inner_block_size=8, outer_block_size=64) if use_triton else {}
        Z_run = sweep_fn(
            W_run, L_G, U_16,
            quantizer=vtq_q,
            super_entry_shape=(16, 16),
            quantizer_inputs=(d_eff_full.reshape(-1), wts_2d.reshape(-1)),
            **kw_blk,
        )
        del U_16, wts_2d
        if reverse_order:
            inv_r = torch.empty_like(idx_r); inv_r[idx_r] = torch.arange(M, device=dev)
            inv_c = torch.empty_like(idx_c); inv_c[idx_c] = torch.arange(Kin, device=dev)
            Z_rot = Z_run[inv_r][:, inv_c].contiguous()
            d_eff_orig = d_eff_static.repeat_interleave(TILE, dim=1)
        else:
            Z_rot = Z_run
            d_eff_orig = d_eff_full

        G_nat_all = Z_rot / d_eff_orig
        u_sb_all = u_sb_static_f.to(torch.uint8)

    suh_fp16 = suh.to(torch.float16)
    if mode.endswith("_refit2d"):
        assert G_out is not None
        d_row_fp16, suh_fp16, W_hat_eff = _refit_2sided(
            W_src, G_nat_all, d_row_fp16, u_sb_all, suh_fp16, H_f32, G_out.float(), rot_block=rot_block, n_sweeps=1
        )
    elif mode.endswith("_refit1d"):
        # Standard 1-sided (diag(row_weights)) VTQ refit
        d_eff_b = d_row_fp16.float().unsqueeze(1) * (u_sb_all.float() * float(U_SB_UNIT))
        W_rot_h = (G_nat_all.view(M, nblocks, TILE) * d_eff_b.unsqueeze(-1)).reshape(M, Kin)
        W_hat_eff = apply_vtq_rot(W_rot_h, rot_block=rot_block, transpose=True) * suh_fp16.float().unsqueeze(0)
        HW = W_src @ H_f32
        HQ = W_hat_eff @ H_f32
        rw = None
        if row_weights is not None:
            rw = row_weights.float().clamp(min=float(row_weights.mean().item()) * 0.1)
            rw = (rw / rw.mean().clamp(min=1e-12)).clamp(0.25, 4.0)
        c_row = (W_hat_eff * HW).sum(dim=1) / (W_hat_eff * HQ).sum(dim=1).clamp(min=1e-12)
        c_row = torch.where(torch.isfinite(c_row) & (c_row > 0.5) & (c_row < 2.0), c_row, torch.ones_like(c_row))
        d_row_cand = (d_row_fp16.float() * c_row).to(torch.float16)
        d_eff_c = d_row_cand.float().unsqueeze(1) * (u_sb_all.float() * float(U_SB_UNIT))
        U_sc = apply_vtq_rot((G_nat_all.view(M, nblocks, TILE) * d_eff_c.unsqueeze(-1)).reshape(M, Kin), rot_block=rot_block, transpose=True)
        U_w = (U_sc * rw.unsqueeze(1)) if rw is not None else U_sc
        A_in = (U_w.T @ U_sc) * H_f32
        b_in = (U_w * HW).sum(dim=0)
        A_in.diagonal().add_(1e-3 * A_in.diagonal().mean().clamp(min=1e-12))
        suh_try = torch.linalg.solve(A_in, b_in).to(torch.float16)
        valid_s = torch.isfinite(suh_try.float()) & (suh_try.float().abs() > 0.05) & (suh_try.float().abs() < 10.0)
        suh_try = torch.where(valid_s, suh_try, suh_fp16)
        W_hat_cand = U_sc * suh_try.float().unsqueeze(0)
        HQ_cand = W_hat_cand @ H_f32
        l_old = float((((HW - HQ) * (W_src - W_hat_eff)).sum(dim=1) * (rw if rw is not None else 1.0)).sum())
        l_new = float((((HW - HQ_cand) * (W_src - W_hat_cand)).sum(dim=1) * (rw if rw is not None else 1.0)).sum())
        if l_new < l_old:
            W_hat_eff = W_hat_cand
    else:
        d_eff_b = d_row_fp16.float().unsqueeze(1) * (u_sb_all.float() * float(U_SB_UNIT))
        W_rot_h = (G_nat_all.view(M, nblocks, TILE) * d_eff_b.unsqueeze(-1)).reshape(M, Kin)
        W_hat_eff = apply_vtq_rot(W_rot_h, rot_block=rot_block, transpose=True) * suh_fp16.float().unsqueeze(0)

    torch.cuda.synchronize()
    return W_hat_eff, time.perf_counter() - t0


# =================================================================================================
# 3. Calibration & Evaluation Traces (exact neutral WikiText-2 cal + 3 held-out traces)
# =================================================================================================

def build_cal_and_eval_traces(model_dir: Path, data_dir: Path, n_cal: int = 64, cal_len: int = 2048):
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    # Neutral WikiText-2 train calibration (same as paper_llama31_8b)
    from datasets import load_dataset
    ds_train = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    train_ids = tok.encode("\n".join(ds_train["text"]), add_special_tokens=False)
    cal_rows = [train_ids[i * cal_len : (i + 1) * cal_len] for i in range(n_cal)]
    cal_tokens = torch.tensor(cal_rows, dtype=torch.long)

    # 3 held-out evaluation traces: WikiText-2 test (40x448), Nemotron (40 seqs), Multilingual (52 seqs)
    wt2_raw = (data_dir / "wikitext2_test.raw").read_text(encoding="utf-8")
    test_ids = tok.encode(wt2_raw, add_special_tokens=False)
    wt2_seqs = []
    for i in range(40):
        chunk = test_ids[i * 448 : (i + 1) * 448]
        wt2_seqs.append({"id": f"wt2_{i}", "token_ids": chunk, "score_spans": [[1, len(chunk)]]})

    traces = {
        "wikitext2": wt2_seqs,
        "nemotron": load_eval_trace(data_dir / "nemotron_trace.json"),
        "multilingual": load_eval_trace(data_dir / "multilingual_trace.json"),
    }
    return cal_tokens, traces


@torch.no_grad()
def collect_single_layer_down_hessians(
    model: nn.Module,
    cal_tokens: torch.Tensor,
    target_layers: List[int],
    batch_size: int = 4,
    low_vram: bool = False,
) -> Dict[int, torch.Tensor]:
    """Runs BF16 teacher forward on cal_tokens and collects exact input Hessians H for target down_proj layers."""
    dev = torch.device("cuda")
    n_seqs, seq_len = cal_tokens.shape
    max_l = max(target_layers)
    if low_vram:
        model.to("cpu")
        torch.cuda.empty_cache()
        model.model.embed_tokens.to(dev)
        model.model.rotary_emb.to(dev)
        hs = [model.model.embed_tokens(cal_tokens[i : i + batch_size].to(dev)).cpu() for i in range(0, n_seqs, batch_size)]
        pos_emb = model.model.rotary_emb(hs[0][:1].to(dev), torch.arange(seq_len, device=dev).unsqueeze(0))
        model.model.embed_tokens.to("cpu")
        model.model.rotary_emb.to("cpu")
    else:
        hs = [model.model.embed_tokens(cal_tokens[i : i + batch_size].to(dev)) for i in range(0, n_seqs, batch_size)]
        pos_emb = model.model.rotary_emb(hs[0][:1], torch.arange(seq_len, device=dev).unsqueeze(0))
    out_H: Dict[int, torch.Tensor] = {}

    for l_idx in range(max_l + 1):
        layer = model.model.layers[l_idx]
        if low_vram:
            layer.to(dev)
        if l_idx in target_layers:
            acc = DualCovAccumulator(need_cross=False)
            hk = layer.mlp.down_proj.register_forward_pre_hook(lambda m, a, ac=acc: ac.add(a[0]))
            for bi in range(len(hs)):
                inp = hs[bi].to(dev) if low_vram else hs[bi]
                o = layer(inp, position_embeddings=pos_emb)
                o = o[0] if isinstance(o, tuple) else o
                hs[bi] = o.cpu() if low_vram else o
            hk.remove()
            H_down, _ = acc.get(gptaq_damp=0.20)
            out_H[l_idx] = H_down.cpu()
            del acc
        else:
            for bi in range(len(hs)):
                inp = hs[bi].to(dev) if low_vram else hs[bi]
                o = layer(inp, position_embeddings=pos_emb)
                o = o[0] if isinstance(o, tuple) else o
                hs[bi] = o.cpu() if low_vram else o
        if low_vram:
            layer.to("cpu")
            torch.cuda.empty_cache()
    del hs
    torch.cuda.empty_cache()
    return out_H


def compute_fisher_offload_llama(
    model: nn.Module,
    cal_tokens: torch.Tensor,
    n_seqs: int = 8,
    seq_len: int = 1024,
) -> Dict[Tuple[int, str], torch.Tensor]:
    """Exact layer-offloaded per-row Fisher (matches `src/fisher_offload.py`, ~2.5 GB peak VRAM)."""
    import torch.nn.functional as F
    dev = torch.device("cuda")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    model.to("cpu")
    torch.cuda.empty_cache()
    inner = model.model
    inner.embed_tokens.to(dev)
    inner.norm.to(dev)
    inner.rotary_emb.to(dev)
    model.lm_head.to(dev)
    layers = inner.layers
    n_layers = len(layers)
    fisher = {
        (l, sub): torch.zeros(layers[l].get_submodule(sub).out_features, dtype=torch.float64, device=dev)
        for l in range(n_layers)
        for sub in lv2.LINEAR_SUBMODULES
    }
    n = min(n_seqs, cal_tokens.shape[0])
    toks = [cal_tokens[i : i + 1, :seq_len].to(dev) for i in range(n)]
    total = sum(t.numel() for t in toks)
    pos = inner.rotary_emb(torch.zeros((1, seq_len, model.config.hidden_size), device=dev, dtype=torch.bfloat16),
                           torch.arange(seq_len, device=dev).unsqueeze(0))
    with torch.no_grad():
        hs = [inner.embed_tokens(t) for t in toks]
        inputs = []
        for l in range(n_layers):
            inputs.append([h.cpu() for h in hs])
            layers[l].to(dev)
            hs_next = []
            for h in hs:
                o = layers[l](h, position_embeddings=pos)
                hs_next.append(o[0] if isinstance(o, tuple) else o)
            hs = hs_next
            layers[l].to("cpu")
    gs = []
    for i, h in enumerate(hs):
        h = h.detach().requires_grad_(True)
        with torch.enable_grad():
            logits = model.lm_head(inner.norm(h)).reshape(-1, model.config.vocab_size)
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
            gs.append(torch.autograd.grad(loss, h)[0].detach().cpu())
        del logits, loss
    del hs
    inner.embed_tokens.to("cpu")
    inner.norm.to("cpu")
    model.lm_head.to("cpu")
    torch.cuda.empty_cache()

    for l in reversed(range(n_layers)):
        layer = layers[l].to(dev)
        hooks = []
        for sub in lv2.LINEAR_SUBMODULES:
            def fwd(m, inp, out, key=(l, sub)):
                def acc(gr, key=key):
                    fisher[key].add_(gr.detach().float().square().sum(dim=(0, 1)).double())
                out.register_hook(acc)
            hooks.append(layer.get_submodule(sub).register_forward_hook(fwd))
        next_gs = []
        for x_cpu, g_cpu in zip(inputs[l], gs):
            x = x_cpu.to(dev).detach().requires_grad_(True)
            g = g_cpu.to(dev)
            with torch.enable_grad():
                o = layer(x, position_embeddings=pos)
                o = o[0] if isinstance(o, tuple) else o
                gx = torch.autograd.grad(o, x, g)[0]
            next_gs.append(gx.detach().cpu())
            del x, g, o, gx
        for hk in hooks:
            hk.remove()
        gs = next_gs
        inputs[l] = None
        layer.to("cpu")
    inner.rotary_emb.to("cpu")
    torch.cuda.empty_cache()
    return {k: (v / float(total)).float().cpu() for k, v in fisher.items()}


def compute_own_g_odown_offload(
    model_path: str,
    cal_tokens: torch.Tensor,
    out_dir: Path,
    n_seqs: int = 32,
    seq_len: int = 1024,
) -> float:
    """
    Computes on-the-fly full M x M (4096 x 4096) empirical Fisher output Hessians G = E[g_y g_y^T]
    ONLY for `self_attn.o_proj` and `mlp.down_proj` across all 32 layers via layer-offloaded backprop
    (~2.5 GB peak VRAM, ~25-40s total, zero external YAQA checkpoints!).
    Saves upper-normalized symmetric FP32 matrices into `out_dir/{l}_{o,down}_ownG.pt`.
    """
    import torch.nn.functional as F
    out_dir.mkdir(parents=True, exist_ok=True)
    # Skip if all 64 files already exist
    if all((out_dir / f"{l}_{sn}_ownG.pt").exists() for l in range(32) for sn in ("o", "down")):
        return 0.0

    t0 = time.perf_counter()
    dev = torch.device("cuda")
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, attn_implementation="sdpa").eval()
    for p in model.parameters():
        p.requires_grad_(False)
    inner = model.model
    inner.embed_tokens.to(dev)
    inner.norm.to(dev)
    inner.rotary_emb.to(dev)
    model.lm_head.to(dev)
    layers = inner.layers
    n_layers = len(layers)
    n = min(n_seqs, cal_tokens.shape[0])
    toks = [cal_tokens[i : i + 1, :seq_len].to(dev) for i in range(n)]
    pos = inner.rotary_emb(
        torch.zeros((1, seq_len, model.config.hidden_size), device=dev, dtype=torch.bfloat16),
        torch.arange(seq_len, device=dev).unsqueeze(0),
    )

    with torch.no_grad():
        hs = [inner.embed_tokens(t) for t in toks]
        inputs = []
        for l in range(n_layers):
            inputs.append([h.cpu() for h in hs])
            layers[l].to(dev)
            hs_next = []
            for h in hs:
                o = layers[l](h, position_embeddings=pos)
                hs_next.append(o[0] if isinstance(o, tuple) else o)
            hs = hs_next
            layers[l].to("cpu")

    gs = []
    for i, h in enumerate(hs):
        h = h.detach().requires_grad_(True)
        with torch.enable_grad():
            logits = model.lm_head(inner.norm(h)).reshape(-1, model.config.vocab_size)
            with torch.no_grad():
                probs = F.softmax(logits.float(), dim=-1)
                u = torch.rand((probs.shape[0], 2), generator=torch.Generator().manual_seed(2000 + i)).to(dev)
                cdf = probs.cumsum(dim=-1)
                y = torch.searchsorted(cdf, (u * cdf[:, -1:]).contiguous()).clamp(max=probs.shape[-1] - 1)
                del probs, cdf
            loss = 0.5 * (
                F.cross_entropy(logits.float(), y[:, 0], reduction="sum")
                + F.cross_entropy(logits.float(), y[:, 1], reduction="sum")
            )
            gs.append(torch.autograd.grad(loss, h)[0].detach().cpu())
        del logits, loss
    del hs
    inner.embed_tokens.to("cpu")
    inner.norm.to("cpu")
    model.lm_head.to("cpu")
    torch.cuda.empty_cache()

    for l in reversed(range(n_layers)):
        layer = layers[l].to(dev)
        M_dim = layer.mlp.down_proj.out_features
        G_acc = {
            "o": torch.zeros((M_dim, M_dim), dtype=torch.float64, device=dev),
            "down": torch.zeros((M_dim, M_dim), dtype=torch.float64, device=dev),
        }
        hooks = []
        for sub, sn in (("self_attn.o_proj", "o"), ("mlp.down_proj", "down")):
            def fwd(m, inp, out, sn=sn):
                def acc(gr, sn=sn):
                    g2 = gr.detach().float().reshape(-1, gr.shape[-1])
                    G_acc[sn].add_((g2.T @ g2).double())
                out.register_hook(acc)
            hooks.append(layer.get_submodule(sub).register_forward_hook(fwd))
        next_gs = []
        for x_cpu, g_cpu in zip(inputs[l], gs):
            x = x_cpu.to(dev).detach().requires_grad_(True)
            g = g_cpu.to(dev)
            with torch.enable_grad():
                o = layer(x, position_embeddings=pos)
                o = o[0] if isinstance(o, tuple) else o
                gx = torch.autograd.grad(o, x, g)[0]
            next_gs.append(gx.detach().cpu())
            del x, g, o, gx
        for hk in hooks:
            hk.remove()
        gs = next_gs
        inputs[l] = None
        layer.to("cpu")
        for sn in ("o", "down"):
            Gm = G_acc[sn]
            Gm = 0.5 * (Gm + Gm.T)
            Gm.div_(Gm.diagonal().mean().clamp(min=1e-12))
            torch.save(Gm.float().cpu(), out_dir / f"{l}_{sn}_ownG.pt")
        del G_acc
    del model, gs, inputs
    gc.collect()
    torch.cuda.empty_cache()
    elapsed = time.perf_counter() - t0
    print(f"  Computed all 64 on-the-fly ownG matrices ({n}x{seq_len} tokens) in {elapsed:.1f}s!", flush=True)
    return elapsed


# =================================================================================================
# 4. Main Benchmark Runner (Phase 0 -> Phase 1 -> Phase 2)
# =================================================================================================

@torch.no_grad()
def run_phase0_selftest(gptq2d_mod, gptq2d_ref, dev: torch.device):
    print("\n=== PHASE 0: Exact Numerical Self-Tests ===", flush=True)
    torch.manual_seed(0)
    # 1) Verify our ldl_of_inverse_with_prec matches gptq_2d_reference.ldl_of_inverse to ~1e-6
    A = torch.randn(128, 128, dtype=torch.float64, device=dev)
    M = A.T @ A / 128.0
    M.diagonal().add_(0.05)
    L_ours, prec_ours, _ = ldl_of_inverse_with_prec(M, size=16)
    L_ref = gptq2d_ref.ldl_of_inverse(M, size=16).float()
    max_diff_L = float((L_ours - L_ref).abs().max().item())
    print(f"  [check 1] ldl_of_inverse(128, size=16) max_diff = {max_diff_L:.3e}", flush=True)
    assert max_diff_L < 1e-5, f"ldl_of_inverse mismatch: {max_diff_L}"

    # 2) Verify Triton gptq_2d_super_entry works on Blackwell GPU and matches PyTorch reference
    G = M.float()
    B = torch.randn(256, 256, dtype=torch.float64, device=dev)
    H = (B.T @ B / 256.0)
    H.diagonal().add_(0.05)
    L_g, _, _ = ldl_of_inverse_with_prec(G, size=16)
    U_h, _, _ = ldl_of_inverse_with_prec(H, size=16)
    U_h = U_h.T.contiguous()
    X = torch.randn(128, 256, dtype=torch.float32, device=dev)
    sc = torch.full((128 * 256,), 0.5, dtype=torch.float32, device=dev)
    wt = torch.ones((128 * 256,), dtype=torch.float32, device=dev)
    vtq_q, _ = _make_vtq_super16_callback("VTQ_200", dev)

    use_triton = True
    try:
        Z_tri = gptq2d_mod.gptq_2d_super_entry(
            X, L_g, U_h, quantizer=vtq_q, super_entry_shape=(16, 16),
            inner_block_size=8, outer_block_size=64, quantizer_inputs=(sc, wt),
        )
        Z_pt = gptq2d_mod.gptq_2d_super_entry_torch(
            X, L_g, U_h, quantizer=vtq_q, super_entry_shape=(16, 16), quantizer_inputs=(sc, wt),
        )
        rel_diff = float((Z_tri - Z_pt).norm() / Z_pt.norm().clamp(min=1e-12))
        print(f"  [check 2] Triton vs PyTorch gptq_2d_super_entry (VTQ_200, 128x256) rel_diff = {rel_diff:.3e}", flush=True)
    except Exception as e:
        print(f"  [check 2] Triton failed ({e!r}), falling back to PyTorch!", flush=True)
        use_triton = False
        Z_tri = gptq2d_mod.gptq_2d_super_entry_torch(
            X, L_g, U_h, quantizer=vtq_q, super_entry_shape=(16, 16), quantizer_inputs=(sc, wt),
        )

    # 3) Verify 2-sided objective drops vs 1-sided (L=I) on synthetic correlated G, H
    Z_1d = (gptq2d_mod.gptq_2d_super_entry if use_triton else gptq2d_mod.gptq_2d_super_entry_torch)(
        X, torch.eye(128, device=dev), U_h, quantizer=vtq_q, super_entry_shape=(16, 16),
        quantizer_inputs=(sc, wt), **({"inner_block_size": 8, "outer_block_size": 64} if use_triton else {}),
    )
    pl_1d = proxy_losses(X, Z_1d, H.float(), G)
    pl_2d = proxy_losses(X, Z_tri, H.float(), G)
    print(f"  [check 3] Synthetic 128x256 2D proxy: 1D={pl_1d['rel_2d']:.5f} -> 2D={pl_2d['rel_2d']:.5f} "
          f"({100.0*(pl_2d['rel_2d']-pl_1d['rel_2d'])/pl_1d['rel_2d']:+.2f}%)", flush=True)
    assert pl_2d["rel_2d"] < pl_1d["rel_2d"], "GPTQ-2D failed to reduce synthetic 2D objective!"

    # 4) Verify L=I GPTQ-2D matches VTQ's native tcq_ldlq_block256 on 128x256 (exact same scales & dw_diag!)
    cfg, tab, gains = vtq._get_spec_gpu_vtq("VTQ_200", dev)
    L_f = torch.linalg.cholesky(H.flip(0, 1))
    U_nat = torch.linalg.solve_triangular(L_f, torch.eye(256, dtype=torch.float64, device=dev), upper=False).flip(0, 1).float()
    U_inv = torch.linalg.solve_triangular(U_nat, torch.eye(256, dtype=torch.float32, device=dev), upper=True)
    dw_blk = (1.0 / (torch.diagonal(U_nat) ** 2).clamp(min=1e-12)).contiguous()
    d_fix = torch.full((128,), 0.5, dtype=torch.float32, device=dev)
    G_nat_cuda, _ = vtq.ldlq_vtq_block256_cuda(
        X.clone(), d_fix, dw_blk, U_nat, U_inv, tab, gains, int(cfg["nl"]), 1 << (int(cfg["K"]) - 1), frac=int(cfg["frac"]), return_code=False
    )
    Z_cuda = G_nat_cuda * 0.5
    Z_1d_dw = (gptq2d_mod.gptq_2d_super_entry if use_triton else gptq2d_mod.gptq_2d_super_entry_torch)(
        X, torch.eye(128, device=dev), U_h, quantizer=vtq_q, super_entry_shape=(16, 16),
        quantizer_inputs=(sc, dw_blk.unsqueeze(0).expand(128, 256).reshape(-1).contiguous()),
        **({"inner_block_size": 8, "outer_block_size": 64} if use_triton else {}),
    )
    rel_parity = float((Z_cuda - Z_1d_dw).norm() / Z_cuda.norm().clamp(min=1e-12))
    p_cuda = proxy_losses(X, Z_cuda, H.float(), G)["rel_1d"]
    p_1d_dw = proxy_losses(X, Z_1d_dw, H.float(), G)["rel_1d"]
    print(f"  [check 4] L=I vs native tcq_ldlq_block256: rel_diff={rel_parity:.4e} | 1D proxy cuda={p_cuda:.6f} vs gptq2d_1d={p_1d_dw:.6f}", flush=True)
    assert abs(p_cuda - p_1d_dw) / max(p_cuda, 1e-12) < 0.02, f"Parity mismatch: {p_cuda} vs {p_1d_dw}"
    return use_triton


@torch.no_grad()
def run_phase1_single_down(
    model: nn.Module,
    cal_tokens: torch.Tensor,
    cached_refs: Dict[str, List[dict]],
    fisher_out: Dict[Tuple[int, str], torch.Tensor],
    hess_dir: Path,
    gptq2d_mod,
    use_triton: bool,
    layers: List[int],
    specs: List[str],
    sigma_reg: float,
    out_path: Path,
    report: dict,
    p1_modes: Optional[List[str]] = None,
):
    print("\n=== PHASE 1: Single down_proj Controlled A/B (L0, L15, L31) ===", flush=True)
    dev = torch.device("cuda")
    dtype = next(model.parameters()).dtype
    low_vram = (torch.cuda.get_device_properties(dev).total_memory / (1024 ** 3)) < 36.0
    H_map = collect_single_layer_down_hessians(model, cal_tokens, layers, batch_size=4, low_vram=low_vram)

    all_modes = [
        ("static_1d", False),
        ("static_1d_w", False),
        ("static_2d", False),
        ("static_2d", True),   # reverse block order (as in YAQA / GPTQ-2D Table 2)
        ("static_2d_w", False),
        ("static_2d_refit2d", False),
        ("blk256_1d", False),
        ("blk256_2d", False),
        ("vtq_prod_nogs", False),
        ("vtq_prod", False),
    ]
    modes = [(m, r) for (m, r) in all_modes if (p1_modes is None or (f"{m}_rev" if r else m) in p1_modes)]

    p1 = report.setdefault("phase1_single_down", {})
    for l_idx in layers:
        mod = model.model.layers[l_idx].mlp.down_proj
        W_orig = mod.weight.data.detach().cpu().clone()
        if low_vram:
            model.to("cpu")
            torch.cuda.empty_cache()
        W_f32_cpu = W_orig.float()
        G_out_gpu, fsr = load_yaqa_hessian(l_idx, "down", "out", W_f32_cpu.shape[0], hess_dir, dev, sigma_reg=sigma_reg)
        L_G_gpu, row_prec_gpu, _ = ldl_of_inverse_with_prec(G_out_gpu, size=SUPER_ROWS)
        g_norm2 = float((G_out_gpu * G_out_gpu).sum().item())
        g_diag2 = float((G_out_gpu.diagonal() ** 2).sum().item())
        offdiag_frac = 1.0 - g_diag2 / max(g_norm2, 1e-12)
        G_out_cpu = G_out_gpu.float().cpu()
        L_G_cpu = L_G_gpu.cpu()
        row_prec_cpu = row_prec_gpu.cpu()
        del G_out_gpu, L_G_gpu, row_prec_gpu
        torch.cuda.empty_cache()
        sr_cpu = fisher_out[(l_idx, "mlp.down_proj")].cpu()

        print(f"\n--- Layer {l_idx} down_proj {tuple(W_f32_cpu.shape)} | YAQA G sigma_reg={fsr:.3f}, offdiag_F2={100*offdiag_frac:.1f}% ---", flush=True)

        for sp in specs:
            key = f"L{l_idx}_{sp}"
            entry = p1.setdefault(key, {"layer": l_idx, "spec": sp, "sigma_reg_G": fsr, "G_offdiag_F2_frac": offdiag_frac, "arms": {}})
            seed_m = 1000 + l_idx * 10 + 6
            for mode, rev in modes:
                arm_name = f"{mode}_rev" if rev else mode
                if arm_name in entry["arms"]:
                    continue
                if low_vram:
                    model.to("cpu")
                    torch.cuda.empty_cache()
                W_f32 = W_f32_cpu.to(dev)
                H_down = H_map[l_idx].to(dev)
                G_out = G_out_cpu.to(dev)
                L_G = L_G_cpu.to(dev)
                row_prec = row_prec_cpu.to(dev)
                sr = sr_cpu.to(dev)
                W_hat, sec = quantize_matrix_vtq_mode(
                    W_f32, H_down, sp, mode,
                    G_out=G_out, L_G_cached=None if rev else L_G, row_prec_cached=None if rev else row_prec,
                    seed=seed_m, damp=0.20, row_weights=sr, rot_block=-1,
                    gptq2d_mod=gptq2d_mod, use_triton=use_triton, reverse_order=rev,
                )
                pl = proxy_losses(W_f32, W_hat, H_down, G_out)
                W_hat_bf16_cpu = W_hat.to(dtype=dtype).cpu()
                del W_f32, H_down, G_out, L_G, row_prec, sr, W_hat
                torch.cuda.empty_cache()
                if low_vram:
                    mod.weight.data.copy_(W_hat_bf16_cpu)
                    model.to(dev)
                else:
                    mod.weight.data.copy_(W_hat_bf16_cpu.to(dev))
                ev = {ds: evaluate_model_on_cached_trace(model, ref, dev) for ds, ref in cached_refs.items()}
                mod.weight.data.copy_(W_orig.to(device=mod.weight.device))

                rec = {"sec": sec, "proxy": pl}
                for ds, res in ev.items():
                    rec[ds] = {
                        "mean_kld": res["mean_kld"],
                        "ci95_lo": res["ci95_lo"],
                        "ci95_hi": res["ci95_hi"],
                        "ppl_quant": res["ppl_quant"],
                        "ppl_ref": res["ppl_ref"],
                        "per_seq_kld_sums": res["per_seq_kld_sums"],
                        "per_seq_tok_counts": res["per_seq_tok_counts"],
                    }
                entry["arms"][arm_name] = rec
                print(
                    f"  [{key:11s}] {arm_name:18s} | {sec:5.2f}s | proxy 1D={pl['rel_1d']:.5f} dG={pl['rel_diagG']:.5f} 2D={pl['rel_2d']:.5f} | "
                    f"KLD wt2={ev['wikitext2']['mean_kld']:.6f} nem={ev['nemotron']['mean_kld']:.6f} mul={ev['multilingual']['mean_kld']:.6f}",
                    flush=True,
                )
                out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

            # Compute paired bootstrap vs static_1d and vs vtq_prod
            paired = {}
            for base_arm in ("static_1d", "vtq_prod"):
                if base_arm not in entry["arms"]:
                    continue
                paired[f"vs_{base_arm}"] = {}
                for arm_name, rec in entry["arms"].items():
                    if arm_name == base_arm:
                        continue
                    paired[f"vs_{base_arm}"][arm_name] = {}
                    for ds in cached_refs:
                        pb = paired_bootstrap_diff(rec[ds], entry["arms"][base_arm][ds], n_boot=4000)
                        pb["rel_pct"] = 100.0 * pb["mean_diff"] / max(entry["arms"][base_arm][ds]["mean_kld"], 1e-12)
                        paired[f"vs_{base_arm}"][arm_name][ds] = pb
            entry["paired"] = paired
            out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

        del G_out_cpu, L_G_cpu, row_prec_cpu, W_orig, W_f32_cpu
        torch.cuda.empty_cache()


@torch.no_grad()
def run_phase2_full_model(
    model_path: str,
    cal_tokens: torch.Tensor,
    cached_refs: Dict[str, List[dict]],
    fisher_out: Dict[Tuple[int, str], torch.Tensor],
    hess_dir: Path,
    gptq2d_mod,
    use_triton: bool,
    spec_name: str,
    sigma_reg: float,
    out_path: Path,
    report: dict,
    arms_to_run: List[Tuple[str, str]],
):
    """
    Full 32-layer streaming VTQ + GPTAQ (alpha=0.50) A/B on Llama-3.1-8B-Instruct.
    On q/k/v/gate/up (160/224 matrices): identical production VTQ across all arms.
    On o_proj + down_proj (64/224 matrices): switches mode per arm (`vtq_prod`, `static_1d`,
    `static_2d`, `static_2d_refit2d`, etc.) while keeping the exact same streaming GPTAQ pipeline.
    Loads clean BF16 weights from local SSD per arm and streams G from disk (0 GB duplicate RAM!).
    """
    print(f"\n=== PHASE 2: Full 32-Layer Streaming GPTAQ A/B ({spec_name}, 64 o+down matrices) ===", flush=True)
    dev = torch.device("cuda")
    low_vram = (torch.cuda.get_device_properties(dev).total_memory / (1024 ** 3)) < 36.0
    n_seqs, seq_len = cal_tokens.shape
    batch_size = 4
    eff_gptaq = 0.50
    damp = 0.20

    # Ensure all 64 YAQA output Hessians are cached on SSD in parallel (8 threads)
    print("  Ensuring all 64 YAQA output Hessians (o_proj + down_proj) are cached on disk...", flush=True)
    t_h = time.perf_counter()
    from concurrent.futures import ThreadPoolExecutor
    fnames = [f"{l_idx}_{short_nm}_hout.pt" for l_idx in range(32) for short_nm in ("o", "down")]
    with ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(lambda fn: hf_hub_download(repo_id=HESS_REPO, filename=fn, local_dir=str(hess_dir)), fnames))
    print(f"  All 64 G files cached on disk in {time.perf_counter()-t_h:.1f}s", flush=True)

    # If any arm uses `ownG`, compute all 64 on-the-fly empirical Fisher G matrices once (~30s!)
    own_g_dir = hess_dir / "ownG_24x1024"
    own_g_sec = 0.0
    if any("ownG" in m for _, m in arms_to_run):
        own_g_sec = compute_own_g_odown_offload(model_path, cal_tokens, own_g_dir, n_seqs=24, seq_len=1024)
        report["ownG_build_sec"] = own_g_sec

    p2 = report.setdefault("phase2_full_model", {"spec": spec_name, "sigma_reg_G": sigma_reg, "arms": {}})

    for arm_name, odown_mode in arms_to_run:
        if arm_name in p2["arms"]:
            print(f"  [skip] {arm_name} already completed", flush=True)
            continue
        print(f"\n--- Running Full-Model Arm: {arm_name} (o+down mode = {odown_mode}) ---", flush=True)
        model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, attn_implementation="sdpa").eval()
        if not low_vram:
            model.to(dev)
        dtype = next(model.parameters()).dtype
        n_layers = model.config.num_hidden_layers
        torch.cuda.synchronize()
        t_arm = time.perf_counter()
        odown_sec = 0.0
        other_sec = 0.0
        proxy_2d_odown = []
        proxy_1d_odown = []

        if low_vram:
            model.model.embed_tokens.to(dev)
            model.model.rotary_emb.to(dev)
            cal_q = [model.model.embed_tokens(cal_tokens[i : i + batch_size].to(dev)).cpu() for i in range(0, n_seqs, batch_size)]
            pos_cal = model.model.rotary_emb(cal_q[0][:1].to(dev), torch.arange(seq_len, device=dev).unsqueeze(0))
            model.model.embed_tokens.to("cpu")
            model.model.rotary_emb.to("cpu")
            cal_r = [c.clone() for c in cal_q]
        else:
            cal_q = [model.model.embed_tokens(cal_tokens[i : i + batch_size].to(dev)) for i in range(0, n_seqs, batch_size)]
            cal_r = [c.clone() for c in cal_q]
            pos_cal = model.model.rotary_emb(cal_q[0][:1], torch.arange(seq_len, device=dev).unsqueeze(0))

        for l_idx in range(n_layers):
            lq = model.model.layers[l_idx]
            if low_vram:
                lq.to(dev)
            lr = copy.deepcopy(lq)

            def q_layer_mod(mod: nn.Linear, H: torch.Tensor, B: torch.Tensor, full_name: str, seed_i: int, dm: float):
                nonlocal odown_sec, other_sec
                W = mod.weight.data.float()
                W_tgt = vtq._drift_target(W, B, eff_gptaq)
                sr = fisher_out[(l_idx, full_name)].to(dev)
                if full_name in ("self_attn.o_proj", "mlp.down_proj"):
                    short_nm = "o" if full_name.endswith("o_proj") else "down"
                    mode_eff = odown_mode
                    sig_eff = sigma_reg
                    use_own_g = False
                    if mode_eff == "static_2d_d01":
                        mode_eff = "static_2d"
                        sig_eff = 0.10
                    elif mode_eff == "static_2d_ownG":
                        mode_eff = "static_2d"
                        sig_eff = 0.10
                        use_own_g = True
                    # Always evaluate 2D proxy against YAQA reference G for apples-to-apples logging
                    G_ref, _ = load_yaqa_hessian(l_idx, short_nm, "out", W.shape[0], hess_dir, dev, sigma_reg=0.01)
                    if use_own_g:
                        G_d = torch.load(own_g_dir / f"{l_idx}_{short_nm}_ownG.pt", map_location="cpu", weights_only=True).to(device=dev, dtype=torch.float64)
                        G_d.diagonal().add_(sig_eff)
                    elif sig_eff != 0.01:
                        G_d, _ = load_yaqa_hessian(l_idx, short_nm, "out", W.shape[0], hess_dir, dev, sigma_reg=sig_eff)
                    else:
                        G_d = G_ref
                    if "2d" in mode_eff:
                        LG_d, rp_d, _ = ldl_of_inverse_with_prec(G_d, size=SUPER_ROWS, use_fp64=True)
                    else:
                        LG_d, rp_d = None, None
                    w_hat, dt = quantize_matrix_vtq_mode(
                        W, H, spec_name, mode_eff,
                        G_out=G_d.float(), L_G_cached=LG_d, row_prec_cached=rp_d,
                        seed=seed_i, damp=damp * dm, row_weights=sr, rot_block=-1,
                        W_target=W_tgt, gptq2d_mod=gptq2d_mod, use_triton=use_triton,
                    )
                    pl = proxy_losses(W_tgt, w_hat, H, G_ref)
                    proxy_2d_odown.append(pl["rel_2d"])
                    proxy_1d_odown.append(pl["rel_1d"])
                    odown_sec += dt
                    del G_d, G_ref, LG_d, rp_d
                else:
                    w_hat, dt = quantize_matrix_vtq_mode(
                        W, H, spec_name, "vtq_prod",
                        seed=seed_i, damp=damp * dm, row_weights=sr, rot_block=-1,
                        W_target=W_tgt,
                    )
                    other_sec += dt
                mod.weight.data.copy_(w_hat.to(dtype=dtype, device=dev))

            # 1) q_proj, k_proj, v_proj
            acc = DualCovAccumulator(True)
            for cq, cr in zip(cal_q, cal_r):
                cq_d, cr_d = (cq.to(dev), cr.to(dev)) if low_vram else (cq, cr)
                acc.add(lq.input_layernorm(cq_d), lr.input_layernorm(cr_d))
            Hqkv, Bqkv = acc.get(gptaq_damp=damp)
            for i_s, nm_s in enumerate(("q_proj", "k_proj", "v_proj")):
                q_layer_mod(getattr(lq.self_attn, nm_s), Hqkv, Bqkv, f"self_attn.{nm_s}", 1000 + l_idx * 10 + i_s, 1.0)
            del Hqkv, Bqkv, acc

            # 2) o_proj + attention residual (matching vtq.py exactly via pre-hook on o_proj; cache pre-o inputs to avoid 2x SDPA)
            acc = DualCovAccumulator(True)
            bq, br = [None], [None]
            hk_q = lq.self_attn.o_proj.register_forward_pre_hook(lambda m, a: bq.__setitem__(0, a[0]))
            hk_r = lr.self_attn.o_proj.register_forward_pre_hook(lambda m, a: br.__setitem__(0, a[0]))
            bq_list = []
            ref_mid = []
            for cq, cr in zip(cal_q, cal_r):
                cq_d, cr_d = (cq.to(dev), cr.to(dev)) if low_vram else (cq, cr)
                sq._attn(lq, cq_d, pos_cal)
                rm = cr_d + sq._attn(lr, cr_d, pos_cal)
                ref_mid.append(rm.cpu() if low_vram else rm)
                acc.add(bq[0], br[0])
                bq_list.append(bq[0].cpu() if low_vram else bq[0])
                bq[0] = br[0] = None
            hk_q.remove()
            hk_r.remove()
            Ho, Bo = acc.get(gptaq_damp=damp)
            q_layer_mod(lq.self_attn.o_proj, Ho, Bo, "self_attn.o_proj", 1000 + l_idx * 10 + 3, 1.0)
            del Ho, Bo, acc
            for bi in range(len(cal_q)):
                cq_d = cal_q[bi].to(dev) if low_vram else cal_q[bi]
                pre_o = bq_list[bi].to(dev) if low_vram else bq_list[bi]
                out_q = cq_d + lq.self_attn.o_proj(pre_o)
                cal_q[bi] = out_q.cpu() if low_vram else out_q
            del bq_list
            cal_r = ref_mid
            del ref_mid

            # 3) gate_proj, up_proj
            acc = DualCovAccumulator(True)
            for cq, cr in zip(cal_q, cal_r):
                cq_d, cr_d = (cq.to(dev), cr.to(dev)) if low_vram else (cq, cr)
                acc.add(lq.post_attention_layernorm(cq_d), lr.post_attention_layernorm(cr_d))
            Hup, Bup = acc.get(gptaq_damp=damp)
            q_layer_mod(lq.mlp.gate_proj, Hup, Bup, "mlp.gate_proj", 1000 + l_idx * 10 + 4, 1.0)
            q_layer_mod(lq.mlp.up_proj, Hup, Bup, "mlp.up_proj", 1000 + l_idx * 10 + 5, 1.0)
            del Hup, Bup, acc

            # 4) down_proj + MLP residual (update cal_r inline in 1 pass; recompute only lq.mlp post-quant)
            acc = DualCovAccumulator(True)
            for bi in range(len(cal_q)):
                cq_d = cal_q[bi].to(dev) if low_vram else cal_q[bi]
                cr_d = cal_r[bi].to(dev) if low_vram else cal_r[bi]
                nq = lq.post_attention_layernorm(cq_d)
                nr = lr.post_attention_layernorm(cr_d)
                mq = lq.mlp.act_fn(lq.mlp.gate_proj(nq)) * lq.mlp.up_proj(nq)
                mr = lr.mlp.act_fn(lr.mlp.gate_proj(nr)) * lr.mlp.up_proj(nr)
                acc.add(mq, mr)
                or_ = cr_d + lr.mlp.down_proj(mr)
                cal_r[bi] = or_.cpu() if low_vram else or_
                del nq, nr, mq, mr, or_
            Hdn, Bdn = acc.get(gptaq_damp=damp)
            q_layer_mod(lq.mlp.down_proj, Hdn, Bdn, "mlp.down_proj", 1000 + l_idx * 10 + 6, 1.0)
            del Hdn, Bdn, acc
            for bi in range(len(cal_q)):
                cq_d = cal_q[bi].to(dev) if low_vram else cal_q[bi]
                oq = cq_d + lq.mlp(lq.post_attention_layernorm(cq_d))
                cal_q[bi] = oq.cpu() if low_vram else oq
            del lr
            if low_vram:
                lq.to("cpu")
                torch.cuda.empty_cache()
            if (l_idx + 1) % 8 == 0:
                print(f"    layer {l_idx+1:2d}/{n_layers} | o+down={odown_sec:.1f}s | mean 2D proxy={np.mean(proxy_2d_odown):.5f}", flush=True)

        del cal_q, cal_r
        if low_vram:
            torch.cuda.empty_cache()
            model.to(dev)
        torch.cuda.synchronize()
        wall_sec = time.perf_counter() - t_arm

        ev = {ds: evaluate_model_on_cached_trace(model, ref, dev) for ds, ref in cached_refs.items()}
        rec = {
            "odown_mode": odown_mode,
            "wall_sec": wall_sec,
            "odown_sec": odown_sec,
            "other_sec": other_sec,
            "mean_proxy_2d_odown": float(np.mean(proxy_2d_odown)),
            "mean_proxy_1d_odown": float(np.mean(proxy_1d_odown)),
        }
        for ds, res in ev.items():
            rec[ds] = {
                "mean_kld": res["mean_kld"],
                "ci95_lo": res["ci95_lo"],
                "ci95_hi": res["ci95_hi"],
                "ppl_quant": res["ppl_quant"],
                "ppl_ref": res["ppl_ref"],
                "top1_match": res["top1_match"],
                "per_seq_kld_sums": res["per_seq_kld_sums"],
                "per_seq_tok_counts": res["per_seq_tok_counts"],
            }
        p2["arms"][arm_name] = rec
        print(
            f"  [DONE {arm_name:20s}] wall={wall_sec:6.1f}s (o+dn={odown_sec:5.1f}s) | "
            f"2D_proxy={rec['mean_proxy_2d_odown']:.5f} | "
            f"KLD wt2={ev['wikitext2']['mean_kld']:.5f} (PPL {ev['wikitext2']['ppl_quant']:.3f}) | "
            f"nem={ev['nemotron']['mean_kld']:.5f} | mul={ev['multilingual']['mean_kld']:.5f}",
            flush=True,
        )
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        del model
        gc.collect()
        torch.cuda.empty_cache()

    # Paired bootstrap summary against both static_1d and vtq_prod
    paired = {}
    for base_arm in ("static_1d", "vtq_prod"):
        if base_arm not in p2["arms"]:
            continue
        paired[f"vs_{base_arm}"] = {}
        for arm_name, rec in p2["arms"].items():
            if arm_name == base_arm:
                continue
            paired[f"vs_{base_arm}"][arm_name] = {}
            for ds in cached_refs:
                pb = paired_bootstrap_diff(rec[ds], p2["arms"][base_arm][ds], n_boot=4000)
                pb["rel_pct"] = 100.0 * pb["mean_diff"] / max(p2["arms"][base_arm][ds]["mean_kld"], 1e-12)
                paired[f"vs_{base_arm}"][arm_name][ds] = pb
    p2["paired"] = paired
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data")
    ap.add_argument("--hess-dir", type=Path, default=Path("/content/work/hessians_8b"))
    ap.add_argument("--gptq2d-dir", type=Path, default=Path("/content/work/GPTQ-2D"))
    ap.add_argument("--output", type=Path, default=ROOT / "results" / "exp32_gptq2d_8b_ab.json")
    ap.add_argument("--layers", type=int, nargs="+", default=[0, 15, 31])
    ap.add_argument("--p1-specs", type=str, nargs="+", default=["VTQ_200", "VTQ_300"])
    ap.add_argument("--p1-modes", type=str, nargs="*", default=None)
    ap.add_argument("--p2-spec", type=str, default="VTQ_200")
    ap.add_argument("--p2-arms", type=str, nargs="*", default=["vtq_prod", "static_1d", "static_2d", "static_2d_w"])
    ap.add_argument("--sigma-reg", type=float, default=0.01)
    ap.add_argument("--skip-phase1", action="store_true")
    ap.add_argument("--skip-phase2", action="store_true")
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    dev = torch.device("cuda")
    args.hess_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    vtq.load_and_register_vtq()
    gptq2d_mod, gptq2d_ref = _import_gptq2d(args.gptq2d_dir)

    use_triton = run_phase0_selftest(gptq2d_mod, gptq2d_ref, dev)

    print(f"\nLoading {args.model} to {torch.cuda.get_device_name()}...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="sdpa").eval().to(dev)
    cal_tokens, traces = build_cal_and_eval_traces(Path(args.model), args.data_dir, n_cal=64, cal_len=2048)

    print("Precomputing BF16 reference logits on WikiText-2 / Nemotron / Multilingual...", flush=True)
    cached_refs = {ds: precompute_ref_trace_data(model, seqs, dev) for ds, seqs in traces.items()}

    print("Computing exact offloaded Fisher row sensitivities (8x1024)...", flush=True)
    fisher_out = compute_fisher_offload_llama(model, cal_tokens, n_seqs=8, seq_len=1024)

    report = json.loads(args.output.read_text()) if args.output.exists() else {
        "model": args.model,
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "use_triton_gptq2d": use_triton,
        "sigma_reg_G": args.sigma_reg,
    }

    if not args.skip_phase1:
        run_phase1_single_down(
            model, cal_tokens, cached_refs, fisher_out, args.hess_dir,
            gptq2d_mod, use_triton, args.layers, args.p1_specs, args.sigma_reg, args.output, report,
            p1_modes=args.p1_modes,
        )

    del model
    gc.collect()
    torch.cuda.empty_cache()

    if not args.skip_phase2:
        arms_p2 = [(nm, nm) for nm in args.p2_arms]
        run_phase2_full_model(
            args.model, cal_tokens, cached_refs, fisher_out, args.hess_dir,
            gptq2d_mod, use_triton, args.p2_spec, args.sigma_reg, args.output, report, arms_p2,
        )

    print(f"\n=== ALL DONE -> saved {args.output} ===", flush=True)


if __name__ == "__main__":
    main()
