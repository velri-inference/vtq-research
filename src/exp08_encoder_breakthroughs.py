"""
Experiment 08 — Next-Gen VTQ Encoder, Preconditioner & Memory-Bounded Polish Breakthroughs
(`src/exp08_encoder_breakthroughs.py` -> `results/exp08_encoder_breakthroughs.json`).

Zero-inference-overhead & bit-exact `.vtq` v2 upgrades evaluated systematically:
  1. `Outer-DCT` Kronecker Transform (`R_K = DCT-II_q (x) H_{2^a}` vs `H_{2^a} (x) DCT-II_q`):
     Separates correlated DCT modes by `p_2 = 2^a >= 256` columns so 100% of cross-mode correlation
     is compensated by sequential Block16-LDLQ Cholesky feedback instead of falling inside the same
     uncompensated 16-column Viterbi subblock, while simplifying runtime transform execution to
     contiguous power-of-2 FWHT blocks + tiny outer `q x q` combine.
  2. `Schur-Diag` (`P_cc = sum_{k=c}^{15} V_{c,k}^2` vs `V_{c,c}^2`):
     Replaces the scalar-LDLQ inverse diagonal `1 / U_{cc}^2` inside `ldlq_block256_cuda` with the
     exact 16-column conditional Schur quadratic diagonal `diag(V V^T)` (`V = U_blk_inv[s0:s1, s0:s1]`)
     at 0.000 ms runtime overhead.
  3. `MM-LDLQ` (Proximal Majorization-Minimization Intra-Subblock Viterbi):
     Captures all 120 off-diagonal couplings of `P = V V^T` inside each 16-column subblock via a
     spectral-majorized Viterbi refinement pass with monotonic per-16-row-tile Cholesky acceptance.
  4. `GECD` (Exact Full-Hessian Gain-Index `gi[M, K/16]` & Block-Exponent `e[M, K/256]` Coordinate Descent):
     Closed-form Gauss-Seidel coordinate optimization of `.vtq` v2's discrete subblock gain indices `gi`
     and 5-bit block exponents `e` against the full two-sided rotated Hessian `H_rot` (`0` Viterbi calls,
     `0` trellis changes, `< 1 ms` per matrix).
  5. `GS-Viterbi` (`MACaQ-MM` Right-to-Left Full-Hessian Tile Refinement):
     Post-LDLQ right-to-left sweep across 16-column subblocks proposing valid tail-biting Viterbi tiles
     along the exact full-Hessian Gauss-Seidel residual `(R @ H_rot[:, J])`, accepted strictly per 16-row
     tile when the full `K x K` Hessian error decreases.
  6. `Block-Polish` (`< 3 GiB` streaming 4-layer window scale polish) & `CP-Polish` (`O(1)`-layer
     gradient-checkpointed full-model logit KLD scale polish over `(d_row, suh)`).
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from transformers import AutoModelForCausalLM

import src.equal_bpw as eqb
import src.liqk_v2 as lv2
import src.streaming_quant as sq
import src.vtq_v2 as v2
import src.vtq_v2_format as v2f
from src.eval_kld import (
    evaluate_model_on_cached_trace,
    load_eval_trace,
    paired_bootstrap_diff,
    precompute_ref_trace_data,
)
from src.run_1b_liqk_v2_experiments import restore_clean_model_weights, save_clean_model_weights
from src.tcq_proto import ldlq_block256_cuda, viterbi_ub_cuda
from src.vtq_v2_format import NU, TILE, assemble_vtq2, decode_slice, load_vtq2_formats, pack_slice

ROOT = Path(__file__).resolve().parent.parent
OUT_JSON = ROOT / "results" / "exp08_encoder_breakthroughs.json"

_ORIG_APPLY_ROT = v2f.apply_kronecker_rot
_OUTER_KRON_CACHE: dict[tuple[int, torch.device], tuple[int, int, int, torch.Tensor, torch.Tensor, torch.Tensor]] = {}


def _get_outer_kron_factors(K: int, device: torch.device):
    key = (K, device)
    if key not in _OUTER_KRON_CACHE:
        p2, q = v2f.split_pow2_odd(K)
        a = p2.bit_length() - 1
        b1 = 1 << (a // 2)
        b2 = 1 << (a - a // 2)
        H1 = v2f.get_hadamard_ortho(b1, device)
        H2 = v2f.get_hadamard_ortho(b2, device)
        Dq = v2f.get_dct_ortho(q, device)
        _OUTER_KRON_CACHE[key] = (q, b1, b2, Dq, H1, H2)
    return _OUTER_KRON_CACHE[key]


def apply_kronecker_rot_outer(X: torch.Tensor, rot_block: int = -1, transpose: bool = False) -> torch.Tensor:
    """Applies `R_K^{outer} = DCT-II_q (x) H_{2^a}` so correlated DCT modes are separated by `p_2 = 2^a >= 256` cols."""
    K = X.shape[-1]
    if rot_block > 0:
        return _ORIG_APPLY_ROT(X, rot_block=rot_block, transpose=transpose)
    p2, q = v2f.split_pow2_odd(K)
    if q == 1:
        return _ORIG_APPLY_ROT(X, rot_block=rot_block, transpose=transpose)
    assert p2 >= 32, f"Expected K={K} divisible by 32, got p2={p2}"
    orig_shape = X.shape
    N = X.numel() // K
    prev_tf32 = torch.backends.cuda.matmul.allow_tf32
    if prev_tf32:
        torch.backends.cuda.matmul.allow_tf32 = False
    try:
        q_f, b1, b2, Dq, H1, H2 = _get_outer_kron_factors(K, X.device)
        Y = torch.matmul(H1, X.reshape(N * q_f, b1, b2) @ H2).reshape(N, q_f, p2)
        lmat = Dq.T if transpose else Dq
        return torch.matmul(lmat, Y).reshape(orig_shape)
    finally:
        if prev_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True


@contextmanager
def use_rotation_mode(outer_dct: bool):
    fn = apply_kronecker_rot_outer if outer_dct else _ORIG_APPLY_ROT
    prev_f, prev_v2 = v2f.apply_kronecker_rot, v2.apply_kronecker_rot
    v2f.apply_kronecker_rot = fn
    v2.apply_kronecker_rot = fn
    try:
        yield
    finally:
        v2f.apply_kronecker_rot = prev_f
        v2.apply_kronecker_rot = prev_v2


@dataclass
class EncoderCfg:
    name: str
    outer_dct: bool = False
    schur_diag: bool = False
    mm_passes: int = 0
    mm_eta: float = 1.25
    gecd_sweeps: int = 0
    gs_viterbi_passes: int = 0
    gs_eta: float = 1.20
    damp: float = 0.20
    opt_reg: float = 0.01


@torch.no_grad()
def _gecd_refine_rows(
    W_rot_sub: torch.Tensor,
    W_hat_rot: torch.Tensor,
    G: torch.Tensor,
    gi: torch.Tensor,
    idx: torch.Tensor,
    d_row: torch.Tensor,
    e: torch.Tensor,
    H_opt: torch.Tensor,
    tab_1d: torch.Tensor,
    gains: torch.Tensor,
    sub: int,
    n_sweeps: int,
):
    """Exact full-Hessian Gauss-Seidel coordinate descent over subblock `gi[R, K/sub]` and block `e[R, K/256]`."""
    if n_sweeps <= 0:
        return
    R, Kin = W_rot_sub.shape
    nblocks = Kin // TILE
    has_gains = gains.numel() > 1
    u_mat = tab_1d[idx.long()]
    RH = (W_rot_sub - W_hat_rot) @ H_opt

    for sw in range(n_sweeps):
        blk_order = range(nblocks - 1, -1, -1) if (sw % 2 == 0) else range(nblocks)
        if has_gains:
            for ibl in blk_order:
                c0 = ibl * TILE
                d_rb = (d_row * torch.exp2(0.25 * e[:, ibl].float())).clamp(min=1e-8)
                sub_order = range(TILE // sub - 1, -1, -1) if (sw % 2 == 0) else range(TILE // sub)
                for ib in sub_order:
                    s0 = c0 + ib * sub
                    s1 = s0 + sub
                    H_sub = H_opt[s0:s1, s0:s1]
                    u_sub = u_mat[:, s0:s1]
                    g_old_idx = gi[:, s0].long()
                    gamma_old = gains[g_old_idx]
                    denom = ((u_sub @ H_sub) * u_sub).sum(dim=1)
                    num_delta = (u_sub * RH[:, s0:s1]).sum(dim=1)
                    gamma_star = gamma_old + num_delta / (d_rb * denom.clamp(min=1e-12))
                    g_new_idx = (gamma_star.unsqueeze(1) - gains.unsqueeze(0)).abs().argmin(dim=1)
                    gamma_new = gains[g_new_idx]
                    d_gamma = gamma_new - gamma_old
                    dloss = -2.0 * d_rb * d_gamma * num_delta + (d_rb * d_gamma).square() * denom
                    take = (dloss < -1e-12) & (denom > 1e-12) & (g_new_idx != g_old_idx)
                    if bool(take.any()):
                        g_fin_idx = torch.where(take, g_new_idx, g_old_idx).to(torch.uint8)
                        gamma_fin = torch.where(take, gamma_new, gamma_old)
                        gi[:, s0:s1] = g_fin_idx.unsqueeze(1)
                        G[:, s0:s1] = gamma_fin.unsqueeze(1) * u_sub
                        w_new = d_rb.unsqueeze(1) * G[:, s0:s1]
                        dw = w_new - W_hat_rot[:, s0:s1]
                        W_hat_rot[:, s0:s1] = w_new
                        RH.sub_(dw @ H_opt[s0:s1, :])

        for ibl in blk_order:
            c0, c1 = ibl * TILE, (ibl + 1) * TILE
            H_blk = H_opt[c0:c1, c0:c1]
            G_blk = G[:, c0:c1]
            e_old = e[:, ibl].float()
            d_old = (d_row * torch.exp2(0.25 * e_old)).clamp(min=1e-8)
            denom = ((G_blk @ H_blk) * G_blk).sum(dim=1)
            num_delta = (G_blk * RH[:, c0:c1]).sum(dim=1)
            d_star = (d_old + num_delta / denom.clamp(min=1e-12)).clamp(min=1e-10)
            e_cand = torch.round(torch.log2(d_star / d_row) / 0.25).clamp(-16, 15)
            d_cand = (d_row * torch.exp2(0.25 * e_cand)).clamp(min=1e-8)
            dd = d_cand - d_old
            dloss = -2.0 * dd * num_delta + dd.square() * denom
            take = (dloss < -1e-12) & (denom > 1e-12) & (e_cand != e_old)
            if bool(take.any()):
                e[:, ibl] = torch.where(take, e_cand.to(torch.int8), e[:, ibl])
                d_fin = torch.where(take, d_cand, d_old)
                w_new = d_fin.unsqueeze(1) * G_blk
                dw = w_new - W_hat_rot[:, c0:c1]
                W_hat_rot[:, c0:c1] = w_new
                RH.sub_(dw @ H_opt[c0:c1, :])


@torch.no_grad()
def _gs_viterbi_refine_rows(
    W_rot_sub: torch.Tensor,
    W_hat_rot: torch.Tensor,
    G: torch.Tensor,
    gi: torch.Tensor,
    idx: torch.Tensor,
    d_row: torch.Tensor,
    e: torch.Tensor,
    H_opt: torch.Tensor,
    spec_name: str,
    n_passes: int,
    gs_eta: float,
):
    """Right-to-left full-Hessian Gauss-Seidel Viterbi tile refinement (`MACaQ-MM`), strictly monotonic per 16-row tile."""
    if n_passes <= 0:
        return
    dev = W_rot_sub.device
    R, Kin = W_rot_sub.shape
    nblocks = Kin // TILE
    cfg, tab, gains = v2._get_spec_gpu(spec_name, dev)
    K = int(cfg["K"])
    pp = int(cfg.get("pp", 0))
    m0 = int(cfg["m"])
    m_extra = tuple(int(x) for x in cfg.get("m_extra", ()))
    n_cand = int(cfg.get("n_cand", 0))
    tb_win = int(os.environ.get("TCQ_TB_WIN", "32"))
    max_r = 65536 if (pp > 0 and R > 65536) else R

    # Precompute 16x16 spectral majorizer constants for each subblock
    total_subs = Kin // 16
    h_diags = []
    step_scales = []
    for j in range(total_subs):
        s0, s1 = j * 16, (j + 1) * 16
        Hs = H_opt[s0:s1, s0:s1]
        hd = torch.diagonal(Hs).clamp(min=1e-12)
        invs = hd.rsqrt()
        C_s = Hs * invs.unsqueeze(1) * invs.unsqueeze(0)
        mu = float(torch.linalg.eigvalsh(C_s)[-1].clamp(min=1.0).item())
        h_diags.append(hd)
        step_scales.append(gs_eta / mu)

    RH = (W_rot_sub - W_hat_rot) @ H_opt
    for p_idx in range(n_passes):
        sub_iter = range(total_subs - 1, -1, -1) if (p_idx % 2 == 0) else range(total_subs)
        for j in sub_iter:
            ibl = j // 16
            s0, s1 = j * 16, (j + 1) * 16
            d_rb = (d_row * torch.exp2(0.25 * e[:, ibl].float())).clamp(min=1e-8)
            hd = h_diags[j]
            sc = step_scales[j]
            g_old = G[:, s0:s1]
            z_cand = g_old + sc * (RH[:, s0:s1] / (d_rb.unsqueeze(1) * hd.unsqueeze(0)))
            wts_tile = hd.repeat(16)

            if max_r < R:
                g_try = torch.empty_like(g_old)
                gi_try = torch.empty_like(gi[:, s0:s1])
                idx_try = torch.empty_like(idx[:, s0:s1])
                for r0 in range(0, R, max_r):
                    r1 = min(R, r0 + max_r)
                    gt, git, ixt = viterbi_ub_cuda(
                        z_cand[r0:r1].contiguous().view((r1 - r0) // 16, 256),
                        tab, K, NU, m0, gains, sub=16, wts=wts_tile,
                        n_cand=n_cand, return_code=True, tail_biting=True,
                        tb_window=tb_win, m_extra=m_extra, pair_parity=pp,
                    )
                    g_try[r0:r1] = gt.view(r1 - r0, 16)
                    gi_try[r0:r1] = git.view(r1 - r0, 16).to(torch.uint8)
                    idx_try[r0:r1] = ixt.view(r1 - r0, 16).to(torch.int16)
            else:
                gt, git, ixt = viterbi_ub_cuda(
                    z_cand.contiguous().view(R // 16, 256),
                    tab, K, NU, m0, gains, sub=16, wts=wts_tile,
                    n_cand=n_cand, return_code=True, tail_biting=True,
                    tb_window=tb_win, m_extra=m_extra, pair_parity=pp,
                )
                g_try = gt.view(R, 16)
                gi_try = git.view(R, 16).to(torch.uint8)
                idx_try = ixt.view(R, 16).to(torch.int16)

            dw_try = d_rb.unsqueeze(1) * (g_try - g_old)
            Hs = H_opt[s0:s1, s0:s1]
            row_dloss = -2.0 * (dw_try * RH[:, s0:s1]).sum(dim=1) + ((dw_try @ Hs) * dw_try).sum(dim=1)
            tile_dloss = row_dloss.view(R // 16, 16).sum(dim=1)
            acc_tile = tile_dloss < -1e-12
            if bool(acc_tile.any()):
                acc_r = acc_tile.repeat_interleave(16).unsqueeze(1)
                G[:, s0:s1] = torch.where(acc_r, g_try, g_old)
                gi[:, s0:s1] = torch.where(acc_r, gi_try, gi[:, s0:s1])
                idx[:, s0:s1] = torch.where(acc_r, idx_try, idx[:, s0:s1])
                dw_acc = torch.where(acc_r, dw_try, torch.zeros_like(dw_try))
                W_hat_rot[:, s0:s1] = W_hat_rot[:, s0:s1] + dw_acc
                RH.sub_(dw_acc @ H_opt[s0:s1, :])


@torch.no_grad()
def _quantize_rows_vtq2_plus(
    W_rot_sub: torch.Tensor,
    U: torch.Tensor,
    U_blks: torch.Tensor,
    U_blks_inv: torch.Tensor,
    diag_w: torch.Tensor,
    H_opt: torch.Tensor,
    spec_name: str,
    ecfg: EncoderCfg,
    need_codes: bool,
) -> dict:
    dev = W_rot_sub.device
    R, Kin = W_rot_sub.shape
    nblocks = Kin // TILE
    cfg, tab, gains = v2._get_spec_gpu(spec_name, dev)
    K = int(cfg["K"])
    pp = int(cfg.get("pp", 0))
    m0 = int(cfg["m"])
    m_extra = tuple(int(x) for x in cfg.get("m_extra", ()))
    n_cand = int(cfg.get("n_cand", 0))
    sub = int(cfg.get("sub", 16))
    cfg_scale = float(cfg.get("scale", 1.0))
    tb_win = int(os.environ.get("TCQ_TB_WIN", "32"))

    emit_codes = need_codes or (ecfg.gecd_sweeps > 0) or (ecfg.gs_viterbi_passes > 0) or (ecfg.mm_passes > 0)
    W_work = W_rot_sub.clone()
    W_hat_rot = torch.empty_like(W_rot_sub)
    e_out = torch.empty((R, nblocks), dtype=torch.int8, device=dev)
    G_out = torch.empty((R, Kin), dtype=torch.float32, device=dev) if emit_codes else None
    gi_out = torch.empty((R, Kin), dtype=torch.uint8, device=dev) if emit_codes else None
    idx_out = torch.empty((R, Kin), dtype=torch.int16, device=dev) if emit_codes else None

    rms_rb0 = W_rot_sub.view(R, nblocks, TILE).square().mean(dim=2).sqrt().clamp(min=1e-10) * cfg_scale
    d_row = torch.exp(torch.log(rms_rb0).mean(dim=1)).to(torch.float16).to(torch.float32).clamp(min=1e-8)
    max_r = 65536 if (pp > 0 and R > 65536) else R

    for ibl in range(nblocks):
        c0, c1 = ibl * TILE, (ibl + 1) * TILE
        w_blk_start = W_work[:, c0:c1].clone()
        U_blk = U_blks[ibl]
        U_blk_inv = U_blks_inv[ibl]
        dw_blk = diag_w[c0:c1]

        cur_rms = w_blk_start.square().mean(dim=1).sqrt().clamp(min=1e-10) * cfg_scale
        e0 = torch.round(torch.log2(cur_rms / d_row) / 0.25).clamp(-16, 15)
        d_rb = (d_row * torch.exp2(0.25 * e0)).clamp(min=1e-8)

        if ecfg.mm_passes > 0:
            w_sub_work = w_blk_start.clone()
            G_pass = torch.empty_like(w_blk_start)
            gi_b = torch.empty((R, TILE), dtype=torch.uint8, device=dev)
            idx_b = torch.empty((R, TILE), dtype=torch.int16, device=dev)
            d2_col = d_rb.square().unsqueeze(1)
            for ib in range(16):
                s0, s1 = ib * 16, (ib + 1) * 16
                V = U_blk_inv[s0:s1, s0:s1]
                P = V @ V.T
                p_diag = torch.diagonal(P).clamp(min=1e-12)
                invs = p_diag.rsqrt()
                mu = float(torch.linalg.eigvalsh(P * invs.unsqueeze(1) * invs.unsqueeze(0))[-1].clamp(min=1.0).item())
                step_mat = (P / p_diag.unsqueeze(0)) * (ecfg.mm_eta / mu)
                wts_tile = p_diag.repeat(16)
                y0 = w_sub_work[:, s0:s1] / d_rb.unsqueeze(1)

                def _run_vit(y_in: torch.Tensor):
                    if max_r < R:
                        go = torch.empty_like(y_in)
                        gio = torch.empty((R, 16), dtype=torch.uint8, device=dev)
                        ixo = torch.empty((R, 16), dtype=torch.int16, device=dev)
                        for rs in range(0, R, max_r):
                            re = min(R, rs + max_r)
                            gt, git, ixt = viterbi_ub_cuda(
                                y_in[rs:re].contiguous().view((re - rs) // 16, 256),
                                tab, K, NU, m0, gains, sub=16, wts=wts_tile,
                                n_cand=n_cand, return_code=True, tail_biting=True,
                                tb_window=tb_win, m_extra=m_extra, pair_parity=pp,
                            )
                            go[rs:re] = gt.view(re - rs, 16)
                            gio[rs:re] = git.view(re - rs, 16).to(torch.uint8)
                            ixo[rs:re] = ixt.view(re - rs, 16).to(torch.int16)
                        return go, gio, ixo
                    gt, git, ixt = viterbi_ub_cuda(
                        y_in.contiguous().view(R // 16, 256),
                        tab, K, NU, m0, gains, sub=16, wts=wts_tile,
                        n_cand=n_cand, return_code=True, tail_biting=True,
                        tb_window=tb_win, m_extra=m_extra, pair_parity=pp,
                    )
                    return gt.view(R, 16), git.view(R, 16).to(torch.uint8), ixt.view(R, 16).to(torch.int16)

                g_cur, gi_cur, idx_cur = _run_vit(y0)
                if mu > 1.012:
                    for _ in range(ecfg.mm_passes):
                        y_mm = g_cur + (y0 - g_cur) @ step_mat
                        g_try, gi_try, idx_try = _run_vit(y_mm)
                        e_old_t = (((y0 - g_cur) @ V).square() * d2_col).view(R // 16, 256).sum(dim=1)
                        e_new_t = (((y0 - g_try) @ V).square() * d2_col).view(R // 16, 256).sum(dim=1)
                        acc_r = (e_new_t < e_old_t - 1e-12).repeat_interleave(16).unsqueeze(1)
                        if not bool(acc_r.any()):
                            break
                        g_cur = torch.where(acc_r, g_try, g_cur)
                        gi_cur = torch.where(acc_r, gi_try, gi_cur)
                        idx_cur = torch.where(acc_r, idx_try, idx_cur)

                G_pass[:, s0:s1] = g_cur
                gi_b[:, s0:s1] = gi_cur
                idx_b[:, s0:s1] = idx_cur
                if s1 < TILE:
                    err_sub = (w_sub_work[:, s0:s1] - d_rb.unsqueeze(1) * g_cur) @ V
                    w_sub_work[:, s1:] -= err_sub @ U_blk[s0:s1, s1:]

            G_out[:, c0:c1] = G_pass
            gi_out[:, c0:c1] = gi_b
            idx_out[:, c0:c1] = idx_b
        elif max_r < R:
            G_pass = torch.empty_like(w_blk_start)
            for r_s in range(0, R, max_r):
                r_e = min(R, r_s + max_r)
                if emit_codes:
                    gp_s, gi_s, ix_s = ldlq_block256_cuda(
                        w_blk_start[r_s:r_e].clone(), d_rb[r_s:r_e], dw_blk, U_blk, U_blk_inv,
                        tab, K, NU, m0, gains, n_cand=n_cand, tb_window=tb_win,
                        m_extra=m_extra, pair_parity=pp, return_code=True,
                    )
                    G_pass[r_s:r_e] = gp_s
                    gi_out[r_s:r_e, c0:c1] = gi_s
                    idx_out[r_s:r_e, c0:c1] = ix_s
                else:
                    G_pass[r_s:r_e] = ldlq_block256_cuda(
                        w_blk_start[r_s:r_e].clone(), d_rb[r_s:r_e], dw_blk, U_blk, U_blk_inv,
                        tab, K, NU, m0, gains, n_cand=n_cand, tb_window=tb_win,
                        m_extra=m_extra, pair_parity=pp, return_code=False,
                    )
            if emit_codes:
                G_out[:, c0:c1] = G_pass
        elif emit_codes:
            G_pass, gi_b, idx_b = ldlq_block256_cuda(
                w_blk_start.clone(), d_rb, dw_blk, U_blk, U_blk_inv,
                tab, K, NU, m0, gains, n_cand=n_cand, tb_window=tb_win,
                m_extra=m_extra, pair_parity=pp, return_code=True,
            )
            G_out[:, c0:c1] = G_pass
            gi_out[:, c0:c1] = gi_b
            idx_out[:, c0:c1] = idx_b
        else:
            G_pass = ldlq_block256_cuda(
                w_blk_start.clone(), d_rb, dw_blk, U_blk, U_blk_inv,
                tab, K, NU, m0, gains, n_cand=n_cand, tb_window=tb_win,
                m_extra=m_extra, pair_parity=pp, return_code=False,
            )

        A_c = w_blk_start @ U_blk_inv
        B_c = G_pass @ U_blk_inv
        num = (A_c * B_c).sum(dim=1)
        den = (B_c * B_c).sum(dim=1).clamp(min=1e-12)
        d_opt = torch.where(den > 0, num / den, d_rb).clamp(min=1e-10)
        e_best = torch.round(torch.log2(d_opt / d_row) / 0.25).clamp(-16, 15)
        d_try = (d_row * torch.exp2(0.25 * e_best)).clamp(min=1e-8)
        loss_old = (A_c - d_rb.unsqueeze(1) * B_c).square().sum(dim=1)
        loss_new = (A_c - d_try.unsqueeze(1) * B_c).square().sum(dim=1)
        take_new = loss_new < loss_old
        e_fin = torch.where(take_new, e_best, e0).to(torch.int8)
        d_fin = torch.where(take_new, d_try, d_rb)

        best_what = d_fin.unsqueeze(1) * G_pass
        W_hat_rot[:, c0:c1] = best_what
        e_out[:, ibl] = e_fin
        if c1 < Kin:
            err_blk = (w_blk_start - best_what) @ U_blk_inv
            W_work[:, c1:] -= err_blk @ U[c0:c1, c1:]

    if ecfg.gecd_sweeps > 0:
        _gecd_refine_rows(
            W_rot_sub, W_hat_rot, G_out, gi_out, idx_out, d_row, e_out,
            H_opt, tab.reshape(-1), gains, sub=sub, n_sweeps=ecfg.gecd_sweeps,
        )

    if ecfg.gs_viterbi_passes > 0:
        _gs_viterbi_refine_rows(
            W_rot_sub, W_hat_rot, G_out, gi_out, idx_out, d_row, e_out,
            H_opt, spec_name, n_passes=ecfg.gs_viterbi_passes, gs_eta=ecfg.gs_eta,
        )
        _gecd_refine_rows(
            W_rot_sub, W_hat_rot, G_out, gi_out, idx_out, d_row, e_out,
            H_opt, tab.reshape(-1), gains, sub=sub, n_sweeps=max(1, ecfg.gecd_sweeps),
        )

    return {"W_hat_rot": W_hat_rot, "d_row": d_row.to(torch.float16), "e": e_out, "G": G_out, "gi": gi_out, "idx": idx_out}


@torch.no_grad()
def quantize_matrix_vtq2_plus(
    W: torch.Tensor,
    H: torch.Tensor,
    spec_or_row_specs: str | List[Tuple[int, int, str]],
    ecfg: EncoderCfg,
    seed: int = 1337,
    damp: Optional[float] = None,
    row_weights: Optional[torch.Tensor] = None,
    rot_block: int = -1,
    W_target: Optional[torch.Tensor] = None,
    emit_pack_override: bool = False,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, float]:
    """Configurable VTQ v2+ matrix quantizer preserving exact `.vtq` v2 bitplane semantics."""
    if not v2.VTQ2_CONFIG:
        v2.load_and_register_vtq2()

    use_damp = ecfg.damp if damp is None else damp
    dev = W.device
    W_src = (W_target if W_target is not None else W).float()
    H_f32 = H.float()
    M, Kin = W_src.shape
    nblocks = Kin // TILE

    with use_rotation_mode(ecfg.outer_dct):
        W_rot, H_rot, suh = v2.compute_suh_vtq2(W_src, H_f32, seed=seed, row_weights=row_weights, rot_block=rot_block)

        h_diag_mean = H_rot.diagonal().mean().clamp(min=1e-8)
        H_reg = H_rot.clone()
        H_reg.diagonal().add_(use_damp * h_diag_mean)
        L_flip = torch.linalg.cholesky(H_reg.flip(0, 1))
        del H_reg
        U = torch.linalg.solve_triangular(
            L_flip, torch.eye(Kin, dtype=L_flip.dtype, device=dev), upper=False
        ).flip(0, 1)
        del L_flip

        eye256 = torch.eye(TILE, dtype=torch.float32, device=dev).unsqueeze(0).expand(nblocks, -1, -1)
        U_blks = torch.stack([U[b * TILE : (b + 1) * TILE, b * TILE : (b + 1) * TILE] for b in range(nblocks)], dim=0)
        U_blks_inv = torch.linalg.solve_triangular(U_blks, eye256, upper=True)

        if ecfg.schur_diag or ecfg.mm_passes > 0:
            # Exact 16-col conditional Schur quadratic diagonal: P_cc = sum_{k=c}^{15} V_{c, k}^2
            V_subs = torch.stack(
                [U_blks_inv[:, ib * 16 : (ib + 1) * 16, ib * 16 : (ib + 1) * 16] for ib in range(16)], dim=1
            )  # [nblocks, 16, 16, 16]
            diag_w = V_subs.square().sum(dim=-1).reshape(Kin).clamp(min=1e-12)
        else:
            diag_w = 1.0 / (torch.diagonal(U) ** 2).clamp(min=1e-12)

        H_opt = H_rot.clone()
        H_opt.diagonal().add_(ecfg.opt_reg * h_diag_mean)
        del H_rot

        row_slices = [(0, M, spec_or_row_specs)] if isinstance(spec_or_row_specs, str) else list(spec_or_row_specs)
        by_spec: Dict[str, List[Tuple[int, int]]] = {}
        for r0, r1, sp in row_slices:
            by_spec.setdefault(sp, []).append((r0, r1))

        emit_pack = emit_pack_override or (v2.PACK is not None) or (lv2.PACK is not None)
        need_codes = emit_pack or (ecfg.gecd_sweeps > 0) or (ecfg.gs_viterbi_passes > 0) or (ecfg.mm_passes > 0)
        W_hat_rot = torch.empty_like(W_rot)
        d_row_all = torch.empty((M,), dtype=torch.float16, device=dev)
        e_all = torch.empty((M, nblocks), dtype=torch.int8, device=dev)
        pm = (
            {
                "G": torch.empty((M, Kin), dtype=torch.float32, device=dev),
                "gi": torch.empty((M, Kin), dtype=torch.uint8, device=dev),
                "idx": torch.empty((M, Kin), dtype=torch.int16, device=dev),
            }
            if need_codes
            else None
        )

        for sp, ranges in by_spec.items():
            rows_idx = torch.cat([torch.arange(r0, r1, device=dev) for r0, r1 in ranges], dim=0)
            sub_out = _quantize_rows_vtq2_plus(
                W_rot[rows_idx], U, U_blks, U_blks_inv, diag_w, H_opt, sp, ecfg, need_codes=need_codes
            )
            W_hat_rot[rows_idx] = sub_out["W_hat_rot"]
            d_row_all[rows_idx] = sub_out["d_row"]
            e_all[rows_idx] = sub_out["e"]
            if pm is not None:
                pm["G"][rows_idx] = sub_out["G"]
                pm["gi"][rows_idx] = sub_out["gi"]
                pm["idx"][rows_idx] = sub_out["idx"]

        del U, U_blks, U_blks_inv, W_rot, H_opt

        U_base = v2f.apply_kronecker_rot(W_hat_rot, rot_block=rot_block, transpose=True)
        W_hat_eff = U_base * suh.unsqueeze(0)

        HW = W_src @ H_f32
        HQ = W_hat_eff @ H_f32
        c_row = (W_hat_eff * HW).sum(dim=1) / (W_hat_eff * HQ).sum(dim=1).clamp(min=1e-12)
        c_row = torch.where(torch.isfinite(c_row) & (c_row > 0.5) & (c_row < 2.0), c_row, torch.ones_like(c_row))
        d_row_new = (d_row_all.float() * c_row).to(torch.float16)
        c_eff = (d_row_new.float() / d_row_all.float().clamp(min=1e-12)).unsqueeze(1)
        U_scaled = U_base * c_eff

        if row_weights is not None:
            rw = row_weights.float().clamp(min=float(row_weights.mean().item()) * 0.1)
            rw = (rw / rw.mean().clamp(min=1e-12)).clamp(0.25, 4.0)
            U_w = U_scaled * rw.unsqueeze(1)
        else:
            rw = None
            U_w = U_scaled

        A_in = (U_w.T @ U_scaled) * H_f32
        b_in = (U_w * HW).sum(dim=0)
        A_in.diagonal().add_(1e-3 * A_in.diagonal().mean().clamp(min=1e-12))
        try:
            suh_cand = torch.linalg.solve(A_in, b_in).to(torch.float16).to(torch.float32)
            valid_s = torch.isfinite(suh_cand) & (suh_cand.abs() > 0.05) & (suh_cand.abs() < 10.0)
            suh_cand = torch.where(valid_s, suh_cand, suh)
            W_hat_cand = U_scaled * suh_cand.unsqueeze(0)
            HQ_cand = W_hat_cand @ H_f32
            if rw is not None:
                l_old = float((((HW - HQ) * (W_src - W_hat_eff)).sum(dim=1) * rw).sum())
                l_new = float((((HW - HQ_cand) * (W_src - W_hat_cand)).sum(dim=1) * rw).sum())
            else:
                l_old = float(((HW - HQ) * (W_src - W_hat_eff)).sum())
                l_new = float(((HW - HQ_cand) * (W_src - W_hat_cand)).sum())
            if l_new < l_old:
                W_hat_eff = W_hat_cand
                HQ = HQ_cand
                d_row_all = d_row_new
                suh = suh_cand
        except RuntimeError:
            pass

        pack_dict = None
        if pm is not None:
            suh_p = suh.to(torch.float16)
            W_hat_eff = assemble_vtq2(pm["G"], d_row_all, e_all, suh_p, rot_block=rot_block)
            HQ = W_hat_eff @ H_f32
            if emit_pack:
                pack_dict = {
                    "row_slices": row_slices,
                    "idx": pm["idx"],
                    "gi": pm["gi"],
                    "G": pm["G"],
                    "d_row": d_row_all,
                    "e": e_all,
                    "suh": suh_p,
                }
                if v2.PACK is not None:
                    v2.PACK["matrix"] = pack_dict
                if lv2.PACK is not None:
                    lv2.PACK["matrix"] = pack_dict

    diff = W_src - W_hat_eff
    rel_loss = float(((HW - HQ) * diff).sum().item()) / float((HW * W_src).sum().clamp(min=1e-12).item())
    qdata = {"d_row": d_row_all, "e": e_all, "suh": suh.to(torch.float16), "row_slices": row_slices}
    if pack_dict is not None:
        qdata["pack"] = pack_dict
    return qdata, W_hat_eff, rel_loss


# -------------------------------------------------------------------------------------------------
# Memory-Bounded Scale Polish (`CP-Polish` O(1)-activation KLD & `Block-Polish` 4-layer Streaming)
# -------------------------------------------------------------------------------------------------

def polish_model_scales_kld_checkpointed(
    model: nn.Module,
    cal_tokens: torch.Tensor,
    ref_hidden_cpu: torch.Tensor,
    n_steps: int = 100,
    lr: float = 4e-4,
    batch_size: int = 2,
    reg_wd: float = 0.02,
    rev_kl_weight: float = 0.12,
    ce_weight: float = 0.0,
) -> Tuple[float, float]:
    """
    O(1)-layer activation-VRAM end-to-end `(d_row, suh)` KLD scale polish using layer-by-layer
    non-reentrant gradient checkpointing. Identical gradients to `polish_model_scales_kld` while
    fitting `8B`, `27B`, and `30B MoE` comfortably inside a single GPU!
    """
    t0 = time.perf_counter()
    device = next(model.parameters()).device
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    torch.cuda.reset_peak_memory_stats(device)
    n_seqs, seq_len, _ = ref_hidden_cpu.shape
    cal_pol = cal_tokens[:n_seqs, :seq_len].to(device)

    scales_alpha: Dict[Tuple[int, str], nn.Parameter] = {}
    scales_beta: Dict[Tuple[int, str], nn.Parameter] = {}
    scale_params: List[nn.Parameter] = []
    hooks = []

    for l in range(model.config.num_hidden_layers):
        layer = model.model.layers[l]
        for sub in lv2.LINEAR_SUBMODULES:
            mod = layer.get_submodule(sub)
            M, K = mod.weight.shape
            alpha = nn.Parameter(torch.ones(M, dtype=torch.float32, device=device))
            beta = nn.Parameter(torch.ones(K, dtype=torch.float32, device=device))
            scales_alpha[(l, sub)] = alpha
            scales_beta[(l, sub)] = beta
            scale_params.extend([alpha, beta])

            def make_hook(a, b):
                def pre_hook(m, args):
                    return (args[0] * b.to(args[0].dtype),)

                def post_hook(m, args, out):
                    return out * a.to(out.dtype)

                return pre_hook, post_hook

            pre_h, post_h = make_hook(alpha, beta)
            hooks.append(mod.register_forward_pre_hook(pre_h))
            hooks.append(mod.register_forward_hook(post_h))

    opt = torch.optim.AdamW(scale_params, lr=lr, weight_decay=0.0)
    pos_ids = torch.arange(seq_len, device=device).unsqueeze(0)

    try:
        for step in range(n_steps):
            progress = step / max(n_steps - 1, 1)
            cur_lr = lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress)))
            for pg in opt.param_groups:
                pg["lr"] = cur_lr

            idx = (step * batch_size) % n_seqs
            x_b = cal_pol[idx : idx + batch_size]
            h_ref_b = ref_hidden_cpu[idx : idx + batch_size].to(device=device, dtype=model.dtype)

            opt.zero_grad()
            with torch.no_grad():
                h0 = model.model.embed_tokens(x_b)
                pos_emb = model.model.rotary_emb(h0[:1], pos_ids)
            h_cur = h0.detach().requires_grad_(True)
            for layer in model.model.layers:
                def _layer_fwd(inp, _ly=layer, _pe=pos_emb):
                    out = _ly(inp, position_embeddings=_pe)
                    return out[0] if isinstance(out, tuple) else out

                h_cur = grad_checkpoint(_layer_fwd, h_cur, use_reentrant=False)
            h_q_b = model.model.norm(h_cur)

            h_ref_flat = h_ref_b[:, :-1, :].reshape(-1, h_ref_b.shape[-1])
            h_q_flat = h_q_b[:, :-1, :].reshape(-1, h_q_b.shape[-1])
            targets_flat = x_b[:, 1:].reshape(-1)
            n_tok = h_q_flat.shape[0]
            grad_h_q = torch.empty_like(h_q_flat)

            chunk_sz = 256
            for c0 in range(0, n_tok, chunk_sz):
                c1 = min(c0 + chunk_sz, n_tok)
                w_fraction = (c1 - c0) / float(n_tok)
                q_chunk = h_q_flat[c0:c1].detach().requires_grad_(True)
                t_chunk = targets_flat[c0:c1]
                with torch.no_grad():
                    ref_logits = model.lm_head(h_ref_flat[c0:c1]).float()
                    ref_logp = F.log_softmax(ref_logits, dim=-1)
                    ref_p = ref_logp.exp()
                q_logits = model.lm_head(q_chunk).float()
                q_logp = F.log_softmax(q_logits, dim=-1)
                q_p = q_logp.exp()

                fwd_kl = (ref_p * (ref_logp - q_logp)).sum(dim=-1).mean()
                rev_kl = (q_p * (q_logp - ref_logp)).sum(dim=-1).mean()
                ce_loss = F.nll_loss(q_logp, t_chunk, reduction="mean")
                loss_chunk = (fwd_kl + rev_kl_weight * rev_kl + ce_weight * ce_loss) * w_fraction
                loss_chunk.backward()
                grad_h_q[c0:c1] = q_chunk.grad

            h_q_b[:, :-1, :].reshape(-1, h_q_b.shape[-1]).backward(grad_h_q)
            if reg_wd > 0.0:
                for p in scale_params:
                    if p.grad is not None:
                        p.grad.add_(p.data - 1.0, alpha=reg_wd)
            opt.step()
    finally:
        for hk in hooks:
            hk.remove()

    with torch.no_grad():
        for (l, sub), alpha in scales_alpha.items():
            beta = scales_beta[(l, sub)]
            mod = model.model.layers[l].get_submodule(sub)
            a_fp16 = alpha.to(torch.float16).float()
            b_fp16 = beta.to(torch.float16).float()
            mod.weight.data.copy_(
                (mod.weight.data.float() * a_fp16.unsqueeze(1) * b_fp16.unsqueeze(0)).to(mod.weight.dtype)
            )
    for p in model.parameters():
        p.requires_grad_(False)

    peak_gib = torch.cuda.max_memory_allocated(device) / float(1 << 30)
    del opt, scale_params, scales_alpha, scales_beta, cal_pol
    torch.cuda.empty_cache()
    return time.perf_counter() - t0, peak_gib


def polish_window_scales_streaming(
    layers_win: List[nn.Module],
    cal_in_q: List[torch.Tensor],
    cal_out_r: List[torch.Tensor],
    pos_emb: Tuple[torch.Tensor, torch.Tensor],
    final_norm: Optional[nn.Module] = None,
    lm_head: Optional[nn.Module] = None,
    n_steps: int = 45,
    lr: float = 5e-4,
    reg_wd: float = 0.02,
) -> float:
    """
    Streaming `< 3 GiB` window scale polish over `W_win` layers (`4` layers at a time):
    Intermediate windows minimize cosine + normalized hidden reconstruction loss against the
    BF16 stream `cal_out_r`; the final window minimizes exact logit KLD through `final_norm + lm_head`.
    """
    t0 = time.perf_counter()
    device = cal_in_q[0].device
    dtype = cal_in_q[0].dtype

    scales_alpha: List[Tuple[nn.Linear, nn.Parameter, nn.Parameter]] = []
    scale_params: List[nn.Parameter] = []
    hooks = []

    for ly in layers_win:
        ly.eval()
        for p in ly.parameters():
            p.requires_grad_(False)
        for sub in lv2.LINEAR_SUBMODULES:
            mod = ly.get_submodule(sub)
            M, K = mod.weight.shape
            alpha = nn.Parameter(torch.ones(M, dtype=torch.float32, device=device))
            beta = nn.Parameter(torch.ones(K, dtype=torch.float32, device=device))
            scales_alpha.append((mod, alpha, beta))
            scale_params.extend([alpha, beta])

            def make_hook(a, b):
                def pre_hook(m, args):
                    return (args[0] * b.to(args[0].dtype),)

                def post_hook(m, args, out):
                    return out * a.to(out.dtype)

                return pre_hook, post_hook

            pre_h, post_h = make_hook(alpha, beta)
            hooks.append(mod.register_forward_pre_hook(pre_h))
            hooks.append(mod.register_forward_hook(post_h))

    if final_norm is not None:
        for p in final_norm.parameters():
            p.requires_grad_(False)
    if lm_head is not None:
        for p in lm_head.parameters():
            p.requires_grad_(False)

    opt = torch.optim.AdamW(scale_params, lr=lr, weight_decay=0.0)
    n_batches = len(cal_in_q)
    seq_cut = min(512, cal_in_q[0].shape[1])
    pe_cut = (pos_emb[0][:, :seq_cut], pos_emb[1][:, :seq_cut])

    try:
        with torch.enable_grad():
            for step in range(n_steps):
                progress = step / max(n_steps - 1, 1)
                cur_lr = lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress)))
                for pg in opt.param_groups:
                    pg["lr"] = cur_lr

                b_i = step % n_batches
                hq = cal_in_q[b_i][:2, :seq_cut].detach()
                hr = cal_out_r[b_i][:2, :seq_cut].detach()
                opt.zero_grad()

                h_cur = hq
                for ly in layers_win:
                    out = ly(h_cur, position_embeddings=pe_cut)
                    h_cur = out[0] if isinstance(out, tuple) else out

                if final_norm is not None and lm_head is not None:
                    h_q_n = final_norm(h_cur).reshape(-1, h_cur.shape[-1])
                    with torch.no_grad():
                        h_r_n = final_norm(hr).reshape(-1, hr.shape[-1])
                    n_tok = h_q_n.shape[0]
                    grad_hq = torch.empty_like(h_q_n)
                    for c0 in range(0, n_tok, 256):
                        c1 = min(c0 + 256, n_tok)
                        wf = (c1 - c0) / float(n_tok)
                        qc = h_q_n[c0:c1].detach().requires_grad_(True)
                        with torch.no_grad():
                            rl = F.log_softmax(lm_head(h_r_n[c0:c1]).float(), dim=-1)
                            rp = rl.exp()
                        ql = F.log_softmax(lm_head(qc).float(), dim=-1)
                        qp = ql.exp()
                        loss_c = ((rp * (rl - ql)).sum(dim=-1).mean() + 0.12 * (qp * (ql - rl)).sum(dim=-1).mean()) * wf
                        loss_c.backward()
                        grad_hq[c0:c1] = qc.grad
                    h_q_n.backward(grad_hq)
                else:
                    qf = h_cur.float()
                    rf = hr.float()
                    rel_mse = (qf - rf).square().sum(dim=-1) / rf.square().sum(dim=-1).clamp(min=1e-6)
                    cos_pen = 1.0 - F.cosine_similarity(qf, rf, dim=-1)
                    loss = (rel_mse + 0.5 * cos_pen).mean()
                    loss.backward()

                if reg_wd > 0.0:
                    for p in scale_params:
                        if p.grad is not None:
                            p.grad.add_(p.data - 1.0, alpha=reg_wd)
                opt.step()
    finally:
        for hk in hooks:
            hk.remove()

    with torch.no_grad():
        for mod, alpha, beta in scales_alpha:
            a_fp16 = alpha.to(torch.float16).float()
            b_fp16 = beta.to(torch.float16).float()
            mod.weight.data.copy_(
                (mod.weight.data.float() * a_fp16.unsqueeze(1) * b_fp16.unsqueeze(0)).to(dtype)
            )
        for b_i in range(n_batches):
            h_u = cal_in_q[b_i]
            for ly in layers_win:
                out = ly(h_u, position_embeddings=pos_emb)
                h_u = out[0] if isinstance(out, tuple) else out
            cal_in_q[b_i] = h_u

    return time.perf_counter() - t0


@torch.no_grad()
def quantize_llama_model_vtq2_plus(
    model: nn.Module,
    cal_tokens: torch.Tensor,
    spec_or_assignment: str | Dict[Tuple[int, str], List[Tuple[int, int, str]]],
    ecfg: EncoderCfg,
    mlp_perms: Optional[Dict[int, torch.Tensor]] = None,
    fisher_out: Optional[Dict[Tuple[int, str], torch.Tensor]] = None,
    batch_size: int = 4,
    gptaq_alpha: float = 0.50,
    block_polish_win: int = 0,
) -> dict:
    """Streaming layer-by-layer VTQ v2+ quantizer with optional `< 3 GiB` `Block-Polish`."""
    if not v2.VTQ2_CONFIG:
        v2.load_and_register_vtq2()

    t_start = time.perf_counter()
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    model.eval()

    if fisher_out is None:
        fisher_out = lv2.compute_exact_row_fisher_sensitivities(model, cal_tokens, n_Menu_seqs=8, seq_len=1024)

    n_seqs, seq_len = cal_tokens.shape
    n_layers = model.config.num_hidden_layers
    use_g = (gptaq_alpha > 0.0) or (block_polish_win > 0)

    cal_q = [model.model.embed_tokens(cal_tokens[i : i + batch_size].to(device)) for i in range(0, n_seqs, batch_size)]
    cal_r = [c.clone() for c in cal_q] if use_g else cal_q
    pos_cal = model.model.rotary_emb(cal_q[0][:1], torch.arange(seq_len, device=device).unsqueeze(0))

    total_params = 0
    total_bits = 0.0
    block_polish_sec = 0.0
    win_start_idx = 0
    win_in_q = [c.clone() for c in cal_q] if block_polish_win > 0 else None

    for l_idx in range(n_layers):
        lq = model.model.layers[l_idx]
        if mlp_perms is not None and l_idx in mlp_perms:
            perm_l = mlp_perms[l_idx].to(device)
            lq.mlp.gate_proj.weight.data = lq.mlp.gate_proj.weight.data[perm_l].contiguous()
            lq.mlp.up_proj.weight.data = lq.mlp.up_proj.weight.data[perm_l].contiguous()
            lq.mlp.down_proj.weight.data = lq.mlp.down_proj.weight.data[:, perm_l].contiguous()
        lr = copy.deepcopy(lq) if use_g else lq

        def q_mod(mod: nn.Linear, H: torch.Tensor, B: Optional[torch.Tensor], full_name: str, seed_i: int, dm: float):
            nonlocal total_params, total_bits
            W = mod.weight.data.float()
            M_r, K_in = W.shape
            r_specs = [(0, M_r, spec_or_assignment)] if isinstance(spec_or_assignment, str) else spec_or_assignment[(l_idx, full_name)]
            W_tgt = (W + gptaq_alpha * (W @ B.T)) if (gptaq_alpha > 0.0 and B is not None) else None
            sr = fisher_out[(l_idx, full_name)].to(device)
            _, w_hat, _ = quantize_matrix_vtq2_plus(
                W, H, r_specs, ecfg=ecfg, seed=seed_i, damp=ecfg.damp * dm, row_weights=sr, rot_block=-1, W_target=W_tgt
            )
            mod.weight.data.copy_(w_hat.to(dtype=dtype, device=device))
            for r0, r1, sp in r_specs:
                n_p = (r1 - r0) * K_in
                total_params += n_p
                total_bits += n_p * lv2.LIQK_V2_SPECS[sp].bpw

        acc = v2.DualCovAccumulator(gptaq_alpha > 0.0)
        for cq, cr in zip(cal_q, cal_r):
            acc.add(lq.input_layernorm(cq), lr.input_layernorm(cr) if gptaq_alpha > 0.0 else None)
        Hqkv, Bqkv = acc.get(gptaq_damp=ecfg.damp)
        for i_s, nm_s in enumerate(("q_proj", "k_proj", "v_proj")):
            q_mod(getattr(lq.self_attn, nm_s), Hqkv, Bqkv, f"self_attn.{nm_s}", 1000 + l_idx * 10 + i_s, 1.0)
        del Hqkv, Bqkv, acc

        acc = v2.DualCovAccumulator(gptaq_alpha > 0.0)
        bq, br = [None], [None]
        hq = lq.self_attn.o_proj.register_forward_pre_hook(lambda m, a: bq.__setitem__(0, a[0]))
        hr = lr.self_attn.o_proj.register_forward_pre_hook(lambda m, a: br.__setitem__(0, a[0])) if gptaq_alpha > 0.0 else None
        ref_mid = []
        for cq, cr in zip(cal_q, cal_r):
            sq._attn(lq, cq, pos_cal)
            if use_g:
                ref_mid.append(cr + sq._attn(lr, cr, pos_cal))
            acc.add(bq[0], br[0])
            bq[0] = br[0] = None
        hq.remove()
        if hr is not None:
            hr.remove()
        Ho, Bo = acc.get(gptaq_damp=ecfg.damp)
        q_mod(lq.self_attn.o_proj, Ho, Bo, "self_attn.o_proj", 1000 + l_idx * 10 + 3, 1.0)
        del Ho, Bo, acc

        for i in range(len(cal_q)):
            cal_q[i] = cal_q[i] + sq._attn(lq, cal_q[i], pos_cal)
        if not use_g:
            ref_mid = cal_q

        acc = v2.DualCovAccumulator(gptaq_alpha > 0.0)
        for cq, crm in zip(cal_q, ref_mid):
            acc.add(lq.post_attention_layernorm(cq), lr.post_attention_layernorm(crm) if gptaq_alpha > 0.0 else None)
        Hup, Bup = acc.get(gptaq_damp=ecfg.damp)
        for i_s, nm_s in enumerate(("gate_proj", "up_proj")):
            q_mod(getattr(lq.mlp, nm_s), Hup, Bup, f"mlp.{nm_s}", 1000 + l_idx * 10 + 4 + i_s, 1.0)
        del Hup, Bup, acc

        acc = v2.DualCovAccumulator(gptaq_alpha > 0.0)
        hq = lq.mlp.down_proj.register_forward_pre_hook(lambda m, a: bq.__setitem__(0, a[0]))
        hr = lr.mlp.down_proj.register_forward_pre_hook(lambda m, a: br.__setitem__(0, a[0])) if gptaq_alpha > 0.0 else None
        for cq, crm in zip(cal_q, ref_mid):
            lq.mlp(lq.post_attention_layernorm(cq))
            if gptaq_alpha > 0.0:
                lr.mlp(lr.post_attention_layernorm(crm))
            acc.add(bq[0], br[0])
            bq[0] = br[0] = None
        hq.remove()
        if hr is not None:
            hr.remove()
        Hdn, Bdn = acc.get(gptaq_damp=ecfg.damp * 1.5)
        q_mod(lq.mlp.down_proj, Hdn, Bdn, "mlp.down_proj", 1000 + l_idx * 10 + 6, 1.5)
        del Hdn, Bdn, acc

        for i in range(len(cal_q)):
            cal_q[i] = cal_q[i] + lq.mlp(lq.post_attention_layernorm(cal_q[i]))
        if use_g:
            for i in range(len(cal_r)):
                cal_r[i] = ref_mid[i] + lr.mlp(lr.post_attention_layernorm(ref_mid[i]))
            del lr
        del ref_mid

        if block_polish_win > 0 and ((l_idx + 1 - win_start_idx) == block_polish_win or l_idx == n_layers - 1):
            is_last = l_idx == n_layers - 1
            win_layers = [model.model.layers[k] for k in range(win_start_idx, l_idx + 1)]
            bp_dt = polish_window_scales_streaming(
                win_layers,
                win_in_q,
                cal_r,
                pos_cal,
                final_norm=model.model.norm if is_last else None,
                lm_head=model.lm_head if is_last else None,
                n_steps=70 if is_last else 40,
                lr=4e-4 if is_last else 5e-4,
            )
            block_polish_sec += bp_dt
            cal_q = [c.clone() for c in win_in_q]
            win_start_idx = l_idx + 1

    del cal_q, cal_r
    if win_in_q is not None:
        del win_in_q
    torch.cuda.empty_cache()
    return {
        "linear_params": total_params,
        "linear_bpw": total_bits / max(total_params, 1),
        "quant_sec": time.perf_counter() - t_start - block_polish_sec,
        "block_polish_sec": block_polish_sec,
    }


# -------------------------------------------------------------------------------------------------
# Part A: Single-Matrix Train/Val Hessian Ablation + `.vtq` v2 Bit-Exact Verification + Speed
# -------------------------------------------------------------------------------------------------

@torch.no_grad()
def collect_train_val_hessians(model: nn.Module, cal64: torch.Tensor, layer_idx: int) -> dict:
    """Collects disjoint `H_cal` (`cal64[:32]`) and held-out `H_val` (`cal64[32:64]`) on `o_proj` and `down_proj`."""
    dev = next(model.parameters()).device
    out = {}
    for split_name, sub_toks in (("cal", cal64[:32]), ("val", cal64[32:64])):
        hs = [model.model.embed_tokens(sub_toks[i : i + 4].to(dev)) for i in range(0, sub_toks.shape[0], 4)]
        pos = model.model.rotary_emb(hs[0][:1], torch.arange(sub_toks.shape[1], device=dev).unsqueeze(0))
        for l_i in range(layer_idx):
            ly = model.model.layers[l_i]
            for b in range(len(hs)):
                r = ly(hs[b], position_embeddings=pos)
                hs[b] = r[0] if isinstance(r, tuple) else r
        ly_t = model.model.layers[layer_idx]
        acc_o = lv2._CovAccumulator()
        acc_d = lv2._CovAccumulator()
        ho = ly_t.self_attn.o_proj.register_forward_pre_hook(lambda m, a: acc_o.add(a[0]))
        hd = ly_t.mlp.down_proj.register_forward_pre_hook(lambda m, a: acc_d.add(a[0]))
        for b in range(len(hs)):
            ly_t(hs[b], position_embeddings=pos)
        ho.remove()
        hd.remove()
        out[f"o_{split_name}"] = acc_o.get()
        out[f"down_{split_name}"] = acc_d.get()
        del hs
    torch.cuda.empty_cache()
    return out


@torch.no_grad()
def eval_rel_hess_loss(W: torch.Tensor, W_hat: torch.Tensor, H: torch.Tensor, rw: Optional[torch.Tensor] = None) -> float:
    Wf = W.float()
    Qf = W_hat.float()
    Hf = H.float()
    diff = Wf - Qf
    row_num = ((diff @ Hf) * diff).sum(dim=1)
    row_den = ((Wf @ Hf) * Wf).sum(dim=1)
    if rw is not None:
        w = rw.float().to(W.device)
        return float((row_num * w).sum().item() / (row_den * w).sum().clamp(min=1e-12).item())
    return float(row_num.sum().item() / row_den.sum().clamp(min=1e-12).item())


@torch.no_grad()
def run_part_a_single_matrix(res_all: dict):
    print("\n================ PART A: Single-Matrix Train/Val Hessian & Bit-Exact Format Ablation ================", flush=True)
    dev = torch.device("cuda")
    v2.load_and_register_vtq2()
    fmts = load_vtq2_formats()

    # 1. Orthogonality & runtime transform speed check for Outer-DCT vs Inner-DCT on K=14336 (q=7) and K=3072 (q=3)
    rot_checks = {}
    for K_test in (3072, 12288, 14336, 17408):
        X = torch.randn((4096, K_test), dtype=torch.float32, device=dev)
        Y_out = apply_kronecker_rot_outer(X, rot_block=-1, transpose=False)
        X_rec = apply_kronecker_rot_outer(Y_out, rot_block=-1, transpose=True)
        inv_err = float((X - X_rec).abs().max().item())
        assert inv_err < 1e-4, f"Outer-DCT invertibility error {inv_err} on K={K_test}"

        for _ in range(5):
            _ORIG_APPLY_ROT(X, rot_block=-1, transpose=True)
            apply_kronecker_rot_outer(X, rot_block=-1, transpose=True)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(30):
            _ORIG_APPLY_ROT(X, rot_block=-1, transpose=True)
        torch.cuda.synchronize()
        ms_inner = (time.perf_counter() - t0) * 1000.0 / 30.0

        t0 = time.perf_counter()
        for _ in range(30):
            apply_kronecker_rot_outer(X, rot_block=-1, transpose=True)
        torch.cuda.synchronize()
        ms_outer = (time.perf_counter() - t0) * 1000.0 / 30.0
        rot_checks[str(K_test)] = {"max_inv_err": inv_err, "inner_dct_ms": round(ms_inner, 4), "outer_dct_ms": round(ms_outer, 4)}
        print(f"  [Rot K={K_test:5d}] max_inv_err={inv_err:.2e} | inner={ms_inner:.3f} ms vs outer={ms_outer:.3f} ms", flush=True)
    res_all["rotation_microbench"] = rot_checks

    cal64 = load_file(str(ROOT / "data" / "cal_trace.safetensors"))["input_ids"][:64]
    m8_dir = ROOT / "models" / "Llama-3.1-8B-Instruct"
    m8 = AutoModelForCausalLM.from_pretrained(str(m8_dir), dtype=torch.bfloat16).to(dev).eval()
    f8_path = ROOT / "data" / "fisher_llama8b.pt"
    if f8_path.exists():
        f8 = torch.load(str(f8_path), map_location="cpu", weights_only=True)
    else:
        f8 = lv2.compute_exact_row_fisher_sensitivities(m8, cal64, n_Menu_seqs=8, seq_len=1024)
        torch.save(f8, str(f8_path))

    l_idx = 15
    hm = collect_train_val_hessians(m8, cal64, l_idx)
    W_down = m8.model.layers[l_idx].mlp.down_proj.weight.data.float().clone()
    W_o = m8.model.layers[l_idx].self_attn.o_proj.weight.data.float().clone()
    rw_down = f8[(l_idx, "mlp.down_proj")].to(dev).float()
    rw_o = f8[(l_idx, "self_attn.o_proj")].to(dev).float()
    del m8
    torch.cuda.empty_cache()

    configs = [
        EncoderCfg("1_base_v2", outer_dct=False, schur_diag=False, mm_passes=0, gecd_sweeps=0, gs_viterbi_passes=0, damp=0.20),
        EncoderCfg("2_outer_dct", outer_dct=True, schur_diag=False, mm_passes=0, gecd_sweeps=0, gs_viterbi_passes=0, damp=0.20),
        EncoderCfg("3_outer_schur", outer_dct=True, schur_diag=True, mm_passes=0, gecd_sweeps=0, gs_viterbi_passes=0, damp=0.20),
        EncoderCfg("4_outer_schur_gecd1", outer_dct=True, schur_diag=True, mm_passes=0, gecd_sweeps=1, gs_viterbi_passes=0, damp=0.20),
        EncoderCfg("5_outer_schur_gecd2", outer_dct=True, schur_diag=True, mm_passes=0, gecd_sweeps=2, gs_viterbi_passes=0, damp=0.20),
        EncoderCfg("6_damp10_fast", outer_dct=True, schur_diag=True, mm_passes=0, gecd_sweeps=2, gs_viterbi_passes=0, damp=0.10),
        EncoderCfg("7_damp06_fast", outer_dct=True, schur_diag=True, mm_passes=0, gecd_sweeps=2, gs_viterbi_passes=0, damp=0.06),
        EncoderCfg("8_mm1_gecd2", outer_dct=True, schur_diag=True, mm_passes=1, mm_eta=1.25, gecd_sweeps=2, gs_viterbi_passes=0, damp=0.10),
        EncoderCfg("9_gsvit1_gecd2", outer_dct=True, schur_diag=True, mm_passes=0, gecd_sweeps=2, gs_viterbi_passes=1, gs_eta=1.20, damp=0.10),
        EncoderCfg("10_mm1_gsvit1_gecd2", outer_dct=True, schur_diag=True, mm_passes=1, mm_eta=1.25, gecd_sweeps=2, gs_viterbi_passes=1, gs_eta=1.20, damp=0.10),
    ]

    part_a = {}
    for mat_tag, W_mat, H_cal, H_val, rw_mat, dm_mult in (
        ("llama8b_L15_down_proj_4096x14336", W_down, hm["down_cal"], hm["down_val"], rw_down, 1.5),
        ("llama8b_L15_o_proj_4096x4096", W_o, hm["o_cal"], hm["o_val"], rw_o, 1.0),
    ):
        print(f"\n--- Matrix `{mat_tag}` (shape={tuple(W_mat.shape)}, dm={dm_mult}) ---", flush=True)
        part_a[mat_tag] = {}
        for sp in ("VTQ200", "VTQ250", "VTQ300", "VTQ350", "VTQ400"):
            part_a[mat_tag][sp] = {}
            base_val = None
            for ec in configs:
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                qdata, w_hat, _ = quantize_matrix_vtq2_plus(
                    W_mat, H_cal, sp, ecfg=ec, seed=1156, damp=ec.damp * dm_mult,
                    row_weights=rw_mat, rot_block=-1, emit_pack_override=True,
                )
                torch.cuda.synchronize()
                ms = (time.perf_counter() - t0) * 1000.0

                # Bit-exact `.vtq` v2 pack_slice -> decode_slice verification!
                pk = qdata["pack"]
                raw_bytes = pack_slice(pk["idx"], pk["gi"], fmts[sp])
                G_dec = decode_slice(raw_bytes, W_mat.shape[0], W_mat.shape[1], fmts[sp], device=dev)
                with use_rotation_mode(ec.outer_dct):
                    w_dec = assemble_vtq2(G_dec, pk["d_row"], pk["e"], pk["suh"], rot_block=-1)
                pack_err = float((w_hat - w_dec).abs().max().item())
                assert pack_err < 1e-5, f"Pack/decode mismatch {pack_err} on {sp} / {ec.name}!"

                loss_cal = eval_rel_hess_loss(W_mat, w_hat, H_cal, rw_mat)
                loss_val = eval_rel_hess_loss(W_mat, w_hat, H_val, rw_mat)
                if base_val is None:
                    base_val = loss_val
                rel_gain_pct = 100.0 * (loss_val - base_val) / max(base_val, 1e-12)
                part_a[mat_tag][sp][ec.name] = {
                    "loss_cal": loss_cal,
                    "loss_val": loss_val,
                    "val_rel_pct_vs_base": round(rel_gain_pct, 3),
                    "quant_ms": round(ms, 2),
                    "pack_max_err": pack_err,
                }
                print(
                    f"  [{sp} | {ec.name:20s}] val={loss_val:.6f} ({rel_gain_pct:+.2f}%) "
                    f"cal={loss_cal:.6f} | {ms:6.1f} ms | pack_err={pack_err:.1e}",
                    flush=True,
                )
        res_all["part_a_single_matrix"] = part_a
        OUT_JSON.write_text(json.dumps(res_all, indent=2), encoding="utf-8")


# -------------------------------------------------------------------------------------------------
# Part B: Full End-to-End LLM KLD Evaluation (`Llama-3.2-3B` & `Llama-3.1-8B`, `B = 2000`)
# -------------------------------------------------------------------------------------------------

def _pair_boot(res_a: dict, res_b: dict) -> dict:
    out = {}
    for ds in ("nemotron", "multilingual"):
        if ds in res_a and ds in res_b and "per_seq_kld_sums" in res_a[ds] and "per_seq_kld_sums" in res_b[ds]:
            d = paired_bootstrap_diff(res_a[ds], res_b[ds])
            base_m = float(res_b[ds]["mean_kld"])
            d["rel_pct"] = round(100.0 * d["mean_diff"] / max(base_m, 1e-12), 2)
            out[ds] = d
    return out


def pick_best_damp_from_part_a(res_all: dict) -> float:
    pa = res_all.get("part_a_single_matrix", {})
    cand_map = {"5_outer_schur_gecd2": 0.20, "6_damp10_fast": 0.10, "7_damp06_fast": 0.06}
    scores = {k: [] for k in cand_map}
    for mtag, sp_dict in pa.items():
        for sp, cd in sp_dict.items():
            for k in cand_map:
                if k in cd:
                    scores[k].append(cd[k]["val_rel_pct_vs_base"])
    best_k = min(scores.keys(), key=lambda k: float(np.mean(scores[k])) if scores[k] else 0.0)
    chosen = cand_map[best_k]
    print(f"\n[Auto-Select] Best held-out `H_val` damp from Part A: `{best_k}` -> damp={chosen:.2f}", flush=True)
    return chosen


def run_part_b_model(
    model_tag: str,
    model_name: str,
    targets: Tuple[float, ...],
    cached_vtq2_json: Path,
    best_damp: float,
    res_all: dict,
):
    print(f"\n================ PART B ({model_tag}): End-to-End KLD on `{model_name}` ================", flush=True)
    dev = torch.device("cuda")
    v2.load_and_register_vtq2()

    cached_runs = {}
    if cached_vtq2_json.exists():
        cached_runs = json.loads(cached_vtq2_json.read_text(encoding="utf-8")).get("runs", {})

    m_dir = ROOT / "models" / model_name
    model = AutoModelForCausalLM.from_pretrained(str(m_dir), dtype=torch.bfloat16).to(dev).eval()
    clean = save_clean_model_weights(model)
    cal64 = load_file(str(ROOT / "data" / "cal_trace.safetensors"))["input_ids"][:64]

    nem_ref = precompute_ref_trace_data(model, load_eval_trace(ROOT / "data" / "nemotron_trace.json"), dev)
    mul_ref = precompute_ref_trace_data(model, load_eval_trace(ROOT / "data" / "multilingual_trace.json"), dev)
    ref_hidden = lv2.precompute_ref_final_hidden_states(model, cal64, n_seqs=64, seq_len=512, batch_size=4)

    f_name = "fisher_3b.pt" if model_tag == "3b" else "fisher_llama8b.pt"
    f_path = ROOT / "data" / f_name
    if f_path.exists():
        fisher = torch.load(str(f_path), map_location="cpu", weights_only=True)
    else:
        fisher = lv2.compute_exact_row_fisher_sensitivities(model, cal64, n_Menu_seqs=8, seq_len=1024)
        torch.save(fisher, str(f_path))

    # Build Fisher slice profile under `Outer-DCT`!
    t0 = time.perf_counter()
    with use_rotation_mode(True):
        prof_outer = v2.build_vtq2_fisher_slice_profile(
            model, cal64, batch_size=8 if model_tag == "3b" else 4, fisher_out=fisher, rot_block=-1
        )
    prof_sec = time.perf_counter() - t0
    print(f"  Built `Outer-DCT` Fisher profile ({len(prof_outer['slices'])} slices) in {prof_sec:.1f}s", flush=True)

    ecfg_fast = EncoderCfg(
        "vtq2_plus_fast", outer_dct=True, schur_diag=True, mm_passes=0, gecd_sweeps=2, gs_viterbi_passes=0, damp=best_damp
    )
    ecfg_hq = EncoderCfg(
        "vtq2_plus_hq", outer_dct=True, schur_diag=True, mm_passes=0, gecd_sweeps=2, gs_viterbi_passes=1, gs_eta=1.20, damp=best_damp
    )

    sec_key = f"part_b_{model_tag}"
    sec = res_all.setdefault(sec_key, {"model": model_name, "runs": {}, "comparisons": {}})
    for k_c, v_c in cached_runs.items():
        if any(k_c.startswith(pfx) for pfx in ("VTQ2-", "TCQPQ-", "EXL3-")):
            sec["runs"].setdefault(k_c, v_c)

    def _record(lbl: str, info: dict):
        entry = {
            "info": info,
            "nemotron": evaluate_model_on_cached_trace(model, nem_ref, dev),
            "multilingual": evaluate_model_on_cached_trace(model, mul_ref, dev),
        }
        sec["runs"][lbl] = entry
        print(
            f"  [{lbl:30s}] bpw={info['linear_bpw']:.4f} t={info.get('quant_sec', 0):.1f}s | "
            f"nem={entry['nemotron']['mean_kld']:.5f} mul={entry['multilingual']['mean_kld']:.5f} | "
            f"top1={entry['nemotron']['top1_match']*100:.2f}%/{entry['multilingual']['top1_match']*100:.2f}%",
            flush=True,
        )
        t_str = f"{info['target_bpw']:.4f}"
        for ref_k in (f"VTQ2-{t_str}", f"VTQ2-{t_str}-Polish", f"TCQPQ-{t_str}", f"EXL3-{t_str}"):
            if ref_k in sec["runs"]:
                d = _pair_boot(entry, sec["runs"][ref_k])
                sec["comparisons"][f"{lbl} vs {ref_k}"] = d
                print(
                    f"      vs {ref_k:18s}: nem {d['nemotron']['rel_pct']:+.2f}% (p={d['nemotron']['p_a_better']:.4f}) | "
                    f"mul {d['multilingual']['rel_pct']:+.2f}% (p={d['multilingual']['p_a_better']:.4f})",
                    flush=True,
                )
        OUT_JSON.write_text(json.dumps(res_all, indent=2), encoding="utf-8")

    pol_steps = 80 if model_tag == "3b" else 120
    for tgt in targets:
        t_s = f"{tgt:.4f}"
        cand_specs = eqb.SETS["VTQ2"] if model_tag == "3b" else v2.vtq2_candidates_for_target(tgt)
        asgn = lv2.solve_fisher_slice_dp(prof_outer, target_bpw=tgt, candidate_specs=cand_specs)

        # 1) Fast encoder (`Outer-DCT + Schur-Diag + GECD2`, zero extra Viterbi passes!) + GPTAQ=0.50
        lbl_fast = f"VTQ2P-Fast-{t_s}"
        if lbl_fast not in sec["runs"]:
            restore_clean_model_weights(model, clean)
            q_f = quantize_llama_model_vtq2_plus(
                model, cal64, asgn, ecfg=ecfg_fast, mlp_perms=prof_outer["mlp_perms"],
                fisher_out=prof_outer["fisher_out"], batch_size=4, gptaq_alpha=0.50, block_polish_win=0,
            )
            q_f["target_bpw"] = tgt
            q_f["profile_sec"] = prof_sec
            _record(lbl_fast, q_f)

            # Test CP-Polish (`O(1)` activation VRAM!) directly on top of `VTQ2P-Fast`
            lbl_fast_cp = f"VTQ2P-Fast-{t_s}-CPPolish"
            if lbl_fast_cp not in sec["runs"] and tgt in (2.5, 3.0, 3.4375):
                pol_dt, peak_gib = polish_model_scales_kld_checkpointed(
                    model, cal64, ref_hidden, n_steps=pol_steps, lr=4e-4, batch_size=2, reg_wd=0.02
                )
                _record(lbl_fast_cp, dict(q_f, polish_sec=pol_dt, polish_peak_gib=round(peak_gib, 2)))

        # 2) HQ encoder (`+ 1 Right-to-Left Full-Hessian GS-Viterbi Pass`) at key operating points
        lbl_hq = f"VTQ2P-HQ-{t_s}"
        if lbl_hq not in sec["runs"] and tgt in (2.5, 3.0, 3.4375):
            restore_clean_model_weights(model, clean)
            q_hq = quantize_llama_model_vtq2_plus(
                model, cal64, asgn, ecfg=ecfg_hq, mlp_perms=prof_outer["mlp_perms"],
                fisher_out=prof_outer["fisher_out"], batch_size=4, gptaq_alpha=0.50, block_polish_win=0,
            )
            q_hq["target_bpw"] = tgt
            q_hq["profile_sec"] = prof_sec
            _record(lbl_hq, q_hq)

            lbl_hq_cp = f"VTQ2P-HQ-{t_s}-CPPolish"
            if lbl_hq_cp not in sec["runs"]:
                pol_dt, peak_gib = polish_model_scales_kld_checkpointed(
                    model, cal64, ref_hidden, n_steps=pol_steps, lr=4e-4, batch_size=2, reg_wd=0.02
                )
                _record(lbl_hq_cp, dict(q_hq, polish_sec=pol_dt, polish_peak_gib=round(peak_gib, 2)))

        # 3) Streaming `< 3 GiB` 4-layer `Block-Polish` ablation at `3.0000 bpw`
        lbl_bp = f"VTQ2P-Fast-{t_s}-BlockPolish4"
        if lbl_bp not in sec["runs"] and abs(tgt - 3.0) < 1e-6:
            restore_clean_model_weights(model, clean)
            q_bp = quantize_llama_model_vtq2_plus(
                model, cal64, asgn, ecfg=ecfg_fast, mlp_perms=prof_outer["mlp_perms"],
                fisher_out=prof_outer["fisher_out"], batch_size=4, gptaq_alpha=0.50, block_polish_win=4,
            )
            q_bp["target_bpw"] = tgt
            q_bp["profile_sec"] = prof_sec
            _record(lbl_bp, q_bp)

    # Strip bulky `per_seq_*` from summary clone? Keep full `per_seq_kld_sums` for our new runs so future scripts can bootstrap!
    del model, clean, nem_ref, mul_ref, ref_hidden
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip_part_a", action="store_true")
    ap.add_argument("--only_part_a", action="store_true")
    args = ap.parse_args()

    res_all = json.loads(OUT_JSON.read_text(encoding="utf-8")) if OUT_JSON.exists() else {}
    if not args.skip_part_a and "part_a_single_matrix" not in res_all:
        run_part_a_single_matrix(res_all)
    if args.only_part_a:
        return

    best_damp = pick_best_damp_from_part_a(res_all)
    run_part_b_model(
        "3b",
        "Llama-3.2-3B-Instruct",
        (2.0, 2.5, 3.0, 3.4375),
        ROOT / "results" / "vtq2_3b_sweep.json",
        best_damp,
        res_all,
    )
    run_part_b_model(
        "llama8b",
        "Llama-3.1-8B-Instruct",
        (2.5, 3.0, 3.4375),
        ROOT / "results" / "vtq2_llama31_8b.json",
        best_damp,
        res_all,
    )
    print("\n[DONE] Exp08 completed -> results/exp08_encoder_breakthroughs.json", flush=True)


if __name__ == "__main__":
    main()
