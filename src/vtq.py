"""
VTQ Production Quantizer, Hardware-Exact Table Trainer & Multi-Architecture Pipeline (`src/vtq.py`):
  1. C++/CUDA wrappers (`viterbi_vtq_cuda`, `ldlq_vtq_block256_cuda`) over `src/cuda/tcq_viterbi.cu`:
     - `16x16` intra-subblock 256-step 2-pass tail-biting 1024-state (`nu = 10`) radix-4 trellis (one trellis per
       16-column group `g` spanning all 16 rows of a `16x256` super-tile),
     - Exact Bellman joint `(gain, 1024-state)` subblock transition (`sub = 16`, `G = 2^h <= 8`, `n_cand = 4` when `G > 4`),
     - Fused 16-stage `Block16-LDLQ` with causal Cholesky feedback across
       uncoded (`frac = 0`), Gosset `[8, 4, 4]` (`frac = 1`), `D4` `[4, 3, 2]` (`frac = 2`), and `D4_eq` `[4, 1, 4]` (`frac = 3`).
  2. `train_vtq_tables`:
     - Jointly trains all 22 hardware-exact `uint8`/`int8` codebooks (`qmax = 127`, `tab / 64.0f`) and `uint8`
       subblock gains (`gain_u8 / 64.0f`, `sub = 16`) via GPU 1024-state Trellis Lloyd-Max + discrete coordinate search,
     - Calibrates `SPEC_DISTORTION_RATIO` on real `Llama-3.2-3B-Instruct` layers.
  3. `quantize_matrix_vtq`:
     - Canonical Kronecker transform `R_K = H_{2^a} (x) DCT-II_q` (`rot_block = -1`) or Block-256 FWHT (`256`),
     - `suh` preconditioning `(p_w = 0.50, p_h = -0.25)`,
     - Natural-column `Block16-LDLQ` + intra-block `256x256` Cholesky residual Viterbi + full-range `uint8 u_sb` (`<= 248`) +
       two-sided regularized Gauss-Seidel `Block-LDLQ` + global closed-form `(c_row -> fp16 d_row, fp16 suh)` solve,
     - Bit-exact `assemble_vtq` reconstruction & `PACK["matrix"]` emission for `VtqWriter`.
  4. `build_vtq_fisher_slice_profile`, `quantize_llama_model_vtq`, `polish_model_scales_kld_checkpointed`,
     `quantize_rings_vtq`, and `install_vtq_hooks` (`streaming_quant.py` & `hybrid_quant.py`).
"""

from __future__ import annotations

import argparse
import copy
import ctypes
import json
import math
import os
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint

import src.liqk_v2 as lv2
from src.iqk_format import make_rademacher_signs
from src.tcq_proto import ldlq_block256_cuda, viterbi_ub_cuda
from src.vtq_v2 import DualCovAccumulator
from src.vtq_stream_cov import ConsumableCov
from src.vtq_format import (
    FMT_META,
    GAIN_UNIT,
    SUPER_ROWS,
    TAB_UNIT,
    TILE,
    U_SB_UNIT,
    VtqFormat,
    _at2_indices_vtq,
    apply_vtq_rot,
    assemble_vtq,
    format_json_vtq,
    hw_to_canon_u8_vtq,
    load_vtq_formats,
    reconstruct_lut_from_hw_vtq,
)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TABLES_PATH = ROOT / "results" / "vtq_tables.json"

PACK: dict | None = None
_MEM_MARKS: Optional[List[Tuple[str, float]]] = None  # set to `[]` to record peak VRAM (GiB) per encoder stage
VTQ_CONFIG: Dict[str, dict] = {}
VTQ_MENU: List[str] = []
_GPU_TAB_CACHE: Dict[Tuple[str, torch.device], Tuple[torch.Tensor, torch.Tensor]] = {}

# Ordered specification table `(name, fmt_code)` covering `1.594 .. 6.219 bpw` in ~0.25 bpw steps:
# All 20 active specs in `VTQ_MENU` (`VTQ_150 .. VTQ_600`) use full `nl = 16` (`[359, 40, 475]`) with
# `s8_32`, `u8`, `at2_16_16` (`K=4`), and `at2z_32_32` (`K=5, 6`), while `VTQ_300_FAST` (`nl=8`) and
# `VTQ_500_FAST` (`nl=4`) provide optional ultra-fast CPU decode modes.
VTQ_SPECS_DEF: List[Tuple[str, int]] = [
    ("VTQ_150",      15),  # 1.59375 bpw (K=2, Gosset84, nl=16, h=1, s8_32)
    ("VTQ_175",      17),  # 1.84375 bpw (K=2, D4,       nl=16, h=1, s8_32)
    ("VTQ_200",      20),  # 2.09375 bpw (K=2, plain,    nl=16, h=1, u8 16B)
    ("VTQ_225",      22),  # 2.34375 bpw (K=3, D4_eq,    nl=16, h=1, u8 32B)
    ("VTQ_250",      25),  # 2.59375 bpw (K=3, Gosset84, nl=16, h=1, u8 32B)
    ("VTQ_250_H2",   24),  # 2.65625 bpw (K=3, Gosset84, nl=16, h=2, u8 32B)
    ("VTQ_275",      26),  # 2.84375 bpw (K=3, D4,       nl=16, h=1, u8 32B)
    ("VTQ_275_H2",   27),  # 2.90625 bpw (K=3, D4,       nl=16, h=2, u8 32B)
    ("VTQ_300_FAST", 30),  # 3.15625 bpw (K=3, plain,    nl=8,  h=2, u8 16B)
    ("VTQ_300",      31),  # 3.15625 bpw (K=3, plain,    nl=16, h=2, u8 32B)
    ("VTQ_325",      32),  # 3.40625 bpw (K=4, D4_eq,    nl=16, h=2, at2_16_16)
    ("VTQ_350",      35),  # 3.65625 bpw (K=4, Gosset84, nl=16, h=2, at2_16_16)
    ("VTQ_375",      37),  # 3.90625 bpw (K=4, D4,       nl=16, h=2, at2_16_16)
    ("VTQ_400",      40),  # 4.21875 bpw (K=4, plain,    nl=16, h=3, at2_16_16)
    ("VTQ_425",      42),  # 4.40625 bpw (K=5, D4_eq,    nl=16, h=2, at2z_32_32)
    ("VTQ_450",      45),  # 4.65625 bpw (K=5, Gosset84, nl=16, h=2, at2z_32_32)
    ("VTQ_475",      47),  # 4.90625 bpw (K=5, D4,       nl=16, h=2, at2z_32_32)
    ("VTQ_500_FAST", 50),  # 5.21875 bpw (K=5, plain,    nl=4,  h=3, u8 32B)
    ("VTQ_500",      51),  # 5.21875 bpw (K=5, plain,    nl=16, h=3, at2z_32_32)
    ("VTQ_550",      55),  # 5.71875 bpw (K=6, Gosset84, nl=16, h=3, at2z_32_32)
    ("VTQ_575",      57),  # 5.96875 bpw (K=6, D4,       nl=16, h=3, at2z_32_32)
    ("VTQ_600",      60),  # 6.21875 bpw (K=6, plain,    nl=16, h=3, at2z_32_32)
]

def _mem_mark(tag: str):
    """Diagnostics only: records `torch.cuda.max_memory_allocated()` (GiB, since the last mark) when `_MEM_MARKS` is a list."""
    if _MEM_MARKS is not None:
        _MEM_MARKS.append((tag, torch.cuda.max_memory_allocated() / 2**30))
        torch.cuda.reset_peak_memory_stats()


_FRAC_TO_TCQ_PP = {0: 0, 1: 4, 2: 2, 3: 3}
_NL_TO_M_EXTRA = {16: (40, 475), 8: (462,), 4: ()}


# =================================================================================================
# 1. Fast Radix-4 `__ballot_sync` CUDA Viterbi & `16x16` Intra-Subblock `Block16-LDLQ` Wrappers
# =================================================================================================

def _code_to_vtq_pack(
    gi_2d: torch.Tensor,     # uint8 [M, 256]
    idx_2d: torch.Tensor,    # int16/int32 [M, 256] (`lb * U + bu`)
    U: int,
    return_labels: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Converts `(gi_2d, idx_2d)` (`[M, 256]` natural order) into `(g_sub, c_all, u_all, lb_out)` for `vtq_st16_k*_t`."""
    M = idx_2d.shape[0]
    ix = idx_2d.to(torch.int32)
    g_sub = gi_2d[:, ::16].to(torch.uint8).contiguous()
    u_all = (ix % U).to(torch.uint8).contiguous()
    lb_2d = ix // U
    sh8 = torch.arange(7, -1, -1, device=idx_2d.device, dtype=torch.int32)
    c_pair = (((lb_2d & 1).view(M, 16, 2, 8) << sh8).sum(dim=-1)).to(torch.uint8)
    c_all = c_pair.flip(-1).contiguous()
    lb_out = lb_2d.to(torch.uint8).contiguous() if return_labels else None
    return g_sub, c_all, u_all, lb_out


@torch.no_grad()
def viterbi_vtq_cuda(
    x: torch.Tensor,                      # float32 [M, 256] (`M % 16 == 0`)
    w: torch.Tensor,                      # float32 [M, 256] or [256]
    tab: torch.Tensor,                    # float32 [nl * U]
    gains: torch.Tensor,                  # float32 [G]
    nl: int,
    U: int,
    frac: int = 0,
    g_sub_fix: Optional[torch.Tensor] = None,
    return_labels: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Runs 2-pass tail-biting 1024-state (`nu = 10`) radix-4 Galois Viterbi across `16x16` subblock tiles of `x [M, 256]`."""
    del g_sub_fix
    dev = x.device
    M, L = x.shape
    assert L == 256 and M % SUPER_ROWS == 0, f"Expected [M % 16 == 0, 256], got {x.shape}"
    K = int(U).bit_length()
    pp = _FRAC_TO_TCQ_PP[int(frac)]
    m_extra = _NL_TO_M_EXTRA[int(nl)]
    G_cnt = int(gains.numel())
    n_cand = 4 if G_cnt > 4 else 0
    tb_win = int(os.environ.get("TCQ_TB_WIN", "32"))

    nbg = M // SUPER_ROWS
    x_tiles = x.view(nbg, 16, 16, 16).permute(0, 2, 1, 3).reshape(nbg * 16, 256).contiguous()
    if w.ndim == 1:
        assert w.numel() == 256
        w_tiles = w.view(1, 16, 1, 16).expand(nbg, 16, 16, 16).reshape(nbg * 16, 256).contiguous()
    else:
        assert w.shape == (M, 256)
        w_tiles = w.view(nbg, 16, 16, 16).permute(0, 2, 1, 3).reshape(nbg * 16, 256).contiguous()

    max_nt = 2048 if pp > 0 else 16384
    nt_total = x_tiles.shape[0]
    if nt_total <= max_nt:
        rec_t, gi_t, idx_t = viterbi_ub_cuda(
            x_tiles, tab, K, 10, 359, gains, sub=16, wts=w_tiles,
            n_cand=n_cand, return_code=True, tail_biting=True, tb_window=tb_win,
            m_extra=m_extra, pair_parity=pp,
        )
    else:
        rec_t = torch.empty_like(x_tiles)
        gi_t = torch.empty((nt_total, 256), dtype=torch.int32, device=dev)
        idx_t = torch.empty((nt_total, 256), dtype=torch.int32, device=dev)
        for t0 in range(0, nt_total, max_nt):
            t1 = min(nt_total, t0 + max_nt)
            r_c, g_c, i_c = viterbi_ub_cuda(
                x_tiles[t0:t1], tab, K, 10, 359, gains, sub=16, wts=w_tiles[t0:t1],
                n_cand=n_cand, return_code=True, tail_biting=True, tb_window=tb_win,
                m_extra=m_extra, pair_parity=pp,
            )
            rec_t[t0:t1] = r_c
            gi_t[t0:t1] = g_c
            idx_t[t0:t1] = i_c

    rec = rec_t.view(nbg, 16, 16, 16).permute(0, 2, 1, 3).reshape(M, 256).contiguous()
    gi_2d = gi_t.view(nbg, 16, 16, 16).permute(0, 2, 1, 3).reshape(M, 256).to(torch.uint8)
    idx_2d = idx_t.view(nbg, 16, 16, 16).permute(0, 2, 1, 3).reshape(M, 256)
    g_out, c_out, u_out, lb_out = _code_to_vtq_pack(gi_2d, idx_2d, U, return_labels=return_labels)
    return rec, g_out, c_out, u_out, lb_out


@torch.no_grad()
def ldlq_vtq_block256_cuda(
    w_work: torch.Tensor,       # float32 [M, 256] (modified in-place by causal Cholesky feedback)
    d_sb: torch.Tensor,         # float32 [M]
    dw_blk: torch.Tensor,       # float32 [256]
    U_blk: torch.Tensor,        # float32 [256, 256]
    U_blk_inv: torch.Tensor,    # float32 [256, 256]
    tab: torch.Tensor,          # float32 [nl * U]
    gains: torch.Tensor,        # float32 [G]
    nl: int,
    U: int,
    frac: int = 0,
    return_code: bool = False,
):
    """Fused 16-stage `Block16-LDLQ` over `16x16` intra-subblock 1024-state tail-biting trellises (`tcq_ldlq_block256`)."""
    dev = w_work.device
    M, cols = w_work.shape
    assert cols == 256 and M % SUPER_ROWS == 0, f"Expected [M % 16 == 0, 256], got {w_work.shape}"
    K = int(U).bit_length()
    pp = _FRAC_TO_TCQ_PP[int(frac)]
    m_extra = _NL_TO_M_EXTRA[int(nl)]
    G_cnt = int(gains.numel())
    n_cand = 4 if G_cnt > 4 else 0
    tb_win = int(os.environ.get("TCQ_TB_WIN", "32"))

    max_r = 8192 if (pp > 0 and M > 8192) else M
    if max_r == M:
        if return_code:
            G_nat, gi_2d, idx_2d = ldlq_block256_cuda(
                w_work, d_sb, dw_blk, U_blk, U_blk_inv, tab, K, 10, 359, gains,
                n_cand=n_cand, tb_window=tb_win, m_extra=m_extra, pair_parity=pp, return_code=True,
            )
            g_sub, c_all, u_all, _ = _code_to_vtq_pack(gi_2d, idx_2d, U, return_labels=False)
            return G_nat, g_sub, c_all, u_all
        G_nat = ldlq_block256_cuda(
            w_work, d_sb, dw_blk, U_blk, U_blk_inv, tab, K, 10, 359, gains,
            n_cand=n_cand, tb_window=tb_win, m_extra=m_extra, pair_parity=pp, return_code=False,
        )
        return G_nat, None

    G_nat = torch.empty((M, 256), dtype=torch.float32, device=dev)
    g_sub = torch.empty((M, 16), dtype=torch.uint8, device=dev) if return_code else None
    c_all = torch.empty((M, 16, 2), dtype=torch.uint8, device=dev) if return_code else None
    u_all = torch.empty((M, 256), dtype=torch.uint8, device=dev) if return_code else None
    for r0 in range(0, M, max_r):
        r1 = min(M, r0 + max_r)
        if return_code:
            gp, gi_s, ix_s = ldlq_block256_cuda(
                w_work[r0:r1].clone(), d_sb[r0:r1], dw_blk, U_blk, U_blk_inv, tab, K, 10, 359, gains,
                n_cand=n_cand, tb_window=tb_win, m_extra=m_extra, pair_parity=pp, return_code=True,
            )
            G_nat[r0:r1] = gp
            gs_s, cs_s, us_s, _ = _code_to_vtq_pack(gi_s, ix_s, U, return_labels=False)
            g_sub[r0:r1] = gs_s
            c_all[r0:r1] = cs_s
            u_all[r0:r1] = us_s
        else:
            G_nat[r0:r1] = ldlq_block256_cuda(
                w_work[r0:r1].clone(), d_sb[r0:r1], dw_blk, U_blk, U_blk_inv, tab, K, 10, 359, gains,
                n_cand=n_cand, tb_window=tb_win, m_extra=m_extra, pair_parity=pp, return_code=False,
            )
    if return_code:
        return G_nat, g_sub, c_all, u_all
    return G_nat, None


# =================================================================================================
# 2. Format Registry (`liqk_v2.LIQK_V2_SPECS`, `SPEC_DISTORTION_RATIO`, `equal_bpw.SETS["VTQ"]`)
# =================================================================================================

def load_and_register_vtq(tables_path: Path | None = None) -> Dict[str, dict]:
    """Loads `results/vtq_tables.json` and registers all VTQ specs into `liqk_v2` and `equal_bpw.SETS`."""
    global VTQ_MENU
    if tables_path is None:
        tables_path = DEFAULT_TABLES_PATH
    tables_path = Path(tables_path)
    if not tables_path.exists():
        return VTQ_CONFIG

    raw = json.loads(tables_path.read_text(encoding="utf-8"))
    VTQ_CONFIG.clear()
    _GPU_TAB_CACHE.clear()

    frac_sub = {0: 0.0, 1: 0.5, 2: 0.25, 3: 0.75}
    for name, c in raw.items():
        fc = int(c["fmt_code"])
        K, frac, nl, h, st_bytes = FMT_META[fc]
        hw = dict(c["hw_codebook"])
        g_raw = c.get("gain_u8", c.get("gain_i"))
        gain_u8 = np.clip(np.asarray(g_raw, dtype=np.int32), 1, 255).astype(np.uint8)
        lut = reconstruct_lut_from_hw_vtq(fc, hw)
        gains_np = gain_u8.astype(np.float32) * GAIN_UNIT

        cfg = dict(c)
        cfg["K"] = K
        cfg["frac"] = frac
        cfg["nl"] = nl
        cfg["h"] = h
        cfg["st_bytes"] = st_bytes
        cfg["gain_u8"] = [int(x) for x in gain_u8.tolist()]
        cfg["lut_np"] = lut
        cfg["gains_np"] = gains_np
        VTQ_CONFIG[name] = cfg

        assert st_bytes % SUPER_ROWS == 0
        bits_256 = (st_bytes * 8.0 / SUPER_ROWS) + 2.0
        lv2.LIQK_V2_SPECS[name] = lv2.LiqkV2Spec(
            name=name,
            bits=float(K - frac_sub[frac]),
            n_levels=nl * (1 << (K - 1)),
            n_pos_states=1 << h,
            block_bytes=bits_256 / 8.0,
        )
        if "distortion_ratio" in c:
            lv2.SPEC_DISTORTION_RATIO[name] = float(c["distortion_ratio"])

    VTQ_MENU = [nm for nm in VTQ_CONFIG if nm not in ("VTQ_300_FAST", "VTQ_500_FAST")]
    try:
        import src.equal_bpw as eqb

        eqb.SETS["VTQ"] = list(VTQ_MENU)
        eqb.SETS["VTQ_FAST"] = [nm for nm in VTQ_CONFIG if nm not in ("VTQ_300", "VTQ_500")]
    except Exception:
        pass
    return VTQ_CONFIG


def vtq_candidates_for_target(target_bpw: float, menu: Optional[List[str]] = None) -> List[str]:
    """Filters `VTQ_MENU` to `[target_bpw - 0.45, 6.30]` so linear DP never starves sensitive slices."""
    if not VTQ_CONFIG:
        load_and_register_vtq()
    base_menu = menu if menu is not None else VTQ_MENU
    lo = float(target_bpw) - 0.45
    cands = [sp for sp in base_menu if lv2.LIQK_V2_SPECS[sp].bpw >= lo]
    below = [sp for sp in cands if lv2.LIQK_V2_SPECS[sp].bpw <= float(target_bpw) - 0.02]
    if not below:
        all_below = sorted(
            [sp for sp in base_menu if lv2.LIQK_V2_SPECS[sp].bpw <= float(target_bpw)],
            key=lambda s: lv2.LIQK_V2_SPECS[s].bpw,
        )
        if all_below and all_below[-1] not in cands:
            cands = [all_below[-1]] + cands
    return cands if cands else list(base_menu)


def _get_spec_gpu_vtq(spec_name: str, dev: torch.device) -> Tuple[dict, torch.Tensor, torch.Tensor]:
    if not VTQ_CONFIG:
        load_and_register_vtq()
    cfg = VTQ_CONFIG[spec_name]
    key = (spec_name, dev)
    if key not in _GPU_TAB_CACHE:
        _GPU_TAB_CACHE[key] = (
            torch.tensor(cfg["lut_np"], dtype=torch.float32, device=dev),
            torch.tensor(cfg["gains_np"], dtype=torch.float32, device=dev),
        )
    tab, gains = _GPU_TAB_CACHE[key]
    return cfg, tab, gains


# =================================================================================================
# 3. Core VTQ Matrix Quantizer (`Outer-DCT` + Natural `Block16-LDLQ` + `fp16 d_row * uint8 u_sb`)
# =================================================================================================

@torch.no_grad()
def _rotate_owned_hessian(buf, rot_block, stripe):
    """For private, contiguous Hessian storage only: reuse rotated row stripes."""
    if stripe <= 0 or buf.shape[0] <= stripe:
        return apply_vtq_rot(buf, rot_block=rot_block, transpose=False)
    for rows in buf.split(stripe, dim=0):
        rows.copy_(apply_vtq_rot(rows, rot_block=rot_block, transpose=False))
    return buf


@torch.no_grad()
def compute_suh_vtq(
    W_f32: torch.Tensor,
    H_f32: torch.Tensor,
    seed: int = 1337,
    row_weights: Optional[torch.Tensor] = None,
    rot_block: int = -1,
    h_stripe: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Computes `suh[K]` `(p_w = 0.50, p_h = -0.25)` and applies `Outer-DCT` (`rot_block=-1`) or `FWHT-256` (`256`)."""
    dev = W_f32.device
    Kin = W_f32.shape[1]
    signs = make_rademacher_signs(Kin, seed=seed, device=dev)

    if row_weights is not None:
        rw_s = row_weights.float().clamp(min=float(row_weights.mean().item()) * 0.1)
        rw_s = rw_s / rw_s.mean().clamp(min=1e-12)
        col_rms = (W_f32.square() * rw_s.unsqueeze(1)).mean(dim=0).sqrt().clamp(min=1e-6)
    else:
        col_rms = W_f32.square().mean(dim=0).sqrt().clamp(min=1e-6)
    col_rms = col_rms / col_rms.mean().clamp(min=1e-6)
    h_std = torch.diagonal(H_f32).clamp(min=1e-8).sqrt()
    h_std = h_std / h_std.mean().clamp(min=1e-6)

    scale_suh = ((col_rms ** 0.50) * (h_std ** -0.25)).clamp(0.25, 4.0)
    suh = (signs * scale_suh).to(torch.float16).to(torch.float32)

    W_rot = apply_vtq_rot(W_f32 / suh.unsqueeze(0), rot_block=rot_block, transpose=False)
    H_tmp = _rotate_owned_hessian(H_f32 * suh.unsqueeze(0), rot_block, h_stripe)
    # H_tmp owns the first rotation. Release its K x K storage before the
    # second rotation rather than carrying it across both factor matmuls.
    H_left = H_tmp.T.mul_(suh.unsqueeze(0)).contiguous()
    del H_tmp
    H_rot = _rotate_owned_hessian(H_left, rot_block, h_stripe)
    del H_left
    H_rot = (H_rot + H_rot.T).mul_(0.5).contiguous()
    return W_rot, H_rot, suh


def _initial_tile_scales(w, cfg_scale, spec_name):
    return w.reshape(w.shape[0], -1, TILE).square().mean(dim=2).sqrt().clamp(min=1e-10) * cfg_scale


def _initial_block_u8(w, cfg_scale, step_row, spec_name):
    rms_cur = w.square().mean(dim=1).sqrt().clamp(min=1e-10) * cfg_scale
    u_init_f = torch.round(rms_cur / step_row).clamp(1.0, 255.0)
    return u_init_f, (step_row * u_init_f).clamp(min=1e-10)


def _refit_block_u8(A_c, B_c, step_row, u_init_f, d_init):
    """Monotone row-wise scale fit for a fixed 256-column code block."""
    den = (B_c * B_c).sum(dim=1)
    d_cont = torch.where(den > 1e-12, (A_c * B_c).sum(dim=1) / den.clamp(min=1e-12), d_init).clamp(min=1e-10)
    u_opt_f = torch.round(d_cont / step_row).clamp(1.0, 255.0)
    d_opt_q = step_row * u_opt_f
    l_init = (A_c - d_init.unsqueeze(1) * B_c).square().sum(dim=1)
    l_opt = (A_c - d_opt_q.unsqueeze(1) * B_c).square().sum(dim=1)
    u_fin_f = torch.where(l_opt < l_init, u_opt_f, u_init_f)
    return u_fin_f, step_row * u_fin_f


@torch.no_grad()
def _quantize_rows_vtq_ldlq(
    W_sub: torch.Tensor,                      # float32 [R, Kin] in natural column order (`R % 16 == 0`)
    U_nat: torch.Tensor,                      # float32 [Kin, Kin]
    U_blks: torch.Tensor,                     # float32 [nblocks, 256, 256]
    U_blks_inv: torch.Tensor,                 # float32 [nblocks, 256, 256]
    dw_diag: torch.Tensor,                    # float32 [nblocks, 256]
    gs_ctx: Optional[dict],                   # low-damp two-sided Hessian context (`H_gs`, `H_bb`, `U_v`, `Ub_gs`, `H_bb_inv`)
    spec_name: str,
    emit_pack: bool = False,
    u_sb_gecd_sweeps: int = 1,
) -> dict:
    """Quantizes `W_sub [R, Kin]` via:
      1. Forward `16x16` intra-subblock `Block16-LDLQ` (`tcq_ldlq_block256`) + monotone Cholesky `u_sb` refinement,
      2. Optional parallel `z_gs` residual Viterbi Gauss-Seidel polish (`u_sb_gecd_sweeps >= 1`).
    """
    dev = W_sub.device
    R, Kin = W_sub.shape
    assert R % SUPER_ROWS == 0, f"VTQ requires row slice multiple of 16, got R={R}"
    nblocks = Kin // TILE
    cfg, tab, gains = _get_spec_gpu_vtq(spec_name, dev)
    K = int(cfg["K"])
    U = 1 << (K - 1)
    nl = int(cfg["nl"])
    frac = int(cfg["frac"])
    cfg_scale = float(cfg.get("scale", 1.0))

    rms_rb = _initial_tile_scales(W_sub, cfg_scale, spec_name)
    d_row_fp16 = (rms_rb.max(dim=1).values * (128.0 / 224.0)).clamp(min=1e-7).to(torch.float16)
    d_row_f32 = d_row_fp16.float().clamp(min=1e-8)
    step_row = (d_row_f32 * float(U_SB_UNIT)).clamp(min=1e-12)

    W_work = W_sub.clone()
    G_nat_all = torch.empty((R, Kin), dtype=torch.float32, device=dev)
    u_sb_all = torch.empty((R, nblocks), dtype=torch.uint8, device=dev)
    g_sub_all = torch.empty((R, nblocks, 16), dtype=torch.uint8, device=dev) if emit_pack else None
    c_all = torch.empty((R, nblocks, 16, 2), dtype=torch.uint8, device=dev) if emit_pack else None
    u_all = torch.empty((R, nblocks, 256), dtype=torch.uint8, device=dev) if emit_pack else None

    # Pass 1: Causal `16x16` intra-subblock Block16-LDLQ + monotone Cholesky-weighted `u_sb` refinement
    for ibl in range(nblocks):
        c0, c1 = ibl * TILE, (ibl + 1) * TILE
        w_blk_start = W_work[:, c0:c1].clone()
        U_blk = U_blks[ibl]
        U_blk_inv = U_blks_inv[ibl]
        dw_blk = dw_diag[ibl]

        u_init_f, d_init = _initial_block_u8(w_blk_start, cfg_scale, step_row, spec_name)

        if emit_pack:
            G_blk, g_b, c_b, u_b = ldlq_vtq_block256_cuda(
                w_blk_start.clone(), d_init, dw_blk, U_blk, U_blk_inv, tab, gains, nl, U, frac=frac, return_code=True
            )
        else:
            G_blk, _ = ldlq_vtq_block256_cuda(
                w_blk_start.clone(), d_init, dw_blk, U_blk, U_blk_inv, tab, gains, nl, U, frac=frac, return_code=False
            )
            g_b = c_b = u_b = None

        A_c = w_blk_start @ U_blk_inv
        B_c = G_blk @ U_blk_inv
        u_fin_f, d_fin = _refit_block_u8(A_c, B_c, step_row, u_init_f, d_init)

        G_nat_all[:, c0:c1] = G_blk
        u_sb_all[:, ibl] = u_fin_f.to(torch.uint8)
        if emit_pack:
            g_sub_all[:, ibl] = g_b
            c_all[:, ibl] = c_b
            u_all[:, ibl] = u_b

        if c1 < Kin:
            err_blk = (w_blk_start - d_fin.unsqueeze(1) * G_blk) @ U_blk_inv
            W_work[:, c1:] -= err_blk @ U_nat[c0:c1, c1:]

    # Pass 2: Monotone two-sided Full-Hessian Gauss-Seidel polish (`u_sb_gecd_sweeps == 1` -> fast parallel `z_gs` Viterbi!)
    if u_sb_gecd_sweeps > 0 and gs_ctx is not None and nblocks > 1:
        H_gs = gs_ctx["H_gs"]
        H_bb_all = gs_ctx["H_bb_all"]
        U_v_all = gs_ctx["U_v_all"]
        Ub_gs_all = gs_ctx["Ub_gs_all"]
        H_bb_inv_all = gs_ctx["H_bb_inv_all"]

        d_eff_rb = step_row.unsqueeze(1) * u_sb_all.float()
        W_hat_nat = (G_nat_all.view(R, nblocks, TILE) * d_eff_rb.unsqueeze(-1)).reshape(R, Kin)
        RH = (W_sub - W_hat_nat) @ H_gs

        run_ldlq_gs = int(u_sb_gecd_sweeps) >= 2
        n_gs_passes = 1 if int(u_sb_gecd_sweeps) == 1 else min(int(u_sb_gecd_sweeps) - 1, 2)
        for sw in range(n_gs_passes):
            blk_iter = range(nblocks - 1, -1, -1) if (sw % 2 == 0) else range(nblocks)
            for ibl in blk_iter:
                c0, c1 = ibl * TILE, (ibl + 1) * TILE
                H_bb = H_bb_all[ibl]
                U_v = U_v_all[ibl]
                Ub_b = Ub_gs_all[ibl]
                diag_uv = torch.diagonal(U_v)
                dwb_gs = (diag_uv * diag_uv).contiguous()

                w_cur_b = W_hat_nat[:, c0:c1]
                w_gs = (w_cur_b + RH[:, c0:c1] @ H_bb_inv_all[ibl]).contiguous()
                d_old = (step_row * u_sb_all[:, ibl].float()).clamp(min=1e-10)
                Ac = w_gs @ U_v

                cands = []
                if run_ldlq_gs and sw == 0:
                    if emit_pack:
                        Gb_ld, gb_ld, cb_ld, ub_ld = ldlq_vtq_block256_cuda(
                            w_gs.clone(), d_old, dwb_gs, Ub_b, U_v, tab, gains, nl, U, frac=frac, return_code=True
                        )
                    else:
                        Gb_ld, _ = ldlq_vtq_block256_cuda(
                            w_gs.clone(), d_old, dwb_gs, Ub_b, U_v, tab, gains, nl, U, frac=frac, return_code=False
                        )
                        gb_ld = cb_ld = ub_ld = None
                    cands.append((Gb_ld, gb_ld, cb_ld, ub_ld))

                # Parallel 1-launch residual Viterbi across all 16 subblocks of `[R, 256]`:
                Gb_cur = G_nat_all[:, c0:c1]
                d_cur = (step_row * u_sb_all[:, ibl].float()).clamp(min=1e-10)
                Bc_cur = Gb_cur @ U_v
                z_gs = d_cur.unsqueeze(1) * Gb_cur + (Ac - d_cur.unsqueeze(1) * Bc_cur) / diag_uv.unsqueeze(0)
                Gb_vt, gb_vt, cb_vt, ub_vt, _ = viterbi_vtq_cuda(
                    z_gs / d_cur.unsqueeze(1), dwb_gs, tab, gains, nl, U, frac=frac, return_labels=False
                )
                cands.append((Gb_vt, gb_vt, cb_vt, ub_vt))

                for Gb_new, gb_new, cb_new, ub_new in cands:
                    Bc_new = Gb_new @ U_v
                    den_n = (Bc_new * Bc_new).sum(dim=1)
                    d_opt = torch.where(den_n > 1e-12, (Ac * Bc_new).sum(dim=1) / den_n.clamp(min=1e-12), d_cur).clamp(min=0.0)
                    u_new_f = torch.round(d_opt / step_row).clamp(1.0, 255.0)
                    d_new = step_row * u_new_f
                    w_cur_b = W_hat_nat[:, c0:c1]
                    dq = d_new.unsqueeze(1) * Gb_new - w_cur_b
                    dloss = -2.0 * (dq * RH[:, c0:c1]).sum(dim=1) + ((dq @ H_bb) * dq).sum(dim=1)
                    take = dloss < -1e-12
                    if bool(take.any()):
                        dq_eff = torch.where(take.unsqueeze(1), dq, torch.zeros_like(dq))
                        G_nat_all[:, c0:c1] = torch.where(take.unsqueeze(1), Gb_new, G_nat_all[:, c0:c1])
                        u_sb_all[:, ibl] = torch.where(take, u_new_f.to(torch.uint8), u_sb_all[:, ibl])
                        if emit_pack:
                            g_sub_all[:, ibl] = torch.where(take.unsqueeze(1), gb_new, g_sub_all[:, ibl])
                            c_all[:, ibl] = torch.where(take.view(-1, 1, 1), cb_new, c_all[:, ibl])
                            u_all[:, ibl] = torch.where(take.unsqueeze(1), ub_new, u_all[:, ibl])
                        W_hat_nat[:, c0:c1] += dq_eff
                        RH.sub_(dq_eff @ H_gs[c0:c1, :])
                        d_cur = (step_row * u_sb_all[:, ibl].float()).clamp(min=1e-10)

    return {
        "G_nat": G_nat_all,
        "d_row": d_row_fp16,
        "u_sb": u_sb_all,
        "g_sub": g_sub_all,
        "c_all": c_all,
        "u_all": u_all,
    }


def resolve_rot_block(subname: Optional[str], rot_block: int = -1, rot_mode: str = "outer") -> int:
    """When `rot_mode == 'selective'`, uses 1-stage fused Block-256 FWHT (`256`) on `q/k/v/gate/up` and
    full `Outer-DCT` (`-1`) only on `o_proj` / `down_proj`."""
    if rot_mode == "selective" and subname is not None:
        if any(tag in subname for tag in ("o_proj", "down_proj", "out_proj")):
            return -1
        return 256
    return rot_block


@torch.no_grad()
def quantize_matrix_vtq(
    W: torch.Tensor,
    H: torch.Tensor,
    spec_or_row_specs: str | List[Tuple[int, int, str]],
    seed: int = 1337,
    damp: float = 0.20,
    row_weights: Optional[torch.Tensor] = None,
    rot_block: int = -1,
    W_target: Optional[torch.Tensor] = None,
    u_sb_gecd_sweeps: Optional[int] = None,
    d_sb_gecd_sweeps: Optional[int] = None,
    compact_wide_workspace: bool = False,
    row_energy_tilt: float = 0.,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, float]:
    """Quantizes `W [M, Kin]` (`Kin % 256 == 0`) into VTQ with bit-exact `assemble_vtq` output."""
    if not VTQ_CONFIG:
        load_and_register_vtq()
    if d_sb_gecd_sweeps is not None:
        u_sb_gecd_sweeps = int(d_sb_gecd_sweeps)
    elif u_sb_gecd_sweeps is None:
        u_sb_gecd_sweeps = int(os.environ.get("VTQ_GS_SWEEPS", "0"))

    dev = W.device
    W_src = (W_target if W_target is not None else W).float()
    H_f32 = H.float()
    M, Kin = W_src.shape
    assert Kin % TILE == 0, f"VTQ requires Kin % 256 == 0, got Kin={Kin}"
    nblocks = Kin // TILE

    compact_wide = compact_wide_workspace and Kin >= 8192
    rot_kw = {'h_stripe': 1024} if compact_wide else {}
    W_rot, H_rot, suh = compute_suh_vtq(W_src, H_f32, seed=seed, row_weights=row_weights, rot_block=rot_block, **rot_kw)
    _mem_mark("suh_rot")
    dmean = H_rot.diagonal().mean().clamp(min=1e-8)
    # `H_rot` is only needed afterwards by the two-sided Gauss-Seidel polish (`u_sb_gecd_sweeps > 0`); otherwise
    # regularize it in place and free it right away (saves one `K x K` buffer at the Cholesky peak).
    keep_h_rot = u_sb_gecd_sweeps > 0 and nblocks > 1
    H_reg = H_rot.clone() if keep_h_rot else H_rot
    H_reg.diagonal().add_(damp * dmean)
    if not keep_h_rot:
        del H_rot
    H_flip = H_reg.flip(0, 1)
    del H_reg
    L_flip = torch.linalg.cholesky(H_flip)
    del H_flip
    _mem_mark("cholesky")
    if compact_wide:
        # A column-major RHS can be overwritten directly by the triangular solve.
        rhs = torch.eye(Kin, dtype=L_flip.dtype, device=dev).T
        U_inv_flip = torch.linalg.solve_triangular(L_flip, rhs, upper=False, out=rhs)
        del rhs
    else:
        U_inv_flip = torch.linalg.solve_triangular(L_flip, torch.eye(Kin, dtype=L_flip.dtype, device=dev), upper=False)
    del L_flip
    U_nat = U_inv_flip.flip(0, 1)
    del U_inv_flip
    _mem_mark("u_nat")

    eye256 = torch.eye(TILE, dtype=torch.float32, device=dev).unsqueeze(0).expand(nblocks, -1, -1)
    U_blks = torch.stack([U_nat[b * TILE : (b + 1) * TILE, b * TILE : (b + 1) * TILE] for b in range(nblocks)], dim=0)
    U_blks_inv = torch.linalg.solve_triangular(U_blks, eye256, upper=True)
    dw_diag = (1.0 / (torch.diagonal(U_nat) ** 2).clamp(min=1e-12)).view(nblocks, TILE).contiguous()

    gs_ctx = None
    if u_sb_gecd_sweeps > 0 and nblocks > 1:
        gs_mult = float(globals().get("_GS_MULT_OVERRIDE", 1.0))
        H_gs = H_rot
        H_gs.diagonal().add_(max(gs_mult * damp, 0.02) * dmean)
        H_bb_all = torch.stack([H_gs[b * TILE : (b + 1) * TILE, b * TILE : (b + 1) * TILE] for b in range(nblocks)], dim=0)
        U_v_all = torch.linalg.cholesky(H_bb_all.flip(1, 2)).flip(1, 2)
        Ub_gs_all = torch.linalg.solve_triangular(U_v_all, eye256, upper=True)
        H_bb_inv_all = torch.cholesky_inverse(torch.linalg.cholesky(H_bb_all))
        gs_ctx = {
            "H_gs": H_gs,
            "H_bb_all": H_bb_all,
            "U_v_all": U_v_all,
            "Ub_gs_all": Ub_gs_all,
            "H_bb_inv_all": H_bb_inv_all,
        }

    row_slices = [(0, M, spec_or_row_specs)] if isinstance(spec_or_row_specs, str) else list(spec_or_row_specs)
    by_spec: Dict[str, List[Tuple[int, int]]] = {}
    for r0, r1, sp in row_slices:
        by_spec.setdefault(sp, []).append((r0, r1))

    emit_pack = (PACK is not None) or (lv2.PACK is not None)
    # Skip gather/scatter (and an entire second set of outputs) for a single full slice.
    full_slice = len(row_slices) == 1 and row_slices[0][:2] == (0, M)
    pm = None
    if not full_slice:
        G_nat_all = torch.empty_like(W_rot)
        d_row_all = torch.empty((M,), dtype=torch.float16, device=dev)
        u_sb_all = torch.empty((M, nblocks), dtype=torch.uint8, device=dev)
        if emit_pack:
            pm = {k: torch.empty((M, nblocks, *shape), dtype=torch.uint8, device=dev)
                  for k, shape in (("g_sub", (16,)), ("c_all", (16, 2)), ("u_all", (256,)))}

    for sp, ranges in by_spec.items():
        rows_idx = slice(None) if full_slice else torch.cat([torch.arange(r0, r1, device=dev) for r0, r1 in ranges])
        sub_out = _quantize_rows_vtq_ldlq(
            W_rot[rows_idx],
            U_nat,
            U_blks,
            U_blks_inv,
            dw_diag,
            gs_ctx,
            sp,
            emit_pack=emit_pack,
            u_sb_gecd_sweeps=u_sb_gecd_sweeps,
        )
        if full_slice:
            G_nat_all, d_row_all, u_sb_all = sub_out["G_nat"], sub_out["d_row"], sub_out["u_sb"]
            if emit_pack:
                pm = {k: sub_out[k] for k in ("g_sub", "c_all", "u_all")}
            del sub_out
            break
        G_nat_all[rows_idx] = sub_out["G_nat"]
        d_row_all[rows_idx] = sub_out["d_row"]
        u_sb_all[rows_idx] = sub_out["u_sb"]
        if pm is not None:
            pm["g_sub"][rows_idx] = sub_out["g_sub"]
            pm["c_all"][rows_idx] = sub_out["c_all"]
            pm["u_all"][rows_idx] = sub_out["u_all"]
        del sub_out

    del U_nat, U_blks, U_blks_inv, gs_ctx, W_rot
    _mem_mark("ldlq")

    # Exact bit-accurate reconstruction + `VTQ_REFIT_SWEEPS` alternating `(c_row -> d_row, suh)` global refit sweeps:
    suh_fp16 = suh.to(torch.float16)
    d_eff_base = d_row_all.float().unsqueeze(1) * (u_sb_all.float() * float(U_SB_UNIT))
    W_rot_hat = (G_nat_all.view(M, nblocks, TILE) * d_eff_base.unsqueeze(-1)).reshape(M, Kin)
    U_base = apply_vtq_rot(W_rot_hat, rot_block=rot_block, transpose=True)
    W_hat_eff = U_base * suh_fp16.float().unsqueeze(0)
    del U_base
    if not emit_pack:
        del W_rot_hat  # only `PACK["matrix"]["W_hat_rot"]` needs it

    HW = W_src @ H_f32
    HQ = W_hat_eff @ H_f32
    if row_weights is not None:
        rw = row_weights.float().clamp(min=float(row_weights.mean().item()) * 0.1)
        rw = (rw / rw.mean().clamp(min=1e-12)).clamp(0.25, 4.0)
    else:
        rw = None

    # Two sweeps give the same SNR as one (<= 0.001 bit on 12 real Llama-3.2-3B matrices) for ~+20% encoder time on wide `K`.
    for _ in range(int(os.environ.get("VTQ_REFIT_SWEEPS", "1"))):
        c_row = (W_hat_eff * HW).sum(dim=1) / (W_hat_eff * HQ).sum(dim=1).clamp(min=1e-12)
        c_row = torch.where(torch.isfinite(c_row) & (c_row > 0.5) & (c_row < 2.0), c_row, torch.ones_like(c_row))
        d_row_cand = (d_row_all.float() * c_row).to(torch.float16)

        d_eff_cand = d_row_cand.float().unsqueeze(1) * (u_sb_all.float() * float(U_SB_UNIT))
        W_rot_cand = (G_nat_all.view(M, nblocks, TILE) * d_eff_cand.unsqueeze(-1)).reshape(M, Kin)
        U_scaled = apply_vtq_rot(W_rot_cand, rot_block=rot_block, transpose=True)
        if not emit_pack:
            del W_rot_cand
        U_w = (U_scaled * rw.unsqueeze(1)) if rw is not None else U_scaled

        A_in = U_w.T @ U_scaled
        A_in.mul_(H_f32)
        b_in = (U_w * HW).sum(dim=0)
        del U_w
        A_in.diagonal().add_(1e-3 * A_in.diagonal().mean().clamp(min=1e-12))
        try:
            suh_try = torch.linalg.solve(A_in, b_in).to(torch.float16)
            del A_in
            valid_s = torch.isfinite(suh_try.float()) & (suh_try.float().abs() > 0.05) & (suh_try.float().abs() < 10.0)
            suh_try = torch.where(valid_s, suh_try, suh_fp16)
            W_hat_cand = U_scaled.mul_(suh_try.float().unsqueeze(0))
            HQ_cand = W_hat_cand @ H_f32
            if rw is not None:
                l_old = float((((HW - HQ) * (W_src - W_hat_eff)).sum(dim=1) * rw).sum())
                l_new = float((((HW - HQ_cand) * (W_src - W_hat_cand)).sum(dim=1) * rw).sum())
            else:
                l_old = float(((HW - HQ) * (W_src - W_hat_eff)).sum())
                l_new = float(((HW - HQ_cand) * (W_src - W_hat_cand)).sum())
            if l_new < l_old:
                d_row_all = d_row_cand
                suh_fp16 = suh_try
                if emit_pack:
                    W_rot_hat = W_rot_cand
                W_hat_eff = W_hat_cand
                HQ = HQ_cand
        except RuntimeError:
            break
    _mem_mark("refit")

    if row_energy_tilt:
        # exp29: least-squares refits may shrink per-row output RMS. Move the
        # existing row scale toward matching E[(Wx)^2] on calibration inputs.
        # Reconstruct from the rounded fp16 scale, not by scaling W_hat_eff.
        if not -1. <= row_energy_tilt <= 1.: raise ValueError('invalid row-energy tilt')
        orig_e = (W_src * HW).sum(1).clamp_min(1e-20)
        recon_e = (W_hat_eff * HQ).sum(1).clamp_min(1e-20)
        ratio = (orig_e / recon_e).sqrt().clamp(.8, 1.25).pow(row_energy_tilt)
        new_d = (d_row_all.float() * ratio).half()
        d_row_all = torch.where(torch.isfinite(new_d) & (new_d > 0), new_d, d_row_all)
        ds = d_row_all.float().unsqueeze(1) * (u_sb_all.float() * float(U_SB_UNIT))
        W_rot_hat = (G_nat_all.view(M, nblocks, TILE) * ds.unsqueeze(-1)).reshape(M, Kin)
        W_hat_eff = apply_vtq_rot(W_rot_hat, rot_block=rot_block, transpose=True) * suh_fp16.float().unsqueeze(0)
        HQ = W_hat_eff @ H_f32

    if emit_pack and pm is not None:
        # W_hat_eff and HQ already correspond to the last accepted fp16 scales;
        # rerunning the inverse rotation and the dense Hessian product is redundant.
        pack_dict = {
            "row_slices": row_slices,
            "c_all": pm["c_all"],
            "u_all": pm["u_all"],
            "g_sub": pm["g_sub"],
            "u_sb": u_sb_all,
            "d_row": d_row_all,
            "suh": suh_fp16,
            "rot_block": rot_block,
            "W_hat_rot": W_rot_hat,
        }
        if PACK is not None:
            PACK["matrix"] = pack_dict
        if lv2.PACK is not None:
            lv2.PACK["matrix"] = pack_dict

    diff = W_src - W_hat_eff
    rel_loss = float(((HW - HQ) * diff).sum().item()) / float((HW * W_src).sum().clamp(min=1e-12).item())
    qdata = {"d_row": d_row_all, "u_sb": u_sb_all, "suh": suh_fp16, "row_slices": row_slices, "rot_block": rot_block}
    return qdata, W_hat_eff, rel_loss


# =================================================================================================
# 4. Hardware-Exact Codebook & Gain Trainer + Real-Layer `SPEC_DISTORTION_RATIO` Calibrator
# =================================================================================================

def _normal_quantiles(n: int) -> np.ndarray:
    """Symmetric `N(0, 1)` mid-probability quantiles without external `scipy` dependency."""
    p = (np.arange(n, dtype=np.float64) + 0.5) / float(n)
    t = torch.tensor(2.0 * p - 1.0, dtype=torch.float64)
    q = (math.sqrt(2.0) * torch.erfinv(t)).numpy()
    q = 0.5 * (q - q[::-1])
    return q.astype(np.float32)


def _fit_additive_u8(pos_mat: np.ndarray, K_base: int, nl: int, tp: str, qmax: int = 127) -> Tuple[list[int], list[int]]:
    """Fits integer `TA_u8, TB_u8 >= 0` with `TA[idxA] + TB[idxB] <= qmax` (zero clipping!) via LSQ + coordinate polish."""
    idxA, idxB = _at2_indices_vtq(K_base, nl, tp)
    nA = 16 if tp == "at2_16_16" else 32
    nB = nA
    u_cnt = pos_mat.shape[0]
    A = np.zeros((u_cnt * nl, nA + nB), dtype=np.float64)
    b = pos_mat.reshape(-1).astype(np.float64)
    for r in range(u_cnt):
        for c in range(nl):
            k = r * nl + c
            A[k, idxA[r, c]] = 1.0
            A[k, nA + idxB[r, c]] = 1.0
    sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    ta = sol[:nA].copy()
    tb = sol[nA:].copy()
    sh_a = min(float(np.min(ta)), 0.0)
    ta -= sh_a
    tb += sh_a
    if float(np.min(tb)) < 0.0:
        sh_b = -float(np.min(tb))
        tb += sh_b
        ta = np.maximum(ta - sh_b, 0.0)
    ta_i = np.clip(np.rint(ta), 0, qmax).astype(np.int32)
    tb_i = np.clip(np.rint(tb), 0, qmax).astype(np.int32)
    while int(np.max(ta_i[idxA] + tb_i[idxB])) > qmax:
        over = np.argwhere((ta_i[idxA] + tb_i[idxB]) > qmax)
        r, c = int(over[0, 0]), int(over[0, 1])
        if ta_i[idxA[r, c]] >= tb_i[idxB[r, c]]:
            ta_i[idxA[r, c]] -= 1
        else:
            tb_i[idxB[r, c]] -= 1
    for _ in range(6):
        for i in range(nA):
            best_e = float(np.mean((ta_i[idxA] + tb_i[idxB] - pos_mat) ** 2))
            for d in (-1, 1):
                if 0 <= ta_i[i] + d <= qmax:
                    ta_i[i] += d
                    if int(np.max(ta_i[idxA] + tb_i[idxB])) <= qmax:
                        e = float(np.mean((ta_i[idxA] + tb_i[idxB] - pos_mat) ** 2))
                        if e < best_e - 1e-12:
                            best_e = e
                            break
                    ta_i[i] -= d
        for i in range(nB):
            best_e = float(np.mean((ta_i[idxA] + tb_i[idxB] - pos_mat) ** 2))
            for d in (-1, 1):
                if 0 <= tb_i[i] + d <= qmax:
                    tb_i[i] += d
                    if int(np.max(ta_i[idxA] + tb_i[idxB])) <= qmax:
                        e = float(np.mean((ta_i[idxA] + tb_i[idxB] - pos_mat) ** 2))
                        if e < best_e - 1e-12:
                            best_e = e
                            break
                    tb_i[i] -= d
    return [int(x) for x in ta_i.tolist()], [int(x) for x in tb_i.tolist()]


def _pos_mags_to_hw(fmt_code: int, pos_vals: np.ndarray, qmax: int = 127) -> Tuple[dict, float]:
    """Stretches `pos_vals` to fill `[0, qmax]` (`max_val -> qmax / 64.0`) and returns `(hw_codebook, stretch)`."""
    K_base, _, nl, _, _ = FMT_META[fmt_code]
    U = 1 << (K_base - 1)
    pos_sorted = np.maximum(pos_vals, 0.0).astype(np.float64)
    vmax = float(np.max(pos_sorted))
    target_max = float(qmax) / 64.0
    stretch = vmax / max(target_max, 1e-8) if vmax > 1e-8 else 1.0
    scaled_f = np.clip(pos_sorted / stretch * 64.0, 0.0, float(qmax))
    if fmt_code in (15, 17):
        pos_s8 = np.rint(scaled_f).astype(np.int8)
        tab_s8 = np.concatenate([-pos_s8[::-1], pos_s8]).astype(np.int8)
        return {"type": "s8_32", "qmax": int(qmax), "tab_div": 64, "tab_s8": [int(x) for x in tab_s8.tolist()]}, float(stretch)
    if K_base == 4 and nl == 16:
        ta_l, tb_l = _fit_additive_u8(scaled_f.reshape(U // 2, nl), K_base, nl, "at2_16_16", qmax=qmax)
        return {"type": "at2_16_16", "qmax": int(qmax), "tab_div": 64, "TA_u8": ta_l, "TB_u8": tb_l}, float(stretch)
    if K_base in (5, 6) and nl == 16:
        ta_l, tb_l = _fit_additive_u8(scaled_f.reshape(U // 2, nl), K_base, nl, "at2z_32_32", qmax=qmax)
        return {"type": "at2z_32_32", "qmax": int(qmax), "tab_div": 64, "TA_u8": ta_l, "TB_u8": tb_l}, float(stretch)
    tab_u8 = np.rint(scaled_f).astype(np.uint8)
    return {"type": "u8", "qmax": int(qmax), "tab_div": 64, "tab_u8": [int(x) for x in tab_u8.tolist()]}, float(stretch)


def _bucket_mapping_vtq(fmt_code: int) -> Tuple[np.ndarray, np.ndarray, int]:
    """Maps `(lb, u)` -> `(param_idx in 0..m_mag-1, sign in {-1.0, +1.0})` matching `reconstruct_lut_from_hw_vtq`."""
    K_base, _, nl, _, _ = FMT_META[fmt_code]
    U = 1 << (K_base - 1)
    p_idx = np.empty((nl, U), dtype=np.int64)
    p_sgn = np.empty((nl, U), dtype=np.float32)
    if fmt_code in (15, 17):
        m_mag = 16
        for lb in range(16):
            p_idx[lb, 0] = 15 - lb
            p_sgn[lb, 0] = -1.0
            p_idx[lb, 1] = lb
            p_sgn[lb, 1] = +1.0
        return p_idx.reshape(-1), p_sgn.reshape(-1), m_mag

    m_mag = nl * (U // 2)
    u_mag_mask = (1 << (K_base - 2)) - 1
    for lb in range(nl):
        for u in range(U):
            neg = (u >> (K_base - 2)) & 1
            u_mag = u & u_mag_mask
            idx = ((u_mag * nl) | lb) ^ ((m_mag - 1) if neg else 0)
            p_idx[lb, u] = idx
            p_sgn[lb, u] = -1.0 if neg else +1.0
    return p_idx.reshape(-1), p_sgn.reshape(-1), m_mag


@torch.no_grad()
def _eval_vtq_cand_on_blocks(
    X_sb: torch.Tensor,         # float32 [N_sb, 256] unit-RMS superblocks
    w_ones: torch.Tensor,       # float32 [256]
    lut_t: torch.Tensor,        # float32 [nl * U]
    gains_t: torch.Tensor,      # float32 [G]
    scale: float,
    nl: int,
    U: int,
    frac: int,
) -> Tuple[float, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluates `(lut, gains, scale)` via exact `L = 256` 2-pass tail-biting joint `(gain, 1024-state)` Viterbi
    and discrete `u_sb` (`1..255`, `* 1/128`, centered at `208` matching `quantize_matrix_vtq`)."""
    d_row_val = float(scale) * (128.0 / 208.0)
    d0 = d_row_val * (208.0 * float(U_SB_UNIT))
    X_norm = (X_sb / max(d0, 1e-10)).contiguous()

    G_sb, g_out, _, u_out, lb_out = viterbi_vtq_cuda(
        X_norm, w_ones, lut_t, gains_t, nl, U, frac=frac, return_labels=True
    )
    num = (X_sb * G_sb).sum(dim=1)
    den = (G_sb * G_sb).sum(dim=1).clamp(min=1e-12)
    d_cont = (num / den).clamp(min=0.0)
    u_opt = torch.round(128.0 * d_cont / max(d_row_val, 1e-10)).clamp(1.0, 255.0)
    d_opt = (d_row_val * (u_opt * float(U_SB_UNIT))).unsqueeze(1)
    mse = float((X_sb - d_opt * G_sb).square().mean().item())
    return mse, d_opt, g_out, lb_out.long() * U + u_out.long(), G_sb


_V2_WARMSTART_MAP: Dict[str, str] = {
    "VTQ_150": "VTQ150",
    "VTQ_175": "VTQ175",
    "VTQ_200": "VTQ200",
    "VTQ_225": "VTQ225",
    "VTQ_250": "VTQ250",
    "VTQ_250_H2": "VTQ250_H2",
    "VTQ_275": "VTQ275",
    "VTQ_275_H2": "VTQ275_H2",
    "VTQ_300": "VTQ300",
    "VTQ_325": "VTQ325",
    "VTQ_350": "VTQ350",
    "VTQ_375": "VTQ375",
    "VTQ_400": "VTQ400",
    "VTQ_425": "VTQ425",
    "VTQ_450": "VTQ450",
    "VTQ_475": "VTQ475",
    "VTQ_500": "VTQ500",
    "VTQ_550": "VTQ550",
    "VTQ_575": "VTQ575",
    "VTQ_600": "VTQ600",
}


@torch.no_grad()
def train_vtq_tables(
    out_path: Path = DEFAULT_TABLES_PATH,
    calibrate_model_dir: Optional[Path] = None,
) -> Dict[str, dict]:
    """Trains all 22 hardware-exact `VTQ` codebooks (`qmax = 127`) & `gain_u8` (`sub = 16`) + calibrates `distortion_ratio`."""
    from src.vtq_v2_format import reconstruct_lut_from_hw as _rec_lut_v2

    t_all = time.perf_counter()
    dev = torch.device("cuda")
    torch.manual_seed(20261116)
    np.random.seed(20261116)

    v2_path = ROOT / "results" / "vtq_v2_tables.json"
    v2_raw = json.loads(v2_path.read_text(encoding="utf-8")) if v2_path.exists() else {}

    N_sb = 6144
    X_gauss = torch.randn((N_sb, TILE), dtype=torch.float32, device=dev)
    X_gauss = X_gauss / X_gauss.square().mean(dim=1, keepdim=True).sqrt().clamp(min=1e-8)

    X_val = torch.randn((4096, TILE), dtype=torch.float32, device=dev)
    X_val = X_val / X_val.square().mean(dim=1, keepdim=True).sqrt().clamp(min=1e-8)
    w_ones = torch.ones((TILE,), dtype=torch.float32, device=dev)

    frac_sub = {0: 0.0, 1: 0.5, 2: 0.25, 3: 0.75}
    trained: Dict[str, dict] = {}
    print(f"=== Training {len(VTQ_SPECS_DEF)} Hardware-Exact VTQ Codebooks (`qmax=127`, `sub=16`) on {torch.cuda.get_device_name(0)} ===", flush=True)

    for name, fc in VTQ_SPECS_DEF:
        t0 = time.perf_counter()
        K, frac, nl, h, st_bytes = FMT_META[fc]
        U = 1 << (K - 1)
        G_cnt = 1 << h
        qmax = 127
        p_idx_np, p_sgn_np, m_mag = _bucket_mapping_vtq(fc)
        p_idx_t = torch.tensor(p_idx_np, dtype=torch.long, device=dev)
        p_sgn_t = torch.tensor(p_sgn_np, dtype=torch.float32, device=dev)

        base_q = _normal_quantiles(2 * m_mag)
        pos_q_asc = np.maximum(base_q[m_mag:], 0.0)
        if fc in (15, 17):
            pos_q_asc[:5] = 0.0
        elif frac > 0:
            n_z = 4 if nl == 16 else (2 if nl == 8 else 1)
            pos_q_asc[:n_z] = 0.0

        seeds_list: list[Tuple[np.ndarray, np.ndarray, int]] = []
        v2_nm = _V2_WARMSTART_MAP.get(name)
        if v2_nm and v2_nm in v2_raw:
            vc = v2_raw[v2_nm]
            vhw = vc.get("hw_codebook") or vc["hw"]
            lut_v2 = _rec_lut_v2(int(vc["K"]), 16, vhw).reshape(16, U)
            pos_v2 = np.maximum(lut_v2[:, U // 2:].T.reshape(-1), 0.0) / max(float(vc.get("scale", 1.0)), 1e-6)
            g_v2 = np.asarray(vc["gains"], dtype=np.float64)
            g_geom0 = float(np.exp(np.mean(np.log(np.maximum(g_v2, 0.1)))))
            g_u8_v2 = np.clip(np.rint(64.0 * (g_v2 / g_geom0)), 16, 220).astype(np.uint8)
            pos_v2_world = pos_v2 * g_geom0
            seeds_list.append((pos_v2_world, g_u8_v2, 10))
        else:
            gain_steps = [0.38, 0.46] if h == 1 else ([0.24, 0.30] if h == 2 else [0.15, 0.19])
            for gs in gain_steps:
                g_exps = np.array([gs * (g - 0.5 * (G_cnt - 1)) for g in range(G_cnt)], dtype=np.float64)
                g_u8_init = np.clip(np.rint(64.0 * (2.0 ** g_exps)), 16, 200).astype(np.uint8)
                for sp_init in (0.96, 1.04):
                    seeds_list.append(((pos_q_asc * sp_init).copy(), g_u8_init.copy(), 12))

        best_cfg = None
        best_val_mse = float("inf")

        def _check_cand(hw_c: dict, gi_c: np.ndarray, sc_c: float):
            nonlocal best_val_mse, best_cfg
            lut_t_c = torch.tensor(reconstruct_lut_from_hw_vtq(fc, hw_c), dtype=torch.float32, device=dev)
            gains_t_c = torch.tensor(gi_c.astype(np.float32) * GAIN_UNIT, dtype=torch.float32, device=dev)
            for sc_mul in (0.95, 0.98, 1.00, 1.02, 1.05):
                sc_try = sc_c * sc_mul
                mse_v, _, _, _, _ = _eval_vtq_cand_on_blocks(X_val, w_ones, lut_t_c, gains_t_c, sc_try, nl, U, frac)
                if mse_v < best_val_mse:
                    best_val_mse = mse_v
                    best_cfg = {
                        "hw_codebook": copy.deepcopy(hw_c),
                        "gain_u8": gi_c.copy(),
                        "scale": float(sc_try),
                    }

        for pos_init, gain_u8_init, n_it in seeds_list:
            hw_cur, stretch = _pos_mags_to_hw(fc, pos_init, qmax)
            scale_cur = float(stretch)
            gain_u8_cur = gain_u8_init.copy()
            # Always evaluate iteration-0 (preserves exact converted VTQ v2 table before Lloyd-Max!)
            _check_cand(hw_cur, gain_u8_cur, scale_cur)

            for it in range(n_it):
                lut_np = reconstruct_lut_from_hw_vtq(fc, hw_cur)
                lut_t = torch.tensor(lut_np, dtype=torch.float32, device=dev)
                gains_t = torch.tensor(gain_u8_cur.astype(np.float32) * GAIN_UNIT, dtype=torch.float32, device=dev)
                _, d_opt, g_out, flat_idx, _ = _eval_vtq_cand_on_blocks(
                    X_gauss, w_ones, lut_t, gains_t, scale_cur, nl, U, frac
                )
                X_norm = X_gauss / d_opt.clamp(min=1e-10)
                X_norm_flat = X_norm.view(-1)
                g_val_sub = gains_t[g_out.long()]
                g_val_flat = g_val_sub.unsqueeze(-1).expand(-1, -1, 16).reshape(-1)
                b_id = p_idx_t[flat_idx.view(-1)]
                s_eff = p_sgn_t[flat_idx.view(-1)] * g_val_flat

                num_b = torch.zeros((m_mag,), dtype=torch.float64, device=dev)
                den_b = torch.zeros((m_mag,), dtype=torch.float64, device=dev)
                num_b.scatter_add_(0, b_id, (s_eff * X_norm_flat).to(torch.float64))
                den_b.scatter_add_(0, b_id, s_eff.square().to(torch.float64))

                if fc in (15, 17):
                    cur_pos_norm = np.asarray(hw_cur["tab_s8"][16:], dtype=np.float64) / 64.0
                else:
                    cur_pos_norm = hw_to_canon_u8_vtq(fc, hw_cur).astype(np.float64) / 64.0
                upd = num_b.cpu().numpy() / np.maximum(den_b.cpu().numpy(), 1e-12)
                has_cnt = den_b.cpu().numpy() > 10.0
                new_pos = np.where(has_cnt, np.maximum(upd, 0.0), cur_pos_norm)
                blended_world = (0.65 * new_pos + 0.35 * cur_pos_norm) * scale_cur
                hw_cur, scale_cur = _pos_mags_to_hw(fc, blended_world, qmax)

                if 1 <= it <= n_it - 2 and G_cnt > 1:
                    lut_new_t = torch.tensor(reconstruct_lut_from_hw_vtq(fc, hw_cur), dtype=torch.float32, device=dev)
                    # Note: each subblock `g` of `X_gauss [N_sb, 256]` has row-wise gains `g_out [N_sb, 16]`
                    tab_vals = lut_new_t[flat_idx].view(N_sb, 16, 16)
                    xn_sub = (X_gauss / (d_opt * (scale_cur / max(stretch, 1e-8))).clamp(min=1e-10)).view(N_sb, 16, 16)
                    num_s = (xn_sub * tab_vals).sum(dim=2).reshape(-1)
                    den_s = (tab_vals * tab_vals).sum(dim=2).reshape(-1)
                    gid_flat = g_out.long().reshape(-1)
                    ng = torch.zeros((G_cnt,), dtype=torch.float64, device=dev)
                    dg = torch.zeros((G_cnt,), dtype=torch.float64, device=dev)
                    ng.scatter_add_(0, gid_flat, num_s.to(torch.float64))
                    dg.scatter_add_(0, gid_flat, den_s.to(torch.float64))
                    g_cont = (ng / dg.clamp(min=1e-12)).cpu().numpy()
                    g_cnt_ok = dg.cpu().numpy() > 20.0
                    g_prev = gain_u8_cur.astype(np.float64) / 64.0
                    g_upd = np.where(g_cnt_ok, np.clip(0.6 * g_cont + 0.4 * g_prev, 0.20, 3.20), g_prev)
                    g_upd = np.sort(g_upd)
                    g_geom = float(np.exp(np.mean(np.log(np.maximum(g_upd, 0.1)))))
                    g_upd = g_upd / max(g_geom, 1e-6)
                    scale_cur *= g_geom
                    g_u8_try = np.clip(np.rint(64.0 * g_upd), 12, 240).astype(np.int32)
                    for gi_k in range(1, G_cnt):
                        if g_u8_try[gi_k] <= g_u8_try[gi_k - 1]:
                            g_u8_try[gi_k] = g_u8_try[gi_k - 1] + 1
                    if g_u8_try[-1] <= 250:
                        gain_u8_cur = g_u8_try.astype(np.uint8)
                stretch = scale_cur

            _check_cand(hw_cur, gain_u8_cur, scale_cur)

        # Discrete coordinate polish (`gain_u8 +- 2`, `+- 1`, and fine `scale`) on `X_val`
        assert best_cfg is not None
        hw_b = best_cfg["hw_codebook"]
        lut_b = torch.tensor(reconstruct_lut_from_hw_vtq(fc, hw_b), dtype=torch.float32, device=dev)
        gi_b = best_cfg["gain_u8"].astype(np.int32).copy()
        sc_b = best_cfg["scale"]

        for delta in (2, 1):
            for g_pos in range(len(gi_b)):
                for sgn in (-1, +1):
                    gi_try = gi_b.copy()
                    gi_try[g_pos] = int(np.clip(int(gi_try[g_pos]) + sgn * delta, 12, 250))
                    if g_pos > 0 and gi_try[g_pos] <= gi_try[g_pos - 1]:
                        continue
                    if g_pos + 1 < len(gi_try) and gi_try[g_pos] >= gi_try[g_pos + 1]:
                        continue
                    gt = torch.tensor(gi_try.astype(np.float32) * GAIN_UNIT, dtype=torch.float32, device=dev)
                    mse_v, _, _, _, _ = _eval_vtq_cand_on_blocks(X_val, w_ones, lut_b, gt, sc_b, nl, U, frac)
                    if mse_v < best_val_mse:
                        best_val_mse = mse_v
                        gi_b = gi_try

        gt_fin = torch.tensor(gi_b.astype(np.float32) * GAIN_UNIT, dtype=torch.float32, device=dev)
        for sc_mul in (0.97, 0.985, 1.00, 1.015, 1.03):
            mse_v, _, _, _, _ = _eval_vtq_cand_on_blocks(X_val, w_ones, lut_b, gt_fin, sc_b * sc_mul, nl, U, frac)
            if mse_v < best_val_mse:
                best_val_mse = mse_v
                sc_b = sc_b * sc_mul

        bpw_256 = (st_bytes * 8.0 / 4096.0) + (2.0 / 256.0)
        r_eff = K - frac_sub[frac]
        snr_bits = 0.5 * math.log2(1.0 / max(best_val_mse, 1e-12))
        gap_shannon = bpw_256 - snr_bits
        gap_trellis = r_eff - snr_bits

        trained[name] = {
            "name": name,
            "fmt_code": fc,
            "K": K,
            "frac": frac,
            "nl": nl,
            "h": h,
            "sub": 16,
            "st_bytes": st_bytes,
            "bpw_256": round(bpw_256, 6),
            "scale": round(float(sc_b), 5),
            "gain_u8": [int(x) for x in gi_b.tolist()],
            "gains": [round(float(x) / 64.0, 6) for x in gi_b.tolist()],
            "hw_codebook": hw_b,
            "mse_gauss": round(best_val_mse, 8),
            "snr_bits_gauss": round(snr_bits, 4),
            "gap_trellis_bits": round(gap_trellis, 4),
            "gap_total_bpw": round(gap_shannon, 4),
            "distortion_ratio": round(best_val_mse / 0.030, 6),
        }
        print(
            f"  [{name:13s}] fmt={fc:2d} bpw={bpw_256:.5f} | MSE={best_val_mse:.6f} "
            f"SNR={snr_bits:.3f}b (trellis_gap={gap_trellis:.3f}b) | gain_u8={gi_b.tolist()} sc={sc_b:.3f} ({time.perf_counter()-t0:.1f}s)",
            flush=True,
        )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(trained, indent=2), encoding="utf-8")
    load_and_register_vtq(out_path)

    if calibrate_model_dir is None:
        cand_dir = ROOT / "models" / "Llama-3.2-3B-Instruct"
        if cand_dir.exists():
            calibrate_model_dir = cand_dir
    if calibrate_model_dir is not None and Path(calibrate_model_dir).exists():
        calibrate_vtq_distortion_ratios(Path(calibrate_model_dir), out_path)
    print(f"=== Saved {len(trained)} VTQ specs to {out_path} in {time.perf_counter()-t_all:.1f}s ===", flush=True)
    return trained


@torch.no_grad()
def calibrate_vtq_distortion_ratios(model_dir: Path, tables_path: Path = DEFAULT_TABLES_PATH):
    """Calibrates `SPEC_DISTORTION_RATIO` on real `Llama-3.2-3B-Instruct` layers (`Loss_VTQ / Loss_diag_LIQK3_V2`)."""
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM

    print(f"\n=== Calibrating VTQ `distortion_ratio` on `{model_dir.name}` ===", flush=True)
    dev = torch.device("cuda")
    load_and_register_vtq(tables_path)
    raw = json.loads(Path(tables_path).read_text(encoding="utf-8"))

    cal_ids = load_file(str(ROOT / "data" / "cal_trace.safetensors"))["input_ids"][:16, :1024]
    model = AutoModelForCausalLM.from_pretrained(str(model_dir), dtype=torch.bfloat16).to("cpu").eval()
    n_layers = model.config.num_hidden_layers
    sample_layers = sorted(set([0, n_layers // 2, n_layers - 1]))

    embed = model.model.embed_tokens.to(dev)
    rotary = model.model.rotary_emb.to(dev)
    H_batches = [embed(cal_ids[i : i + 4].to(dev)) for i in range(0, 16, 4)]
    pos_emb = rotary(H_batches[0][:1], torch.arange(cal_ids.shape[1], device=dev).unsqueeze(0))
    embed.cpu()

    ratios_per_spec: Dict[str, List[float]] = {nm: [] for nm in raw}

    for l_idx in range(n_layers):
        layer = model.model.layers[l_idx].to(dev)
        if l_idx in sample_layers:
            cov_acc = {k: lv2._CovAccumulator() for k in ("qkv", "o", "up", "down")}
            hooks = [
                layer.self_attn.q_proj.register_forward_pre_hook(lambda m, a, ca=cov_acc: ca["qkv"].add(a[0])),
                layer.self_attn.o_proj.register_forward_pre_hook(lambda m, a, ca=cov_acc: ca["o"].add(a[0])),
                layer.mlp.gate_proj.register_forward_pre_hook(lambda m, a, ca=cov_acc: ca["up"].add(a[0])),
                layer.mlp.down_proj.register_forward_pre_hook(lambda m, a, ca=cov_acc: ca["down"].add(a[0])),
            ]
            for b_i in range(len(H_batches)):
                out_b = layer(H_batches[b_i], position_embeddings=pos_emb)
                H_batches[b_i] = out_b[0] if isinstance(out_b, tuple) else out_b
            for hk in hooks:
                hk.remove()
            H_map = {
                "self_attn.q_proj": cov_acc["qkv"].get(),
                "self_attn.o_proj": cov_acc["o"].get(),
                "mlp.gate_proj": cov_acc["up"].get(),
                "mlp.down_proj": cov_acc["down"].get(),
            }
            del cov_acc
            for sub_i, (subname, H_m) in enumerate(H_map.items()):
                W_full = layer.get_submodule(subname).weight.data.float()
                M_full = W_full.shape[0]
                if M_full > 512:
                    r_idx = torch.linspace(0, M_full - 1, 512, device=dev).long()
                    W_m = W_full[r_idx].contiguous()
                else:
                    W_m = W_full
                del W_full
                seed_m = 1000 + l_idx * 10 + sub_i
                W_rot, H_rot, _ = compute_suh_vtq(W_m, H_m, seed=seed_m, rot_block=-1)
                diag_h = torch.diagonal(H_rot).clamp(min=1e-8)
                diff3 = W_rot - lv2.quantize_liqk_v2_diag_rot(W_rot, diag_h, "LIQK3_V2")
                base_loss = float(((diff3 @ H_rot) * diff3).sum().clamp(min=1e-12).item())
                del W_rot, H_rot, diff3

                HW = W_m @ H_m
                for nm in raw:
                    _, W_hat, _ = quantize_matrix_vtq(
                        W_m, H_m, nm, seed=seed_m, damp=0.20, rot_block=-1, u_sb_gecd_sweeps=None
                    )
                    dW = W_m - W_hat
                    vtq_loss = float(((HW - (W_hat @ H_m)) * dW).sum().clamp(min=1e-12).item())
                    ratios_per_spec[nm].append(vtq_loss / base_loss)
            del H_map
        else:
            for b_i in range(len(H_batches)):
                out_b = layer(H_batches[b_i], position_embeddings=pos_emb)
                H_batches[b_i] = out_b[0] if isinstance(out_b, tuple) else out_b
        layer.cpu()
        torch.cuda.empty_cache()

    del model, H_batches
    torch.cuda.empty_cache()

    ordered = sorted(
        raw.keys(),
        key=lambda k: (raw[k]["bpw_256"], 1 if k.endswith(("_FAST", "_HQ")) else 0),
    )
    prev_dr = float("inf")
    for nm in ordered:
        vals = np.array(ratios_per_spec[nm], dtype=np.float64)
        geom = float(np.exp(np.mean(np.log(np.maximum(vals, 1e-12)))))
        if not nm.endswith(("_FAST", "_HQ")):
            geom = min(geom, prev_dr * 0.985)
            prev_dr = geom
        raw[nm]["distortion_ratio"] = round(geom, 6)
        print(f"  [Calibrated] {nm:13s} bpw={raw[nm]['bpw_256']:.5f} -> distortion_ratio={geom:.5f}", flush=True)

    Path(tables_path).write_text(json.dumps(raw, indent=2), encoding="utf-8")
    load_and_register_vtq(tables_path)


# =================================================================================================
# 5. Fisher Slice Profiler & Full-Model Streaming GPTAQ + Checkpointed Scale Polish
# =================================================================================================

@torch.no_grad()
def profile_matrix_vtq(
    slices: list,
    l_idx: int,
    full: str,
    W: torch.Tensor,
    H: torch.Tensor,
    seed: int,
    step: int,
    s_row: Optional[torch.Tensor] = None,
    rot_block: int = -1,
    rot_mode: str = "outer",
):
    """Profiles per-slice Fisher-weighted distortion under VTQ preconditioning (`suh` + `Outer-DCT` / `FWHT-256`)."""
    dev = W.device
    rw = s_row.to(dev).float() if s_row is not None else None
    rb = resolve_rot_block(full, rot_block=rot_block, rot_mode=rot_mode)
    W_rot, H_rot, _ = compute_suh_vtq(W.float(), H.float(), seed=seed, row_weights=rw, rot_block=rb)
    diag_h = torch.diagonal(H_rot).clamp(min=1e-8)
    diff = W_rot - lv2.quantize_liqk_v2_diag_rot(W_rot, diag_h, "LIQK3_V2")
    row_err = ((diff @ H_rot) * diff).sum(dim=1).clamp(min=1e-12)
    M, K = W.shape
    if rw is not None:
        row_err = row_err * rw
    for i, v in enumerate((0.5 * row_err).view(M // step, step).sum(dim=1).tolist()):
        slices.append(
            {"layer": l_idx, "subname": full, "r0": i * step, "r1": (i + 1) * step, "params": step * K, "raw_kld3": v}
        )


@torch.no_grad()
def build_vtq_fisher_slice_profile(
    model: nn.Module,
    cal_tokens: torch.Tensor,
    candidate_specs: Optional[List[str]] = None,
    batch_size: int = 8,
    attn_slice_rows: int = 256,
    kv_slice_rows: int = 128,
    mlp_slice_rows: int = 512,
    fisher_out: Optional[Dict[Tuple[int, str], torch.Tensor]] = None,
    rot_block: int = -1,
    rot_mode: str = "outer",
) -> dict:
    """Builds the VTQ Fisher slice profile with 128-row GQA KV slices and lossless SwiGLU `mlp_perms`."""
    if not VTQ_CONFIG:
        load_and_register_vtq()
    if candidate_specs is None:
        candidate_specs = list(VTQ_MENU)

    device = next(model.parameters()).device
    n_layers = model.config.num_hidden_layers
    if fisher_out is None:
        fisher_out = lv2.compute_exact_row_fisher_sensitivities(model, cal_tokens, n_Menu_seqs=8, seq_len=1024)

    n_seqs = min(16, cal_tokens.shape[0])
    H_batches = [model.model.embed_tokens(cal_tokens[i : i + batch_size].to(device)) for i in range(0, n_seqs, batch_size)]
    pos_emb = model.model.rotary_emb(H_batches[0][:1], torch.arange(cal_tokens.shape[1], device=device).unsqueeze(0))

    slices_info = []
    mlp_perms: Dict[int, torch.Tensor] = {}
    fisher_out_perm: Dict[Tuple[int, str], torch.Tensor] = {k: v.clone() for k, v in fisher_out.items()}

    for l_idx in range(n_layers):
        layer = model.model.layers[l_idx]
        cov_acc = {name: lv2._CovAccumulator() for name in ("qkv", "o", "up", "down")}
        hooks = [
            layer.self_attn.q_proj.register_forward_pre_hook(lambda m, a, ca=cov_acc: ca["qkv"].add(a[0])),
            layer.self_attn.o_proj.register_forward_pre_hook(lambda m, a, ca=cov_acc: ca["o"].add(a[0])),
            layer.mlp.gate_proj.register_forward_pre_hook(lambda m, a, ca=cov_acc: ca["up"].add(a[0])),
            layer.mlp.down_proj.register_forward_pre_hook(lambda m, a, ca=cov_acc: ca["down"].add(a[0])),
        ]
        for b_i in range(len(H_batches)):
            out_b = layer(H_batches[b_i], position_embeddings=pos_emb)
            H_batches[b_i] = out_b[0] if isinstance(out_b, tuple) else out_b
        for hk in hooks:
            hk.remove()

        H_qkv, H_o, H_up, H_down = (cov_acc[k].get() for k in ("qkv", "o", "up", "down"))
        del cov_acc

        wg = layer.mlp.gate_proj.weight.data.float()
        wu = layer.mlp.up_proj.weight.data.float()
        sg = fisher_out[(l_idx, "mlp.gate_proj")].to(device).float()
        su = fisher_out[(l_idx, "mlp.up_proj")].to(device).float()
        diag_up = torch.diagonal(H_up).clamp(min=1e-8)
        mlp_imp = sg * (wg * wg * diag_up.unsqueeze(0)).sum(dim=1) + su * (wu * wu * diag_up.unsqueeze(0)).sum(dim=1)
        mlp_imp_q = torch.round(torch.log10(mlp_imp.clamp(min=1e-30)) * 1000.0)
        perm_desc = torch.sort(mlp_imp_q, descending=True, stable=True).indices
        Kd = wg.shape[0]
        perm_bal2 = perm_desc.view(Kd // 512, TILE, 2).transpose(1, 2).reshape(Kd)
        mlp_perms[l_idx] = perm_bal2.cpu()

        fisher_out_perm[(l_idx, "mlp.gate_proj")] = sg[perm_bal2].cpu()
        fisher_out_perm[(l_idx, "mlp.up_proj")] = su[perm_bal2].cpu()
        H_down_perm = H_down[perm_bal2][:, perm_bal2]

        hess_map = {
            "self_attn.q_proj": H_qkv,
            "self_attn.k_proj": H_qkv,
            "self_attn.v_proj": H_qkv,
            "self_attn.o_proj": H_o,
            "mlp.gate_proj": H_up,
            "mlp.up_proj": H_up,
            "mlp.down_proj": H_down_perm,
        }
        for idx_s, subname in enumerate(lv2.LINEAR_SUBMODULES):
            W_f32 = layer.get_submodule(subname).weight.data.float()
            if subname in ("mlp.gate_proj", "mlp.up_proj"):
                W_f32 = W_f32[perm_bal2]
            elif subname == "mlp.down_proj":
                W_f32 = W_f32[:, perm_bal2]
            M_r = W_f32.shape[0]
            if subname in ("mlp.gate_proj", "mlp.up_proj"):
                step = min(mlp_slice_rows, M_r)
            elif subname in ("self_attn.k_proj", "self_attn.v_proj") and M_r % kv_slice_rows == 0:
                step = min(kv_slice_rows, M_r)
            else:
                step = min(attn_slice_rows, M_r)
            profile_matrix_vtq(
                slices_info,
                l_idx,
                subname,
                W_f32,
                hess_map[subname],
                seed=1000 + l_idx * 10 + idx_s,
                step=step,
                s_row=fisher_out_perm[(l_idx, subname)],
                rot_block=rot_block,
                rot_mode=rot_mode,
            )

    kpp = np.array([s["raw_kld3"] / s["params"] for s in slices_info], dtype=np.float64)
    mean_kpp = float(np.mean(kpp))
    for s in slices_info:
        kpp_clamped = min(max(s["raw_kld3"] / s["params"], 0.20 * mean_kpp), 4.5 * mean_kpp)
        base_c3 = kpp_clamped * s["params"]
        s["costs"] = {sp: base_c3 * lv2.SPEC_DISTORTION_RATIO[sp] for sp in candidate_specs}

    return {
        "slices": slices_info,
        "mlp_perms": mlp_perms,
        "fisher_out": {k: v.cpu() for k, v in fisher_out_perm.items()},
    }


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
) -> float:
    """O(1)-activation-VRAM gradient-checkpointed `(fp16 d_row, fp16 suh)` KLD scale polish (0 format overhead)."""
    t0 = time.perf_counter()
    device = next(model.parameters()).device
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

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
                w_frac = (c1 - c0) / float(n_tok)
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
                loss_chunk = (fwd_kl + rev_kl_weight * rev_kl + ce_weight * ce_loss) * w_frac
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

    del opt, scale_params, scales_alpha, scales_beta, cal_pol
    torch.cuda.empty_cache()
    return time.perf_counter() - t0


def _drift_target(w, b, alpha):
    return w * (1.0 + alpha * b) if b.ndim == 1 else w + alpha * (w @ b.T)


@torch.no_grad()
def quantize_llama_model_vtq(
    model: nn.Module,
    cal_tokens: torch.Tensor,
    spec_or_assignment: str | Dict[Tuple[int, str], List[Tuple[int, int, str]]],
    mlp_perms: Optional[Dict[int, torch.Tensor]] = None,
    fisher_out: Optional[Dict[Tuple[int, str], torch.Tensor]] = None,
    batch_size: int = 4,
    damp: float = 0.20,
    gptaq_alpha: Optional[float] = None,
    rot_block: int = -1,
    rot_mode: str = "outer",
    d_sb_gecd_sweeps: int = 1,
    polish_steps: int = 0,
    polish_lr: float = 4e-4,
    ref_hidden_cpu: Optional[torch.Tensor] = None,
    final_calibration=None,
    down_output_fit=None,
    reuse_reference_mlp: bool = False,
    compact_gptaq: bool = False,
) -> dict:
    """Layer-by-layer streaming VTQ quantizer with asymmetric `GPTAQ` (`Qwen` QK-Norm safe via `sq._attn`)
    and optional `O(1)`-layer gradient-checkpointed scale polish."""
    if not VTQ_CONFIG:
        load_and_register_vtq()

    import src.streaming_quant as sq

    t_start = time.perf_counter()
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    model.eval()

    has_qknorm = hasattr(model.model.layers[0].self_attn, "q_norm")
    eff_gptaq = (0.0 if has_qknorm else 0.50) if gptaq_alpha is None else float(gptaq_alpha)

    if polish_steps > 0 and ref_hidden_cpu is None:
        ref_hidden_cpu = lv2.precompute_ref_final_hidden_states(model, cal_tokens, n_seqs=64, seq_len=512, batch_size=2)
    if fisher_out is None:
        fisher_out = lv2.compute_exact_row_fisher_sensitivities(model, cal_tokens, n_Menu_seqs=8, seq_len=1024)

    n_seqs, seq_len = cal_tokens.shape
    n_layers = model.config.num_hidden_layers
    use_g = eff_gptaq > 0.0
    cov_factory = ConsumableCov if compact_gptaq else DualCovAccumulator
    reuse_reference_mlp = reuse_reference_mlp or compact_gptaq

    cal_q = [model.model.embed_tokens(cal_tokens[i : i + batch_size].to(device)) for i in range(0, n_seqs, batch_size)]
    cal_r = [c.clone() for c in cal_q] if use_g else cal_q
    pos_cal = model.model.rotary_emb(cal_q[0][:1], torch.arange(seq_len, device=device).unsqueeze(0))

    total_params = 0
    total_bits = 0.0

    for l_idx in range(n_layers):
        lq = model.model.layers[l_idx]
        if mlp_perms is not None and l_idx in mlp_perms:
            perm_l = mlp_perms[l_idx].to(device)
            lq.mlp.gate_proj.weight.data = lq.mlp.gate_proj.weight.data[perm_l].contiguous()
            lq.mlp.up_proj.weight.data = lq.mlp.up_proj.weight.data[perm_l].contiguous()
            lq.mlp.down_proj.weight.data = lq.mlp.down_proj.weight.data[:, perm_l].contiguous()
        lr = copy.deepcopy(lq) if use_g else lq

        def q_mod(mod: nn.Linear, H: torch.Tensor, B: Optional[torch.Tensor], full_name: str, seed_i: int, dm: float, *, output_delta=None):
            nonlocal total_params, total_bits
            W = mod.weight.data.float()
            M_r, K_in = W.shape
            r_specs = [(0, M_r, spec_or_assignment)] if isinstance(spec_or_assignment, str) else spec_or_assignment[(l_idx, full_name)]
            W_tgt = None
            if output_delta is not None:
                W_tgt = W + eff_gptaq * output_delta
            elif use_g and B is not None:
                W_tgt = _drift_target(W, B, eff_gptaq)
            sr = fisher_out[(l_idx, full_name)].to(device)
            rb = resolve_rot_block(full_name, rot_block=rot_block, rot_mode=rot_mode)
            _, w_hat, _ = quantize_matrix_vtq(
                W, H, r_specs, seed=seed_i, damp=damp * dm, row_weights=sr,
                rot_block=rb, W_target=W_tgt, u_sb_gecd_sweeps=d_sb_gecd_sweeps,
            )
            mod.weight.data.copy_(w_hat.to(dtype=dtype, device=device))
            for r0, r1, sp in r_specs:
                n_p = (r1 - r0) * K_in
                total_params += n_p
                total_bits += n_p * lv2.LIQK_V2_SPECS[sp].bpw

        # 1) q_proj, k_proj, v_proj
        g_damp = max(damp, 0.20)
        acc = cov_factory(use_g)
        for cq, cr in zip(cal_q, cal_r):
            acc.add(lq.input_layernorm(cq), lr.input_layernorm(cr) if use_g else None)
        Hqkv, Bqkv = acc.get(gptaq_damp=g_damp)
        for i_s, nm_s in enumerate(("q_proj", "k_proj", "v_proj")):
            q_mod(getattr(lq.self_attn, nm_s), Hqkv, Bqkv, f"self_attn.{nm_s}", 1000 + l_idx * 10 + i_s, 1.0)
        del Hqkv, Bqkv, acc

        # 2) o_proj + attention residual update
        acc = cov_factory(use_g)
        bq, br = [None], [None]
        hq = lq.self_attn.o_proj.register_forward_pre_hook(lambda m, a: bq.__setitem__(0, a[0]))
        hr = lr.self_attn.o_proj.register_forward_pre_hook(lambda m, a: br.__setitem__(0, a[0])) if use_g else None
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
        Ho, Bo = acc.get(gptaq_damp=g_damp)
        q_mod(lq.self_attn.o_proj, Ho, Bo, "self_attn.o_proj", 1000 + l_idx * 10 + 3, 1.0)
        del Ho, Bo, acc

        for i in range(len(cal_q)):
            cal_q[i] = cal_q[i] + sq._attn(lq, cal_q[i], pos_cal)
        if not use_g:
            ref_mid = cal_q

        # 3) gate_proj, up_proj
        acc = cov_factory(use_g)
        for cq, crm in zip(cal_q, ref_mid):
            acc.add(lq.post_attention_layernorm(cq), lr.post_attention_layernorm(crm) if use_g else None)
        Hup, Bup = acc.get(gptaq_damp=g_damp)
        for i_s, nm_s in enumerate(("gate_proj", "up_proj")):
            q_mod(getattr(lq.mlp, nm_s), Hup, Bup, f"mlp.{nm_s}", 1000 + l_idx * 10 + 4 + i_s, 1.0)
        del Hup, Bup, acc

        # 4) down_proj + MLP residual update
        fit_output = down_output_fit is not None and use_g
        acc = down_output_fit() if fit_output else cov_factory(use_g)
        hq = lq.mlp.down_proj.register_forward_pre_hook(lambda m, a: bq.__setitem__(0, a[0]))
        hr = lr.mlp.down_proj.register_forward_pre_hook(lambda m, a: br.__setitem__(0, a[0])) if use_g and not fit_output else None
        for i, (cq, crm) in enumerate(zip(cal_q, ref_mid)):
            if fit_output:
                yq = lq.mlp(lq.post_attention_layernorm(cq))
                yr = lr.mlp(lr.post_attention_layernorm(crm))
                acc.add_outputs(bq[0], yq, yr, cq, crm)
                cal_r[i] = crm + yr
                del yq, yr
            else:
                lq.mlp(lq.post_attention_layernorm(cq))
                if use_g:
                    if reuse_reference_mlp:
                        cal_r[i] = crm + lr.mlp(lr.post_attention_layernorm(crm))
                    else:
                        lr.mlp(lr.post_attention_layernorm(crm))
                acc.add(bq[0], br[0])
            bq[0] = br[0] = None
        hq.remove()
        if hr is not None:
            hr.remove()
        Hdn, Bdn = acc.get(gptaq_damp=g_damp * 1.5)
        q_mod(lq.mlp.down_proj, Hdn, None if fit_output else Bdn, "mlp.down_proj", 1000 + l_idx * 10 + 6, 1.5,
              output_delta=Bdn if fit_output else None)
        del Hdn, Bdn, acc

        for i in range(len(cal_q)):
            cal_q[i] = cal_q[i] + lq.mlp(lq.post_attention_layernorm(cal_q[i]))
        if use_g:
            if not (fit_output or reuse_reference_mlp):
                for i in range(len(cal_r)):
                    cal_r[i] = ref_mid[i] + lr.mlp(lr.post_attention_layernorm(ref_mid[i]))
            del lr
        del ref_mid

    if final_calibration is not None:
        if not use_g:
            raise ValueError('paired final calibration requires a distinct reference stream')
        final_calibration(model, cal_q, cal_r)
    del cal_q, cal_r
    torch.cuda.empty_cache()
    quant_sec = time.perf_counter() - t_start

    polish_sec = 0.0
    if polish_steps > 0 and ref_hidden_cpu is not None:
        with torch.enable_grad():
            polish_sec = polish_model_scales_kld_checkpointed(
                model, cal_tokens, ref_hidden_cpu, n_steps=polish_steps, lr=polish_lr, batch_size=2
            )

    return {
        "linear_params": total_params,
        "linear_bytes": int(round(total_bits / 8.0)),
        "linear_bpw": total_bits / max(total_params, 1),
        "quant_sec": quant_sec,
        "polish_sec": polish_sec,
    }


# =================================================================================================
# 6. `quantize_rings_vtq` (Embeddings, `lm_head`, Gemma-4 PLE) & `install_vtq_hooks`
# =================================================================================================

@torch.no_grad()
def quantize_rings_vtq(
    W: torch.Tensor,
    spec_name: str,
    ring_len: int = TILE,
    rot: bool = True,
    chunk: int = 8192,
    scales: tuple[float, ...] = (0.94, 1.00, 1.07),
) -> Tuple[torch.Tensor, float]:
    """Quantizes 2D table `W [rows, n * ring_len]` (`ring_len <= 256`) using VTQ 256-state tail-biting rings."""
    if not VTQ_CONFIG:
        load_and_register_vtq()
    assert 0 < ring_len <= TILE, f"VTQ rings require ring_len <= 256, got {ring_len}"
    dev = torch.device("cuda")
    cfg, lut_t, gains_t = _get_spec_gpu_vtq(spec_name, dev)
    K = int(cfg["K"])
    U = 1 << (K - 1)
    nl = int(cfg["nl"])
    frac = int(cfg["frac"])
    sc_base = float(cfg.get("scale", 1.0))
    signs = make_rademacher_signs(ring_len, seed=4242, device=dev)

    def _rot_ring(x: torch.Tensor, inv: bool) -> torch.Tensor:
        if not rot:
            return x
        if not inv:
            return apply_vtq_rot(x * signs.unsqueeze(0), rot_block=-1, transpose=False)
        return apply_vtq_rot(x, rot_block=-1, transpose=True) * signs.unsqueeze(0)

    rings = W.view(-1, ring_len)
    out = torch.empty_like(rings)
    pp = _FRAC_TO_TCQ_PP[frac]
    m_extra = _NL_TO_M_EXTRA[nl]
    n_cand = 4 if gains_t.numel() > 4 else 0
    tb_win = 32 if ring_len >= 128 else 0
    eff_chunk = min(chunk, 8192) if frac > 0 else chunk

    for i in range(0, rings.shape[0], eff_chunk):
        x_raw = _rot_ring(rings[i : i + eff_chunk].to(dev).float(), inv=False)
        rms = x_raw.square().mean(dim=1, keepdim=True).sqrt().clamp(min=1e-8) * sc_base
        best, best_err = None, None
        for sc in scales:
            d0 = (rms * sc).half().float().clamp(min=1e-8)
            G_r = viterbi_ub_cuda(
                x_raw / d0,
                lut_t,
                K,
                10,
                359,
                gains_t,
                sub=16,
                n_cand=n_cand,
                tail_biting=True,
                tb_window=tb_win,
                m_extra=m_extra,
                pair_parity=pp,
            )
            d_opt = ((x_raw * G_r).sum(dim=1) / (G_r * G_r).sum(dim=1).clamp(min=1e-12)).clamp(min=1e-8).half().float().unsqueeze(1)
            rec = d_opt * G_r
            err = (rec - x_raw).square().sum(dim=1)
            if best is None:
                best, best_err = rec, err
            else:
                imp = err < best_err
                best = torch.where(imp.unsqueeze(1), rec, best)
                best_err = torch.where(imp, err, best_err)
        out[i : i + eff_chunk] = _rot_ring(best, inv=True).to(out.dtype).cpu()

    # Exact bit rate per ring element (scaled linearly if `ring_len < 256` + `fp16` row scale):
    bpv = (cfg["st_bytes"] * 8.0 / 4096.0) + (16.0 / ring_len if ring_len < TILE else 16.0 / max(W.shape[1], ring_len))
    return out.view_as(W), bpv


_HOOKS_VTQ_INSTALLED = False


def install_vtq_hooks():
    """Patches `streaming_quant` and `hybrid_quant` so any `VTQ*` format or `VTQ` DP assignment
    automatically executes `quantize_matrix_vtq` and `profile_matrix_vtq`."""
    global _HOOKS_VTQ_INSTALLED
    load_and_register_vtq()
    if _HOOKS_VTQ_INSTALLED:
        return

    import src.streaming_quant as sqm

    orig_quant = sqm._quantize_linear_weight
    orig_prof = sqm.profile_matrix
    orig_dp = lv2.solve_fisher_slice_dp

    def _is_vtq(spec) -> bool:
        if isinstance(spec, str):
            return spec in VTQ_CONFIG
        if isinstance(spec, list) and spec:
            return spec[0][2] in VTQ_CONFIG
        return False

    def _patched_quant(W: torch.Tensor, H: torch.Tensor, spec, seed: int, damp: float, full: str | None = None) -> torch.Tensor:
        if _is_vtq(spec):
            rw = sqm._row_w(full, W.device) if full else None
            vtq_damp = damp * 10.0 if damp < 0.08 else damp
            rot_mode = os.environ.get("VTQ_ROT_MODE", "outer")
            rb = resolve_rot_block(full, rot_block=-1, rot_mode=rot_mode)
            _, w_hat, _ = quantize_matrix_vtq(W, H, spec, seed=seed, damp=vtq_damp, row_weights=rw, rot_block=rb)
            return w_hat.to(W.dtype)
        return orig_quant(W, H, spec, seed, damp, full)

    def _patched_prof(slices: list, l_idx: int, full: str, W: torch.Tensor, H: torch.Tensor, seed: int, step: int):
        if os.environ.get("VTQ_PROFILE", "0") == "1":
            s_row = sqm.FISHER[(l_idx, full)].to(W.device).float() if sqm.FISHER is not None else None
            rot_mode = os.environ.get("VTQ_ROT_MODE", "outer")
            return profile_matrix_vtq(slices, l_idx, full, W, H, seed, step, s_row=s_row, rot_block=-1, rot_mode=rot_mode)
        return orig_prof(slices, l_idx, full, W, H, seed, step)

    def _patched_dp(profile: dict, target_bpw: float, candidate_specs: Optional[List[str]] = None, sens_power: float = 1.0):
        if candidate_specs is not None:
            s_set = set(candidate_specs)
            if VTQ_MENU and s_set == set(VTQ_MENU):
                candidate_specs = vtq_candidates_for_target(target_bpw, VTQ_MENU)
            elif VTQ_CONFIG and s_set.issubset(set(VTQ_CONFIG.keys())) and len(s_set) >= 10:
                candidate_specs = vtq_candidates_for_target(target_bpw, list(candidate_specs))
        return orig_dp(profile, target_bpw=target_bpw, candidate_specs=candidate_specs, sens_power=sens_power)

    sqm._quantize_linear_weight = _patched_quant
    sqm.profile_matrix = _patched_prof
    lv2.solve_fisher_slice_dp = _patched_dp
    try:
        import src.hybrid_quant as hqm

        hqm._quantize_linear_weight = _patched_quant
        hqm.profile_matrix = _patched_prof
    except Exception:
        pass
    _HOOKS_VTQ_INSTALLED = True


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-tables", action="store_true")
    ap.add_argument("--calibrate-only", action="store_true")
    ap.add_argument("--out", type=Path, default=DEFAULT_TABLES_PATH)
    ap.add_argument("--model-dir", type=Path, default=ROOT / "models" / "Llama-3.2-3B-Instruct")
    args = ap.parse_args()
    if args.train_tables:
        train_vtq_tables(out_path=args.out, calibrate_model_dir=args.model_dir)
    elif args.calibrate_only:
        calibrate_vtq_distortion_ratios(args.model_dir, tables_path=args.out)
