"""
VTQ v2 Production Quantizer & Multi-Architecture Pipeline (`src/vtq_v2.py`):
  1. Loads `results/vtq_v2_tables.json` (`26` hardware-legal `uint8` codebooks `VTQ150..VTQ600`, `1.59–6.22 bpw`)
     and registers them into `liqk_v2.LIQK_V2_SPECS`, `liqk_v2.SPEC_DISTORTION_RATIO`, and `equal_bpw.SETS["VTQ2"]`.
  2. `quantize_matrix_vtq2`:
     - Canonical Kronecker rotation `R_K = H_{2^a} (x) DCT-II_q` across full matrix width `K` (`256 | K`),
     - Optimal `suh` preconditioning `(p_w = 0.50, p_h = -0.25)` with Rademacher signs,
     - Fused C++/CUDA `ldlq_block256_cuda` supporting Extended Hamming `[8, 4, 4]` Gosset E8 (`pp = 4`),
       `D4` (`pp = 2`), `D4-eq` (`pp = 3`), `D2` (`pp = 1`), and full uncoded (`pp = 0`),
     - Both single-format (`str`) and mixed-precision DP row-slice (`[(r0, r1, spec), ...]`) assignments,
     - Blockwise Cholesky `refit_exp2` (`d_row * 2^(0.25 * e)`, `e in [-16, 15]`) + Fisher-weighted `c_row / suh` refit,
     - Exact `.vtq` v2 bitplane emission (`PACK["matrix"]`) whose `VtqFile.decode_matrix` matches `W_hat` bit-for-bit,
     - Optional asymmetric stream-compensated target `W_target = W + alpha * (W @ B^T)` (`GPTAQ`).
  3. `build_vtq2_fisher_slice_profile` & `quantize_llama_model_vtq2`:
     - 128-row GQA KV head slices + lossless SwiGLU `mlp_perms` + Kronecker `R_K` Fisher slice profiling,
     - Full-model streaming `GPTAQ` (`alpha = 0.50`) + optional end-to-end `fp16` scale polish (`polish_model_scales_kld`).
  4. `quantize_rings_vtq2`:
     - Tail-biting `VtqRing` quantizer (`L = 256` or `L = 160`) with Hadamard/Kronecker rotation for `embed_tokens`,
       `lm_head`, and Gemma 4 PLE (`embed_tokens_per_layer`).
  5. `install_vtq2_hooks`:
     - Transparently routes `VTQ*` formats and `VTQ2` DP assignments in `streaming_quant.py` (8B / 30B MoE) and
       `hybrid_quant.py` (Qwen3.5 DeltaNet / Gemma 4 E4B PLE) to `quantize_matrix_vtq2`.
"""

from __future__ import annotations

import copy
import json
import math
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import src.liqk_v2 as lv2
from src.iqk_format import make_rademacher_signs
from src.tcq_proto import ldlq_block256_cuda, viterbi_ub_cuda
from src.vtq_v2_format import NU, TILE, apply_kronecker_rot, assemble_vtq2, reconstruct_lut_from_hw

PP_SUB = {0: 0.0, 1: 0.5, 2: 0.25, 3: 0.75, 4: 0.5}
PACK: dict | None = None

VTQ2_CONFIG: Dict[str, dict] = {}
VTQ2_MENU: List[str] = []
_GPU_TAB_CACHE: Dict[Tuple[str, torch.device], Tuple[torch.Tensor, torch.Tensor]] = {}


def load_and_register_vtq2(tables_path: Path | None = None) -> Dict[str, dict]:
    """Loads `results/vtq_v2_tables.json` and registers all VTQ v2 formats into `liqk_v2` and `equal_bpw.SETS`."""
    global VTQ2_MENU
    if tables_path is None:
        tables_path = Path(__file__).resolve().parent.parent / "results" / "vtq_v2_tables.json"
    tables_path = Path(tables_path)
    if not tables_path.exists():
        return VTQ2_CONFIG

    raw = json.loads(tables_path.read_text(encoding="utf-8"))
    VTQ2_CONFIG.clear()
    for name, c in raw.items():
        K = int(c["K"])
        masks = (int(c["m"]),) + tuple(int(x) for x in c.get("m_extra", ()))
        nl = 2 ** len(masks) * 2
        hw = dict(c.get("hw_codebook") or c["hw"])
        lut = reconstruct_lut_from_hw(K, nl, hw)
        cfg = dict(c)
        cfg["lut_np"] = lut
        cfg["gains_np"] = np.asarray(c["gains"], dtype=np.float32)
        VTQ2_CONFIG[name] = cfg

        pp = int(c.get("pp", 0))
        sub = int(c.get("sub", 16))
        h = int(c["h"])
        bits_256 = TILE * (K - PP_SUB[pp]) + (TILE // sub) * h + 5 + 2
        lv2.LIQK_V2_SPECS[name] = lv2.LiqkV2Spec(
            name=name,
            bits=float(K - PP_SUB[pp]),
            n_levels=nl * (1 << (K - 1)),
            n_pos_states=1 << h,
            block_bytes=bits_256 / 8.0,
        )
        if "distortion_ratio" in c:
            lv2.SPEC_DISTORTION_RATIO[name] = float(c["distortion_ratio"])

    # Active DP menu excludes D2 control `VTQ150_D2` and Gaussian-overrated low-`h` duplicates (`VTQ200_H0`, `VTQ300_H1`, `VTQ400_H2`, `VTQ500_H2`, `VTQ600_H2`)
    VTQ2_MENU = [
        nm for nm in VTQ2_CONFIG
        if nm not in ("VTQ150_D2", "VTQ200_H0", "VTQ300_H1", "VTQ400_H2", "VTQ500_H2", "VTQ600_H2")
    ]
    try:
        import src.equal_bpw as eqb

        eqb.SETS["VTQ2"] = list(VTQ2_MENU)
        # Sub-menus for ultra-low (<2.1 bpw) or standard (>=2.0 bpw) budgets
        eqb.SETS["VTQ2_STD"] = [nm for nm in VTQ2_MENU if not nm.startswith(("VTQ150", "VTQ175"))]
    except Exception:
        pass
    return VTQ2_CONFIG


def vtq2_candidates_for_target(target_bpw: float) -> List[str]:
    """Returns `VTQ2_MENU` filtered to `[target_bpw - 0.45, 6.25]` so linear DP never starves slices >0.45 bpw below budget."""
    if not VTQ2_CONFIG:
        load_and_register_vtq2()
    lo = float(target_bpw) - 0.45
    cands = [sp for sp in VTQ2_MENU if lv2.LIQK_V2_SPECS[sp].bpw >= lo]
    below = [sp for sp in cands if lv2.LIQK_V2_SPECS[sp].bpw <= float(target_bpw) - 0.02]
    if not below:
        all_below = sorted(
            [sp for sp in VTQ2_MENU if lv2.LIQK_V2_SPECS[sp].bpw <= float(target_bpw)],
            key=lambda s: lv2.LIQK_V2_SPECS[s].bpw,
        )
        if all_below and all_below[-1] not in cands:
            cands = [all_below[-1]] + cands
    return cands if cands else list(VTQ2_MENU)


def _get_spec_gpu(spec_name: str, dev: torch.device) -> Tuple[dict, torch.Tensor, torch.Tensor]:
    if not VTQ2_CONFIG:
        load_and_register_vtq2()
    cfg = VTQ2_CONFIG[spec_name]
    key = (spec_name, dev)
    if key not in _GPU_TAB_CACHE:
        _GPU_TAB_CACHE[key] = (
            torch.tensor(cfg["lut_np"], dtype=torch.float32, device=dev),
            torch.tensor(cfg["gains_np"], dtype=torch.float32, device=dev),
        )
    tab, gains = _GPU_TAB_CACHE[key]
    return cfg, tab, gains


# ------------------------------------------------------------------------------------------------
# Asymmetric GPTAQ Dual Covariance Accumulator
# ------------------------------------------------------------------------------------------------

class DualCovAccumulator:
    """Accumulates quantized input Hessian `H = X_q^T X_q / N` and BF16 drift cross-covariance
    `C = X_q^T (X_ref - X_q) / N` in fp64, producing `B = H_reg^{-1} C` so `W_target = W + alpha * (W @ B.T)`."""

    def __init__(self, need_cross: bool = True):
        self.need_cross = need_cross
        self.sum_xx: torch.Tensor | None = None
        self.sum_qc: torch.Tensor | None = None
        self.n_tokens: int = 0

    @torch.no_grad()
    def add(self, x_q: torch.Tensor, x_ref: torch.Tensor | None = None):
        xf = x_q.reshape(-1, x_q.shape[-1]).float()
        xx = torch.mm(xf.T, xf).to(torch.float64)
        self.sum_xx = xx if self.sum_xx is None else self.sum_xx.add_(xx)
        if self.need_cross and x_ref is not None:
            rf = x_ref.reshape(-1, x_ref.shape[-1]).float()
            qc = torch.mm(xf.T, rf - xf).to(torch.float64)
            self.sum_qc = qc if self.sum_qc is None else self.sum_qc.add_(qc)
        self.n_tokens += xf.shape[0]

    @torch.no_grad()
    def get(self, gptaq_damp: float = 0.20) -> Tuple[torch.Tensor, torch.Tensor | None]:
        assert self.sum_xx is not None and self.n_tokens > 0
        H = (self.sum_xx / self.n_tokens).float()
        H = 0.5 * (H + H.T)
        if not self.need_cross or self.sum_qc is None:
            return H, None
        C = (self.sum_qc / self.n_tokens).float()
        H_reg = H.clone()
        H_reg.diagonal().add_(gptaq_damp * H_reg.diagonal().mean().clamp(min=1e-8))
        B = torch.linalg.solve(H_reg, C).contiguous()
        return H, B


# ------------------------------------------------------------------------------------------------
# Core VTQ v2 Preconditioning + Fused Block16-LDLQ Matrix Quantizer
# ------------------------------------------------------------------------------------------------

@torch.no_grad()
def compute_suh_vtq2(
    W_f32: torch.Tensor,
    H_f32: torch.Tensor,
    seed: int = 1337,
    row_weights: Optional[torch.Tensor] = None,
    rot_block: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Computes `suh[K]` with `(p_w = 0.50, p_h = -0.25)` and applies canonical Kronecker rotation `R_K`."""
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

    W_rot = apply_kronecker_rot(W_f32 / suh.unsqueeze(0), rot_block=rot_block, transpose=False)
    H_tmp = apply_kronecker_rot(H_f32 * suh.unsqueeze(0), rot_block=rot_block, transpose=False)
    H_rot = apply_kronecker_rot((H_tmp.T * suh.unsqueeze(0)).contiguous(), rot_block=rot_block, transpose=False).T.contiguous()
    del H_tmp
    return W_rot, H_rot, suh


@torch.no_grad()
def _quantize_rows_vtq2_ldlq(
    W_rot_sub: torch.Tensor,
    U: torch.Tensor,
    U_blks: torch.Tensor,
    U_blks_inv: torch.Tensor,
    diag_w: torch.Tensor,
    spec_name: str,
    emit_pack: bool = False,
) -> dict:
    """Quantizes a subset of rows `W_rot_sub [R, K]` with a single VTQ v2 format using fused `ldlq_block256_cuda`."""
    dev = W_rot_sub.device
    R, Kin = W_rot_sub.shape
    nblocks = Kin // TILE
    cfg, tab, gains = _get_spec_gpu(spec_name, dev)
    K = int(cfg["K"])
    pp = int(cfg.get("pp", 0))
    m0 = int(cfg["m"])
    m_extra = tuple(int(x) for x in cfg.get("m_extra", ()))
    n_cand = int(cfg.get("n_cand", 0))
    cfg_scale = float(cfg.get("scale", 1.0))
    tb_win = int(os.environ.get("TCQ_TB_WIN", "32"))

    W_work = W_rot_sub.clone()
    W_hat_rot = torch.empty_like(W_rot_sub)
    e_out = torch.empty((R, nblocks), dtype=torch.int8, device=dev)
    G_out = torch.empty((R, Kin), dtype=torch.float32, device=dev) if emit_pack else None
    gi_out = torch.empty((R, Kin), dtype=torch.uint8, device=dev) if emit_pack else None
    idx_out = torch.empty((R, Kin), dtype=torch.int16, device=dev) if emit_pack else None

    rms_rb0 = W_rot_sub.view(R, nblocks, TILE).square().mean(dim=2).sqrt().clamp(min=1e-10) * cfg_scale
    d_row = torch.exp(torch.log(rms_rb0).mean(dim=1)).to(torch.float16).to(torch.float32).clamp(min=1e-8)

    for ibl in range(nblocks):
        c0, c1 = ibl * TILE, (ibl + 1) * TILE
        w_blk_start = W_work[:, c0:c1].clone()
        U_blk = U_blks[ibl]
        U_blk_inv = U_blks_inv[ibl]
        dw_blk = diag_w[c0:c1]

        cur_rms = w_blk_start.square().mean(dim=1).sqrt().clamp(min=1e-10) * cfg_scale
        e0 = torch.round(torch.log2(cur_rms / d_row) / 0.25).clamp(-16, 15)
        d_rb = (d_row * torch.exp2(0.25 * e0)).clamp(min=1e-8)

        max_r = 65536 if (pp > 0 and R > 65536) else R
        if max_r < R:
            G_pass = torch.empty_like(w_blk_start)
            for r_s in range(0, R, max_r):
                r_e = min(R, r_s + max_r)
                if emit_pack:
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
            if emit_pack:
                G_out[:, c0:c1] = G_pass
        elif emit_pack:
            G_pass, gi_b, idx_b = ldlq_block256_cuda(
                w_blk_start.clone(),
                d_rb,
                dw_blk,
                U_blk,
                U_blk_inv,
                tab,
                K,
                NU,
                m0,
                gains,
                n_cand=n_cand,
                tb_window=tb_win,
                m_extra=m_extra,
                pair_parity=pp,
                return_code=True,
            )
            G_out[:, c0:c1] = G_pass
            gi_out[:, c0:c1] = gi_b
            idx_out[:, c0:c1] = idx_b
        else:
            G_pass = ldlq_block256_cuda(
                w_blk_start.clone(),
                d_rb,
                dw_blk,
                U_blk,
                U_blk_inv,
                tab,
                K,
                NU,
                m0,
                gains,
                n_cand=n_cand,
                tb_window=tb_win,
                m_extra=m_extra,
                pair_parity=pp,
                return_code=False,
            )

        # Cholesky-weighted 5-bit exponent refinement (`e in [-16, 15]`)
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

    return {"W_hat_rot": W_hat_rot, "d_row": d_row.to(torch.float16), "e": e_out, "G": G_out, "gi": gi_out, "idx": idx_out}


@torch.no_grad()
def quantize_matrix_vtq2(
    W: torch.Tensor,
    H: torch.Tensor,
    spec_or_row_specs: str | List[Tuple[int, int, str]],
    seed: int = 1337,
    damp: float = 0.20,
    row_weights: Optional[torch.Tensor] = None,
    rot_block: int = -1,
    W_target: Optional[torch.Tensor] = None,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, float]:
    """Quantizes `W [M, K]` with input Hessian `H [K, K]` into VTQ v2 (uniform `spec_name` or DP `row_slices`)."""
    if not VTQ2_CONFIG:
        load_and_register_vtq2()

    dev = W.device
    W_src = (W_target if W_target is not None else W).float()
    H_f32 = H.float()
    M, Kin = W_src.shape
    nblocks = Kin // TILE

    W_rot, H_rot, suh = compute_suh_vtq2(W_src, H_f32, seed=seed, row_weights=row_weights, rot_block=rot_block)

    H_reg = H_rot.clone()
    del H_rot
    H_reg.diagonal().add_(damp * H_reg.diagonal().mean().clamp(min=1e-8))
    L_flip = torch.linalg.cholesky(H_reg.flip(0, 1))
    del H_reg
    U = torch.linalg.solve_triangular(L_flip, torch.eye(Kin, dtype=L_flip.dtype, device=dev), upper=False).flip(0, 1)
    del L_flip
    diag_w = 1.0 / (torch.diagonal(U) ** 2).clamp(min=1e-12)

    eye256 = torch.eye(TILE, dtype=torch.float32, device=dev).unsqueeze(0).expand(nblocks, -1, -1)
    U_blks = torch.stack([U[b * TILE : (b + 1) * TILE, b * TILE : (b + 1) * TILE] for b in range(nblocks)], dim=0)
    U_blks_inv = torch.linalg.solve_triangular(U_blks, eye256, upper=True)

    row_slices = [(0, M, spec_or_row_specs)] if isinstance(spec_or_row_specs, str) else list(spec_or_row_specs)
    by_spec: Dict[str, List[Tuple[int, int]]] = {}
    for r0, r1, sp in row_slices:
        by_spec.setdefault(sp, []).append((r0, r1))

    emit_pack = (PACK is not None) or (lv2.PACK is not None)
    W_hat_rot = torch.empty_like(W_rot)
    d_row_all = torch.empty((M,), dtype=torch.float16, device=dev)
    e_all = torch.empty((M, nblocks), dtype=torch.int8, device=dev)
    pm = (
        {
            "G": torch.empty((M, Kin), dtype=torch.float32, device=dev),
            "gi": torch.empty((M, Kin), dtype=torch.uint8, device=dev),
            "idx": torch.empty((M, Kin), dtype=torch.int16, device=dev),
        }
        if emit_pack
        else None
    )

    for sp, ranges in by_spec.items():
        rows_idx = torch.cat([torch.arange(r0, r1, device=dev) for r0, r1 in ranges], dim=0)
        sub_out = _quantize_rows_vtq2_ldlq(W_rot[rows_idx], U, U_blks, U_blks_inv, diag_w, sp, emit_pack=emit_pack)
        W_hat_rot[rows_idx] = sub_out["W_hat_rot"]
        d_row_all[rows_idx] = sub_out["d_row"]
        e_all[rows_idx] = sub_out["e"]
        if pm is not None:
            pm["G"][rows_idx] = sub_out["G"]
            pm["gi"][rows_idx] = sub_out["gi"]
            pm["idx"][rows_idx] = sub_out["idx"]

    del U, U_blks, U_blks_inv, W_rot

    U_base = apply_kronecker_rot(W_hat_rot, rot_block=rot_block, transpose=True)
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
    c_pack = torch.ones((M,), dtype=torch.float32, device=dev)
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
            c_pack = c_eff.squeeze(1)
    except RuntimeError:
        pass

    if emit_pack and pm is not None:
        suh_p = suh.to(torch.float16)
        W_hat_eff = assemble_vtq2(pm["G"], d_row_all, e_all, suh_p, rot_block=rot_block)
        HQ = W_hat_eff @ H_f32
        pack_dict = {
            "row_slices": row_slices,
            "idx": pm["idx"],
            "gi": pm["gi"],
            "G": pm["G"],
            "d_row": d_row_all,
            "e": e_all,
            "suh": suh_p,
        }
        if PACK is not None:
            PACK["matrix"] = pack_dict
        if lv2.PACK is not None:
            lv2.PACK["matrix"] = pack_dict

    diff = W_src - W_hat_eff
    rel_loss = float(((HW - HQ) * diff).sum().item()) / float((HW * W_src).sum().clamp(min=1e-12).item())
    qdata = {"d_row": d_row_all, "e": e_all, "suh": suh.to(torch.float16), "row_slices": row_slices}
    return qdata, W_hat_eff, rel_loss


# ------------------------------------------------------------------------------------------------
# Fisher Slice Profiler (128-row GQA KV Head Slices + Kronecker R_K + SwiGLU Permutation)
# ------------------------------------------------------------------------------------------------

@torch.no_grad()
def profile_matrix_vtq2(
    slices: list,
    l_idx: int,
    full: str,
    W: torch.Tensor,
    H: torch.Tensor,
    seed: int,
    step: int,
    s_row: Optional[torch.Tensor] = None,
    rot_block: int = -1,
):
    """Profiles per-slice Fisher-weighted distortion under VTQ v2 preconditioning (`suh` + Kronecker `R_K`)."""
    dev = W.device
    rw = s_row.to(dev).float() if s_row is not None else None
    W_rot, H_rot, _ = compute_suh_vtq2(W.float(), H.float(), seed=seed, row_weights=rw, rot_block=rot_block)
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
def build_vtq2_fisher_slice_profile(
    model: nn.Module,
    cal_tokens: torch.Tensor,
    candidate_specs: Optional[List[str]] = None,
    batch_size: int = 8,
    attn_slice_rows: int = 256,
    kv_slice_rows: int = 128,
    mlp_slice_rows: int = 512,
    fisher_out: Optional[Dict[Tuple[int, str], torch.Tensor]] = None,
    rot_block: int = -1,
) -> dict:
    """Builds the VTQ v2 Fisher slice profile with 128-row GQA KV head granularity and lossless SwiGLU `mlp_perms`."""
    if not VTQ2_CONFIG:
        load_and_register_vtq2()
    if candidate_specs is None:
        candidate_specs = list(VTQ2_MENU)

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
        for h in hooks:
            h.remove()

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
            profile_matrix_vtq2(
                slices_info,
                l_idx,
                subname,
                W_f32,
                hess_map[subname],
                seed=1000 + l_idx * 10 + idx_s,
                step=step,
                s_row=fisher_out_perm[(l_idx, subname)],
                rot_block=rot_block,
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


# ------------------------------------------------------------------------------------------------
# Full-Model Streaming GPTAQ Quantizer (1B / 3B / 8B In-VRAM Models)
# ------------------------------------------------------------------------------------------------

@torch.no_grad()
def quantize_llama_model_vtq2(
    model: nn.Module,
    cal_tokens: torch.Tensor,
    spec_or_assignment: str | Dict[Tuple[int, str], List[Tuple[int, int, str]]],
    mlp_perms: Optional[Dict[int, torch.Tensor]] = None,
    fisher_out: Optional[Dict[Tuple[int, str], torch.Tensor]] = None,
    batch_size: int = 4,
    damp: float = 0.20,
    gptaq_alpha: float = 0.0,
    rot_block: int = -1,
    polish_steps: int = 0,
    polish_lr: float = 4e-4,
) -> dict:
    """Layer-by-layer streaming VTQ v2 quantizer with optional asymmetric `GPTAQ` and optional scale polish."""
    if not VTQ2_CONFIG:
        load_and_register_vtq2()

    import src.streaming_quant as sq

    t_start = time.perf_counter()
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    model.eval()

    ref_hidden_cpu = None
    if polish_steps > 0:
        ref_hidden_cpu = lv2.precompute_ref_final_hidden_states(model, cal_tokens, n_seqs=64, seq_len=512, batch_size=4)
    if fisher_out is None:
        fisher_out = lv2.compute_exact_row_fisher_sensitivities(model, cal_tokens, n_Menu_seqs=8, seq_len=1024)

    n_seqs, seq_len = cal_tokens.shape
    n_layers = model.config.num_hidden_layers
    use_g = gptaq_alpha > 0.0

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

        def q_mod(mod: nn.Linear, H: torch.Tensor, B: Optional[torch.Tensor], full_name: str, seed_i: int, dm: float):
            nonlocal total_params, total_bits
            W = mod.weight.data.float()
            M_r, K_in = W.shape
            r_specs = [(0, M_r, spec_or_assignment)] if isinstance(spec_or_assignment, str) else spec_or_assignment[(l_idx, full_name)]
            W_tgt = (W + gptaq_alpha * (W @ B.T)) if (use_g and B is not None) else None
            sr = fisher_out[(l_idx, full_name)].to(device)
            _, w_hat, _ = quantize_matrix_vtq2(
                W, H, r_specs, seed=seed_i, damp=damp * dm, row_weights=sr, rot_block=rot_block, W_target=W_tgt
            )
            mod.weight.data.copy_(w_hat.to(dtype=dtype, device=device))
            for r0, r1, sp in r_specs:
                n_p = (r1 - r0) * K_in
                total_params += n_p
                total_bits += n_p * lv2.LIQK_V2_SPECS[sp].bpw

        # 1) q_proj, k_proj, v_proj
        acc = DualCovAccumulator(use_g)
        for cq, cr in zip(cal_q, cal_r):
            acc.add(lq.input_layernorm(cq), lr.input_layernorm(cr) if use_g else None)
        Hqkv, Bqkv = acc.get(gptaq_damp=damp)
        for i_s, nm_s in enumerate(("q_proj", "k_proj", "v_proj")):
            q_mod(getattr(lq.self_attn, nm_s), Hqkv, Bqkv, f"self_attn.{nm_s}", 1000 + l_idx * 10 + i_s, 1.0)
        del Hqkv, Bqkv, acc

        # 2) o_proj + attention residual update
        acc = DualCovAccumulator(use_g)
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
        Ho, Bo = acc.get(gptaq_damp=damp)
        q_mod(lq.self_attn.o_proj, Ho, Bo, "self_attn.o_proj", 1000 + l_idx * 10 + 3, 1.0)
        del Ho, Bo, acc

        for i in range(len(cal_q)):
            cal_q[i] = cal_q[i] + sq._attn(lq, cal_q[i], pos_cal)
        if not use_g:
            ref_mid = cal_q

        # 3) gate_proj, up_proj
        acc = DualCovAccumulator(use_g)
        for cq, crm in zip(cal_q, ref_mid):
            acc.add(lq.post_attention_layernorm(cq), lr.post_attention_layernorm(crm) if use_g else None)
        Hup, Bup = acc.get(gptaq_damp=damp)
        for i_s, nm_s in enumerate(("gate_proj", "up_proj")):
            q_mod(getattr(lq.mlp, nm_s), Hup, Bup, f"mlp.{nm_s}", 1000 + l_idx * 10 + 4 + i_s, 1.0)
        del Hup, Bup, acc

        # 4) down_proj + MLP residual update
        acc = DualCovAccumulator(use_g)
        hq = lq.mlp.down_proj.register_forward_pre_hook(lambda m, a: bq.__setitem__(0, a[0]))
        hr = lr.mlp.down_proj.register_forward_pre_hook(lambda m, a: br.__setitem__(0, a[0])) if use_g else None
        for cq, crm in zip(cal_q, ref_mid):
            lq.mlp(lq.post_attention_layernorm(cq))
            if use_g:
                lr.mlp(lr.post_attention_layernorm(crm))
            acc.add(bq[0], br[0])
            bq[0] = br[0] = None
        hq.remove()
        if hr is not None:
            hr.remove()
        Hdn, Bdn = acc.get(gptaq_damp=damp * 1.5)
        q_mod(lq.mlp.down_proj, Hdn, Bdn, "mlp.down_proj", 1000 + l_idx * 10 + 6, 1.5)
        del Hdn, Bdn, acc

        for i in range(len(cal_q)):
            cal_q[i] = cal_q[i] + lq.mlp(lq.post_attention_layernorm(cal_q[i]))
        if use_g:
            for i in range(len(cal_r)):
                cal_r[i] = ref_mid[i] + lr.mlp(lr.post_attention_layernorm(ref_mid[i]))
            del lr
        del ref_mid

    del cal_q, cal_r
    torch.cuda.empty_cache()
    quant_sec = time.perf_counter() - t_start

    polish_sec = 0.0
    if polish_steps > 0 and ref_hidden_cpu is not None:
        with torch.enable_grad():
            polish_sec = lv2.polish_model_scales_kld(
                model, cal_tokens, ref_hidden_cpu, n_steps=polish_steps, lr=polish_lr, batch_size=2
            )

    return {
        "linear_params": total_params,
        "linear_bytes": int(round(total_bits / 8.0)),
        "linear_bpw": total_bits / max(total_params, 1),
        "quant_sec": quant_sec,
        "polish_sec": polish_sec,
    }


# ------------------------------------------------------------------------------------------------
# `VtqRing` Short-Row / Embedding / Head / PLE Table Quantizer (`L = 256` or `L = 160`)
# ------------------------------------------------------------------------------------------------

@torch.no_grad()
def quantize_rings_vtq2(
    W: torch.Tensor,
    spec_name: str,
    ring_len: int = TILE,
    rot: bool = True,
    chunk: int = 1 << 16,
    scales: tuple[float, ...] = (0.92, 1.00, 1.10),
) -> Tuple[torch.Tensor, float]:
    """Quantizes 2D table `W [rows, n * ring_len]` into tail-biting `VtqRing` rows (`L = 256` or `160`)."""
    from src.vtq_v2_format import load_vtq2_formats

    if not VTQ2_CONFIG:
        load_and_register_vtq2()
    fmt = load_vtq2_formats(menu=[spec_name])[spec_name]
    cfg, lut, gains = _get_spec_gpu(spec_name, torch.device("cuda"))
    signs = make_rademacher_signs(ring_len, seed=4242, device=torch.device("cuda"))
    p2, q = (ring_len & -ring_len), ring_len // (ring_len & -ring_len)

    def _rot_ring(x: torch.Tensor, inv: bool) -> torch.Tensor:
        if not rot:
            return x
        if not inv:
            return apply_kronecker_rot(x * signs.unsqueeze(0), rot_block=-1, transpose=False)
        return apply_kronecker_rot(x, rot_block=-1, transpose=True) * signs.unsqueeze(0)

    rings = W.view(-1, ring_len)
    out = torch.empty_like(rings)
    sc_base = float(cfg.get("scale", 1.0))
    n_cand = int(cfg.get("n_cand", 0))
    eff_chunk = min(chunk, 8192) if fmt.frac > 0 else chunk
    for i in range(0, rings.shape[0], eff_chunk):
        x = _rot_ring(rings[i : i + eff_chunk].cuda().float(), inv=False)
        rms = x.square().mean(dim=1, keepdim=True).sqrt().clamp(min=1e-8) * sc_base
        best, best_err = None, None
        for sc in scales:
            d = (rms * sc).half().float().clamp(min=1e-8)
            rec = (
                viterbi_ub_cuda(
                    x / d,
                    lut,
                    fmt.K,
                    NU,
                    fmt.masks[0],
                    gains,
                    sub=fmt.sub,
                    tail_biting=True,
                    m_extra=fmt.masks[1:],
                    pair_parity=fmt.frac,
                    n_cand=n_cand,
                )
                * d
            )
            err = (rec - x).square().sum(dim=1)
            if best is None:
                best, best_err = rec, err
            else:
                imp = err < best_err
                best = torch.where(imp.unsqueeze(1), rec, best)
                best_err = torch.where(imp, err, best_err)
        out[i : i + eff_chunk] = _rot_ring(best, inv=True).to(out.dtype).cpu()

    bpv = fmt.tile_bytes(ring_len) * 8.0 / ring_len + 16.0 / ring_len
    return out.view_as(W), bpv


# ------------------------------------------------------------------------------------------------
# Transparent Hooks for `streaming_quant.py` (8B / 30B MoE) & `hybrid_quant.py` (Qwen3.5 / Gemma 4)
# ------------------------------------------------------------------------------------------------

_HOOKS_INSTALLED = False


def install_vtq2_hooks():
    """Patches `streaming_quant._quantize_linear_weight` and `streaming_quant.profile_matrix` (and `hybrid_quant`
    references) so any `VTQ*` format or `VTQ2` DP assignment automatically runs `quantize_matrix_vtq2`."""
    global _HOOKS_INSTALLED
    load_and_register_vtq2()
    if _HOOKS_INSTALLED:
        return

    import src.streaming_quant as sqm

    orig_quant = sqm._quantize_linear_weight
    orig_prof = sqm.profile_matrix

    def _is_vtq2(spec) -> bool:
        if isinstance(spec, str):
            return spec in VTQ2_CONFIG
        if isinstance(spec, list) and spec:
            return spec[0][2] in VTQ2_CONFIG
        return False

    def _patched_quant(W: torch.Tensor, H: torch.Tensor, spec, seed: int, damp: float, full: str | None = None) -> torch.Tensor:
        if _is_vtq2(spec):
            rw = sqm._row_w(full, W.device) if full else None
            # streaming_quant / hybrid_quant pass base damp=0.02 (0.03 on down_proj); scale by 10x -> 0.20 (0.30) as in TCQ
            vtq_damp = damp * 10.0 if damp < 0.08 else damp
            _, w_hat, _ = quantize_matrix_vtq2(W, H, spec, seed=seed, damp=vtq_damp, row_weights=rw, rot_block=-1)
            return w_hat.to(W.dtype)
        return orig_quant(W, H, spec, seed, damp, full)

    def _patched_prof(slices: list, l_idx: int, full: str, W: torch.Tensor, H: torch.Tensor, seed: int, step: int):
        if os.environ.get("VTQ2_PROFILE", "0") == "1":
            s_row = sqm.FISHER[(l_idx, full)].to(W.device).float() if sqm.FISHER is not None else None
            return profile_matrix_vtq2(slices, l_idx, full, W, H, seed, step, s_row=s_row, rot_block=-1)
        return orig_prof(slices, l_idx, full, W, H, seed, step)

    sqm._quantize_linear_weight = _patched_quant
    sqm.profile_matrix = _patched_prof
    orig_dp = lv2.solve_fisher_slice_dp

    def _patched_dp(profile: dict, target_bpw: float, candidate_specs: Optional[List[str]] = None, sens_power: float = 1.0):
        if candidate_specs is not None and len(candidate_specs) == len(VTQ2_MENU) and set(candidate_specs) == set(VTQ2_MENU):
            candidate_specs = vtq2_candidates_for_target(target_bpw)
        return orig_dp(profile, target_bpw=target_bpw, candidate_specs=candidate_specs, sens_power=sens_power)

    lv2.solve_fisher_slice_dp = _patched_dp
    try:
        import src.hybrid_quant as hqm

        hqm._quantize_linear_weight = _patched_quant
        hqm.profile_matrix = _patched_prof
    except Exception:
        pass
    _HOOKS_INSTALLED = True
