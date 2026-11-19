"""
Reference implementation of the VTQ format and `.vtq` binary container (`src/vtq_format.py`):
  1. `16 rows x 256 cols = 4,096 weights` Natural-Column Super-Tile (`SUPER_ROWS = 16`, `TILE = 256`):
     - Shared byte-for-byte across CPU AVX2 (`src/cpu/vtq_avx2_bench.c`), GPU Joint-1024 (`src/cuda/vtq_matvec_bench.cu`),
       and the `.vtq` container (`VtqWriter` / `VtqFile`), with zero column permutations (`0..255` natural order)
       and native little-endian `uint16_t` `c[16][16][2]` (`c[g][r][0] = cols 8..15`, `c[g][r][1] = cols 0..7`).
  2. 1024-state (`nu = 10`) Tail-Biting Shift-Register Trellis (`9` history bits in `*(uint16_t*)c[15][r] & 0x1FF`)
     with optimal Ungerboeck GF(2) subset label maps:
       `nl == 16 -> (s & 1) | (p10(s & 0x2CE) << 1) | (p10(s & 0x050) << 2) | (p10(s & 0x3B6) << 3)` (`[359, 40, 475]`),
       `nl == 8  -> (s & 1) | (p10(s & 0x2CE) << 1) | (p10(s & 0x39C) << 2)` (`[359, 462]`),
       `nl == 4  -> (s & 1) | (p10(s & 0x2CE) << 1)` (`[359]`).
  3. Exact `uint8` / `int8` (`qmax = 127`) dyadic codebooks (`tab * 1/64`) + `uint8 gain_u8[2^h]` (`* 1/64`, `sub = 16`)
     + 8-bit linear superblock scale `uint8 u_sb[16]` (`* 1/128`) + per-row `fp16 d_row[M]` + per-col `fp16 suh[K]`.
  4. Canonical Kronecker transform `R_K = H_{2^a} (x) DCT-II_q` (`rot_block = -1`) and Block-256 FWHT (`rot_block = 256`).
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import struct
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from src.vtq_v2_format import (
    CW8_TO_IDX,
    DTYPES,
    HAM_CW8,
    TORCH_TO_VTQ,
    _align,
    apply_kronecker_rot as _apply_inner_rot,
    get_dct_ortho,
    get_hadamard_ortho,
    split_pow2_odd,
)

TILE = 256
SUPER_ROWS = 16
MAGIC = b"VTQ\0"
VERSION = 1

TAB_UNIT = np.float32(1.0 / 64.0)
GAIN_UNIT = np.float32(1.0 / 64.0)
U_SB_UNIT = np.float32(1.0 / 128.0)

# Format metadata: fmt_code -> (K_base, frac, nl, h, st_bytes for 16x256 = 4,096 weights)
# `frac`: 0=plain, 1=Gosset[8,4,4] (-0.50 bpw), 2=D4[4,3,2] (-0.25 bpw), 3=D4_eq[4,1,4] (-0.75 bpw)
FMT_META: dict[int, tuple[int, int, int, int, int]] = {
    15: (2, 1, 16, 1,  816),  # 1.59375 bpw (vtq_st16_k15_t, s8_32)
    17: (2, 2, 16, 1,  944),  # 1.84375 bpw (s8_32)
    20: (2, 0, 16, 1, 1072),  # 2.09375 bpw (vtq_st16_k20_t, u8 16B)
    22: (3, 3, 16, 1, 1200),  # 2.34375 bpw (u8 32B)
    25: (3, 1, 16, 1, 1328),  # 2.59375 bpw (vtq_st16_k25_t, h=1, u8 32B)
    24: (3, 1, 16, 2, 1360),  # 2.65625 bpw (VTQ_250_H2, h=2, u8 32B)
    26: (3, 2, 16, 1, 1456),  # 2.84375 bpw (VTQ_275, h=1, u8 32B)
    27: (3, 2, 16, 2, 1488),  # 2.90625 bpw (VTQ_275_H2, h=2, u8 32B)
    30: (3, 0,  8, 2, 1616),  # 3.15625 bpw (vtq_st16_k30_t, nl=8 fast, u8 16B)
    31: (3, 0, 16, 2, 1616),  # 3.15625 bpw (vtq_st16_k30_t, nl=16 HQ, u8 32B)
    32: (4, 3, 16, 2, 1744),  # 3.40625 bpw (nl=16, at2_16_16 32B)
    35: (4, 1, 16, 2, 1872),  # 3.65625 bpw (vtq_st16_k35_t, nl=16, at2_16_16 32B)
    37: (4, 2, 16, 2, 2000),  # 3.90625 bpw (nl=16, at2_16_16 32B)
    40: (4, 0, 16, 3, 2160),  # 4.21875 bpw (vtq_st16_k40_t, nl=16, at2_16_16 32B)
    42: (5, 3, 16, 2, 2256),  # 4.40625 bpw (nl=16, at2z_32_32 64B)
    45: (5, 1, 16, 2, 2384),  # 4.65625 bpw (nl=16, at2z_32_32 64B)
    47: (5, 2, 16, 2, 2512),  # 4.90625 bpw (nl=16, at2z_32_32 64B)
    50: (5, 0,  4, 3, 2672),  # 5.21875 bpw (vtq_st16_k50_t, nl=4 fast, u8 32B)
    51: (5, 0, 16, 3, 2672),  # 5.21875 bpw (vtq_st16_k50_t, nl=16 HQ, at2z_32_32 64B)
    55: (6, 1, 16, 3, 2928),  # 5.71875 bpw (nl=16, at2z_32_32 64B)
    57: (6, 2, 16, 3, 3056),  # 5.96875 bpw (nl=16, at2z_32_32 64B)
    60: (6, 0, 16, 3, 3184),  # 6.21875 bpw (vtq_st16_k60_t, nl=16, at2z_32_32 64B)
}

_OUTER_KRON_CACHE: dict[tuple[int, torch.device], tuple[int, int, int, torch.Tensor, torch.Tensor, torch.Tensor]] = {}

_GALOIS_MASKS: dict[int, tuple[int, ...]] = {
    16: (0x001, 0x2CE, 0x050, 0x3B6),
    8:  (0x001, 0x2CE, 0x39C),
    4:  (0x001, 0x2CE),
}
_GALOIS_LUT: dict[int, np.ndarray] = {
    nl: np.array(
        [sum(((bin(s & m).count("1") & 1) << i) for i, m in enumerate(ms)) for s in range(1024)],
        dtype=np.uint8,
    )
    for nl, ms in _GALOIS_MASKS.items()
}


def galois_lb_vtq(st: np.ndarray, nl: int) -> np.ndarray:
    """Optimal 1024-state (`nu = 10`, 9 history bits) Ungerboeck GF(2) subset label map matching C AVX2 & CUDA Joint-1024."""
    return _GALOIS_LUT[nl][np.asarray(st, dtype=np.uint16) & 0x3FF]


def _get_outer_kron_factors(K: int, device: torch.device):
    key = (K, device)
    if key not in _OUTER_KRON_CACHE:
        p2, q = split_pow2_odd(K)
        a = p2.bit_length() - 1
        b1 = 1 << (a // 2)
        b2 = 1 << (a - a // 2)
        H1 = get_hadamard_ortho(b1, device)
        H2 = get_hadamard_ortho(b2, device)
        Dq = get_dct_ortho(q, device)
        _OUTER_KRON_CACHE[key] = (q, b1, b2, Dq, H1, H2)
    return _OUTER_KRON_CACHE[key]


def apply_vtq_rot(X: torch.Tensor, rot_block: int = -1, transpose: bool = False) -> torch.Tensor:
    """Applies `R_K^{outer} = DCT-II_q (x) H_{2^a}` (`rot_block = -1`), `Inner-DCT` (`-2`), or Block-FWHT (`> 0`)."""
    K = X.shape[-1]
    if rot_block > 0:
        return _apply_inner_rot(X, rot_block=rot_block, transpose=transpose)
    if rot_block in (-1, -2):
        return _apply_inner_rot(X, rot_block=-1, transpose=transpose)
    p2, q = split_pow2_odd(K)
    if q == 1:
        return _apply_inner_rot(X, rot_block=-1, transpose=transpose)
    assert p2 >= 32, f"Expected K={K} divisible by 32, got p2={p2}"
    orig_shape = X.shape
    N = X.numel() // K
    prev_tf32 = torch.backends.cuda.matmul.allow_tf32
    if prev_tf32:
        torch.backends.cuda.matmul.allow_tf32 = False
    try:
        q_f, b1, b2, Dq, H1, H2 = _get_outer_kron_factors(K, X.device)
        if not transpose:
            Y = torch.matmul(H1, X.reshape(N * q_f, b1, b2) @ H2).reshape(N, q_f, p2)
            return torch.matmul(Dq, Y).reshape(orig_shape)
        else:
            Z = torch.matmul(Dq.T, X.reshape(N, q_f, p2)).reshape(N * q_f, b1, b2)
            return torch.matmul(H1, Z @ H2).reshape(orig_shape)
    finally:
        if prev_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True


def _at2_indices_vtq(K_base: int, nl: int, tp: str) -> tuple[np.ndarray, np.ndarray]:
    """Maps `(u_mag, lb)` -> `(idxA, idxB)` of shape `(U // 2, nl)` (`idx_canon = (u_mag * nl) | lb`)."""
    u_cnt = 1 << max(K_base - 2, 0)
    ul = np.arange(u_cnt, dtype=np.int64)[:, None]
    lb = np.arange(nl, dtype=np.int64)[None, :]
    if tp == "at2_16_16":
        assert K_base == 4 and nl == 16
        return np.broadcast_to(ul | ((lb & 3) << 2), (u_cnt, nl)), np.broadcast_to(ul | ((lb >> 2) << 2), (u_cnt, nl))
    if tp == "at2z_32_32":
        assert nl == 16 and K_base in (5, 6)
        if K_base == 5:
            return np.broadcast_to(ul | ((lb & 3) << 3), (u_cnt, nl)), np.broadcast_to(ul | ((lb >> 2) << 3), (u_cnt, nl))
        return np.broadcast_to(ul | ((lb & 1) << 4), (u_cnt, nl)), np.broadcast_to((ul >> 2) | ((lb >> 1) << 2), (u_cnt, nl))
    raise ValueError(f"Unsupported additive codebook type={tp!r} for K={K_base}, nl={nl}")


def hw_to_canon_u8_vtq(fmt_code: int, hw: dict) -> np.ndarray:
    """Expands `hw_codebook` (`u8`, `at2_16_16`, or `at2z_32_32`) to canonical `uint8 [nl * (U // 2)]` (`(u_mag * nl) | lb`)."""
    K_base, _, nl, _, _ = FMT_META[fmt_code]
    U = 1 << (K_base - 1)
    m_mag = nl * max(U // 2, 1)
    qmax = int(hw.get("qmax", 127))
    tp = hw["type"]
    if tp == "s8_32":
        return np.asarray(hw["tab_s8"], dtype=np.int8).view(np.uint8)
    if tp == "u8":
        tab_u8 = np.asarray(hw["tab_u8"], dtype=np.uint8)
        assert tab_u8.shape == (m_mag,), f"Expected tab_u8 shape ({m_mag},), got {tab_u8.shape}"
        return tab_u8
    if tp in ("at2_16_16", "at2z_32_32"):
        idxA, idxB = _at2_indices_vtq(K_base, nl, tp)
        ta = np.asarray(hw["TA_u8"], dtype=np.int32)
        tb = np.asarray(hw["TB_u8"], dtype=np.int32)
        return np.clip(ta[idxA] + tb[idxB], 0, qmax).astype(np.uint8).reshape(m_mag)
    raise ValueError(f"Unknown VTQ hw_codebook type: {tp!r}")


def reconstruct_lut_from_hw_vtq(fmt_code: int, hw: dict) -> np.ndarray:
    """Deterministically reconstructs exact dyadic `float32 [nl * U]` table from `s8_32`, `u8`, `at2_16_16`, or `at2z_32_32`."""
    K_base, _, nl, _, _ = FMT_META[fmt_code]
    U = 1 << (K_base - 1)
    tab_div = np.float32(1.0 / float(hw.get("tab_div", 64)))
    out = np.empty((nl, U), dtype=np.float32)
    if fmt_code in (15, 17):
        tab_s8 = np.asarray(hw["tab_s8"], dtype=np.int8)
        assert tab_s8.shape == (32,)
        for lb in range(16):
            for u in range(2):
                out[lb, u] = np.float32(int(tab_s8[(u << 4) | lb])) * tab_div
        return out.reshape(-1)

    m_mag = nl * (U // 2)
    tab_u8 = hw_to_canon_u8_vtq(fmt_code, hw)
    assert tab_u8.shape == (m_mag,), f"Expected tab_u8 shape ({m_mag},), got {tab_u8.shape}"
    u_mag_mask = (1 << (K_base - 2)) - 1
    for lb in range(nl):
        for u in range(U):
            neg = (u >> (K_base - 2)) & 1
            u_mag = u & u_mag_mask
            idx = ((u_mag * nl) | lb) ^ ((m_mag - 1) if neg else 0)
            val = int(tab_u8[idx])
            out[lb, u] = np.float32(-val if neg else val) * tab_div
    return out.reshape(-1)


@dataclass(frozen=True)
class VtqFormat:
    name: str
    fmt_code: int         # 15..60
    K: int                # base integer bits (2..6)
    gosset: int           # 0=plain, 1=Gosset[8,4,4], 2=D4[4,3,2], 3=D4_eq[4,1,4]
    nl: int               # number of Galois subset labels (4, 8, or 16)
    h: int                # gain index bits per 16-weight subblock (1, 2, or 3)
    sub: int              # 16 weights per gain index
    hw_codebook: dict     # {"type": "s8_32" | "u8", "qmax": 127, "tab_div": 64, "tab_s8"|"tab_u8": [...]}
    gain_u8: np.ndarray   # uint8 [2^h] integer multipliers (`gains = gain_u8 / 64.0f`)
    lut: np.ndarray       # float32 [nl * U] exact dyadic codebook
    gain_tab: np.ndarray  # float32 [2^h] exact dyadic gains
    scale: float = 1.0    # optimal RMS pre-scale for encoder initialization

    @property
    def U(self) -> int:
        return 1 << (self.K - 1)

    @property
    def st_bytes(self) -> int:
        return FMT_META[self.fmt_code][4]

    @property
    def bits_256(self) -> float:
        return self.st_bytes * 8.0 / 16.0  # super-tile bits per 256-weight block (`c + u + hdr + u_sb`)

    def bpw(self, m_rows: int = 4096, k_cols: int = 4096) -> float:
        """Total bits/weight including super-tile (`c + u + hdr + u_sb`), `fp16 d_row[M]`, and `fp16 suh[K]`."""
        return self.st_bytes * 8.0 / 4096.0 + 16.0 / k_cols + 16.0 / m_rows


def load_vtq_formats(tables_path: Path | None = None, menu: list[str] | None = None) -> dict[str, VtqFormat]:
    if tables_path is None:
        tables_path = Path(__file__).resolve().parent.parent / "results" / "vtq_tables.json"
    raw = json.loads(Path(tables_path).read_text(encoding="utf-8"))
    names = menu if menu is not None else list(raw.keys())
    out: dict[str, VtqFormat] = {}
    for nm in names:
        out[nm] = format_from_json_vtq(dict(raw[nm], name=nm))
    return out


def format_json_vtq(f: VtqFormat) -> dict:
    return {
        "name": f.name,
        "fmt_code": f.fmt_code,
        "hw_codebook": f.hw_codebook,
        "gain_u8": [int(x) for x in f.gain_u8.tolist()],
        "scale": float(f.scale),
    }


def format_from_json_vtq(d: dict) -> VtqFormat:
    fc = int(d["fmt_code"])
    K, gosset, nl, h, _ = FMT_META[fc]
    hw = dict(d["hw_codebook"])
    g_raw = d.get("gain_u8", d.get("gain_i"))
    gain_u8 = np.clip(np.asarray(g_raw, dtype=np.int32), 1, 255).astype(np.uint8)
    return VtqFormat(
        name=d["name"],
        fmt_code=fc,
        K=K,
        gosset=gosset,
        nl=nl,
        h=h,
        sub=16,
        hw_codebook=hw,
        gain_u8=gain_u8,
        lut=reconstruct_lut_from_hw_vtq(fc, hw),
        gain_tab=gain_u8.astype(np.float32) * GAIN_UNIT,
        scale=float(d.get("scale", 1.0)),
    )


# ------------------------------------------------------------------------------------------------
# Vectorized `16x256` Super-Tile Packing & Reference Decoding (`vtq_st16_k15_t .. vtq_st16_k60_t`)
# ------------------------------------------------------------------------------------------------

def _pack_eo_plane(bits: np.ndarray) -> np.ndarray:
    """Packs `[nbg, nsb, 16_g, 16_r, 16_j]` `{0,1}` into `[nbg, nsb, 512]` (`plane[g][16*odd + j]` bit `p` = `r=2*p+odd`)."""
    nbg, nsb, _, _, _ = bits.shape
    eo = bits.reshape(nbg, nsb, 16, 8, 2, 16)
    sh = np.arange(8, dtype=np.uint8)[None, None, None, :, None, None]
    packed = np.sum((eo.astype(np.uint8) & 1) << sh, axis=3, dtype=np.uint8)  # [nbg, nsb, 16_g, 2_odd, 16_j]
    return packed.reshape(nbg, nsb, 512)


def _unpack_eo_plane(buf512: np.ndarray, nbg: int, nsb: int) -> np.ndarray:
    """Unpacks `[nbg, nsb, 512]` into `{0,1}` `int32 [nbg, nsb, 16_g, 16_r, 16_j]`."""
    eo = buf512.reshape(nbg, nsb, 16, 2, 16)
    sh = np.arange(8, dtype=np.uint8)[None, None, None, :, None, None]
    bits = ((eo[:, :, :, None, :, :] >> sh) & 1).astype(np.int32)  # [nbg, nsb, 16_g, 8_p, 2_odd, 16_j]
    return bits.reshape(nbg, nsb, 16, 16, 16)


def _pack_u2_plane(u2_grj: np.ndarray) -> np.ndarray:
    """Packs `[nbg, nsb, 16_g, 16_r, 16_j]` `2`-bit values into `u2[16][2][32]` (`[nbg, nsb, 1024]`)."""
    nbg, nsb, _, _, _ = u2_grj.shape
    v = (u2_grj.astype(np.uint8) & 3).reshape(nbg, nsb, 16, 2, 4, 2, 16)
    sh = (2 * np.arange(4, dtype=np.uint8))[None, None, None, None, :, None, None]
    packed = np.sum(v << sh, axis=4, dtype=np.uint8)  # [nbg, nsb, 16_g, 2_hp, 2_odd, 16_j]
    return packed.reshape(nbg, nsb, 1024)


def _unpack_u2_plane(buf1024: np.ndarray, nbg: int, nsb: int) -> np.ndarray:
    """Unpacks `[nbg, nsb, 1024]` `u2[16][2][32]` into `int32 [nbg, nsb, 16_g, 16_r, 16_j]`."""
    p = buf1024.reshape(nbg, nsb, 16, 2, 2, 16)
    sh = (2 * np.arange(4, dtype=np.uint8))[None, None, None, None, :, None, None]
    v = ((p[:, :, :, :, None, :, :] >> sh) & 3).astype(np.int32)
    return v.reshape(nbg, nsb, 16, 16, 16)


def _pack_u4_plane(u4_grj: np.ndarray) -> np.ndarray:
    """Packs `[nbg, nsb, 16_g, 16_r, 16_j]` `4`-bit values into `u4[16][4][32]` (`[nbg, nsb, 2048]`)."""
    nbg, nsb, _, _, _ = u4_grj.shape
    v = (u4_grj.astype(np.uint8) & 15).reshape(nbg, nsb, 16, 4, 2, 2, 16)
    sh = (4 * np.arange(2, dtype=np.uint8))[None, None, None, None, :, None, None]
    packed = np.sum(v << sh, axis=4, dtype=np.uint8)  # [nbg, nsb, 16_g, 4_qp, 2_odd, 16_j]
    return packed.reshape(nbg, nsb, 2048)


def _unpack_u4_plane(buf2048: np.ndarray, nbg: int, nsb: int) -> np.ndarray:
    """Unpacks `[nbg, nsb, 2048]` `u4[16][4][32]` into `int32 [nbg, nsb, 16_g, 16_r, 16_j]`."""
    p = buf2048.reshape(nbg, nsb, 16, 4, 2, 16)
    sh = (4 * np.arange(2, dtype=np.uint8))[None, None, None, None, :, None, None]
    v = ((p[:, :, :, :, None, :, :] >> sh) & 15).astype(np.int32)
    return v.reshape(nbg, nsb, 16, 16, 16)


def _pack_ham16(u0_grj: np.ndarray) -> np.ndarray:
    """Packs `[nbg, nsb, 16_g, 16_r, 16_j]` Gosset `[8,4,4]` bits into `ham[16][16]` (`[nbg, nsb, 256]`)."""
    nbg, nsb, _, _, _ = u0_grj.shape
    halves = (u0_grj.astype(np.uint8) & 1).reshape(nbg, nsb, 16, 16, 2, 8)
    sh = np.arange(7, -1, -1, dtype=np.uint8)[None, None, None, None, None, :]
    cw = np.sum(halves << sh, axis=-1, dtype=np.uint8)
    nibs = CW8_TO_IDX[cw]
    if np.any(nibs < 0):
        raise ValueError("Invalid Gosset [8, 4, 4] codeword encountered in u0_grj!")
    nibs_u8 = nibs.astype(np.uint8)
    return (nibs_u8[..., 0] | (nibs_u8[..., 1] << 4)).reshape(nbg, nsb, 256)


def _unpack_ham16(buf256: np.ndarray, nbg: int, nsb: int) -> np.ndarray:
    """Unpacks `[nbg, nsb, 256]` `ham[16][16]` into `{0,1}` `int32 [nbg, nsb, 16_g, 16_r, 16_j]`."""
    ham = buf256.reshape(nbg, nsb, 16, 16)
    cw = np.stack([HAM_CW8[ham & 0x0F], HAM_CW8[(ham >> 4) & 0x0F]], axis=-1)  # [nbg, nsb, 16, 16, 2]
    sh = np.arange(7, -1, -1, dtype=np.uint8)[None, None, None, None, None, :]
    return ((cw[..., None] >> sh) & 1).reshape(nbg, nsb, 16, 16, 16).astype(np.int32)


def _pack_d4_16(u0_grj: np.ndarray) -> np.ndarray:
    """Packs `[nbg, nsb, 16_g, 16_r, 16_j]` D4 `[4, 3, 2]` bits into `[nbg, nsb, 384]`."""
    nbg, nsb, _, _, _ = u0_grj.shape
    q = (u0_grj.astype(np.uint8) & 1).reshape(nbg, nsb, 16, 16, 4, 4)
    if np.any(((q[..., 0] ^ q[..., 1] ^ q[..., 2] ^ q[..., 3]) & 1) != 0):
        raise ValueError("Invalid D4 [4, 3, 2] parity encountered in u0_grj!")
    # Pack each of `t = 0, 1, 2` (`[nbg, nsb, 16_g, 16_r, 4_q4]`) -> `2` rows per byte (`128 B` per `t`, `384 B` total)
    sh4 = np.arange(8, dtype=np.uint8)[None, None, None, None, :]
    planes = [
        np.sum(q[..., t].reshape(nbg, nsb, 16, 8, 8) << sh4, axis=-1, dtype=np.uint8).reshape(nbg, nsb, 128)
        for t in range(3)
    ]
    return np.concatenate(planes, axis=-1)


def _unpack_d4_16(buf384: np.ndarray, nbg: int, nsb: int) -> np.ndarray:
    sh4 = np.arange(8, dtype=np.uint8)[None, None, None, None, :]
    p = [
        ((buf384[:, :, 128 * t : 128 * (t + 1)].reshape(nbg, nsb, 16, 8)[..., None] >> sh4) & 1)
        .reshape(nbg, nsb, 16, 16, 4)
        .astype(np.int32)
        for t in range(3)
    ]
    p3 = p[0] ^ p[1] ^ p[2]
    return np.stack([p[0], p[1], p[2], p3], axis=-1).reshape(nbg, nsb, 16, 16, 16)


def _pack_d4eq_16(u0_grj: np.ndarray) -> np.ndarray:
    """Packs `[nbg, nsb, 16_g, 16_r, 16_j]` D4_eq `[4, 1, 4]` bits into `[nbg, nsb, 128]`."""
    nbg, nsb, _, _, _ = u0_grj.shape
    q = (u0_grj.astype(np.uint8) & 1).reshape(nbg, nsb, 16, 16, 4, 4)
    if np.any(q[..., 1:] != q[..., :1]):
        raise ValueError("Invalid D4_eq [4, 1, 4] codeword encountered in u0_grj!")
    sh4 = np.arange(8, dtype=np.uint8)[None, None, None, None, :]
    return np.sum(q[..., 0].reshape(nbg, nsb, 16, 8, 8) << sh4, axis=-1, dtype=np.uint8).reshape(nbg, nsb, 128)


def _unpack_d4eq_16(buf128: np.ndarray, nbg: int, nsb: int) -> np.ndarray:
    sh4 = np.arange(8, dtype=np.uint8)[None, None, None, None, :]
    p0 = ((buf128.reshape(nbg, nsb, 16, 8)[..., None] >> sh4) & 1).reshape(nbg, nsb, 16, 16, 4).astype(np.int32)
    return np.repeat(p0[..., None], 4, axis=-1).reshape(nbg, nsb, 16, 16, 16)


def _pack_lsb16(u0_grj: np.ndarray, frac: int) -> np.ndarray:
    if frac == 0:
        return _pack_eo_plane(u0_grj)
    if frac == 1:
        return _pack_ham16(u0_grj)
    if frac == 2:
        return _pack_d4_16(u0_grj)
    if frac == 3:
        return _pack_d4eq_16(u0_grj)
    raise ValueError(f"Unsupported frac={frac}")


def _unpack_lsb16(buf: np.ndarray, off: int, frac: int, nbg: int, nsb: int) -> tuple[np.ndarray, int]:
    if frac == 0:
        return _unpack_eo_plane(buf[:, :, off : off + 512], nbg, nsb), off + 512
    if frac == 1:
        return _unpack_ham16(buf[:, :, off : off + 256], nbg, nsb), off + 256
    if frac == 2:
        return _unpack_d4_16(buf[:, :, off : off + 384], nbg, nsb), off + 384
    if frac == 3:
        return _unpack_d4eq_16(buf[:, :, off : off + 128], nbg, nsb), off + 128
    raise ValueError(f"Unsupported frac={frac}")


def pack_supertiles_vtq(
    c_all: np.ndarray,    # uint8 [M, nsb, 16, 2] (MSB-first coded byte per g=0..15, [0]=cols 0..7, [1]=cols 8..15)
    u_all: np.ndarray,    # uint8 [M, nsb, 256] (natural column order 0..255)
    g_sub: np.ndarray,    # uint8 [M, nsb, 16] (gain index per g=0..15)
    u_sb: np.ndarray,     # uint8 [M, nsb] (linear superblock scale 0..255)
    fmt: VtqFormat,
) -> np.ndarray:
    """Packs `[M, nsb]` quantized superblocks into contiguous `vtq_st16_k*_t` bytes (pads `M` to `% 16 == 0` if needed)."""
    M, nsb, g_dim, c_dim = c_all.shape
    assert g_dim == 16 and c_dim == 2, f"Expected c_all [M, nsb, 16, 2], got {c_all.shape}"
    M_pad = _align(M, SUPER_ROWS)
    if M_pad != M:
        pad = M_pad - M
        c_all = np.pad(c_all, ((0, pad), (0, 0), (0, 0), (0, 0)))
        u_all = np.pad(u_all, ((0, pad), (0, 0), (0, 0)))
        g_sub = np.pad(g_sub, ((0, pad), (0, 0), (0, 0)))
        u_sb = np.pad(u_sb, ((0, pad), (0, 0)))

    nbg = M_pad // SUPER_ROWS
    fc = fmt.fmt_code

    # `c`: [nbg, 16_r, nsb, 16_g, 2] -> [nbg, nsb, 16_g, 16_r, 2] = 512 B
    c_st = c_all.reshape(nbg, 16, nsb, 16, 2).transpose(0, 2, 3, 1, 4).astype(np.uint8)
    # `u`: [nbg, 16_r, nsb, 16_g, 16_j] -> [nbg, nsb, 16_g, 16_r, 16_j]
    u_st = u_all.reshape(nbg, 16, nsb, 16, 16).transpose(0, 2, 3, 1, 4).astype(np.uint8)

    # `hdr[16][h]` (`uint16` per `(g, hb)`, bit `r` = `(g_sub[r, g] >> hb) & 1`):
    g_gr = g_sub.reshape(nbg, 16, nsb, 16).transpose(0, 2, 3, 1).astype(np.uint16)  # [nbg, nsb, 16_g, 16_r]
    sh16 = np.arange(16, dtype=np.uint16)[None, None, None, :]
    hdr_planes = [
        np.sum(((g_gr >> hb) & 1) << sh16, axis=-1, dtype=np.uint16).astype("<u2")
        for hb in range(fmt.h)
    ]
    hdr_u16 = np.stack(hdr_planes, axis=-1)  # [nbg, nsb, 16_g, h]
    hdr_bytes = np.ascontiguousarray(hdr_u16).view(np.uint8).reshape(nbg, nsb, fmt.h * 32)

    # `u_sb[16]` (`uint8` per row `r = 0..15`):
    usb_bytes = np.ascontiguousarray(u_sb.reshape(nbg, 16, nsb).transpose(0, 2, 1), dtype=np.uint8).reshape(nbg, nsb, 16)

    parts = [c_st.reshape(nbg, nsb, 512)]
    if fc in (15, 17):
        parts.append(_pack_lsb16(u_st & 1, fmt.gosset))
    elif fc == 20:
        parts.append(_pack_eo_plane(u_st & 1))
    elif fc == 40:
        parts.append(_pack_eo_plane((u_st >> 2) & 1))
        parts.append(_pack_u2_plane(u_st & 3))
    elif fc in (50, 51):
        parts.append(_pack_eo_plane((u_st >> 3) & 1))
        parts.append(_pack_u2_plane(u_st & 3))
        parts.append(_pack_eo_plane((u_st >> 2) & 1))
    elif fc == 60:
        parts.append(_pack_eo_plane((u_st >> 4) & 1))
        parts.append(_pack_u4_plane(u_st & 15))
    else:
        parts.append(_pack_eo_plane((u_st >> (fmt.K - 2)) & 1))
        for bit in range(fmt.K - 3, 0, -1):
            parts.append(_pack_eo_plane((u_st >> bit) & 1))
        parts.append(_pack_lsb16(u_st & 1, fmt.gosset))

    parts.append(hdr_bytes)
    parts.append(usb_bytes)
    packed = np.concatenate(parts, axis=-1)
    assert packed.shape == (nbg, nsb, fmt.st_bytes), f"Expected {(nbg, nsb, fmt.st_bytes)}, got {packed.shape}"
    return np.ascontiguousarray(packed.reshape(-1))


def decode_supertiles_vtq(
    raw: bytes | memoryview | np.ndarray,
    M: int,
    K: int,
    fmt: VtqFormat,
    d_row: np.ndarray | None = None,
    *, gain_luts: np.ndarray | None = None,
) -> np.ndarray:
    """Vectorized reference decoder; gain_luts is for the separate TSB1 research container only."""
    assert K % TILE == 0
    M_pad = _align(M, SUPER_ROWS)
    nbg, nsb = M_pad // SUPER_ROWS, K // TILE
    fc = fmt.fmt_code
    buf = np.frombuffer(raw, dtype=np.uint8, count=nbg * nsb * fmt.st_bytes).reshape(nbg, nsb, fmt.st_bytes)

    # Unpack native little-endian `uint16_t` `c[16][16][2]` (`[0]=cols 8..15, [1]=cols 0..7`) and step intra-subblock 16x16 1024-state shift register:
    c_gr2 = buf[:, :, :512].reshape(nbg, nsb, 16, 16, 2)
    c_msb_pair = c_gr2[..., ::-1]  # [..., 0]=cols 0..7, [..., 1]=cols 8..15
    sh8 = np.arange(7, -1, -1, dtype=np.uint8)[None, None, None, None, None, :]
    c_bits = ((c_msb_pair[..., None] >> sh8) & 1).reshape(nbg, nsb, 16, 256)
    st_seq = np.empty((nbg, nsb, 16, 256), dtype=np.uint16)
    st = ((c_gr2[:, :, :, 15, 1].astype(np.uint16) & 1) << 8) | c_gr2[:, :, :, 15, 0].astype(np.uint16)
    for step in range(256):
        st = ((st << 1) | c_bits[..., step].astype(np.uint16)) & 0x3FF
        st_seq[..., step] = st
    lb_grj = galois_lb_vtq(st_seq, fmt.nl).reshape(nbg, nsb, 16, 16, 16).astype(np.int32)

    if fc in (15, 17):
        u_grj, _ = _unpack_lsb16(buf, 512, fmt.gosset, nbg, nsb)
    elif fc == 20:
        u_grj = _unpack_eo_plane(buf[:, :, 512:1024], nbg, nsb)
    elif fc == 40:
        neg = _unpack_eo_plane(buf[:, :, 512:1024], nbg, nsb)
        u01 = _unpack_u2_plane(buf[:, :, 1024:2048], nbg, nsb)
        u_grj = (neg << 2) | u01
    elif fc in (50, 51):
        neg = _unpack_eo_plane(buf[:, :, 512:1024], nbg, nsb)
        u01 = _unpack_u2_plane(buf[:, :, 1024:2048], nbg, nsb)
        b2 = _unpack_eo_plane(buf[:, :, 2048:2560], nbg, nsb)
        u_grj = (neg << 3) | (b2 << 2) | u01
    elif fc == 60:
        neg = _unpack_eo_plane(buf[:, :, 512:1024], nbg, nsb)
        u_mag = _unpack_u4_plane(buf[:, :, 1024:3072], nbg, nsb)
        u_grj = (neg << 4) | u_mag
    else:
        neg = _unpack_eo_plane(buf[:, :, 512:1024], nbg, nsb)
        u_grj = neg << (fmt.K - 2)
        off = 1024
        for bit in range(fmt.K - 3, 0, -1):
            b_pl = _unpack_eo_plane(buf[:, :, off : off + 512], nbg, nsb)
            u_grj |= b_pl << bit
            off += 512
        u0, _ = _unpack_lsb16(buf, off, fmt.gosset, nbg, nsb)
        u_grj |= u0

    hdr_off = fmt.st_bytes - 16 - fmt.h * 32
    hdr_u16 = np.frombuffer(np.ascontiguousarray(buf[:, :, hdr_off : fmt.st_bytes - 16]), dtype="<u2").reshape(nbg, nsb, 16, fmt.h)
    sh16 = np.arange(16, dtype=np.uint16)[None, None, None, :]
    g_idx = np.zeros((nbg, nsb, 16, 16), dtype=np.int32)
    for hb in range(fmt.h):
        g_idx |= (((hdr_u16[:, :, :, hb, None] >> sh16) & 1).astype(np.int32)) << hb

    u_sb_f = buf[:, :, -16:].astype(np.float32) * U_SB_UNIT  # [nbg, nsb, 16_r]
    if gain_luts is None:
        vals = fmt.lut.reshape(fmt.nl, fmt.U)[lb_grj, u_grj]
    else:
        if gain_luts.shape != (1 << fmt.h, fmt.nl, fmt.U):
            raise ValueError("invalid gain-conditioned lookup table shape")
        vals = gain_luts[g_idx[..., None], lb_grj, u_grj]
    w_grj = vals * fmt.gain_tab[g_idx][..., None] * u_sb_f[:, :, None, :, None]
    W_pad = w_grj.transpose(0, 3, 1, 2, 4).reshape(M_pad, K).astype(np.float32)
    W_out = W_pad[:M]
    if d_row is not None:
        dr = np.asarray(d_row, dtype=np.float16).astype(np.float32).reshape(M, 1)
        W_out = W_out * dr
    return W_out


def assemble_vtq(
    G_nat: torch.Tensor,   # float32 [M, K] in natural column order (`gains * tab`)
    d_row: torch.Tensor,   # float16 [M]
    u_sb: torch.Tensor,    # uint8 [M, K // 256]
    suh: torch.Tensor,     # float16 [K]
    rot_block: int = -1,
) -> torch.Tensor:
    """Reconstructs `W_hat [M, K]` bit-identically to `VtqFile.decode_matrix`."""
    M, K = G_nat.shape
    nsb = K // TILE
    d_eff = d_row.float().unsqueeze(1) * (u_sb.float() * float(U_SB_UNIT))
    W_rot = (G_nat.view(M, nsb, TILE) * d_eff.unsqueeze(-1)).reshape(M, K)
    return apply_vtq_rot(W_rot, rot_block=rot_block, transpose=True) * suh.float().unsqueeze(0)


# ------------------------------------------------------------------------------------------------
# `.vtq` Binary Container (`VtqWriter` & `VtqFile`)
# ------------------------------------------------------------------------------------------------

class VtqWriter:
    def __init__(self, path: Path, formats: list[VtqFormat], reserve: int = (4 << 20) - 16, *, codec=None):
        # The caller (never checkpoint metadata) may supply an explicit CPU codec.
        self._pack_tiles = pack_supertiles_vtq if codec is None else codec.pack
        self.f = open(path, "w+b")
        self.reserve = _align(16 + reserve, 64) - 16
        self.data_off = 16 + self.reserve
        self.f.write(b"\0" * self.data_off)
        self.pos = 0
        self.tensors: dict[str, dict] = {}
        self.formats = formats
        self.fmt_idx = {fmt.name: i for i, fmt in enumerate(formats)}

    def _pad(self, a: int = 64):
        p = _align(self.pos, a) - self.pos
        if p:
            self.f.write(b"\0" * p)
            self.pos += p

    def add_bytes(self, name: str, buf, dtype: str, shape: list[int]):
        self._pad(64)
        mv = memoryview(buf).cast("B")
        self.tensors[name] = {"dtype": dtype, "shape": list(shape), "offset": self.pos}
        self.f.write(mv)
        self.pos += len(mv)

    def add_tensor(self, name: str, t: torch.Tensor):
        t = t.detach().cpu().contiguous()
        dt = TORCH_TO_VTQ[t.dtype]
        arr = t.view(torch.uint16).numpy() if t.dtype == torch.bfloat16 else t.numpy()
        self.add_bytes(name, arr, dt, list(t.shape))

    def add_matrix(self, prefix: str, m: dict, verify: bool = True):
        """Writes a quantized VTQ matrix (`row_slices`, `c_all`, `u_all`, `g_sub`, `u_sb`, `d_row`, `suh`, `rot_block`)."""
        c_np = m["c_all"].cpu().numpy() if isinstance(m["c_all"], torch.Tensor) else m["c_all"]
        u_np = m["u_all"].cpu().numpy() if isinstance(m["u_all"], torch.Tensor) else m["u_all"]
        g_np = m["g_sub"].cpu().numpy() if isinstance(m["g_sub"], torch.Tensor) else m["g_sub"]
        usb_np = m["u_sb"].cpu().numpy() if isinstance(m["u_sb"], torch.Tensor) else m["u_sb"]
        d_row_t = m["d_row"] if isinstance(m["d_row"], torch.Tensor) else torch.tensor(m["d_row"], dtype=torch.float16)
        d_row_np = d_row_t.detach().cpu().to(torch.float16).numpy()
        K = c_np.shape[1] * TILE
        rot_block = int(m.get("rot_block", -1))

        blobs, meta, off = [], [], 0
        for r0, r1, name in m["row_slices"]:
            fi = self.fmt_idx[name]
            fmt = self.formats[fi]
            b = self._pack_tiles(c_np[r0:r1], u_np[r0:r1], g_np[r0:r1], usb_np[r0:r1], fmt).tobytes()
            pad = (-len(b)) & 63
            blobs.append(b + b"\0" * pad)
            meta.append([r0, r1, fi, off, rot_block])
            off += len(b) + pad
            if verify and "W_hat_rot" in m:
                dec = decode_supertiles_vtq(b, r1 - r0, K, fmt, d_row=d_row_np[r0:r1])
                ref = m["W_hat_rot"][r0:r1].cpu().numpy()
                if not np.allclose(dec, ref, atol=1e-6, rtol=1e-6):
                    mx = float(np.max(np.abs(dec - ref)))
                    raise AssertionError(f"{prefix} [{r0}:{r1}] {name}: supertile decode mismatch max_diff={mx}")

        self.add_bytes(f"{prefix}.vtq.data", b"".join(blobs), "u8", [off])
        self.add_bytes(f"{prefix}.vtq.slices", np.array(meta, dtype=np.int32), "i32", [len(meta), 5])
        self.add_tensor(f"{prefix}.vtq.d_row", d_row_t.to(torch.float16))
        self.add_tensor(f"{prefix}.vtq.suh", m["suh"].to(torch.float16))

    def finalize(self, meta: dict):
        hdr = json.dumps({**meta, "tensors": self.tensors}, separators=(",", ":")).encode("utf-8")
        assert len(hdr) <= self.reserve, f"header {len(hdr)} B > reserve {self.reserve} B"
        self.f.seek(0)
        self.f.write(MAGIC + struct.pack("<I", VERSION) + struct.pack("<Q", len(hdr)) + hdr + b" " * (self.reserve - len(hdr)))
        self.f.close()


class VtqFile:
    def __init__(self, path: Path, *, codec=None):
        import mmap

        self._unpack_tiles = decode_supertiles_vtq if codec is None else codec.unpack

        self.f = open(path, "rb")
        self.mm = mmap.mmap(self.f.fileno(), 0, access=mmap.ACCESS_READ)
        magic, ver, hlen = struct.unpack_from("<4sIQ", self.mm, 0)
        assert magic == MAGIC and ver in (VERSION, 3), f"not a .vtq file: {magic!r} v{ver}"
        self.header = json.loads(bytes(self.mm[16 : 16 + hlen]))
        self.formats = [format_from_json_vtq(d) for d in self.header["quantization"]["formats"]]
        # VtqWriter pads the header with spaces but the tensor payload can
        # itself begin with 0x20 (a perfectly valid packed trellis byte).
        # Its final tensor ends at EOF, so scan no further than the last
        # possible aligned start that still fits all declared tensor bytes.
        extent = max((t["offset"] + math.prod(t["shape"]) * np.dtype(DTYPES[t["dtype"]][0]).itemsize
                      for t in self.header["tensors"].values()), default=0)
        last_start = (len(self.mm) - extent) // 64 * 64
        end = 16 + hlen
        if _align(end, 64) > last_start:
            self.mm.close()
            self.f.close()
            raise ValueError("truncated VTQ tensor payload")
        while end < last_start and self.mm[end] == 0x20:
            end += 1
        self.data_off = _align(end, 64)

    def raw(self, name: str) -> memoryview:
        t = self.header["tensors"][name]
        n = int(np.prod(t["shape"])) * np.dtype(DTYPES[t["dtype"]][0]).itemsize
        o = self.data_off + t["offset"]
        return memoryview(self.mm[o : o + n])

    def tensor(self, name: str) -> torch.Tensor:
        t = self.header["tensors"][name]
        np_dt, _ = DTYPES[t["dtype"]]
        arr = torch.from_numpy(np.frombuffer(self.raw(name), dtype=np_dt).copy()).reshape(t["shape"])
        return arr.view(torch.bfloat16) if t["dtype"] == "bf16" else arr

    def decode_matrix(self, prefix: str, device="cuda") -> torch.Tensor:
        suh = self.tensor(f"{prefix}.vtq.suh").to(device)
        d_row = self.tensor(f"{prefix}.vtq.d_row").cpu().numpy()
        slices = self.tensor(f"{prefix}.vtq.slices").tolist()
        M = max(s[1] for s in slices)
        K = suh.shape[0]
        data = self.raw(f"{prefix}.vtq.data")
        W_rot = torch.empty((M, K), dtype=torch.float32, device=device)
        rot_block = slices[0][4] if len(slices[0]) >= 5 else -1
        for row_s in slices:
            r0, r1, fi, off = row_s[:4]
            fmt = self.formats[fi]
            nbg = _align(r1 - r0, SUPER_ROWS) // SUPER_ROWS
            nbytes = nbg * (K // TILE) * fmt.st_bytes
            dec_np = self._unpack_tiles(data[off : off + nbytes], r1 - r0, K, fmt, d_row=d_row[r0:r1])
            W_rot[r0:r1] = torch.from_numpy(dec_np).to(device)
        return apply_vtq_rot(W_rot, rot_block=rot_block, transpose=True) * suh.float().unsqueeze(0)


# ------------------------------------------------------------------------------------------------
# Comprehensive Self-Test (CPU C `vtq_avx2_bench.c` Cross-Check + CUDA Viterbi + `.vtq` Container)
# ------------------------------------------------------------------------------------------------

def self_test(cpu_only: bool = False) -> dict:
    rng = np.random.default_rng(2026)
    synth_fmts: list[VtqFormat] = []
    for fc, (K, gosset, nl, h, _) in FMT_META.items():
        if fc in (15, 17):
            pos = np.sort(rng.integers(0, 127, size=16, dtype=np.int8))
            tab_s8 = np.concatenate([-pos[::-1], pos]).astype(np.int8)
            hw = {"type": "s8_32", "qmax": 127, "tab_div": 64, "tab_s8": tab_s8.tolist()}
        elif K == 4 and nl == 16:
            ta = np.sort(rng.integers(0, 64, size=16, dtype=np.uint8))
            tb = np.sort(rng.integers(1, 64, size=16, dtype=np.uint8))
            hw = {"type": "at2_16_16", "qmax": 127, "tab_div": 64, "TA_u8": ta.tolist(), "TB_u8": tb.tolist()}
        elif K in (5, 6) and nl == 16:
            ta = np.sort(rng.integers(0, 64, size=32, dtype=np.uint8))
            tb = np.sort(rng.integers(1, 64, size=32, dtype=np.uint8))
            hw = {"type": "at2z_32_32", "qmax": 127, "tab_div": 64, "TA_u8": ta.tolist(), "TB_u8": tb.tolist()}
        else:
            m_mag = nl * ((1 << (K - 1)) // 2)
            tab_u8 = np.sort(rng.integers(1, 128, size=m_mag, dtype=np.uint8))
            hw = {"type": "u8", "qmax": 127, "tab_div": 64, "tab_u8": tab_u8.tolist()}
        gain_u8 = np.linspace(44, 96, 1 << h).round().astype(np.uint8)
        synth_fmts.append(
            VtqFormat(
                name=f"TEST_{fc}",
                fmt_code=fc,
                K=K,
                gosset=gosset,
                nl=nl,
                h=h,
                sub=16,
                hw_codebook=hw,
                gain_u8=gain_u8,
                lut=reconstruct_lut_from_hw_vtq(fc, hw),
                gain_tab=gain_u8.astype(np.float32) * GAIN_UNIT,
                scale=1.0,
            )
        )

    report: dict = {"formats": {}}
    M, K = 32, 512
    nsb = K // TILE
    a_i8 = rng.integers(-127, 128, size=K, dtype=np.int8)
    d_row = rng.uniform(0.005, 0.03, size=M).astype(np.float16)

    # Compile a tiny shared helper from `src/cpu/vtq_avx2_bench.c` (`scalar_ref_vtq`) to verify C byte-compatibility!
    c_src_path = Path(__file__).resolve().parent / "cpu" / "vtq_avx2_bench.c"
    c_lib = None
    tmp_so = None
    if c_src_path.exists():
        tmp_so = Path(tempfile.mktemp(suffix=".so"))
        wrap_c = Path(tempfile.mktemp(suffix=".c"))
        wrap_c.write_text(
            f'#define M_ROWS {M}\n#define K_COLS {K}\n#define main vtq_bench_unused_main\n'
            f'#include "{c_src_path}"\n#undef main\n'
            f'void c_scalar_check(const void* st, size_t stride, const uint8_t* tab, const uint8_t* g8, '
            f'const int8_t* a, const float* dr, float* y, int fmt) {{\n'
            f'    scalar_ref_vtq(st, stride, tab, g8, a, dr, y, fmt);\n}}\n',
            encoding="utf-8",
        )
        subprocess.run(["gcc", "-O2", "-mavx2", "-mfma", "-shared", "-fPIC", str(wrap_c), "-lm", "-o", str(tmp_so)], check=True)
        wrap_c.unlink(missing_ok=True)
        c_lib = ctypes.CDLL(str(tmp_so))
        c_lib.c_scalar_check.argtypes = [
            ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
        ]

    try:
        for fmt in synth_fmts:
            c_all = rng.integers(0, 256, size=(M, nsb, 16, 2), dtype=np.uint8)
            if fmt.gosset == 1:
                cw_idx = rng.integers(0, 16, size=(M, nsb, 32), dtype=np.int32)
                cw_bytes = HAM_CW8[cw_idx]
                u0 = ((cw_bytes[..., None] >> np.arange(7, -1, -1, dtype=np.uint8)) & 1).astype(np.uint8).reshape(M, nsb, 256)
            elif fmt.gosset == 2:
                p012 = rng.integers(0, 2, size=(M, nsb, 64, 3), dtype=np.uint8)
                p3 = (p012[..., 0] ^ p012[..., 1] ^ p012[..., 2])[..., None]
                u0 = np.concatenate([p012, p3], axis=-1).reshape(M, nsb, 256)
            elif fmt.gosset == 3:
                b1 = rng.integers(0, 2, size=(M, nsb, 64, 1), dtype=np.uint8)
                u0 = np.repeat(b1, 4, axis=-1).reshape(M, nsb, 256)
            else:
                u0 = None

            if u0 is not None:
                if fmt.U == 2:
                    u_nat = u0
                else:
                    hi = rng.integers(0, fmt.U // 2, size=(M, nsb, 256), dtype=np.uint8)
                    u_nat = ((hi << 1) | u0).astype(np.uint8)
            else:
                u_nat = rng.integers(0, fmt.U, size=(M, nsb, 256), dtype=np.uint8)
            g_sub = rng.integers(0, 1 << fmt.h, size=(M, nsb, 16), dtype=np.uint8)
            u_sb = rng.integers(32, 240, size=(M, nsb), dtype=np.uint8)

            packed = pack_supertiles_vtq(c_all, u_nat, g_sub, u_sb, fmt)
            W_dec = decode_supertiles_vtq(packed, M, K, fmt, d_row=d_row)
            y_py = W_dec @ a_i8.astype(np.float32)

            c_rel_err = 0.0
            if c_lib is not None and fmt.fmt_code in (15, 20, 25, 30, 31, 35, 40, 50, 51, 60):
                tab_raw = np.ascontiguousarray(hw_to_canon_u8_vtq(fmt.fmt_code, fmt.hw_codebook), dtype=np.uint8)
                g8_pad = np.zeros(8, dtype=np.uint8)
                g8_pad[: len(fmt.gain_u8)] = fmt.gain_u8
                dr_eff = (d_row.astype(np.float32) * float(TAB_UNIT * GAIN_UNIT * U_SB_UNIT)).astype(np.float32)
                y_c = np.empty(M, dtype=np.float32)
                c_lib.c_scalar_check(
                    packed.ctypes.data, fmt.st_bytes, tab_raw.ctypes.data, g8_pad.ctypes.data,
                    a_i8.ctypes.data, dr_eff.ctypes.data, y_c.ctypes.data, int(fmt.fmt_code),
                )
                c_rel_err = float(np.linalg.norm(y_py - y_c) / max(np.linalg.norm(y_c), 1e-12))
                assert c_rel_err < 1e-6, f"C struct mismatch in {fmt.name}: c_rel_err={c_rel_err}"

            report["formats"][fmt.name] = {
                "fmt_code": fmt.fmt_code,
                "st_bytes": fmt.st_bytes,
                "bpw_4096": round(fmt.bpw(4096, 4096), 5),
                "c_rel_err": c_rel_err,
            }
            print(
                f"  [CPU-Pack] {fmt.name:8s} (fmt={fmt.fmt_code:2d}): {fmt.st_bytes:4d} B/st16 "
                f"({fmt.bpw(4096, 4096):.4f} bpw) c_err={c_rel_err:.2e} OK",
                flush=True,
            )
    finally:
        if tmp_so is not None:
            tmp_so.unlink(missing_ok=True)

    if not cpu_only and torch.cuda.is_available():
        from src.vtq import viterbi_vtq_cuda

        dev = torch.device("cuda")
        for fmt in synth_fmts:
            N_seq = 32
            x = torch.randn(N_seq, 256, device=dev)
            w = torch.rand(N_seq, 256, device=dev) + 0.5
            tab_t = torch.tensor(fmt.lut, dtype=torch.float32, device=dev)
            gains_t = torch.tensor(fmt.gain_tab, dtype=torch.float32, device=dev)
            rec, g_out, c_out, u_out, _ = viterbi_vtq_cuda(x, w, tab_t, gains_t, fmt.nl, fmt.U, frac=fmt.gosset)

            d_one = np.ones(N_seq, dtype=np.float16)
            usb_128 = np.full((N_seq, 1), 128, dtype=np.uint8)  # 128 * (1/128) == 1.0 exactly!
            st_bytes = pack_supertiles_vtq(
                c_out.cpu().numpy().reshape(N_seq, 1, 16, 2),
                u_out.cpu().numpy().reshape(N_seq, 1, 256),
                g_out.cpu().numpy().reshape(N_seq, 1, 16),
                usb_128,
                fmt,
            )
            dec_nat = decode_supertiles_vtq(st_bytes, N_seq, 256, fmt, d_row=d_one)
            max_diff = float(np.max(np.abs(dec_nat - rec.cpu().numpy())))
            assert max_diff == 0.0, f"Bit-exact mismatch in {fmt.name}: max_diff={max_diff}"
            report["formats"][fmt.name]["gpu_viterbi_max_diff"] = max_diff
            print(f"  [GPU-Vit ] {fmt.name:8s} bit-exact max_diff={max_diff:.1e} OK", flush=True)

        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "selftest_v3.vtq"
            wr = VtqWriter(p, synth_fmts)
            M_c, K_c = 32, 512
            suh = torch.ones(K_c, dtype=torch.float16, device=dev) * 0.75
            c_m = rng.integers(0, 256, size=(M_c, 2, 16, 2), dtype=np.uint8)
            cw_idx = rng.integers(0, 16, size=(M_c, 2, 32), dtype=np.int32)
            u0 = ((HAM_CW8[cw_idx][..., None] >> np.arange(7, -1, -1, dtype=np.uint8)) & 1).astype(np.uint8).reshape(M_c, 2, 256)
            g_m = np.zeros((M_c, 2, 16), dtype=np.uint8)
            usb_m = rng.integers(64, 200, size=(M_c, 2), dtype=np.uint8)
            dr_m = rng.uniform(0.01, 0.03, size=M_c).astype(np.float16)
            slices = [(0, 16, "TEST_25"), (16, 32, "TEST_40")]
            W_rot_ref = np.zeros((M_c, K_c), dtype=np.float32)
            for r0, r1, snm in slices:
                f_obj = next(f for f in synth_fmts if f.name == snm)
                W_rot_ref[r0:r1] = decode_supertiles_vtq(
                    pack_supertiles_vtq(c_m[r0:r1], u0[r0:r1], g_m[r0:r1], usb_m[r0:r1], f_obj),
                    r1 - r0, K_c, f_obj, d_row=dr_m[r0:r1],
                )
            wr.add_matrix(
                "model.layers.0.self_attn.q_proj",
                {
                    "row_slices": slices,
                    "c_all": c_m,
                    "u_all": u0,
                    "g_sub": g_m,
                    "u_sb": usb_m,
                    "d_row": dr_m,
                    "suh": suh,
                    "rot_block": -1,
                    "W_hat_rot": torch.from_numpy(W_rot_ref),
                },
                verify=True,
            )
            wr.finalize({"format": "vtq", "quantization": {"formats": [format_json_vtq(f) for f in synth_fmts]}})
            vf = VtqFile(p)
            W_reloaded = vf.decode_matrix("model.layers.0.self_attn.q_proj", device=dev)
            W_expected = apply_vtq_rot(torch.from_numpy(W_rot_ref).to(dev), rot_block=-1, transpose=True) * suh.float().unsqueeze(0)
            c_diff = float((W_reloaded - W_expected).abs().max().item())
            assert c_diff < 1e-6, f"Container roundtrip diff={c_diff}"
            report["container_roundtrip_max_diff"] = c_diff
            print(f"  [Container] `.vtq` roundtrip max_diff={c_diff:.2e} OK", flush=True)

    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu-only", action="store_true")
    args = ap.parse_args()
    self_test(cpu_only=args.cpu_only)
