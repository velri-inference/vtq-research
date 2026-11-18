"""
Reference implementation of the VTQ v2 format and `.vtq` v2 binary container:
  1. Unified CPU AVX2 + GPU CUDA bitplane tile layout (`code` + `usgn` + `u_mid` + `lsb` + `gains`):
     - Separating `usgn[32B]` makes uncoded magnitude bitplanes exact powers of two (`0` @ K=2, `1` @ K=3,
       aligned `uint16` @ K=4, `2+1` @ K=5, aligned `uint32` @ K=6) and eliminates signed `vpshufb` limits on AVX2.
  2. Lossless hardware `uint8` antipodal / additive codebooks (`qmax = 128`, immune to `vpmaddubsw` `int16` saturation):
     - `at`: unsigned magnitude `mag_u8` (`32 B` @ K=2, `64 B` @ K=3)
     - `at2_16_16`: zoned additive `TA_u8[16] + TB_u8[16]` (`32 B` @ K=4)
     - `at2z_32_32`: 2-zone additive `TA_u8[32] + TB_u8[32]` (`64 B` @ K=5, 6)
  3. Extended Hamming `[8, 4, 4]` Gosset E8 LSB (`frac = 4`, `"gosset84"`, rate `K - 0.5`):
     - Packs 8 consecutive LSBs into one 4-bit codeword nibble (`16 B` per 256-weight tile), decoded on AVX2 via a
       single `vpshufb` over `HAM_CW8[16]` and on CUDA via bank-conflict-free shared `sham[16]`.
  4. Canonical Kronecker orthogonal transform `R_K = H_{2^a} (x) DCT-II_q` across full matrix width `K` (`256 | K`).
  5. `.vtq` v2 container (`VtqWriter` / `VtqFile`, `MAGIC = b"VTQ\\0"`, `VERSION = 2`) + `VtqRing` short-row codec (`L = 160, 256`).

Self-test (`python -m src.vtq_v2_format`):
  Verifies all 26 formats of `results/vtq_v2_tables.json` on `L = 256` tiles and `L = 160` short row rings against
  `viterbi_ub_cuda` (`0` label mismatches, `0` fractional-mode / Gosset E8 violations, `0` bit-exact decode mismatches),
  plus a full `.vtq` v2 container write/read/decode roundtrip with mixed-precision DP row slices and Kronecker `R_K`.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from src.iqk_format import fwht_inplace

NU = 10
TILE = 256
MAGIC = b"VTQ\0"
VERSION = 2

DTYPES = {
    "f32": (np.float32, torch.float32),
    "f16": (np.float16, torch.float16),
    "bf16": (np.uint16, torch.bfloat16),
    "i32": (np.int32, torch.int32),
    "i8": (np.int8, torch.int8),
    "u8": (np.uint8, torch.uint8),
}
TORCH_TO_VTQ = {
    torch.float32: "f32",
    torch.float16: "f16",
    torch.bfloat16: "bf16",
    torch.int32: "i32",
    torch.int8: "i8",
    torch.uint8: "u8",
}
FRAC_NAMES = ["none", "d2", "d4", "d4_eq", "gosset84"]

# 16 codewords of the [8, 4, 4] Extended Hamming (Gosset E8 binary) code, matching tcq_viterbi.cu (`pp = 4`)
HAM_CW8 = np.array(
    [0x00, 0xFF, 0xF0, 0x0F, 0xCC, 0x33, 0x3C, 0xC3, 0xAA, 0x55, 0x5A, 0xA5, 0x66, 0x99, 0x96, 0x69],
    dtype=np.uint8,
)
CW8_TO_IDX = np.full(256, -1, dtype=np.int16)
for _i, _cw in enumerate(HAM_CW8.tolist()):
    CW8_TO_IDX[_cw] = _i

HAM_BITS8 = ((HAM_CW8[:, None] >> np.arange(8, dtype=np.uint8)) & 1).astype(np.uint32)  # [16, 8]


def _align(x: int, a: int) -> int:
    return (x + a - 1) // a * a


# ------------------------------------------------------------------------------------------------
# Kronecker Orthogonal Transform R_K = H_{2^a} (x) DCT-II_q
# ------------------------------------------------------------------------------------------------

_DCT_CACHE: dict[tuple[int, torch.device], torch.Tensor] = {}
_HAD_CACHE: dict[tuple[int, torch.device], torch.Tensor] = {}
_KRON_CACHE: dict[tuple[int, torch.device], tuple[int, int, torch.Tensor, torch.Tensor]] = {}


def split_pow2_odd(K: int) -> tuple[int, int]:
    p2 = K & -K
    return p2, K // p2


def get_dct_ortho(q: int, device: torch.device) -> torch.Tensor:
    key = (q, device)
    if key not in _DCT_CACHE:
        m = torch.arange(q, dtype=torch.float64, device=device).unsqueeze(1)
        n = torch.arange(q, dtype=torch.float64, device=device).unsqueeze(0)
        mat = torch.cos(math.pi * m * (2.0 * n + 1.0) / (2.0 * q))
        scale = torch.full((q, 1), math.sqrt(2.0 / q), dtype=torch.float64, device=device)
        scale[0, 0] = math.sqrt(1.0 / q)
        _DCT_CACHE[key] = (mat * scale).float().contiguous()
    return _DCT_CACHE[key]


def get_hadamard_ortho(n: int, device: torch.device) -> torch.Tensor:
    key = (n, device)
    if key not in _HAD_CACHE:
        h = torch.tensor([[1.0]], dtype=torch.float64, device=device)
        cur = 1
        s2 = math.sqrt(0.5)
        while cur < n:
            h = torch.cat([torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)], dim=0) * s2
            cur *= 2
        _HAD_CACHE[key] = h.float().contiguous()
    return _HAD_CACHE[key]


def _get_kron_factors(K: int, device: torch.device) -> tuple[int, int, torch.Tensor, torch.Tensor]:
    key = (K, device)
    if key not in _KRON_CACHE:
        p2, q = split_pow2_odd(K)
        a = p2.bit_length() - 1
        b1 = 1 << (a // 2)
        b2 = 1 << (a - a // 2)
        H1 = get_hadamard_ortho(b1, device)
        H2 = get_hadamard_ortho(b2, device)
        M_right = H2 if q == 1 else torch.kron(H2.double(), get_dct_ortho(q, device).double()).float().contiguous()
        _KRON_CACHE[key] = (b1, b2 * q, H1, M_right)
    return _KRON_CACHE[key]


def apply_kronecker_rot(X: torch.Tensor, rot_block: int = -1, transpose: bool = False) -> torch.Tensor:
    """Applies canonical orthogonal transform R_K (`rot_block = -1`) or block FWHT (`rot_block = 256, ...`).
    Uses factored IEEE-FP32 matrix multiplication `H_{b1} (x) (H_{b2} (x) DCT_q)` without mutating `X`."""
    K = X.shape[-1]
    orig_shape = X.shape
    prev_tf32 = torch.backends.cuda.matmul.allow_tf32
    if prev_tf32:
        torch.backends.cuda.matmul.allow_tf32 = False
    try:
        if rot_block > 0:
            assert K % rot_block == 0 and (rot_block & (rot_block - 1)) == 0
            if rot_block <= 1024:
                Hb = get_hadamard_ortho(rot_block, X.device)
                return (X.reshape(-1, rot_block) @ Hb).reshape(orig_shape)
            return fwht_inplace(X.clone().contiguous(), block_size=rot_block)
        p2, _ = split_pow2_odd(K)
        assert p2 >= 32, f"Expected K={K} divisible by 32, got p2={p2}"
        b1, b2q, H1, M_right = _get_kron_factors(K, X.device)
        rmat = M_right if transpose else M_right.T
        Y = X.reshape(-1, b1, b2q) @ rmat
        return torch.matmul(H1, Y).reshape(orig_shape)
    finally:
        if prev_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True


# ------------------------------------------------------------------------------------------------
# Hardware `uint8` Codebook Reconstruction & VtqFormat Spec
# ------------------------------------------------------------------------------------------------

def _at2_indices_np(K: int, nl: int, u_low_n: int, mode: str) -> tuple[np.ndarray, np.ndarray]:
    lb = np.arange(nl, dtype=np.int64)[:, None]
    ul = np.arange(u_low_n, dtype=np.int64)[None, :]
    if mode == "at2_16_16":
        assert K == 4
        return np.broadcast_to(ul | ((lb & 3) << 2), (nl, u_low_n)), np.broadcast_to(ul | ((lb >> 2) << 2), (nl, u_low_n))
    if mode == "at2z_32_32":
        if K == 5:
            return np.broadcast_to(ul | ((lb & 3) << 3), (nl, u_low_n)), np.broadcast_to(ul | ((lb >> 2) << 3), (nl, u_low_n))
        if K == 6:
            return np.broadcast_to(ul | ((lb & 1) << 4), (nl, u_low_n)), np.broadcast_to((ul >> 2) | ((lb >> 1) << 2), (nl, u_low_n))
    raise ValueError(f"Unsupported AT2 mode={mode!r} for K={K}")


def reconstruct_lut_from_hw(K: int, nl: int, hw: dict) -> np.ndarray:
    """Deterministically reconstructs the float32 `[nl * 2^(K-1)]` table from the header's `uint8` codebook."""
    ub = K - 1
    U = 1 << ub
    u_low_n = 1 << max(ub - 1, 0)
    unit = np.float32(hw["unit"])
    qmax = int(hw.get("qmax", 128))
    tp = hw["type"]
    if tp == "at":
        q_u8 = np.asarray(hw["mag_u8"], dtype=np.int32).reshape(nl, u_low_n)
        rec_pos = q_u8.astype(np.float32) * unit
    elif tp in ("at2_16_16", "at2z_32_32"):
        idxA, idxB = _at2_indices_np(K, nl, u_low_n, tp)
        ta = np.asarray(hw["TA_u8"], dtype=np.int32)
        tb = np.asarray(hw["TB_u8"], dtype=np.int32)
        sum_u8 = np.clip(ta[idxA] + tb[idxB], 0, qmax)
        rec_pos = sum_u8.astype(np.float32) * unit
    else:
        raise ValueError(f"Unknown hw codebook type: {tp!r}")
    out = np.empty((nl, U), dtype=np.float32)
    out[:, u_low_n:] = rec_pos
    out[:, :u_low_n] = -rec_pos[::-1, ::-1]
    return out.reshape(-1)


@dataclass(frozen=True)
class VtqFormat:
    name: str
    K: int                # integer base bits/weight: 1 coded bit + (K - 1) uncoded bits before frac
    frac: int             # 0=none, 1=d2 (K-0.5), 2=d4 (K-0.25), 3=d4_eq (K-0.75), 4=gosset84 (K-0.5)
    masks: tuple[int, ...]  # (359, 40, 475) -> nl = 16 subsets
    h: int                # gain index bits per `sub` weights
    step: float           # gain log2 step
    sub: int              # weights per gain index (16 or 32)
    hw_codebook: dict     # lossless uint8 codebook ("at" | "at2_16_16" | "at2z_32_32", unit, qmax=128)
    lut: np.ndarray       # float32 [nl * 2^(K-1)] reconstructed identically from hw_codebook
    gain_tab: np.ndarray  # float32 [2^h] explicit gain multipliers

    @property
    def n_labels(self) -> int:
        return 2 ** len(self.masks) * 2

    @property
    def U(self) -> int:
        return 1 << (self.K - 1)

    def usgn_words(self, L: int = TILE) -> int:
        # At K=2 with frac!=0, the single uncoded bit `u & 1` IS the fractional-constrained plane (`lsb_words`).
        return (L // 16) if (self.K >= 3 or self.frac == 0) else 0

    def umid_words(self, L: int = TILE) -> int:
        return max(self.K - 3, 0) * (L // 16)

    def lsb_words(self, L: int = TILE) -> int:
        if self.K == 2 and self.frac == 0:
            return 0
        n_bits = {0: L, 1: L // 2, 2: 3 * L // 4, 3: L // 4, 4: L // 2}[self.frac]
        return (n_bits + 15) // 16

    def gain_words(self, L: int = TILE) -> int:
        return (L // self.sub * self.h + 15) // 16 if self.h > 0 else 0

    def tile_bytes(self, L: int = TILE) -> int:
        return 2 * (L // 16 + self.usgn_words(L) + self.umid_words(L) + self.lsb_words(L) + self.gain_words(L))

    def bpw(self, k_in: int = 4096) -> float:
        """File bits/weight for a matrix of width `k_in`: tiles + int8 `e` per 256-block + fp16 `d_row`."""
        return self.tile_bytes() * 8 / TILE + 8 / TILE + 16 / k_in


def load_vtq2_formats(tables_path: Path | None = None, menu: list[str] | None = None) -> dict[str, VtqFormat]:
    if tables_path is None:
        tables_path = Path(__file__).resolve().parent.parent / "results" / "vtq_v2_tables.json"
    raw = json.loads(Path(tables_path).read_text(encoding="utf-8"))
    names = menu if menu is not None else list(raw.keys())
    out: dict[str, VtqFormat] = {}
    for nm in names:
        c = raw[nm]
        K = int(c["K"])
        masks = (int(c["m"]),) + tuple(int(x) for x in c.get("m_extra", ()))
        nl = 2 ** len(masks) * 2
        hw = dict(c.get("hw_codebook") or c["hw"])
        lut = reconstruct_lut_from_hw(K, nl, hw)
        out[nm] = VtqFormat(
            name=nm,
            K=K,
            frac=int(c.get("pp", 0)),
            masks=masks,
            h=int(c["h"]),
            step=float(c.get("step", 0.0)),
            sub=int(c.get("sub", 16)),
            hw_codebook=hw,
            lut=lut,
            gain_tab=np.asarray(c["gains"], dtype=np.float32),
        )
    return out


def format_json(f: VtqFormat) -> dict:
    return {
        "name": f.name,
        "K": f.K,
        "frac": FRAC_NAMES[f.frac],
        "masks": [int(m) for m in f.masks],
        "h": f.h,
        "sub": f.sub,
        "hw_codebook": f.hw_codebook,
        "gain": [float(v) for v in np.asarray(f.gain_tab, np.float32)],
    }


def format_from_json(d: dict) -> VtqFormat:
    K = int(d["K"])
    masks = tuple(int(m) for m in d["masks"])
    nl = 2 ** len(masks) * 2
    hw = dict(d["hw_codebook"])
    return VtqFormat(
        name=d["name"],
        K=K,
        frac=FRAC_NAMES.index(d["frac"]),
        masks=masks,
        h=int(d["h"]),
        step=0.0,
        sub=int(d["sub"]),
        hw_codebook=hw,
        lut=reconstruct_lut_from_hw(K, nl, hw),
        gain_tab=np.asarray(d["gain"], dtype=np.float32),
    )


# ------------------------------------------------------------------------------------------------
# Single-Tile / Row-Ring Packing & Decoding (NumPy Reference)
# ------------------------------------------------------------------------------------------------

def parity(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.uint32)
    for s in (16, 8, 4, 2, 1):
        x ^= x >> s
    return x & 1


def states_from_codes(c: np.ndarray) -> np.ndarray:
    L = len(c)
    t = np.arange(L)
    s = np.zeros(L, dtype=np.uint32)
    for j in range(NU):
        s |= c[(t - j) % L].astype(np.uint32) << j
    return s


def labels_from_states(s: np.ndarray, fmt: VtqFormat) -> np.ndarray:
    lb = (s & 1) | (parity((s >> 1) & fmt.masks[0]) << 1)
    for i, m in enumerate(fmt.masks[1:]):
        lb |= parity((s >> 1) & m) << (2 + i)
    return lb


def _bits_to_words(bits: np.ndarray, n_words: int) -> np.ndarray:
    if n_words == 0:
        return np.zeros(0, dtype=np.uint16)
    out = np.zeros(n_words * 16, dtype=np.uint32)
    out[: len(bits)] = bits.astype(np.uint32)
    return ((out.reshape(n_words, 16) << np.arange(16, dtype=np.uint32)).sum(axis=1)).astype(np.uint16)


def _words_to_bits(words: np.ndarray, n_bits: int) -> np.ndarray:
    if n_bits == 0:
        return np.zeros(0, dtype=np.uint32)
    return ((words.astype(np.uint32)[:, None] >> np.arange(16, dtype=np.uint32)) & 1).reshape(-1)[:n_bits].astype(np.uint32)


def _pack_lsb_np(lsb_all: np.ndarray, fmt: VtqFormat, L: int) -> np.ndarray:
    nw = fmt.lsb_words(L)
    if nw == 0:
        return np.zeros(0, dtype=np.uint16)
    if fmt.frac == 0:
        return _bits_to_words(lsb_all, nw)
    if fmt.frac == 1:
        return _bits_to_words(lsb_all[0::2], nw)
    if fmt.frac == 2:
        return _bits_to_words(lsb_all.reshape(-1, 4)[:, :3].reshape(-1), nw)
    if fmt.frac == 3:
        return _bits_to_words(lsb_all[0::4], nw)
    if fmt.frac == 4:
        b8 = lsb_all.reshape(-1, 8).astype(np.uint32)
        bytes8 = (b8 << np.arange(8, dtype=np.uint32)).sum(axis=1)
        idx4 = CW8_TO_IDX[bytes8]
        if np.any(idx4 < 0):
            raise ValueError("Invalid Gosset [8, 4, 4] codeword encountered during packing")
        nibs_bits = ((idx4.astype(np.uint32)[:, None] >> np.arange(4, dtype=np.uint32)) & 1).reshape(-1)
        return _bits_to_words(nibs_bits, nw)
    raise ValueError(f"Unsupported frac={fmt.frac}")


def _unpack_lsb_np(lsb_words: np.ndarray, fmt: VtqFormat, L: int) -> np.ndarray:
    if fmt.frac == 0:
        return _words_to_bits(lsb_words, L)
    if fmt.frac == 1:
        return np.repeat(_words_to_bits(lsb_words, L // 2), 2)
    if fmt.frac == 2:
        b3 = _words_to_bits(lsb_words, 3 * L // 4).reshape(-1, 3)
        return np.concatenate([b3, b3.sum(axis=1, keepdims=True) & 1], axis=1).reshape(-1)
    if fmt.frac == 3:
        return np.repeat(_words_to_bits(lsb_words, L // 4), 4)
    if fmt.frac == 4:
        nb = _words_to_bits(lsb_words, L // 2).reshape(-1, 4)
        idx4 = (nb << np.arange(4, dtype=np.uint32)).sum(axis=1)
        return HAM_BITS8[idx4].reshape(-1)
    raise ValueError(f"Unsupported frac={fmt.frac}")


def pack_tile(labels: np.ndarray, u: np.ndarray, gidx: np.ndarray, fmt: VtqFormat) -> dict:
    """Packs one tile (`L = 256`) or short row ring (`L % 16 == 0`) into VTQ v2 unified bitplane fields."""
    L = len(labels)
    W = L // 16
    codes = _bits_to_words(labels & 1, W)
    if fmt.K >= 3 or fmt.frac == 0:
        usgn = _bits_to_words((u >> (fmt.K - 2)) & 1, W)
    else:
        usgn = np.zeros(0, dtype=np.uint16)
    umid = (
        np.stack([_bits_to_words((u >> p) & 1, W) for p in range(1, fmt.K - 2)])
        if fmt.K >= 4
        else np.zeros((0, W), dtype=np.uint16)
    )
    lsb = _pack_lsb_np(u & 1, fmt, L)
    if fmt.h > 0:
        g = gidx[:: fmt.sub].astype(np.uint32)
        gbits = ((g[:, None] >> np.arange(fmt.h, dtype=np.uint32)) & 1).reshape(-1)
        gains = _bits_to_words(gbits, fmt.gain_words(L))
    else:
        gains = np.zeros(0, dtype=np.uint16)
    return {"L": L, "codes": codes, "usgn": usgn, "umid": umid, "lsb": lsb, "gains": gains}


def decode_tile(f: dict, fmt: VtqFormat) -> np.ndarray:
    """Decodes unscaled tile or row-ring values `[L]` float32 from packed VTQ v2 bitplane fields."""
    L = f.get("L", TILE)
    c = _words_to_bits(f["codes"], L)
    lb = labels_from_states(states_from_codes(c), fmt)
    if fmt.K == 2 and fmt.frac == 0:
        u = _words_to_bits(f["usgn"], L)
    elif fmt.K == 2:
        u = _unpack_lsb_np(f["lsb"], fmt, L)
    else:
        u = _unpack_lsb_np(f["lsb"], fmt, L).copy()
        for p in range(1, fmt.K - 2):
            u |= _words_to_bits(f["umid"][p - 1], L) << p
        u |= _words_to_bits(f["usgn"], L) << (fmt.K - 2)
    if fmt.h > 0:
        gb = _words_to_bits(f["gains"], L // fmt.sub * fmt.h).reshape(-1, fmt.h)
        g = (gb << np.arange(fmt.h, dtype=np.uint32)).sum(axis=1)
        gain = fmt.gain_tab[np.repeat(g, fmt.sub)]
    else:
        gain = np.ones(L, dtype=np.float32)
    return (gain * fmt.lut[lb * fmt.U + u]).astype(np.float32)


# ------------------------------------------------------------------------------------------------
# Vectorized GPU / CPU Matrix Slice Packing & Decoding + `.vtq` v2 Container
# ------------------------------------------------------------------------------------------------

def _words_pt(bits: torch.Tensor) -> torch.Tensor:
    T_n, n = bits.shape
    if n == 0:
        return bits.new_zeros((T_n, 0), dtype=torch.int32)
    n16 = _align(n, 16)
    if n16 != n:
        bits = torch.cat([bits, bits.new_zeros(T_n, n16 - n)], dim=1)
    sh = torch.arange(16, device=bits.device, dtype=torch.int32)
    return (bits.view(T_n, -1, 16).to(torch.int32) << sh).sum(dim=2)


def _bits_pt(words: torch.Tensor, n: int) -> torch.Tensor:
    sh = torch.arange(16, device=words.device, dtype=torch.int32)
    return ((words.to(torch.int32).unsqueeze(2) >> sh) & 1).reshape(words.shape[0], -1)[:, :n]


def _tiles(x: torch.Tensor) -> torch.Tensor:
    R, K = x.shape
    return x.reshape(R // 16, 16, K // 16, 16).permute(0, 2, 1, 3).reshape(-1, TILE)


def _untiles(t: torch.Tensor, R: int, K: int) -> torch.Tensor:
    return t.reshape(R // 16, K // 16, 16, 16).permute(0, 2, 1, 3).reshape(R, K)


def _label_table(f: VtqFormat) -> torch.Tensor:
    s = np.arange(1 << NU)
    lab = s & 1
    for i, m in enumerate(f.masks):
        v = (s >> 1) & m
        par = np.zeros_like(v)
        while v.any():
            par ^= v & 1
            v >>= 1
        lab |= par << (i + 1)
    return torch.tensor(lab, dtype=torch.int64)


def pack_slice(idx: torch.Tensor, gi: torch.Tensor, f: VtqFormat) -> bytes:
    """Packs `[R, K]` matrix slice into 5 64-byte-aligned SoA regions: `[codes, usgn, umid, lsb, gains]`."""
    ix, g = _tiles(idx.to(torch.int32)), _tiles(gi.to(torch.int32))
    T_n = ix.shape[0]
    dev = ix.device
    lab, u = ix // f.U, ix % f.U

    r_codes = _words_pt(lab & 1)
    r_usgn = _words_pt((u >> (f.K - 2)) & 1) if (f.K >= 3 or f.frac == 0) else ix.new_zeros((T_n, 0))
    r_umid = (
        torch.stack([_words_pt((u >> p) & 1) for p in range(1, f.K - 2)], dim=1).reshape(T_n, -1)
        if f.K >= 4
        else ix.new_zeros((T_n, 0))
    )

    if f.K == 2 and f.frac == 0:
        r_lsb = ix.new_zeros((T_n, 0))
    else:
        lsb = u & 1
        if f.frac == 1:
            lsb = lsb[:, 0::2]
        elif f.frac == 2:
            lsb = lsb.view(T_n, -1, 4)[:, :, :3].reshape(T_n, -1)
        elif f.frac == 3:
            lsb = lsb[:, 0::4]
        elif f.frac == 4:
            sh8 = torch.arange(8, device=dev, dtype=torch.int32)
            bytes8 = (lsb.view(T_n, -1, 8).to(torch.int32) << sh8).sum(dim=2)
            lut_inv = torch.tensor(CW8_TO_IDX, device=dev, dtype=torch.int32)
            idx4 = lut_inv[bytes8]
            if bool((idx4 < 0).any()):
                raise RuntimeError(f"Invalid Gosset [8, 4, 4] codeword in slice for format {f.name}")
            sh4 = torch.arange(4, device=dev, dtype=torch.int32)
            lsb = ((idx4.unsqueeze(2) >> sh4) & 1).reshape(T_n, -1)
        r_lsb = _words_pt(lsb)

    if f.h > 0:
        gs = g[:, :: f.sub]
        sh_h = torch.arange(f.h, device=dev, dtype=torch.int32)
        r_gains = _words_pt(((gs.unsqueeze(2) >> sh_h) & 1).reshape(T_n, -1))
    else:
        r_gains = ix.new_zeros((T_n, 0))

    out = bytearray()
    for r in (r_codes, r_usgn, r_umid, r_lsb, r_gains):
        out += bytes(_align(len(out), 64) - len(out))
        if r.numel() > 0:
            out += r.to(torch.int32).cpu().numpy().astype(np.uint16).tobytes()
    return bytes(out)


def decode_slice(data: memoryview | bytes, R: int, K: int, f: VtqFormat, device) -> torch.Tensor:
    """Vectorized spec decoder `[R, K]` float32 from 5 64-byte-aligned SoA regions."""
    T_n = (R // 16) * (K // 16)
    sizes = [16, f.usgn_words(), f.umid_words(), f.lsb_words(), f.gain_words()]
    regions, off = [], 0
    for n in sizes:
        off = _align(off, 64)
        regions.append(
            torch.frombuffer(bytearray(data[off : off + 2 * T_n * n]), dtype=torch.int16).to(device).view(T_n, n)
            if n > 0
            else None
        )
        off += 2 * T_n * n

    c = _bits_pt(regions[0], TILE)
    s = sum(torch.roll(c, j, dims=1) << j for j in range(NU))
    lab = _label_table(f).to(device)[s]

    if f.K == 2 and f.frac == 0:
        u = _bits_pt(regions[1], TILE).to(torch.int64)
    else:
        n_lsb_bits = {0: TILE, 1: TILE // 2, 2: 3 * TILE // 4, 3: TILE // 4, 4: TILE // 2}[f.frac]
        lb = _bits_pt(regions[3], n_lsb_bits)
        if f.frac == 1:
            lb = lb.repeat_interleave(2, dim=1)
        elif f.frac == 2:
            b3 = lb.view(T_n, -1, 3)
            lb = torch.cat([b3, b3.sum(dim=2, keepdim=True) & 1], dim=2).reshape(T_n, TILE)
        elif f.frac == 3:
            lb = lb.repeat_interleave(4, dim=1)
        elif f.frac == 4:
            sh4 = torch.arange(4, device=device, dtype=torch.int32)
            idx4 = (lb.view(T_n, -1, 4) << sh4).sum(dim=2).long()
            ham_pt = torch.tensor(HAM_BITS8, device=device, dtype=torch.int64)
            lb = ham_pt[idx4].reshape(T_n, TILE)
        u = lb.to(torch.int64)
        if f.K >= 3:
            for p in range(1, f.K - 2):
                u = u | (_bits_pt(regions[2][:, 16 * (p - 1) : 16 * p], TILE).to(torch.int64) << p)
            u = u | (_bits_pt(regions[1], TILE).to(torch.int64) << (f.K - 2))

    if f.h > 0:
        gb = _bits_pt(regions[4], TILE // f.sub * f.h).view(T_n, -1, f.h)
        sh_h = torch.arange(f.h, device=device, dtype=torch.int32)
        g = (gb << sh_h).sum(dim=2).long().repeat_interleave(f.sub, dim=1)
        gain = torch.tensor(f.gain_tab, device=device)[g]
    else:
        gain = torch.ones((T_n, TILE), device=device)

    vals = gain * torch.tensor(f.lut, device=device)[lab * f.U + u]
    return _untiles(vals, R, K)


def assemble_vtq2(
    G: torch.Tensor, d_row: torch.Tensor, e: torch.Tensor, suh: torch.Tensor, rot_block: int = -1
) -> torch.Tensor:
    """Reconstructs `W = R_K^T(d · G) ⊙ suh` with `d[m, b] = d_row[m] · 2^(0.25 · e[m, b])`."""
    M, K = G.shape
    scale = d_row.float().unsqueeze(1) * torch.exp2(0.25 * e.float())
    W_scaled = (scale.unsqueeze(2) * G.view(M, K // TILE, TILE)).view(M, K)
    return apply_kronecker_rot(W_scaled, rot_block=rot_block, transpose=True) * suh.float().unsqueeze(0)


class VtqWriter:
    """Streams `.vtq` v2 binary container (`MAGIC = b'VTQ\\0'`, `VERSION = 2`) with bit-exact slice verification."""

    def __init__(self, path: Path, formats: list[VtqFormat], reserve: int = (64 << 20) - 16):
        assert (16 + reserve) % 4096 == 0
        self.path, self.reserve = Path(path), reserve
        self.formats = formats
        self.fmt_index = {f.name: i for i, f in enumerate(formats)}
        self.f = open(self.path, "wb")
        self.f.write(bytes(16 + reserve))
        self.cur = 0
        self.tensors: dict[str, dict] = {}
        self.quantized: list[str] = []

    def _write(self, name: str, raw: bytes, dtype: str, shape: list, align: int = 64):
        pad = _align(self.cur, align) - self.cur
        if pad:
            self.f.write(bytes(pad))
            self.cur += pad
        self.tensors[name] = {"dtype": dtype, "shape": [int(s) for s in shape], "offset": self.cur}
        self.f.write(raw)
        self.cur += len(raw)

    def add_tensor(self, name: str, t: torch.Tensor):
        t = t.detach().cpu().contiguous()
        dt = TORCH_TO_VTQ[t.dtype]
        raw = t.view(torch.int16).numpy().tobytes() if t.dtype == torch.bfloat16 else t.numpy().tobytes()
        self._write(name, raw, dt, list(t.shape))

    def add_matrix(self, p: str, m: dict, verify: bool = True) -> int:
        M, K = m["idx"].shape
        rows, data = [], bytearray()
        for r0, r1, sp in m["row_slices"]:
            f = self.formats[self.fmt_index[sp]]
            data += bytes(_align(len(data), 64) - len(data))
            rows.append([r0, r1, self.fmt_index[sp], len(data)])
            raw = pack_slice(m["idx"][r0:r1], m["gi"][r0:r1], f)
            if verify:
                G = m["G"][r0:r1]
                bad = int((decode_slice(raw, r1 - r0, K, f, G.device) != G).sum())
                if bad:
                    raise RuntimeError(f"{p} [{r0}:{r1}] {sp}: VTQ v2 decode disagrees with quantizer in {bad} weights")
            data += raw
        self.add_tensor(f"{p}.vtq.slices", torch.tensor(rows, dtype=torch.int32))
        self._write(f"{p}.vtq.data", bytes(data), "u8", [len(data)])
        self.add_tensor(f"{p}.vtq.d_row", m["d_row"].to(torch.float16))
        self.add_tensor(f"{p}.vtq.e", m["e"].to(torch.int8))
        self.add_tensor(f"{p}.vtq.suh", m["suh"].to(torch.float16))
        self.quantized.append(p)
        return len(data) + 16 * len(rows) + 2 * M + M * (K // TILE) + 2 * K

    def finalize(self, header: dict):
        header = dict(header, tensors=self.tensors)
        raw = json.dumps(header, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        assert len(raw) <= self.reserve, f"header of {len(raw)} B exceeds reserve {self.reserve} B"
        self.f.seek(0)
        self.f.write(MAGIC + struct.pack("<IQ", VERSION, self.reserve) + raw + b" " * (self.reserve - len(raw)))
        self.f.close()


class VtqFile:
    """Memory-mapped `.vtq` v2 reader and full-matrix decoder."""

    def __init__(self, path: Path):
        with open(path, "rb") as f:
            magic, (ver, H) = f.read(4), struct.unpack("<IQ", f.read(12))
            assert magic == MAGIC and ver == VERSION, (magic, ver)
            self.header = json.loads(f.read(H).decode("utf-8"))
        self.data_off = 16 + H
        assert self.data_off % 4096 == 0
        self.mm = np.memmap(path, dtype=np.uint8, mode="r")
        q = self.header["quantization"]
        self.formats = [format_from_json(d) for d in q["formats"]]
        self.rot_block = int(q.get("rot_block", -1))

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

    def decode_matrix(self, p: str, device="cuda") -> torch.Tensor:
        d_row, e, suh = (self.tensor(f"{p}.vtq.{k}").to(device) for k in ("d_row", "e", "suh"))
        M, K = d_row.shape[0], suh.shape[0]
        data = self.raw(f"{p}.vtq.data")
        G = torch.empty((M, K), device=device)
        for r0, r1, fi, off in self.tensor(f"{p}.vtq.slices").tolist():
            G[r0:r1] = decode_slice(data[off:], r1 - r0, K, self.formats[fi], device)
        return assemble_vtq2(G, d_row, e, suh, rot_block=self.rot_block)


# ------------------------------------------------------------------------------------------------
# Self-Test (`python -m src.vtq_v2_format`)
# ------------------------------------------------------------------------------------------------

def self_test(n_tiles: int = 256, seed: int = 0, L: int = TILE, tables_path: Path | None = None) -> dict:
    from src.tcq_proto import viterbi_ub_cuda

    torch.manual_seed(seed)
    fmts = load_vtq2_formats(tables_path=tables_path)
    res = {}
    for name, fmt in fmts.items():
        if L != TILE and fmt.sub != 16:
            continue
        x = torch.randn(n_tiles, L, device="cuda")
        gains = torch.tensor(fmt.gain_tab, device="cuda")
        n_cand = 4 if fmt.h > 2 else 0
        rec, gi, idx = viterbi_ub_cuda(
            x,
            torch.tensor(fmt.lut, device="cuda"),
            fmt.K,
            NU,
            fmt.masks[0],
            gains,
            sub=fmt.sub,
            return_code=True,
            tail_biting=True,
            m_extra=fmt.masks[1:],
            pair_parity=fmt.frac,
            n_cand=n_cand,
        )
        rec_np = rec.cpu().numpy()
        gi_np = gi.cpu().numpy().astype(np.uint32)
        idx_np = idx.cpu().numpy().astype(np.uint32)
        bad_lab = bad_frac = bad_dec = 0
        for n in range(n_tiles):
            lab, u = idx_np[n] // fmt.U, idx_np[n] % fmt.U
            if not np.array_equal(labels_from_states(states_from_codes(lab & 1), fmt), lab):
                bad_lab += 1
            lsb = u & 1
            if fmt.frac == 1 and np.any(lsb[0::2] != lsb[1::2]):
                bad_frac += 1
            elif fmt.frac == 2 and np.any(lsb.reshape(-1, 4).sum(axis=1) & 1):
                bad_frac += 1
            elif fmt.frac == 3 and np.any(lsb.reshape(-1, 4) != lsb[0::4, None]):
                bad_frac += 1
            elif fmt.frac == 4:
                bytes8 = (lsb.reshape(-1, 8).astype(np.uint32) << np.arange(8, dtype=np.uint32)).sum(axis=1)
                if np.any(CW8_TO_IDX[bytes8] < 0):
                    bad_frac += 1
            if not np.array_equal(decode_tile(pack_tile(lab, u, gi_np[n], fmt), fmt), rec_np[n]):
                bad_dec += 1
        assert bad_lab == 0 and bad_frac == 0 and bad_dec == 0, (name, L, bad_lab, bad_frac, bad_dec)
        res[name] = {
            "tile_bytes": fmt.tile_bytes(L),
            "bpw_k4096": round(fmt.bpw(4096), 4),
            "label_mismatch": bad_lab,
            "frac_violations": bad_frac,
            "decode_mismatch": bad_dec,
        }
        print(
            f"{name:10s} L={L:3d} {fmt.tile_bytes(L):3d} B, {fmt.bpw(4096):.4f} bpw | "
            f"mismatches: labels {bad_lab}, frac {bad_frac}, decode {bad_dec} / {n_tiles}",
            flush=True,
        )
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", type=Path, default=None, help="Verify an existing .vtq v2 file")
    ap.add_argument("--n", type=int, default=4)
    args = ap.parse_args()
    if args.check is not None:
        vf = VtqFile(args.check)
        q = vf.header["quantization"]
        print(
            f"{args.check}: {args.check.stat().st_size / 1e9:.2f} GB, tensors {len(vf.header['tensors'])}, "
            f"formats {len(q['formats'])}, bpw {q.get('bits_per_weight')}"
        )
        names = sorted({k[: -len(".vtq.data")] for k in vf.header["tensors"] if k.endswith(".vtq.data")})
        for p in names[:: max(1, len(names) // args.n)][: args.n]:
            W = vf.decode_matrix(p)
            print(f"{p}: {tuple(W.shape)}, rms {W.square().mean().sqrt():.4g}, finite {bool(torch.isfinite(W).all())}")
        return

    out = {"tiles_256": self_test(n_tiles=256, L=256), "rings_160": self_test(n_tiles=128, L=160)}
    out_path = Path(__file__).resolve().parent.parent / "results" / "vtq_v2_format_selftest.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Saved self-test report to {out_path}", flush=True)


if __name__ == "__main__":
    main()
