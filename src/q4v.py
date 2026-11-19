"""Experimental Q4V1 folded 4-D vector codec. Independent of the VTQ wire format.

One 256-weight row tile = 64 centroid IDs + 32 sign bytes + 4 gain bytes +
1 scale byte. All integers in the file are little endian. GPU search grid is
encoder-only; the independent NumPy decoder only needs the serialized bytes.
"""

from dataclasses import dataclass
from pathlib import Path
import struct

import numpy as np
import torch
import triton
import triton.language as tl

from src.iqk_format import make_rademacher_signs
from src.vtq import compute_suh_vtq
from src.vtq_format import apply_vtq_rot

HEADER = struct.Struct("<4sII")
MAGIC = b"Q4V1"
GAINS = (0.6875, 0.875, 1.0625, 1.375)


@torch.no_grad()
def train_book(device="cuda", seed=711, steps=48):
    """Offline Lloyd updates under a joint per-16 gain; never sees model weights."""
    rng = torch.Generator(device=device).manual_seed(seed)
    book = torch.randn(256, 4, generator=rng, device=device).abs()
    for _ in range(steps):
        x = torch.randn(512, 256, generator=rng, device=device)
        x = (x / x.square().mean(1, keepdim=True).sqrt()).reshape(-1, 4).abs()
        c = (book * 64).round().clamp(0, 255) / 64
        dot = x @ c.T
        c2 = c.square().sum(1)
        best_loss = torch.full((len(x) // 4,), float("inf"), device=device)
        best_id = torch.zeros(len(x), dtype=torch.long, device=device)
        best_g = torch.ones(len(x), device=device)
        for g in GAINS:
            dist, idx = ((g * g) * c2 - (2 * g) * dot).min(dim=1)
            loss = dist.view(-1, 4).sum(1)
            take = loss < best_loss
            best_loss = torch.minimum(loss, best_loss)
            take = take.repeat_interleave(4)
            best_id = torch.where(take, idx, best_id)
            best_g = torch.where(take, g, best_g)
        num = torch.zeros_like(book).index_add_(0, best_id, x * best_g[:, None])
        den = torch.zeros(256, device=device).index_add_(0, best_id, best_g.square())
        book = torch.where(den[:, None] > 0, num / den[:, None].clamp_min(1e-12), book)
    return (book * 64).round().clamp(0, 255).byte().cpu().numpy()


@triton.jit
def _pack_tiles(
    X, D, BOOK, GRID, OUT, RECON, NB: tl.constexpr, WRITE_REC: tl.constexpr
):
    tile = tl.program_id(0)
    v = tl.arange(0, 64)
    off = tile * 256 + 4 * v
    x0, x1 = tl.load(X + off), tl.load(X + off + 1)
    x2, x3 = tl.load(X + off + 2), tl.load(X + off + 3)
    rowstep = tl.load(D + tile // NB).to(tl.float32) / 128.0
    rms = tl.sqrt(tl.sum(x0 * x0 + x1 * x1 + x2 * x2 + x3 * x3, 0) / 256.0)
    u = tl.minimum(255.0, tl.maximum(1.0, tl.floor(rms / rowstep + 0.5)))
    inv = 1.0 / (rowstep * u)
    a, b, c, d = tl.abs(x0) * inv, tl.abs(x1) * inv, tl.abs(x2) * inv, tl.abs(x3) * inv
    groupbest = tl.full((16,), float("inf"), tl.float32)
    gainid = tl.full((16,), 0, tl.int32)
    ids = tl.full((64,), 0, tl.int32)
    for gi in tl.static_range(4):
        gain = (0.6875, 0.875, 1.0625, 1.375)[gi]
        qa = tl.minimum((a * (4.0 / gain)).to(tl.int32), 15)
        qb = tl.minimum((b * (4.0 / gain)).to(tl.int32), 15)
        qc = tl.minimum((c * (4.0 / gain)).to(tl.int32), 15)
        qd = tl.minimum((d * (4.0 / gain)).to(tl.int32), 15)
        cand = tl.load(GRID + qa + 16 * qb + 256 * qc + 4096 * qd)
        best = tl.full((64,), float("inf"), tl.float32)
        idx = tl.full((64,), 0, tl.int32)
        for j in tl.static_range(4):
            ix = (cand >> (8 * j)) & 255
            w = tl.load(BOOK + ix)
            e0 = a - (w & 255) * (gain / 64.0)
            e1 = b - ((w >> 8) & 255) * (gain / 64.0)
            e2 = c - ((w >> 16) & 255) * (gain / 64.0)
            e3 = d - ((w >> 24) & 255) * (gain / 64.0)
            err = e0 * e0 + e1 * e1 + e2 * e2 + e3 * e3
            idx = tl.where(err < best, ix, idx)
            best = tl.minimum(err, best)
        err16 = tl.sum(tl.reshape(best, (16, 4)), 1)
        take = err16 < groupbest
        ids = tl.reshape(
            tl.where(take[:, None], tl.reshape(idx, (16, 4)), tl.reshape(ids, (16, 4))),
            (64,),
        )
        gainid = tl.where(take, gi, gainid)
        groupbest = tl.minimum(groupbest, err16)
    # Refit the 8-bit tile scale for the selected codes, without another search.
    gc = tl.reshape(tl.broadcast_to(gainid[:, None], (16, 4)), (64,))
    gg = (
        tl.where(
            gc == 0, 0.6875, tl.where(gc == 1, 0.875, tl.where(gc == 2, 1.0625, 1.375))
        )
        / 64.0
    )
    w = tl.load(BOOK + ids)
    m0, m1 = (w & 255) * gg, ((w >> 8) & 255) * gg
    m2, m3 = ((w >> 16) & 255) * gg, ((w >> 24) & 255) * gg
    num = tl.sum(
        tl.abs(x0) * m0 + tl.abs(x1) * m1 + tl.abs(x2) * m2 + tl.abs(x3) * m3, 0
    )
    den = tl.sum(m0 * m0 + m1 * m1 + m2 * m2 + m3 * m3, 0)
    u = tl.minimum(
        255.0, tl.maximum(0.0, tl.floor(num / tl.maximum(den * rowstep, 1e-20) + 0.5))
    )
    sign = (
        (x0 < 0).to(tl.int32)
        + 2 * (x1 < 0).to(tl.int32)
        + 4 * (x2 < 0).to(tl.int32)
        + 8 * (x3 < 0).to(tl.int32)
    )
    base = OUT + tile * 101
    tl.store(base + v, ids)
    tl.store(
        base + 64 + tl.arange(0, 32),
        tl.sum(tl.reshape(sign, (32, 2)) << (4 * tl.arange(0, 2)[None, :]), 1),
    )
    tl.store(
        base + 96 + tl.arange(0, 4),
        tl.sum(tl.reshape(gainid, (4, 4)) << (2 * tl.arange(0, 4)[None, :]), 1),
    )
    tl.store(base + 100, u)
    if WRITE_REC:
        scale = rowstep * u
        tl.store(RECON + off, tl.where(x0 < 0, -m0, m0) * scale)
        tl.store(RECON + off + 1, tl.where(x1 < 0, -m1, m1) * scale)
        tl.store(RECON + off + 2, tl.where(x2 < 0, -m2, m2) * scale)
        tl.store(RECON + off + 3, tl.where(x3 < 0, -m3, m3) * scale)


@dataclass
class Packed:
    tiles: torch.Tensor  # uint8 [M,K/256,101]
    row: torch.Tensor  # fp16 [M]
    suh: torch.Tensor  # fp16 [K]

    def serialize(self, book):
        m, k = self.row.numel(), self.suh.numel()
        return b"".join(
            (
                HEADER.pack(MAGIC, m, k),
                np.asarray(book, np.uint8).reshape(256, 4).tobytes(),
                self.row.cpu().numpy().astype("<f2").tobytes(),
                self.suh.cpu().numpy().astype("<f2").tobytes(),
                self.tiles.cpu().numpy().tobytes(),
            )
        )


class Encoder:
    def __init__(self, book, device="cuda"):
        self.book = np.asarray(book, np.uint8).reshape(256, 4).copy()
        c = torch.tensor(self.book, device=device).float() / 64
        self.words = torch.tensor(self.book.view("<i4").reshape(-1), device=device)
        # Offline inverse map: top four IDs at the centre of each 4-D grid cell.
        grid = []
        shifts = torch.arange(4, device=device) * 4
        for q in torch.arange(65536, device=device).split(2048):
            centre = (((q[:, None] >> shifts) & 15) + 0.5) / 4
            ix = (c.square().sum(1) - 2 * (centre @ c.T)).topk(4, largest=False).indices
            grid.append((ix << (2 * shifts)).sum(1).int())
        self.grid = torch.cat(grid)

    @torch.no_grad()
    def encode(self, w, hdiag, seed=1337):
        m, k = w.shape
        if k % 256 or not w.is_cuda or hdiag.shape != (k,):
            raise ValueError("expected CUDA W[M, K], 256 | K and input variances [K]")
        # Same family of signed channel scaling and rotation as VTQ, but no
        # K x K covariance, Cholesky, feedback or trellis workspace.
        w = w.float()
        col = w.norm(dim=0).div_(m**0.5).clamp_min_(1e-6)
        h = hdiag.float().clamp_min(1e-8).sqrt()
        s = ((col / col.mean()).sqrt() * (h / h.mean()).pow(-0.25)).clamp_(0.25, 4.0)
        suh = (make_rademacher_signs(k, seed=seed, device=w.device) * s).half()
        rot = apply_vtq_rot(w / suh.float(), transpose=False).contiguous()
        rms = rot.view(m, -1, 256).norm(dim=-1) / 16
        row = (rms.amax(dim=1) * (128.0 / 224.0)).clamp_min_(1e-7).half()
        out = torch.empty((m, k // 256, 101), dtype=torch.uint8, device=w.device)
        _pack_tiles[(m * (k // 256),)](
            rot,
            row,
            self.words,
            self.grid,
            out,
            rot,
            k // 256,
            False,
            enable_fp_fusion=False,
        )
        return Packed(out, row, suh)

    @torch.no_grad()
    def encode_hessian(self, w, h, seed=1337):
        """Same bytes. Spend more encoder time to feed errors into later tiles
        using the calibration covariance, then fit the existing fp16 scales."""
        m, k = w.shape
        if k % 256 or h.shape != (k, k):
            raise ValueError("expected W[M,K], 256 | K and full H[K,K]")
        rot, hr, suh = compute_suh_vtq(w.float(), h.float(), seed=seed)
        hr.diagonal().add_(0.2 * hr.diagonal().mean().clamp_min(1e-8))
        l = torch.linalg.cholesky(hr.flip((0, 1)))
        del hr
        u = torch.linalg.solve_triangular(
            l, torch.eye(k, device=w.device), upper=False
        ).flip((0, 1))
        del l
        n = k // 256
        # Diagonal blocks of U^{-1}; needed to propagate errors across tile boundaries.
        diag = torch.stack([u[j : j + 256, j : j + 256] for j in range(0, k, 256)])
        ui = torch.linalg.solve_triangular(
            diag, torch.eye(256, device=w.device).expand(n, -1, -1), upper=True
        )
        row = (
            (rot.view(m, n, 256).norm(dim=-1).amax(dim=1) * (8.0 / 224.0))
            .clamp_min_(1e-7)
            .half()
        )
        out = torch.empty((m, n, 101), dtype=torch.uint8, device=w.device)
        for j in range(n):
            s = 256 * j
            x = rot[:, s : s + 256].contiguous()
            y = torch.empty_like(x)
            block = torch.empty((m, 101), device=w.device, dtype=torch.uint8)
            _pack_tiles[(m,)](
                x, row, self.words, self.grid, block, y, 1, True, enable_fp_fusion=False
            )
            out[:, j] = block
            if s + 256 < k:
                rot[:, s + 256 :].sub_(((x - y) @ ui[j]) @ u[s : s + 256, s + 256 :])
            rot[:, s : s + 256] = y
        del u, ui, diag
        # Alternating diagonal least squares with frozen vector IDs and gains.
        basis = apply_vtq_rot(rot, transpose=True)
        q = basis * suh
        hw = w.float() @ h
        hq = q @ h
        c = (q * hw).sum(1) / (q * hq).sum(1).clamp_min(1e-12)
        row_try = (row.float() * torch.where((c > 0.5) & (c < 2), c, 1.0)).half()
        b = basis * (row_try.float() / row.float())[:, None]
        a = (b.T @ b).mul_(h)
        a.diagonal().add_(1e-3 * a.diagonal().mean().clamp_min(1e-12))
        sol, info = torch.linalg.solve_ex(a, (b * hw).sum(0))
        st = sol.half()
        st = torch.where(
            torch.isfinite(st) & (st.abs() > 0.05) & (st.abs() < 10), st, suh.half()
        )
        qt = b * st.float()
        err = w - qt
        if info.item() == 0 and ((err @ h) * err).sum() < ((w - q) * (hw - hq)).sum():
            row, suh = row_try, st
        return Packed(out, row, suh.half())


def decode(blob, device="cpu"):
    """Independent unpacker; no encoder, grid, floats cached from encoding, or fitted seeds."""
    if len(blob) < HEADER.size:
        raise ValueError("truncated Q4V1 header")
    magic, m, k = HEADER.unpack_from(blob)
    if (
        magic != MAGIC
        or not m
        or not k
        or k % 256
        or len(blob) != HEADER.size + 1024 + 2 * (m + k) + m * (k // 256) * 101
    ):
        raise ValueError("invalid Q4V1 matrix")
    book = np.frombuffer(blob, np.uint8, 1024, HEADER.size).reshape(256, 4)
    p = HEADER.size + 1024
    row = np.frombuffer(blob, "<f2", m, p).astype(np.float32)
    s = np.frombuffer(blob, "<f2", k, p + 2 * m).astype(np.float32)
    tiles = np.frombuffer(blob, np.uint8, offset=p + 2 * (m + k)).reshape(m, -1, 101)
    mag = book[tiles[:, :, :64]].reshape(m, -1, 256).astype(np.float32) / 64
    signs = np.unpackbits(tiles[:, :, 64:96], axis=-1, bitorder="little")
    gi = (
        (tiles[:, :, 96:100, None] >> np.arange(0, 8, 2, dtype=np.uint8)) & 3
    ).reshape(m, -1, 16)
    scale = (row[:, None] * (tiles[:, :, 100].astype(np.float32) / 128))[:, :, None]
    rot = (
        np.where(signs, -mag, mag)
        * np.repeat(np.asarray(GAINS, np.float32)[gi], 16, axis=-1)
        * scale
    )
    return apply_vtq_rot(
        torch.from_numpy(rot.reshape(m, k)).to(device), transpose=True
    ) * torch.from_numpy(s).to(device)


if __name__ == "__main__":
    Path("results/q4v_book.bin").write_bytes(train_book().tobytes())
