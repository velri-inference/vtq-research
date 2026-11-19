import unittest
from pathlib import Path

import numpy as np
import torch

from src import q4v
from src.vtq_format import apply_vtq_rot


class Q4V1Bytes(unittest.TestCase):
    def test_known_tile(self):
        # All 16 per-vector sign masks, four gain values, and both ID extremes.
        book = np.zeros((256, 4), np.uint8)
        book[0] = [2, 8, 16, 64]
        book[255] = [3, 12, 24, 96]
        idx = bytes([0, 255] * 32)
        sign = bytes(i | ((i + 1) << 4) for i in list(range(0, 16, 2)) * 4)
        gains = bytes([0b11100100] * 4)
        blob = (
            q4v.HEADER.pack(b"Q4V1", 1, 256)
            + book.tobytes()
            + np.full(1, 0.5, "<f2").tobytes()
            + np.ones(256, "<f2").tobytes()
            + idx
            + sign
            + gains
            + bytes([192])
        )
        expect = []
        for v, ix in enumerate(idx):
            for j in range(4):
                polarity = -1 if (v % 16) & (1 << j) else 1
                expect.append(
                    float(book[ix, j]) / 64 * polarity * q4v.GAINS[(v // 4) % 4] * 0.75
                )
        ref = apply_vtq_rot(torch.tensor(expect).view(1, 256), transpose=True)
        self.assertTrue(torch.equal(q4v.decode(blob), ref))
        for bad in (blob[:4], blob[:-1], blob + b"\0", b"NOPE" + blob[4:]):
            with self.assertRaises(ValueError):
                q4v.decode(bad)

    @unittest.skipUnless(torch.cuda.is_available(), "requires Triton/CUDA")
    def test_pack_kernel_matches_independent_decoder(self):
        book = np.fromfile(
            Path(__file__).resolve().parents[1] / "results/q4v_book.bin", np.uint8
        ).reshape(256, 4)
        enc = q4v.Encoder(book)
        m, k = 3, 512
        x = torch.randn(
            m,
            k,
            device="cuda",
            generator=torch.Generator(device="cuda").manual_seed(12),
        )
        x[0, :256] = 0
        row = torch.tensor([0.5, 0.75, 1.5], device="cuda").half()
        suh = torch.linspace(-1, 2, k, device="cuda").half()
        tiles = torch.empty((m, k // 256, 101), dtype=torch.uint8, device="cuda")
        recon = torch.empty_like(x)
        q4v._pack_tiles[(m * k // 256,)](
            x,
            row,
            enc.words,
            enc.grid,
            tiles,
            recon,
            k // 256,
            True,
            enable_fp_fusion=False,
        )
        blob = q4v.Packed(tiles, row, suh).serialize(book)
        self.assertEqual(len(blob), 12 + 1024 + 2 * (m + k) + m * k // 256 * 101)
        self.assertEqual(int(tiles[0, 0, 100]), 0)
        self.assertTrue(
            torch.equal(
                apply_vtq_rot(recon, transpose=True) * suh, q4v.decode(blob, "cuda")
            )
        )


if __name__ == "__main__":
    unittest.main()
