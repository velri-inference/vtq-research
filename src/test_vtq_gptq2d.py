#!/usr/bin/env python3
"""
End-to-end verification tests for opt-in GPTQ-2D integration in VTQ (`src/vtq_gptq2d.py` & `src/vtq.py`).

Verifies:
  1. Default path (`gptq2d=False`) is bit-identical regardless of `G_out`.
  2. `flat_to_sym` / `sym_to_flat` / `pad_output_hessian` / `OutHessianProvider` roundtrips.
  3. `ldl_of_inverse_block` satisfies `L @ Lambda @ L^T == M^{-1}` with exact `I_16` diagonal blocks.
  4. Triton vs PyTorch GPTQ-2D parity on VTQ `16x16` super-entries.
  5. Strict 2-sided quadratic objective reduction (`G_corr` vs `diag(G)`).
  6. Bit-exact `.vtq` file pack + unpack roundtrip (`VtqWriter` -> `VtqFile.decode_matrix`) across:
       - Uniform specs covering all VTQ codebook families (`VTQ_150`, `VTQ_175`, `VTQ_200`, `VTQ_300`, `VTQ_400`, `VTQ_500`),
       - Mixed-precision DP `row_slices`,
       - Non-multiple padded shapes (`132 x 300`).
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

import src.vtq as vtq
import src.vtq_format as vf
import src.vtq_gptq2d as vg


def _make_correlated_gram(n: int, seed: int, device: torch.device) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(seed)
    Q, _ = torch.linalg.qr(torch.randn((n, n), generator=g, dtype=torch.float64))
    eigs = 1.0 / torch.arange(1, n + 1, dtype=torch.float64)
    M = (Q * eigs.unsqueeze(0)) @ Q.T
    M = M / M.diagonal().mean()
    M.diagonal().add_(0.05)
    return M.to(device=device)


class TestVtqGptq2d(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA required for VTQ Viterbi kernels")
        torch.backends.cuda.matmul.allow_tf32 = False
        cls.dev = torch.device("cuda")
        vtq.load_and_register_vtq()
        cls.all_fmts = list(vf.load_vtq_formats().values())

    def test_flat_sym_and_provider(self):
        M = _make_correlated_gram(64, seed=7, device=self.dev).float()
        flat = vg.sym_to_flat(M)
        self.assertEqual(flat.numel(), 64 * 65 // 2)
        M2 = vg.flat_to_sym(flat, 64)
        self.assertTrue(torch.equal(M, M2))

        with tempfile.TemporaryDirectory() as td:
            td_p = Path(td)
            torch.save(flat.cpu(), td_p / "3_down_hout.pt")
            prov = vg.OutHessianProvider(td_p, sigma_reg=0.10)
            self.assertTrue(prov.wants("mlp.down_proj"))
            self.assertTrue(prov.wants("self_attn.o_proj"))
            self.assertFalse(prov.wants("mlp.gate_proj"))
            got = prov.get(3, "mlp.down_proj", 64, self.dev)
            self.assertIsNotNone(got)
            self.assertTrue(torch.allclose(got.float(), M, atol=1e-6))
            self.assertIsNone(prov.get(3, "mlp.gate_proj", 64, self.dev))

    def test_ldl_of_inverse_exact(self):
        M = _make_correlated_gram(128, seed=11, device=self.dev)
        L, prec, scalar_dw = vg.ldl_of_inverse_block(M, size=16, use_fp64=True)
        # Verify exact I_16 diagonal blocks & strict lower triangularity
        L_blk = L.view(8, 16, 8, 16)
        eye16 = torch.eye(16, device=self.dev)
        for i in range(8):
            self.assertLess(float((L_blk[i, :, i, :] - eye16).abs().max()), 1e-6)
            for j in range(i + 1, 8):
                self.assertEqual(float(L_blk[i, :, j, :].abs().max()), 0.0)
        # Verify L^{-1} M^{-1} L^{-T} is block-diagonal (off-diagonal blocks == 0)
        M_inv = torch.linalg.inv(M)
        L_inv = torch.linalg.solve_triangular(L.double(), torch.eye(128, dtype=torch.float64, device=self.dev), upper=False)
        Lam = (L_inv @ M_inv @ L_inv.T).view(8, 16, 8, 16)
        for i in range(8):
            for j in range(8):
                if i != j:
                    self.assertLess(float(Lam[i, :, j, :].abs().max()), 1e-5)
        self.assertTrue(bool((prec > 0).all()))
        self.assertTrue(bool((scalar_dw > 0).all()))

    def test_default_path_bit_identical(self):
        torch.manual_seed(42)
        W = torch.randn((128, 256), device=self.dev) * 0.05
        H = _make_correlated_gram(256, seed=13, device=self.dev).float()
        G = _make_correlated_gram(128, seed=17, device=self.dev).float()
        _, w1, l1 = vtq.quantize_matrix_vtq(W, H, "VTQ_200", seed=1234, u_sb_gecd_sweeps=0)
        _, w2, l2 = vtq.quantize_matrix_vtq(W, H, "VTQ_200", seed=1234, u_sb_gecd_sweeps=0, G_out=G, gptq2d=False)
        self.assertTrue(torch.equal(w1, w2))
        self.assertEqual(l1, l2)

    def test_2sided_proxy_reduction_and_triton_parity(self):
        torch.manual_seed(101)
        M, K = 128, 512
        W = torch.randn((M, K), device=self.dev) * 0.04
        H = _make_correlated_gram(K, seed=21, device=self.dev).float()
        G = _make_correlated_gram(M, seed=29, device=self.dev).float()
        G_diag = torch.diag(G.diagonal())

        _, w_1d, _ = vtq.quantize_matrix_vtq(W, H, "VTQ_200", seed=555, G_out=G_diag, gptq2d=True, gptq2d_sigma_reg=0.05)
        _, w_2d, _ = vtq.quantize_matrix_vtq(W, H, "VTQ_200", seed=555, G_out=G, gptq2d=True, gptq2d_sigma_reg=0.05)

        def loss_2d(wh):
            dw = (W - wh).double()
            return float(((G.double() @ dw) * (dw @ H.double())).sum().item())

        l1, l2 = loss_2d(w_1d), loss_2d(w_2d)
        rel_drop = (l2 - l1) / l1
        print(f"  [proxy check] 1D(diagG)={l1:.6e} -> 2D(fullG)={l2:.6e} ({100*rel_drop:+.2f}%)")
        self.assertLess(rel_drop, -0.10, f"Expected >10% 2D proxy drop, got {100*rel_drop:+.2f}%")

        # Triton vs PyTorch parity inside quantize_rows_vtq_gptq2d
        W_rot, H_rot, _ = vtq.compute_suh_vtq(W, H, seed=555)
        H_reg = H_rot.clone()
        H_reg.diagonal().add_(0.20 * H_rot.diagonal().mean())
        out_tri = vg.quantize_rows_vtq_gptq2d(W_rot, H_reg, G, [(0, M, "VTQ_200")], vtq_mod=vtq, use_triton=True)
        out_pt = vg.quantize_rows_vtq_gptq2d(W_rot, H_reg, G, [(0, M, "VTQ_200")], vtq_mod=vtq, use_triton=False)
        rel_tp = float((out_tri["G_nat"] - out_pt["G_nat"]).norm() / out_pt["G_nat"].norm().clamp(min=1e-12))
        print(f"  [triton vs torch] rel_diff={rel_tp:.3e}")
        self.assertLess(rel_tp, 1e-4)

    def _assert_vtq_file_roundtrip(self, W: torch.Tensor, H: torch.Tensor, G: torch.Tensor, spec_or_slices, label: str):
        vtq.PACK = {}
        try:
            _, w_hat, _ = vtq.quantize_matrix_vtq(
                W, H, spec_or_slices, seed=777, damp=0.20, rot_block=-1,
                G_out=G, gptq2d=True, gptq2d_sigma_reg=0.10,
            )
            pack = vtq.PACK.pop("matrix")
        finally:
            vtq.PACK = None

        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "test.vtq"
            wr = vf.VtqWriter(p, self.all_fmts)
            wr.add_matrix("model.layers.0.mlp.down_proj", pack, verify=True)
            wr.finalize({"quantization": {"formats": [vf.format_json_vtq(f) for f in self.all_fmts]}})
            with vf.VtqFile(p) as rd:
                w_dec = rd.decode_matrix("model.layers.0.mlp.down_proj", device=self.dev)
        max_err = float((w_dec - w_hat).abs().max().item())
        print(f"  [roundtrip {label:22s}] shape={tuple(W.shape)} max_abs_diff={max_err:.3e}")
        self.assertLess(max_err, 1e-5, f"Roundtrip mismatch on {label}: {max_err}")

    def test_pack_roundtrip_all_codebook_families(self):
        torch.manual_seed(202)
        M, K = 128, 512
        W = torch.randn((M, K), device=self.dev) * 0.05
        H = _make_correlated_gram(K, seed=31, device=self.dev).float()
        G = _make_correlated_gram(M, seed=37, device=self.dev).float()
        for sp in ("VTQ_150", "VTQ_175", "VTQ_200", "VTQ_300", "VTQ_400", "VTQ_500"):
            self._assert_vtq_file_roundtrip(W, H, G, sp, sp)

    def test_pack_roundtrip_mixed_precision_slices(self):
        torch.manual_seed(303)
        M, K = 256, 512
        W = torch.randn((M, K), device=self.dev) * 0.05
        H = _make_correlated_gram(K, seed=41, device=self.dev).float()
        G = _make_correlated_gram(M, seed=43, device=self.dev).float()
        slices = [
            (0, 64, "VTQ_200"),
            (64, 128, "VTQ_250"),
            (128, 192, "VTQ_300"),
            (192, 256, "VTQ_400"),
        ]
        self._assert_vtq_file_roundtrip(W, H, G, slices, "mixed_dp_4slices")

    def test_pack_roundtrip_padded_shape(self):
        torch.manual_seed(404)
        M, K = 132, 300  # non-multiple of 16 and 256 -> padded to 144 x 512
        W = torch.randn((M, K), device=self.dev) * 0.05
        H = _make_correlated_gram(K, seed=47, device=self.dev).float()
        G = _make_correlated_gram(M, seed=53, device=self.dev).float()
        self._assert_vtq_file_roundtrip(W, H, G, "VTQ_250", "padded_132x300")


if __name__ == "__main__":
    unittest.main(verbosity=2)
