"""Exercise the production matrix encoder (including its uniform-slice fast path)."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch

from src import vtq
from src.vtq_format import VtqFile, VtqWriter, format_json_vtq, load_vtq_formats


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA Viterbi required')
class MatrixRoundTrip(unittest.TestCase):
    def test_uniform_and_mixed(self):
        rng = torch.Generator(device='cuda').manual_seed(121)
        w = torch.randn(48, 512, device='cuda', generator=rng)
        x = torch.randn(256, 512, device='cuda', generator=rng)
        h = x.T @ x / len(x)
        formats = list(load_vtq_formats(menu=['VTQ_250', 'VTQ_300']).values())
        # Non-adjacent slices of the same format also exercise the gather path.
        for specs in ('VTQ_300', [(0, 16, 'VTQ_300'), (16, 32, 'VTQ_250'), (32, 48, 'VTQ_300')]):
            with self.subTest(specs=specs), TemporaryDirectory() as td, patch.object(vtq, 'PACK', {}):
                _, q, _ = vtq.quantize_matrix_vtq(w, h, specs)
                path = Path(td) / 'matrix.vtq'
                writer = VtqWriter(path, formats)
                writer.add_matrix('w', vtq.PACK['matrix'], verify=True)
                writer.finalize({'quantization': {'formats': [format_json_vtq(f) for f in formats]}})
                reader = VtqFile(path)
                try:
                    self.assertTrue(torch.equal(q, reader.decode_matrix('w')))
                finally:
                    reader.mm.close()
                    reader.f.close()
                with patch.object(vtq, 'PACK', None):
                    _, unpacked, _ = vtq.quantize_matrix_vtq(w, h, specs)
                self.assertTrue(torch.equal(q, unpacked))


if __name__ == '__main__':
    unittest.main()
