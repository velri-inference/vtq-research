"""G4 packed matrix A/B. Prepare src/vtq_baseline.py as in bench_vtq_encoder_ab.py."""
import json
from pathlib import Path
import time

import torch

from src import vtq, vtq_baseline


@torch.inference_mode()
def main():
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    report = dict(gpu=torch.cuda.get_device_name(), spec='VTQ_300', pack=True, seed=42, cases=[])
    for m, k in ((768, 3072), (8192, 3072), (3072, 8192)):
        w = torch.randn(m, k, device='cuda') * 0.02
        x = torch.randn(1024, k, device='cuda')
        h = x.T @ x / len(x)
        del x
        case = dict(shape=[m, k], runs=[])
        report['cases'].append(case)
        for rep in range(5):
            for enc in ((vtq_baseline, vtq) if rep % 2 == 0 else (vtq, vtq_baseline)):
                enc.PACK = {}
                torch.cuda.synchronize()
                base = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                start = time.perf_counter()
                _, q, loss = enc.quantize_matrix_vtq(w, h, report['spec'])
                torch.cuda.synchronize()
                seconds = time.perf_counter() - start
                peak = torch.cuda.max_memory_allocated() - base
                if rep == 0:  # prime both encoders, compare the full FP32 reconstruction outside timing
                    if enc is vtq_baseline:
                        ref = q.cpu()
                    else:
                        assert torch.equal(ref, q.cpu())
                        del ref
                else:
                    case['runs'].append(dict(encoder=enc.__name__, s=seconds, peak_extra_bytes=peak, rel_loss=loss))
                enc.PACK = None
                del q
        del w, h
    Path('results/vtq_g4_pack_shapes_ab.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
