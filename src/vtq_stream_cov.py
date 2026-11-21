"""Single-use covariance accumulator for the layerwise VTQ encoder."""
import torch

from src.vtq_v2 import DualCovAccumulator


class ConsumableCov(DualCovAccumulator):
    """Same FP32 batch products, FP64 sums and damped solve as DualCovAccumulator.

    get() consumes the sufficient statistics: each FP64 buffer can be freed as
    soon as its normalized FP32 matrix is materialized, before the dense solve.
    Callers which need to try more than one damping must use DualCovAccumulator.
    """

    @staticmethod
    def _update(total, batch):
        return batch.double() if total is None else total.add_(batch)

    @torch.no_grad()
    def add(self, x_q, x_ref=None):
        if self.n_tokens < 0:
            raise RuntimeError('cannot add after finalization')
        x = x_q.reshape(-1, x_q.shape[-1]).float()
        self.sum_xx = self._update(self.sum_xx, x.T @ x)
        if self.need_cross and x_ref is not None:
            err = x_ref.reshape_as(x).float() - x
            self.sum_qc = self._update(self.sum_qc, x.T @ err)
        self.n_tokens += len(x)

    @torch.no_grad()
    def get(self, gptaq_damp=.20):
        if self.sum_xx is None or self.n_tokens <= 0:
            raise RuntimeError('empty or already finalized')
        n, self.n_tokens = self.n_tokens, -1
        h = self.sum_xx.div_(n).float()
        self.sum_xx = None
        h = (h + h.T).mul_(.5)
        if self.sum_qc is None:
            return h, None
        c = self.sum_qc.div_(n).float()
        self.sum_qc = None
        reg = h.clone()
        reg.diagonal().add_(gptaq_damp * reg.diagonal().mean().clamp(min=1e-8))
        return h, torch.linalg.solve(reg, c).contiguous()
