"""
Opt-in Two-Sided GPTQ-2D Rounding Engine for VTQ (`src/vtq_gptq2d.py`).

Propagates quantization rounding error across BOTH input columns (`U_H`) and output rows (`L_G`)
over VTQ's native `16x16` intra-subblock tail-biting trellis super-entries with ZERO changes
to the `.vtq` file format or inference kernels.

Based on:
  - GPTQ-2D: Cubic-Time Two-Sided Adaptive Rounding (arXiv:2607.27042v2),
    vendored MIT engine in `src/third_party_gptq2d.py`.
  - YAQA Sketch-B Kronecker output Hessian format (`flatH` lower-triangular `.pt` files).

Empirically validated in `exp32` (`results/exp32_gptq2d_8b_ab.json`) on Llama-3.1-8B-Instruct
(`VTQ_200`, 32 layers + streaming GPTAQ, applying GPTQ-2D only to the 64 `o_proj + down_proj`
matrices with `sigma_reg = 0.10`):
  - vs matched `static_1d`: `-2.60%*` WikiText-2, `-5.01%*` Nemotron, `-4.39%*` Multilingual KLD
  - vs prod `vtq_prod`:     `-2.07%`  WikiText-2, `-5.67%*` Nemotron, `-7.16%*` Multilingual KLD
  - 2.2x faster on `o_proj + down_proj` than prod VTQ (`53.8s` vs `120.9s` on L4).
"""

from __future__ import annotations

import fnmatch
import gc
import os
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

import src.third_party_gptq2d as g2d
from src.tcq_proto import viterbi_ub_cuda
from src.vtq_format import SUPER_ROWS, TILE, U_SB_UNIT

DEFAULT_GPTQ2D_SIGMA_REG = 0.10
DEFAULT_GPTQ2D_TARGETS: Tuple[str, ...] = (
    "self_attn.o_proj",
    "linear_attn.out_proj",
    "mlp.down_proj",
    "mlp.shared_expert.down_proj",
    "experts.*.down_proj",
)
FIVE_OUT_GPTQ2D_TARGETS: Tuple[str, ...] = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "linear_attn.out_proj",
    "mlp.down_proj",
    "mlp.shared_expert.down_proj",
    "experts.*.down_proj",
)

# Mapping from canonical submodule suffixes to YAQA / RelaxML short Hessian names
_SUBNAME_TO_YAQA_SHORT = {
    "q_proj": "q",
    "k_proj": "k",
    "v_proj": "v",
    "o_proj": "o",
    "out_proj": "out",
    "gate_proj": "gate",
    "up_proj": "up",
    "down_proj": "down",
}


# =================================================================================================
# 1. Symmetric Packing, Padding, Regularization, and Exact Block-LDL-of-Inverse
# =================================================================================================

def flat_to_sym(v: torch.Tensor, n: int) -> torch.Tensor:
    """Unpacks a lower-triangular `flatH` vector (or passes through an `[n, n]` matrix) to symmetric `[n, n]`."""
    if v.ndim == 2 and v.shape == (n, n):
        return (v + v.T).mul_(0.5)
    expected = n * (n + 1) // 2
    if v.numel() != expected:
        raise ValueError(f"Expected flatH of length {expected} or ({n}, {n}) matrix, got shape {tuple(v.shape)}")
    a = torch.zeros((n, n), dtype=v.dtype, device=v.device)
    idxs = torch.tril_indices(n, n, device=v.device)
    a[idxs[0], idxs[1]] = v
    a[idxs[1], idxs[0]] = v
    return a


def sym_to_flat(a: torch.Tensor) -> torch.Tensor:
    """Packs a symmetric `[n, n]` matrix into lower-triangular `flatH` vector (YAQA / RelaxML format)."""
    n = a.shape[0]
    idxs = torch.tril_indices(n, n, device=a.device)
    return a[idxs[0], idxs[1]].contiguous()


@torch.no_grad()
def pad_output_hessian(G: torch.Tensor, M_pad: int) -> torch.Tensor:
    """Pads output Hessian `G [M_orig, M_orig]` to `[M_pad, M_pad]` with uncoupled identity-scaled dummy rows."""
    M_orig = G.shape[0]
    if M_pad == M_orig:
        return G
    assert M_pad > M_orig
    G_pad = torch.zeros((M_pad, M_pad), dtype=G.dtype, device=G.device)
    G_pad[:M_orig, :M_orig] = G
    dmean = float(G.diagonal().mean().clamp(min=1e-12).item())
    G_pad[M_orig:, M_orig:].diagonal().fill_(dmean)
    return G_pad


@torch.no_grad()
def regularize_output_hessian(
    G: torch.Tensor,
    sigma_reg: float = DEFAULT_GPTQ2D_SIGMA_REG,
) -> Tuple[torch.Tensor, float]:
    """
    Symmetrizes, unit-diagonal-normalizes, and ridge-damps `G` in float64 until SPD.
    Returns `(G_reg_f64, final_sigma_reg)`.
    """
    G64 = G.to(torch.float64)
    G64 = (G64 + G64.T).mul_(0.5)
    G64.div_(G64.diagonal().mean().clamp(min=1e-12))
    step = max(float(sigma_reg), 1e-4)
    fsr = 0.0
    while True:
        G64.diagonal().add_(step)
        fsr += step
        _, info = torch.linalg.cholesky_ex(G64)
        if int(info.item()) == 0:
            break
        if fsr > 10.0:
            raise RuntimeError(f"Failed to make output Hessian SPD even at sigma_reg={fsr}")
    return G64, fsr


@torch.no_grad()
def ldl_of_inverse_block(
    M: torch.Tensor,
    size: int = SUPER_ROWS,
    use_fp64: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes unit block-lower-triangular `L` (`M^{-1} = L Lambda L^T` with `size x size` identity diagonal blocks),
    local Schur-complement precision diagonal `prec_diag = diag(Lambda^{-1})`, and scalar diagonal weights
    `scalar_dw = diag(Um)^2` (matching VTQ's `dw_diag = 1 / diag(U_nat)^2`) in one pass.
    """
    work_dt = torch.float64 if use_fp64 else torch.float32
    Mw = M.to(work_dt)
    n = Mw.shape[0]
    assert n % size == 0, f"Matrix dimension {n} must be a multiple of super-entry size {size}"
    k = n // size
    # Reverse-order Cholesky: M = Um @ Um.T with Um upper triangular
    Um = torch.linalg.cholesky(Mw.flip(dims=(0, 1))).flip(dims=(0, 1))
    del Mw
    scalar_dw = (Um.diagonal() ** 2).clamp(min=1e-12)
    eye = torch.eye(n, dtype=work_dt, device=M.device)
    L_raw = torch.linalg.solve_triangular(Um, eye, upper=True).T.contiguous()  # lower triangular = Um^{-T}
    del eye
    # Diagonal blocks of L_raw are D_j = Um_jj^{-T}, so D_j^{-1} = Um_jj^T and Lambda_j^{-1} = Um_jj @ Um_jj^T
    Um_bb = Um.reshape(k, size, k, size).diagonal(dim1=0, dim2=2).permute(2, 0, 1).contiguous()
    del Um
    prec_diag = (Um_bb * Um_bb).sum(dim=-1).reshape(n)
    # Right-multiply block-column j of L_raw by Um_bb[j].T via batched matmul [k, n, size] @ [k, size, size]
    L = torch.bmm(L_raw.reshape(n, k, size).transpose(0, 1), Um_bb.transpose(1, 2)).transpose(0, 1).reshape(n, n).contiguous()
    del L_raw, Um_bb
    L.tril_()
    L_blk = L.view(k, size, k, size)
    idx = torch.arange(k, device=M.device)
    L_blk[idx, :, idx, :] = torch.eye(size, dtype=work_dt, device=M.device)
    return L.float().contiguous(), prec_diag.float().contiguous(), scalar_dw.float().contiguous()


@torch.no_grad()
def ldl_of_inverse_block_batched(
    M: torch.Tensor,
    size: int = SUPER_ROWS,
    use_fp64: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Batched version of `ldl_of_inverse_block` for `M [B, n, n]`.
    Returns `(L [B, n, n], prec_diag [B, n], scalar_dw [B, n])` in float32.
    """
    work_dt = torch.float64 if use_fp64 else torch.float32
    Mw = M.to(work_dt)
    B, n, _ = Mw.shape
    assert n % size == 0, f"Matrix dimension {n} must be a multiple of super-entry size {size}"
    k = n // size
    Um = torch.linalg.cholesky(Mw.flip(dims=(-2, -1))).flip(dims=(-2, -1))
    del Mw
    scalar_dw = (Um.diagonal(dim1=-2, dim2=-1) ** 2).clamp(min=1e-12)
    eye = torch.eye(n, dtype=work_dt, device=M.device).expand(B, n, n)
    L_raw = torch.linalg.solve_triangular(Um, eye, upper=True).transpose(-2, -1).contiguous()
    del eye
    Um_bb = Um.reshape(B, k, size, k, size).diagonal(dim1=1, dim2=3).permute(0, 3, 1, 2).contiguous()
    del Um
    prec_diag = (Um_bb * Um_bb).sum(dim=-1).reshape(B, n)
    L = torch.bmm(
        L_raw.reshape(B, n, k, size).transpose(1, 2).reshape(B * k, n, size),
        Um_bb.transpose(-2, -1).reshape(B * k, size, size),
    ).reshape(B, k, n, size).transpose(1, 2).reshape(B, n, n).contiguous()
    del L_raw, Um_bb
    L.tril_()
    L_blk = L.view(B, k, size, k, size)
    idx = torch.arange(k, device=M.device)
    L_blk[:, idx, :, idx, :] = torch.eye(size, dtype=work_dt, device=M.device)
    return L.float().contiguous(), prec_diag.float().contiguous(), scalar_dw.float().contiguous()


# =================================================================================================
# 2. Output Hessian Loader / Provider (Local Dir, HF Repo, or Dict) & Sketch-B Builder
# =================================================================================================

def load_output_hessian_file(path: Union[str, Path], n: int, device: torch.device) -> torch.Tensor:
    """Loads a raw `.pt` output Hessian (`flatH` tensor, dict with `flatH`/`H`, or `[n, n]` tensor) into float64."""
    raw = torch.load(str(path), map_location="cpu", weights_only=True)
    if isinstance(raw, dict):
        if "flatH" in raw:
            v = raw["flatH"]
        elif "H" in raw:
            v = raw["H"]
        else:
            raise KeyError(f"Unsupported Hessian dict keys in {path}: {list(raw.keys())}")
    else:
        v = raw
    return flat_to_sym(v.to(device=device, dtype=torch.float64), n)


def _match_target(full_name: str, targets: Sequence[str]) -> bool:
    for t in targets:
        if "*" in t:
            if fnmatch.fnmatch(full_name, t) or fnmatch.fnmatch(full_name, "*." + t):
                return True
        elif full_name == t or full_name.endswith("." + t) or full_name.endswith(t):
            return True
    return False


def _short_yaqa_name(full_name: str) -> Optional[str]:
    if "experts." in full_name and full_name.endswith(".down_proj"):
        return "moe_down"
    if "shared_expert" in full_name and full_name.endswith(".down_proj"):
        return "shared_down"
    tail = full_name.rsplit(".", 1)[-1]
    return _SUBNAME_TO_YAQA_SHORT.get(tail)


class OutHessianProvider:
    """
    Resolves `(layer_idx, subname)` -> raw symmetric output Hessian `G_out [M, M]` or
    cached `(L_G, row_prec)` block-16 LDL factors for VTQ GPTQ-2D.

    `source` may be:
      - `Path` or local directory string containing `{l}_{short}_hout.pt` or `{l}_{subname}_hout.pt`
        (including layer-shared `{l}_moe_down_hout.pt` for MoE `experts.*.down_proj`),
      - HuggingFace Hub repo id string (e.g. `"relaxml/Llama-3.1-8B-Instruct-Hessians-2Sided"`),
      - In-memory mapping `Dict[Tuple[int, str], torch.Tensor]`.
    """

    def __init__(
        self,
        source: Union[str, Path, Dict[Tuple[int, str], torch.Tensor]],
        *,
        targets: Sequence[str] = DEFAULT_GPTQ2D_TARGETS,
        sigma_reg: float = DEFAULT_GPTQ2D_SIGMA_REG,
        cache_dir: Optional[Union[str, Path]] = None,
    ):
        self.targets = tuple(targets)
        self.sigma_reg = float(sigma_reg)
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self._dict: Optional[Dict[Tuple[int, str], torch.Tensor]] = None
        self._local_dir: Optional[Path] = None
        self._hf_repo: Optional[str] = None
        self._factor_cache_layer: int = -1
        self._factor_cache: Dict[Tuple[str, int, str], Tuple[torch.Tensor, torch.Tensor]] = {}

        if isinstance(source, dict):
            self._dict = source
        else:
            p = Path(source)
            if p.exists() and p.is_dir():
                self._local_dir = p
            else:
                self._hf_repo = str(source)

    def wants(self, full_name: str) -> bool:
        return _match_target(full_name, self.targets)

    def _candidate_names(self, layer_idx: int, full_name: str) -> List[str]:
        short = _short_yaqa_name(full_name)
        cand = [f"{layer_idx}_{full_name}_hout.pt"]
        if short is not None:
            cand.append(f"{layer_idx}_{short}_hout.pt")
        if full_name.endswith("out_proj") and f"{layer_idx}_o_hout.pt" not in cand:
            cand.append(f"{layer_idx}_o_hout.pt")
        return cand

    def _resolve_raw(
        self, layer_idx: int, full_name: str, M: int, device: torch.device
    ) -> Tuple[Optional[str], Optional[torch.Tensor]]:
        if not self.wants(full_name):
            return None, None
        if self._dict is not None:
            key_used = full_name
            t = self._dict.get((layer_idx, full_name))
            if t is None:
                short = _short_yaqa_name(full_name)
                if short is not None:
                    key_used = short
                    t = self._dict.get((layer_idx, short))
            if t is None:
                return None, None
            return key_used, flat_to_sym(t.to(device=device, dtype=torch.float64), M)

        cand_names = self._candidate_names(layer_idx, full_name)
        if self._local_dir is not None:
            for fn in cand_names:
                fp = self._local_dir / fn
                if fp.exists():
                    return fn, load_output_hessian_file(fp, M, device)
            return None, None

        if self._hf_repo is not None:
            from huggingface_hub import hf_hub_download

            kw = {"local_dir": str(self.cache_dir)} if self.cache_dir is not None else {}
            for fn in reversed(cand_names):  # YAQA `{l}_{short}_hout.pt` first on HF
                try:
                    fp = hf_hub_download(repo_id=self._hf_repo, filename=fn, **kw)
                    return fn, load_output_hessian_file(fp, M, device)
                except Exception:
                    continue
        return None, None

    def get(self, layer_idx: int, full_name: str, M: int, device: torch.device) -> Optional[torch.Tensor]:
        _, G = self._resolve_raw(layer_idx, full_name, M, device)
        return G

    @torch.no_grad()
    def get_factors(
        self, layer_idx: int, full_name: str, M_orig: int, device: torch.device
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Returns cached `(L_G, row_prec)` on `device` (padded to `M_pad = ceil(M_orig/16)*16`),
        caching per-layer so all 128-256 MoE experts sharing `{l}_moe_down_hout.pt` factor `G` ONCE!
        """
        if not self.wants(full_name):
            return None
        M_pad = ((M_orig + SUPER_ROWS - 1) // SUPER_ROWS) * SUPER_ROWS
        if layer_idx != self._factor_cache_layer:
            self._factor_cache.clear()
            self._factor_cache_layer = layer_idx

        # Fast path: check canonical key before touching disk
        short = _short_yaqa_name(full_name)
        dev_key = str(device)
        if self._local_dir is not None:
            for fn in self._candidate_names(layer_idx, full_name):
                ck = (fn, M_pad, dev_key)
                if ck in self._factor_cache:
                    return self._factor_cache[ck]
        elif self._dict is not None:
            for nm in (full_name, short):
                if nm is not None and (nm, M_pad, dev_key) in self._factor_cache:
                    return self._factor_cache[(nm, M_pad, dev_key)]

        resolved_key, G_raw = self._resolve_raw(layer_idx, full_name, M_orig, device)
        if G_raw is None or resolved_key is None:
            return None
        ck = (resolved_key, M_pad, dev_key)
        if ck not in self._factor_cache:
            G_pad = pad_output_hessian(G_raw, M_pad)
            del G_raw
            G_reg, _ = regularize_output_hessian(G_pad, sigma_reg=self.sigma_reg)
            del G_pad
            L_G, row_prec, _ = ldl_of_inverse_block(G_reg, size=SUPER_ROWS, use_fp64=True)
            del G_reg
            self._factor_cache[ck] = (L_G, row_prec)
        else:
            del G_raw
        return self._factor_cache[ck]


class InHessianProvider:
    """
    Resolves `(layer_idx, subname)` -> normalized KL-weighted input Hessian `H_in [K, K]` (float32)
    from a YAQA / Sketch-A directory (`{l}_{short}_hin.pt`), HF repo, or dict, and blends it with
    the quantized-stream activation covariance `H_stream`:
        H_use = blend * norm(H_in) + (1 - blend) * norm(H_stream)
    Validated in `exp35` (`blend = 0.5`): combines end-to-end KL weighting with cross-layer
    quantized-stream error compensation across all projections without altering `.vtq` v2.
    """

    def __init__(
        self,
        source: Union[str, Path, Dict[Tuple[int, str], torch.Tensor]],
        *,
        blend: float = 0.50,
        cache_dir: Optional[Union[str, Path]] = None,
    ):
        self.blend = float(blend)
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self._dict: Optional[Dict[Tuple[int, str], torch.Tensor]] = None
        self._local_dir: Optional[Path] = None
        self._hf_repo: Optional[str] = None
        if isinstance(source, dict):
            self._dict = source
        else:
            p = Path(source)
            if p.exists() and p.is_dir():
                self._local_dir = p
            else:
                self._hf_repo = str(source)

    @staticmethod
    def _norm_spd(H: torch.Tensor) -> torch.Tensor:
        H32 = H.float()
        H32 = 0.5 * (H32 + H32.T)
        dm = H32.diagonal().mean().clamp(min=1e-12)
        return H32 / dm

    def _candidate_names(self, layer_idx: int, full_name: str) -> List[str]:
        short = _short_yaqa_name(full_name)
        cand = [f"{layer_idx}_{full_name}_hin.pt"]
        if short is not None:
            cand.append(f"{layer_idx}_{short}_hin.pt")
        return cand

    @torch.no_grad()
    def get(self, layer_idx: int, full_name: str, K: int, device: torch.device) -> Optional[torch.Tensor]:
        if self._dict is not None:
            t = self._dict.get((layer_idx, full_name))
            if t is None:
                short = _short_yaqa_name(full_name)
                if short is not None:
                    t = self._dict.get((layer_idx, short))
            if t is None:
                return None
            H64 = flat_to_sym(t.to(device=device, dtype=torch.float64), K)
            H32 = self._norm_spd(H64)
            del H64
            return H32

        cand_names = self._candidate_names(layer_idx, full_name)
        if self._local_dir is not None:
            for fn in cand_names:
                fp = self._local_dir / fn
                if fp.exists():
                    H64 = load_output_hessian_file(fp, K, device)
                    H32 = self._norm_spd(H64)
                    del H64
                    return H32
            return None

        if self._hf_repo is not None:
            from huggingface_hub import hf_hub_download

            kw = {"local_dir": str(self.cache_dir)} if self.cache_dir is not None else {}
            for fn in reversed(cand_names):
                try:
                    fp = hf_hub_download(repo_id=self._hf_repo, filename=fn, **kw)
                    H64 = load_output_hessian_file(fp, K, device)
                    H32 = self._norm_spd(H64)
                    del H64
                    return H32
                except Exception:
                    continue
        return None

    @torch.no_grad()
    def blend_with(
        self,
        layer_idx: int,
        full_name: str,
        K: int,
        device: torch.device,
        H_stream: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if self.blend <= 0.0:
            return H_stream
        H_in = self.get(layer_idx, full_name, K, device)
        if H_in is None:
            return H_stream
        if H_stream is None or self.blend >= 1.0:
            return H_in
        Hs = self._norm_spd(H_stream)
        out = self.blend * H_in + (1.0 - self.blend) * Hs
        del H_in, Hs
        return out


@torch.no_grad()
def _sample_inverse_cdf_targets(logits: torch.Tensor, seed: int, n_draws: int = 2) -> torch.Tensor:
    probs = F.softmax(logits.float(), dim=-1)
    u = torch.rand((probs.shape[0], n_draws), generator=torch.Generator().manual_seed(seed)).to(logits.device)
    cdf = probs.cumsum(dim=-1)
    return torch.searchsorted(cdf, (u * cdf[:, -1:]).contiguous()).clamp(max=probs.shape[-1] - 1)


def compute_sketch_b_output_hessians(
    model: nn.Module,
    cal_tokens: torch.Tensor,
    out_dir: Union[str, Path],
    *,
    targets: Sequence[str] = DEFAULT_GPTQ2D_TARGETS,
    n_seqs: int = 128,
    seq_len: int = 1024,
    n_mc: int = 2,
) -> Dict[Tuple[int, str], Path]:
    """
    Experimental self-contained YAQA Sketch-B output Hessian builder (layer-offloaded, ~2.5 GB peak VRAM).

    Computes the exact left Kronecker marginal of the weight-gradient Fisher:
        G_l = E_seq[ (grad_W) (grad_W)^T ] = E_seq[ (g_seq^T x_seq) (g_seq^T x_seq)^T ]
    (matching Cornell-RelaxML/yaqa-quantization `custom_linear_B.py`) ONLY for `targets`
    (default: `self_attn.o_proj` and `mlp.down_proj`, where `M = d_model` is compact),
    and saves lower-triangular `flatH` `.pt` files into `out_dir/{l}_{short}_hout.pt`.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    model.to("cpu")
    torch.cuda.empty_cache()

    inner = model.model
    layers = inner.layers
    n_layers = len(layers)
    n = min(int(n_seqs), int(cal_tokens.shape[0]))
    seq_len = min(int(seq_len), int(cal_tokens.shape[1]))

    # Resolve target submodules per layer
    layer_targets: List[List[Tuple[str, str, int]]] = []
    for l in range(n_layers):
        lt = []
        for sub in targets:
            try:
                mod = layers[l].get_submodule(sub)
            except AttributeError:
                continue
            if isinstance(mod, nn.Linear):
                sn = _short_yaqa_name(sub) or sub
                lt.append((sub, sn, int(mod.out_features)))
        layer_targets.append(lt)

    inner.embed_tokens.to(dev)
    inner.norm.to(dev)
    inner.rotary_emb.to(dev)
    model.lm_head.to(dev)
    pos = inner.rotary_emb(
        torch.zeros((1, seq_len, model.config.hidden_size), device=dev, dtype=next(model.parameters()).dtype),
        torch.arange(seq_len, device=dev).unsqueeze(0),
    )

    # Stream in chunks of 16 sequences so CPU activation storage stays <= 4 GB
    chunk_seqs = 16
    accum: Dict[Tuple[int, str], torch.Tensor] = {}
    for l in range(n_layers):
        for _, sn, m_dim in layer_targets[l]:
            accum[(l, sn)] = torch.zeros((m_dim, m_dim), dtype=torch.float64, device="cpu")

    t0 = time.perf_counter()
    for s0 in range(0, n, chunk_seqs):
        s1 = min(n, s0 + chunk_seqs)
        toks = [cal_tokens[i : i + 1, :seq_len].to(dev) for i in range(s0, s1)]
        with torch.no_grad():
            inner.embed_tokens.to(dev)
            hs = [inner.embed_tokens(t) for t in toks]
            inner.embed_tokens.to("cpu")
            inputs = []
            for l in range(n_layers):
                inputs.append([h.cpu() for h in hs])
                layers[l].to(dev)
                hs_next = []
                for h in hs:
                    o = layers[l](h, position_embeddings=pos)
                    hs_next.append(o[0] if isinstance(o, tuple) else o)
                hs = hs_next
                layers[l].to("cpu")

        gs = []
        inner.norm.to(dev)
        model.lm_head.to(dev)
        for j, h in enumerate(hs):
            h = h.detach().requires_grad_(True)
            with torch.enable_grad():
                logits = model.lm_head(inner.norm(h)).reshape(-1, model.config.vocab_size)
                y = _sample_inverse_cdf_targets(logits, seed=3000 + s0 + j, n_draws=n_mc)
                loss = sum(F.cross_entropy(logits.float(), y[:, d], reduction="sum") for d in range(n_mc)) / float(n_mc)
                gs.append(torch.autograd.grad(loss, h)[0].detach().cpu())
            del logits, loss
        del hs
        inner.norm.to("cpu")
        model.lm_head.to("cpu")
        torch.cuda.empty_cache()

        for l in reversed(range(n_layers)):
            layer = layers[l].to(dev)
            lt = layer_targets[l]
            G_gpu = {sn: torch.zeros((m_dim, m_dim), dtype=torch.float64, device=dev) for _, sn, m_dim in lt}
            saved_x: Dict[str, Optional[torch.Tensor]] = {sn: None for _, sn, _ in lt}
            hooks = []
            for sub, sn, _ in lt:
                def fwd_hk(m, inp, out, sn=sn):
                    saved_x[sn] = inp[0].detach().float().reshape(-1, inp[0].shape[-1])
                    def bwd_hk(gr, sn=sn):
                        g2 = gr.detach().float().reshape(-1, gr.shape[-1])
                        x2 = saved_x[sn]
                        saved_x[sn] = None
                        # Sketch-B weight gradient outer product: grad_W = g^T x -> G += grad_W @ grad_W^T
                        gw = g2.T @ x2  # [M, K]
                        G_gpu[sn].add_((gw @ gw.T).double())
                        del gw, g2, x2
                    out.register_hook(bwd_hk)
                hooks.append(layer.get_submodule(sub).register_forward_hook(fwd_hk))

            next_gs = []
            for x_cpu, g_cpu in zip(inputs[l], gs):
                x = x_cpu.to(dev).detach().requires_grad_(True)
                g = g_cpu.to(dev)
                with torch.enable_grad():
                    o = layer(x, position_embeddings=pos)
                    o = o[0] if isinstance(o, tuple) else o
                    gx = torch.autograd.grad(o, x, g)[0]
                next_gs.append(gx.detach().cpu())
                del x, g, o, gx
            for hk in hooks:
                hk.remove()
            gs = next_gs
            inputs[l] = None
            layer.to("cpu")
            for _, sn, _ in lt:
                accum[(l, sn)].add_(G_gpu[sn].cpu())
            del G_gpu
            torch.cuda.empty_cache()
        del gs, inputs

    inner.rotary_emb.to("cpu")
    saved_paths: Dict[Tuple[int, str], Path] = {}
    for l in range(n_layers):
        for sub, sn, _ in layer_targets[l]:
            Gm = accum[(l, sn)] / float(max(n, 1))
            Gm = (Gm + Gm.T).mul_(0.5)
            out_p = out_dir / f"{l}_{sn}_hout.pt"
            torch.save(sym_to_flat(Gm.float()), out_p)
            saved_paths[(l, sub)] = out_p
    del accum
    gc.collect()
    torch.cuda.empty_cache()
    print(f"[vtq_gptq2d] Saved {len(saved_paths)} Sketch-B output Hessians ({n}x{seq_len}) to {out_dir} in {time.perf_counter()-t0:.1f}s", flush=True)
    return saved_paths


# =================================================================================================
# 3. Core VTQ `16x16` Super-Entry GPTQ-2D Matrix Quantizer (Uniform + Mixed-Precision + Exact Pack)
# =================================================================================================

def _build_stage_spec_runs(
    row_slices: List[Tuple[int, int, str]],
    M: int,
    Kin: int,
) -> List[List[Tuple[int, int, str]]]:
    """
    Precomputes contiguous `(k0, k1, spec_name)` tile runs for each coarse anti-diagonal stage
    `s = 0 .. mbar + nbar - 2` on CPU so the quantizer callback has ZERO GPU->CPU syncs.
    """
    mbar = M // SUPER_ROWS
    nbar = Kin // 16
    tile_specs: List[str] = [""] * mbar
    for r0, r1, sp in row_slices:
        assert r0 % SUPER_ROWS == 0 and r1 % SUPER_ROWS == 0, f"Row slice ({r0}, {r1}) not multiple of 16"
        for ib in range(r0 // SUPER_ROWS, r1 // SUPER_ROWS):
            tile_specs[ib] = sp
    assert all(tile_specs), "Uncovered 16-row tile in row_slices"

    stages: List[List[Tuple[int, int, str]]] = []
    for s in range(mbar + nbar - 1):
        i_lo = max(0, s - nbar + 1)
        i_hi = min(s + 1, mbar)
        runs: List[Tuple[int, int, str]] = []
        k = 0
        d = i_hi - i_lo
        while k < d:
            sp = tile_specs[i_lo + k]
            k1 = k + 1
            while k1 < d and tile_specs[i_lo + k1] == sp:
                k1 += 1
            runs.append((k, k1, sp))
            k = k1
        stages.append(runs)
    return stages


@torch.no_grad()
def quantize_rows_vtq_gptq2d(
    W_rot: torch.Tensor,                      # float32 [M, Kin] rotated weights (`M % 16 == 0, Kin % 256 == 0`)
    H_reg: torch.Tensor,                      # float32 [Kin, Kin] regularized rotated input Hessian
    G_out: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],  # [M, M] output Hessian OR pre-factored `(L_G, row_prec)`
    row_slices: List[Tuple[int, int, str]],   # [(r0, r1, spec_name), ...] covering [0, M)
    *,
    vtq_mod,                                  # `src.vtq` module (avoids circular import)
    sigma_reg: float = DEFAULT_GPTQ2D_SIGMA_REG,
    weight_mode: str = "dw",                  # `"dw"` (verified exp32 default) or `"schur"`
    emit_pack: bool = False,
    use_triton: Optional[bool] = None,
) -> Dict[str, Optional[torch.Tensor]]:
    """
    Runs two-sided `16x16` super-entry GPTQ-2D over VTQ's 1024-state tail-biting trellis.
    Handles both uniform and mixed-precision `row_slices`, and emits bit-exact `(g_sub, c_all, u_all)`
    for `VtqWriter` when `emit_pack=True`.
    """
    dev = W_rot.device
    M, Kin = W_rot.shape
    assert M % SUPER_ROWS == 0 and Kin % TILE == 0, f"Expected (M%16==0, K%256==0), got {(M, Kin)}"
    nblocks = Kin // TILE

    # 1) Regularize output Hessian G (float64) and compute block-16 feedback factors L_G, U_H
    if isinstance(G_out, tuple):
        L_G, row_prec = G_out
        assert L_G.shape == (M, M), f"Pre-factored L_G shape mismatch: {tuple(L_G.shape)} vs {(M, M)}"
        owns_lg = False
    else:
        G_reg, _ = regularize_output_hessian(G_out, sigma_reg=sigma_reg)
        L_G, row_prec, _ = ldl_of_inverse_block(G_reg, size=SUPER_ROWS, use_fp64=True)
        del G_reg
        owns_lg = True
    L_H, col_prec, dw_col = ldl_of_inverse_block(H_reg, size=16, use_fp64=False)
    U_16 = L_H.T.contiguous()
    del L_H

    # 2) Compute per-row fp16 `d_row` and static uint8 `u_sb` per slice (matching each slice's `cfg_scale`)
    d_row_all = torch.empty((M,), dtype=torch.float16, device=dev)
    u_sb_all = torch.empty((M, nblocks), dtype=torch.uint8, device=dev)
    d_eff_static = torch.empty((M, nblocks), dtype=torch.float32, device=dev)
    for r0, r1, sp in row_slices:
        cfg = vtq_mod.VTQ_CONFIG[sp]
        cfg_scale = float(cfg.get("scale", 1.0))
        rms_rb = vtq_mod._initial_tile_scales(W_rot[r0:r1], cfg_scale, sp)
        d_r_fp16 = (rms_rb.max(dim=1).values * (128.0 / 224.0)).clamp(min=1e-7).to(torch.float16)
        step_r = (d_r_fp16.float().clamp(min=1e-8) * float(U_SB_UNIT)).clamp(min=1e-12)
        u_f = torch.round(rms_rb / step_r.unsqueeze(1)).clamp(1.0, 255.0)
        d_row_all[r0:r1] = d_r_fp16
        u_sb_all[r0:r1] = u_f.to(torch.uint8)
        d_eff_static[r0:r1] = (d_r_fp16.float().unsqueeze(1) * (u_f * float(U_SB_UNIT))).clamp(min=1e-10)

    d_eff_full = d_eff_static.repeat_interleave(TILE, dim=1).contiguous()
    if weight_mode == "schur":
        rp = row_prec / row_prec.mean().clamp(min=1e-12)
        cp = col_prec / col_prec.mean().clamp(min=1e-12)
        wts_2d = (rp.unsqueeze(1) * (d_eff_full * d_eff_full) * cp.unsqueeze(0)).clamp(min=1e-12)
        wts_2d = wts_2d / wts_2d.mean().clamp(min=1e-12)
    elif weight_mode == "dw":
        wts_2d = dw_col.unsqueeze(0).expand(M, Kin).contiguous()
    else:
        raise ValueError(f"Unknown gptq2d weight_mode: {weight_mode!r}")

    # 3) Preload per-spec Viterbi tables & precompute stage runs (zero GPU->CPU sync in callback)
    tb_win = int(os.environ.get("TCQ_TB_WIN", "32"))
    spec_params = {}
    for _, _, sp in row_slices:
        if sp not in spec_params:
            cfg, tab, gains = vtq_mod._get_spec_gpu_vtq(sp, dev)
            K = int(cfg["K"])
            nl = int(cfg["nl"])
            frac = int(cfg["frac"])
            pp = vtq_mod._FRAC_TO_TCQ_PP[frac]
            m_extra = vtq_mod._NL_TO_M_EXTRA[nl]
            n_cand = 4 if int(gains.numel()) > 4 else 0
            spec_params[sp] = (K, tab, gains, pp, m_extra, n_cand)

    stage_runs = _build_stage_spec_runs(row_slices, M, Kin)
    single_spec = next(iter(spec_params)) if len(spec_params) == 1 else None

    # Exact `G_nat` buffer (and integer code buffers when `emit_pack=True`) written directly via views
    g_nat_full = torch.empty((M, Kin), dtype=torch.float32, device=dev)
    g_nat_flat = g_nat_full.reshape(-1)
    gi_full = torch.empty((M, Kin), dtype=torch.uint8, device=dev) if emit_pack else None
    idx_full = torch.empty((M, Kin), dtype=torch.int32, device=dev) if emit_pack else None
    gi_flat = gi_full.reshape(-1) if emit_pack else None
    idx_flat = idx_full.reshape(-1) if emit_pack else None
    stage_ctr = [0]

    def _run_viterbi(sp: str, x_t: torch.Tensor, w_t: torch.Tensor):
        K, tab, gains, pp, m_extra, n_cand = spec_params[sp]
        max_nt = 2048 if pp > 0 else 16384
        nt = x_t.shape[0]
        if nt <= max_nt:
            return viterbi_ub_cuda(
                x_t, tab, K, 10, 359, gains, sub=16, wts=w_t,
                n_cand=n_cand, return_code=emit_pack, tail_biting=True, tb_window=tb_win,
                m_extra=m_extra, pair_parity=pp,
            )
        rec_o = torch.empty_like(x_t)
        gi_o = torch.empty((nt, 256), dtype=torch.int32, device=dev) if emit_pack else None
        ix_o = torch.empty((nt, 256), dtype=torch.int32, device=dev) if emit_pack else None
        for t0 in range(0, nt, max_nt):
            t1 = min(nt, t0 + max_nt)
            r = viterbi_ub_cuda(
                x_t[t0:t1], tab, K, 10, 359, gains, sub=16, wts=w_t[t0:t1],
                n_cand=n_cand, return_code=emit_pack, tail_biting=True, tb_window=tb_win,
                m_extra=m_extra, pair_parity=pp,
            )
            if emit_pack:
                rec_o[t0:t1], gi_o[t0:t1], ix_o[t0:t1] = r
            else:
                rec_o[t0:t1] = r
        return (rec_o, gi_o, ix_o) if emit_pack else rec_o

    def vtq_super16_q(y_flat: torch.Tensor, select, scale_flat: torch.Tensor, wts_flat: torch.Tensor, *, out: torch.Tensor):
        s_idx = stage_ctr[0]
        stage_ctr[0] = s_idx + 1
        y_sub = select(y_flat)       # [d, 16, 16]
        sc_sub = select(scale_flat)  # [d, 16, 16]
        wt_sub = select(wts_flat)    # [d, 16, 16]
        d = y_sub.shape[0]
        x_tiles = (y_sub / sc_sub).reshape(d, 256).contiguous()
        w_tiles = wt_sub.reshape(d, 256).contiguous()

        if single_spec is not None:
            res = _run_viterbi(single_spec, x_tiles, w_tiles)
            if emit_pack:
                rec_t, gi_t, ix_t = res
                select(gi_flat).copy_(gi_t.to(torch.uint8).view(d, 16, 16))
                select(idx_flat).copy_(ix_t.view(d, 16, 16))
            else:
                rec_t = res
        else:
            runs = stage_runs[s_idx]
            assert runs[-1][1] == d, f"Antidiagonal tile count mismatch at stage {s_idx}: {runs[-1][1]} vs {d}"
            rec_t = torch.empty((d, 256), dtype=torch.float32, device=dev)
            gi_t = torch.empty((d, 256), dtype=torch.uint8, device=dev) if emit_pack else None
            ix_t = torch.empty((d, 256), dtype=torch.int32, device=dev) if emit_pack else None
            for k0, k1, sp in runs:
                res = _run_viterbi(sp, x_tiles[k0:k1], w_tiles[k0:k1])
                if emit_pack:
                    r_c, g_c, i_c = res
                    rec_t[k0:k1] = r_c
                    gi_t[k0:k1] = g_c.to(torch.uint8)
                    ix_t[k0:k1] = i_c
                else:
                    rec_t[k0:k1] = res
            if emit_pack:
                select(gi_flat).copy_(gi_t.view(d, 16, 16))
                select(idx_flat).copy_(ix_t.view(d, 16, 16))

        rec_view = rec_t.view(d, 16, 16)
        select(g_nat_flat).copy_(rec_view)
        torch.mul(rec_view, sc_sub, out=out)

    # 4) Execute blocked two-level Triton GPTQ-2D (or PyTorch fallback if Triton unavailable)
    do_triton = (g2d.HAS_TRITON and dev.type == "cuda") if use_triton is None else bool(use_triton)
    if do_triton:
        g2d.gptq_2d_super_entry(
            W_rot, L_G, U_16,
            quantizer=vtq_super16_q,
            super_entry_shape=(SUPER_ROWS, 16),
            inner_block_size=8,
            outer_block_size=64,
            quantizer_inputs=(d_eff_full.reshape(-1), wts_2d.reshape(-1)),
        )
    else:
        g2d.gptq_2d_super_entry_torch(
            W_rot, L_G, U_16,
            quantizer=vtq_super16_q,
            super_entry_shape=(SUPER_ROWS, 16),
            quantizer_inputs=(d_eff_full.reshape(-1), wts_2d.reshape(-1)),
        )
    assert stage_ctr[0] == len(stage_runs), f"Incomplete sweep: {stage_ctr[0]} vs {len(stage_runs)}"
    if owns_lg:
        del L_G
    del U_16, wts_2d, d_eff_full

    # 5) Pack `(g_sub, c_all, u_all)` bit-exactly per slice when `emit_pack=True`
    g_sub_all = c_all = u_all = None
    if emit_pack:
        g_sub_all = torch.empty((M, nblocks, 16), dtype=torch.uint8, device=dev)
        c_all = torch.empty((M, nblocks, 16, 2), dtype=torch.uint8, device=dev)
        u_all = torch.empty((M, nblocks, TILE), dtype=torch.uint8, device=dev)
        for r0, r1, sp in row_slices:
            U_sp = 1 << (int(vtq_mod.VTQ_CONFIG[sp]["K"]) - 1)
            R_sl = r1 - r0
            gi_sl = gi_full[r0:r1].view(R_sl, nblocks, TILE).reshape(R_sl * nblocks, TILE)
            ix_sl = idx_full[r0:r1].view(R_sl, nblocks, TILE).reshape(R_sl * nblocks, TILE)
            gs, ca, ua, _ = vtq_mod._code_to_vtq_pack(gi_sl, ix_sl, U_sp, return_labels=False)
            g_sub_all[r0:r1] = gs.view(R_sl, nblocks, 16)
            c_all[r0:r1] = ca.view(R_sl, nblocks, 16, 2)
            u_all[r0:r1] = ua.view(R_sl, nblocks, TILE)
        del gi_full, idx_full

    return {
        "G_nat": g_nat_full,
        "d_row": d_row_all,
        "u_sb": u_sb_all,
        "g_sub": g_sub_all,
        "c_all": c_all,
        "u_all": u_all,
    }


@torch.no_grad()
def quantize_rows_vtq_gptq2d_batched(
    W_rot: torch.Tensor,                                   # float32 [B, M, Kin]
    H_reg: torch.Tensor,                                   # float32 [B, Kin, Kin] already regularized
    G_factors: Tuple[torch.Tensor, torch.Tensor],          # shared `(L_G [M, M], row_prec [M])`
    spec_name: str,
    *,
    vtq_mod,
    weight_mode: str = "dw",
    emit_pack: bool = False,
    use_triton: Optional[bool] = None,
) -> Dict[str, Optional[torch.Tensor]]:
    """
    Batched uniform-spec `16x16` super-entry GPTQ-2D for `B` MoE experts sharing one layer-level
    `(L_G [M, M], row_prec [M])` with per-expert input Hessians `H_reg [B, Kin, Kin]`.
    Mathematically identical to calling `quantize_rows_vtq_gptq2d` on each expert independently,
    while saturating GPU SMs (~10-20x faster on 128-256 expert MoEs).
    """
    dev = W_rot.device
    B, M, Kin = W_rot.shape
    assert M % SUPER_ROWS == 0 and Kin % TILE == 0, f"Expected (M%16==0, K%256==0), got {(M, Kin)}"
    L_G, row_prec = G_factors
    assert L_G.shape == (M, M), f"Shared L_G shape mismatch: {tuple(L_G.shape)} vs {(M, M)}"
    nblocks = Kin // TILE

    L_H, col_prec, dw_col = ldl_of_inverse_block_batched(H_reg, size=16, use_fp64=False)
    U_16 = L_H.transpose(-2, -1).contiguous()
    del L_H

    cfg = vtq_mod.VTQ_CONFIG[spec_name]
    cfg_scale = float(cfg.get("scale", 1.0))
    rms_rb = vtq_mod._initial_tile_scales(W_rot.reshape(B * M, Kin), cfg_scale, spec_name)
    d_r_fp16 = (rms_rb.max(dim=1).values * (128.0 / 224.0)).clamp(min=1e-7).to(torch.float16)
    step_r = (d_r_fp16.float().clamp(min=1e-8) * float(U_SB_UNIT)).clamp(min=1e-12)
    u_f = torch.round(rms_rb / step_r.unsqueeze(1)).clamp(1.0, 255.0)
    d_row_all = d_r_fp16.view(B, M).contiguous()
    u_sb_all = u_f.to(torch.uint8).view(B, M, nblocks).contiguous()
    d_eff_static = (d_r_fp16.float().unsqueeze(1) * (u_f * float(U_SB_UNIT))).clamp(min=1e-10).view(B, M, nblocks)
    d_eff_full = d_eff_static.repeat_interleave(TILE, dim=2).contiguous()

    if weight_mode == "schur":
        rp = row_prec / row_prec.mean().clamp(min=1e-12)
        cp = col_prec / col_prec.mean(dim=1, keepdim=True).clamp(min=1e-12)
        wts_2d = (rp.view(1, M, 1) * (d_eff_full * d_eff_full) * cp.view(B, 1, Kin)).clamp(min=1e-12)
        wts_2d = wts_2d / wts_2d.mean(dim=(1, 2), keepdim=True).clamp(min=1e-12)
    elif weight_mode == "dw":
        wts_2d = dw_col.unsqueeze(1).expand(B, M, Kin).contiguous()
    else:
        raise ValueError(f"Unknown gptq2d weight_mode: {weight_mode!r}")

    tb_win = int(os.environ.get("TCQ_TB_WIN", "32"))
    cfg_g, tab, gains = vtq_mod._get_spec_gpu_vtq(spec_name, dev)
    K_sp = int(cfg_g["K"])
    nl = int(cfg_g["nl"])
    frac = int(cfg_g["frac"])
    pp = vtq_mod._FRAC_TO_TCQ_PP[frac]
    m_extra = vtq_mod._NL_TO_M_EXTRA[nl]
    n_cand = 4 if int(gains.numel()) > 4 else 0
    max_nt = 2048 if pp > 0 else 16384

    g_nat_full = torch.empty((B, M, Kin), dtype=torch.float32, device=dev)
    g_nat_flat = g_nat_full.reshape(B, M * Kin)
    gi_full = torch.empty((B, M, Kin), dtype=torch.uint8, device=dev) if emit_pack else None
    idx_full = torch.empty((B, M, Kin), dtype=torch.int32, device=dev) if emit_pack else None
    gi_flat = gi_full.reshape(B, M * Kin) if emit_pack else None
    idx_flat = idx_full.reshape(B, M * Kin) if emit_pack else None

    def _run_viterbi(x_t: torch.Tensor, w_t: torch.Tensor):
        nt = x_t.shape[0]
        if nt <= max_nt:
            return viterbi_ub_cuda(
                x_t, tab, K_sp, 10, 359, gains, sub=16, wts=w_t,
                n_cand=n_cand, return_code=emit_pack, tail_biting=True, tb_window=tb_win,
                m_extra=m_extra, pair_parity=pp,
            )
        rec_o = torch.empty_like(x_t)
        gi_o = torch.empty((nt, 256), dtype=torch.int32, device=dev) if emit_pack else None
        ix_o = torch.empty((nt, 256), dtype=torch.int32, device=dev) if emit_pack else None
        for t0 in range(0, nt, max_nt):
            t1 = min(nt, t0 + max_nt)
            r = viterbi_ub_cuda(
                x_t[t0:t1], tab, K_sp, 10, 359, gains, sub=16, wts=w_t[t0:t1],
                n_cand=n_cand, return_code=emit_pack, tail_biting=True, tb_window=tb_win,
                m_extra=m_extra, pair_parity=pp,
            )
            if emit_pack:
                rec_o[t0:t1], gi_o[t0:t1], ix_o[t0:t1] = r
            else:
                rec_o[t0:t1] = r
        return (rec_o, gi_o, ix_o) if emit_pack else rec_o

    def vtq_super16_q_batched(
        y_flat: torch.Tensor, select, scale_flat: torch.Tensor, wts_flat: torch.Tensor, *, out: torch.Tensor
    ):
        y_sub = select(y_flat)       # [B, d, 16, 16]
        sc_sub = select(scale_flat)  # [B, d, 16, 16]
        wt_sub = select(wts_flat)    # [B, d, 16, 16]
        d = y_sub.shape[1]
        x_tiles = (y_sub / sc_sub).reshape(B * d, 256).contiguous()
        w_tiles = wt_sub.reshape(B * d, 256).contiguous()
        res = _run_viterbi(x_tiles, w_tiles)
        if emit_pack:
            rec_t, gi_t, ix_t = res
            select(gi_flat).copy_(gi_t.to(torch.uint8).view(B, d, 16, 16))
            select(idx_flat).copy_(ix_t.view(B, d, 16, 16))
        else:
            rec_t = res
        rec_view = rec_t.view(B, d, 16, 16)
        select(g_nat_flat).copy_(rec_view)
        torch.mul(rec_view, sc_sub, out=out)

    do_triton = (g2d.HAS_TRITON and dev.type == "cuda") if use_triton is None else bool(use_triton)
    if do_triton:
        g2d.gptq_2d_super_entry(
            W_rot, L_G, U_16,
            quantizer=vtq_super16_q_batched,
            super_entry_shape=(SUPER_ROWS, 16),
            inner_block_size=8,
            outer_block_size=64,
            quantizer_inputs=(d_eff_full.reshape(B, M * Kin), wts_2d.reshape(B, M * Kin)),
        )
    else:
        g2d.gptq_2d_super_entry_torch(
            W_rot, L_G, U_16,
            quantizer=vtq_super16_q_batched,
            super_entry_shape=(SUPER_ROWS, 16),
            quantizer_inputs=(d_eff_full.reshape(B, M * Kin), wts_2d.reshape(B, M * Kin)),
        )
    del U_16, wts_2d, d_eff_full

    g_sub_all = c_all = u_all = None
    if emit_pack:
        U_sp = 1 << (K_sp - 1)
        gi_2d = gi_full.view(B * M * nblocks, TILE)
        ix_2d = idx_full.view(B * M * nblocks, TILE)
        gs, ca, ua, _ = vtq_mod._code_to_vtq_pack(gi_2d, ix_2d, U_sp, return_labels=False)
        g_sub_all = gs.view(B, M, nblocks, 16)
        c_all = ca.view(B, M, nblocks, 16, 2)
        u_all = ua.view(B, M, nblocks, TILE)
        del gi_full, idx_full

    return {
        "G_nat": g_nat_full,
        "d_row": d_row_all,
        "u_sb": u_sb_all,
        "g_sub": g_sub_all,
        "c_all": c_all,
        "u_all": u_all,
    }
