"""
End-to-End Paper Benchmark Suite (`src/run_paper_benchmarks.py`):
Compares 3 formats (`GGUF`, `EXL3`, `VTQ`) across 4 target bitrates (`2.5, 3.5, 4.5, 6.0` bpw)
on 3 architectures:
  1. `llama31_8b` : `Llama-3.1-8B-Instruct` (dense 8B GQA, `K_down = 14336`)
  2. `gemma4_26b` : `gemma-4-26B-A4B-it` (30-layer 128-expert MoE + shared MLP, tied 262k vocab, `K = 704 / 2112`)
  3. `qwen36_35b` : `Qwen3.6-35B-A3B` (40-layer 256-expert MoE + shared expert + Gated DeltaNet hybrid attention)

Methodology:
  - Shared domain-neutral calibration corpus (`wikitext-2-raw-v1` train, `64 x 2048 = 131,072` tokens),
    strictly disjoint from all 3 evaluation traces (`nemotron`, `multilingual`, `wikitext2` test).
  - Full end-to-end text-model quantization (including `embed_tokens` and `lm_head` per each format's rules),
    reporting exact `total_bpw = total_text_file_bytes * 8 / total_text_params`, `linear_bpw`, `quant_sec`,
    and teacher-forced `KLD`, `PPL`, and `Top-1` match vs `BF16` (`B = 2000` paired bootstrap).
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

import src.gguf_utils as gguf_utils
import src.hybrid_quant as hqm
import src.liqk_v2 as lv2
import src.streaming_quant as sqm
import src.vtq as vtq
import src.vtq_format as vf
from src.eval_kld import (
    get_scored_positions_and_targets,
    load_eval_trace,
    paired_bootstrap_diff,
)
from src.vtq_native import NativeVtqCodec

ROOT = Path(__file__).resolve().parent.parent

MODEL_SPECS = {
    "llama31_8b": {
        "hf_repos": "unsloth/Llama-3.1-8B-Instruct|NousResearch/Meta-Llama-3.1-8B-Instruct",
        "local_name": "Llama-3.1-8B-Instruct",
        "data_dir": ROOT / "data",
        "out_json": ROOT / "results" / "paper_llama31_8b.json",
    },
    "gemma4_26b": {
        "hf_repos": "unsloth/gemma-4-26B-A4B-it|google/gemma-4-26B-A4B-it",
        "local_name": "gemma-4-26B-A4B-it",
        "data_dir": ROOT / "data" / "gemma4",
        "out_json": ROOT / "results" / "paper_gemma4_26b.json",
    },
    "qwen36_35b": {
        "hf_repos": "Qwen/Qwen3.6-35B-A3B",
        "local_name": "Qwen3.6-35B-A3B",
        "data_dir": ROOT / "data" / "qwen35",
        "out_json": ROOT / "results" / "paper_qwen36_35b.json",
    },
}

TARGET_BPWS = (2.5, 3.5, 4.5, 6.0)
GGUF_PRESETS = ("IQ2_S", "IQ3_M", "Q4_K_S", "Q5_K_M", "Q6_K")


# =================================================================================================
# 1. Native Helpers (`NativeVtqCodec` & `libggml` Universal Row Dequantizer Shim)
# =================================================================================================

def build_vtq_native_codec(work_dir: Path) -> NativeVtqCodec:
    so_path = work_dir / "libvtq_io.so"
    if not so_path.exists():
        subprocess.run(
            [
                "c++", "-O3", "-std=c++17", "-ffp-contract=off", "-fPIC", "-shared",
                str(ROOT / "src/native/vtq_pack_tiles.cpp"),
                str(ROOT / "src/native/vtq_unpack_tiles.cpp"),
                "-o", str(so_path),
            ],
            check=True,
        )
    return NativeVtqCodec(so_path)


class LibGgmlShim:
    """Direct ctypes wrapper around upstream `llama.cpp` `ggml_get_type_traits(type)->to_float`."""

    def __init__(self, llama_cpp_dir: Path, work_dir: Path):
        so_path = work_dir / "libgguf_deq_shim.so"
        lib_bin = llama_cpp_dir / "build" / "bin"
        if not so_path.exists():
            c_src = work_dir / "gguf_deq_shim.c"
            c_src.write_text(
                '#include <stdint.h>\n'
                '#include <stddef.h>\n'
                '#include "ggml.h"\n'
                'size_t shim_row_bytes(int type, int64_t ne0) {\n'
                '    return ggml_row_size((enum ggml_type)type, ne0);\n'
                '}\n'
                'void shim_dequant_rows(int type, const void* src, float* dst, int64_t nrows, int64_t ne0) {\n'
                '    const struct ggml_type_traits* tr = ggml_get_type_traits((enum ggml_type)type);\n'
                '    size_t rb = ggml_row_size((enum ggml_type)type, ne0);\n'
                '    if (tr->to_float != NULL) {\n'
                '        #pragma omp parallel for schedule(static) if (nrows >= 64)\n'
                '        for (int64_t r = 0; r < nrows; r++) {\n'
                '            tr->to_float((const char*)src + (size_t)r * rb, dst + (size_t)r * (size_t)ne0, ne0);\n'
                '        }\n'
                '    }\n'
                '}\n',
                encoding="utf-8",
            )
            inc_dirs = [
                f"-I{llama_cpp_dir / 'ggml' / 'include'}",
                f"-I{llama_cpp_dir / 'include'}",
            ]
            subprocess.run(
                [
                    "gcc", "-O3", "-fopenmp", "-fPIC", "-shared", *inc_dirs,
                    str(c_src),
                    f"-L{lib_bin}", f"-Wl,-rpath,{lib_bin}", "-lggml-base",
                    "-o", str(so_path),
                ],
                check=True,
            )
        self._lib = ctypes.CDLL(str(so_path))
        self._row_bytes = self._lib.shim_row_bytes
        self._row_bytes.argtypes = [ctypes.c_int, ctypes.c_int64]
        self._row_bytes.restype = ctypes.c_size_t
        self._deq = self._lib.shim_dequant_rows
        self._deq.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int64, ctypes.c_int64]
        self._deq.restype = None

    def parse_tensors(self, gguf_path: Path) -> Tuple[int, Dict[str, dict]]:
        with open(gguf_path, "rb") as f:
            magic = f.read(4)
            assert magic == b"GGUF", f"Not GGUF: {magic!r}"
            _, n_tensors, n_kv = struct.unpack("<IQQ", f.read(20))
            alignment = 32
            for _ in range(n_kv):
                key = gguf_utils._read_gguf_string(f)
                vtype = struct.unpack("<I", f.read(4))[0]
                if key == "general.alignment" and vtype == 4:
                    alignment = struct.unpack("<I", f.read(4))[0]
                else:
                    gguf_utils._skip_gguf_value(f, vtype)
            tensors = {}
            for _ in range(n_tensors):
                name = gguf_utils._read_gguf_string(f)
                n_dims = struct.unpack("<I", f.read(4))[0]
                dims = tuple(struct.unpack(f"<{n_dims}Q", f.read(8 * n_dims)))
                ggml_type, offset = struct.unpack("<IQ", f.read(12))
                ne0 = int(dims[0])
                nrows = 1
                for d in dims[1:]:
                    nrows *= int(d)
                rb = int(self._row_bytes(int(ggml_type), ne0))
                n_bytes = rb * nrows
                tensors[name] = {
                    "dims": dims,
                    "ggml_type": int(ggml_type),
                    "offset": int(offset),
                    "n_elements": ne0 * nrows,
                    "n_bytes": n_bytes,
                }
            pos = f.tell()
            data_offset = (pos + alignment - 1) // alignment * alignment
        return data_offset, tensors

    def dequant(self, raw_mv: np.ndarray, ggml_type: int, dims_gguf: Tuple[int, ...]) -> torch.Tensor:
        ne0 = int(dims_gguf[0])
        nrows = 1
        for d in dims_gguf[1:]:
            nrows *= int(d)
        pt_shape = tuple(int(d) for d in reversed(dims_gguf))
        if ggml_type == 0:  # F32
            arr = np.frombuffer(raw_mv, dtype=np.float32, count=ne0 * nrows).copy()
            return torch.from_numpy(arr).reshape(pt_shape)
        out = np.empty(nrows * ne0, dtype=np.float32)
        src_buf = np.ascontiguousarray(raw_mv, dtype=np.uint8)
        self._deq(
            int(ggml_type),
            src_buf.ctypes.data_as(ctypes.c_void_p),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            nrows,
            ne0,
        )
        return torch.from_numpy(out).reshape(pt_shape)


import struct


# =================================================================================================
# 2. Neutral Calibration & Evaluation Trace Builder
# =================================================================================================

def prepare_neutral_cal_and_traces(
    model_dir: Path,
    data_dir: Path,
    work_dir: Path,
    tag: str,
    n_cal_rows: int = 64,
    cal_seq_len: int = 2048,
) -> Tuple[Path, Path, Dict[str, List[Tuple[torch.Tensor, List[int], List[int]]]]]:
    cal_st = work_dir / f"cal_neutral_{tag}.safetensors"
    cal_txt = work_dir / f"cal_neutral_{tag}.txt"
    wt2_cache = work_dir / f"wt2_eval_{tag}.json"

    tok = AutoTokenizer.from_pretrained(str(model_dir))
    if not cal_st.exists() or not cal_txt.exists() or not wt2_cache.exists():
        raw_test = (ROOT / "data" / "wikitext2_test.raw").read_text(encoding="utf-8")
        test_ids = tok.encode(raw_test, add_special_tokens=False)
        eval_seqs = []
        chunk_len = 512
        for i in range(40):
            seg = test_ids[i * chunk_len : (i + 1) * chunk_len]
            if len(seg) < 128:
                break
            eval_seqs.append({"id": f"wt2_{i}", "token_ids": seg, "score_spans": [[64, len(seg)]]})
        wt2_cache.write_text(json.dumps({"sequences": eval_seqs}), encoding="utf-8")

        from datasets import load_dataset
        ds_train = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
        articles = [t.strip() for t in ds_train["text"] if t and len(t.strip()) > 60]
        train_text = "\n\n".join(articles)
        train_ids = tok.encode(train_text, add_special_tokens=False)
        need = n_cal_rows * cal_seq_len
        if len(train_ids) < need:
            train_ids = (train_ids * ((need // len(train_ids)) + 2))[:need]
        cal_ids = torch.tensor(train_ids[:need], dtype=torch.long).view(n_cal_rows, cal_seq_len).contiguous()
        save_file({"input_ids": cal_ids}, str(cal_st))

        # Exact decoded text of those 64 x 2048 chunks for `llama-imatrix`
        decoded_chunks = [tok.decode(cal_ids[r].tolist(), skip_special_tokens=True) for r in range(n_cal_rows)]
        cal_txt.write_text("\n\n".join(decoded_chunks), encoding="utf-8")
        print(f"=== Prepared neutral WikiText-2 calibration ({n_cal_rows}x{cal_seq_len}) -> {cal_st.name} ===", flush=True)

    traces: Dict[str, List[Tuple[torch.Tensor, List[int], List[int]]]] = {}
    for ds_name, p in [
        ("nemotron", data_dir / "nemotron_trace.json"),
        ("multilingual", data_dir / "multilingual_trace.json"),
        ("wikitext2", wt2_cache),
    ]:
        items = []
        for sq_item in load_eval_trace(p):
            pos, tg = get_scored_positions_and_targets(sq_item)
            if pos:
                items.append((torch.tensor([sq_item["token_ids"][:8192]], dtype=torch.long), pos, tg))
        traces[ds_name] = items
    return cal_st, cal_txt, traces


# =================================================================================================
# 3. Streaming TextModel Helper (Supports Llama-3.1, Gemma-4 MoE, Qwen3.6 Hybrid MoE)
# =================================================================================================

class StreamingCausalEvaluator:
    """Layer-streaming CausalLM wrapper for BF16 reference, Fisher offload, VTQ quantization, and multi-format eval."""

    def __init__(self, model_dir: Path, device: torch.device):
        self.model_dir = model_dir
        self.device = device
        full_cfg = AutoConfig.from_pretrained(str(model_dir))
        self.config = getattr(full_cfg, "text_config", full_cfg)
        with torch.device("meta"):
            self.model = AutoModelForCausalLM.from_config(self.config, dtype=torch.bfloat16, attn_implementation="sdpa")
        self.model.eval()
        self.store = sqm.TensorStore(model_dir)
        self.prefix = "model.language_model" if self.store.has("model.language_model.embed_tokens.weight") else "model"
        self.tm = self.model.model
        self.tie_word_embeddings = bool(
            getattr(self.config, "tie_word_embeddings", False) or getattr(full_cfg, "tie_word_embeddings", False)
        )
        if not self.store.has("lm_head.weight"):
            self.tie_word_embeddings = True

        # Materialize all non-layer top-level children of `self.tm` (just like `hybrid_quant.TextModel`)
        for name, mod in self.tm.named_children():
            if name in ("layers", "rotary_emb"):
                continue
            hqm._load_module(mod, f"{self.prefix}.{name}", self.store, "cpu" if name == "embed_tokens_per_layer" else device)
        with torch.device(device):
            self.tm.rotary_emb = type(self.tm.rotary_emb)(config=self.config)
        if self.tie_word_embeddings:
            self.model.lm_head.to_empty(device=device)
            self.model.lm_head.weight = self.tm.embed_tokens.weight
        else:
            hqm._load_module(self.model.lm_head, "lm_head", self.store, device)
        self.softcap = getattr(self.config, "final_logit_softcapping", None)

        self.layers = list(self.tm.layers)
        self.n_layers = len(self.layers)
        self.tm.layers = nn.ModuleList([hqm._Catcher(self, i) for i in range(self.n_layers)])
        self.want, self.caught = -1, None

        # Save clean BF16 embed_tokens & lm_head on CPU for fast restoration between methods
        self.clean_embed_cpu = self.tm.embed_tokens.weight.detach().cpu().clone()
        self.clean_head_cpu = None if self.tie_word_embeddings else self.model.lm_head.weight.detach().cpu().clone()

    def restore_clean_embed_and_head(self):
        self.tm.embed_tokens.weight.data.copy_(self.clean_embed_cpu.to(self.device))
        if self.tie_word_embeddings:
            self.model.lm_head.weight = self.tm.embed_tokens.weight
        else:
            if self.model.lm_head.weight.data_ptr() == self.tm.embed_tokens.weight.data_ptr():
                self.model.lm_head.weight = nn.Parameter(torch.empty_like(self.clean_head_cpu, device=self.device), requires_grad=False)
            self.model.lm_head.weight.data.copy_(self.clean_head_cpu.to(self.device))

    @torch.no_grad()
    def capture_input_and_ctx(self, ids: torch.Tensor) -> Tuple[torch.Tensor, List[tuple]]:
        """Captures `h_0` (on CPU) and per-layer `(rest, kwargs)` contexts (masks, RoPE) across all layers."""
        ids_gpu = ids.to(self.device)
        self.want = 0
        try:
            self.tm(input_ids=ids_gpu, use_cache=False)
        except hqm._Stop:
            pass
        a0, kw0 = self.caught
        self.caught = None
        h0_cpu = a0[0].detach().cpu()

        # Check which distinct layer context types exist (e.g. sliding vs full attention, linear vs full)
        ctxs = []
        seen_by_type = {}
        layer_types = getattr(self.config, "layer_types", None)
        for l_idx in range(self.n_layers):
            lt_key = layer_types[l_idx] if layer_types is not None else "default"
            if lt_key in seen_by_type:
                ctxs.append(seen_by_type[lt_key])
                continue
            self.want = l_idx
            try:
                self.tm(input_ids=ids_gpu, use_cache=False)
            except hqm._Stop:
                pass
            args, kwargs = self.caught
            self.caught = None
            kwargs = {k: hqm._own(v) for k, v in kwargs.items()}
            c = (tuple(hqm._own(a) for a in args[1:]), kwargs)
            seen_by_type[lt_key] = c
            ctxs.append(c)
        return h0_cpu, ctxs

    def load_layer_bf16(self, l_idx: int) -> nn.Module:
        layer = self.layers[l_idx]
        key_pfx = f"{self.prefix}.layers.{l_idx}"
        layer.to_empty(device=self.device)
        for name, t in list(layer.named_parameters()) + list(layer.named_buffers()):
            full_k = f"{key_pfx}.{name}"
            if self.store.has(full_k):
                t.data.copy_(self.store.get(full_k, self.device))
            elif name.endswith("experts.gate_up_proj"):
                base = full_k[: -len("gate_up_proj")]
                t.data.copy_(
                    torch.stack(
                        [
                            torch.cat([self.store.get(f"{base}{e}.gate_proj.weight", self.device),
                                       self.store.get(f"{base}{e}.up_proj.weight", self.device)])
                            for e in range(t.shape[0])
                        ]
                    )
                )
            elif name.endswith("experts.down_proj"):
                base = full_k[: -len("down_proj")]
                t.data.copy_(torch.stack([self.store.get(f"{base}{e}.down_proj.weight", self.device) for e in range(t.shape[0])]))
            elif name.endswith("embed_scale"):
                pass
            else:
                raise KeyError(f"{full_k} not found in checkpoint")
        for m in layer.modules():
            if hasattr(m, "scalar_embed_scale"):
                m.embed_scale = torch.tensor(m.scalar_embed_scale, device=self.device)
        if len(self.store._handles) > 3:
            self.store._handles.clear()
        return layer

    def release_layer(self, l_idx: int):
        self.layers[l_idx] = self.layers[l_idx].to("meta")

    def call_layer(self, layer: nn.Module, h_cpu: torch.Tensor, ctx: tuple) -> torch.Tensor:
        rest, kwargs = ctx
        out = layer(h_cpu.to(self.device), *rest, **kwargs)
        out = out[0] if isinstance(out, tuple) else out
        return out.detach().cpu()

    def logits(self, h_gpu: torch.Tensor) -> torch.Tensor:
        out = self.model.lm_head(self.tm.norm(h_gpu))
        if self.softcap:
            out = torch.tanh(out.float() / self.softcap) * self.softcap
        return out

    def count_text_params_and_breakdown(self) -> dict:
        emb_p = int(self.clean_embed_cpu.numel())
        head_p = 0 if self.tie_word_embeddings else int(self.clean_head_cpu.numel())
        norm_top = sum(int(p.numel()) for p in self.tm.norm.parameters())
        lin_q_p = 0
        other_layer_p = 0
        for l_idx in range(self.n_layers):
            lay = self.layers[l_idx]
            for name, mod in lay.named_modules():
                if isinstance(mod, nn.Linear):
                    n = mod.out_features * mod.in_features
                    if is_quantizable_linear(name, mod):
                        lin_q_p += n
                    else:
                        other_layer_p += n
                    if mod.bias is not None:
                        other_layer_p += mod.bias.numel()
            ex_obj, _ = get_moe_experts_and_prefix(lay)
            if ex_obj is not None:
                lin_q_p += int(ex_obj.gate_up_proj.numel()) + int(ex_obj.down_proj.numel())
            # Count remaining non-Linear params/buffers in layer
            all_lp = sum(int(t.numel()) for t in list(lay.parameters()) + list(lay.buffers()))
            lin_all_in_lay = sum(
                m.out_features * m.in_features + (m.bias.numel() if m.bias is not None else 0)
                for _, m in lay.named_modules() if isinstance(m, nn.Linear)
            )
            if ex_obj is not None:
                lin_all_in_lay += int(ex_obj.gate_up_proj.numel()) + int(ex_obj.down_proj.numel())
            other_layer_p += (all_lp - lin_all_in_lay)

        total_text = emb_p + head_p + norm_top + lin_q_p + other_layer_p
        return {
            "total_text_params": total_text,
            "embed_params": emb_p,
            "lm_head_params": head_p,
            "quantizable_decoder_linear_params": lin_q_p,
            "unchanged_other_params": norm_top + other_layer_p,
            "tie_word_embeddings": self.tie_word_embeddings,
        }


def is_quantizable_linear(name: str, mod: nn.Linear) -> bool:
    if name.endswith(("mlp.gate", "router.proj", "shared_expert_gate")):
        return False
    if mod.out_features < 128 or mod.in_features < 128:
        return False
    return True


def get_moe_experts_and_prefix(layer: nn.Module) -> Tuple[Optional[nn.Module], str]:
    """Returns `(experts_module, prefix_str)` for Qwen3.6 (`mlp.experts`) or Gemma-4 (`experts`), else `(None, '')`."""
    if hasattr(layer, "mlp") and hasattr(layer.mlp, "experts") and hasattr(layer.mlp.experts, "gate_up_proj"):
        return layer.mlp.experts, "mlp.experts"
    if hasattr(layer, "experts") and hasattr(layer.experts, "gate_up_proj"):
        return layer.experts, "experts"
    return None, ""


# =================================================================================================
# 4. Architecture-General Row-Fisher Offload (Dense + Qwen3.6 MoE + Gemma-4 MoE)
# =================================================================================================

def compute_general_fisher_offload(
    ev: StreamingCausalEvaluator,
    cal_ids: torch.Tensor,
    fisher_path: Path,
    n_seqs: int = 8,
    seq_len: int = 1024,
) -> Dict[Tuple[int, str], torch.Tensor]:
    if fisher_path.exists():
        return torch.load(str(fisher_path), map_location="cpu", weights_only=True)

    print(f"\n=== Computing Row-Fisher Sensitivities ({n_seqs}x{seq_len}) -> {fisher_path.name} ===", flush=True)
    t0 = time.perf_counter()
    dev = ev.device
    ev.restore_clean_embed_and_head()

    n = min(n_seqs, cal_ids.shape[0])
    toks = [cal_ids[i : i + 1, :seq_len] for i in range(n)]
    total_toks = sum(int(t.numel()) for t in toks)

    hs_cpu = []
    ctxs_per_seq = []
    for t in toks:
        h0, ctxs = ev.capture_input_and_ctx(t)
        hs_cpu.append(h0)
        ctxs_per_seq.append(ctxs)

    layer_inputs: List[List[torch.Tensor]] = []
    fisher: Dict[Tuple[int, str], torch.Tensor] = {}

    for l_idx in range(ev.n_layers):
        layer_inputs.append(hs_cpu)
        lay = ev.load_layer_bf16(l_idx)
        for p in lay.parameters():
            p.requires_grad_(False)
        for name, mod in lay.named_modules():
            if isinstance(mod, nn.Linear) and is_quantizable_linear(name, mod):
                fisher[(l_idx, name)] = torch.zeros(mod.out_features, dtype=torch.float64, device="cpu")
        ex_obj, ex_pfx = get_moe_experts_and_prefix(lay)
        if ex_obj is not None:
            E = ex_obj.num_experts
            gu_out = ex_obj.gate_up_proj.shape[1]
            dn_out = ex_obj.down_proj.shape[1]
            for e in range(E):
                fisher[(l_idx, f"{ex_pfx}.{e}.gate_up_proj")] = torch.zeros(gu_out, dtype=torch.float64, device="cpu")
                fisher[(l_idx, f"{ex_pfx}.{e}.down_proj")] = torch.zeros(dn_out, dtype=torch.float64, device="cpu")

        with torch.no_grad():
            hs_cpu = [ev.call_layer(lay, hs_cpu[s], ctxs_per_seq[s][l_idx]) for s in range(n)]
        ev.release_layer(l_idx)

    # Sample inverse-CDF targets at final logits and backpropagate layer-by-layer
    gs_cpu = []
    for i, h_c in enumerate(hs_cpu):
        h = h_c.to(dev).detach().requires_grad_(True)
        with torch.enable_grad():
            logits = ev.logits(h).reshape(-1, ev.config.vocab_size)
            with torch.no_grad():
                probs = F.softmax(logits.float(), dim=-1)
                u = torch.rand((probs.shape[0], 2), generator=torch.Generator().manual_seed(1000 + i)).to(dev)
                cdf = probs.cumsum(dim=-1)
                y = torch.searchsorted(cdf, (u * cdf[:, -1:]).contiguous()).clamp(max=probs.shape[-1] - 1)
                del probs, cdf
            loss = 0.5 * (
                F.cross_entropy(logits.float(), y[:, 0], reduction="sum")
                + F.cross_entropy(logits.float(), y[:, 1], reduction="sum")
            )
            g_h = torch.autograd.grad(loss, h)[0].detach().cpu()
            gs_cpu.append(g_h)
        del logits, loss, h
    del hs_cpu
    torch.cuda.empty_cache()

    for l_idx in reversed(range(ev.n_layers)):
        lay = ev.load_layer_bf16(l_idx)
        for p in lay.parameters():
            p.requires_grad_(False)
        gpu_acc: Dict[str, torch.Tensor] = {}
        hooks = []
        for name, mod in lay.named_modules():
            if isinstance(mod, nn.Linear) and is_quantizable_linear(name, mod):
                gpu_acc[name] = torch.zeros(mod.out_features, dtype=torch.float64, device=dev)

                def _make_fwd(sub_nm: str):
                    def _fwd(m, inp, out):
                        if out.requires_grad:
                            out.register_hook(
                                lambda gr, k=sub_nm: (
                                    gpu_acc[k].add_(gr.detach().float().reshape(-1, gr.shape[-1]).square().sum(dim=0).double()),
                                    None,
                                )[1]
                            )
                    return _fwd

                hooks.append(mod.register_forward_hook(_make_fwd(name)))

        ex_obj, ex_pfx = get_moe_experts_and_prefix(lay)
        if ex_obj is not None:
            for e in range(ex_obj.num_experts):
                gpu_acc[f"{ex_pfx}.{e}.gate_up_proj"] = torch.zeros(ex_obj.gate_up_proj.shape[1], dtype=torch.float64, device=dev)
                gpu_acc[f"{ex_pfx}.{e}.down_proj"] = torch.zeros(ex_obj.down_proj.shape[1], dtype=torch.float64, device=dev)

            def _add_ex(sub_nm: str, g: torch.Tensor):
                gpu_acc[sub_nm].add_(g.detach().float().square().sum(dim=0).double())

            def _ex_fwd(hidden_states, top_k_index, top_k_weights, _ex=ex_obj, _pfx=ex_pfx):
                out = torch.zeros_like(hidden_states)
                with torch.no_grad():
                    mask = F.one_hot(top_k_index, num_classes=_ex.num_experts).permute(2, 1, 0)
                    hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()
                for e in hit[:, 0].tolist():
                    pos_i, tok_i = torch.where(mask[e])
                    gu = F.linear(hidden_states[tok_i], _ex.gate_up_proj[e])
                    if gu.requires_grad:
                        gu.register_hook(lambda gr, ee=e: (_add_ex(f"{_pfx}.{ee}.gate_up_proj", gr), None)[1])
                    gate_t, up_t = gu.chunk(2, dim=-1)
                    dn = F.linear(_ex.act_fn(gate_t) * up_t, _ex.down_proj[e])
                    if dn.requires_grad:
                        dn.register_hook(lambda gr, ee=e: (_add_ex(f"{_pfx}.{ee}.down_proj", gr), None)[1])
                    out.index_add_(0, tok_i, (dn * top_k_weights[tok_i, pos_i, None]).to(out.dtype))
                return out

            ex_obj.forward = _ex_fwd

        for s in range(n):
            x = layer_inputs[l_idx][s].to(dev).detach().requires_grad_(True)
            rest, kwargs = ctxs_per_seq[s][l_idx]
            with torch.enable_grad():
                out = lay(x, *rest, **kwargs)
                out = out[0] if isinstance(out, tuple) else out
                gs_cpu[s] = torch.autograd.grad(out, x, grad_outputs=gs_cpu[s].to(dev))[0].detach().cpu()
            del x, out

        for hk in hooks:
            hk.remove()
        if ex_obj is not None:
            del ex_obj.forward
        for sub_nm, acc_v in gpu_acc.items():
            fisher[(l_idx, sub_nm)] = (acc_v / max(total_toks, 1)).float().cpu()
        del gpu_acc
        ev.release_layer(l_idx)
        layer_inputs[l_idx] = []
        torch.cuda.empty_cache()
        if l_idx % 8 == 0 or l_idx == ev.n_layers - 1:
            print(f"  [Fisher] layer {l_idx}/{ev.n_layers - 1} done (+{time.perf_counter()-t0:.1f}s)", flush=True)

    fisher_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(fisher, str(fisher_path))
    print(f"=== Saved Fisher ({len(fisher)} matrices) in {time.perf_counter()-t0:.1f}s ===", flush=True)
    return fisher


# =================================================================================================
# 5. Layer-wise MoE & Linear Hessian Helper for VTQ Profile & Quantization
# =================================================================================================

@torch.no_grad()
def run_layer_moe_hessians(
    layer: nn.Module,
    hs_in_cpu: List[torch.Tensor],
    device: torch.device,
    on_gate_up,
    on_down,
    shrink: float = 64.0,
):
    """Computes routed per-expert Hessians `(gate_up_proj, down_proj)` for both Qwen3.6 MoE and Gemma-4 MoE."""
    ex_obj, ex_pfx = get_moe_experts_and_prefix(layer)
    assert ex_obj is not None
    E = ex_obj.num_experts
    gu_w = ex_obj.gate_up_proj
    dn_w = ex_obj.down_proj
    H_dim = gu_w.shape[2]
    I_dim = dn_w.shape[2]

    routes = []
    xs_gpu_inputs = []
    S = torch.zeros((E, H_dim, H_dim), dtype=torch.float32, device=device)
    n = torch.zeros(E, dtype=torch.float64)
    acc_all = lv2._CovAccumulator()

    is_gemma4 = (ex_pfx == "experts")
    for h_c in hs_in_cpu:
        h_g = h_c.to(device)
        if is_gemma4:
            # In Gemma4DecoderLayer, residual before MLP is post_attention_layernorm(attn) + h_in;
            # caller passes `residual_post_attn` directly!
            flat_r = h_g.reshape(-1, H_dim)
            _, _, sel = layer.router(flat_r)
            x_exp = layer.pre_feedforward_layernorm_2(flat_r)
        else:
            flat_r = layer.post_attention_layernorm(h_g).reshape(-1, H_dim)
            _, _, sel = layer.mlp.gate(flat_r)
            x_exp = flat_r
        routes.append(sel.cpu())
        xs_gpu_inputs.append(x_exp.cpu())
        acc_all.add(x_exp)
        for e in torch.unique(sel).tolist():
            if e < E:
                xe = x_exp[(sel == e).any(dim=-1)]
                S[e] += torch.mm(xe.T, xe, out_dtype=torch.float32)
                n[e] += xe.shape[0]

    H_layer = acc_all.get()
    for e in range(E):
        on_gate_up(e, (S[e] + shrink * H_layer) / (float(n[e]) + shrink))
    del S, H_layer
    torch.cuda.empty_cache()

    S_d = torch.zeros((E, I_dim, I_dim), dtype=torch.float32, device=device)
    n_d = torch.zeros(E, dtype=torch.float64)
    acc_mid = lv2._CovAccumulator()
    for x_exp_c, sel_c in zip(xs_gpu_inputs, routes):
        x_exp = x_exp_c.to(device)
        sel = sel_c.to(device)
        for e in torch.unique(sel).tolist():
            if e < E:
                xe = x_exp[(sel == e).any(dim=-1)]
                gate_t, up_t = F.linear(xe, ex_obj.gate_up_proj[e]).chunk(2, dim=-1)
                mid = ex_obj.act_fn(gate_t) * up_t
                S_d[e] += torch.mm(mid.T, mid, out_dtype=torch.float32)
                n_d[e] += mid.shape[0]
                acc_mid.add(mid)

    H_mid = acc_mid.get()
    for e in range(E):
        on_down(e, (S_d[e] + shrink * H_mid) / (float(n_d[e]) + shrink))
    del S_d, H_mid, xs_gpu_inputs, routes
    torch.cuda.empty_cache()


class _StopCapture(Exception):
    pass


@torch.no_grad()
def discover_linear_groups(layer: nn.Module, run_once) -> List[List[str]]:
    seen = []
    hooks = [
        mod.register_forward_pre_hook(lambda m, a, n=name: seen.append((n, a[0])))
        for name, mod in layer.named_modules()
        if isinstance(mod, nn.Linear) and is_quantizable_linear(name, mod)
    ]
    try:
        run_once()
    finally:
        for h in hooks:
            h.remove()
    groups, reps = [], []
    for name, x in seen:
        for g, r in zip(groups, reps):
            if r is x:
                if name not in g:
                    g.append(name)
                break
        else:
            if not any(name in g for g in groups):
                groups.append([name])
                reps.append(x)
    return groups


def slice_step_for_matrix(name: str, M: int) -> int:
    if "experts." in name:
        return M  # 1 slice per expert matrix (already fine-grained across 128-256 experts!)
    for s in (256, 128, 64, 32, 16):
        if M >= s and M % s == 0:
            if any(k in name for k in ("k_proj", "v_proj")):
                return min(s, 128)
            if "mlp" in name and M >= 512 and M % 256 == 0:
                return 256
            return min(s, 128)
    return 16


@torch.no_grad()
def build_vtq_profile_and_head_hessian(
    ev: StreamingCausalEvaluator,
    cal_ids: torch.Tensor,
    fisher: Dict[Tuple[int, str], torch.Tensor],
    profile_cache_path: Path,
    n_profile_rows: int = 16,
) -> Tuple[dict, torch.Tensor]:
    if profile_cache_path.exists():
        saved = torch.load(str(profile_cache_path), map_location="cpu", weights_only=False)
        return saved["profile"], saved["H_head"]

    print(f"\n=== Building VTQ Slice Distortion Profile & Head Hessian ({n_profile_rows}x{cal_ids.shape[1]}) ===", flush=True)
    t0 = time.perf_counter()
    dev = ev.device
    ev.restore_clean_embed_and_head()
    vtq.load_and_register_vtq()

    n_p = min(n_profile_rows, cal_ids.shape[0])
    hs_cpu = []
    ctxs = []
    for i in range(n_p):
        h0, c = ev.capture_input_and_ctx(cal_ids[i : i + 1])
        hs_cpu.append(h0)
        ctxs.append(c)

    slices = []
    for l_idx in range(ev.n_layers):
        lay = ev.load_layer_bf16(l_idx)
        mods = dict(lay.named_modules())
        ex_obj, ex_pfx = get_moe_experts_and_prefix(lay)

        def _probe_run():
            rest, kwargs = ctxs[0][l_idx]
            lay(hs_cpu[0].to(dev), *rest, **kwargs)

        groups = discover_linear_groups(lay, _probe_run)

        for g in groups:
            q_names = [nm for nm in g if is_quantizable_linear(nm, mods[nm])]
            if not q_names:
                continue
            acc = lv2._CovAccumulator()

            def _cap_hook(m, a, ca=acc):
                ca.add(a[0])
                raise _StopCapture()

            h_cap = mods[q_names[0]].register_forward_pre_hook(_cap_hook)
            try:
                for s in range(n_p):
                    rest, kwargs = ctxs[s][l_idx]
                    try:
                        lay(hs_cpu[s].to(dev), *rest, **kwargs)
                    except _StopCapture:
                        pass
            finally:
                h_cap.remove()
            H_g = acc.get()
            for nm in q_names:
                W = mods[nm].weight.data
                rw = fisher[(l_idx, nm)].to(dev)
                st = slice_step_for_matrix(nm, W.shape[0])
                vtq.profile_matrix_vtq(slices, l_idx, nm, W, H_g, sqm._seed(l_idx, nm), st, s_row=rw, rot_block=-1)
            del H_g

        # Advance `hs_cpu` through BF16 layer while capturing exact post-attention residual if MoE is present
        post_attn_res_cpu = []
        if ex_obj is not None:
            is_gemma4 = (ex_pfx == "experts")
            target_norm = lay.pre_feedforward_layernorm if is_gemma4 else lay.post_attention_layernorm
            hk = target_norm.register_forward_pre_hook(lambda m, a: post_attn_res_cpu.append(a[0].detach().cpu()))
        else:
            hk = None

        next_hs = []
        try:
            for s in range(n_p):
                next_hs.append(ev.call_layer(lay, hs_cpu[s], ctxs[s][l_idx]))
        finally:
            if hk is not None:
                hk.remove()

        if ex_obj is not None:
            def _prof_gu(e, He):
                full_nm = f"{ex_pfx}.{e}.gate_up_proj"
                W = ex_obj.gate_up_proj.data[e]
                rw = fisher[(l_idx, full_nm)].to(dev)
                vtq.profile_matrix_vtq(slices, l_idx, full_nm, W, He, sqm._seed(l_idx, f"gu{e}"), W.shape[0], s_row=rw, rot_block=-1)

            def _prof_dn(e, He):
                full_nm = f"{ex_pfx}.{e}.down_proj"
                W = ex_obj.down_proj.data[e]
                rw = fisher[(l_idx, full_nm)].to(dev)
                vtq.profile_matrix_vtq(slices, l_idx, full_nm, W, He, sqm._seed(l_idx, f"dn{e}"), W.shape[0], s_row=rw, rot_block=-1)

            run_layer_moe_hessians(lay, post_attn_res_cpu, dev, _prof_gu, _prof_dn)
            del post_attn_res_cpu

        hs_cpu = next_hs
        ev.release_layer(l_idx)
        torch.cuda.empty_cache()
        if l_idx % 8 == 0 or l_idx == ev.n_layers - 1:
            print(f"  [Profile] layer {l_idx}/{ev.n_layers - 1} done (+{time.perf_counter()-t0:.1f}s)", flush=True)

    # Pre-`lm_head` Hessian `H_head = E[norm(h)^T norm(h)]`
    acc_head = lv2._CovAccumulator()
    for h_c in hs_cpu:
        acc_head.add(ev.tm.norm(h_c.to(dev)))
    H_head = acc_head.get().cpu()

    profile = sqm.finalize_profile(slices)
    torch.save({"profile": profile, "H_head": H_head}, str(profile_cache_path))
    print(f"=== Saved VTQ Profile ({len(slices)} slices) in {time.perf_counter()-t0:.1f}s ===", flush=True)
    return profile, H_head


# =================================================================================================
# 6. Full End-to-End VTQ Quantizer & `.vtq` v2 Container Exporter
# =================================================================================================

def choose_vtq_embed_head_specs(target_bpw: float) -> Tuple[str, str]:
    if target_bpw <= 2.6:
        return "VTQ_225", "VTQ_350"
    if target_bpw <= 3.6:
        return "VTQ_325", "VTQ_450"
    if target_bpw <= 4.6:
        return "VTQ_425", "VTQ_550"
    return "VTQ_600", "VTQ_600"


@torch.no_grad()
def quantize_and_export_vtq_full(
    ev: StreamingCausalEvaluator,
    cal_ids: torch.Tensor,
    fisher: Dict[Tuple[int, str], torch.Tensor],
    profile: dict,
    H_head_cpu: torch.Tensor,
    target_bpw: float,
    out_vtq_path: Path,
    codec: NativeVtqCodec,
) -> dict:
    dev = ev.device
    ev.restore_clean_embed_and_head()
    vtq.load_and_register_vtq()
    all_fmts = list(vf.load_vtq_formats().values())

    info = ev.count_text_params_and_breakdown()
    total_params = info["total_text_params"]
    emb_spec, head_spec = choose_vtq_embed_head_specs(target_bpw)
    if ev.tie_word_embeddings:
        # Single shared table serves both lookup and output logits: allocate slightly richer spec
        emb_spec = {"VTQ_225": "VTQ_275_H2", "VTQ_325": "VTQ_375", "VTQ_425": "VTQ_475", "VTQ_600": "VTQ_600"}[emb_spec]

    def _matrix_bytes(M: int, K: int, sp_name: str) -> int:
        M_p = ((M + 15) // 16) * 16
        K_p = ((K + 255) // 256) * 256
        st_b = vtq.VTQ_CONFIG[sp_name]["st_bytes"]
        data_b = (M_p // 16) * (K_p // 256) * st_b
        return ((data_b + 63) & ~63) + ((2 * M_p + 63) & ~63) + ((2 * K_p + 63) & ~63) + 64

    V, H_dim = ev.clean_embed_cpu.shape
    emb_bytes = _matrix_bytes(V, H_dim, emb_spec)
    head_bytes = 0 if ev.tie_word_embeddings else _matrix_bytes(V, H_dim, head_spec)
    unchanged_bytes = info["unchanged_other_params"] * 2 + 131072  # BF16/FP32 norms/routers + ~128 KB metadata/alignment

    # Compute exact `d_row + suh` header bytes across all decoder linear matrices (excluding `u_sb` which is in `st_bytes`)
    dec_scale_bytes = 0
    dec_padded_params = sum(int(s["params"]) for s in profile["slices"])
    seen_mat = set()
    for s in profile["slices"]:
        mk = (s["layer"], s["subname"])
        if mk not in seen_mat:
            seen_mat.add(mk)
            K_p = s["params"] // (s["r1"] - s["r0"])
            dec_scale_bytes += 2 * K_p + 128  # suh + alignment
        dec_scale_bytes += 2 * (s["r1"] - s["r0"])

    target_total_bytes = target_bpw * total_params / 8.0
    avail_dec_tile_bytes = target_total_bytes - emb_bytes - head_bytes - unchanged_bytes - dec_scale_bytes
    # Note: `LIQK_V2_SPECS[sp].bpw` includes `+2.0 / 256.0 = 0.0078125` bpw for `d_row`, so match its definition:
    dp_target_bpw = (avail_dec_tile_bytes * 8.0 / max(dec_padded_params, 1)) + (2.0 / 256.0)
    dp_target_bpw = max(min(dp_target_bpw, 6.22), 1.61)

    cands = vtq.vtq_candidates_for_target(dp_target_bpw, vtq.VTQ_MENU)
    assignment = lv2.solve_fisher_slice_dp(profile, target_bpw=dp_target_bpw, candidate_specs=cands)
    ach_bits = sum(
        (r1 - r0) * (next(s["params"] // (s["r1"] - s["r0"]) for s in profile["slices"] if s["layer"] == k[0] and s["subname"] == k[1]))
        * lv2.LIQK_V2_SPECS[sp].bpw
        for k, sls in assignment.items() for r0, r1, sp in sls
    )
    ach_bpw = ach_bits / max(dec_padded_params, 1)

    print(
        f"\n=== [VTQ {target_bpw:.2f} bpw] DP target={dp_target_bpw:.4f} (achieved={ach_bpw:.4f}) | "
        f"emb={emb_spec} head={'TIED' if ev.tie_word_embeddings else head_spec} ===",
        flush=True,
    )

    out_vtq_path.unlink(missing_ok=True)
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    wr = vf.VtqWriter(out_vtq_path, all_fmts, codec=codec)
    vtq.PACK = {}
    quantized_source_keys = set()
    lin_bytes_sum = 0

    try:
        # 1. Quantize `embed_tokens` (using `rot_block=256`; blended with `H_head` if tied!)
        W_emb = ev.clean_embed_cpu.to(dev, dtype=torch.float32)
        if ev.tie_word_embeddings:
            Hh = H_head_cpu.to(dev, dtype=torch.float32)
            Hh_norm = Hh / Hh.diagonal().mean().clamp(min=1e-8)
            H_emb = 0.85 * Hh_norm + 0.15 * torch.eye(H_dim, dtype=torch.float32, device=dev)
        else:
            H_emb = torch.eye(H_dim, dtype=torch.float32, device=dev)
        _, W_emb_hat, _ = vtq.quantize_matrix_vtq(W_emb, H_emb, emb_spec, seed=9001, damp=0.20, rot_block=256)
        wr.add_matrix(f"{ev.prefix}.embed_tokens", vtq.PACK.pop("matrix"), verify=False)
        quantized_source_keys.add(f"{ev.prefix}.embed_tokens.weight")
        ev.tm.embed_tokens.weight.data.copy_(W_emb_hat.to(torch.bfloat16))
        if ev.tie_word_embeddings:
            ev.model.lm_head.weight = ev.tm.embed_tokens.weight
        del W_emb, H_emb, W_emb_hat
        torch.cuda.empty_cache()

        # 2. Capture initial hidden states on `cal_ids` from the quantized embedding table!
        n_cal = cal_ids.shape[0]
        hs_cpu = []
        ctxs = []
        for i in range(n_cal):
            h0, c = ev.capture_input_and_ctx(cal_ids[i : i + 1])
            hs_cpu.append(h0)
            ctxs.append(c)

        # 3. Stream layer-by-layer quantization
        for l_idx in range(ev.n_layers):
            lay = ev.load_layer_bf16(l_idx)
            mods = dict(lay.named_modules())
            ex_obj, ex_pfx = get_moe_experts_and_prefix(lay)
            lay_pfx = f"{ev.prefix}.layers.{l_idx}"

            def _probe_run():
                rest, kwargs = ctxs[0][l_idx]
                lay(hs_cpu[0].to(dev), *rest, **kwargs)

            groups = discover_linear_groups(lay, _probe_run)
            for g in groups:
                q_names = [nm for nm in g if is_quantizable_linear(nm, mods[nm])]
                if not q_names:
                    continue
                acc = lv2._CovAccumulator()

                def _cap_q(m, a, ca=acc):
                    ca.add(a[0])
                    raise _StopCapture()

                h_cap = mods[q_names[0]].register_forward_pre_hook(_cap_q)
                try:
                    for s in range(n_cal):
                        rest, kwargs = ctxs[s][l_idx]
                        try:
                            lay(hs_cpu[s].to(dev), *rest, **kwargs)
                        except _StopCapture:
                            pass
                finally:
                    h_cap.remove()
                H_g = acc.get()
                for nm in q_names:
                    W = mods[nm].weight.data
                    rw = fisher[(l_idx, nm)].to(dev)
                    row_sl = assignment[(l_idx, nm)]
                    dm = 0.30 if nm.endswith(("down_proj", "o_proj", "out_proj")) else 0.20
                    pos_before = wr.pos
                    _, w_hat, _ = vtq.quantize_matrix_vtq(
                        W, H_g, row_sl, seed=sqm._seed(l_idx, nm), damp=dm, row_weights=rw, rot_block=-1
                    )
                    wr.add_matrix(f"{lay_pfx}.{nm}", vtq.PACK.pop("matrix"), verify=False)
                    lin_bytes_sum += (wr.pos - pos_before)
                    quantized_source_keys.add(f"{lay_pfx}.{nm}.weight")
                    mods[nm].weight.data.copy_(w_hat.to(W.dtype))
                    del w_hat
                del H_g
                torch.cuda.empty_cache()

            if ex_obj is not None:
                is_gemma4 = (ex_pfx == "experts")
                target_norm = lay.pre_feedforward_layernorm if is_gemma4 else lay.post_attention_layernorm
                post_attn_res_cpu = []

                def _cap_moe_res(m, a):
                    post_attn_res_cpu.append(a[0].detach().cpu())
                    raise _StopCapture()

                hk = target_norm.register_forward_pre_hook(_cap_moe_res)
                try:
                    for s in range(n_cal):
                        rest, kwargs = ctxs[s][l_idx]
                        try:
                            lay(hs_cpu[s].to(dev), *rest, **kwargs)
                        except _StopCapture:
                            pass
                finally:
                    hk.remove()
                moe_in = post_attn_res_cpu

                def _q_gu(e, He):
                    nonlocal lin_bytes_sum
                    full_nm = f"{ex_pfx}.{e}.gate_up_proj"
                    W = ex_obj.gate_up_proj.data[e]
                    rw = fisher[(l_idx, full_nm)].to(dev)
                    row_sl = assignment[(l_idx, full_nm)]
                    pos_before = wr.pos
                    _, w_hat, _ = vtq.quantize_matrix_vtq(
                        W, He, row_sl, seed=sqm._seed(l_idx, f"gu{e}"), damp=0.20, row_weights=rw, rot_block=-1
                    )
                    wr.add_matrix(f"{lay_pfx}.{full_nm}", vtq.PACK.pop("matrix"), verify=False)
                    lin_bytes_sum += (wr.pos - pos_before)
                    ex_obj.gate_up_proj.data[e].copy_(w_hat.to(W.dtype))
                    del w_hat

                def _q_dn(e, He):
                    nonlocal lin_bytes_sum
                    full_nm = f"{ex_pfx}.{e}.down_proj"
                    W = ex_obj.down_proj.data[e]
                    rw = fisher[(l_idx, full_nm)].to(dev)
                    row_sl = assignment[(l_idx, full_nm)]
                    pos_before = wr.pos
                    _, w_hat, _ = vtq.quantize_matrix_vtq(
                        W, He, row_sl, seed=sqm._seed(l_idx, f"dn{e}"), damp=0.30, row_weights=rw, rot_block=-1
                    )
                    wr.add_matrix(f"{lay_pfx}.{full_nm}", vtq.PACK.pop("matrix"), verify=False)
                    lin_bytes_sum += (wr.pos - pos_before)
                    ex_obj.down_proj.data[e].copy_(w_hat.to(W.dtype))
                    del w_hat

                run_layer_moe_hessians(lay, moe_in, dev, _q_gu, _q_dn)
                # Mark fused or per-expert checkpoint keys as replaced
                for kind in ("gate_up_proj", "down_proj"):
                    fk = f"{lay_pfx}.{ex_pfx}.{kind}"
                    if ev.store.has(fk):
                        quantized_source_keys.add(fk)
                    else:
                        for e in range(ex_obj.num_experts):
                            if kind == "gate_up_proj":
                                quantized_source_keys.add(f"{lay_pfx}.{ex_pfx}.{e}.gate_proj.weight")
                                quantized_source_keys.add(f"{lay_pfx}.{ex_pfx}.{e}.up_proj.weight")
                            else:
                                quantized_source_keys.add(f"{lay_pfx}.{ex_pfx}.{e}.down_proj.weight")

            # Advance quantized calibration stream to next layer
            hs_cpu = [ev.call_layer(lay, hs_cpu[s], ctxs[s][l_idx]) for s in range(n_cal)]
            ev.release_layer(l_idx)
            torch.cuda.empty_cache()
            if l_idx % 8 == 0 or l_idx == ev.n_layers - 1:
                print(f"  [VTQ {target_bpw:.1f}] layer {l_idx}/{ev.n_layers - 1} done (+{time.perf_counter()-t0:.1f}s)", flush=True)

        # 4. Quantize untied `lm_head` using the quantized stream's final `norm(h)` Hessian
        if not ev.tie_word_embeddings:
            acc_qh = lv2._CovAccumulator()
            for h_c in hs_cpu:
                acc_qh.add(ev.tm.norm(h_c.to(dev)))
            H_qh = acc_qh.get()
            W_hd = ev.clean_head_cpu.to(dev, dtype=torch.float32)
            _, W_hd_hat, _ = vtq.quantize_matrix_vtq(W_hd, H_qh, head_spec, seed=9002, damp=0.20, rot_block=-1)
            wr.add_matrix("lm_head", vtq.PACK.pop("matrix"), verify=False)
            quantized_source_keys.add("lm_head.weight")
            del W_hd, H_qh, W_hd_hat
        del hs_cpu
        torch.cuda.empty_cache()

        # 5. Copy all unchanged text-model tensors (norms, routers, conv1d, scalars; skip vision/audio/mtp)
        skip_prefixes = ("model.visual.", "model.vision_tower.", "model.embed_vision.", "model.audio_tower.", "mtp.")
        for k in sorted(ev.store.keys()):
            if k in quantized_source_keys or k.startswith(skip_prefixes):
                continue
            wr.add_tensor(k, ev.store.get(k, "cpu"))

        meta = vf.source_metadata(ev.model_dir)
        meta["quantization"] = {
            "formats": [vf.format_json_vtq(f) for f in all_fmts],
            "target_bpw": target_bpw,
            "embed_spec": emb_spec,
            "head_spec": "tied" if ev.tie_word_embeddings else head_spec,
            "rot_block": -1,
        }
        wr.finalize(meta)
        with open(out_vtq_path, "rb") as fs:
            os.fsync(fs.fileno())
    finally:
        vtq.PACK = None

    torch.cuda.synchronize()
    quant_sec = time.perf_counter() - t0
    file_bytes = out_vtq_path.stat().st_size
    with open(out_vtq_path, "rb") as fs:
        sha256 = hashlib.file_digest(fs, "sha256").hexdigest()

    return {
        "file_bytes": file_bytes,
        "total_bpw": round(file_bytes * 8.0 / total_params, 5),
        "linear_bpw": round(lin_bytes_sum * 8.0 / max(info["quantizable_decoder_linear_params"], 1), 5),
        "quant_sec": round(quant_sec, 2),
        "sha256": sha256,
        "embed_spec": emb_spec,
        "head_spec": "tied" if ev.tie_word_embeddings else head_spec,
    }


# =================================================================================================
# 7. Multi-Format Streaming Dequantizer & Evaluator (`VTQ`, `GGUF`, `EXL3`)
# =================================================================================================

def _best_cand(cands: List[torch.Tensor], ref: torch.Tensor, label: str) -> torch.Tensor:
    errs = [float((c.float() - ref.float()).norm() / ref.float().norm().clamp(min=1e-8)) for c in cands]
    best = int(np.argmin(errs))
    if errs[best] > 0.78:
        raise RuntimeError(f"Layout/reconstruction mismatch on {label}: rel_errs={errs}")
    return cands[best]


class VtqDiskLoader:
    def __init__(self, vtq_path: Path, codec: NativeVtqCodec, prefix: str, tie_word_embeddings: bool):
        self.rd = vf.VtqFile(vtq_path, codec=codec)
        self.prefix = prefix
        self.tie = tie_word_embeddings

    def close(self):
        self.rd.close()

    def load_embed_and_head(self, ev: StreamingCausalEvaluator):
        w_e = self.rd.decode_matrix(f"{self.prefix}.embed_tokens", device=ev.device).to(torch.bfloat16)
        _best_cand([w_e], ev.clean_embed_cpu.to(ev.device), "vtq.embed_tokens")
        ev.tm.embed_tokens.weight.data.copy_(w_e)
        if self.tie:
            ev.model.lm_head.weight = ev.tm.embed_tokens.weight
        else:
            w_h = self.rd.decode_matrix("lm_head", device=ev.device).to(torch.bfloat16)
            _best_cand([w_h], ev.clean_head_cpu.to(ev.device), "vtq.lm_head")
            ev.model.lm_head.weight.data.copy_(w_h)

    def load_layer(self, lay: nn.Module, l_idx: int, dev: torch.device):
        lpfx = f"{self.prefix}.layers.{l_idx}"
        for nm, mod in lay.named_modules():
            if isinstance(mod, nn.Linear) and is_quantizable_linear(nm, mod):
                wq = self.rd.decode_matrix(f"{lpfx}.{nm}", device=dev).to(mod.weight.dtype)
                if l_idx < 2:
                    _best_cand([wq], mod.weight.data, f"vtq.L{l_idx}.{nm}")
                mod.weight.data.copy_(wq)
        ex_obj, ex_pfx = get_moe_experts_and_prefix(lay)
        if ex_obj is not None:
            for e in range(ex_obj.num_experts):
                gu = self.rd.decode_matrix(f"{lpfx}.{ex_pfx}.{e}.gate_up_proj", device=dev).to(ex_obj.gate_up_proj.dtype)
                dn = self.rd.decode_matrix(f"{lpfx}.{ex_pfx}.{e}.down_proj", device=dev).to(ex_obj.down_proj.dtype)
                if l_idx == 0 and e == 0:
                    _best_cand([gu], ex_obj.gate_up_proj.data[e], f"vtq.L0.{ex_pfx}.0.gate_up")
                    _best_cand([dn], ex_obj.down_proj.data[e], f"vtq.L0.{ex_pfx}.0.down")
                ex_obj.gate_up_proj.data[e].copy_(gu)
                ex_obj.down_proj.data[e].copy_(dn)


class GgufDiskLoader:
    GGUF_MAP = {
        "linear_attn.in_proj_qkv": "attn_qkv",
        "linear_attn.in_proj_z": "attn_gate",
        "linear_attn.out_proj": "ssm_out",
        "linear_attn.in_proj_a": "ssm_alpha",
        "linear_attn.in_proj_b": "ssm_beta",
        "self_attn.q_proj": "attn_q",
        "self_attn.k_proj": "attn_k",
        "self_attn.v_proj": "attn_v",
        "self_attn.o_proj": "attn_output",
        "mlp.gate_proj": "ffn_gate",
        "mlp.up_proj": "ffn_up",
        "mlp.down_proj": "ffn_down",
        "mlp.shared_expert.gate_proj": "ffn_gate_shexp",
        "mlp.shared_expert.up_proj": "ffn_up_shexp",
        "mlp.shared_expert.down_proj": "ffn_down_shexp",
    }

    def __init__(self, gguf_path: Path, shim: LibGgmlShim, config):
        self.shim = shim
        self.data_off, self.tensors = shim.parse_tensors(gguf_path)
        self.mm = np.memmap(gguf_path, dtype=np.uint8, mode="r")
        self.config = config
        self.nk = getattr(config, "linear_num_key_heads", 1)
        self.nv = getattr(config, "linear_num_value_heads", 1)
        self.dk = getattr(config, "linear_key_head_dim", 0)
        self.dv = getattr(config, "linear_value_head_dim", 0)
        self.nq = getattr(config, "num_attention_heads", 1)
        self.nkv = getattr(config, "num_key_value_heads", 1)
        self.ngkv = getattr(config, "num_global_key_value_heads", self.nkv)
        self._perm_cache: Dict[str, int] = {}

    def close(self):
        del self.mm

    def _raw_deq(self, key: str, dev: torch.device) -> torch.Tensor:
        info = self.tensors[key]
        o = self.data_off + info["offset"]
        return self.shim.dequant(self.mm[o : o + info["n_bytes"]], info["ggml_type"], info["dims"]).to(dev)

    def _grouped_v(self, W: torch.Tensor, dim: int) -> torch.Tensor:
        n = W.shape[dim]
        idx = torch.arange(n, device=W.device).reshape(self.nv // self.nk, self.nk, self.dv).transpose(0, 1).reshape(n)
        return W.index_select(dim, idx)

    def load_embed_and_head(self, ev: StreamingCausalEvaluator):
        w_e = self._raw_deq("token_embd.weight", ev.device).to(torch.bfloat16)
        _best_cand([w_e], ev.clean_embed_cpu.to(ev.device), "gguf.token_embd")
        ev.tm.embed_tokens.weight.data.copy_(w_e)
        if ev.tie_word_embeddings or "output.weight" not in self.tensors:
            ev.model.lm_head.weight = ev.tm.embed_tokens.weight
        else:
            w_h = self._raw_deq("output.weight", ev.device).to(torch.bfloat16)
            _best_cand([w_h], ev.clean_head_cpu.to(ev.device), "gguf.output")
            ev.model.lm_head.weight.data.copy_(w_h)

    def load_layer(self, lay: nn.Module, l_idx: int, dev: torch.device):
        for nm, mod in lay.named_modules():
            if isinstance(mod, nn.Linear) and nm in self.GGUF_MAP:
                gkey = f"blk.{l_idx}.{self.GGUF_MAP[nm]}.weight"
                if gkey not in self.tensors:
                    continue
                W = self._raw_deq(gkey, dev)
                ref = mod.weight.data
                cands = [W]
                if nm == "self_attn.q_proj":
                    cands.append(gguf_utils.unpermute_qk(W, self.nq))
                    if getattr(self.config, "attn_output_gate", False):
                        hd = W.shape[0] // (2 * self.nq)
                        # Qwen3.5/3.6 gated attention: GGUF may interleave [nq, 2, hd, K] vs [2, nq, hd, K]
                        w_deint = W.view(self.nq, 2, hd, W.shape[1]).transpose(0, 1).reshape_as(W).contiguous()
                        cands.append(w_deint)
                        q_p, g_p = w_deint.chunk(2, dim=0)
                        cands.append(torch.cat([gguf_utils.unpermute_qk(q_p, self.nq), g_p], dim=0))
                elif nm == "self_attn.k_proj":
                    for nh_cand in {self.nkv, self.ngkv}:
                        if nh_cand > 0 and W.shape[0] % (2 * nh_cand) == 0:
                            cands.append(gguf_utils.unpermute_qk(W, nh_cand))
                elif self.nv != self.nk:
                    if nm == "linear_attn.in_proj_qkv":
                        q = 2 * self.nk * self.dk
                        cands.append(torch.cat([W[:q], self._grouped_v(W[q:], 0)], dim=0))
                    elif nm == "linear_attn.in_proj_z":
                        cands.append(self._grouped_v(W, 0))
                    elif nm == "linear_attn.out_proj":
                        cands.append(self._grouped_v(W, 1))

                cache_k = f"{nm}:{W.shape[0]}x{W.shape[1]}"
                if cache_k in self._perm_cache and len(cands) > self._perm_cache[cache_k]:
                    chosen = cands[self._perm_cache[cache_k]]
                    if l_idx < 2:
                        _best_cand([chosen], ref, gkey)
                else:
                    errs = [float((c.float() - ref.float()).norm() / ref.float().norm().clamp(min=1e-8)) for c in cands]
                    best_i = int(np.argmin(errs))
                    if errs[best_i] > 0.78:
                        raise RuntimeError(f"GGUF layout mismatch on {gkey}: errs={errs}")
                    self._perm_cache[cache_k] = best_i
                    chosen = cands[best_i]
                mod.weight.data.copy_(chosen.to(mod.weight.dtype))

        ex_obj, ex_pfx = get_moe_experts_and_prefix(lay)
        if ex_obj is not None:
            gu_key = f"blk.{l_idx}.ffn_gate_up_exps.weight"
            if gu_key in self.tensors:
                gu = self._raw_deq(gu_key, dev)
            else:
                g_e = self._raw_deq(f"blk.{l_idx}.ffn_gate_exps.weight", dev)
                u_e = self._raw_deq(f"blk.{l_idx}.ffn_up_exps.weight", dev)
                gu = torch.cat([g_e, u_e], dim=1)
            dn = self._raw_deq(f"blk.{l_idx}.ffn_down_exps.weight", dev)
            if l_idx == 0:
                _best_cand([gu[0]], ex_obj.gate_up_proj.data[0], f"gguf.L0.{ex_pfx}.0.gate_up")
                _best_cand([dn[0]], ex_obj.down_proj.data[0], f"gguf.L0.{ex_pfx}.0.down")
            ex_obj.gate_up_proj.data.copy_(gu.to(ex_obj.gate_up_proj.dtype))
            ex_obj.down_proj.data.copy_(dn.to(ex_obj.down_proj.dtype))


class Exl3DiskLoader:
    def __init__(self, exl3_dir: Path, prefix: str, device: torch.device):
        self.store = sqm.TensorStore(exl3_dir)
        self.prefix = prefix
        self.device = device

    def close(self):
        self.store._handles.clear()

    def _reconstruct_linear(self, key: str, M_orig: int, K_orig: int) -> torch.Tensor:
        from exllamav3.ext import exllamav3_ext as ext
        from exllamav3.modules.quant.exl3 import LinearEXL3

        if self.store.has(f"{key}.weight") and not self.store.has(f"{key}.trellis"):
            return self.store.get(f"{key}.weight", self.device)[:M_orig, :K_orig].float()

        M_pad = ((M_orig + 127) // 128) * 128
        K_pad = ((K_orig + 127) // 128) * 128

        def _get(sub):
            fk = f"{key}.{sub}"
            return self.store.get(fk, self.device) if self.store.has(fk) else None

        lin = LinearEXL3(
            config=None,
            in_features=K_pad,
            out_features=M_pad,
            scale=None,
            su=_get("su"),
            sv=_get("sv"),
            suh=_get("suh"),
            svh=_get("svh"),
            trellis=_get("trellis"),
            mcg=_get("mcg"),
            mul1=_get("mul1"),
            out_dtype=torch.float16,
            key=key,
        )
        w_t = torch.empty((K_pad, M_pad), dtype=torch.float16, device=self.device)
        ext.reconstruct_had_slice(w_t, lin.trellis, lin.suh, lin.svh, lin.K, lin.mcg, lin.mul1, 0)
        return w_t[:K_orig, :M_orig].T.contiguous()

    def _reconstruct_embedding(self, key: str, V: int, H_dim: int) -> torch.Tensor:
        if self.store.has(f"{key}.weight"):
            return self.store.get(f"{key}.weight", self.device)[:V, :H_dim]
        from exllamav3.ext import exllamav3_ext as ext
        from exllamav3.modules.quant.exl3_lib.ngram_codec import GROUP_DIM

        # Quantized RowTable (`<key>.trellis` or sharded `<key>.trellis.0000` + `<key>.signs`)
        signs = self.store.get(f"{key}.signs", self.device)
        if self.store.has(f"{key}.trellis"):
            t_rows = [self.store.get(f"{key}.trellis", "cpu")]
        else:
            shard_keys = sorted(k for k in self.store.keys() if k.startswith(f"{key}.trellis."))
            t_rows = [self.store.get(sk, "cpu") for sk in shard_keys]
        full_tbl = torch.cat(t_rows, dim=0)  # [V, row_words]
        groups = signs.shape[0]
        row_words = full_tbl.shape[1]
        K_bits = (row_words // groups - 1) // 16
        out_full = torch.empty((V, H_dim), dtype=torch.bfloat16, device=self.device)
        chunk = 8192
        for r0 in range(0, V, chunk):
            r1 = min(r0 + chunk, V)
            U = r1 - r0
            rings = full_tbl[r0:r1].to(self.device).view(U * groups, -1).contiguous()
            dec = torch.empty((U * groups, GROUP_DIM), dtype=torch.float16, device=self.device)
            ext.ngram_dequant(rings, K_bits, None, signs, dec, True)
            out_full[r0:r1] = dec.view(U, -1)[:, :H_dim].to(torch.bfloat16)
        return out_full

    def text_bytes_and_linear_bytes(self) -> Tuple[int, int]:
        skip_prefixes = ("model.visual.", "model.vision_tower.", "model.embed_vision.", "model.audio_tower.", "mtp.")
        dtype_sz = {
            torch.float32: 4, torch.float16: 2, torch.bfloat16: 2,
            torch.int64: 8, torch.int32: 4, torch.int16: 2, torch.uint16: 2, torch.int8: 1, torch.uint8: 1,
        }
        tot_b = 0
        lin_b = 0
        for k in self.store.keys():
            if k.startswith(skip_prefixes):
                continue
            sl = self.store._handle(k).get_slice(k)
            dt_str = sl.get_dtype()
            bpe = {"F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4, "I16": 2, "U16": 2, "I8": 1, "U8": 1}[dt_str]
            nb = math.prod(sl.get_shape()) * bpe
            tot_b += nb
            if ".layers." in k and any(k.endswith(suf) for suf in (".trellis", ".suh", ".svh", ".su", ".sv", ".mcg", ".mul1")):
                lin_b += nb
        if len(self.store._handles) > 2:
            self.store._handles.clear()
        return tot_b, lin_b

    def load_embed_and_head(self, ev: StreamingCausalEvaluator):
        V, H_dim = ev.clean_embed_cpu.shape
        w_e = self._reconstruct_embedding(f"{self.prefix}.embed_tokens", V, H_dim).to(torch.bfloat16)
        _best_cand([w_e], ev.clean_embed_cpu.to(ev.device), "exl3.embed_tokens")
        ev.tm.embed_tokens.weight.data.copy_(w_e)
        if self.store.has("lm_head.trellis") or self.store.has("lm_head.weight"):
            if ev.model.lm_head.weight.data_ptr() == ev.tm.embed_tokens.weight.data_ptr():
                ev.model.lm_head.weight = nn.Parameter(torch.empty((V, H_dim), dtype=torch.bfloat16, device=ev.device), requires_grad=False)
            ref_h = ev.clean_embed_cpu if ev.tie_word_embeddings else ev.clean_head_cpu
            w_h = self._reconstruct_linear("lm_head", V, H_dim).to(torch.bfloat16)
            _best_cand([w_h], ref_h.to(ev.device), "exl3.lm_head")
            ev.model.lm_head.weight.data.copy_(w_h)
        else:
            ev.model.lm_head.weight = ev.tm.embed_tokens.weight

    def load_layer(self, lay: nn.Module, l_idx: int, dev: torch.device):
        lpfx = f"{self.prefix}.layers.{l_idx}"
        for nm, mod in lay.named_modules():
            if isinstance(mod, nn.Linear) and is_quantizable_linear(nm, mod):
                fk = f"{lpfx}.{nm}"
                if self.store.has(f"{fk}.trellis") or self.store.has(f"{fk}.weight"):
                    wq = self._reconstruct_linear(fk, mod.out_features, mod.in_features).to(mod.weight.dtype)
                    if l_idx < 2:
                        _best_cand([wq], mod.weight.data, f"exl3.L{l_idx}.{nm}")
                    mod.weight.data.copy_(wq)
        ex_obj, ex_pfx = get_moe_experts_and_prefix(lay)
        if ex_obj is not None:
            I_dim = ex_obj.down_proj.shape[2]
            H_dim = ex_obj.gate_up_proj.shape[2]
            for e in range(ex_obj.num_experts):
                base = f"{lpfx}.{ex_pfx}.{e}"
                g = self._reconstruct_linear(f"{base}.gate_proj", I_dim, H_dim)
                u = self._reconstruct_linear(f"{base}.up_proj", I_dim, H_dim)
                gu = torch.cat([g, u], dim=0).to(ex_obj.gate_up_proj.dtype)
                dn = self._reconstruct_linear(f"{base}.down_proj", H_dim, I_dim).to(ex_obj.down_proj.dtype)
                if l_idx == 0 and e == 0:
                    _best_cand([gu], ex_obj.gate_up_proj.data[0], f"exl3.L0.{ex_pfx}.0.gate_up")
                    _best_cand([dn], ex_obj.down_proj.data[0], f"exl3.L0.{ex_pfx}.0.down")
                ex_obj.gate_up_proj.data[e].copy_(gu)
                ex_obj.down_proj.data[e].copy_(dn)
        if len(self.store._handles) > 2:
            self.store._handles.clear()


@torch.no_grad()
def compute_bf16_ref_final_states(
    ev: StreamingCausalEvaluator,
    traces: Dict[str, List[Tuple[torch.Tensor, List[int], List[int]]]],
    ref_cache_path: Path,
) -> Dict[str, List[torch.Tensor]]:
    if ref_cache_path.exists():
        return torch.load(str(ref_cache_path), map_location="cpu", weights_only=True)

    print("\n=== Computing BF16 Reference Hidden States Across All 3 Held-Out Traces ===", flush=True)
    t0 = time.perf_counter()
    ev.restore_clean_embed_and_head()

    flat_ids = []
    ds_slices = {}
    for ds_name, items in traces.items():
        i0 = len(flat_ids)
        for ids, _, _ in items:
            flat_ids.append(ids)
        ds_slices[ds_name] = (i0, len(flat_ids))

    hs_cpu = []
    ctxs = []
    for ids in flat_ids:
        h0, c = ev.capture_input_and_ctx(ids)
        hs_cpu.append(h0)
        ctxs.append(c)

    for l_idx in range(ev.n_layers):
        lay = ev.load_layer_bf16(l_idx)
        hs_cpu = [ev.call_layer(lay, hs_cpu[i], ctxs[i][l_idx]) for i in range(len(flat_ids))]
        ev.release_layer(l_idx)
        torch.cuda.empty_cache()

    ref_final: Dict[str, List[torch.Tensor]] = {}
    for ds_name, items in traces.items():
        i0, i1 = ds_slices[ds_name]
        ref_final[ds_name] = [hs_cpu[i0 + k][0, pos].contiguous() for k, (_, pos, _) in enumerate(items)]
    torch.save(ref_final, str(ref_cache_path))
    print(f"=== Cached BF16 Reference States in {time.perf_counter()-t0:.1f}s ===", flush=True)
    return ref_final


@torch.no_grad()
def evaluate_disk_loader(
    ev: StreamingCausalEvaluator,
    loader,
    traces: Dict[str, List[Tuple[torch.Tensor, List[int], List[int]]]],
    ref_final: Dict[str, List[torch.Tensor]],
) -> Dict[str, dict]:
    t0 = time.perf_counter()
    dev = ev.device
    ev.restore_clean_embed_and_head()

    # 1. Load quantized `embed_tokens` + `lm_head` and capture `h_0` from the quantized embedding table!
    loader.load_embed_and_head(ev)
    flat_ids = []
    ds_slices = {}
    for ds_name, items in traces.items():
        i0 = len(flat_ids)
        for ids, _, _ in items:
            flat_ids.append(ids)
        ds_slices[ds_name] = (i0, len(flat_ids))

    hs_cpu = []
    ctxs = []
    for ids in flat_ids:
        h0, c = ev.capture_input_and_ctx(ids)
        hs_cpu.append(h0)
        ctxs.append(c)

    # 2. Stream layers 0 .. n_layers - 1
    for l_idx in range(ev.n_layers):
        lay = ev.load_layer_bf16(l_idx)
        loader.load_layer(lay, l_idx, dev)
        hs_cpu = [ev.call_layer(lay, hs_cpu[i], ctxs[i][l_idx]) for i in range(len(flat_ids))]
        ev.release_layer(l_idx)
        torch.cuda.empty_cache()

    # 3. Compute exact end-to-end KLD / PPL / Top-1 using quantized `lm_head` vs BF16 `lm_head`!
    q_head_fn = lambda h: ev.logits(h)

    # Build a separate clean BF16 `logits_fn` so `lr` uses clean BF16 `lm_head` while `lq` uses quantized `lm_head`!
    clean_head_w = (ev.clean_embed_cpu if ev.tie_word_embeddings else ev.clean_head_cpu).to(dev)

    def ref_head_fn(h: torch.Tensor) -> torch.Tensor:
        out = F.linear(ev.tm.norm(h), clean_head_w)
        if ev.softcap:
            out = torch.tanh(out.float() / ev.softcap) * ev.softcap
        return out

    metrics_out = {}
    for ds_name, items in traces.items():
        i0, i1 = ds_slices[ds_name]
        q_final = [hs_cpu[i0 + k][0, pos].contiguous() for k, (_, pos, _) in enumerate(items)]
        targets = [tg for _, _, tg in items]
        metrics_out[ds_name] = trace_metrics_dual_head(ref_final[ds_name], q_final, targets, dev, ref_head_fn, q_head_fn)

    del clean_head_w, hs_cpu
    ev.restore_clean_embed_and_head()
    torch.cuda.empty_cache()
    metrics_out["eval_sec"] = round(time.perf_counter() - t0, 2)
    return metrics_out


@torch.no_grad()
def trace_metrics_dual_head(
    ref_final: list,
    q_final: list,
    targets: list,
    device: torch.device,
    ref_logits_fn,
    q_logits_fn,
    n_boot: int = 2000,
    seed: int = 42,
    chunk: int = 512,
) -> dict:
    seq_kld, seq_cnt, seq_nll_q, all_kld = [], [], [], []
    top1 = tot = 0
    nll_ref_sum = nll_q_sum = 0.0
    for hr_all, hq_all, tg_all in zip(ref_final, q_final, targets):
        s_kld = s_nq = 0.0
        for c0 in range(0, len(tg_all), chunk):
            hr = hr_all[c0 : c0 + chunk].to(device)
            hq = hq_all[c0 : c0 + chunk].to(device)
            tg = tg_all[c0 : c0 + chunk]
            lr = ref_logits_fn(hr).float()
            lq = q_logits_fn(hq).float()
            tgt = torch.tensor(tg, device=device, dtype=torch.long)
            lpr = F.log_softmax(lr, dim=-1)
            lpq = F.log_softmax(lq, dim=-1)
            kld = (lpr.exp() * (lpr - lpq)).sum(dim=-1).clamp(min=0.0)
            s_kld += float(kld.sum().item())
            all_kld.append(kld.cpu().numpy())
            top1 += int((lr.argmax(dim=-1) == lq.argmax(dim=-1)).sum().item())
            nr = -lpr.gather(-1, tgt[:, None]).squeeze(-1)
            nq = -lpq.gather(-1, tgt[:, None]).squeeze(-1)
            nll_ref_sum += float(nr.sum().item())
            nll_q_sum += float(nq.sum().item())
            s_nq += float(nq.sum().item())
            tot += len(tg)
        seq_kld.append(s_kld)
        seq_cnt.append(len(tg_all))
        seq_nll_q.append(s_nq)

    sk = np.asarray(seq_kld, dtype=np.float64)
    sc = np.asarray(seq_cnt, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(sk), size=(n_boot, len(sk)))
    boot = (sk[idx] * 1.0).sum(axis=1) / sc[idx].sum(axis=1)
    flat_k = np.concatenate(all_kld) if all_kld else np.zeros(1)
    ppl_ref = float(math.exp(min(nll_ref_sum / max(tot, 1), 30.0)))
    ppl_q = float(math.exp(min(nll_q_sum / max(tot, 1), 30.0)))
    return {
        "mean_kld": float(sk.sum() / max(sc.sum(), 1)),
        "ci95_lo": float(np.percentile(boot, 2.5)),
        "ci95_hi": float(np.percentile(boot, 97.5)),
        "median_kld": float(np.median(flat_k)),
        "p99_kld": float(np.percentile(flat_k, 99.0)),
        "top1_match": float(top1 / max(tot, 1)),
        "ppl_ref": ppl_ref,
        "ppl_quant": ppl_q,
        "delta_ppl": ppl_q - ppl_ref,
        "n_tokens": int(tot),
        "n_seqs": int(len(sk)),
        "per_seq_kld_sums": [float(x) for x in seq_kld],
        "per_seq_token_counts": [int(x) for x in seq_cnt],
        "per_seq_nll_sums": [float(x) for x in seq_nll_q],
    }


# =================================================================================================
# 8. GGUF & EXL3 End-to-End Orchestration
# =================================================================================================

def run_gguf_phase(
    ev: StreamingCausalEvaluator,
    model_dir: Path,
    llama_cpp_dir: Path,
    shim: LibGgmlShim,
    cal_txt: Path,
    traces: dict,
    ref_final: dict,
    results: dict,
    out_json: Path,
    work_dir: Path,
):
    info = ev.count_text_params_and_breakdown()
    total_params = info["total_text_params"]
    todo = [p for p in GGUF_PRESETS if f"GGUF:{p}" not in results["methods"]]
    if not todo:
        return

    bin_dir = llama_cpp_dir / "build" / "bin"
    bf16_gguf = work_dir / "model-bf16.gguf"
    imatrix_dat = work_dir / "imatrix.gguf"

    if not bf16_gguf.exists():
        print(f"\n=== [GGUF] Converting {model_dir.name} to BF16 GGUF ===", flush=True)
        t_c = time.perf_counter()
        has_mtp = any(k.startswith("mtp.") for k in ev.store.keys())
        base_cmd = [
            sys.executable, str(llama_cpp_dir / "convert_hf_to_gguf.py"),
            str(model_dir), "--outtype", "bf16", "--outfile", str(bf16_gguf),
        ]
        if has_mtp:
            rc_mtp = subprocess.run([*base_cmd, "--no-mtp"])
            if rc_mtp.returncode != 0:
                bf16_gguf.unlink(missing_ok=True)
                subprocess.run(base_cmd, check=True)
        else:
            subprocess.run(base_cmd, check=True)
        print(f"  Converted BF16 GGUF ({bf16_gguf.stat().st_size / (1<<30):.2f} GiB) in {time.perf_counter()-t_c:.1f}s", flush=True)

    # Verify BF16 GGUF layer-0 + first full_attention layer + embed/head tensor mapping is bit-exact!
    print("  [GGUF] Verifying BF16 GGUF -> PyTorch tensor bijection...", flush=True)
    bf_loader = GgufDiskLoader(bf16_gguf, shim, ev.config)
    bf_loader.load_embed_and_head(ev)
    verify_layers = {0}
    ltypes = getattr(ev.config, "layer_types", None) or []
    for li, lt in enumerate(ltypes):
        if lt == "full_attention":
            verify_layers.add(li)
            break
    for li in sorted(verify_layers):
        if li < ev.n_layers:
            lay_v = ev.load_layer_bf16(li)
            bf_loader.load_layer(lay_v, li, ev.device)
            ev.release_layer(li)
    bf_loader.close()
    ev.restore_clean_embed_and_head()
    torch.cuda.empty_cache()
    print("  [GGUF] BF16 GGUF bijection verified OK!", flush=True)

    imatrix_sec = results.get("gguf_imatrix_sec")
    if not imatrix_dat.exists():
        print("\n=== [GGUF] Computing `llama-imatrix` on shared 64x2048 WikiText-2 calibration ===", flush=True)
        # Release GPU tensors temporarily so `llama-imatrix -ngl 99` has the full 96 GB VRAM!
        ev.tm.embed_tokens.to("cpu")
        ev.model.lm_head.to("cpu")
        torch.cuda.empty_cache()
        t_im = time.perf_counter()
        subprocess.run(
            [
                str(bin_dir / "llama-imatrix"),
                "-m", str(bf16_gguf),
                "-f", str(cal_txt),
                "-o", str(imatrix_dat),
                "-c", "2048", "-b", "512", "--chunks", "64", "-ngl", "99",
            ],
            check=True,
        )
        imatrix_sec = round(time.perf_counter() - t_im, 2)
        results["gguf_imatrix_sec"] = imatrix_sec
        out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
        ev.tm.embed_tokens.to(ev.device)
        ev.model.lm_head.to(ev.device)
        ev.restore_clean_embed_and_head()
        print(f"  Computed imatrix in {imatrix_sec:.1f}s", flush=True)

    # Quantize all requested presets first, then delete the 50-70 GB `bf16.gguf` immediately to save SSD!
    q_paths: Dict[str, Tuple[Path, float]] = {}
    n_threads = min(os.cpu_count() or 16, 32)
    for preset in todo:
        q_p = work_dir / f"model-{preset}.gguf"
        q_p.unlink(missing_ok=True)
        print(f"\n=== [GGUF] Running `llama-quantize --imatrix` -> `{preset}` ===", flush=True)
        t_q = time.perf_counter()
        subprocess.run(
            [
                str(bin_dir / "llama-quantize"),
                "--imatrix", str(imatrix_dat),
                str(bf16_gguf), str(q_p), preset, str(n_threads),
            ],
            check=True,
        )
        q_sec = round(time.perf_counter() - t_q, 2)
        q_paths[preset] = (q_p, q_sec)

    bf16_gguf.unlink(missing_ok=True)

    for preset in todo:
        q_p, q_sec = q_paths[preset]
        raw_file_bytes = q_p.stat().st_size
        loader = GgufDiskLoader(q_p, shim, ev.config)
        non_text_bytes = sum(
            v["n_bytes"] for k, v in loader.tensors.items()
            if k.startswith(("v.", "mm.", "mtp.", "nextn."))
        )
        file_bytes = raw_file_bytes - non_text_bytes
        lin_bytes = sum(
            v["n_bytes"] for k, v in loader.tensors.items()
            if k.startswith("blk.") and any(tag in k for tag in ("attn_q", "attn_k", "attn_v", "attn_output", "attn_gate", "ssm_out", "ffn_gate.", "ffn_up.", "ffn_down.", "_shexp.", "_exps.weight"))
        )
        ev_res = evaluate_disk_loader(ev, loader, traces, ref_final)
        loader.close()
        q_p.unlink(missing_ok=True)

        entry = {
            "format": "GGUF",
            "preset": preset,
            "file_bytes": file_bytes,
            "total_bpw": round(file_bytes * 8.0 / total_params, 5),
            "linear_bpw": round(lin_bytes * 8.0 / max(info["quantizable_decoder_linear_params"], 1), 5),
            "quant_sec": q_sec,
            "quant_plus_imatrix_sec": round(q_sec + float(imatrix_sec or 0.0), 2),
            **ev_res,
        }
        results["methods"][f"GGUF:{preset}"] = entry
        out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(
            f"  [GGUF:{preset:7s}] total_bpw={entry['total_bpw']:.4f} | "
            f"nem={entry['nemotron']['mean_kld']:.5f} mul={entry['multilingual']['mean_kld']:.5f} wt2={entry['wikitext2']['mean_kld']:.5f}",
            flush=True,
        )


def choose_exl3_eb_hb(target_bpw: float) -> Tuple[int, int]:
    if target_bpw <= 2.6:
        return 3, 4
    if target_bpw <= 3.6:
        return 4, 5
    if target_bpw <= 4.6:
        return 5, 6
    return 6, 6


def run_exl3_phase(
    ev: StreamingCausalEvaluator,
    model_dir: Path,
    exl3_repo_dir: Path,
    cal_st: Path,
    traces: dict,
    ref_final: dict,
    results: dict,
    out_json: Path,
    work_dir: Path,
):
    info = ev.count_text_params_and_breakdown()
    total_params = info["total_text_params"]
    emb_p = info["embed_params"]
    # Note: EXL3 always writes a separate `lm_head` tensor even when `tie_word_embeddings=True`!
    head_p_exl3 = emb_p if info["tie_word_embeddings"] else info["lm_head_params"]
    other_bits = info["unchanged_other_params"] * 16.0
    dec_p = info["quantizable_decoder_linear_params"]

    for b_tgt in TARGET_BPWS:
        mkey = f"EXL3:{b_tgt:.1f}"
        if mkey in results["methods"]:
            continue

        eb, hb = choose_exl3_eb_hb(b_tgt)
        # Solve exact `--bits` so total text-model EXL3 size hits `b_tgt`
        # Account for scale vectors `suh + svh` (~0.02 bpw) and Gemma-4 `704->768` padding if present
        pad_factor = 1.038 if "gemma-4" in model_dir.name.lower() else 1.008
        b_dec = (b_tgt * total_params - eb * emb_p - hb * head_p_exl3 - other_bits) / max(dec_p * pad_factor, 1.0)
        b_dec = round(max(min(b_dec, 7.95), 1.55), 4)

        w_dir = work_dir / f"exl3_work_{b_tgt:.1f}"
        o_dir = work_dir / f"exl3_out_{b_tgt:.1f}"
        shutil.rmtree(w_dir, ignore_errors=True)
        shutil.rmtree(o_dir, ignore_errors=True)
        w_dir.mkdir(parents=True, exist_ok=True)
        o_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n=== [{mkey}] Running `exllamav3/convert.py` (--bits {b_dec} -eb {eb} -hb {hb}) ===", flush=True)
        ev.tm.embed_tokens.to("cpu")
        ev.model.lm_head.to("cpu")
        torch.cuda.empty_cache()

        env = dict(os.environ)
        env["PYTHONPATH"] = str(exl3_repo_dir) + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        t_q = time.perf_counter()
        subprocess.run(
            [
                sys.executable, "-u", str(exl3_repo_dir / "convert.py"),
                "-i", str(model_dir),
                "-w", str(w_dir),
                "-o", str(o_dir),
                "-b", str(b_dec),
                "-eb", str(eb),
                "-hb", str(hb),
                "-vb", "16",
                "-mb", "16",
                "-cb", "mul1",
                "-cd", str(cal_st),
                "-cr", "64",
                "-cc", "2048",
            ],
            env=env,
            check=True,
        )
        q_sec = round(time.perf_counter() - t_q, 2)
        shutil.rmtree(w_dir, ignore_errors=True)

        ev.tm.embed_tokens.to(ev.device)
        ev.model.lm_head.to(ev.device)
        ev.restore_clean_embed_and_head()

        sys.path.insert(0, str(exl3_repo_dir))
        loader = Exl3DiskLoader(o_dir, ev.prefix, ev.device)
        tot_b, lin_b = loader.text_bytes_and_linear_bytes()
        ev_res = evaluate_disk_loader(ev, loader, traces, ref_final)
        loader.close()
        shutil.rmtree(o_dir, ignore_errors=True)

        entry = {
            "format": "EXL3",
            "target_bpw": b_tgt,
            "dec_bits_arg": b_dec,
            "embed_bits": eb,
            "head_bits": hb,
            "file_bytes": tot_b,
            "total_bpw": round(tot_b * 8.0 / total_params, 5),
            "linear_bpw": round(lin_b * 8.0 / max(dec_p, 1), 5),
            "quant_sec": q_sec,
            **ev_res,
        }
        results["methods"][mkey] = entry
        out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(
            f"  [{mkey:8s}] total_bpw={entry['total_bpw']:.4f} (quant={q_sec:.1f}s) | "
            f"nem={entry['nemotron']['mean_kld']:.5f} mul={entry['multilingual']['mean_kld']:.5f} wt2={entry['wikitext2']['mean_kld']:.5f}",
            flush=True,
        )


def run_vtq_phase(
    ev: StreamingCausalEvaluator,
    cal_ids: torch.Tensor,
    fisher_path: Path,
    profile_path: Path,
    codec: NativeVtqCodec,
    traces: dict,
    ref_final: dict,
    results: dict,
    out_json: Path,
    work_dir: Path,
):
    todo = [b for b in TARGET_BPWS if f"VTQ:{b:.1f}" not in results["methods"]]
    if not todo:
        return

    t_prep0 = time.perf_counter()
    fisher = compute_general_fisher_offload(ev, cal_ids, fisher_path)
    profile, H_head = build_vtq_profile_and_head_hessian(ev, cal_ids, fisher, profile_path)
    prep_sec = round(time.perf_counter() - t_prep0, 2)
    if "vtq_shared_profile_fisher_sec" not in results and prep_sec > 2.0:
        results["vtq_shared_profile_fisher_sec"] = prep_sec

    for b_tgt in todo:
        mkey = f"VTQ:{b_tgt:.1f}"
        vtq_file = work_dir / f"model_vtq_{b_tgt:.1f}.vtq"
        q_meta = quantize_and_export_vtq_full(ev, cal_ids, fisher, profile, H_head, b_tgt, vtq_file, codec)

        loader = VtqDiskLoader(vtq_file, codec, ev.prefix, ev.tie_word_embeddings)
        ev_res = evaluate_disk_loader(ev, loader, traces, ref_final)
        loader.close()
        vtq_file.unlink(missing_ok=True)

        entry = {
            "format": "VTQ",
            "target_bpw": b_tgt,
            **q_meta,
            **ev_res,
        }
        results["methods"][mkey] = entry
        out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(
            f"  [{mkey:8s}] total_bpw={entry['total_bpw']:.4f} (quant={entry['quant_sec']:.1f}s) | "
            f"nem={entry['nemotron']['mean_kld']:.5f} mul={entry['multilingual']['mean_kld']:.5f} wt2={entry['wikitext2']['mean_kld']:.5f}",
            flush=True,
        )


def compute_summary_comparisons(results: dict):
    comps = {}
    ms = results.get("methods", {})
    gguf_match = {2.5: "GGUF:IQ2_S", 3.5: "GGUF:IQ3_M", 4.5: "GGUF:Q4_K_S", 6.0: "GGUF:Q6_K"}
    for b in TARGET_BPWS:
        vk = f"VTQ:{b:.1f}"
        if vk not in ms:
            continue
        for rk in (f"EXL3:{b:.1f}", gguf_match[b], "GGUF:Q5_K_M" if b == 6.0 else None):
            if rk and rk in ms:
                pair_d = {}
                for ds in ("nemotron", "multilingual", "wikitext2"):
                    if ds in ms[vk] and ds in ms[rk]:
                        d = paired_bootstrap_diff(ms[vk][ds], ms[rk][ds])
                        d["rel_pct"] = round(100.0 * d["mean_diff"] / max(float(ms[rk][ds]["mean_kld"]), 1e-12), 2)
                        pair_d[ds] = d
                comps[f"{vk} ({ms[vk]['total_bpw']:.3f}) vs {rk} ({ms[rk]['total_bpw']:.3f})"] = pair_d
    results["comparisons"] = comps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(MODEL_SPECS), required=True)
    ap.add_argument("--models-root", type=Path, default=ROOT / "models")
    ap.add_argument("--llama-cpp-dir", type=Path, default=Path("/content/llama.cpp"))
    ap.add_argument("--exl3-dir", type=Path, default=Path("/content/exllamav3"))
    ap.add_argument("--work-dir", type=Path, default=Path("/content/work_paper"))
    ap.add_argument("--phases", nargs="+", choices=["vtq", "gguf", "exl3"], default=["vtq", "gguf", "exl3"])
    args = ap.parse_args()

    spec = MODEL_SPECS[args.model]
    model_dir = args.models_root / spec["local_name"]
    out_json: Path = spec["out_json"]
    out_json.parent.mkdir(parents=True, exist_ok=True)
    m_work = args.work_dir / args.model
    m_work.mkdir(parents=True, exist_ok=True)

    dev = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False

    cal_st, cal_txt, traces = prepare_neutral_cal_and_traces(model_dir, spec["data_dir"], m_work, args.model)
    cal_ids = load_file(str(cal_st))["input_ids"]

    ev = StreamingCausalEvaluator(model_dir, dev)
    model_info = ev.count_text_params_and_breakdown()

    results = (
        json.loads(out_json.read_text(encoding="utf-8"))
        if out_json.exists()
        else {
            "model_tag": args.model,
            "model_name": spec["local_name"],
            "calibration": "wikitext-2-raw-v1 train (64x2048 = 131,072 tokens, shared across VTQ / EXL3 / GGUF)",
            "gpu_name": torch.cuda.get_device_name(0),
            "model_info": model_info,
            "methods": {},
            "comparisons": {},
        }
    )
    results["model_info"] = model_info

    ref_final = compute_bf16_ref_final_states(ev, traces, m_work / "ref_final.pt")

    for ph in args.phases:
        if ph == "vtq":
            codec = build_vtq_native_codec(args.work_dir)
            run_vtq_phase(
                ev, cal_ids, m_work / "fisher_neutral.pt", m_work / "vtq_profile.pt",
                codec, traces, ref_final, results, out_json, m_work,
            )
        elif ph == "gguf":
            shim = LibGgmlShim(args.llama_cpp_dir, args.work_dir)
            run_gguf_phase(
                ev, model_dir, args.llama_cpp_dir, shim, cal_txt,
                traces, ref_final, results, out_json, m_work,
            )
        elif ph == "exl3":
            run_exl3_phase(
                ev, model_dir, args.exl3_dir, cal_st,
                traces, ref_final, results, out_json, m_work,
            )

    compute_summary_comparisons(results)
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n=== Completed {args.model} -> {out_json} ===", flush=True)


if __name__ == "__main__":
    main()
