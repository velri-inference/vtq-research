"""Single-batch calibration only. Quantize each eligible projection just before
its first call. Downstream inputs therefore see the same quantized prefix as
in sequential group replay. Never use first-batch Hessians as a substitute for
multi-batch Hessians. Intended for dense (not routed/shared projection) layers.
"""
from contextlib import ExitStack

import torch
from src.hybrid_quant import quantizable


def single_call_layer(layer,run,encode):
    last=None; gram=None; seen=set()
    def before(name,inputs):
        nonlocal last,gram
        if name in seen:raise RuntimeError(f'repeated linear {name}; use group replay')
        seen.add(name);x=inputs[0]
        if x is not last:
            last=x;gram=None
            xf=x.reshape(-1,x.shape[-1]).float();n=len(xf);h=xf.T@xf
            # Reference path sums one FP32 product in FP64, divides, rounds
            # once to FP32 and symmetrizes. Scaling by 2^-k is exact here.
            h=h.mul_(1./n) if n>0 and n&(n-1)==0 else h.double().div_(n).float()
            gram=(h+h.T).mul_(.5);del h,xf
        encode(name,gram)
    with ExitStack() as stack:
        for name,m in layer.named_modules():
            if quantizable(name,m):stack.callback(m.register_forward_pre_hook(lambda mod,ins,n=name:before(n,ins)).remove)
        return run()
