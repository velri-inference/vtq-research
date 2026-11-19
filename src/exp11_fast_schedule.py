"""Bounded-memory prototype for the exp09 (first pass only) act-desc schedule.
Gathers both Hessian axes without materializing broadcast int64 index grids.
The second kernel fuses the exact same discrete objective, including its
monotonic accept/reject check, rather than dropping encoder work from timings.
"""
from contextlib import contextmanager, ExitStack
from unittest.mock import patch

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

from src import vtq
from src.exp09_group_order import order_first_pass


@triton.jit
def _square(X,P,Y,K:tl.constexpr,S0:tl.constexpr,S1:tl.constexpr):
    r=32*tl.program_id(0)+tl.arange(0,32); c=32*tl.program_id(1)+tl.arange(0,32)
    i=tl.load(P+r); j=tl.load(P+c)
    tl.store(Y+r[:,None]*K+c[None,:],tl.load(X+i[:,None]*S0+j[None,:]*S1))


def gather_square(h,cols):
    k=h.shape[0]; assert h.shape==(k,k) and k%32==0 and cols.shape==(k,)
    y=torch.empty((k,k),device=h.device,dtype=h.dtype)
    _square[(k//32,k//32)](h,cols,y,k,*h.stride()); return y


@triton.jit
def _refit(A,B,STEP,UI,DI,U,D,SA0:tl.constexpr,SA1:tl.constexpr,SB0:tl.constexpr,SB1:tl.constexpr):
    r=tl.program_id(0); j=tl.arange(0,256)
    a=tl.load(A+r*SA0+j*SA1); b=tl.load(B+r*SB0+j*SB1)
    di=tl.load(DI+r); step=tl.load(STEP+r); ui=tl.load(UI+r)
    den=tl.sum(b*b,0); num=tl.sum(a*b,0)
    dc=tl.maximum(tl.where(den>1e-12,tl.div_rn(num,tl.maximum(den,1e-12)),di),1e-10)
    un=tl.maximum(1.,tl.minimum(255.,libdevice.rint(tl.div_rn(dc,step))))
    e0=a-di*b; e1=a-(step*un)*b
    u=tl.where(tl.sum(e1*e1,0)<tl.sum(e0*e0,0),un,ui)
    tl.store(U+r,u); tl.store(D+r,step*u)


def refit_block(a,b,step,ui,di):
    assert a.shape==b.shape and a.shape[1]==256
    u,d=torch.empty_like(ui),torch.empty_like(di)
    _refit[(a.shape[0],)](a,b,step,ui,di,u,d,*a.stride(),*b.stride(),enable_fp_fusion=False)
    return u,d


@contextmanager
def schedule(mode):
    with ExitStack() as st:
        if mode in ('fast-order','fast-fused'):
            st.enter_context(order_first_pass('act-desc',lowmem=True))
        elif mode=='act-desc':
            st.enter_context(order_first_pass(mode))
        elif mode not in ('natural','fused-only'):
            raise ValueError(mode)
        if mode in ('fast-fused','fused-only'):
            st.enter_context(patch.object(vtq,'_refit_block_u8',refit_block))
        yield
