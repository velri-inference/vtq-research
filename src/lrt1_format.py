"""LRT1: pay for an activation-aware low-rank correction with 1 freed gain bit/16 weights.

Canonical VTQ_300 coding planes; only two gains => 1584 vs 1616 bytes per
supertile. A signed int8 residual pair A[M,r] B[r,K] uses r*(M+K+4) bytes
including one FP32 product scale per rank. Even the standalone file must fit
the scalar VTQ tile+row+channel payload *without* counting VTQ's own header.
"""
import struct

import numpy as np
import torch

from src.vtq_format import VtqFormat, apply_vtq_rot, decode_supertiles_vtq, format_from_json_vtq, pack_supertiles_vtq

HEAD=struct.Struct('<4sIII')
GAINS=np.asarray([57,72],np.uint8)
OVERHEAD=HEAD.size+32+2


class LrtFormat(VtqFormat):
    @property
    def st_bytes(self): return 1584


def make_format(table,gains):
    base=format_from_json_vtq(dict(name='LRT1',fmt_code=31,gain_u8=gains,hw_codebook=dict(type='u8',qmax=127,tab_div=64,tab_u8=np.asarray(table).tolist())))
    return LrtFormat(**dict(vars(base),h=1))


def budget_rank(m,k):
    ntiles=(m//16)*(k//256)
    return max(0,(32*ntiles-OVERHEAD)//(m+k+4))


def apply_residual(q,a,b,s):
    if not len(s):return q
    return torch.addmm(q,a.float()*s[None,:],b.float())


def fit_residual(w,h,q,r,seed,weighted=True):
    """Randomized eigenbasis for E (H + .1 mean(diag H) I) E.T, followed by
    integer factor packing. Never materializes an M x M covariance or a K x K Cholesky."""
    m,k=w.shape;dev=w.device
    if not r:return (torch.empty((m,0),device=dev,dtype=torch.int8),torch.empty((0,k),device=dev,dtype=torch.int8),torch.empty(0,device=dev))
    e=w.float() if q is None else w.float()-q
    l=min(r+8,m,k); generator=torch.Generator(device=dev).manual_seed(seed)
    v=torch.randn(m,l,device=dev,generator=generator); reg=.1*h.diagonal().mean() if weighted else 0.
    def hmul(t): return (h@t).add_(reg*t) if weighted else t
    # Keep the low-dimensional sketches numerically conditioned after each pass.
    for _ in range(3):
        v=torch.linalg.qr(e@hmul(e.T@v),mode='reduced')[0]
    t=e.T@v; gram=t.T@hmul(t); _,rot=torch.linalg.eigh((gram+gram.T)*.5);rot=rot[:,-r:]
    a=v@rot; b=rot.T@t.T
    sa=a.abs().amax(0).clamp_min(1e-20)/127;sb=b.abs().amax(1).clamp_min(1e-20)/127
    return ((a/sa).round().clamp(-127,127).to(torch.int8),(b/sb[:,None]).round().clamp(-127,127).to(torch.int8),sa*sb)


def serialize(p,fmt):
    def arr(t):return t.detach().cpu().contiguous().numpy()
    dr=arr(p['d_row']).astype('<f2');s=arr(p['suh']).astype('<f2');m,k=len(dr),len(s)
    a,b,rs=map(arr,p['lr']);r=len(rs);cap=1616*(m//16)*(k//256)+2*(m+k)
    if m%16 or k%256 or p['rot_block']!=-1 or r>budget_rank(m,k):raise ValueError('invalid LRT1 matrix')
    tiles=pack_supertiles_vtq(arr(p['c_all']),arr(p['u_all']),arr(p['g_sub']),arr(p['u_sb']),fmt)
    blob=b''.join((HEAD.pack(b'LRT1',m,k,r),np.asarray(fmt.hw_codebook['tab_u8'],np.uint8).tobytes(),fmt.gain_u8.tobytes(),dr.tobytes(),s.tobytes(),tiles.tobytes(),a.tobytes(),b.tobytes(),rs.astype('<f4').tobytes()))
    if len(blob)>cap:raise ValueError('even rank zero cannot fit budget for this tiny matrix')
    return blob


def decode(blob,device='cpu'):
    if len(blob)<OVERHEAD:raise ValueError('short LRT1 header')
    tag,m,k,r=HEAD.unpack_from(blob);nt=(m//16)*(k//256); cap=1616*nt+2*(m+k)
    if tag!=b'LRT1' or not m or not k or m%16 or k%256 or len(blob)>cap or len(blob)!=OVERHEAD+1584*nt+2*(m+k)+r*(m+k+4):raise ValueError('invalid LRT1 matrix')
    off=HEAD.size
    def get(dtype,n):
        nonlocal off
        v=np.frombuffer(blob,dtype,n,off);off+=v.nbytes;return v
    table=get(np.uint8,32);g=get(np.uint8,2);fmt=make_format(table,g);dr=get('<f2',m);suh=get('<f2',k).astype(np.float32)
    rot=decode_supertiles_vtq(get(np.uint8,1584*nt),m,k,fmt,dr)
    q=apply_vtq_rot(torch.from_numpy(rot).to(device),transpose=True)*torch.from_numpy(suh).to(device)
    a=torch.from_numpy(get(np.int8,m*r).copy().reshape(m,r)).to(device);b=torch.from_numpy(get(np.int8,r*k).copy().reshape(r,k)).to(device);sc=torch.from_numpy(get('<f4',r).copy()).to(device)
    return apply_residual(q,a,b,sc)
