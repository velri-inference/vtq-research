"""WRC1: at exactly the VTQ_300 tile stride, trade 2 low scale bits for
error along a shared, calibration-selected Walsh direction per input tile.
The reference reader needs no Hessian, RNG, original matrix or fit state.
"""
import argparse
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import torch
from src import vtq,wrc1_format as fmt1
from src.exp09_group_order import run_model_ab,vtq_roundtrip
from src.vtq_format import apply_vtq_rot

MODES=['natural','scale6-only','walsh-dc','walsh-energy','walsh-fit','walsh-fit-zero']


def refit(w,h,p,hbb,walsh,mode):
    m,k=w.shape; nb=k//256;dev=w.device;blocks=torch.arange(nb,device=dev)
    dr=p['d_row'].float(); suh=p['suh'].float(); u=p['u_sb'].float()
    # G is exactly an integer multiple of 1/4096 in VTQ_300. Recover it
    # without carrying a second MxK FP32 array out of the main quantizer.
    origscale=dr[:,None]*(u/128.)
    g=((p['W_hat_rot'].view(m,nb,256)/origscale.clamp_min(1e-30)[:,:,None])*4096).round()/4096
    ix=(torch.round(u/4).clamp(1,64)-1).byte(); scale=dr[:,None]*((ix.float()+1)*(4./128.))
    rot=g*scale[:,:,None]; del g
    def inv(x):return apply_vtq_rot(x.reshape(m,k),transpose=True)*suh
    q=inv(rot)
    if mode=='scale6-only' or m<=16:
        p['wrc']=(ix,torch.zeros_like(ix),torch.zeros(nb,device=dev,dtype=torch.uint8),torch.zeros(nb,device=dev,dtype=torch.float16),torch.zeros(nb,device=dev,dtype=torch.uint8))
        return q
    if mode=='walsh-dc': candidates=torch.zeros((nb,1),device=dev,dtype=torch.long)
    else:
        var=((walsh@hbb)*walsh).sum(-1)  # unnormalized variance of each +/- Walsh projection
        candidates=var.topk(1 if mode=='walsh-energy' else 8,dim=1).indices
    J=candidates.shape[1];r0=torch.arange(nb*J,device=dev)
    vec=torch.zeros(nb*J,nb,256,device=dev);vec[r0,r0//J]=walsh[candidates.flatten()]
    basis=apply_vtq_rot(vec.reshape(nb*J,k),transpose=True)*suh;del vec
    reg=.10*h.diagonal().mean().clamp_min(1e-12);bh=(basis@h).add_(reg*basis); rhs=(w-q)@bh.T
    diag=(basis*bh).sum(1).clamp_min(1e-12); scores=(rhs[16:].square().sum(0)/diag).view(nb,J)
    chosen=scores.argmax(1); jj=blocks*J+chosen;direction=candidates[blocks,chosen]
    b=basis[jj]; gram=bh[jj]@b.T;gram=(gram+gram.T)*.5;rhs=rhs[:,jj].contiguous();del basis,bh,b
    # Joint continuous coefficients only initialize the per-tile four levels.
    gram.diagonal().add_(1e-4*gram.diagonal().mean().clamp_min(1e-12)); cc=torch.linalg.solve(gram,rhs.T).T
    normalized=(cc[16:]/scale[16:].clamp_min(1e-20)).abs();weight=scale[16:].square()
    sig=((weight*normalized.square()).sum(0)/weight.sum(0).clamp_min(1e-25)).sqrt();inner=.45*sig;outer=1.5*sig
    for _ in range(5):
        if mode=='walsh-fit-zero':inner.zero_()
        high=normalized>((inner+outer)*.5); wh=weight*high;wl=weight*(~high)
        inner=torch.where(wl.sum(0)>1e-24,(wl*normalized).sum(0)/wl.sum(0).clamp_min(1e-25),inner)
        outer=torch.where(wh.sum(0)>1e-24,(wh*normalized).sum(0)/wh.sum(0).clamp_min(1e-25),outer)
    if mode=='walsh-fit-zero': inner.zero_()
    outer=outer.clamp(0,65504).half();ratio=(256*inner/outer.float().clamp_min(1e-20)).round().clamp(0,255).byte();out=outer.float();inn=out*(ratio.float()/256.)
    lev=torch.stack([-out,-inn,inn,out],1);val=torch.zeros_like(scale);code=torch.zeros_like(ix);res=rhs.clone()
    for order in (range(nb),range(nb-1,-1,-1)):
        for j in order:
            old=val[:,j];target=old+res[:,j]/gram[j,j].clamp_min(1e-20); trial=scale[:,j,None]*lev[j];best=(trial-target[:,None]).abs().argmin(1)
            new=scale[:,j]*lev[j,best];new[:16]=0;delta=new-old;res.sub_(delta[:,None]*gram[j]);val[:,j]=new;code[:,j]=best.byte()
    p['wrc']=(ix,code,direction.byte(),outer,ratio)
    return inv(rot+val[:,:,None]*walsh[direction])


def experiment():
    walsh=torch.from_numpy(fmt1.walsh()).cuda();meta={}
    @contextmanager
    def ctx(mode):
        meta.clear();meta.update(params=0,tile_and_scale_bytes=0,file_bytes=0)
        if mode=='natural':yield meta;return
        orig_s,orig_q=vtq.compute_suh_vtq,vtq.quantize_matrix_vtq;tmp={}
        def prep(*a,**kw):
            wr,hr,s=orig_s(*a,**kw)
            if mode not in ('scale6-only','walsh-dc'):
                n=hr.shape[0]//256;b=torch.arange(n,device=hr.device);tmp['hbb']=hr.view(n,256,n,256)[b,:,b,:].contiguous()
            return wr,hr,s
        def quant(w,h,*a,**kw):
            qd,q,_=orig_q(w,h,*a,**kw);del q;wt=kw.get('W_target');p=vtq.PACK['matrix']
            return qd,refit(w if wt is None else wt,h,p,tmp.pop('hbb',None),walsh,mode),float('nan')
        with patch.object(vtq,'compute_suh_vtq',prep),patch.object(vtq,'quantize_matrix_vtq',quant):yield meta
    def reload(path,p,fmt,method):
        m,k=p['d_row'].numel(),p['suh'].numel();payload=1616*(m//16)*(k//256)+2*(m+k);meta['params']+=m*k;meta['tile_and_scale_bytes']+=payload
        if method=='natural':
            q=vtq_roundtrip(path,p,fmt,method);meta['file_bytes']+=path.stat().st_size;return q
        blob=fmt1.serialize(p,fmt);assert len(blob)==payload+48;meta['file_bytes']+=len(blob);return fmt1.decode(blob,'cuda')
    return ctx,reload


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--model',required=True);ap.add_argument('--mode',choices=['discover','confirm'],required=True);ap.add_argument('--text',type=Path,default=Path('data/wikitext2_test.raw'));ap.add_argument('--output',type=Path,required=True);ap.add_argument('--seeds',nargs='+',type=int,default=[1000,2000]);ap.add_argument('--winner');a=ap.parse_args();a.specs=['VTQ_300'];modes=MODES
    if a.mode=='confirm':
        if a.winner not in MODES:ap.error('fix a discovery winner before confirmation')
        modes=list(dict.fromkeys(['natural',a.winner]))
    ctx,reload=experiment();run_model_ab(a,modes,ctx,reload,dict(format='WRC1',tile_bytes=1616,scale_bits=6,coefficient_bits=2))


if __name__=='__main__':main()
