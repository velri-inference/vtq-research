"""WRC1: 6-bit block scale + 2-bit Walsh-direction residual, exactly 1616 B/st16.

For each column block the first 16-row supertile stores twelve bytes of 6-bit
scales and four bytes of shared direction/level metadata. These first 16 rows
have no correction. Every other row uses one byte: (scale_index<<2)|level.
No increase in tile stride or number/size of global lookup tables.
"""
import struct
import numpy as np
import torch

from src.vtq_format import apply_vtq_rot,decode_supertiles_vtq,format_from_json_vtq,pack_supertiles_vtq

HEAD=struct.Struct('<4sII')


def walsh():
    a=np.arange(256,dtype=np.uint8); v=a[:,None]&a[None,:]
    v=v^(v>>4);v=v^(v>>2);v=v^(v>>1)
    return 1.-2.*(v&1).astype(np.float32)


def serialize(p,fmt):
    def ar(t):return t.detach().cpu().contiguous().numpy()
    dr,s=ar(p['d_row']).astype('<f2'),ar(p['suh']).astype('<f2'); m,k=len(dr),len(s);ng=m//16; nb=k//256
    if not ng or m%16 or k%256 or fmt.fmt_code!=31 or p['rot_block']!=-1:raise ValueError('unsupported WRC1 matrix')
    idx,g,direction,outer,ratio=[ar(t) for t in p['wrc']]
    tiles=pack_supertiles_vtq(ar(p['c_all']),ar(p['u_all']),ar(p['g_sub']),ar(p['u_sb']),fmt).reshape(ng,nb,1616)
    hdr=tiles[:,:,-16:]; ix=idx.reshape(ng,16,nb).transpose(0,2,1); code=g.reshape(ng,16,nb).transpose(0,2,1)
    hdr[1:]=(ix[1:]<<2)|code[1:]
    bits=(ix[0,:,:,None]>>np.arange(6,dtype=np.uint8))&1
    hdr[0,:,:12]=np.packbits(bits.reshape(nb,96),axis=1,bitorder='little');hdr[0,:,12]=direction
    hdr[0,:,13:15]=np.ascontiguousarray(outer.astype('<f2')).view(np.uint8).reshape(nb,2);hdr[0,:,15]=ratio
    return b''.join([HEAD.pack(b'WRC1',m,k),np.asarray(fmt.hw_codebook['tab_u8'],np.uint8).tobytes(),fmt.gain_u8.tobytes(),dr.tobytes(),s.tobytes(),tiles.tobytes()])


def decode(blob,device='cpu'):
    if len(blob)<HEAD.size+36:raise ValueError('short WRC1 header')
    tag,m,k=HEAD.unpack_from(blob);ng=m//16;nb=k//256
    if tag!=b'WRC1' or not m or not k or m%16 or k%256 or len(blob)!=HEAD.size+36+2*(m+k)+1616*ng*nb:raise ValueError('invalid WRC1 matrix')
    off=HEAD.size;t=np.frombuffer(blob,np.uint8,32,off);gain=np.frombuffer(blob,np.uint8,4,off+32);off+=36
    fmt=format_from_json_vtq(dict(name='WRC1',fmt_code=31,gain_u8=gain,hw_codebook=dict(type='u8',qmax=127,tab_div=64,tab_u8=t.tolist())))
    dr=np.frombuffer(blob,'<f2',m,off).astype(np.float32);s=np.frombuffer(blob,'<f2',k,off+2*m).astype(np.float32)
    raw=np.frombuffer(blob,np.uint8,offset=off+2*(m+k)).reshape(ng,nb,1616);hdr=raw[:,:,-16:]
    ix=hdr>>2;bits=np.unpackbits(hdr[0,:,:12],axis=1,bitorder='little').reshape(nb,16,6);ix[0]=(bits<<np.arange(6,dtype=np.uint8)).sum(2)
    outer=np.ascontiguousarray(hdr[0,:,13:15]).view('<f2').reshape(nb).astype(np.float32);inner=outer*(hdr[0,:,15].astype(np.float32)/256.)
    levels=np.stack([-outer,-inner,inner,outer],axis=1);code=(hdr&3).transpose(0,2,1).reshape(m,nb)
    corr=levels[np.arange(nb),code];corr[:16]=0;patterns=walsh()[hdr[0,:,12]]
    scale=dr[:,None]*((ix.transpose(0,2,1).reshape(m,nb).astype(np.float32)+1)*(4./128.))
    data=raw.copy();data[:,:,-16:]=128; base=decode_supertiles_vtq(data.reshape(-1),m,k,fmt).reshape(m,nb,256)
    rot=base*scale[:,:,None]+(scale*corr)[:,:,None]*patterns[None,:,:]
    return apply_vtq_rot(torch.from_numpy(rot.reshape(m,k)).to(device),transpose=True)*torch.from_numpy(s).to(device)
