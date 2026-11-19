"""P2T1 reference matrix container: separate 2-bit trellis symbols and 4-bit 2D-vector IDs.
A 256-weight ring has 32 symbol bytes, 64 vector-ID bytes and 4 gain bytes.
16 rings + 16 block scales = 1616 bytes per supertile, as in VTQ_300.
This standalone matrix file includes the shared 8192-byte int8 book in full.
"""
import struct
import numpy as np
import torch

from src.exp12_pair_trellis import pair_labels
from src.vtq_format import apply_vtq_rot

HEADER=struct.Struct('<4sIIH')


def serialize(packed,book,gains,states):
    def arr(k):return packed[k].detach().cpu().contiguous().numpy()
    dr,s=arr('d_row').astype('<f2'),arr('suh').astype('<f2'); m,k=len(dr),len(s); ng,nb=m//16,k//256
    if packed['rot_block']!=-1 or m%16 or k%256 or states not in (256,1024,2048): raise ValueError('unsupported P2T1 matrix')
    c=arr('c_all').reshape(ng,16,nb,16,2).transpose(0,2,3,1,4).reshape(ng,nb,16,32)
    u=arr('u_all')[:,:,::2].reshape(ng,16,nb,16,8).transpose(0,2,3,1,4).reshape(ng,nb,16,128)
    u=u[...,::2]|(u[...,1::2]<<4)
    g=arr('g_sub').reshape(ng,16,nb,16).transpose(0,2,3,1);g=g[...,::4]|(g[...,1::4]<<2)|(g[...,2::4]<<4)|(g[...,3::4]<<6)
    usb=arr('u_sb').reshape(ng,16,nb).transpose(0,2,1)
    data=np.concatenate([np.concatenate([c,u,g],axis=-1).reshape(ng,nb,1600),usb],axis=-1)
    return b''.join((HEADER.pack(b'P2T1',m,k,states),np.asarray(book,np.int8).reshape(256,16,2).tobytes(),
                     np.asarray(gains,np.uint8).reshape(4).tobytes(),dr.tobytes(),s.tobytes(),data.tobytes()))


def decode(blob,device='cpu'):
    if len(blob)<HEADER.size:raise ValueError('short P2T1 header')
    magic,m,k,S=HEADER.unpack_from(blob); ng=m//16; nb=k//256
    if magic!=b'P2T1' or not m or not k or m%16 or k%256 or S not in (256,1024,2048) or len(blob)!=HEADER.size+8196+2*(m+k)+ng*nb*1616: raise ValueError('invalid P2T1 file')
    off=HEADER.size;book=np.frombuffer(blob,np.int8,8192,off).reshape(256,16,2).astype(np.float32)/64;off+=8192
    gains=np.frombuffer(blob,np.uint8,4,off).astype(np.float32)/64;off+=4
    dr=np.frombuffer(blob,'<f2',m,off).astype(np.float32);off+=2*m;s=np.frombuffer(blob,'<f2',k,off).astype(np.float32);off+=2*k
    data=np.frombuffer(blob,np.uint8,offset=off).reshape(ng,nb,1616);rings=data[:,:,:1600].reshape(ng,nb,16,100)
    def unpack(v,bits):return ((v[...,None]>>(bits*np.arange(8//bits,dtype=np.uint8)))&((1<<bits)-1)).reshape(*v.shape[:-1],-1)
    codes=unpack(rings[...,:32],2).astype(np.uint16); index=unpack(rings[...,32:96],4); g=np.repeat(unpack(rings[...,96:],2),8,axis=-1)
    state=codes.copy()
    for lag in range(1,((S.bit_length()-1)+1)//2):state |= np.roll(codes,lag,axis=-1)<<(2*lag)
    values=book[pair_labels(S)[state&(S-1)],index]*gains[g][...,None]
    mat=values.reshape(ng,nb,16,16,16).transpose(0,3,1,2,4).reshape(m,nb,256)
    usb=data[:,:,-16:].transpose(0,2,1).reshape(m,nb).astype(np.float32)/128
    rot=(mat*(dr[:,None]*usb)[:,:,None]).reshape(m,k)
    return apply_vtq_rot(torch.from_numpy(rot).to(device),transpose=True)*torch.from_numpy(s).to(device)
