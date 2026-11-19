"""TSB1 experimental 3-bit trellis shape-bank matrix. Not readable as .vtq.
Fixed 16-state-label map and 16x256 tile from VTQ_300; the group gain header
also indexes a 32-byte magnitude table. Sizes/offsets are fully deterministic.
"""
import struct

import numpy as np
import torch

from src.vtq_format import (apply_vtq_rot, decode_supertiles_vtq, format_from_json_vtq,
                            pack_supertiles_vtq, reconstruct_lut_from_hw_vtq)

HEAD = struct.Struct('<4sII')
MAGIC = b'TSB1'


def hw(bank):
    return dict(type='u8',qmax=127,tab_div=64,tab_u8=bank.tolist())


def encode_bytes(packed, banks, fmt):
    banks=np.asarray(banks,dtype=np.uint8)
    if fmt.fmt_code!=31 or banks.shape!=(4,32) or banks.max()>127 or packed['rot_block']!=-1:
        raise ValueError('TSB1 requires four VTQ_300 magnitude banks and default rotation')
    def arr(k): return packed[k].detach().cpu().contiguous().numpy()
    d,s=arr('d_row').astype('<f2'),arr('suh').astype('<f2')
    if packed['row_slices']!=[(0,len(d),'VTQ_300')]: raise ValueError('expected one uniform matrix')
    tiles=pack_supertiles_vtq(arr('c_all'),arr('u_all'),arr('g_sub'),arr('u_sb'),fmt)
    return b''.join([HEAD.pack(MAGIC,len(d),len(s)),banks.tobytes(),fmt.gain_u8.tobytes(),d.tobytes(),s.tobytes(),tiles.tobytes()])


def decode_bytes(blob,device='cpu'):
    if len(blob)<HEAD.size: raise ValueError('short TSB1 header')
    magic,m,k=HEAD.unpack_from(blob)
    if magic!=MAGIC or not m or not k or m%16 or k%256 or len(blob)!=HEAD.size+132+2*(m+k)+(m//16)*(k//256)*1616:
        raise ValueError('invalid TSB1 matrix')
    banks=np.frombuffer(blob,np.uint8,128,HEAD.size).reshape(4,32)
    gains=np.frombuffer(blob,np.uint8,4,HEAD.size+128)
    if banks.max()>127 or not gains.all(): raise ValueError('invalid levels or gains')
    fmt=format_from_json_vtq(dict(name='TSB1',fmt_code=31,hw_codebook=hw(banks[0]),gain_u8=gains))
    tables=np.stack([reconstruct_lut_from_hw_vtq(31,hw(b)).reshape(16,4) for b in banks])
    off=HEAD.size+132
    dr=np.frombuffer(blob,'<f2',m,off); suh=np.frombuffer(blob,'<f2',k,off+2*m).astype(np.float32)
    rot=decode_supertiles_vtq(memoryview(blob)[off+2*(m+k):],m,k,fmt,dr,gain_luts=tables)
    return apply_vtq_rot(torch.from_numpy(rot).to(device),transpose=True)*torch.from_numpy(suh).to(device)
