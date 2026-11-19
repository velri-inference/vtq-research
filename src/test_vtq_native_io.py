"""Explicit shared-library interoperability gate. Build elsewhere; no JIT on import."""
import argparse
from contextlib import closing
import json
from pathlib import Path
import tempfile

import numpy as np
import torch
from src import vtq_format as vf
from src.exp25_pack import data
from src.vtq_native import NativeVtqCodec


def check(library,device):
    codec=NativeVtqCodec(library);formats=list(vf.load_vtq_formats().values());ns=3;planes=[[] for _ in range(4)];slices=[];offset=0;checks=0
    for i,fmt in enumerate(formats):
        m=15+(i%5);inputs=data(m,ns*256,fmt,100+i);slab=vf.pack_supertiles_vtq(*inputs,fmt);dr=np.linspace(.25,2,m,dtype=np.float16)
        assert np.array_equal(codec.pack(*inputs,fmt),slab)
        assert np.array_equal(codec.unpack(slab,m,ns*256,fmt,dr).view('u4'),vf.decode_supertiles_vtq(slab,m,ns*256,fmt,dr).view('u4'))
        try:codec.unpack(slab[:-1],m,ns*256,fmt,dr)
        except ValueError:checks+=1
        else:raise AssertionError('accepted truncated tile')
        if fmt.gosset:
            bad=inputs[1].copy();bad.flat[0]^=1
            try:codec.pack(inputs[0],bad,inputs[2],inputs[3],fmt)
            except ValueError:checks+=1
            else:raise AssertionError('accepted invalid lattice codeword')
        for arr,x in zip(planes,inputs):arr.append(x)
        slices.append((offset,offset+m,fmt.name));offset+=m
    rng=np.random.default_rng(405);p=dict(zip(['c_all','u_all','g_sub','u_sb'],[np.concatenate(arr,axis=0) for arr in planes]));p.update(d_row=torch.from_numpy(rng.normal(size=offset).astype(np.float16)),suh=torch.from_numpy(rng.choice([-1.,1.],size=ns*256).astype(np.float16)),row_slices=slices,rot_block=-1);meta={'quantization':{'formats':[vf.format_json_vtq(f) for f in formats]}}
    with tempfile.TemporaryDirectory() as td:
        paths=[Path(td)/n for n in ('reference.vtq','native.vtq')]
        for path,impl in zip(paths,(None,codec)):
            w=vf.VtqWriter(path,formats,reserve=65536,codec=impl);w.add_matrix('m',p,verify=False);w.add_tensor('norm',torch.arange(96,dtype=torch.bfloat16));w.finalize(meta)
        assert paths[0].read_bytes()==paths[1].read_bytes();outputs=[]
        for path,impl in zip(paths[::-1],(None,codec)):
            r=vf.VtqFile(path,codec=impl)
            try:outputs.append(r.decode_matrix('m',device=device));assert torch.equal(r.tensor('norm'),torch.arange(96,dtype=torch.bfloat16))
            finally:r.mm.close();r.f.close()
        assert torch.equal(outputs[0].view(torch.int32),outputs[1].view(torch.int32))
    return dict(formats=len(formats),negative_checks=checks,mixed_slice_file_identical=True,decoded_fp32_bits_identical=True,device=device)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--library',type=Path,required=True);ap.add_argument('--output',type=Path);a=ap.parse_args();result=[check(a.library,d) for d in ('cpu','cuda')];print(result);a.output and a.output.write_text(json.dumps(result)+'\n')
