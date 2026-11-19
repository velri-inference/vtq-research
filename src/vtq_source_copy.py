"""Copy unchanged safetensors payloads to a VTQ writer without materializing tensors.

Only use on known-unchanged source keys. Layout conversion, updated tensors and
newly quantized rows must still use the normal VTQ tensor/matrix writer.
"""
import errno
import json
import math
import os
from pathlib import Path
import struct

_TYPES = {'F32': ('f32',4), 'F16': ('f16',2), 'BF16': ('bf16',2), 'I32': ('i32',4), 'I8': ('i8',1), 'U8': ('u8',1)}


class FrozenTensorCopier:
    def __init__(self):
        self._files = {}
        self._buf = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        for f,_,_,_ in self._files.values():
            f.close()
        self._files.clear()

    def _shard(self, path):
        path=Path(path)
        if path not in self._files:
            f=path.open('rb', buffering=0)
            try:
                size=os.fstat(f.fileno()).st_size
                header_len,=struct.unpack('<Q',f.read(8))
                if header_len>min(size-8,100_000_000):raise ValueError('bad safetensors header length')
                hdr=json.loads(f.read(header_len));self._files[path]=(f,hdr,header_len+8,size)
            except BaseException:
                f.close();raise
        return self._files[path]

    def add(self, writer, name, path):
        f,hdr,base,size=self._shard(path);meta=hdr[name];dtype,width=_TYPES[meta['dtype']];shape=meta['shape'];start,end=meta['data_offsets']
        if (not isinstance(shape,list) or not all(type(d) is int and d>=0 for d in shape) or type(start) is not int or type(end) is not int or not 0<=start<=end<=size-base or end-start!=width*math.prod(shape)):
            raise ValueError(f'invalid source tensor {name}')
        if name in writer.tensors:raise ValueError(f'duplicate VTQ tensor {name}')
        writer._pad(64);writer.f.flush();n=end-start;off=base+start;remaining=n
        while remaining:
            try:copied=os.sendfile(writer.f.fileno(),f.fileno(),off,min(remaining,1<<30))
            except (AttributeError,NotImplementedError):break
            except OSError as e:
                if e.errno in (errno.EINVAL,errno.EXDEV,errno.ENOSYS,errno.EOPNOTSUPP):break
                raise
            if copied==0:raise EOFError(f'truncated source {name}')
            remaining-=copied;off+=copied
        if remaining:
            if self._buf is None:self._buf=bytearray(1<<20)
            f.seek(off);buf=memoryview(self._buf)
            while remaining:
                got=f.readinto(buf[:min(remaining,len(buf))]);
                if not got:raise EOFError(f'truncated source {name}')
                writer.f.write(buf[:got]);remaining-=got
        writer.tensors[name]={'dtype':dtype,'shape':list(shape),'offset':writer.pos};writer.pos+=n
        return n
