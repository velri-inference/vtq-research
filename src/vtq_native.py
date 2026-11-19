"""Optional native CPU serializer and FP32 materializer for the VTQ Python IO API.

Build the shared library explicitly (see docs/RU/vtq-native-io.md). Importing this
module does not compile anything or load third-party code. Format files never
control the library path. This is not an inference matvec implementation.
"""
import ctypes as ct
import operator
from pathlib import Path

import numpy as np

from src.vtq_format import FMT_META, VtqFormat, galois_lb_vtq
from src.vtq_v2_format import CW8_TO_IDX, HAM_CW8


def _dimension(value):
    value = operator.index(value)
    if not 0 < value < 2**31:
        raise ValueError(f'invalid VTQ dimension: {value}')
    return value


def _check_format(fmt):
    if FMT_META.get(fmt.fmt_code) != (fmt.K, fmt.gosset, fmt.nl, fmt.h, fmt.st_bytes):
        raise ValueError('unsupported VTQ format geometry')


class NativeVtqCodec:
    """Pass the same instance as codec= to VtqWriter and/or VtqFile."""

    def __init__(self, library: Path):
        self._so = ct.CDLL(str(Path(library).resolve(strict=True)))
        abi = self._so.vtq_io_abi
        abi.argtypes, abi.restype = [], ct.c_int
        if abi() != 1:
            raise ValueError('incompatible VTQ IO native ABI')
        self._pack = self._so.vtq_pack
        self._pack.argtypes = [ct.c_void_p]*6 + [ct.c_int]*6 + [ct.c_size_t]
        self._pack.restype = ct.c_int
        self._unpack = self._so.vtq_unpack
        self._unpack.argtypes = [ct.c_void_p, ct.c_size_t]*2 + [ct.c_void_p]*5 + [ct.c_int]*6
        self._unpack.restype = ct.c_int
        self._nibbles = np.ascontiguousarray(CW8_TO_IDX, dtype=np.int16)
        self._ham = np.ascontiguousarray(HAM_CW8, dtype=np.uint8)
        self._labels = {nl: np.ascontiguousarray(galois_lb_vtq(np.arange(1024), nl), dtype=np.uint8)
                        for nl in (4, 8, 16)}
        assert self._nibbles.shape == (256,) and self._ham.shape == (16,)

    def pack(self, c, u, g, usb, fmt: VtqFormat):
        _check_format(fmt)
        c, u, g, usb = (np.ascontiguousarray(x, dtype=np.uint8) for x in (c, u, g, usb))
        if c.ndim != 4:
            raise ValueError('expected a four-dimensional coded bit plane')
        m, ns = (_dimension(i) for i in c.shape[:2])
        sh = (m, ns)
        if (c.shape, u.shape, g.shape, usb.shape) != (sh+(16,2), sh+(256,), sh+(16,), sh):
            raise ValueError('incompatible VTQ code plane shapes')
        out = np.empty(((m+15)//16)*ns*fmt.st_bytes, dtype=np.uint8)
        err = self._pack(*(v.ctypes.data for v in (c,u,g,usb,self._nibbles,out)),
                         m,ns,fmt.K,fmt.gosset,fmt.h,fmt.st_bytes,out.nbytes)
        if err:
            raise ValueError(f'invalid VTQ packing input (status {err})')
        return out

    def unpack(self, raw, M, K, fmt: VtqFormat, d_row=None):
        _check_format(fmt)
        m,k = _dimension(M),_dimension(K)
        if k%256:
            raise ValueError('K must be a multiple of 256')
        ns = k//256
        buf = np.frombuffer(raw, dtype=np.uint8)
        if buf.nbytes < ((m+15)//16)*ns*fmt.st_bytes:
            raise ValueError('truncated VTQ tile data')
        tab = np.ascontiguousarray(fmt.lut, dtype=np.float32)
        gains = np.ascontiguousarray(fmt.gain_tab, dtype=np.float32)
        if tab.size != fmt.nl*fmt.U or gains.size != 1<<fmt.h:
            raise ValueError('invalid LUT or gain table length')
        dr = (np.ones(m, dtype=np.float32) if d_row is None else
              np.ascontiguousarray(np.asarray(d_row, dtype=np.float16), dtype=np.float32).reshape(m))
        out = np.empty((m,k), dtype=np.float32)
        err = self._unpack(buf.ctypes.data,buf.nbytes,out.ctypes.data,out.size,
                           *(v.ctypes.data for v in (tab,gains,self._labels[fmt.nl],self._ham,dr)),
                           m,ns,fmt.K,fmt.gosset,fmt.h,fmt.st_bytes)
        if err:
            raise ValueError(f'invalid VTQ unpacking input (status {err})')
        return out
