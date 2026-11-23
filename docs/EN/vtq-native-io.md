# Optional Native CPU Backend for the Python VTQ Writer and Reader

`src.vtq_native.NativeVtqCodec` packs already-computed quantization codes into `.vtq` supertiles and unpacks `W_rot` to FP32. It does **not** alter Viterbi search, format selection, on-disk byte layout, or runtime matvec kernels. Omitting `codec=` keeps the NumPy reference path. The public API accepts the codec object explicitly: importing the module never invokes a compiler, downloads code, or reads a shared-library path from a quantized checkpoint.

Example build on the target host (validated on Colab G4):

```sh
c++ -O3 -std=c++17 -march=native -ffp-contract=off -fPIC -shared \
    src/native/vtq_pack_tiles.cpp src/native/vtq_unpack_tiles.cpp -o /tmp/libvtq_io.so
python -m src.test_vtq_native_io --library /tmp/libvtq_io.so
```

For portable builds, omit `-march=native`; never pass `-ffast-math`. Load only trusted local shared libraries.

```python
from pathlib import Path
from src.vtq_native import NativeVtqCodec
from src.vtq_format import VtqWriter, VtqFile

codec = NativeVtqCodec(Path('/tmp/libvtq_io.so'))
# Writing: VtqWriter(path, formats, codec=codec); add_matrix / finalize as usual.
reader = VtqFile(path, codec=codec)
try:
    weights = reader.decode_matrix(prefix, device='cuda')
finally:
    reader.mm.close()
    reader.f.close()
```

Compatibility requires no file-header changes. `test_vtq_native_io` verifies all four cross-combinations of writer/reader, partial tiles, multi-format row slices within a single matrix, invalid codewords, raw BF16 tensors, and FP32 outputs after inverse rotation. [Colab G4 results](../../results/exp26_native_io_api.json): files match byte-for-byte, reconstructed FP32 weights match bit-for-bit on CPU and CUDA, and 37 negative tests pass on every path. Additionally, the [public `codec=` API on real checkpoints](../../results/exp26_public_api_samples.json) was verified on the largest saved matrix of each of the 8 standard and 2 control models: three load pairs per model produced identical FP32 hashes. Note that `verify=True` in `VtqWriter` intentionally retains the independent NumPy decoder. Experimental TSB1 with gain-conditioned LUT banks is outside standard VTQ and is not supported by this API. Benchmarks distinguish tile packing time, full model export time, GPU weight materialization, and inference decode: see the [format & benchmark specification](vtq-format.md).
