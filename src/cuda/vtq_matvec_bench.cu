// VTQ `16x256` Super-Tile Bank-Conflict-Free GPU Blackwell Benchmark
// (`nvcc -O3 -std=c++17 -arch=sm_120 src/cuda/vtq_matvec_bench.cu -o vtq_matvec_bench`)
//
// Core VTQ GPU Blackwell (`sm_120`) hardware innovations over VTQ v1 (`LTQ`) and VTQ v2:
//   1. `16 rows x 256 cols = 4,096 weights` Super-Tile Shared Byte-for-Byte with CPU AVX2 (`vtq_st16_k*_t`):
//      - Reuses each activation `x[j]` `8x..16x` in registers across all 16 rows of the super-tile (`16x` lower `x` traffic),
//      - Stores `u_sb[16]` (`uint8` linear superblock row scale) + per-row `fp16 d_row[M]` + per-format `float32 unit` (`0` inner-loop `fp16->fp32`).
//   2. Bank-Conflict-Free Replicated Shared-Memory Codebooks (`s_lut[N_ENT][32]`, `4..16 KB`):
//      - Because `nu = 8` (`256` states) uses `nl in {4, 8, 16}` subsets, the entire signed + complemented codebook has only
//        `32`, `64`, or `128` entries (`4 KB`, `8 KB`, or `16 KB` replicated across all 32 shared-memory banks!), eliminating the
//        1024-state shared-memory bank conflicts that bottleneck VTQ v1 and VTQ v2 at `25..50%` of GDDR7 bandwidth.
//   3. Single-Shuffle Packed `(gi << 24) | w23` 256-State Tail-Biting Window + Aligned Even/Odd Planes:
//      - Lanes `0..15` load `c[g][r][0..1]` + `hdr[g]` once per subblock `g`, pack `(gi_row << 24) | w23_row`, and deliver both
//        the 23-bit trellis window and the subblock gain index in a single `__shfl_sync`;
//      - `usgn`, `ub*`, `u2`, `u4` are laid out as `[even_rows[16] | odd_rows[16]]`, giving each lane `(odd, j)` all 8 rows'
//        uncoded bits from 1 coalesced 32-byte sector load with ZERO shuffles (and in `__dp4a` GEMM, a single `uint32_t` word
//        `4 * (r_lo & 1) + q` feeds BOTH `r0 = r_lo` and `r1 = 8 + r_lo` across `B = 1, 8, 32` tokens!).
#include <cuda_fp16.h>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CK(x) do { \
    cudaError_t e = (x); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA %s: %s (%s:%d)\n", #x, cudaGetErrorString(e), __FILE__, __LINE__); \
        exit(1); \
    } \
} while (0)

static __host__ __device__ inline uint32_t hash32(uint64_t i, uint32_t seed) {
    uint64_t x = i * 0x9E3779B97F4A7C15ull + seed * 0xD1B54A32D192ED03ull;
    x ^= x >> 31; x *= 0xBF58476D1CE4E5B9ull; x ^= x >> 29; x *= 0x94D049BB133111EBull; x ^= x >> 32;
    return (uint32_t)x;
}

__global__ void fill_bytes(uint8_t* p, size_t n, uint32_t seed) {
    size_t idx4 = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    uint32_t* p4 = (uint32_t*)p;
    size_t n4 = n >> 2;
    for (size_t i = idx4; i < n4; i += stride) p4[i] = hash32(i, seed);
}

// Identical `16x256` (`4,096`-weight) super-tile structs to `src/cpu/vtq_avx2_bench.c`:
struct vtq_st16_k15_t { uint8_t c[16][16][2]; uint8_t ham[16][16]; uint16_t hdr[16][1]; uint8_t u_sb[16]; };
struct vtq_st16_k20_t { uint8_t c[16][16][2]; uint8_t usgn[16][32]; uint16_t hdr[16][1]; uint8_t u_sb[16]; };
struct vtq_st16_k25_t { uint8_t c[16][16][2]; uint8_t usgn[16][32]; uint8_t ham[16][16]; uint16_t hdr[16][1]; uint8_t u_sb[16]; };
struct vtq_st16_k30_t { uint8_t c[16][16][2]; uint8_t usgn[16][32]; uint8_t ub0[16][32]; uint16_t hdr[16][2]; uint8_t u_sb[16]; };
struct vtq_st16_k35_t { uint8_t c[16][16][2]; uint8_t usgn[16][32]; uint8_t ub1[16][32]; uint8_t ham[16][16]; uint16_t hdr[16][2]; uint8_t u_sb[16]; };
struct vtq_st16_k40_t { uint8_t c[16][16][2]; uint8_t usgn[16][32]; uint8_t u2[16][2][32]; uint16_t hdr[16][3]; uint8_t u_sb[16]; };
struct vtq_st16_k50_t { uint8_t c[16][16][2]; uint8_t usgn[16][32]; uint8_t u2[16][2][32]; uint8_t ub2[16][32]; uint16_t hdr[16][3]; uint8_t u_sb[16]; };
struct vtq_st16_k60_t { uint8_t c[16][16][2]; uint8_t usgn[16][32]; uint8_t u4[16][4][32]; uint16_t hdr[16][3]; uint8_t u_sb[16]; };

static_assert(sizeof(vtq_st16_k15_t) ==  816, "st16_k15");
static_assert(sizeof(vtq_st16_k20_t) == 1072, "st16_k20");
static_assert(sizeof(vtq_st16_k25_t) == 1328, "st16_k25");
static_assert(sizeof(vtq_st16_k30_t) == 1616, "st16_k30");
static_assert(sizeof(vtq_st16_k35_t) == 1872, "st16_k35");
static_assert(sizeof(vtq_st16_k40_t) == 2160, "st16_k40");
static_assert(sizeof(vtq_st16_k50_t) == 2672, "st16_k50");
static_assert(sizeof(vtq_st16_k60_t) == 3184, "st16_k60");

template <int FMT>
__host__ __device__ __forceinline__ constexpr size_t st_bytes() {
    if constexpr (FMT == 15) return sizeof(vtq_st16_k15_t);
    if constexpr (FMT == 20) return sizeof(vtq_st16_k20_t);
    if constexpr (FMT == 25) return sizeof(vtq_st16_k25_t);
    if constexpr (FMT == 30 || FMT == 31) return sizeof(vtq_st16_k30_t);
    if constexpr (FMT == 35) return sizeof(vtq_st16_k35_t);
    if constexpr (FMT == 40) return sizeof(vtq_st16_k40_t);
    if constexpr (FMT == 50 || FMT == 51) return sizeof(vtq_st16_k50_t);
    return sizeof(vtq_st16_k60_t);
}

template <int FMT>
__host__ __device__ __forceinline__ constexpr int fmt_nl() {
    if constexpr (FMT == 30) return 8;
    if constexpr (FMT == 50) return 4;
    return 16;
}

template <int FMT>
__host__ __device__ __forceinline__ constexpr int fmt_ub() {
    if constexpr (FMT == 20) return 0;
    if constexpr (FMT == 15 || FMT == 25 || FMT == 30 || FMT == 31) return 1;
    if constexpr (FMT == 35 || FMT == 40) return 2;
    if constexpr (FMT == 50 || FMT == 51) return 3;
    return 4;
}

template <int FMT>
__host__ __device__ __forceinline__ constexpr int fmt_h() {
    if constexpr (FMT <= 25) return 1;
    if constexpr (FMT <= 35) return 2;
    return 3;
}

template <int FMT>
__host__ __device__ __forceinline__ constexpr int signed_lut_entries() {
    if constexpr (FMT == 15) return 32;
    return 2 * fmt_nl<FMT>() * (1 << fmt_ub<FMT>());
}

__constant__ uint8_t C_HAM_CW8[16] = {
    0x00, 0xff, 0xf0, 0x0f, 0xcc, 0x33, 0x3c, 0xc3,
    0xaa, 0x55, 0x5a, 0xa5, 0x66, 0x99, 0x96, 0x69
};

static const uint8_t HOST_HAM_CW8[16] = {
    0x00, 0xff, 0xf0, 0x0f, 0xcc, 0x33, 0x3c, 0xc3,
    0xaa, 0x55, 0x5a, 0xa5, 0x66, 0x99, 0x96, 0x69
};

__host__ __device__ __forceinline__ uint32_t p10_u32(uint32_t x) {
#ifdef __CUDA_ARCH__
    return (uint32_t)(__popc(x & 0x3FFu) & 1);
#else
    return (uint32_t)(__builtin_popcount(x & 0x3FFu) & 1);
#endif
}

template <int NL>
__host__ __device__ __forceinline__ uint32_t galois_lb_u32(uint32_t st) {
    if constexpr (NL == 16) {
        return (st & 1u) | (p10_u32(st & 0x2CEu) << 1) | (p10_u32(st & 0x050u) << 2) | (p10_u32(st & 0x3B6u) << 3);
    }
    if constexpr (NL == 8) {
        return (st & 1u) | (p10_u32(st & 0x2CEu) << 1) | (p10_u32(st & 0x39Cu) << 2);
    }
    return (st & 1u) | (p10_u32(st & 0x2CEu) << 1);
}

// =================================================================================================
// Coalesced 512-Byte Plane Loader for `lane = 2 * g + hc` (`1` `LDG.E.128` + `2` `SHFL`s -> 0 smem!)
// =================================================================================================
__device__ __forceinline__ void load_plane512_eo(
    const uint8_t* __restrict__ ptr, int lane, int hc, uint2& ev, uint2& od)
{
    uint4 v = __ldg(reinterpret_cast<const uint4*>(ptr) + lane);
    uint32_t s0 = hc ? v.x : v.z;
    uint32_t s1 = hc ? v.y : v.w;
    uint32_t r0 = __shfl_xor_sync(0xffffffffu, s0, 1);
    uint32_t r1 = __shfl_xor_sync(0xffffffffu, s1, 1);
    ev = hc ? make_uint2(r0, r1) : make_uint2(v.x, v.y);
    od = hc ? make_uint2(v.z, v.w) : make_uint2(r0, r1);
}

// =================================================================================================
// 1. Zero-Galois-ALU Joint-256 `16x256` FP32 GEMV (`B = 1`, `lane = 2 * g + hc` = 8 cols x 16 rows)
//    - Pre-expands `(u_full, st_8bit) -> fp16` (`1..16 KB` lossless for `[-127, 127]`):
//      0 runtime Galois shifts/XORs, 0 `s_pk` staging, `x` loaded ONCE across all 16 rows!
// =================================================================================================
template <int FMT, int SPLIT_K>
__global__ void __launch_bounds__(256) vtq_fp32_gemv(
    const uint8_t* __restrict__ ST_raw,
    const __half* __restrict__ d_row,
    float unit,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int Kd,
    const float* __restrict__ d_lut_f32,
    const float* __restrict__ d_gains)
{
    constexpr size_t STRIDE = st_bytes<FMT>();
    constexpr int NL      = fmt_nl<FMT>();
    constexpr int LBITS   = (NL == 4) ? 2 : ((NL == 8) ? 3 : 4);
    constexpr int H       = fmt_h<FMT>();
    constexpr int N_ENT   = signed_lut_entries<FMT>();
    constexpr int UFULL   = N_ENT / NL;
    constexpr int N_JOINT = 1024 * UFULL; // 2 KB .. 32 KB lossless int8 Joint-1024 LUT!

    __shared__ int8_t s_joint[N_JOINT];
    __shared__ float  s_gain[8];
    __shared__ float  s_red[8][16];

    for (int idx = threadIdx.x; idx < N_JOINT; idx += blockDim.x) {
        uint32_t st_i = idx & 1023u;
        uint32_t uf   = idx >> 10;
        uint32_t lb   = galois_lb_u32<NL>(st_i);
        s_joint[idx]  = (int8_t)(int)d_lut_f32[(uf << LBITS) | lb];
    }
    if (threadIdx.x < 8) s_gain[threadIdx.x] = d_gains[threadIdx.x];
    __syncthreads();

    const int wid_cta   = threadIdx.x >> 5;
    const int lane      = threadIdx.x & 31;
    const int g         = lane >> 1;   // 0..15 subblock
    const int hc        = lane & 1;    // 0 (cols 0..7) or 1 (cols 8..15)
    const int nsb      = Kd >> 8;
    const int total_bg = M >> 4;
    const float4* __restrict__ x4_ptr = reinterpret_cast<const float4*>(x);

    const int step_work = (SPLIT_K == 8) ? gridDim.x : (gridDim.x * (blockDim.x >> 5));
    const int init_work = (SPLIT_K == 8) ? blockIdx.x : (blockIdx.x * (blockDim.x >> 5) + wid_cta);
    const int sb_beg    = (SPLIT_K == 8) ? wid_cta : 0;
    const int sb_inc    = (SPLIT_K == 8) ? 8 : 1;

    for (int bg = init_work; bg < total_bg; bg += step_work) {
        float acc[16];
        #pragma unroll
        for (int r = 0; r < 16; r++) acc[r] = 0.f;

        for (int sb = sb_beg; sb < nsb; sb += sb_inc) {
            const uint8_t* base = ST_raw + ((size_t)bg * nsb + sb) * STRIDE;

            // 1. Load 256 activations ONCE across the warp (`8` floats per lane reused across 16 rows!)
            const float4 xa = __ldg(x4_ptr + sb * 64 + 2 * lane + 0);
            const float4 xb = __ldg(x4_ptr + sb * 64 + 2 * lane + 1);
            const float xs[8] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w};

            // 2. Load 512 B `c[16][16][2]` in ONE `LDG.E.128` — intra-subblock `16x16` trellis needs ZERO cross-g shuffles!
            uint4 cv = __ldg(reinterpret_cast<const uint4*>(base) + lane);
            uint32_t ox = __shfl_xor_sync(0xffffffffu, cv.x, 1);
            uint32_t oy = __shfl_xor_sync(0xffffffffu, cv.y, 1);
            uint32_t oz = __shfl_xor_sync(0xffffffffu, cv.z, 1);
            uint32_t ow = __shfl_xor_sync(0xffffffffu, cv.w, 1);
            uint4 lo4 = hc ? make_uint4(ox, oy, oz, ow) : cv;
            uint4 hi4 = hc ? cv : make_uint4(ox, oy, oz, ow);
            const uint32_t cw_pair[8] = {lo4.x, lo4.y, lo4.z, lo4.w, hi4.x, hi4.y, hi4.z, hi4.w};

            // 3. Load payload planes (coalesced `uint4`/`uint2` loads across the entire super-tile!)
            uint2 sv_ev = {0, 0}, sv_od = {0, 0};
            uint2 ub_ev = {0, 0}, ub_od = {0, 0};
            uint2 u2_ev0 = {0, 0}, u2_od0 = {0, 0}, u2_ev1 = {0, 0}, u2_od1 = {0, 0};
            uint2 h_lo = {0, 0}, h_hi = {0, 0};

            if constexpr (FMT >= 20) {
                load_plane512_eo(base + 512, lane, hc, sv_ev, sv_od);
            }
            if constexpr (FMT == 30 || FMT == 31 || FMT == 35) {
                load_plane512_eo(base + 1024, lane, hc, ub_ev, ub_od);
            } else if constexpr (FMT == 50 || FMT == 51) {
                load_plane512_eo(base + 2048, lane, hc, ub_ev, ub_od);
            }
            if constexpr (FMT == 40 || FMT == 50 || FMT == 51) {
                const uint8_t* p_u2 = base + 1024 + 64 * g + 8 * hc;
                u2_ev0 = __ldg(reinterpret_cast<const uint2*>(p_u2 + 0));
                u2_od0 = __ldg(reinterpret_cast<const uint2*>(p_u2 + 16));
                u2_ev1 = __ldg(reinterpret_cast<const uint2*>(p_u2 + 32));
                u2_od1 = __ldg(reinterpret_cast<const uint2*>(p_u2 + 48));
            }
            if constexpr (FMT == 15 || FMT == 25 || FMT == 35) {
                constexpr size_t HAM_OFF = (FMT == 15) ? 512 : ((FMT == 25) ? 1024 : 1536);
                uint2 hv = __ldg(reinterpret_cast<const uint2*>(base + HAM_OFF) + lane);
                uint32_t hx = __shfl_xor_sync(0xffffffffu, hv.x, 1);
                uint32_t hy = __shfl_xor_sync(0xffffffffu, hv.y, 1);
                h_lo = hc ? make_uint2(hx, hy) : hv;
                h_hi = hc ? hv : make_uint2(hx, hy);
            }

            const uint16_t* hdr_ptr = reinterpret_cast<const uint16_t*>(base + STRIDE - 16 - 32 * H);
            const uint32_t h0 = __ldg(hdr_ptr + g * H + 0);
            const uint32_t h1 = (H >= 2) ? __ldg(hdr_ptr + g * H + 1) : 0u;
            const uint32_t h2 = (H >= 3) ? __ldg(hdr_ptr + g * H + 2) : 0u;
            const uint4 usb4  = __ldg(reinterpret_cast<const uint4*>(base + STRIDE - 16));
            const uint32_t usb_w[4] = {usb4.x, usb4.y, usb4.z, usb4.w};

            #pragma unroll
            for (int r = 0; r < 16; r++) {
                const int p     = r >> 1;
                const int odd   = r & 1;
                const int r_prv = (r + 15) & 15;
                const uint32_t cw_cur = (cw_pair[r >> 1]     >> (16 * (r & 1)))     & 0xFFFFu;
                const uint32_t cw_prv = (cw_pair[r_prv >> 1] >> (16 * (r_prv & 1))) & 0x1FFu;
                const uint32_t w25    = (cw_prv << 16) | cw_cur;
                const uint32_t w17    = hc ? w25 : (w25 >> 8);

                uint32_t h_cw = 0u;
                if constexpr (FMT == 15 || FMT == 25 || FMT == 35) {
                    uint32_t hw = (r < 4) ? h_lo.x : ((r < 8) ? h_lo.y : ((r < 12) ? h_hi.x : h_hi.y));
                    uint32_t hb = (hw >> (8 * (r & 3))) & 0xFFu;
                    h_cw = C_HAM_CW8[hc ? (hb >> 4) : (hb & 15)];
                }

                uint2 sv_p = odd ? sv_od : sv_ev;
                uint2 ub_p = odd ? ub_od : ub_ev;
                uint2 u2_p = (p < 4) ? (odd ? u2_od0 : u2_ev0) : (odd ? u2_od1 : u2_ev1);
                uint2 u4_p = {0, 0};
                if constexpr (FMT == 60) {
                    u4_p = __ldg(reinterpret_cast<const uint2*>(base + 1024 + 128 * g + 32 * (p >> 1) + 16 * odd + 8 * hc));
                }

                float part = 0.f;
                #pragma unroll
                for (int j8 = 0; j8 < 8; j8++) {
                    const uint32_t st = (w17 >> (7 - j8)) & 0x3FFu;
                    const int sh_j = 8 * (j8 & 3);
                    uint32_t uf = 0u;

                    if constexpr (FMT == 15) {
                        uf = (h_cw >> (7 - j8)) & 1u;
                    } else {
                        const uint32_t sv_b = ((j8 < 4 ? sv_p.x : sv_p.y) >> sh_j) & 0xFFu;
                        const uint32_t neg  = (sv_b >> p) & 1u;
                        if constexpr (FMT == 20) {
                            uf = neg;
                        } else if constexpr (FMT == 25) {
                            uf = (neg << 1) | ((h_cw >> (7 - j8)) & 1u);
                        } else if constexpr (FMT == 30 || FMT == 31) {
                            const uint32_t ub_b = ((j8 < 4 ? ub_p.x : ub_p.y) >> sh_j) & 0xFFu;
                            uf = (neg << 1) | ((ub_b >> p) & 1u);
                        } else if constexpr (FMT == 35) {
                            const uint32_t ub_b = ((j8 < 4 ? ub_p.x : ub_p.y) >> sh_j) & 0xFFu;
                            uf = (neg << 2) | (((ub_b >> p) & 1u) << 1) | ((h_cw >> (7 - j8)) & 1u);
                        } else if constexpr (FMT == 40) {
                            const uint32_t u2_b = ((j8 < 4 ? u2_p.x : u2_p.y) >> sh_j) & 0xFFu;
                            uf = (neg << 2) | ((u2_b >> (2 * (p & 3))) & 3u);
                        } else if constexpr (FMT == 50 || FMT == 51) {
                            const uint32_t ub_b = ((j8 < 4 ? ub_p.x : ub_p.y) >> sh_j) & 0xFFu;
                            const uint32_t u2_b = ((j8 < 4 ? u2_p.x : u2_p.y) >> sh_j) & 0xFFu;
                            uf = (neg << 3) | (((ub_b >> p) & 1u) << 2) | ((u2_b >> (2 * (p & 3))) & 3u);
                        } else if constexpr (FMT == 60) {
                            const uint32_t u4_b = ((j8 < 4 ? u4_p.x : u4_p.y) >> sh_j) & 0xFFu;
                            uf = (neg << 4) | ((u4_b >> (4 * (p & 1))) & 15u);
                        }
                    }
                    part = fmaf((float)s_joint[(uf << 10) | st], xs[j8], part);
                }

                uint32_t gi = (h0 >> r) & 1u;
                if constexpr (H >= 2) gi |= ((h1 >> r) & 1u) << 1;
                if constexpr (H >= 3) gi |= ((h2 >> r) & 1u) << 2;
                float u_f = (float)((usb_w[r >> 2] >> (8 * (r & 3))) & 0xFFu);
                acc[r] = fmaf(part, s_gain[gi] * u_f, acc[r]);
            }
        }

        // Reduce across all 32 lanes (`256` columns) for each of the 16 rows
        #pragma unroll
        for (int r = 0; r < 16; r++) {
            float v = acc[r];
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
            acc[r] = v;
        }

        if constexpr (SPLIT_K == 8) {
            if (lane == 0) {
                #pragma unroll
                for (int r = 0; r < 16; r++) s_red[wid_cta][r] = acc[r];
            }
            __syncthreads();
            if (wid_cta == 0 && lane < 16) {
                float sum = 0.f;
                #pragma unroll
                for (int w = 0; w < 8; w++) sum += s_red[w][lane];
                float dr = __half2float(__ldg(d_row + bg * 16 + lane)) * unit * (1.0f / 64.0f);
                y[bg * 16 + lane] = sum * dr;
            }
            __syncthreads();
        } else {
            if (lane == 0) {
                #pragma unroll
                for (int r = 0; r < 16; r++) {
                    float dr = __half2float(__ldg(d_row + bg * 16 + r)) * unit * (1.0f / 64.0f);
                    y[bg * 16 + r] = acc[r] * dr;
                }
            }
        }
    }
}

// =================================================================================================
// Shared Super-Tile Metadata Unpacker for `__dp4a` Multi-Token GEMM
// =================================================================================================
template <int FMT>
__device__ __forceinline__ void stage_supertile_meta(
    const uint8_t* __restrict__ base,
    int lane,
    uint32_t s_pk_w[16][16],
    uint32_t* __restrict__ s_h16_flat,
    uint16_t s_tail_w[64],
    const uint16_t* __restrict__ s_ham16)
{
    constexpr size_t STRIDE  = st_bytes<FMT>();
    constexpr int H          = fmt_h<FMT>();
    constexpr int TAIL_WORDS = (32 * H + 16) >> 2;

    uint4 cv = __ldg(reinterpret_cast<const uint4*>(base) + lane);
    if (lane < TAIL_WORDS) {
        reinterpret_cast<uint32_t*>(s_tail_w)[lane] =
            __ldg(reinterpret_cast<const uint32_t*>(base + STRIDE - 16 - 32 * H) + lane);
    }

    const uint32_t other_w  = __shfl_xor_sync(0xffffffffu, cv.w, 1);
    const uint32_t prv_wrap = (other_w >> 16) & 0x1FFu;
    __syncwarp();

    const int cg = lane >> 1;
    const int r_base = (lane & 1) << 3;
    const uint32_t h0 = s_tail_w[cg * H + 0];
    const uint32_t h1 = (H >= 2) ? s_tail_w[cg * H + 1] : 0u;
    const uint32_t h2 = (H >= 3) ? s_tail_w[cg * H + 2] : 0u;

    const uint32_t cw_v[4] = {cv.x, cv.y, cv.z, cv.w};
    uint32_t prv9 = prv_wrap;
    #pragma unroll
    for (int k = 0; k < 8; k++) {
        const int r = r_base + k;
        const uint32_t cw16 = (cw_v[k >> 1] >> (16 * (k & 1))) & 0xFFFFu;
        const uint32_t w25  = (prv9 << 16) | cw16;
        uint32_t gi = (h0 >> r) & 1u;
        if constexpr (H >= 2) gi |= ((h1 >> r) & 1u) << 1;
        if constexpr (H >= 3) gi |= ((h2 >> r) & 1u) << 2;
        s_pk_w[cg][r] = w25 | (gi << 25);
        prv9 = cw16 & 0x1FFu;
    }

    if constexpr (FMT == 15 || FMT == 25 || FMT == 35) {
        constexpr size_t HAM_OFF = (FMT == 15) ? 512 : ((FMT == 25) ? 1024 : 1536);
        uint2 hv = __ldg(reinterpret_cast<const uint2*>(base + HAM_OFF) + lane);
        #pragma unroll
        for (int b = 0; b < 4; b++) {
            s_h16_flat[16 * cg + r_base + b]     = s_ham16[(hv.x >> (8 * b)) & 0xFFu];
            s_h16_flat[16 * cg + r_base + 4 + b] = s_ham16[(hv.y >> (8 * b)) & 0xFFu];
        }
    }
    __syncwarp();
}

// =================================================================================================
// 2a. Zero-Galois-ALU Joint-1024 `16x256` INT8 `__dp4a` Single-Token GEMV (`B = 1`, `lane = 2*g + hc`)
// =================================================================================================
template <int FMT, int SPLIT_K>
__global__ void __launch_bounds__(256) vtq_dp4a_gemv_b1(
    const uint8_t* __restrict__ ST_raw,
    const __half* __restrict__ d_row,
    float unit,
    const int32_t* __restrict__ x_q8x4,   // [Kd / 4]
    const float* __restrict__ d_x,        // [Kd / 256]
    float* __restrict__ y,                // [M]
    int M, int Kd,
    const int32_t* __restrict__ d_lut_i32,
    const int32_t* __restrict__ d_gains_i)
{
    constexpr size_t STRIDE = st_bytes<FMT>();
    constexpr int NL      = fmt_nl<FMT>();
    constexpr int LBITS   = (NL == 4) ? 2 : ((NL == 8) ? 3 : 4);
    constexpr int H       = fmt_h<FMT>();
    constexpr int N_ENT   = signed_lut_entries<FMT>();
    constexpr int UFULL   = N_ENT / NL;
    constexpr int N_JOINT = 1024 * UFULL;

    __shared__ uint8_t s_joint[N_JOINT];
    __shared__ int32_t s_gain[8];
    __shared__ float   s_red[8][16];

    for (int idx = threadIdx.x; idx < N_JOINT; idx += blockDim.x) {
        uint32_t st_i = idx & 1023u;
        uint32_t uf   = idx >> 10;
        uint32_t lb   = galois_lb_u32<NL>(st_i);
        s_joint[idx]  = (uint8_t)(d_lut_i32[(uf << LBITS) | lb] & 0xFF);
    }
    if (threadIdx.x < 8) s_gain[threadIdx.x] = d_gains_i[threadIdx.x];
    __syncthreads();

    const int wid_cta  = threadIdx.x >> 5;
    const int lane     = threadIdx.x & 31;
    const int g        = lane >> 1;
    const int hc       = lane & 1;
    const int nsb      = Kd >> 8;
    const int total_bg = M >> 4;
    const int2* __restrict__ x8_ptr = reinterpret_cast<const int2*>(x_q8x4);

    const int step_work = (SPLIT_K == 8) ? gridDim.x : (gridDim.x * (blockDim.x >> 5));
    const int init_work = (SPLIT_K == 8) ? blockIdx.x : (blockIdx.x * (blockDim.x >> 5) + wid_cta);
    const int sb_beg    = (SPLIT_K == 8) ? wid_cta : 0;
    const int sb_inc    = (SPLIT_K == 8) ? 8 : 1;

    for (int bg = init_work; bg < total_bg; bg += step_work) {
        float acc[16];
        #pragma unroll
        for (int r = 0; r < 16; r++) acc[r] = 0.f;

        for (int sb = sb_beg; sb < nsb; sb += sb_inc) {
            const uint8_t* base = ST_raw + ((size_t)bg * nsb + sb) * STRIDE;
            const int2 xv2 = __ldg(x8_ptr + sb * 32 + lane);
            const float dx = __ldg(d_x + sb);

            uint4 cv = __ldg(reinterpret_cast<const uint4*>(base) + lane);
            uint32_t ox = __shfl_xor_sync(0xffffffffu, cv.x, 1);
            uint32_t oy = __shfl_xor_sync(0xffffffffu, cv.y, 1);
            uint32_t oz = __shfl_xor_sync(0xffffffffu, cv.z, 1);
            uint32_t ow = __shfl_xor_sync(0xffffffffu, cv.w, 1);
            uint4 lo4 = hc ? make_uint4(ox, oy, oz, ow) : cv;
            uint4 hi4 = hc ? cv : make_uint4(ox, oy, oz, ow);
            const uint32_t cw_pair[8] = {lo4.x, lo4.y, lo4.z, lo4.w, hi4.x, hi4.y, hi4.z, hi4.w};

            uint2 sv_ev = {0, 0}, sv_od = {0, 0};
            uint2 ub_ev = {0, 0}, ub_od = {0, 0};
            uint2 u2_ev0 = {0, 0}, u2_od0 = {0, 0}, u2_ev1 = {0, 0}, u2_od1 = {0, 0};
            uint2 h_lo = {0, 0}, h_hi = {0, 0};

            if constexpr (FMT >= 20) load_plane512_eo(base + 512, lane, hc, sv_ev, sv_od);
            if constexpr (FMT == 30 || FMT == 31 || FMT == 35) load_plane512_eo(base + 1024, lane, hc, ub_ev, ub_od);
            else if constexpr (FMT == 50 || FMT == 51)         load_plane512_eo(base + 2048, lane, hc, ub_ev, ub_od);
            if constexpr (FMT == 40 || FMT == 50 || FMT == 51) {
                const uint8_t* p_u2 = base + 1024 + 64 * g + 8 * hc;
                u2_ev0 = __ldg(reinterpret_cast<const uint2*>(p_u2 + 0));
                u2_od0 = __ldg(reinterpret_cast<const uint2*>(p_u2 + 16));
                u2_ev1 = __ldg(reinterpret_cast<const uint2*>(p_u2 + 32));
                u2_od1 = __ldg(reinterpret_cast<const uint2*>(p_u2 + 48));
            }
            if constexpr (FMT == 15 || FMT == 25 || FMT == 35) {
                constexpr size_t HAM_OFF = (FMT == 15) ? 512 : ((FMT == 25) ? 1024 : 1536);
                uint2 hv = __ldg(reinterpret_cast<const uint2*>(base + HAM_OFF) + lane);
                uint32_t hx = __shfl_xor_sync(0xffffffffu, hv.x, 1);
                uint32_t hy = __shfl_xor_sync(0xffffffffu, hv.y, 1);
                h_lo = hc ? make_uint2(hx, hy) : hv;
                h_hi = hc ? hv : make_uint2(hx, hy);
            }

            const uint16_t* hdr_ptr = reinterpret_cast<const uint16_t*>(base + STRIDE - 16 - 32 * H);
            const uint32_t h0 = __ldg(hdr_ptr + g * H + 0);
            const uint32_t h1 = (H >= 2) ? __ldg(hdr_ptr + g * H + 1) : 0u;
            const uint32_t h2 = (H >= 3) ? __ldg(hdr_ptr + g * H + 2) : 0u;
            const uint4 usb4  = __ldg(reinterpret_cast<const uint4*>(base + STRIDE - 16));
            const uint32_t usb_w[4] = {usb4.x, usb4.y, usb4.z, usb4.w};

            #pragma unroll
            for (int r = 0; r < 16; r++) {
                const int p     = r >> 1;
                const int odd   = r & 1;
                const int r_prv = (r + 15) & 15;
                const uint32_t cw_cur = (cw_pair[r >> 1]     >> (16 * (r & 1)))     & 0xFFFFu;
                const uint32_t cw_prv = (cw_pair[r_prv >> 1] >> (16 * (r_prv & 1))) & 0x1FFu;
                const uint32_t w25    = (cw_prv << 16) | cw_cur;
                const uint32_t w17    = hc ? w25 : (w25 >> 8);

                uint32_t h_cw = 0u;
                if constexpr (FMT == 15 || FMT == 25 || FMT == 35) {
                    uint32_t hw = (r < 4) ? h_lo.x : ((r < 8) ? h_lo.y : ((r < 12) ? h_hi.x : h_hi.y));
                    uint32_t hb = (hw >> (8 * (r & 3))) & 0xFFu;
                    h_cw = C_HAM_CW8[hc ? (hb >> 4) : (hb & 15)];
                }

                uint2 sv_p = odd ? sv_od : sv_ev;
                uint2 ub_p = odd ? ub_od : ub_ev;
                uint2 u2_p = (p < 4) ? (odd ? u2_od0 : u2_ev0) : (odd ? u2_od1 : u2_ev1);
                uint2 u4_p = {0, 0};
                if constexpr (FMT == 60) {
                    u4_p = __ldg(reinterpret_cast<const uint2*>(base + 1024 + 128 * g + 32 * (p >> 1) + 16 * odd + 8 * hc));
                }

                uint32_t wp_a = 0u, wp_b = 0u;
                #pragma unroll
                for (int j8 = 0; j8 < 8; j8++) {
                    const uint32_t st = (w17 >> (7 - j8)) & 0x3FFu;
                    const int sh_j = 8 * (j8 & 3);
                    uint32_t uf = 0u;
                    if constexpr (FMT == 15) {
                        uf = (h_cw >> (7 - j8)) & 1u;
                    } else {
                        const uint32_t sv_b = ((j8 < 4 ? sv_p.x : sv_p.y) >> sh_j) & 0xFFu;
                        const uint32_t neg  = (sv_b >> p) & 1u;
                        if constexpr (FMT == 20) uf = neg;
                        else if constexpr (FMT == 25) uf = (neg << 1) | ((h_cw >> (7 - j8)) & 1u);
                        else if constexpr (FMT == 30 || FMT == 31) {
                            const uint32_t ub_b = ((j8 < 4 ? ub_p.x : ub_p.y) >> sh_j) & 1u * 0xFFu;
                            uf = (neg << 1) | ((ub_b >> p) & 1u);
                        } else if constexpr (FMT == 35) {
                            const uint32_t ub_b = ((j8 < 4 ? ub_p.x : ub_p.y) >> sh_j) & 0xFFu;
                            uf = (neg << 2) | (((ub_b >> p) & 1u) << 1) | ((h_cw >> (7 - j8)) & 1u);
                        } else if constexpr (FMT == 40) {
                            const uint32_t u2_b = ((j8 < 4 ? u2_p.x : u2_p.y) >> sh_j) & 0xFFu;
                            uf = (neg << 2) | ((u2_b >> (2 * (p & 3))) & 3u);
                        } else if constexpr (FMT == 50 || FMT == 51) {
                            const uint32_t ub_b = ((j8 < 4 ? ub_p.x : ub_p.y) >> sh_j) & 0xFFu;
                            const uint32_t u2_b = ((j8 < 4 ? u2_p.x : u2_p.y) >> sh_j) & 0xFFu;
                            uf = (neg << 3) | (((ub_b >> p) & 1u) << 2) | ((u2_b >> (2 * (p & 3))) & 3u);
                        } else if constexpr (FMT == 60) {
                            const uint32_t u4_b = ((j8 < 4 ? u4_p.x : u4_p.y) >> sh_j) & 0xFFu;
                            uf = (neg << 4) | ((u4_b >> (4 * (p & 1))) & 15u);
                        }
                    }
                    uint32_t qb = s_joint[(uf << 10) | st];
                    if (j8 < 4) wp_a |= qb << sh_j;
                    else        wp_b |= qb << sh_j;
                }

                int32_t dot8 = __dp4a((int32_t)wp_b, xv2.y, __dp4a((int32_t)wp_a, xv2.x, 0));
                uint32_t gi = (h0 >> r) & 1u;
                if constexpr (H >= 2) gi |= ((h1 >> r) & 1u) << 1;
                if constexpr (H >= 3) gi |= ((h2 >> r) & 1u) << 2;
                float u_f = (float)((usb_w[r >> 2] >> (8 * (r & 3))) & 0xFFu);
                acc[r] = fmaf((float)(dot8 * s_gain[gi]), u_f * dx, acc[r]);
            }
        }

        #pragma unroll
        for (int r = 0; r < 16; r++) {
            float v = acc[r];
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
            acc[r] = v;
        }

        if constexpr (SPLIT_K == 8) {
            if (lane == 0) {
                #pragma unroll
                for (int r = 0; r < 16; r++) s_red[wid_cta][r] = acc[r];
            }
            __syncthreads();
            if (wid_cta == 0 && lane < 16) {
                float sum = 0.f;
                #pragma unroll
                for (int w = 0; w < 8; w++) sum += s_red[w][lane];
                float dr = __half2float(__ldg(d_row + bg * 16 + lane)) * unit * (1.0f / 64.0f);
                y[bg * 16 + lane] = sum * dr;
            }
            __syncthreads();
        } else {
            if (lane == 0) {
                #pragma unroll
                for (int r = 0; r < 16; r++) {
                    float dr = __half2float(__ldg(d_row + bg * 16 + r)) * unit * (1.0f / 64.0f);
                    y[bg * 16 + r] = acc[r] * dr;
                }
            }
        }
    }
}

// =================================================================================================
// 2b. Zero-Galois-ALU Joint-256 `16x256` INT8 `__dp4a` Multi-Token GEMM (`B_TILE = 8`, `grid.y = B_TOK / 8`)
//     Zero register spills (`<= 96` regs), `0.5..8 KB` `s_joint`!
// =================================================================================================
template <int FMT, int B_TILE, int SPLIT_K>
__global__ void vtq_dp4a_gemm(
    const uint8_t* __restrict__ ST_raw,
    const __half* __restrict__ d_row,
    float unit,
    const int32_t* __restrict__ x_q8x4,   // [B_TOK, Kd / 4]
    const float* __restrict__ d_x,        // [B_TOK, Kd / 256]
    float* __restrict__ y,                // [B_TOK, M]
    int M, int Kd,
    const int32_t* __restrict__ d_lut_i32,
    const int32_t* __restrict__ d_gains_i)
{
    constexpr size_t STRIDE = st_bytes<FMT>();
    constexpr int NL      = fmt_nl<FMT>();
    constexpr int LBITS   = (NL == 4) ? 2 : ((NL == 8) ? 3 : 4);
    constexpr int H       = fmt_h<FMT>();
    constexpr int N_ENT   = signed_lut_entries<FMT>();
    constexpr int UFULL   = N_ENT / NL;
    constexpr int N_JOINT = 1024 * UFULL;
    constexpr bool HAS_HAM = (FMT == 15 || FMT == 25 || FMT == 35);

    __shared__ uint8_t  s_joint[N_JOINT];
    __shared__ int32_t  s_gain[8];
    __shared__ uint16_t s_ham16[HAS_HAM ? 256 : 1];
    __shared__ uint32_t s_pk[8][16][16];
    __shared__ uint32_t s_h16[HAS_HAM ? 8 : 1][HAS_HAM ? 16 : 1][HAS_HAM ? 16 : 1];
    __shared__ uint16_t s_tail[8][64];
    __shared__ float    s_red[8][16];

    for (int idx = threadIdx.x; idx < N_JOINT; idx += blockDim.x) {
        uint32_t st_i = idx & 1023u;
        uint32_t uf   = idx >> 10;
        uint32_t lb   = galois_lb_u32<NL>(st_i);
        s_joint[idx]  = (uint8_t)(d_lut_i32[(uf << LBITS) | lb] & 0xFF);
    }
    if (threadIdx.x < 8) s_gain[threadIdx.x] = d_gains_i[threadIdx.x];
    if constexpr (HAS_HAM) {
        for (int hb = threadIdx.x; hb < 256; hb += blockDim.x) {
            s_ham16[hb] = (uint16_t)((uint16_t(C_HAM_CW8[hb & 15]) << 8) | uint16_t(C_HAM_CW8[hb >> 4]));
        }
    }
    __syncthreads();

    const int b_off   = blockIdx.y * B_TILE;
    const int wid_cta = threadIdx.x >> 5;
    const int lane    = threadIdx.x & 31;
    const int q       = lane >> 3;
    const int r_lo    = lane & 7;
    const int odd     = r_lo & 1;
    const int p0      = r_lo >> 1;
    const int w_src   = 4 * odd + q;
    const int nsb     = Kd >> 8;
    const int k4      = Kd >> 2;
    const int total_bg = M >> 4;

    const int step_work = (SPLIT_K == 8) ? gridDim.x : (gridDim.x * (blockDim.x >> 5));
    const int init_work = (SPLIT_K == 8) ? blockIdx.x : (blockIdx.x * (blockDim.x >> 5) + wid_cta);
    const int sb_beg    = (SPLIT_K == 8) ? wid_cta : 0;
    const int sb_inc    = (SPLIT_K == 8) ? 8 : 1;

    for (int bg = init_work; bg < total_bg; bg += step_work) {
        float acc0[B_TILE], acc1[B_TILE];
        #pragma unroll
        for (int bt = 0; bt < B_TILE; bt++) { acc0[bt] = 0.f; acc1[bt] = 0.f; }

        for (int sb = sb_beg; sb < nsb; sb += sb_inc) {
            const uint8_t* base = ST_raw + ((size_t)bg * nsb + sb) * STRIDE;
            stage_supertile_meta<FMT>(base, lane, s_pk[wid_cta], &s_h16[HAS_HAM ? wid_cta : 0][0][0], s_tail[wid_cta], s_ham16);

            int32_t isum0[B_TILE], isum1[B_TILE];
            #pragma unroll
            for (int bt = 0; bt < B_TILE; bt++) { isum0[bt] = 0; isum1[bt] = 0; }

            #pragma unroll 1
            for (int gq = 0; gq < 4; gq++) {
                uint32_t sw_l = 0u, uw_l = 0u, u2_l0 = 0u, u2_l1 = 0u;
                if constexpr (FMT >= 20) {
                    sw_l = __ldg(reinterpret_cast<const uint32_t*>(base + 512 + 128 * gq) + lane);
                }
                if constexpr (FMT == 30 || FMT == 31 || FMT == 35) {
                    uw_l = __ldg(reinterpret_cast<const uint32_t*>(base + 1024 + 128 * gq) + lane);
                } else if constexpr (FMT == 50 || FMT == 51) {
                    uw_l = __ldg(reinterpret_cast<const uint32_t*>(base + 2048 + 128 * gq) + lane);
                }
                if constexpr (FMT == 40 || FMT == 50 || FMT == 51) {
                    u2_l0 = __ldg(reinterpret_cast<const uint32_t*>(base + 1024 + 256 * gq) + lane);
                    u2_l1 = __ldg(reinterpret_cast<const uint32_t*>(base + 1024 + 256 * gq + 128) + lane);
                }

                #pragma unroll
                for (int s = 0; s < 4; s++) {
                    const int g = 4 * gq + s;
                    const uint32_t pk0 = s_pk[wid_cta][g][r_lo];
                    const uint32_t pk1 = s_pk[wid_cta][g][8 + r_lo];
                    const int32_t g0 = s_gain[pk0 >> 25];
                    const int32_t g1 = s_gain[pk1 >> 25];

                    uint32_t sv_w = 0u, ub_w = 0u, u2a_w = 0u, u2b_w = 0u, u40_w = 0u, u41_w = 0u, h16_0 = 0u, h16_1 = 0u;
                    if constexpr (HAS_HAM) {
                        h16_0 = s_h16[HAS_HAM ? wid_cta : 0][HAS_HAM ? g : 0][HAS_HAM ? r_lo : 0];
                        h16_1 = s_h16[HAS_HAM ? wid_cta : 0][HAS_HAM ? g : 0][HAS_HAM ? (8 + r_lo) : 0];
                    }
                    if constexpr (FMT >= 20) {
                        sv_w = __shfl_sync(0xffffffffu, sw_l, 8 * s + w_src);
                    }
                    if constexpr (FMT == 30 || FMT == 31 || FMT == 35 || FMT == 50 || FMT == 51) {
                        ub_w = __shfl_sync(0xffffffffu, uw_l, 8 * s + w_src);
                    }
                    if constexpr (FMT == 40 || FMT == 50 || FMT == 51) {
                        const uint32_t src_u2 = (s < 2) ? u2_l0 : u2_l1;
                        const int bs = (s & 1) * 16;
                        u2a_w = __shfl_sync(0xffffffffu, src_u2, bs + w_src);
                        u2b_w = __shfl_sync(0xffffffffu, src_u2, bs + 8 + w_src);
                    }
                    if constexpr (FMT == 60) {
                        uint32_t u4_l = __ldg(reinterpret_cast<const uint32_t*>(base + 1024 + 128 * g) + lane);
                        u40_w = __shfl_sync(0xffffffffu, u4_l, 8 * (p0 >> 1) + w_src);
                        u41_w = __shfl_sync(0xffffffffu, u4_l, 16 + 8 * (p0 >> 1) + w_src);
                    }

                    uint32_t wp0 = 0u, wp1 = 0u;
                    #pragma unroll
                    for (int i = 0; i < 4; i++) {
                        const int sh_c = 15 - (4 * q + i);
                        const int sh_b = 8 * i;
                        const uint32_t st0 = (pk0 >> sh_c) & 0x3FFu;
                        const uint32_t st1 = (pk1 >> sh_c) & 0x3FFu;
                        uint32_t uf0 = 0u, uf1 = 0u;

                        if constexpr (FMT == 15) {
                            uf0 = (h16_0 >> sh_c) & 1u;
                            uf1 = (h16_1 >> sh_c) & 1u;
                        } else {
                            uint32_t sb_i = (sv_w >> sh_b) & 0xFFu;
                            uint32_t n0 = (sb_i >> p0) & 1u;
                            uint32_t n1 = (sb_i >> (p0 + 4)) & 1u;
                            if constexpr (FMT == 20) {
                                uf0 = n0; uf1 = n1;
                            } else if constexpr (FMT == 25) {
                                uf0 = (n0 << 1) | ((h16_0 >> sh_c) & 1u);
                                uf1 = (n1 << 1) | ((h16_1 >> sh_c) & 1u);
                            } else if constexpr (FMT == 30 || FMT == 31) {
                                uint32_t ub_i = (ub_w >> sh_b) & 0xFFu;
                                uf0 = (n0 << 1) | ((ub_i >> p0) & 1u);
                                uf1 = (n1 << 1) | ((ub_i >> (p0 + 4)) & 1u);
                            } else if constexpr (FMT == 35) {
                                uint32_t ub_i = (ub_w >> sh_b) & 0xFFu;
                                uf0 = (n0 << 2) | (((ub_i >> p0) & 1u) << 1) | ((h16_0 >> sh_c) & 1u);
                                uf1 = (n1 << 2) | (((ub_i >> (p0 + 4)) & 1u) << 1) | ((h16_1 >> sh_c) & 1u);
                            } else if constexpr (FMT == 40) {
                                uf0 = (n0 << 2) | (((u2a_w >> sh_b) >> (2 * p0)) & 3u);
                                uf1 = (n1 << 2) | (((u2b_w >> sh_b) >> (2 * p0)) & 3u);
                            } else if constexpr (FMT == 50 || FMT == 51) {
                                uint32_t ub_i = (ub_w >> sh_b) & 0xFFu;
                                uf0 = (n0 << 3) | (((ub_i >> p0) & 1u) << 2) | (((u2a_w >> sh_b) >> (2 * p0)) & 3u);
                                uf1 = (n1 << 3) | (((ub_i >> (p0 + 4)) & 1u) << 2) | (((u2b_w >> sh_b) >> (2 * p0)) & 3u);
                            } else if constexpr (FMT == 60) {
                                const int nsh = 4 * (p0 & 1);
                                uf0 = (n0 << 4) | (((u40_w >> sh_b) >> nsh) & 15u);
                                uf1 = (n1 << 4) | (((u41_w >> sh_b) >> nsh) & 15u);
                            }
                        }
                        wp0 |= uint32_t(s_joint[(uf0 << 10) | st0]) << sh_b;
                        wp1 |= uint32_t(s_joint[(uf1 << 10) | st1]) << sh_b;
                    }

                    const int x_col4 = sb * 64 + 4 * g + q;
                    #pragma unroll
                    for (int bt = 0; bt < B_TILE; bt++) {
                        int32_t xv = __ldg(x_q8x4 + (size_t)(b_off + bt) * k4 + x_col4);
                        isum0[bt] += __dp4a((int32_t)wp0, xv, 0) * g0;
                        isum1[bt] += __dp4a((int32_t)wp1, xv, 0) * g1;
                    }
                }
            }

            const uint8_t* usb_s = reinterpret_cast<const uint8_t*>(&s_tail[wid_cta][16 * H]);
            const float u0_f = (float)usb_s[r_lo];
            const float u1_f = (float)usb_s[8 + r_lo];
            #pragma unroll
            for (int bt = 0; bt < B_TILE; bt++) {
                float dx = __ldg(d_x + (size_t)(b_off + bt) * nsb + sb);
                acc0[bt] = fmaf((float)isum0[bt], u0_f * dx, acc0[bt]);
                acc1[bt] = fmaf((float)isum1[bt], u1_f * dx, acc1[bt]);
            }
            __syncwarp();
        }

        #pragma unroll
        for (int bt = 0; bt < B_TILE; bt++) {
            acc0[bt] += __shfl_xor_sync(0xffffffffu, acc0[bt], 8);
            acc0[bt] += __shfl_xor_sync(0xffffffffu, acc0[bt], 16);
            acc1[bt] += __shfl_xor_sync(0xffffffffu, acc1[bt], 8);
            acc1[bt] += __shfl_xor_sync(0xffffffffu, acc1[bt], 16);
        }

        float dr0 = __half2float(__ldg(d_row + bg * 16 + r_lo))     * unit * (1.0f / 64.0f);
        float dr1 = __half2float(__ldg(d_row + bg * 16 + 8 + r_lo)) * unit * (1.0f / 64.0f);

        if constexpr (SPLIT_K == 8) {
            for (int bt = 0; bt < B_TILE; bt++) {
                if (q == 0) {
                    s_red[wid_cta][r_lo]     = acc0[bt];
                    s_red[wid_cta][8 + r_lo] = acc1[bt];
                }
                __syncthreads();
                if (wid_cta == 0 && lane < 16) {
                    float sum = 0.f;
                    #pragma unroll
                    for (int w = 0; w < 8; w++) sum += s_red[w][lane];
                    float dr = __half2float(__ldg(d_row + bg * 16 + lane)) * unit * (1.0f / 64.0f);
                    y[(size_t)(b_off + bt) * M + bg * 16 + lane] = sum * dr;
                }
                __syncthreads();
            }
        } else {
            if (q == 0) {
                #pragma unroll
                for (int bt = 0; bt < B_TILE; bt++) {
                    y[(size_t)(b_off + bt) * M + bg * 16 + r_lo]     = acc0[bt] * dr0;
                    y[(size_t)(b_off + bt) * M + bg * 16 + 8 + r_lo] = acc1[bt] * dr1;
                }
            }
        }
    }
}

// =================================================================================================
// 3. Fused Block-256 FWHT + INT8 Dynamic Activation Quantization (`< 5 us` for `KD = 8192`)
// =================================================================================================
__global__ void fused_fwht256_quant_q8(
    const float* __restrict__ x_in,
    const int8_t* __restrict__ suh_sgn,
    int32_t* __restrict__ x_q8x4,
    float* __restrict__ d_x,
    int Kd)
{
    __shared__ float s_x[256];
    __shared__ float s_max[8];
    const int sb  = blockIdx.x;
    const int tok = blockIdx.y;
    const int tid = threadIdx.x;

    float v = x_in[(size_t)tok * Kd + sb * 256 + tid] * (suh_sgn[sb * 256 + tid] >= 0 ? 1.0f : -1.0f);
    s_x[tid] = v;
    __syncthreads();

    for (int st = 1; st < 256; st <<= 1) {
        int lo = tid & (st - 1);
        int i0 = ((tid - lo) << 1) + lo;
        int i1 = i0 + st;
        float a = 0.f, b = 0.f;
        if (tid < 128) { a = s_x[i0]; b = s_x[i1]; }
        __syncthreads();
        if (tid < 128) { s_x[i0] = a + b; s_x[i1] = a - b; }
        __syncthreads();
    }
    v = s_x[tid] * (1.0f / 16.0f);
    s_x[tid] = v;

    float amax = fabsf(v);
    for (int off = 16; off > 0; off >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off));
    if ((tid & 31) == 0) s_max[tid >> 5] = amax;
    __syncthreads();
    if (tid < 8) {
        float m = s_max[tid];
        for (int off = 4; off > 0; off >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffu, m, off));
        s_max[0] = m;
    }
    __syncthreads();

    float scale = fmaxf(s_max[0], 1e-12f) * (1.0f / 127.0f);
    if (tid == 0) d_x[(size_t)tok * (Kd >> 8) + sb] = scale;
    if (tid < 64) {
        float inv = 1.0f / scale;
        uint32_t pk = 0u;
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int qi = __float2int_rn(s_x[4 * tid + i] * inv);
            qi = max(-127, min(127, qi));
            pk |= (uint32_t)(uint8_t(int8_t(qi))) << (8 * i);
        }
        x_q8x4[(size_t)tok * (Kd >> 2) + sb * 64 + tid] = (int32_t)pk;
    }
}

// =================================================================================================
// Host Table Builder & Exact Reference Checkers
// =================================================================================================
struct VtqTabHost {
    float   signed_f32[512];
    int32_t signed_i32[512];
    float   gain_f[8];
    int32_t gain_i[8];
};

template <int FMT>
static VtqTabHost build_signed_tab(const uint8_t* tab_canon, const uint8_t* gain_u8) {
    VtqTabHost th{};
    constexpr int NL    = fmt_nl<FMT>();
    constexpr int LBITS = (NL == 4) ? 2 : ((NL == 8) ? 3 : 4);
    constexpr int UB    = fmt_ub<FMT>();
    for (int i = 0; i < 8; i++) {
        th.gain_i[i] = (int32_t)gain_u8[i];
        th.gain_f[i] = (float)gain_u8[i];
    }
    if constexpr (FMT == 15) {
        const int8_t* ts = (const int8_t*)tab_canon;
        for (int i = 0; i < 32; i++) {
            th.signed_i32[i] = (int32_t)ts[i];
            th.signed_f32[i] = (float)ts[i];
        }
    } else {
        constexpr int M_MAG = NL * (1 << UB);
        for (int neg = 0; neg < 2; neg++) {
            for (int u_mag = 0; u_mag < (1 << UB); u_mag++) {
                for (int lb = 0; lb < NL; lb++) {
                    int s_idx   = (neg << (LBITS + UB)) | (u_mag << LBITS) | lb;
                    int mag_idx = ((u_mag << LBITS) | lb) ^ (neg ? (M_MAG - 1) : 0);
                    int w_s     = neg ? -(int)tab_canon[mag_idx] : (int)tab_canon[mag_idx];
                    th.signed_i32[s_idx] = w_s;
                    th.signed_f32[s_idx] = (float)w_s;
                }
            }
        }
    }
    return th;
}

template <int FMT>
static double host_verify_fp32(const uint8_t* d_ST, const __half* d_row, float unit, const float* d_x, const float* d_y,
                               const uint8_t* tab_canon, const uint8_t* gain_u8, int Kd, int n_rows_chk = 32) {
    constexpr size_t STRIDE = st_bytes<FMT>();
    constexpr int NL = fmt_nl<FMT>();
    constexpr int H  = fmt_h<FMT>();
    const int nsb = Kd >> 8;
    const int nbg = n_rows_chk >> 4;
    const int8_t* tab_s8 = (const int8_t*)tab_canon;

    std::vector<uint8_t> h_st((size_t)nbg * nsb * STRIDE);
    std::vector<__half>  h_dr(n_rows_chk);
    std::vector<float>   h_x(Kd), h_y(n_rows_chk);
    CK(cudaMemcpy(h_st.data(), d_ST,  h_st.size(),   cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(h_dr.data(), d_row, n_rows_chk * 2, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(h_x.data(),  d_x,   Kd * 4,         cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(h_y.data(),  d_y,   n_rows_chk * 4, cudaMemcpyDeviceToHost));

    double err = 0.0, nrm = 0.0;
    for (int bg = 0; bg < nbg; bg++) {
        double acc[16] = {0};
        for (int sb = 0; sb < nsb; sb++) {
            const uint8_t* base = h_st.data() + ((size_t)bg * nsb + sb) * STRIDE;
            const uint8_t (*c)[16][2] = (const uint8_t (*)[16][2])base;
            const uint16_t* hdr_p = (const uint16_t*)(base + STRIDE - 16 - (size_t)H * 32);
            const uint8_t*  u_sb  = base + STRIDE - 16;

            double sb_sum[16] = {0};

            for (int g = 0; g < 16; g++) {
                uint16_t st = (uint16_t)(((c[g][15][1] & 1u) << 8) | c[g][15][0]);
                for (int r = 0; r < 16; r++) {
                    int gi = (hdr_p[g * H + 0] >> r) & 1;
                    if (H >= 2) gi |= ((hdr_p[g * H + 1] >> r) & 1) << 1;
                    if (H >= 3) gi |= ((hdr_p[g * H + 2] >> r) & 1) << 2;
                    double g_val = (double)gain_u8[gi];
                    const int p = r >> 1, odd = r & 1;
                    for (int j = 0; j < 16; j++) {
                        uint8_t cb = (j < 8) ? c[g][r][1] : c[g][r][0];
                        uint8_t cbit = (cb >> (7 - (j & 7))) & 1u;
                        st = (uint16_t)(((st << 1) | cbit) & 0x3FFu);
                        int lb = (int)galois_lb_u32<NL>(st);
                        int w_s = 0;
                        if constexpr (FMT == 15) {
                            const uint8_t (*ham)[16] = (const uint8_t (*)[16])(base + 512);
                            int nib = (j < 8) ? (ham[g][r] & 15) : (ham[g][r] >> 4);
                            int u0 = (HOST_HAM_CW8[nib] >> (7 - (j & 7))) & 1;
                            w_s = (int)tab_s8[(u0 << 4) | lb];
                        } else {
                            const uint8_t (*usgn)[32] = (const uint8_t (*)[32])(base + 512);
                            int lane = 16 * odd + j;
                            int neg = (usgn[g][lane] >> p) & 1;
                            int wm = 0;
                            if constexpr (FMT == 20) {
                                wm = tab_canon[lb ^ (neg ? 15 : 0)];
                            } else if constexpr (FMT == 25) {
                                const uint8_t (*ham)[16] = (const uint8_t (*)[16])(base + 1024);
                                int nib = (j < 8) ? (ham[g][r] & 15) : (ham[g][r] >> 4);
                                int b0 = (HOST_HAM_CW8[nib] >> (7 - (j & 7))) & 1;
                                wm = tab_canon[((b0 << 4) | lb) ^ (neg ? 31 : 0)];
                            } else if constexpr (FMT == 30) {
                                const uint8_t (*ub0)[32] = (const uint8_t (*)[32])(base + 1024);
                                wm = tab_canon[((((ub0[g][lane] >> p) & 1) << 3) | lb) ^ (neg ? 15 : 0)];
                            } else if constexpr (FMT == 31) {
                                const uint8_t (*ub0)[32] = (const uint8_t (*)[32])(base + 1024);
                                wm = tab_canon[((((ub0[g][lane] >> p) & 1) << 4) | lb) ^ (neg ? 31 : 0)];
                            } else if constexpr (FMT == 35) {
                                const uint8_t (*ub1)[32] = (const uint8_t (*)[32])(base + 1024);
                                const uint8_t (*ham)[16] = (const uint8_t (*)[16])(base + 1536);
                                int nib = (j < 8) ? (ham[g][r] & 15) : (ham[g][r] >> 4);
                                int b0 = (HOST_HAM_CW8[nib] >> (7 - (j & 7))) & 1;
                                int b1 = (ub1[g][lane] >> p) & 1;
                                wm = tab_canon[((b1 << 5) | (b0 << 4) | lb) ^ (neg ? 63 : 0)];
                            } else if constexpr (FMT == 40) {
                                const uint8_t (*u2)[2][32] = (const uint8_t (*)[2][32])(base + 1024);
                                int um = (u2[g][p >> 2][lane] >> (2 * (p & 3))) & 3;
                                wm = tab_canon[((um << 4) | lb) ^ (neg ? 63 : 0)];
                            } else if constexpr (FMT == 50) {
                                const uint8_t (*u2)[2][32] = (const uint8_t (*)[2][32])(base + 1024);
                                const uint8_t (*ub2)[32]   = (const uint8_t (*)[32])(base + 2048);
                                int um = (((ub2[g][lane] >> p) & 1) << 2) | ((u2[g][p >> 2][lane] >> (2 * (p & 3))) & 3);
                                wm = tab_canon[((um << 2) | lb) ^ (neg ? 31 : 0)];
                            } else if constexpr (FMT == 51) {
                                const uint8_t (*u2)[2][32] = (const uint8_t (*)[2][32])(base + 1024);
                                const uint8_t (*ub2)[32]   = (const uint8_t (*)[32])(base + 2048);
                                int um = (((ub2[g][lane] >> p) & 1) << 2) | ((u2[g][p >> 2][lane] >> (2 * (p & 3))) & 3);
                                wm = tab_canon[((um << 4) | lb) ^ (neg ? 127 : 0)];
                            } else if constexpr (FMT == 60) {
                                const uint8_t (*u4)[4][32] = (const uint8_t (*)[4][32])(base + 1024);
                                int um = (u4[g][p >> 1][lane] >> (4 * (p & 1))) & 15;
                                wm = tab_canon[((um << 4) | lb) ^ (neg ? 255 : 0)];
                            }
                            w_s = neg ? -wm : wm;
                        }
                        sb_sum[r] += (double)w_s * g_val * (double)h_x[sb * 256 + 16 * g + j];
                    }
                }
            }
            for (int r = 0; r < 16; r++) acc[r] += sb_sum[r] * (double)u_sb[r];
        }
        for (int r = 0; r < 16; r++) {
            double dr = (double)__half2float(h_dr[bg * 16 + r]) * (double)unit * (1.0 / 64.0);
            double ref = acc[r] * dr;
            err += std::fabs((double)h_y[bg * 16 + r] - ref);
            nrm += std::fabs(ref);
        }
    }
    return err / (nrm + 1e-12);
}

template <int FMT, int B_TOK>
static double host_verify_dp4a(const uint8_t* d_ST, const __half* d_row, float unit,
                               const int32_t* d_xq8, const float* d_dx, const float* d_y,
                               const uint8_t* tab_canon, const uint8_t* gain_u8, int M, int Kd, int n_rows_chk = 32) {
    constexpr size_t STRIDE = st_bytes<FMT>();
    constexpr int NL = fmt_nl<FMT>();
    constexpr int H  = fmt_h<FMT>();
    const int nsb = Kd >> 8;
    const int nbg = n_rows_chk >> 4;
    const int8_t* tab_s8 = (const int8_t*)tab_canon;

    std::vector<uint8_t> h_st((size_t)nbg * nsb * STRIDE);
    std::vector<__half>  h_dr(n_rows_chk);
    std::vector<int8_t>  h_xq8((size_t)B_TOK * Kd);
    std::vector<float>   h_dx((size_t)B_TOK * nsb);
    CK(cudaMemcpy(h_st.data(),  d_ST,  h_st.size(),     cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(h_dr.data(),  d_row, n_rows_chk * 2,  cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(h_xq8.data(), d_xq8, h_xq8.size(),    cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(h_dx.data(),  d_dx,  h_dx.size() * 4, cudaMemcpyDeviceToHost));

    double err = 0.0, nrm = 0.0;
    for (int bt = 0; bt < B_TOK; bt++) {
        std::vector<float> h_y(n_rows_chk);
        CK(cudaMemcpy(h_y.data(), d_y + (size_t)bt * M, n_rows_chk * 4, cudaMemcpyDeviceToHost));
        for (int bg = 0; bg < nbg; bg++) {
            double acc[16] = {0};
            for (int sb = 0; sb < nsb; sb++) {
                const uint8_t* base = h_st.data() + ((size_t)bg * nsb + sb) * STRIDE;
                const uint8_t (*c)[16][2] = (const uint8_t (*)[16][2])base;
                const uint16_t* hdr_p = (const uint16_t*)(base + STRIDE - 16 - (size_t)H * 32);
                const uint8_t*  u_sb  = base + STRIDE - 16;

                int32_t isum[16] = {0};

                for (int g = 0; g < 16; g++) {
                    uint16_t st = (uint16_t)(((c[g][15][1] & 1u) << 8) | c[g][15][0]);
                    for (int r = 0; r < 16; r++) {
                        int gi = (hdr_p[g * H + 0] >> r) & 1;
                        if (H >= 2) gi |= ((hdr_p[g * H + 1] >> r) & 1) << 1;
                        if (H >= 3) gi |= ((hdr_p[g * H + 2] >> r) & 1) << 2;
                        int32_t g_val = (int32_t)gain_u8[gi];
                        const int p = r >> 1, odd = r & 1;
                        int32_t sub_s = 0;
                        for (int j = 0; j < 16; j++) {
                            uint8_t cb = (j < 8) ? c[g][r][1] : c[g][r][0];
                            uint8_t cbit = (cb >> (7 - (j & 7))) & 1u;
                            st = (uint16_t)(((st << 1) | cbit) & 0x3FFu);
                            int lb = (int)galois_lb_u32<NL>(st);
                            int w_s = 0;
                            if constexpr (FMT == 15) {
                                const uint8_t (*ham)[16] = (const uint8_t (*)[16])(base + 512);
                                int nib = (j < 8) ? (ham[g][r] & 15) : (ham[g][r] >> 4);
                                int u0 = (HOST_HAM_CW8[nib] >> (7 - (j & 7))) & 1;
                                w_s = (int)tab_s8[(u0 << 4) | lb];
                            } else {
                                const uint8_t (*usgn)[32] = (const uint8_t (*)[32])(base + 512);
                                int lane = 16 * odd + j;
                                int neg = (usgn[g][lane] >> p) & 1;
                                int wm = 0;
                                if constexpr (FMT == 20) {
                                    wm = tab_canon[lb ^ (neg ? 15 : 0)];
                                } else if constexpr (FMT == 25) {
                                    const uint8_t (*ham)[16] = (const uint8_t (*)[16])(base + 1024);
                                    int nib = (j < 8) ? (ham[g][r] & 15) : (ham[g][r] >> 4);
                                    int b0 = (HOST_HAM_CW8[nib] >> (7 - (j & 7))) & 1;
                                    wm = tab_canon[((b0 << 4) | lb) ^ (neg ? 31 : 0)];
                                } else if constexpr (FMT == 30) {
                                    const uint8_t (*ub0)[32] = (const uint8_t (*)[32])(base + 1024);
                                    wm = tab_canon[((((ub0[g][lane] >> p) & 1) << 3) | lb) ^ (neg ? 15 : 0)];
                                } else if constexpr (FMT == 31) {
                                    const uint8_t (*ub0)[32] = (const uint8_t (*)[32])(base + 1024);
                                    wm = tab_canon[((((ub0[g][lane] >> p) & 1) << 4) | lb) ^ (neg ? 31 : 0)];
                                } else if constexpr (FMT == 35) {
                                    const uint8_t (*ub1)[32] = (const uint8_t (*)[32])(base + 1024);
                                    const uint8_t (*ham)[16] = (const uint8_t (*)[16])(base + 1536);
                                    int nib = (j < 8) ? (ham[g][r] & 15) : (ham[g][r] >> 4);
                                    int b0 = (HOST_HAM_CW8[nib] >> (7 - (j & 7))) & 1;
                                    int b1 = (ub1[g][lane] >> p) & 1;
                                    wm = tab_canon[((b1 << 5) | (b0 << 4) | lb) ^ (neg ? 63 : 0)];
                                } else if constexpr (FMT == 40) {
                                    const uint8_t (*u2)[2][32] = (const uint8_t (*)[2][32])(base + 1024);
                                    int um = (u2[g][p >> 2][lane] >> (2 * (p & 3))) & 3;
                                    wm = tab_canon[((um << 4) | lb) ^ (neg ? 63 : 0)];
                                } else if constexpr (FMT == 50) {
                                    const uint8_t (*u2)[2][32] = (const uint8_t (*)[2][32])(base + 1024);
                                    const uint8_t (*ub2)[32]   = (const uint8_t (*)[32])(base + 2048);
                                    int um = (((ub2[g][lane] >> p) & 1) << 2) | ((u2[g][p >> 2][lane] >> (2 * (p & 3))) & 3);
                                    wm = tab_canon[((um << 2) | lb) ^ (neg ? 31 : 0)];
                                } else if constexpr (FMT == 51) {
                                    const uint8_t (*u2)[2][32] = (const uint8_t (*)[2][32])(base + 1024);
                                    const uint8_t (*ub2)[32]   = (const uint8_t (*)[32])(base + 2048);
                                    int um = (((ub2[g][lane] >> p) & 1) << 2) | ((u2[g][p >> 2][lane] >> (2 * (p & 3))) & 3);
                                    wm = tab_canon[((um << 4) | lb) ^ (neg ? 127 : 0)];
                                } else if constexpr (FMT == 60) {
                                    const uint8_t (*u4)[4][32] = (const uint8_t (*)[4][32])(base + 1024);
                                    int um = (u4[g][p >> 1][lane] >> (4 * (p & 1))) & 15;
                                    wm = tab_canon[((um << 4) | lb) ^ (neg ? 255 : 0)];
                                }
                                w_s = neg ? -wm : wm;
                            }
                            sub_s += w_s * (int)h_xq8[(size_t)bt * Kd + sb * 256 + 16 * g + j];
                        }
                        isum[r] += sub_s * g_val;
                    }
                }
                double dx = (double)h_dx[(size_t)bt * nsb + sb];
                for (int r = 0; r < 16; r++) acc[r] += (double)isum[r] * (double)u_sb[r] * dx;
            }
            for (int r = 0; r < 16; r++) {
                double dr = (double)__half2float(h_dr[bg * 16 + r]) * (double)unit * (1.0 / 64.0);
                double ref = acc[r] * dr;
                err += std::fabs((double)h_y[bg * 16 + r] - ref);
                nrm += std::fabs(ref);
            }
        }
    }
    return err / (nrm + 1e-12);
}

static int g_sms = 0;
static const int KD = 8192;
static float   *g_x = nullptr, *g_y = nullptr, *g_dx = nullptr;
static __half  *g_drow = nullptr;
static int32_t *g_xq8 = nullptr;
static float   *g_lut_f = nullptr, *g_gain_f = nullptr;
static int32_t *g_lut_i = nullptr, *g_gain_i = nullptr;

template <typename F>
static float time_ms(F launch, int iters) {
    cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    for (int i = 0; i < 3; i++) launch();
    CK(cudaEventRecord(a));
    for (int i = 0; i < iters; i++) launch();
    CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b)); CK(cudaGetLastError());
    float ms = 0.0f; CK(cudaEventElapsedTime(&ms, a, b));
    cudaEventDestroy(a); cudaEventDestroy(b);
    return ms / iters;
}

static const size_t MAX_M = 1572864;

template <int FMT>
static void bench_fp32(size_t target_bytes, const char* tag, const char* label,
                       const uint8_t* tab_canon, const uint8_t* gain_u8) {
    constexpr size_t STRIDE = st_bytes<FMT>();
    const int nsb = KD >> 8;
    const int nbg = std::min((int)(MAX_M / 16), std::max(8, (int)(target_bytes / (STRIDE * nsb))));
    const int M   = nbg * 16;
    const size_t nst = (size_t)nbg * nsb;
    const size_t total_bytes = nst * STRIDE;
    const float unit = 1.0f / 32.0f;

    uint8_t* d_ST = nullptr;
    CK(cudaMalloc(&d_ST, total_bytes));
    fill_bytes<<<1024, 256>>>(d_ST, total_bytes, 100 + FMT);

    VtqTabHost th = build_signed_tab<FMT>(tab_canon, gain_u8);
    CK(cudaMemcpy(g_lut_f,  th.signed_f32, sizeof(th.signed_f32), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(g_gain_f, th.gain_f,     sizeof(th.gain_f),     cudaMemcpyHostToDevice));

    int nb1 = 0, nb8 = 0;
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb1, vtq_fp32_gemv<FMT, 1>, 256, 0));
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb8, vtq_fp32_gemv<FMT, 8>, 256, 0));
    const int iters = target_bytes > (256u << 20) ? 20 : 200;

    int grid1 = std::max(1, std::min(nb1 * g_sms, (nbg + 7) / 8));
    int grid8 = std::max(1, std::min(nb8 * g_sms, nbg));
    float ms1 = time_ms([&] { vtq_fp32_gemv<FMT, 1><<<grid1, 256>>>(d_ST, g_drow, unit, g_x, g_y, M, KD, g_lut_f, g_gain_f); }, iters);
    double rel1 = host_verify_fp32<FMT>(d_ST, g_drow, unit, g_x, g_y, tab_canon, gain_u8, KD, 32);
    float ms8 = time_ms([&] { vtq_fp32_gemv<FMT, 8><<<grid8, 256>>>(d_ST, g_drow, unit, g_x, g_y, M, KD, g_lut_f, g_gain_f); }, iters);
    double rel8 = host_verify_fp32<FMT>(d_ST, g_drow, unit, g_x, g_y, tab_canon, gain_u8, KD, 32);

    const bool use_sk8 = (ms8 <= ms1);
    float ms   = use_sk8 ? ms8 : ms1;
    double rel = std::max(rel1, rel8);
    double bpw = STRIDE * 8.0 / 4096.0 + 16.0 / KD;
    double bytes_total = (double)total_bytes + (double)M * 2.0;
    double wts = (double)M * KD;
    printf("  %-4s SK=%d %-27s %.3f bpw | err %.1e | %7.1f GB/s | %7.1f G weights/s\n",
           tag, use_sk8 ? 8 : 1, label, bpw, rel, bytes_total / ms / 1e6, wts / ms / 1e6);
    if (rel > 1e-5) { fprintf(stderr, "FP32 MISMATCH in %s: rel=%.3e\n", label, rel); exit(1); }
    CK(cudaFree(d_ST));
}

template <int FMT, int B_TOK>
static void bench_dp4a(size_t target_bytes, const char* tag, const char* label,
                       const uint8_t* tab_canon, const uint8_t* gain_u8) {
    constexpr size_t STRIDE = st_bytes<FMT>();
    const int nsb = KD >> 8;
    const int nbg = std::min((int)(MAX_M / 16), std::max(8, (int)(target_bytes / (STRIDE * nsb))));
    const int M   = nbg * 16;
    const size_t nst = (size_t)nbg * nsb;
    const size_t total_bytes = nst * STRIDE;
    const float unit = 1.0f / 32.0f;

    uint8_t* d_ST = nullptr;
    CK(cudaMalloc(&d_ST, total_bytes));
    fill_bytes<<<1024, 256>>>(d_ST, total_bytes, 300 + FMT);

    VtqTabHost th = build_signed_tab<FMT>(tab_canon, gain_u8);
    CK(cudaMemcpy(g_lut_i,  th.signed_i32, sizeof(th.signed_i32), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(g_gain_i, th.gain_i,     sizeof(th.gain_i),     cudaMemcpyHostToDevice));

    const int iters = target_bytes > (256u << 20) ? 20 : 200;
    float ms = 0.0f;
    double rel = 0.0;
    int sk_used = 1;

    if constexpr (B_TOK == 1) {
        int nb1 = 0, nb8 = 0;
        CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb1, vtq_dp4a_gemv_b1<FMT, 1>, 256, 0));
        CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb8, vtq_dp4a_gemv_b1<FMT, 8>, 256, 0));
        int grid1 = std::max(1, std::min(nb1 * g_sms, (nbg + 7) / 8));
        int grid8 = std::max(1, std::min(nb8 * g_sms, nbg));
        float ms1 = time_ms([&] { vtq_dp4a_gemv_b1<FMT, 1><<<grid1, 256>>>(d_ST, g_drow, unit, g_xq8, g_dx, g_y, M, KD, g_lut_i, g_gain_i); }, iters);
        double rel1 = host_verify_dp4a<FMT, 1>(d_ST, g_drow, unit, g_xq8, g_dx, g_y, tab_canon, gain_u8, M, KD, 32);
        float ms8 = time_ms([&] { vtq_dp4a_gemv_b1<FMT, 8><<<grid8, 256>>>(d_ST, g_drow, unit, g_xq8, g_dx, g_y, M, KD, g_lut_i, g_gain_i); }, iters);
        double rel8 = host_verify_dp4a<FMT, 1>(d_ST, g_drow, unit, g_xq8, g_dx, g_y, tab_canon, gain_u8, M, KD, 32);
        ms = (ms8 < ms1) ? ms8 : ms1;
        sk_used = (ms8 < ms1) ? 8 : 1;
        rel = std::max(rel1, rel8);
    } else {
        constexpr int B_TILE = (B_TOK >= 16) ? 16 : B_TOK;
        int nb1 = 0;
        CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb1, vtq_dp4a_gemm<FMT, B_TILE, 1>, 256, 0));
        int grid1 = std::max(1, std::min(nb1 * g_sms, (nbg + 7) / 8));
        dim3 grid_2d(grid1, B_TOK / B_TILE);
        ms = time_ms([&] { vtq_dp4a_gemm<FMT, B_TILE, 1><<<grid_2d, 256>>>(d_ST, g_drow, unit, g_xq8, g_dx, g_y, M, KD, g_lut_i, g_gain_i); }, iters);
        rel = host_verify_dp4a<FMT, B_TOK>(d_ST, g_drow, unit, g_xq8, g_dx, g_y, tab_canon, gain_u8, M, KD, 32);
    }
    (void)sk_used;
    double bpw = STRIDE * 8.0 / 4096.0 + 16.0 / KD;
    double bytes_total = (double)total_bytes + (double)M * 2.0;
    double wts_ops = (double)B_TOK * M * KD;
    printf("  %-4s B=%-2d  %-26s %.3f bpw | err %.1e | %7.1f GB/s | %8.1f G w*tok/s\n",
           tag, B_TOK, label, bpw, rel, bytes_total / ms / 1e6, wts_ops / ms / 1e6);
    if (rel > 1e-5) { fprintf(stderr, "DP4A MISMATCH in %s (B=%d): rel=%.3e\n", label, B_TOK, rel); exit(1); }
    CK(cudaFree(d_ST));
}

int main() {
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0)); g_sms = p.multiProcessorCount;
    const size_t max_m = MAX_M;
    CK(cudaMalloc(&g_x,      KD * 4));
    CK(cudaMalloc(&g_drow,   max_m * sizeof(__half)));
    CK(cudaMalloc(&g_xq8,    32 * KD));
    CK(cudaMalloc(&g_dx,     32 * (KD / 256) * 4));
    CK(cudaMalloc(&g_y,      32ull * max_m * 4));
    CK(cudaMalloc(&g_lut_f,  512 * 4));
    CK(cudaMalloc(&g_gain_f, 8 * 4));
    CK(cudaMalloc(&g_lut_i,  512 * 4));
    CK(cudaMalloc(&g_gain_i, 8 * 4));

    {
        std::vector<float>  hx(KD), hdx(32 * (KD / 256));
        std::vector<__half> hdr(max_m);
        std::vector<int8_t> hq8(32 * KD);
        for (int i = 0; i < KD; i++) hx[i] = ((int)(hash32(i, 99) & 0xffff) - 32768) / 32768.0f;
        for (size_t i = 0; i < max_m; i++) hdr[i] = __float2half(0.01f + 0.003f * (i & 15));
        for (size_t i = 0; i < hq8.size(); i++) hq8[i] = (int8_t)((int)(hash32(i, 101) % 255) - 127);
        for (size_t i = 0; i < hdx.size(); i++) hdx[i] = 0.01f + 0.005f * (i % 7);
        CK(cudaMemcpy(g_x,    hx.data(),  KD * 4,     cudaMemcpyHostToDevice));
        CK(cudaMemcpy(g_drow, hdr.data(), max_m * 2,  cudaMemcpyHostToDevice));
        CK(cudaMemcpy(g_xq8,  hq8.data(), hq8.size(), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(g_dx,   hdx.data(), hdx.size() * 4, cudaMemcpyHostToDevice));
    }

    uint8_t tab_s8[32] = {0}, tab_u8[256] = {0};
    for (int i = 0; i < 32; i++)  tab_s8[i] = (uint8_t)(int8_t)((int)(hash32(i, 11) % 255) - 127);
    for (int i = 0; i < 256; i++) tab_u8[i] = (uint8_t)(1 + (hash32(i, 22) % 127));
    const uint8_t gain_u8[8] = {42, 49, 56, 63, 71, 79, 89, 101};

    printf("=== VTQ 16x256 Super-Tile Bank-Conflict-Free GPU Benchmark (%s, %d SM) ===\n", p.name, g_sms);

    for (int pass = 0; pass < 2; pass++) {
        size_t tb = (pass == 0) ? (2048ull << 20) : (size_t)(p.l2CacheSize * 0.4);
        const char* tag = (pass == 0) ? "DRAM" : "L2";
        printf("\n-- FP32 GEMV %s (%.0f MB, batch=1, 16x256 Super-Tile + 0 bank conflicts!) --\n", tag, tb / 1048576.0);
        bench_fp32<15>(tb, tag, "VTQ K=1.5 Gosset84",    tab_s8, gain_u8);
        bench_fp32<20>(tb, tag, "VTQ K=2.0 nl=16",       tab_u8, gain_u8);
        bench_fp32<25>(tb, tag, "VTQ K=2.5 Gosset84",    tab_u8, gain_u8);
        bench_fp32<30>(tb, tag, "VTQ K=3.0 nl=8",        tab_u8, gain_u8);
        bench_fp32<31>(tb, tag, "VTQ K=3.0 nl=16",       tab_u8, gain_u8);
        bench_fp32<35>(tb, tag, "VTQ K=3.5 Gosset84",    tab_u8, gain_u8);
        bench_fp32<40>(tb, tag, "VTQ K=4.0 nl=16",       tab_u8, gain_u8);
        bench_fp32<50>(tb, tag, "VTQ K=5.0 nl=4",        tab_u8, gain_u8);
        bench_fp32<51>(tb, tag, "VTQ K=5.0 nl=16",       tab_u8, gain_u8);
        bench_fp32<60>(tb, tag, "VTQ K=6.0 nl=16",       tab_u8, gain_u8);
    }

    for (int pass = 0; pass < 2; pass++) {
        size_t tb = (pass == 0) ? (2048ull << 20) : (size_t)(p.l2CacheSize * 0.4);
        const char* tag = (pass == 0) ? "DRAM" : "L2";
        printf("\n-- DP4A INT8 GEMV / Multi-Token GEMM %s (%.0f MB, 16x256 Super-Tile + 0 bank conflicts!) --\n", tag, tb / 1048576.0);
        bench_dp4a<15, 1>(tb, tag, "VTQ-DP4A K=1.5 Gosset84", tab_s8, gain_u8);
        bench_dp4a<20, 1>(tb, tag, "VTQ-DP4A K=2.0 nl=16",    tab_u8, gain_u8);
        bench_dp4a<25, 1>(tb, tag, "VTQ-DP4A K=2.5 Gosset84", tab_u8, gain_u8);
        bench_dp4a<30, 1>(tb, tag, "VTQ-DP4A K=3.0 nl=8",     tab_u8, gain_u8);
        bench_dp4a<31, 1>(tb, tag, "VTQ-DP4A K=3.0 nl=16",    tab_u8, gain_u8);
        bench_dp4a<35, 1>(tb, tag, "VTQ-DP4A K=3.5 Gosset84", tab_u8, gain_u8);
        bench_dp4a<40, 1>(tb, tag, "VTQ-DP4A K=4.0 nl=16",    tab_u8, gain_u8);
        bench_dp4a<50, 1>(tb, tag, "VTQ-DP4A K=5.0 nl=4",     tab_u8, gain_u8);
        bench_dp4a<51, 1>(tb, tag, "VTQ-DP4A K=5.0 nl=16",    tab_u8, gain_u8);
        bench_dp4a<60, 1>(tb, tag, "VTQ-DP4A K=6.0 nl=16",    tab_u8, gain_u8);
        if (pass == 0) {
            bench_dp4a<20, 8>(tb,  tag, "VTQ-DP4A K=2.0 nl=16", tab_u8, gain_u8);
            bench_dp4a<20, 32>(tb, tag, "VTQ-DP4A K=2.0 nl=16", tab_u8, gain_u8);
            bench_dp4a<30, 8>(tb,  tag, "VTQ-DP4A K=3.0 nl=8",  tab_u8, gain_u8);
            bench_dp4a<30, 32>(tb, tag, "VTQ-DP4A K=3.0 nl=8",  tab_u8, gain_u8);
            bench_dp4a<31, 8>(tb,  tag, "VTQ-DP4A K=3.0 nl=16", tab_u8, gain_u8);
            bench_dp4a<31, 32>(tb, tag, "VTQ-DP4A K=3.0 nl=16", tab_u8, gain_u8);
            bench_dp4a<40, 8>(tb,  tag, "VTQ-DP4A K=4.0 nl=16", tab_u8, gain_u8);
            bench_dp4a<40, 32>(tb, tag, "VTQ-DP4A K=4.0 nl=16", tab_u8, gain_u8);
        }
    }

    {
        float* d_xin = nullptr; int8_t* d_sgn = nullptr;
        CK(cudaMalloc(&d_xin, 32 * KD * 4));
        CK(cudaMalloc(&d_sgn, KD));
        fill_bytes<<<64, 256>>>((uint8_t*)d_sgn, KD, 777);
        float ms_b1  = time_ms([&] { fused_fwht256_quant_q8<<<dim3(KD / 256, 1),  256>>>(d_xin, d_sgn, g_xq8, g_dx, KD); }, 500);
        float ms_b32 = time_ms([&] { fused_fwht256_quant_q8<<<dim3(KD / 256, 32), 256>>>(d_xin, d_sgn, g_xq8, g_dx, KD); }, 500);
        printf("\n-- Fused Block-256 FWHT + INT8 Dynamic Quantization Overhead (KD=%d) --\n", KD);
        printf("  B=1 : %.4f ms (%.2f us) | B=32 : %.4f ms (%.2f us)\n", ms_b1, ms_b1 * 1000.0f, ms_b32, ms_b32 * 1000.0f);
        CK(cudaFree(d_xin)); CK(cudaFree(d_sgn));
    }

    CK(cudaFree(g_x)); CK(cudaFree(g_drow)); CK(cudaFree(g_xq8)); CK(cudaFree(g_dx)); CK(cudaFree(g_y));
    CK(cudaFree(g_lut_f)); CK(cudaFree(g_gain_f)); CK(cudaFree(g_lut_i)); CK(cudaFree(g_gain_i));
    return 0;
}
