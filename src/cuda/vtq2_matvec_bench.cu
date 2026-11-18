// VTQ v2 Unified Bitplane GPU Blackwell GEMV Benchmark (`nvcc -O3 -arch=sm_120 src/cuda/vtq2_matvec_bench.cu -o vtq2_matvec_bench`)
// Verifies unified tile layout shared with CPU AVX2:
//   code[32B] + contiguous unc[UNC_WORDS * 4B] = { usgn[32B], power-of-two u_low planes } + hdr[4B] + scale[2B].
// Separating `usgn[32B]` makes K=4 `u_low` 16-bit/lane (`uint16`) and K=6 `u_low` 32-bit/lane (`uint32`) inside `unc[]`
// — zero unaligned 24-bit/40-bit cross-word bit-unpacking and zero separate-buffer TLB misses!
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA %s: %s (%s:%d)\n", #x, cudaGetErrorString(e), __FILE__, __LINE__); exit(1); } } while (0)

static __host__ __device__ inline uint32_t hash32(uint64_t i, uint32_t seed) {
    uint64_t x = i * 0x9E3779B97F4A7C15ull + seed * 0xD1B54A32D192ED03ull;
    x ^= x >> 31; x *= 0xBF58476D1CE4E5B9ull; x ^= x >> 29; x *= 0x94D049BB133111EBull; x ^= x >> 32;
    return (uint32_t)x;
}

__global__ void fill_u32(uint32_t* p, size_t n, uint32_t seed) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) p[i] = hash32(i, seed);
}
__global__ void fill_half(__half* p, size_t n, uint32_t seed, float lo, float hi) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
        p[i] = __float2half(lo + (hi - lo) * (hash32(i, seed) >> 8) * (1.0f / 16777216.0f));
}

struct Vtq2UniDev {
    const uint32_t *code, *unc, *hdr;
    const __half* scale;
};

template <int FMT>
__device__ __host__ __forceinline__ constexpr int k_ceil() {
    if constexpr (FMT == 20) return 2;
    if constexpr (FMT == 25 || FMT == 30) return 3;
    if constexpr (FMT == 35 || FMT == 40) return 4;
    if constexpr (FMT == 50) return 5;
    return 6;
}

template <int FMT>
__device__ __host__ __forceinline__ constexpr int unc_words() {
    if constexpr (FMT == 20) return 8;    // 32B usgn
    if constexpr (FMT == 25) return 12;   // 32B usgn + 16B ham4
    if constexpr (FMT == 30) return 16;   // 32B usgn + 32B ub0
    if constexpr (FMT == 35) return 20;   // 32B usgn + 32B ub1 + 16B ham4
    if constexpr (FMT == 40) return 24;   // 32B usgn + 64B u2 (aligned uint16 per lane!)
    if constexpr (FMT == 50) return 32;   // 32B usgn + 64B u2 + 32B ub0
    return 40;                            // 32B usgn + 128B u4 (aligned uint32 per lane!)
}

template <int FMT, int R>
__global__ void __launch_bounds__(256) vtq2_uni_matvec(
    Vtq2UniDev d, const float* __restrict__ x, float* __restrict__ y, int M, int Kd,
    const __half* __restrict__ gtab, const float* __restrict__ gains)
{
    constexpr int KC = k_ceil<FMT>();
    constexpr int UL_BITS = KC - 2;
    constexpr int UW = unc_words<FMT>();
    extern __shared__ __align__(16) unsigned char smem[];
    float* sg = (float*)smem;
    uint32_t* sham = (uint32_t*)(smem + 32);  // 16 x uint32 bank-conflict-free Gosset [8,4,4] masks in shared memory
    if (threadIdx.x < 8) sg[threadIdx.x] = gains[threadIdx.x];
    if (threadIdx.x < 16) {
        constexpr uint8_t cw[16] = {
            0x00, 0xff, 0xf0, 0x0f, 0xcc, 0x33, 0x3c, 0xc3,
            0xaa, 0x55, 0x5a, 0xa5, 0x66, 0x99, 0x96, 0x69
        };
        sham[threadIdx.x] = cw[threadIdx.x];
    }
    __half* st = (__half*)(smem + 96);
    for (int i = threadIdx.x; i < (1024 << UL_BITS); i += blockDim.x) st[i] = gtab[i];
    __syncthreads();

    const int l = threadIdx.x & 31, o = 8 * (l & 3);
    const int warps = gridDim.x * (blockDim.x >> 5), nt = Kd >> 8;
    for (int row0 = (blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5)) * R; row0 < M; row0 += warps * R) {
        float accs[R];
        #pragma unroll
        for (int r = 0; r < R; r++) accs[r] = 0.f;
        #pragma unroll 2
        for (int j = 0; j < nt; j++) {
            const float4 xa = __ldg((const float4*)(x + (size_t)j * 256 + 8 * l));
            const float4 xb = __ldg((const float4*)(x + (size_t)j * 256 + 8 * l + 4));
            const float xs[8] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w};
            #pragma unroll
            for (int r = 0; r < R; r++) {
                const size_t tile = (size_t)(row0 + r) * nt + j;
                const uint32_t myw = __ldg(d.code + tile * 8 + (l & 7));
                const uint32_t* ut = d.unc + tile * UW;
                const uint32_t sbyte = (__ldg(ut + (l >> 2)) >> o) & 0xffu;
                const uint32_t hdr = __ldg(d.hdr + tile);
                const float sc = __half2float(d.scale[tile]);

                const uint32_t cw = __shfl_sync(0xffffffffu, myw, l >> 2);
                const uint32_t pw = __shfl_sync(0xffffffffu, myw, ((l >> 2) + 7) & 7);
                const uint32_t win17 = (uint32_t)((((uint64_t)cw << 32) | pw) >> (23 + o));

                uint32_t u_main_l = 0, u_aux_l = 0;
                if constexpr (FMT == 25) {
                    const uint32_t hw = __ldg(ut + 8 + (l >> 3));
                    u_main_l = sham[(hw >> (4 * (l & 7))) & 0xfu];
                } else if constexpr (FMT == 30) {
                    u_main_l = (__ldg(ut + 8 + (l >> 2)) >> o) & 0xffu;
                } else if constexpr (FMT == 35) {
                    u_main_l = (__ldg(ut + 8 + (l >> 2)) >> o) & 0xffu;
                    const uint32_t hw = __ldg(ut + 16 + (l >> 3));
                    u_aux_l = sham[(hw >> (4 * (l & 7))) & 0xfu];
                } else if constexpr (FMT == 40) {
                    u_main_l = (__ldg(ut + 8 + (l >> 1)) >> (16 * (l & 1))) & 0xffffu;
                } else if constexpr (FMT == 50) {
                    u_main_l = (__ldg(ut + 8 + (l >> 1)) >> (16 * (l & 1))) & 0xffffu;
                    u_aux_l = (__ldg(ut + 24 + (l >> 2)) >> o) & 0xffu;
                } else if constexpr (FMT == 60) {
                    u_main_l = __ldg(ut + 8 + l);
                }

                float part = 0.f;
                #pragma unroll
                for (int i = 0; i < 8; i++) {
                    const uint32_t w10 = (win17 >> i) & 0x3ffu;
                    const uint32_t neg = (sbyte >> i) & 1u;
                    uint32_t ul = 0;
                    if constexpr (FMT == 25 || FMT == 30) {
                        ul = ((u_main_l >> i) & 1u) ^ neg;
                    } else if constexpr (FMT == 35) {
                        const uint32_t raw = (((u_main_l >> i) & 1u) << 1) | ((u_aux_l >> i) & 1u);
                        ul = raw ^ (neg ? 3u : 0u);
                    } else if constexpr (FMT == 40) {
                        ul = ((u_main_l >> (2 * i)) & 3u) ^ (neg ? 3u : 0u);
                    } else if constexpr (FMT == 50) {
                        const uint32_t raw = (((u_aux_l >> i) & 1u) << 2) | ((u_main_l >> (2 * i)) & 3u);
                        ul = raw ^ (neg ? 7u : 0u);
                    } else if constexpr (FMT == 60) {
                        ul = ((u_main_l >> (4 * i)) & 15u) ^ (neg ? 15u : 0u);
                    }
                    const uint32_t w_fold = neg ? (w10 ^ 0x3ffu) : w10;
                    float v = __half2float(st[(w_fold << UL_BITS) | ul]);
                    if (neg) v = -v;
                    part = fmaf(v, xs[i], part);
                }
                accs[r] = fmaf(part, sg[(hdr >> (2 * (l >> 1))) & 3] * sc, accs[r]);
            }
        }
        #pragma unroll
        for (int r = 0; r < R; r++) {
            float acc = accs[r];
            #pragma unroll
            for (int s = 16; s; s >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, s);
            if (l == 0) y[row0 + r] = acc;
        }
    }
}

static int g_sms;
static const int KD = 8192;
static float *g_x, *g_y;

template <int FMT>
static double verify_host_ref(const Vtq2UniDev& d, const __half* d_comb, const float* d_gains, int n_rows_chk = 16) {
    constexpr int KC = k_ceil<FMT>();
    constexpr int UL_BITS = KC - 2;
    constexpr int UW = unc_words<FMT>();
    const int nt = KD >> 8;
    const size_t ntile = (size_t)n_rows_chk * nt;

    std::vector<uint32_t> h_code(ntile * 8), h_unc(ntile * UW), h_hdr(ntile);
    std::vector<__half> h_sc(ntile), h_comb(1024 << UL_BITS);
    std::vector<float> h_x(KD), h_y(n_rows_chk), h_g(8);

    CK(cudaMemcpy(h_code.data(), d.code, ntile * 32, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(h_unc.data(),  d.unc,  ntile * UW * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(h_hdr.data(),  d.hdr,  ntile * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(h_sc.data(),   d.scale, ntile * 2, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(h_comb.data(), d_comb, (1024 << UL_BITS) * 2, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(h_x.data(),    g_x,    KD * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(h_y.data(),    g_y,    n_rows_chk * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(h_g.data(),    d_gains, 32, cudaMemcpyDeviceToHost));

    static const uint8_t cw8[16] = {
        0x00, 0xff, 0xf0, 0x0f, 0xcc, 0x33, 0x3c, 0xc3,
        0xaa, 0x55, 0x5a, 0xa5, 0x66, 0x99, 0x96, 0x69
    };

    double err = 0.0, nrm = 0.0;
    for (int row = 0; row < n_rows_chk; row++) {
        double acc = 0.0;
        for (int j = 0; j < nt; j++) {
            size_t tile = (size_t)row * nt + j;
            const uint32_t* tc = &h_code[tile * 8];
            const uint32_t* ut = &h_unc[tile * UW];
            uint32_t hdr = h_hdr[tile];
            float sc = __half2float(h_sc[tile]);
            for (int l = 0; l < 32; l++) {
                int o = 8 * (l & 3);
                uint32_t sbyte = (ut[l >> 2] >> o) & 0xffu;
                uint32_t cw = tc[l >> 2];
                uint32_t pw = tc[((l >> 2) + 7) & 7];
                uint32_t win17 = (uint32_t)((((uint64_t)cw << 32) | pw) >> (23 + o));
                uint32_t u_main_l = 0, u_aux_l = 0;
                if constexpr (FMT == 25) {
                    u_main_l = cw8[(ut[8 + (l >> 3)] >> (4 * (l & 7))) & 0xfu];
                } else if constexpr (FMT == 30) {
                    u_main_l = (ut[8 + (l >> 2)] >> o) & 0xffu;
                } else if constexpr (FMT == 35) {
                    u_main_l = (ut[8 + (l >> 2)] >> o) & 0xffu;
                    u_aux_l = cw8[(ut[16 + (l >> 3)] >> (4 * (l & 7))) & 0xfu];
                } else if constexpr (FMT == 40) {
                    u_main_l = (ut[8 + (l >> 1)] >> (16 * (l & 1))) & 0xffffu;
                } else if constexpr (FMT == 50) {
                    u_main_l = (ut[8 + (l >> 1)] >> (16 * (l & 1))) & 0xffffu;
                    u_aux_l = (ut[24 + (l >> 2)] >> o) & 0xffu;
                } else if constexpr (FMT == 60) {
                    u_main_l = ut[8 + l];
                }
                double part = 0.0;
                for (int i = 0; i < 8; i++) {
                    uint32_t w10 = (win17 >> i) & 0x3ffu;
                    uint32_t neg = (sbyte >> i) & 1u;
                    uint32_t ul = 0;
                    if constexpr (FMT == 25 || FMT == 30) ul = ((u_main_l >> i) & 1u) ^ neg;
                    else if constexpr (FMT == 35) ul = ((((u_main_l >> i) & 1u) << 1) | ((u_aux_l >> i) & 1u)) ^ (neg ? 3u : 0u);
                    else if constexpr (FMT == 40) ul = ((u_main_l >> (2 * i)) & 3u) ^ (neg ? 3u : 0u);
                    else if constexpr (FMT == 50) ul = ((((u_aux_l >> i) & 1u) << 2) | ((u_main_l >> (2 * i)) & 3u)) ^ (neg ? 7u : 0u);
                    else if constexpr (FMT == 60) ul = ((u_main_l >> (4 * i)) & 15u) ^ (neg ? 15u : 0u);
                    uint32_t w_fold = neg ? (w10 ^ 0x3ffu) : w10;
                    float v = __half2float(h_comb[(w_fold << UL_BITS) | ul]);
                    if (neg) v = -v;
                    part += (double)v * (double)h_x[j * 256 + 8 * l + i];
                }
                acc += part * (double)(h_g[(hdr >> (2 * (l >> 1))) & 3] * sc);
            }
        }
        err += std::fabs(h_y[row] - acc);
        nrm += std::fabs(acc);
    }
    return err / (nrm + 1e-12);
}

template <typename F>
static float time_ms(F launch, int iters) {
    cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    for (int i = 0; i < 3; i++) launch();
    CK(cudaEventRecord(a));
    for (int i = 0; i < iters; i++) launch();
    CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b)); CK(cudaGetLastError());
    float ms; CK(cudaEventElapsedTime(&ms, a, b));
    cudaEventDestroy(a); cudaEventDestroy(b);
    return ms / iters;
}

template <int FMT, int R>
static void bench_uni(size_t target_bytes, const char* tag, const char* label,
                      const __half* comb, const float* gains) {
    constexpr int KC = k_ceil<FMT>();
    constexpr int UL_BITS = KC - 2;
    constexpr int UW = unc_words<FMT>();
    const size_t bpt = 32 + (size_t)UW * 4 + 6;
    const int M = (int)(target_bytes / (bpt * (KD / 256))) & ~7;
    const size_t ntile = (size_t)M * (KD / 256);
    uint32_t *code, *unc, *hdr; __half* scale;
    CK(cudaMalloc(&code, ntile * 32));
    CK(cudaMalloc(&unc,  ntile * UW * 4));
    CK(cudaMalloc(&hdr,  ntile * 4));
    CK(cudaMalloc(&scale, ntile * 2));
    fill_u32<<<1024, 256>>>(code, ntile * 8,  10 + FMT);
    fill_u32<<<1024, 256>>>(unc,  ntile * UW, 20 + FMT);
    fill_u32<<<1024, 256>>>(hdr,  ntile,      30 + FMT);
    fill_half<<<1024, 256>>>(scale, ntile,    40 + FMT, 0.5f, 1.5f);

    const size_t smem = 96 + (size_t)(1024 << UL_BITS) * 2;
    CK(cudaFuncSetAttribute(vtq2_uni_matvec<FMT, R>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    int nb; CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, vtq2_uni_matvec<FMT, R>, 256, smem));
    const int grid = nb * g_sms;
    const int iters = target_bytes > (256u << 20) ? 20 : 200;
    Vtq2UniDev d{code, unc, hdr, scale};
    float ms = time_ms([&] {
        vtq2_uni_matvec<FMT, R><<<grid, 256, smem>>>(d, g_x, g_y, M, KD, comb, gains);
    }, iters);
    double rel = verify_host_ref<FMT>(d, comb, gains);
    double bytes = (double)ntile * bpt, wts = (double)M * KD;
    printf("  %-4s R=%d %-24s %.3f bpw | err %.1e | %7.1f GB/s | %7.1f G weights/s\n",
           tag, R, label, bpt * 8.0 / 256.0, rel, bytes / ms / 1e6, wts / ms / 1e6);
    if (rel > 1e-5) { fprintf(stderr, "MISMATCH in %s: rel=%.3e\n", label, rel); exit(1); }
    cudaFree(code); cudaFree(unc); cudaFree(hdr); cudaFree(scale);
}

int main() {
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0)); g_sms = p.multiProcessorCount;
    CK(cudaMalloc(&g_x, KD * 4)); CK(cudaMalloc(&g_y, 64u << 20));
    fill_half<<<256, 256>>>((__half*)g_x, KD * 2, 1, -1.0f, 1.0f);
    // Initialize `g_x` with clean float32 values in [-1, 1]
    {
        std::vector<float> hx(KD);
        for (int i = 0; i < KD; i++) hx[i] = ((int)(hash32(i, 99) & 0xffff) - 32768) / 32768.0f;
        CK(cudaMemcpy(g_x, hx.data(), KD * 4, cudaMemcpyHostToDevice));
    }
    __half* comb; float* gains;
    CK(cudaMalloc(&comb, (1024 << 4) * 2)); fill_half<<<256, 256>>>(comb, 1024 << 4, 2, 0.f, 2.f);
    float hg[8] = {0.70f, 0.80f, 0.92f, 1.00f, 1.08f, 1.20f, 1.32f, 1.45f};
    CK(cudaMalloc(&gains, 32)); CK(cudaMemcpy(gains, hg, 32, cudaMemcpyHostToDevice));

    printf("=== VTQ v2 Unified Bitplane GPU Blackwell GEMV (%s, %d SM) ===\n", p.name, g_sms);
    for (int pass = 0; pass < 2; pass++) {
        size_t tb = pass == 0 ? (2048ull << 20) : (size_t)(p.l2CacheSize * 0.4);
        const char* tag = pass == 0 ? "DRAM" : "L2";
        printf("\n-- %s (%.0f MB) --\n", tag, tb / 1048576.0);
        bench_uni<20, 2>(tb, tag, "VTQ2 K=2.0 AT-Comb",      comb, gains);
        bench_uni<25, 4>(tb, tag, "VTQ2 K=2.5 Gosset84",     comb, gains);
        bench_uni<30, 4>(tb, tag, "VTQ2 K=3.0 AT-Comb",      comb, gains);
        bench_uni<35, 4>(tb, tag, "VTQ2 K=3.5 Gosset84",     comb, gains);
        bench_uni<40, 4>(tb, tag, "VTQ2 K=4.0 AT-Comb(u16)", comb, gains);
        bench_uni<50, 4>(tb, tag, "VTQ2 K=5.0 AT-Comb",      comb, gains);
        bench_uni<60, 4>(tb, tag, "VTQ2 K=6.0 AT-Comb(u32)", comb, gains);
    }
    return 0;
}
