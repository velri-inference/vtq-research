// VTQ v2 Compact Unified Bitplane CPU AVX2 Benchmark (`gcc -O3 -mavx2 -mfma -mno-avx512f src/cpu/vtq2_avx2_bench.c -lm`)
// Benchmarks exact compact per-format structs (zero wasted padding), AT-32 / AT-64 / AT2(16+16) / AT2Z(32+32) `qmax=128`
// codebooks, and Extended Hamming [8, 4, 4] Gosset E_8 fractional planes (K=2.5, 3.5) with scalar reference verification.
#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define M_ROWS 256
#define K_COLS 2048
#define NTILE  (K_COLS / 16)

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static inline float hsum4(__m128 x) {
    __m128 s = _mm_add_ps(x, _mm_movehl_ps(x, x));
    s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
    return _mm_cvtss_f32(s);
}

static inline __m256i lut_pair(const uint8_t L[256][16], int i0, int i1) {
    return _mm256_set_m128i(_mm_loadu_si128((const __m128i*)L[i1]),
                            _mm_loadu_si128((const __m128i*)L[i0]));
}

static const int WM[3] = {359, 40, 475};
static uint8_t LH[512][16], L0[256][16], L1[256][16];

// Subset label bits placed at bit positions [bit_base .. bit_base + 3]
static void build_luts_at(int bit_base) {
    memset(LH, 0, sizeof(LH));
    memset(L0, 0, sizeof(L0));
    memset(L1, 0, sizeof(L1));
    for (int hi = 0; hi < 512; hi++) {
        for (int j = 0; j < 16; j++) {
            int lb = 0;
            for (int b = 0; b < 9; b++) {
                int pos = j - 1 - b;
                if (pos < 0) {
                    int bit = (hi >> (8 + pos + 1)) & 1;
                    for (int k = 0; k < 3; k++) if ((WM[k] >> b) & 1) lb ^= bit << (bit_base + 1 + k);
                }
            }
            LH[hi][j] = (uint8_t)lb;
        }
    }
    for (int byte = 0; byte < 256; byte++) {
        for (int j = 0; j < 16; j++) {
            int lb0 = (j < 8) ? (((byte >> j) & 1) << bit_base) : 0;
            int lb1 = (j >= 8) ? (((byte >> (j - 8)) & 1) << bit_base) : 0;
            for (int b = 0; b < 9; b++) {
                int pos = j - 1 - b;
                if (pos >= 0 && pos < 8) {
                    int bit = (byte >> pos) & 1;
                    for (int k = 0; k < 3; k++) if ((WM[k] >> b) & 1) lb0 ^= bit << (bit_base + 1 + k);
                } else if (pos >= 8 && pos < 16) {
                    int bit = (byte >> (pos - 8)) & 1;
                    for (int k = 0; k < 3; k++) if ((WM[k] >> b) & 1) lb1 ^= bit << (bit_base + 1 + k);
                }
            }
            L0[byte][j] = (uint8_t)lb0;
            L1[byte][j] = (uint8_t)lb1;
        }
    }
}

// Compact per-format structs (16x16 = 256 weights per tile, row-interleaved across 8 row-pairs g=0..7):
typedef struct { uint16_t c[16]; uint8_t usgn[32]; uint32_t hdr; float sc; } tile_k2_t;                             // 72 B
typedef struct { uint16_t c[16]; uint8_t usgn[32]; uint8_t ham[16]; uint32_t hdr; float sc; } tile_k25_t;           // 88 B
typedef struct { uint16_t c[16]; uint8_t usgn[32]; uint8_t ub0[32]; uint32_t hdr; float sc; } tile_k3_t;            // 104 B
typedef struct { uint16_t c[16]; uint8_t usgn[32]; uint8_t ub1[32]; uint8_t ham[16]; uint32_t hdr; float sc; } tile_k35_t; // 120 B
typedef struct { uint16_t c[16]; uint8_t usgn[32]; uint8_t u2[64];  uint32_t hdr; float sc; } tile_k4_t;            // 136 B
typedef struct { uint16_t c[16]; uint8_t usgn[32]; uint8_t u2[64];  uint8_t ub0[32]; uint32_t hdr; float sc; } tile_k5_t;  // 168 B
typedef struct { uint16_t c[16]; uint8_t usgn[32]; uint8_t u2[64];  uint8_t u2h[64]; uint32_t hdr; float sc; } tile_k6_t;  // 200 B

// 16 Extended Hamming [8, 4, 4] codewords expanded with a single `_mm256_shuffle_epi8` instruction
static const uint8_t HAM_CW8[16] = {
    0x00, 0xff, 0xf0, 0x0f, 0xcc, 0x33, 0x3c, 0xc3,
    0xaa, 0x55, 0x5a, 0xa5, 0x66, 0x99, 0x96, 0x69
};

static inline int state_lb16(const uint16_t c[16], int r, int j) {
    int lb = (c[r] >> j) & 1;
    for (int b = 0; b < 9; b++) {
        int pos = j - 1 - b;
        int bit = (pos >= 0) ? ((c[r] >> pos) & 1) : ((c[(r + 15) & 15] >> (16 + pos)) & 1);
        for (int k = 0; k < 3; k++) if ((WM[k] >> b) & 1) lb ^= bit << (1 + k);
    }
    return lb;
}

static void scalar_ref(const void* T_raw, size_t stride, const uint8_t* tab, const float* gains, const int8_t* a, float* y, int fmt) {
    for (int b = 0; b < M_ROWS / 16; b++) {
        double acc[16] = {0};
        for (int t = 0; t < NTILE; t++) {
            const uint8_t* p = (const uint8_t*)T_raw + (b * NTILE + t) * stride;
            const uint16_t* c = (const uint16_t*)p;
            const uint8_t* usgn = p + 32;
            uint32_t hdr = *(const uint32_t*)(p + stride - 8);
            float sc = *(const float*)(p + stride - 4);
            for (int g = 0; g < 8; g++) {
                for (int half = 0; half < 2; half++) {
                    int r = 2 * g + half;
                    float rsc = sc * gains[(hdr >> (2 * r)) & 3];
                    int sum_i = 0;
                    for (int j = 0; j < 16; j++) {
                        int lane = half * 16 + j;
                        int neg = (usgn[lane] >> g) & 1;
                        int lb = state_lb16(c, r, j);
                        int ul = 0, w_mag = 0;
                        if (fmt == 20) {
                            int fidx = lb ^ (neg ? 15 : 0);
                            w_mag = tab[fidx];
                        } else if (fmt == 25) {
                            const uint8_t* ham = p + 64;
                            int nib = (half == 0) ? (ham[j] & 0x0f) : (ham[j] >> 4);
                            ul = (HAM_CW8[nib] >> g) & 1;
                            int fidx = ((lb << 1) | ul) ^ (neg ? 31 : 0);
                            w_mag = tab[fidx];
                        } else if (fmt == 30) {
                            const uint8_t* ub0 = p + 64;
                            ul = (ub0[lane] >> g) & 1;
                            int fidx = ((lb << 1) | ul) ^ (neg ? 31 : 0);
                            w_mag = tab[fidx];
                        } else if (fmt == 35) {
                            const uint8_t* ub1 = p + 64;
                            const uint8_t* ham = p + 96;
                            int nib = (half == 0) ? (ham[j] & 0x0f) : (ham[j] >> 4);
                            ul = (((ub1[lane] >> g) & 1) << 1) | ((HAM_CW8[nib] >> g) & 1);
                            int fv = ((lb << 2) | ul) ^ (neg ? 63 : 0);
                            int idxA = (fv & 3) | (((fv >> 2) & 3) << 2);
                            int idxB = (fv & 3) | (((fv >> 4) & 3) << 2);
                            int s = tab[idxA] + tab[16 + idxB];
                            w_mag = s > 255 ? 255 : s;
                        } else if (fmt == 40) {
                            const uint8_t* u2 = p + 64;
                            int src_b = u2[(g >= 4 ? 32 : 0) + lane];
                            ul = (src_b >> (2 * (g & 3))) & 3;
                            int fv = ((lb << 2) | ul) ^ (neg ? 63 : 0);
                            int idxA = (fv & 3) | (((fv >> 2) & 3) << 2);
                            int idxB = (fv & 3) | (((fv >> 4) & 3) << 2);
                            int s = tab[idxA] + tab[16 + idxB];
                            w_mag = s > 255 ? 255 : s;
                        } else if (fmt == 50) {
                            const uint8_t* u2 = p + 64;
                            const uint8_t* ub0 = p + 128;
                            int u01 = (u2[(g >= 4 ? 32 : 0) + lane] >> (2 * (g & 3))) & 3;
                            int b2 = (ub0[lane] >> g) & 1;
                            ul = u01 | (b2 << 2);
                            int fv = ((lb << 3) | ul) ^ (neg ? 127 : 0);
                            int ul_f = fv & 7, lb_f = (fv >> 3) & 15;
                            int idxA = ul_f | ((lb_f & 3) << 3);
                            int idxB = ul_f | ((lb_f >> 2) << 3);
                            int s = tab[idxA] + tab[32 + idxB];
                            w_mag = s > 255 ? 255 : s;
                        } else if (fmt == 60) {
                            const uint8_t* u2 = p + 64;
                            const uint8_t* u2h = p + 128;
                            int u01 = (u2[(g >= 4 ? 32 : 0) + lane] >> (2 * (g & 3))) & 3;
                            int u23 = (u2h[(g >= 4 ? 32 : 0) + lane] >> (2 * (g & 3))) & 3;
                            ul = u01 | (u23 << 2);
                            int fv = ((lb << 4) | ul) ^ (neg ? 255 : 0);
                            int ul_f = fv & 15, lb_f = (fv >> 4) & 15;
                            int idxA = ul_f | ((lb_f & 1) << 4);
                            int idxB = (ul_f >> 2) | ((lb_f >> 1) << 2);
                            int s = tab[idxA] + tab[32 + idxB];
                            w_mag = s > 255 ? 255 : s;
                        }
                        int av = a[16 * t + j];
                        sum_i += w_mag * (neg ? -av : av);
                    }
                    acc[r] += (double)sum_i * rsc;
                }
            }
        }
        for (int r = 0; r < 16; r++) y[b * 16 + r] = (float)acc[r];
    }
}

#define AVX2_PROLOGUE() \
    const __m256i zero = _mm256_setzero_si256(), one = _mm256_set1_epi8(1), m4 = _mm256_set1_epi8(0x0f), m80 = _mm256_set1_epi8((char)0x80); \
    const __m256i sh_lo = _mm256_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14), sh_hi = _mm256_setr_epi32(16, 18, 20, 22, 24, 26, 28, 30); \
    const __m256i three = _mm256_set1_epi32(3); \
    const __m256 gv = _mm256_castps128_ps256(_mm_loadu_ps(gains)); \
    const __m256i H_LUT = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)HAM_CW8));

#define AVX2_STEP_V_NEG(g, c, sv, v, neg) \
    const int r0 = 2 * (g), r1 = 2 * (g) + 1; \
    __m256i v = lut_pair(LH, (c)[(r0 + 15) & 15] >> 7, (c)[r0] >> 7); \
    v = _mm256_xor_si256(v, lut_pair(L0, (c)[r0] & 0xff, (c)[r1] & 0xff)); \
    v = _mm256_xor_si256(v, lut_pair(L1, (c)[r0] >> 8, (c)[r1] >> 8)); \
    __m256i sbit = _mm256_and_si256((g) == 7 ? (sv) : _mm256_slli_epi16((sv), 7 - (g)), m80); \
    __m256i neg = _mm256_cmpgt_epi8(zero, sbit);

#define AVX2_ACCUM(g, w_mag, neg, a2, sc_lo, sc_hi, acc) \
    __m256i p = _mm256_maddubs_epi16((w_mag), _mm256_sign_epi8((a2), _mm256_or_si256((neg), one))); \
    __m256i s = _mm256_madd_epi16(p, _mm256_set1_epi16(1)); \
    __m256 scs = ((g) < 4) ? (sc_lo) : (sc_hi); \
    const int q = (2 * (g)) & 7; \
    __m256 sp = _mm256_permutevar8x32_ps(scs, _mm256_setr_epi32(q, q, q, q, q + 1, q + 1, q + 1, q + 1)); \
    (acc)[g] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(s), sp, (acc)[g]);

static void kern_k2(const void* T_raw, const uint8_t* tab, const float* gains, const int8_t* a, float* y) {
    const tile_k2_t* T = (const tile_k2_t*)T_raw;
    const __m256i T0 = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)tab));
    AVX2_PROLOGUE(); (void)H_LUT;
    for (int b = 0; b < M_ROWS / 16; b++) {
        __m256 acc[8]; for (int g = 0; g < 8; g++) acc[g] = _mm256_setzero_ps();
        for (int j = 0; j < NTILE; j++) {
            const tile_k2_t* tl = &T[b * NTILE + j];
            const __m256i a2 = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)(a + 16 * j)));
            const __m256 scv = _mm256_set1_ps(tl->sc);
            const __m256i hv = _mm256_set1_epi32((int)tl->hdr);
            const __m256 sc_lo = _mm256_mul_ps(scv, _mm256_permutevar8x32_ps(gv, _mm256_and_si256(_mm256_srlv_epi32(hv, sh_lo), three)));
            const __m256 sc_hi = _mm256_mul_ps(scv, _mm256_permutevar8x32_ps(gv, _mm256_and_si256(_mm256_srlv_epi32(hv, sh_hi), three)));
            const __m256i sv = _mm256_loadu_si256((const __m256i*)tl->usgn);
            #pragma GCC unroll 8
            for (int g = 0; g < 8; g++) {
                AVX2_STEP_V_NEG(g, tl->c, sv, v, neg);
                __m256i fidx = _mm256_and_si256(_mm256_xor_si256(v, neg), m4);
                __m256i w_mag = _mm256_shuffle_epi8(T0, fidx);
                AVX2_ACCUM(g, w_mag, neg, a2, sc_lo, sc_hi, acc);
            }
        }
        for (int g = 0; g < 8; g++) { y[b * 16 + 2 * g] = hsum4(_mm256_castps256_ps128(acc[g])); y[b * 16 + 2 * g + 1] = hsum4(_mm256_extractf128_ps(acc[g], 1)); }
    }
}

static void kern_k25_gosset(const void* T_raw, const uint8_t* tab, const float* gains, const int8_t* a, float* y) {
    const tile_k25_t* T = (const tile_k25_t*)T_raw;
    const __m256i T0 = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)tab));
    const __m256i T1 = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)(tab + 16)));
    AVX2_PROLOGUE();
    for (int b = 0; b < M_ROWS / 16; b++) {
        __m256 acc[8]; for (int g = 0; g < 8; g++) acc[g] = _mm256_setzero_ps();
        for (int j = 0; j < NTILE; j++) {
            const tile_k25_t* tl = &T[b * NTILE + j];
            const __m256i a2 = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)(a + 16 * j)));
            const __m256 scv = _mm256_set1_ps(tl->sc);
            const __m256i hv = _mm256_set1_epi32((int)tl->hdr);
            const __m256 sc_lo = _mm256_mul_ps(scv, _mm256_permutevar8x32_ps(gv, _mm256_and_si256(_mm256_srlv_epi32(hv, sh_lo), three)));
            const __m256 sc_hi = _mm256_mul_ps(scv, _mm256_permutevar8x32_ps(gv, _mm256_and_si256(_mm256_srlv_epi32(hv, sh_hi), three)));
            const __m256i sv = _mm256_loadu_si256((const __m256i*)tl->usgn);
            const __m128i h16 = _mm_loadu_si128((const __m128i*)tl->ham);
            const __m256i nibs = _mm256_and_si256(_mm256_set_m128i(_mm_srli_epi16(h16, 4), h16), m4);
            const __m256i ub0v = _mm256_shuffle_epi8(H_LUT, nibs);
            #pragma GCC unroll 8
            for (int g = 0; g < 8; g++) {
                AVX2_STEP_V_NEG(g, tl->c, sv, v, neg);
                __m256i u_low = _mm256_and_si256(g ? _mm256_srli_epi16(ub0v, g) : ub0v, one);
                __m256i fidx = _mm256_xor_si256(_mm256_or_si256(v, u_low), neg);
                __m256i lo = _mm256_and_si256(fidx, m4);
                __m256i w_mag = _mm256_blendv_epi8(_mm256_shuffle_epi8(T0, lo), _mm256_shuffle_epi8(T1, lo), _mm256_slli_epi16(fidx, 3));
                AVX2_ACCUM(g, w_mag, neg, a2, sc_lo, sc_hi, acc);
            }
        }
        for (int g = 0; g < 8; g++) { y[b * 16 + 2 * g] = hsum4(_mm256_castps256_ps128(acc[g])); y[b * 16 + 2 * g + 1] = hsum4(_mm256_extractf128_ps(acc[g], 1)); }
    }
}

static void kern_k3(const void* T_raw, const uint8_t* tab, const float* gains, const int8_t* a, float* y) {
    const tile_k3_t* T = (const tile_k3_t*)T_raw;
    const __m256i T0 = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)tab));
    const __m256i T1 = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)(tab + 16)));
    AVX2_PROLOGUE(); (void)H_LUT;
    for (int b = 0; b < M_ROWS / 16; b++) {
        __m256 acc[8]; for (int g = 0; g < 8; g++) acc[g] = _mm256_setzero_ps();
        for (int j = 0; j < NTILE; j++) {
            const tile_k3_t* tl = &T[b * NTILE + j];
            const __m256i a2 = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)(a + 16 * j)));
            const __m256 scv = _mm256_set1_ps(tl->sc);
            const __m256i hv = _mm256_set1_epi32((int)tl->hdr);
            const __m256 sc_lo = _mm256_mul_ps(scv, _mm256_permutevar8x32_ps(gv, _mm256_and_si256(_mm256_srlv_epi32(hv, sh_lo), three)));
            const __m256 sc_hi = _mm256_mul_ps(scv, _mm256_permutevar8x32_ps(gv, _mm256_and_si256(_mm256_srlv_epi32(hv, sh_hi), three)));
            const __m256i sv = _mm256_loadu_si256((const __m256i*)tl->usgn);
            const __m256i ub0v = _mm256_loadu_si256((const __m256i*)tl->ub0);
            #pragma GCC unroll 8
            for (int g = 0; g < 8; g++) {
                AVX2_STEP_V_NEG(g, tl->c, sv, v, neg);
                __m256i u_low = _mm256_and_si256(g ? _mm256_srli_epi16(ub0v, g) : ub0v, one);
                __m256i fidx = _mm256_xor_si256(_mm256_or_si256(v, u_low), neg);
                __m256i lo = _mm256_and_si256(fidx, m4);
                __m256i w_mag = _mm256_blendv_epi8(_mm256_shuffle_epi8(T0, lo), _mm256_shuffle_epi8(T1, lo), _mm256_slli_epi16(fidx, 3));
                AVX2_ACCUM(g, w_mag, neg, a2, sc_lo, sc_hi, acc);
            }
        }
        for (int g = 0; g < 8; g++) { y[b * 16 + 2 * g] = hsum4(_mm256_castps256_ps128(acc[g])); y[b * 16 + 2 * g + 1] = hsum4(_mm256_extractf128_ps(acc[g], 1)); }
    }
}

static void kern_k35_gosset(const void* T_raw, const uint8_t* tab, const float* gains, const int8_t* a, float* y) {
    const tile_k35_t* T = (const tile_k35_t*)T_raw;
    const __m256i TA0 = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)tab));
    const __m256i TB0 = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)(tab + 16)));
    AVX2_PROLOGUE();
    for (int b = 0; b < M_ROWS / 16; b++) {
        __m256 acc[8]; for (int g = 0; g < 8; g++) acc[g] = _mm256_setzero_ps();
        for (int j = 0; j < NTILE; j++) {
            const tile_k35_t* tl = &T[b * NTILE + j];
            const __m256i a2 = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)(a + 16 * j)));
            const __m256 scv = _mm256_set1_ps(tl->sc);
            const __m256i hv = _mm256_set1_epi32((int)tl->hdr);
            const __m256 sc_lo = _mm256_mul_ps(scv, _mm256_permutevar8x32_ps(gv, _mm256_and_si256(_mm256_srlv_epi32(hv, sh_lo), three)));
            const __m256 sc_hi = _mm256_mul_ps(scv, _mm256_permutevar8x32_ps(gv, _mm256_and_si256(_mm256_srlv_epi32(hv, sh_hi), three)));
            const __m256i sv = _mm256_loadu_si256((const __m256i*)tl->usgn);
            const __m256i ub1v = _mm256_loadu_si256((const __m256i*)tl->ub1);
            const __m128i h16 = _mm_loadu_si128((const __m128i*)tl->ham);
            const __m256i nibs = _mm256_and_si256(_mm256_set_m128i(_mm_srli_epi16(h16, 4), h16), m4);
            const __m256i ub0v = _mm256_shuffle_epi8(H_LUT, nibs);
            #pragma GCC unroll 8
            for (int g = 0; g < 8; g++) {
                AVX2_STEP_V_NEG(g, tl->c, sv, v, neg);
                __m256i b0 = _mm256_and_si256(g ? _mm256_srli_epi16(ub0v, g) : ub0v, one);
                __m256i b1 = _mm256_and_si256(g == 0 ? _mm256_slli_epi16(ub1v, 1) : (g == 1 ? ub1v : _mm256_srli_epi16(ub1v, g - 1)), _mm256_set1_epi8(2));
                __m256i fv = _mm256_xor_si256(_mm256_or_si256(v, _mm256_or_si256(b0, b1)), neg);
                __m256i wA = _mm256_shuffle_epi8(TA0, _mm256_and_si256(fv, m4));
                __m256i idxB = _mm256_or_si256(_mm256_and_si256(fv, _mm256_set1_epi8(3)), _mm256_and_si256(_mm256_srli_epi16(fv, 2), _mm256_set1_epi8(0x0c)));
                __m256i w_mag = _mm256_adds_epu8(wA, _mm256_shuffle_epi8(TB0, idxB));
                AVX2_ACCUM(g, w_mag, neg, a2, sc_lo, sc_hi, acc);
            }
        }
        for (int g = 0; g < 8; g++) { y[b * 16 + 2 * g] = hsum4(_mm256_castps256_ps128(acc[g])); y[b * 16 + 2 * g + 1] = hsum4(_mm256_extractf128_ps(acc[g], 1)); }
    }
}

static void kern_k4(const void* T_raw, const uint8_t* tab, const float* gains, const int8_t* a, float* y) {
    const tile_k4_t* T = (const tile_k4_t*)T_raw;
    const __m256i TA0 = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)tab));
    const __m256i TB0 = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)(tab + 16)));
    AVX2_PROLOGUE(); (void)H_LUT;
    for (int b = 0; b < M_ROWS / 16; b++) {
        __m256 acc[8]; for (int g = 0; g < 8; g++) acc[g] = _mm256_setzero_ps();
        for (int j = 0; j < NTILE; j++) {
            const tile_k4_t* tl = &T[b * NTILE + j];
            const __m256i a2 = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)(a + 16 * j)));
            const __m256 scv = _mm256_set1_ps(tl->sc);
            const __m256i hv = _mm256_set1_epi32((int)tl->hdr);
            const __m256 sc_lo = _mm256_mul_ps(scv, _mm256_permutevar8x32_ps(gv, _mm256_and_si256(_mm256_srlv_epi32(hv, sh_lo), three)));
            const __m256 sc_hi = _mm256_mul_ps(scv, _mm256_permutevar8x32_ps(gv, _mm256_and_si256(_mm256_srlv_epi32(hv, sh_hi), three)));
            const __m256i sv = _mm256_loadu_si256((const __m256i*)tl->usgn);
            const __m256i u_lo0 = _mm256_loadu_si256((const __m256i*)(tl->u2));
            const __m256i u_lo1 = _mm256_loadu_si256((const __m256i*)(tl->u2 + 32));
            #pragma GCC unroll 8
            for (int g = 0; g < 8; g++) {
                AVX2_STEP_V_NEG(g, tl->c, sv, v, neg);
                int sh = 2 * (g & 3);
                __m256i s0 = (g < 4) ? u_lo0 : u_lo1;
                __m256i u01 = _mm256_and_si256(sh ? _mm256_srli_epi16(s0, sh) : s0, _mm256_set1_epi8(3));
                __m256i fv = _mm256_xor_si256(_mm256_or_si256(v, u01), neg);
                __m256i wA = _mm256_shuffle_epi8(TA0, _mm256_and_si256(fv, m4));
                __m256i idxB = _mm256_or_si256(_mm256_and_si256(fv, _mm256_set1_epi8(3)), _mm256_and_si256(_mm256_srli_epi16(fv, 2), _mm256_set1_epi8(0x0c)));
                __m256i w_mag = _mm256_adds_epu8(wA, _mm256_shuffle_epi8(TB0, idxB));
                AVX2_ACCUM(g, w_mag, neg, a2, sc_lo, sc_hi, acc);
            }
        }
        for (int g = 0; g < 8; g++) { y[b * 16 + 2 * g] = hsum4(_mm256_castps256_ps128(acc[g])); y[b * 16 + 2 * g + 1] = hsum4(_mm256_extractf128_ps(acc[g], 1)); }
    }
}

static void kern_k5_at2z32(const void* T_raw, const uint8_t* tab, const float* gains, const int8_t* a, float* y) {
    const tile_k5_t* T = (const tile_k5_t*)T_raw;
    const __m256i TA0 = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)tab));
    const __m256i TA1 = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)(tab + 16)));
    const __m256i TB0 = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)(tab + 32)));
    const __m256i TB1 = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)(tab + 48)));
    AVX2_PROLOGUE(); (void)H_LUT;
    for (int b = 0; b < M_ROWS / 16; b++) {
        __m256 acc[8]; for (int g = 0; g < 8; g++) acc[g] = _mm256_setzero_ps();
        for (int j = 0; j < NTILE; j++) {
            const tile_k5_t* tl = &T[b * NTILE + j];
            const __m256i a2 = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)(a + 16 * j)));
            const __m256 scv = _mm256_set1_ps(tl->sc);
            const __m256i hv = _mm256_set1_epi32((int)tl->hdr);
            const __m256 sc_lo = _mm256_mul_ps(scv, _mm256_permutevar8x32_ps(gv, _mm256_and_si256(_mm256_srlv_epi32(hv, sh_lo), three)));
            const __m256 sc_hi = _mm256_mul_ps(scv, _mm256_permutevar8x32_ps(gv, _mm256_and_si256(_mm256_srlv_epi32(hv, sh_hi), three)));
            const __m256i sv = _mm256_loadu_si256((const __m256i*)tl->usgn);
            const __m256i u_lo0 = _mm256_loadu_si256((const __m256i*)(tl->u2));
            const __m256i u_lo1 = _mm256_loadu_si256((const __m256i*)(tl->u2 + 32));
            const __m256i ub0v = _mm256_loadu_si256((const __m256i*)tl->ub0);
            #pragma GCC unroll 8
            for (int g = 0; g < 8; g++) {
                AVX2_STEP_V_NEG(g, tl->c, sv, v, neg);
                int sh = 2 * (g & 3);
                __m256i s0 = (g < 4) ? u_lo0 : u_lo1;
                __m256i u01 = _mm256_and_si256(sh ? _mm256_srli_epi16(s0, sh) : s0, _mm256_set1_epi8(3));
                __m256i b2 = g >= 2 ? _mm256_srli_epi16(ub0v, g - 2) : _mm256_slli_epi16(ub0v, 2 - g);
                __m256i fv = _mm256_xor_si256(_mm256_or_si256(v, _mm256_or_si256(u01, _mm256_and_si256(b2, _mm256_set1_epi8(4)))), neg);
                __m256i loA = _mm256_and_si256(fv, m4);
                __m256i wA = _mm256_blendv_epi8(_mm256_shuffle_epi8(TA0, loA), _mm256_shuffle_epi8(TA1, loA), _mm256_slli_epi16(fv, 3));
                __m256i idxB = _mm256_or_si256(_mm256_and_si256(fv, _mm256_set1_epi8(7)), _mm256_and_si256(_mm256_srli_epi16(fv, 2), _mm256_set1_epi8(0x18)));
                __m256i loB = _mm256_and_si256(idxB, m4);
                __m256i wB = _mm256_blendv_epi8(_mm256_shuffle_epi8(TB0, loB), _mm256_shuffle_epi8(TB1, loB), _mm256_slli_epi16(idxB, 3));
                __m256i w_mag = _mm256_adds_epu8(wA, wB);
                AVX2_ACCUM(g, w_mag, neg, a2, sc_lo, sc_hi, acc);
            }
        }
        for (int g = 0; g < 8; g++) { y[b * 16 + 2 * g] = hsum4(_mm256_castps256_ps128(acc[g])); y[b * 16 + 2 * g + 1] = hsum4(_mm256_extractf128_ps(acc[g], 1)); }
    }
}

static void kern_k6_at2z32(const void* T_raw, const uint8_t* tab, const float* gains, const int8_t* a, float* y) {
    const tile_k6_t* T = (const tile_k6_t*)T_raw;
    const __m256i TA0 = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)tab));
    const __m256i TA1 = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)(tab + 16)));
    const __m256i TB0 = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)(tab + 32)));
    const __m256i TB1 = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)(tab + 48)));
    AVX2_PROLOGUE(); (void)H_LUT;
    for (int b = 0; b < M_ROWS / 16; b++) {
        __m256 acc[8]; for (int g = 0; g < 8; g++) acc[g] = _mm256_setzero_ps();
        for (int j = 0; j < NTILE; j++) {
            const tile_k6_t* tl = &T[b * NTILE + j];
            const __m256i a2 = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)(a + 16 * j)));
            const __m256 scv = _mm256_set1_ps(tl->sc);
            const __m256i hv = _mm256_set1_epi32((int)tl->hdr);
            const __m256 sc_lo = _mm256_mul_ps(scv, _mm256_permutevar8x32_ps(gv, _mm256_and_si256(_mm256_srlv_epi32(hv, sh_lo), three)));
            const __m256 sc_hi = _mm256_mul_ps(scv, _mm256_permutevar8x32_ps(gv, _mm256_and_si256(_mm256_srlv_epi32(hv, sh_hi), three)));
            const __m256i sv = _mm256_loadu_si256((const __m256i*)tl->usgn);
            const __m256i u_lo0 = _mm256_loadu_si256((const __m256i*)(tl->u2));
            const __m256i u_lo1 = _mm256_loadu_si256((const __m256i*)(tl->u2 + 32));
            const __m256i u_hi0 = _mm256_loadu_si256((const __m256i*)(tl->u2h));
            const __m256i u_hi1 = _mm256_loadu_si256((const __m256i*)(tl->u2h + 32));
            #pragma GCC unroll 8
            for (int g = 0; g < 8; g++) {
                AVX2_STEP_V_NEG(g, tl->c, sv, v, neg);
                int sh = 2 * (g & 3);
                __m256i s0 = (g < 4) ? u_lo0 : u_lo1;
                __m256i s1 = (g < 4) ? u_hi0 : u_hi1;
                __m256i u01 = _mm256_and_si256(sh ? _mm256_srli_epi16(s0, sh) : s0, _mm256_set1_epi8(3));
                __m256i u23 = _mm256_and_si256(_mm256_slli_epi16(sh ? _mm256_srli_epi16(s1, sh) : s1, 2), _mm256_set1_epi8(0x0c));
                __m256i fv = _mm256_xor_si256(_mm256_or_si256(v, _mm256_or_si256(u01, u23)), neg);
                __m256i loA = _mm256_and_si256(fv, m4);
                __m256i wA = _mm256_blendv_epi8(_mm256_shuffle_epi8(TA0, loA), _mm256_shuffle_epi8(TA1, loA), _mm256_slli_epi16(fv, 3));
                __m256i idxB = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(fv, 2), _mm256_set1_epi8(3)),
                                               _mm256_and_si256(_mm256_srli_epi16(fv, 3), _mm256_set1_epi8(0x1c)));
                __m256i loB = _mm256_and_si256(idxB, m4);
                __m256i wB = _mm256_blendv_epi8(_mm256_shuffle_epi8(TB0, loB), _mm256_shuffle_epi8(TB1, loB), _mm256_slli_epi16(idxB, 3));
                __m256i w_mag = _mm256_adds_epu8(wA, wB);
                AVX2_ACCUM(g, w_mag, neg, a2, sc_lo, sc_hi, acc);
            }
        }
        for (int g = 0; g < 8; g++) { y[b * 16 + 2 * g] = hsum4(_mm256_castps256_ps128(acc[g])); y[b * 16 + 2 * g + 1] = hsum4(_mm256_extractf128_ps(acc[g], 1)); }
    }
}

typedef void (*avx_fn_t)(const void*, const uint8_t*, const float*, const int8_t*, float*);

static void bench_fmt(const char* name, avx_fn_t fn, size_t stride, int bit_base, int fmt, double file_bpw,
                      const uint8_t* tab, const float* gains, const int8_t* a) {
    const int ntiles = (M_ROWS / 16) * NTILE;
    uint8_t* buf = aligned_alloc(64, stride * ntiles);
    srand(100 + fmt);
    for (size_t i = 0; i < stride * ntiles; i++) buf[i] = (uint8_t)(rand() & 0xff);
    for (int i = 0; i < ntiles; i++) *(float*)(buf + i * stride + stride - 4) = 0.05f + 0.01f * (i & 7);
    float *y = malloc(4 * M_ROWS), *yr = malloc(4 * M_ROWS);

    build_luts_at(bit_base);
    scalar_ref(buf, stride, tab, gains, a, yr, fmt);
    fn(buf, tab, gains, a, y);
    double err = 0, nrm = 0;
    for (int i = 0; i < M_ROWS; i++) { err += fabs(y[i] - yr[i]); nrm += fabs(yr[i]); }
    double rel = err / (nrm + 1e-12);

    const int iters = 400;
    double t0 = now_s();
    for (int it = 0; it < iters; it++) { fn(buf, tab, gains, a, y); __asm__ volatile("" ::: "memory"); }
    double dt = (now_s() - t0) / iters;
    double gw = (double)M_ROWS * K_COLS / dt * 1e-9;
    printf("%-38s | %3zuB | err %.1e | %6.2f G weights/s | %5.1f GB/s @ %.3f bpw\n",
           name, stride, rel, gw, gw * file_bpw / 8.0, file_bpw);
    if (rel > 1e-5) { fprintf(stderr, "MISMATCH in %s: rel=%.3e\n", name, rel); exit(1); }
    free(buf); free(y); free(yr);
}

int main(void) {
    int8_t a[K_COLS]; uint8_t tab_at[64], tab_at2[64];
    srand(7);
    for (int i = 0; i < K_COLS; i++) a[i] = (int8_t)((rand() % 255) - 127);
    // QMAX_U8 = 128 guarantees 2 * 128 * 127 = 32512 <= 32767 (zero int16 saturation in `_mm256_maddubs_epi16`)
    for (int i = 0; i < 64; i++) {
        tab_at[i]  = (uint8_t)((rand() % 128) + 1);
        tab_at2[i] = (uint8_t)((rand() % 64) + 1);
    }
    const float gains[4] = {0.75f, 0.92f, 1.08f, 1.31f};

    printf("=== VTQ v2 Compact Unified Bitplane CPU AVX2 Benchmark ===\n");
    bench_fmt("VTQ2 K=2.0 AT-32   nl=16 (1x pshufb)",   kern_k2,         sizeof(tile_k2_t),  0, 20, 2.090, tab_at,  gains, a);
    bench_fmt("VTQ2 K=2.5 Gosset84 nl=16 (AT-64)",      kern_k25_gosset, sizeof(tile_k25_t), 1, 25, 2.590, tab_at,  gains, a);
    bench_fmt("VTQ2 K=3.0 AT-64   nl=16 (2x pshufb)",   kern_k3,         sizeof(tile_k3_t),  1, 30, 3.152, tab_at,  gains, a);
    bench_fmt("VTQ2 K=3.5 Gosset84 nl=16 (AT2 16+16)",  kern_k35_gosset, sizeof(tile_k35_t), 2, 35, 3.652, tab_at2, gains, a);
    bench_fmt("VTQ2 K=4.0 AT2     nl=16 (16B+16B add)", kern_k4,         sizeof(tile_k4_t),  2, 40, 4.152, tab_at2, gains, a);
    bench_fmt("VTQ2 K=5.0 AT2Z_32 nl=16 (32B+32B add)", kern_k5_at2z32,  sizeof(tile_k5_t),  3, 50, 5.152, tab_at2, gains, a);
    bench_fmt("VTQ2 K=6.0 AT2Z_32 nl=16 (32B+32B add)", kern_k6_at2z32,  sizeof(tile_k6_t),  4, 60, 6.152, tab_at2, gains, a);
    return 0;
}
