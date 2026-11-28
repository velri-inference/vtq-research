// VTQ Super-Tile (`16 rows x 256 cols = 4,096 weights`) Galois-1024 (`nu = 10`) Trellis CPU AVX2 Benchmark
// (`gcc -O3 -mavx2 -mfma -mno-avx512f src/cpu/vtq_avx2_bench.c -lm -o vtq_avx2_bench`)
//
// Core VTQ CPU AVX2 hardware innovations over VTQ v1 (TCQ) and VTQ v2:
//   1. `16x256` (`4,096`-weight) Natural-Column Super-Tile + 1024-State (`nu = 10`) Tail-Biting Trellis:
//      - `16 KB` Native Little-Endian `uint16_t` `GF(2)` L1 LUT (`LH9[512][16]` `8 KB` + `L0[256][16]` `4 KB` + `L1[256][16]` `4 KB`):
//        storing `c[g][r][0] = cols 8..15` and `c[g][r][1] = cols 0..7` lets x86-64 load all `9` history bits in a single
//        `movzwl + and $0x1ff` (`LH9[r0 & 0x1FF]`) with ZERO byte-swaps or bit-shifts across all 16 columns.
//   2. Zero-Load Even/Odd `vpshufb` `int16` Gain Fusion into `vpmaddwd`:
//      - Expands `uint8 gain_u8[2^h]` once per 16-col subblock `g` into `[even rows | odd rows]`, broadcasts
//        `[int16(g_{2p}) x8 | int16(g_{2p+1}) x8]` via `_mm256_shuffle_epi8(geo, gsh)` (bit-7 auto-zeros high byte),
//        and fuses subblock gain multiplication directly into `_mm256_madd_epi16(prd, gp)` — cutting `vcvtdq2ps + vfmadd231ps`
//        by `16x` vs VTQ v2 (once per `256` weights instead of once per `16` weights!).
//   3. Exact `uint8` / `int8` (`qmax = 127`) Codebooks + Hardware Bit-7 `vpshufb` Zeroing (`0` `vpblendvb`!):
//      - `16`-entry (`K=2.0, K=3.0 nl=8`): `1x vpshufb`
//      - `32`-entry (`K=1.5, 2.5, 3.0 nl=16, 3.5, 4.0, 5.0 nl=4`): `2x vpshufb + vpor` via bit-7 (`0x80`) zeroing
//      - `64`-entry (`K=5.0 nl=8, K=6.0 nl=4`): `1-XOR / 4-ADD` (`0x70 / 0xD0`) `4x vpshufb` with ZERO `vpblendvb`
//   4. Coalesced Packed `u2[16][2][32]` (`K=4, 5`) and `u4[16][4][32]` (`K=6`) + Off-Register `acc_f` Accumulators:
//      - Eliminates YMM register spills across all bit-rates `1.5..6.0 bpw` while remaining 32-byte warp-coalesced on GPU.
#include <assert.h>
#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// Matches `(M_ROWS = 256, K_COLS = 2048)` in `src/cpu/vtq2_avx2_bench.c` for exact apples-to-apples L2 comparison:
#ifndef M_ROWS
#define M_ROWS 256
#endif
#ifndef K_COLS
#define K_COLS 2048
#endif
#define NSB (K_COLS / 256)

// 16-byte naturally aligned `16x256` (`4,096`-weight) super-tile structs (zero padding, shared with GPU & `.vtq`):
typedef struct {
    uint8_t  c[16][16][2];  //  512 B (1.00000 bpw, LE uint16 pair, MSB-first per byte: c[g][r][0]=cols 8..15, c[g][r][1]=cols 0..7)
    uint8_t  ham[16][16];   //  256 B (0.50000 bpw, Gosset [8,4,4]: low nib=cols 0..7, high nib=cols 8..15)
    uint16_t hdr[16][1];    //   32 B (0.06250 bpw, h=1 gain bit per 16 weights: bit r = row r)
    uint8_t  u_sb[16];      //   16 B (0.03125 bpw, uint8 linear superblock row scale) ->  816 B = 1.59375 bpw
} vtq_st16_k15_t;

typedef struct {
    uint8_t  c[16][16][2];  //  512 B
    uint8_t  usgn[16][32];  //  512 B ([0..15]=even rows 2*p bit p, [16..31]=odd rows 2*p+1 bit p)
    uint16_t hdr[16][1];    //   32 B (h=1)
    uint8_t  u_sb[16];      //   16 B -> 1072 B = 2.09375 bpw
} vtq_st16_k20_t;

typedef struct {
    uint8_t  c[16][16][2];  //  512 B
    uint8_t  usgn[16][32];  //  512 B
    uint8_t  ham[16][16];   //  256 B
    uint16_t hdr[16][1];    //   32 B (h=1)
    uint8_t  u_sb[16];      //   16 B -> 1328 B = 2.59375 bpw
} vtq_st16_k25_t;

typedef struct {
    uint8_t  c[16][16][2];  //  512 B
    uint8_t  usgn[16][32];  //  512 B
    uint8_t  ub0[16][32];   //  512 B
    uint16_t hdr[16][2];    //   64 B (h=2)
    uint8_t  u_sb[16];      //   16 B -> 1616 B = 3.15625 bpw
} vtq_st16_k30_t;

typedef struct {
    uint8_t  c[16][16][2];  //  512 B
    uint8_t  usgn[16][32];  //  512 B
    uint8_t  ub1[16][32];   //  512 B
    uint8_t  ham[16][16];   //  256 B
    uint16_t hdr[16][2];    //   64 B (h=2)
    uint8_t  u_sb[16];      //   16 B -> 1872 B = 3.65625 bpw
} vtq_st16_k35_t;

typedef struct {
    uint8_t  c[16][16][2];  //  512 B
    uint8_t  usgn[16][32];  //  512 B
    uint8_t  u2[16][2][32]; // 1024 B (u2[g][hp][16*odd + j] bits 2*q..2*q+1 for p = 4*hp + q)
    uint16_t hdr[16][3];    //   96 B (h=3)
    uint8_t  u_sb[16];      //   16 B -> 2160 B = 4.21875 bpw
} vtq_st16_k40_t;

typedef struct {
    uint8_t  c[16][16][2];  //  512 B
    uint8_t  usgn[16][32];  //  512 B
    uint8_t  u2[16][2][32]; // 1024 B
    uint8_t  ub2[16][32];   //  512 B
    uint16_t hdr[16][3];    //   96 B (h=3)
    uint8_t  u_sb[16];      //   16 B -> 2672 B = 5.21875 bpw
} vtq_st16_k50_t;

typedef struct {
    uint8_t  c[16][16][2];  //  512 B
    uint8_t  usgn[16][32];  //  512 B
    uint8_t  u4[16][4][32]; // 2048 B (u4[g][p >> 1][16*odd + j] bits 4*(p&1)..4*(p&1)+3)
    uint16_t hdr[16][3];    //   96 B (h=3)
    uint8_t  u_sb[16];      //   16 B -> 3184 B = 6.21875 bpw
} vtq_st16_k60_t;

_Static_assert(sizeof(vtq_st16_k15_t) ==  816, "st16_k15 size");
_Static_assert(sizeof(vtq_st16_k20_t) == 1072, "st16_k20 size");
_Static_assert(sizeof(vtq_st16_k25_t) == 1328, "st16_k25 size");
_Static_assert(sizeof(vtq_st16_k30_t) == 1616, "st16_k30 size");
_Static_assert(sizeof(vtq_st16_k35_t) == 1872, "st16_k35 size");
_Static_assert(sizeof(vtq_st16_k40_t) == 2160, "st16_k40 size");
_Static_assert(sizeof(vtq_st16_k50_t) == 2672, "st16_k50 size");
_Static_assert(sizeof(vtq_st16_k60_t) == 3184, "st16_k60 size");

static const uint8_t HAM_CW8[16] __attribute__((aligned(16))) = {
    0x00, 0xff, 0xf0, 0x0f, 0xcc, 0x33, 0x3c, 0xc3,
    0xaa, 0x55, 0x5a, 0xa5, 0x66, 0x99, 0x96, 0x69
};

// 16 KB Galois-1024 LUTs + 4 KB Gosset byte-expansion LUT = 20 KB total in L1d (48 KB on Zen 5 / Raptor Lake)
static uint8_t  LH9[512][16]    __attribute__((aligned(64)));
static uint8_t  L0[256][16]     __attribute__((aligned(64)));
static uint8_t  L1[256][16]     __attribute__((aligned(64)));
static uint8_t  HAM_LUT[256][16] __attribute__((aligned(64)));

// Optimal 1024-state (`nu = 10`, 9 history bits) Ungerboeck GF(2) subset label map (`nl in {4, 8, 16}`):
static inline int p10_gf2(int x) { return __builtin_popcount((unsigned int)(x & 0x3FF)) & 1; }
static inline int galois_lb(int st, int nl) {
    if (nl == 16) {
        return (st & 1) | (p10_gf2(st & 0x2CE) << 1) | (p10_gf2(st & 0x050) << 2) | (p10_gf2(st & 0x3B6) << 3);
    }
    if (nl == 8) {
        return (st & 1) | (p10_gf2(st & 0x2CE) << 1) | (p10_gf2(st & 0x39C) << 2);
    }
    return (st & 1) | (p10_gf2(st & 0x2CE) << 1);
}

static inline uint8_t pack_lb_bits(int lb, int lbits, const int pos[4]) {
    uint8_t out = 0;
    for (int i = 0; i < lbits; i++) {
        if ((lb >> i) & 1) out |= (uint8_t)(1u << pos[i]);
    }
    return out;
}

// Builds `LH9 + L0 + L1` for `(ub, nl, lut_mode)` and `HAM_LUT` placing the Gosset bit at `ham_bit_pos`:
//   - `lut_mode == 0`: `16`-entry or `32`-entry bit-7 zeroing (`lb` bits at `ub + i` if `<= 3`, else bit `7`).
//   - `lut_mode == 2`: `AT2 (16B+16B)` `K=4, nl=16` (`lb` bits at `{2, 3, 4, 5}`, bit `7 == 0`).
//   - `lut_mode == 3`: `AT2Z (32B+32B)` `K=5, nl=16` (`lb` bits at `{3, 7, 4, 6}` -> bit-7 zeroing for both `TA` and `TB`!).
//   - `lut_mode == 4`: `AT2Z (32B+32B)` `K=6, nl=16` (`lb` bits at `{7, 4, 5, 6}` -> bit-7 zeroing for both `TA` and `TB`!).
static void build_vtq_luts(int ub, int nl, int lut_mode, int ham_bit_pos) {
    int lbits = (nl == 4) ? 2 : ((nl == 8) ? 3 : 4);
    int pos[4] = {0, 0, 0, 0};
    if (lut_mode == 2) {
        pos[0] = 2; pos[1] = 3; pos[2] = 6; pos[3] = 7;
    } else if (lut_mode == 3) {
        pos[0] = 3; pos[1] = 7; pos[2] = 4; pos[3] = 6;
    } else if (lut_mode == 4) {
        pos[0] = 7; pos[1] = 4; pos[2] = 5; pos[3] = 6;
    } else {
        for (int i = 0; i < lbits; i++) {
            pos[i] = (lut_mode == 1) ? (ub + i) : ((ub + i <= 3) ? (ub + i) : 7);
        }
    }
    memset(LH9, 0, sizeof(LH9));
    for (int h9 = 0; h9 < 512; h9++) {
        uint32_t w_h = ((uint32_t)h9) << 16;
        for (int j = 0; j < 9; j++) {
            int st_h = (int)((w_h >> (15 - j)) & 0x3FFu);
            LH9[h9][j] = pack_lb_bits(galois_lb(st_h, nl), lbits, pos);
        }
    }
    for (int byte = 0; byte < 256; byte++) {
        uint32_t w0 = ((uint32_t)byte) << 8;
        uint32_t w1 = (uint32_t)byte;
        for (int j = 0; j < 16; j++) {
            int sh = 15 - j;
            L0[byte][j] = pack_lb_bits(galois_lb((int)((w0 >> sh) & 0x3FFu), nl), lbits, pos);
            L1[byte][j] = pack_lb_bits(galois_lb((int)((w1 >> sh) & 0x3FFu), nl), lbits, pos);
        }
    }
    if (ham_bit_pos >= 0) {
        uint8_t mask = (lut_mode == 2 && ham_bit_pos == 0) ? 0x11u : (uint8_t)(1u << ham_bit_pos);
        for (int hb = 0; hb < 256; hb++) {
            uint8_t cw0 = HAM_CW8[hb & 0x0F];
            uint8_t cw1 = HAM_CW8[hb >> 4];
            for (int j = 0; j < 8; j++) {
                HAM_LUT[hb][j]     = ((cw0 >> (7 - j)) & 1) ? mask : 0;
                HAM_LUT[hb][8 + j] = ((cw1 >> (7 - j)) & 1) ? mask : 0;
            }
        }
    }
}

// Load-time codebook bit-permutation from canonical `(u_mag * nl) | lb` to AVX2 `(lb_bits_hi | u_mag_lo)`:
// Because bit complement commutes with any bit permutation (`perm(x ^ mask) == perm(x) ^ mask`),
// XORing with `neg` in AVX2 remains 100% bit-exact for every signed entry!
static void permute_tab_for_avx2(const uint8_t* tab_canon, uint8_t* tab_avx, int ub, int nl) {
    int lbits = (nl == 4) ? 2 : ((nl == 8) ? 3 : 4);
    int u_cnt = 1 << ub;
    for (int u_mag = 0; u_mag < u_cnt; u_mag++) {
        for (int lb = 0; lb < nl; lb++) {
            int idx_canon = (u_mag << lbits) | lb;
            int idx_avx   = (lb << ub) | u_mag;
            tab_avx[idx_avx] = tab_canon[idx_canon];
        }
    }
}

// Load-time `TB_u8[32]` bit-permutation for `K=5 AT2Z (32B+32B)` (`(lb3<<4)|(lb2<<3)|u_mag` -> `(lb3<<4)|(u_mag<<1)|lb2`):
static void prepare_at2z_k5_for_avx2(const uint8_t* ta_u8, const uint8_t* tb_u8, uint8_t tab_avx[64]) {
    memcpy(tab_avx, ta_u8, 32);
    for (int lb3 = 0; lb3 < 2; lb3++) {
        for (int lb2 = 0; lb2 < 2; lb2++) {
            for (int um = 0; um < 8; um++) {
                tab_avx[32 + ((lb3 << 4) | (um << 1) | lb2)] = tb_u8[(lb3 << 4) | (lb2 << 3) | um];
            }
        }
    }
}

// Intra-subblock `16x16` 2-row 1024-state (`nu = 10`) Galois state unpacker (zero `cb_prv` pointer!):
static inline __m256i step_v_pair(const uint8_t* cb_cur, int p) {
    const int b0  = 4 * p, b1 = 4 * p + 2;
    const int bp0 = (4 * p + 30) & 31;
    uint16_t r0, r1;
    memcpy(&r0, cb_cur + bp0, 2);
    memcpy(&r1, cb_cur + b0,  2);
    __m256i vh = _mm256_set_m128i(_mm_load_si128((const __m128i*)LH9[r1 & 0x1FF]),
                                  _mm_load_si128((const __m128i*)LH9[r0 & 0x1FF]));
    __m256i v0 = _mm256_set_m128i(_mm_load_si128((const __m128i*)L0[cb_cur[b1 + 1]]),
                                  _mm_load_si128((const __m128i*)L0[cb_cur[b0 + 1]]));
    __m256i v1 = _mm256_set_m128i(_mm_load_si128((const __m128i*)L1[cb_cur[b1 + 0]]),
                                  _mm_load_si128((const __m128i*)L1[cb_cur[b0 + 0]]));
    return _mm256_xor_si256(vh, _mm256_xor_si256(v0, v1));
}

static inline __m256i load_ham_pair(const uint8_t ham_g[16], int r_base) {
    return _mm256_set_m128i(_mm_load_si128((const __m128i*)HAM_LUT[ham_g[r_base + 1]]),
                            _mm_load_si128((const __m128i*)HAM_LUT[ham_g[r_base + 0]]));
}

// Full 16-row even/odd gain expanders (low 128b bytes 0..7 = even rows 2*p, high 128b bytes 0..7 = odd rows 2*p+1)
static const uint8_t C_EO_SEL[32] __attribute__((aligned(32))) = {
    0,0,0,0, 1,1,1,1,  0,0,0,0, 0,0,0,0,
    0,0,0,0, 1,1,1,1,  0,0,0,0, 0,0,0,0
};
static const uint8_t C_EO_MSK[32] __attribute__((aligned(32))) = {
    0x01,0x04,0x10,0x40, 0x01,0x04,0x10,0x40,  0,0,0,0,0,0,0,0,
    0x02,0x08,0x20,0x80, 0x02,0x08,0x20,0x80,  0,0,0,0,0,0,0,0
};
static const uint8_t C_EO8_MSK[32] __attribute__((aligned(32))) = {
    0x01,0x04,0x10,0x40,  0,0,0,0,0,0,0,0,0,0,0,0,
    0x02,0x08,0x20,0x80,  0,0,0,0,0,0,0,0,0,0,0,0
};

static inline __m256i expand_gains_eo_h1(uint16_t h0, __m256i glut2) {
    __m256i bsel = *(const __m256i*)C_EO_SEL;
    __m256i bmsk = *(const __m256i*)C_EO_MSK;
    __m256i w0   = _mm256_shuffle_epi8(_mm256_set1_epi16((short)h0), bsel);
    __m256i gi   = _mm256_and_si256(_mm256_cmpeq_epi8(_mm256_and_si256(w0, bmsk), bmsk), _mm256_set1_epi8(1));
    return _mm256_shuffle_epi8(glut2, gi);
}

static inline __m256i expand_gains_half_h1(uint8_t hb0, __m256i glut2) {
    __m256i bmsk = *(const __m256i*)C_EO8_MSK;
    __m256i gi   = _mm256_and_si256(_mm256_cmpeq_epi8(_mm256_and_si256(_mm256_set1_epi8((char)hb0), bmsk), bmsk), _mm256_set1_epi8(1));
    return _mm256_shuffle_epi8(glut2, gi);
}

static inline __m256i expand_gains_eo_h2(const uint16_t hdr[2], __m256i glut2) {
    __m256i bsel = *(const __m256i*)C_EO_SEL;
    __m256i bmsk = *(const __m256i*)C_EO_MSK;
    __m256i w0   = _mm256_shuffle_epi8(_mm256_set1_epi16((short)hdr[0]), bsel);
    __m256i w1   = _mm256_shuffle_epi8(_mm256_set1_epi16((short)hdr[1]), bsel);
    __m256i gi   = _mm256_or_si256(
        _mm256_and_si256(_mm256_cmpeq_epi8(_mm256_and_si256(w0, bmsk), bmsk), _mm256_set1_epi8(1)),
        _mm256_and_si256(_mm256_cmpeq_epi8(_mm256_and_si256(w1, bmsk), bmsk), _mm256_set1_epi8(2)));
    return _mm256_shuffle_epi8(glut2, gi);
}

static inline __m256i expand_gains_eo_h3(const uint16_t hdr[3], __m256i glut2) {
    __m256i bsel = *(const __m256i*)C_EO_SEL;
    __m256i bmsk = *(const __m256i*)C_EO_MSK;
    __m256i w0   = _mm256_shuffle_epi8(_mm256_set1_epi16((short)hdr[0]), bsel);
    __m256i w1   = _mm256_shuffle_epi8(_mm256_set1_epi16((short)hdr[1]), bsel);
    __m256i w2   = _mm256_shuffle_epi8(_mm256_set1_epi16((short)hdr[2]), bsel);
    __m256i gi   = _mm256_or_si256(
        _mm256_and_si256(_mm256_cmpeq_epi8(_mm256_and_si256(w0, bmsk), bmsk), _mm256_set1_epi8(1)),
        _mm256_or_si256(
            _mm256_and_si256(_mm256_cmpeq_epi8(_mm256_and_si256(w1, bmsk), bmsk), _mm256_set1_epi8(2)),
            _mm256_and_si256(_mm256_cmpeq_epi8(_mm256_and_si256(w2, bmsk), bmsk), _mm256_set1_epi8(4))));
    return _mm256_shuffle_epi8(glut2, gi);
}

static inline float hsum4(__m128 x) {
    __m128 s = _mm_add_ps(x, _mm_movehl_ps(x, x));
    s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
    return _mm_cvtss_f32(s);
}

static inline void flush_usb16(float acc_f[8][8], const __m256i isum[8], const uint8_t u_sb[16]) {
    __m256 uf0 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*)(u_sb + 0))));
    __m256 uf1 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*)(u_sb + 8))));
    for (int p = 0; p < 4; p++) {
        const int q = 2 * p;
        __m256 up = _mm256_permutevar8x32_ps(uf0, _mm256_setr_epi32(q, q, q, q, q + 1, q + 1, q + 1, q + 1));
        _mm256_store_ps(acc_f[p], _mm256_fmadd_ps(_mm256_cvtepi32_ps(isum[p]), up, _mm256_load_ps(acc_f[p])));
    }
    for (int p = 0; p < 4; p++) {
        const int q = 2 * p;
        __m256 up = _mm256_permutevar8x32_ps(uf1, _mm256_setr_epi32(q, q, q, q, q + 1, q + 1, q + 1, q + 1));
        _mm256_store_ps(acc_f[4 + p], _mm256_fmadd_ps(_mm256_cvtepi32_ps(isum[4 + p]), up, _mm256_load_ps(acc_f[4 + p])));
    }
}

static inline void finish_y16(float* y_bg, const float* dr_bg, const float acc_f[8][8]) {
    for (int p = 0; p < 8; p++) {
        __m256 v = _mm256_load_ps(acc_f[p]);
        y_bg[2 * p + 0] = hsum4(_mm256_castps256_ps128(v))   * dr_bg[2 * p + 0];
        y_bg[2 * p + 1] = hsum4(_mm256_extractf128_ps(v, 1)) * dr_bg[2 * p + 1];
    }
}

static inline void flush_half4(float acc_f[4][8], const __m256i isum[4], const uint8_t u_sb_half[8]) {
    __m256 uf = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*)u_sb_half)));
    for (int q = 0; q < 4; q++) {
        const int r = 2 * q;
        __m256 up = _mm256_permutevar8x32_ps(uf, _mm256_setr_epi32(r, r, r, r, r + 1, r + 1, r + 1, r + 1));
        _mm256_store_ps(acc_f[q], _mm256_fmadd_ps(_mm256_cvtepi32_ps(isum[q]), up, _mm256_load_ps(acc_f[q])));
    }
}

static inline void finish_half_y(float* y_half, const float* dr_half, const float acc_f[4][8]) {
    for (int q = 0; q < 4; q++) {
        __m256 v = _mm256_load_ps(acc_f[q]);
        y_half[2 * q + 0] = hsum4(_mm256_castps256_ps128(v))   * dr_half[2 * q + 0];
        y_half[2 * q + 1] = hsum4(_mm256_extractf128_ps(v, 1)) * dr_half[2 * q + 1];
    }
}

// =================================================================================================
// Scalar Shift-Register Reference (verifies canonical `tab_canon`, MSB-first `c`, Gosset, `u2/u4`, `hdr`, `u_sb`)
// =================================================================================================
static void scalar_ref_vtq(const void* ST_raw, size_t stride, const uint8_t* tab_canon,
                            const uint8_t* gain_u8, const int8_t* a, const float* d_row_eff,
                            float* y, int fmt) {
    const int8_t* tab_s8 = (const int8_t*)tab_canon;
    const int h = (fmt <= 25) ? 1 : ((fmt <= 35) ? 2 : 3);
    const int nl = (fmt == 30) ? 8 : ((fmt == 50) ? 4 : 16);

    for (int bg = 0; bg < M_ROWS / 16; bg++) {
        double acc[16] = {0};
        for (int sb = 0; sb < NSB; sb++) {
            const uint8_t* base = (const uint8_t*)ST_raw + (size_t)(bg * NSB + sb) * stride;
            const uint8_t (*c)[16][2] = (const uint8_t (*)[16][2])base;
            const uint16_t* hdr_p = (const uint16_t*)(base + stride - 16 - (size_t)h * 32);
            const uint8_t*  u_sb  = base + stride - 16;

            int32_t isum_row[16] = {0};
            for (int g = 0; g < 16; g++) {
                uint16_t st = (uint16_t)(((c[g][15][1] & 1u) << 8) | c[g][15][0]);
                int32_t sub_sum[16] = {0};
                for (int r = 0; r < 16; r++) {
                    const int p = r >> 1;
                    const int odd = r & 1;
                    for (int j = 0; j < 16; j++) {
                        const uint8_t cb = (j < 8) ? c[g][r][1] : c[g][r][0];
                        const uint8_t cbit = (cb >> (7 - (j & 7))) & 1u;
                        st = (uint16_t)(((st << 1) | cbit) & 0x3FFu);
                        const int lb = galois_lb(st, nl);
                        const int av = a[sb * 256 + 16 * g + j];

                        if (fmt == 15) {
                            const uint8_t (*ham)[16] = (const uint8_t (*)[16])(base + 512);
                            const int nib = (j < 8) ? (ham[g][r] & 0x0F) : (ham[g][r] >> 4);
                            const int u0  = (HAM_CW8[nib] >> (7 - (j & 7))) & 1;
                            sub_sum[r] += (int)tab_s8[(u0 << 4) | lb] * av;
                        } else {
                            const uint8_t (*usgn)[32] = (const uint8_t (*)[32])(base + 512);
                            const int lane = 16 * odd + j;
                            const int neg = (usgn[g][lane] >> p) & 1;
                            int w_mag = 0;
                            if (fmt == 20) {
                                w_mag = tab_canon[lb ^ (neg ? 15 : 0)];
                            } else if (fmt == 25) {
                                const uint8_t (*ham)[16] = (const uint8_t (*)[16])(base + 1024);
                                const int nib = (j < 8) ? (ham[g][r] & 0x0F) : (ham[g][r] >> 4);
                                const int b0  = (HAM_CW8[nib] >> (7 - (j & 7))) & 1;
                                w_mag = tab_canon[((b0 << 4) | lb) ^ (neg ? 31 : 0)];
                            } else if (fmt == 30) {
                                const uint8_t (*ub0)[32] = (const uint8_t (*)[32])(base + 1024);
                                const int b0 = (ub0[g][lane] >> p) & 1;
                                w_mag = tab_canon[((b0 << 3) | lb) ^ (neg ? 15 : 0)];
                            } else if (fmt == 31) {
                                const uint8_t (*ub0)[32] = (const uint8_t (*)[32])(base + 1024);
                                const int b0 = (ub0[g][lane] >> p) & 1;
                                w_mag = tab_canon[((b0 << 4) | lb) ^ (neg ? 31 : 0)];
                            } else if (fmt == 35) {
                                const uint8_t (*ub1)[32] = (const uint8_t (*)[32])(base + 1024);
                                const uint8_t (*ham)[16] = (const uint8_t (*)[16])(base + 1536);
                                const int b1  = (ub1[g][lane] >> p) & 1;
                                const int nib = (j < 8) ? (ham[g][r] & 0x0F) : (ham[g][r] >> 4);
                                const int b0  = (HAM_CW8[nib] >> (7 - (j & 7))) & 1;
                                w_mag = tab_canon[((b1 << 5) | (b0 << 4) | lb) ^ (neg ? 63 : 0)];
                            } else if (fmt == 40) {
                                const uint8_t (*u2)[2][32] = (const uint8_t (*)[2][32])(base + 1024);
                                const int u_mag = (u2[g][p >> 2][lane] >> (2 * (p & 3))) & 3;
                                w_mag = tab_canon[((u_mag << 4) | lb) ^ (neg ? 63 : 0)];
                            } else if (fmt == 50) {
                                const uint8_t (*u2)[2][32] = (const uint8_t (*)[2][32])(base + 1024);
                                const uint8_t (*ub2)[32]   = (const uint8_t (*)[32])(base + 2048);
                                const int u01 = (u2[g][p >> 2][lane] >> (2 * (p & 3))) & 3;
                                const int b2  = (ub2[g][lane] >> p) & 1;
                                const int u_mag = (b2 << 2) | u01;
                                w_mag = tab_canon[((u_mag << 2) | lb) ^ (neg ? 31 : 0)];
                            } else if (fmt == 51) {
                                const uint8_t (*u2)[2][32] = (const uint8_t (*)[2][32])(base + 1024);
                                const uint8_t (*ub2)[32]   = (const uint8_t (*)[32])(base + 2048);
                                const int u01 = (u2[g][p >> 2][lane] >> (2 * (p & 3))) & 3;
                                const int b2  = (ub2[g][lane] >> p) & 1;
                                const int u_mag = (b2 << 2) | u01;
                                w_mag = tab_canon[((u_mag << 4) | lb) ^ (neg ? 127 : 0)];
                            } else if (fmt == 60) {
                                const uint8_t (*u4)[4][32] = (const uint8_t (*)[4][32])(base + 1024);
                                const int u_mag = (u4[g][p >> 1][lane] >> (4 * (p & 1))) & 15;
                                w_mag = tab_canon[((u_mag << 4) | lb) ^ (neg ? 255 : 0)];
                            }
                            sub_sum[r] += (neg ? -w_mag : w_mag) * av;
                        }
                    }
                }
                for (int r = 0; r < 16; r++) {
                    int gi = (hdr_p[g * h + 0] >> r) & 1;
                    if (h >= 2) gi |= ((hdr_p[g * h + 1] >> r) & 1) << 1;
                    if (h >= 3) gi |= ((hdr_p[g * h + 2] >> r) & 1) << 2;
                    isum_row[r] += sub_sum[r] * (int32_t)gain_u8[gi];
                }
            }
            for (int r = 0; r < 16; r++) {
                acc[r] += (double)isum_row[r] * (double)u_sb[r];
            }
        }
        for (int r = 0; r < 16; r++) {
            y[bg * 16 + r] = (float)(acc[r] * (double)d_row_eff[bg * 16 + r]);
        }
    }
}

// =================================================================================================
// VTQ CPU AVX2 Decode Kernels (`16x256` Super-Tile + Intra-Subblock `16x16` Trellis)
// =================================================================================================

// 1. K = 1.5 Gosset [8,4,4] (`nl = 16, h = 1, tab_s8[32]`, 1.594 bpw)
__attribute__((noinline))
void vtq_avx2_k15(const vtq_st16_k15_t* ST, const uint8_t* tab_avx, const uint8_t* gain_u8,
                   const int8_t* a, const float* d_row_eff, float* y) {
    const __m256i T0    = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)(tab_avx + 0)));
    const __m256i T1    = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)(tab_avx + 16)));
    const __m256i glut2 = _mm256_broadcastsi128_si256(_mm_loadl_epi64((const __m128i*)gain_u8));
    const __m256i zero  = _mm256_setzero_si256();
    const __m256i m80   = _mm256_set1_epi8((char)0x80);

    for (int bg = 0; bg < M_ROWS / 16; bg++) {
        float acc_f[8][8] __attribute__((aligned(32))) = {{0}};
        for (int sb = 0; sb < NSB; sb++) {
            const vtq_st16_k15_t* t = &ST[bg * NSB + sb];
            __m256i isum[8]; for (int p = 0; p < 8; p++) isum[p] = zero;
            for (int g = 0; g < 16; g++) {
                const uint8_t* cb_cur = (const uint8_t*)t->c[g];
                const __m256i a2  = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)(a + sb * 256 + 16 * g)));
                const __m256i geo = expand_gains_eo_h1(t->hdr[g][0], glut2);
                __m256i gsh = _mm256_set1_epi16((short)0x8000);
                #pragma GCC unroll 8
                for (int p = 0; p < 8; p++) {
                    __m256i v   = step_v_pair(cb_cur, p);
                    __m256i ul  = load_ham_pair(t->ham[g], 2 * p);
                    __m256i f   = _mm256_or_si256(v, ul);
                    __m256i ws  = _mm256_or_si256(_mm256_shuffle_epi8(T0, f), _mm256_shuffle_epi8(T1, _mm256_xor_si256(f, m80)));
                    __m256i prd = _mm256_maddubs_epi16(_mm256_sign_epi8(ws, ws), _mm256_sign_epi8(a2, ws));
                    __m256i gp  = _mm256_shuffle_epi8(geo, gsh);
                    gsh = _mm256_add_epi16(gsh, _mm256_set1_epi16(1));
                    isum[p] = _mm256_add_epi32(isum[p], _mm256_madd_epi16(prd, gp));
                }
            }
            flush_usb16(acc_f, isum, t->u_sb);
        }
        finish_y16(y + bg * 16, d_row_eff + bg * 16, acc_f);
    }
}

// 2. K = 2.0 (`nl = 16, h = 1, tab_u8[16]`, 2.094 bpw — single-pass `isum[8]`, zero redundant `usgn/a` loads!)
__attribute__((noinline))
void vtq_avx2_k20(const vtq_st16_k20_t* ST, const uint8_t* tab_avx, const uint8_t* gain_u8,
                   const int8_t* a, const float* d_row_eff, float* y) {
    const __m256i T0    = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)tab_avx));
    const __m256i glut2 = _mm256_broadcastsi128_si256(_mm_loadl_epi64((const __m128i*)gain_u8));
    const __m256i zero  = _mm256_setzero_si256();
    const __m256i one   = _mm256_set1_epi8(1);
    const __m256i m0f   = _mm256_set1_epi8(0x0F);

    for (int bg = 0; bg < M_ROWS / 16; bg++) {
        float acc_f[8][8] __attribute__((aligned(32))) = {{0}};
        for (int sb = 0; sb < NSB; sb++) {
            const vtq_st16_k20_t* t = &ST[bg * NSB + sb];
            __m256i isum[8]; for (int p = 0; p < 8; p++) isum[p] = zero;
            for (int g = 0; g < 16; g++) {
                const uint8_t* cb_cur = (const uint8_t*)t->c[g];
                const __m256i sv  = _mm256_loadu_si256((const __m256i*)t->usgn[g]);
                const __m256i a2  = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)(a + sb * 256 + 16 * g)));
                const __m256i geo = expand_gains_eo_h1(t->hdr[g][0], glut2);
                __m256i gsh = _mm256_set1_epi16((short)0x8000);
                #pragma GCC unroll 8
                for (int p = 0; p < 8; p++) {
                    __m256i v   = step_v_pair(cb_cur, p);
                    __m256i ss  = (p == 7) ? sv : _mm256_slli_epi16(sv, 7 - p);
                    __m256i neg = _mm256_cmpgt_epi8(zero, ss);
                    __m256i wm  = _mm256_shuffle_epi8(T0, _mm256_and_si256(_mm256_xor_si256(v, neg), m0f));
                    __m256i prd = _mm256_maddubs_epi16(wm, _mm256_sign_epi8(a2, _mm256_or_si256(ss, one)));
                    __m256i gp  = _mm256_shuffle_epi8(geo, gsh);
                    gsh = _mm256_add_epi16(gsh, _mm256_set1_epi16(1));
                    isum[p] = _mm256_add_epi32(isum[p], _mm256_madd_epi16(prd, gp));
                }
            }
            flush_usb16(acc_f, isum, t->u_sb);
        }
        finish_y16(y + bg * 16, d_row_eff + bg * 16, acc_f);
    }
}

// 3. K = 2.5 Gosset [8,4,4] (`nl = 16, h = 1, tab_u8[32]`, 2.594 bpw)
__attribute__((noinline))
void vtq_avx2_k25(const vtq_st16_k25_t* ST, const uint8_t* tab_avx, const uint8_t* gain_u8,
                   const int8_t* a, const float* d_row_eff, float* y) {
    const __m256i T0    = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)(tab_avx + 0)));
    const __m256i T1    = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)(tab_avx + 16)));
    const __m256i glut2 = _mm256_broadcastsi128_si256(_mm_loadl_epi64((const __m128i*)gain_u8));
    const __m256i zero  = _mm256_setzero_si256();
    const __m256i one   = _mm256_set1_epi8(1);
    const __m256i m80   = _mm256_set1_epi8((char)0x80);

    for (int bg = 0; bg < M_ROWS / 16; bg++) {
        float acc_f[8][8] __attribute__((aligned(32))) = {{0}};
        for (int sb = 0; sb < NSB; sb++) {
            const vtq_st16_k25_t* t = &ST[bg * NSB + sb];
            __m256i isum[8]; for (int p = 0; p < 8; p++) isum[p] = zero;
            for (int g = 0; g < 16; g++) {
                const uint8_t* cb_cur = (const uint8_t*)t->c[g];
                const __m256i sv  = _mm256_loadu_si256((const __m256i*)t->usgn[g]);
                const __m256i a2  = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)(a + sb * 256 + 16 * g)));
                const __m256i geo = expand_gains_eo_h1(t->hdr[g][0], glut2);
                __m256i gsh = _mm256_set1_epi16((short)0x8000);
                #pragma GCC unroll 8
                for (int p = 0; p < 8; p++) {
                    __m256i v   = step_v_pair(cb_cur, p);
                    __m256i sb7 = _mm256_and_si256(p == 7 ? sv : _mm256_slli_epi16(sv, 7 - p), m80);
                    __m256i neg = _mm256_cmpgt_epi8(zero, sb7);
                    __m256i ul  = load_ham_pair(t->ham[g], 2 * p);
                    __m256i f   = _mm256_xor_si256(_mm256_or_si256(v, ul), neg);
                    __m256i wm  = _mm256_or_si256(_mm256_shuffle_epi8(T0, f), _mm256_shuffle_epi8(T1, _mm256_xor_si256(f, m80)));
                    __m256i prd = _mm256_maddubs_epi16(wm, _mm256_sign_epi8(a2, _mm256_or_si256(neg, one)));
                    __m256i gp  = _mm256_shuffle_epi8(geo, gsh);
                    gsh = _mm256_add_epi16(gsh, _mm256_set1_epi16(1));
                    isum[p] = _mm256_add_epi32(isum[p], _mm256_madd_epi16(prd, gp));
                }
            }
            flush_usb16(acc_f, isum, t->u_sb);
        }
        finish_y16(y + bg * 16, d_row_eff + bg * 16, acc_f);
    }
}

// 4. K = 3.0 Fast (`nl = 8, h = 2, tab_u8[16]`, 3.156 bpw — 1x `vpshufb`)
__attribute__((noinline))
void vtq_avx2_k30_nl8(const vtq_st16_k30_t* ST, const uint8_t* tab_avx, const uint8_t* gain_u8,
                       const int8_t* a, const float* d_row_eff, float* y) {
    const __m256i T0    = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)tab_avx));
    const __m256i glut2 = _mm256_broadcastsi128_si256(_mm_loadl_epi64((const __m128i*)gain_u8));
    const __m256i zero  = _mm256_setzero_si256();
    const __m256i one   = _mm256_set1_epi8(1);
    const __m256i m80   = _mm256_set1_epi8((char)0x80);
    const __m256i m7f   = _mm256_set1_epi8(0x7F);

    for (int bg = 0; bg < M_ROWS / 16; bg++) {
        float acc_f[8][8] __attribute__((aligned(32))) = {{0}};
        for (int sb = 0; sb < NSB; sb++) {
            const vtq_st16_k30_t* t = &ST[bg * NSB + sb];
            __m256i isum[8]; for (int p = 0; p < 8; p++) isum[p] = zero;
            for (int g = 0; g < 16; g++) {
                const uint8_t* cb_cur = (const uint8_t*)t->c[g];
                const __m256i sv  = _mm256_loadu_si256((const __m256i*)t->usgn[g]);
                const __m256i ub0 = _mm256_loadu_si256((const __m256i*)t->ub0[g]);
                const __m256i a2  = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)(a + sb * 256 + 16 * g)));
                const __m256i geo = expand_gains_eo_h2(t->hdr[g], glut2);
                __m256i gsh = _mm256_set1_epi16((short)0x8000);
                #pragma GCC unroll 8
                for (int p = 0; p < 8; p++) {
                    __m256i v   = step_v_pair(cb_cur, p);
                    __m256i sb7 = _mm256_and_si256(p == 7 ? sv : _mm256_slli_epi16(sv, 7 - p), m80);
                    __m256i neg = _mm256_cmpgt_epi8(zero, sb7);
                    __m256i ul  = _mm256_and_si256(p == 0 ? ub0 : _mm256_srli_epi16(ub0, p), one);
                    __m256i wm  = _mm256_shuffle_epi8(T0, _mm256_and_si256(_mm256_xor_si256(_mm256_or_si256(v, ul), neg), m7f));
                    __m256i prd = _mm256_maddubs_epi16(wm, _mm256_sign_epi8(a2, _mm256_or_si256(neg, one)));
                    __m256i gp  = _mm256_shuffle_epi8(geo, gsh);
                    gsh = _mm256_add_epi16(gsh, _mm256_set1_epi16(1));
                    isum[p] = _mm256_add_epi32(isum[p], _mm256_madd_epi16(prd, gp));
                }
            }
            flush_usb16(acc_f, isum, t->u_sb);
        }
        finish_y16(y + bg * 16, d_row_eff + bg * 16, acc_f);
    }
}

// 5. K = 3.0 HQ (`nl = 16, h = 2, tab_u8[32]`, 3.156 bpw — 2x `vpshufb` bit-7 zeroing)
__attribute__((noinline))
void vtq_avx2_k30_nl16(const vtq_st16_k30_t* ST, const uint8_t* tab_avx, const uint8_t* gain_u8,
                        const int8_t* a, const float* d_row_eff, float* y) {
    const __m256i T0    = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)(tab_avx + 0)));
    const __m256i T1    = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)(tab_avx + 16)));
    const __m256i glut2 = _mm256_broadcastsi128_si256(_mm_loadl_epi64((const __m128i*)gain_u8));
    const __m256i zero  = _mm256_setzero_si256();
    const __m256i one   = _mm256_set1_epi8(1);
    const __m256i m80   = _mm256_set1_epi8((char)0x80);

    for (int bg = 0; bg < M_ROWS / 16; bg++) {
        float acc_f[8][8] __attribute__((aligned(32))) = {{0}};
        for (int sb = 0; sb < NSB; sb++) {
            const vtq_st16_k30_t* t = &ST[bg * NSB + sb];
            __m256i isum[8]; for (int p = 0; p < 8; p++) isum[p] = zero;
            for (int g = 0; g < 16; g++) {
                const uint8_t* cb_cur = (const uint8_t*)t->c[g];
                const __m256i sv  = _mm256_loadu_si256((const __m256i*)t->usgn[g]);
                const __m256i ub0 = _mm256_loadu_si256((const __m256i*)t->ub0[g]);
                const __m256i a2  = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)(a + sb * 256 + 16 * g)));
                const __m256i geo = expand_gains_eo_h2(t->hdr[g], glut2);
                __m256i gsh = _mm256_set1_epi16((short)0x8000);
                #pragma GCC unroll 8
                for (int p = 0; p < 8; p++) {
                    __m256i v   = step_v_pair(cb_cur, p);
                    __m256i sb7 = _mm256_and_si256(p == 7 ? sv : _mm256_slli_epi16(sv, 7 - p), m80);
                    __m256i neg = _mm256_cmpgt_epi8(zero, sb7);
                    __m256i ul  = _mm256_and_si256(p == 0 ? ub0 : _mm256_srli_epi16(ub0, p), one);
                    __m256i f   = _mm256_xor_si256(_mm256_or_si256(v, ul), neg);
                    __m256i wm  = _mm256_or_si256(_mm256_shuffle_epi8(T0, f), _mm256_shuffle_epi8(T1, _mm256_xor_si256(f, m80)));
                    __m256i prd = _mm256_maddubs_epi16(wm, _mm256_sign_epi8(a2, _mm256_or_si256(neg, one)));
                    __m256i gp  = _mm256_shuffle_epi8(geo, gsh);
                    gsh = _mm256_add_epi16(gsh, _mm256_set1_epi16(1));
                    isum[p] = _mm256_add_epi32(isum[p], _mm256_madd_epi16(prd, gp));
                }
            }
            flush_usb16(acc_f, isum, t->u_sb);
        }
        finish_y16(y + bg * 16, d_row_eff + bg * 16, acc_f);
    }
}

// 6. K = 3.5 Gosset [8,4,4] (`nl = 16, h = 2, AT2 16B+16B`, 3.656 bpw — dual-nibble `2x vpshufb + vpaddusb`)
__attribute__((noinline))
void vtq_avx2_k35(const vtq_st16_k35_t* ST, const uint8_t* tab_avx, const uint8_t* gain_u8,
                   const int8_t* a, const float* d_row_eff, float* y) {
    const __m256i TA0   = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)(tab_avx + 0)));
    const __m256i TB0   = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)(tab_avx + 16)));
    const __m256i glut2 = _mm256_broadcastsi128_si256(_mm_loadl_epi64((const __m128i*)gain_u8));
    const __m256i zero  = _mm256_setzero_si256();
    const __m256i one   = _mm256_set1_epi8(1);
    const __m256i two   = _mm256_set1_epi8(2);
    const __m256i m0f   = _mm256_set1_epi8(0x0F);
    const __m256i m80   = _mm256_set1_epi8((char)0x80);

    for (int bg = 0; bg < M_ROWS / 16; bg++) {
        float acc_f[8][8] __attribute__((aligned(32))) = {{0}};
        for (int sb = 0; sb < NSB; sb++) {
            const vtq_st16_k35_t* t = &ST[bg * NSB + sb];
            __m256i isum[8]; for (int p = 0; p < 8; p++) isum[p] = zero;
            for (int g = 0; g < 16; g++) {
                const uint8_t* cb_cur = (const uint8_t*)t->c[g];
                const __m256i sv  = _mm256_loadu_si256((const __m256i*)t->usgn[g]);
                const __m256i ub1 = _mm256_loadu_si256((const __m256i*)t->ub1[g]);
                const __m256i a2  = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)(a + sb * 256 + 16 * g)));
                const __m256i geo = expand_gains_eo_h2(t->hdr[g], glut2);
                __m256i gsh = _mm256_set1_epi16((short)0x8000);
                #pragma GCC unroll 8
                for (int p = 0; p < 8; p++) {
                    __m256i v   = step_v_pair(cb_cur, p);
                    __m256i sb7 = _mm256_and_si256(p == 7 ? sv : _mm256_slli_epi16(sv, 7 - p), m80);
                    __m256i neg = _mm256_cmpgt_epi8(zero, sb7);
                    __m256i b0  = load_ham_pair(t->ham[g], 2 * p);
                    __m256i b1  = _mm256_and_si256(p == 0 ? _mm256_slli_epi16(ub1, 1) : (p == 1 ? ub1 : _mm256_srli_epi16(ub1, p - 1)), two);
                    __m256i b1d = _mm256_or_si256(b1, _mm256_slli_epi16(b1, 4));
                    __m256i fv  = _mm256_xor_si256(_mm256_or_si256(v, _mm256_or_si256(b1d, b0)), neg);
                    __m256i wm  = _mm256_adds_epu8(_mm256_shuffle_epi8(TA0, _mm256_and_si256(fv, m0f)),
                                                   _mm256_shuffle_epi8(TB0, _mm256_and_si256(_mm256_srli_epi16(fv, 4), m0f)));
                    __m256i prd = _mm256_maddubs_epi16(wm, _mm256_sign_epi8(a2, _mm256_or_si256(neg, one)));
                    __m256i gp  = _mm256_shuffle_epi8(geo, gsh);
                    gsh = _mm256_add_epi16(gsh, _mm256_set1_epi16(1));
                    isum[p] = _mm256_add_epi32(isum[p], _mm256_madd_epi16(prd, gp));
                }
            }
            flush_usb16(acc_f, isum, t->u_sb);
        }
        finish_y16(y + bg * 16, d_row_eff + bg * 16, acc_f);
    }
}

// 7. K = 4.0 (`nl = 16, h = 3, AT2 16B+16B`, 4.219 bpw — dual-nibble `2x vpshufb + vpaddusb`)
__attribute__((noinline))
void vtq_avx2_k40(const vtq_st16_k40_t* ST, const uint8_t* tab_avx, const uint8_t* gain_u8,
                   const int8_t* a, const float* d_row_eff, float* y) {
    const __m256i TA0   = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)(tab_avx + 0)));
    const __m256i TB0   = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)(tab_avx + 16)));
    const __m256i glut2 = _mm256_broadcastsi128_si256(_mm_loadl_epi64((const __m128i*)gain_u8));
    const __m256i zero  = _mm256_setzero_si256();
    const __m256i one   = _mm256_set1_epi8(1);
    const __m256i m03   = _mm256_set1_epi8(3);
    const __m256i m0f   = _mm256_set1_epi8(0x0F);
    const __m256i m80   = _mm256_set1_epi8((char)0x80);

    for (int bg = 0; bg < M_ROWS / 16; bg++) {
        float acc_f[8][8] __attribute__((aligned(32))) = {{0}};
        for (int sb = 0; sb < NSB; sb++) {
            const vtq_st16_k40_t* t = &ST[bg * NSB + sb];
            __m256i isum[8]; for (int p = 0; p < 8; p++) isum[p] = zero;
            for (int g = 0; g < 16; g++) {
                const uint8_t* cb_cur = (const uint8_t*)t->c[g];
                const __m256i sv  = _mm256_loadu_si256((const __m256i*)t->usgn[g]);
                const __m256i a2  = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)(a + sb * 256 + 16 * g)));
                const __m256i geo = expand_gains_eo_h3(t->hdr[g], glut2);
                __m256i gsh = _mm256_set1_epi16((short)0x8000);
                __m256i u_cur = _mm256_loadu_si256((const __m256i*)t->u2[g][0]);
                #pragma GCC unroll 8
                for (int p = 0; p < 8; p++) {
                    if (p == 4) u_cur = _mm256_loadu_si256((const __m256i*)t->u2[g][1]);
                    __m256i v   = step_v_pair(cb_cur, p);
                    __m256i sb7 = _mm256_and_si256(p == 7 ? sv : _mm256_slli_epi16(sv, 7 - p), m80);
                    __m256i neg = _mm256_cmpgt_epi8(zero, sb7);
                    const int sh = 2 * (p & 3);
                    __m256i ul  = _mm256_and_si256(sh ? _mm256_srli_epi16(u_cur, sh) : u_cur, m03);
                    __m256i uld = _mm256_or_si256(ul, _mm256_slli_epi16(ul, 4));
                    __m256i fv  = _mm256_xor_si256(_mm256_or_si256(v, uld), neg);
                    __m256i wm  = _mm256_adds_epu8(_mm256_shuffle_epi8(TA0, _mm256_and_si256(fv, m0f)),
                                                   _mm256_shuffle_epi8(TB0, _mm256_and_si256(_mm256_srli_epi16(fv, 4), m0f)));
                    __m256i prd = _mm256_maddubs_epi16(wm, _mm256_sign_epi8(a2, _mm256_or_si256(neg, one)));
                    __m256i gp  = _mm256_shuffle_epi8(geo, gsh);
                    gsh = _mm256_add_epi16(gsh, _mm256_set1_epi16(1));
                    isum[p] = _mm256_add_epi32(isum[p], _mm256_madd_epi16(prd, gp));
                }
            }
            flush_usb16(acc_f, isum, t->u_sb);
        }
        finish_y16(y + bg * 16, d_row_eff + bg * 16, acc_f);
    }
}

// 8. K = 5.0 Fast (`nl = 4, h = 3, tab_u8[32]`, 5.219 bpw — 2x `vpshufb` bit-7 zeroing)
__attribute__((noinline))
void vtq_avx2_k50_nl4(const vtq_st16_k50_t* ST, const uint8_t* tab_avx, const uint8_t* gain_u8,
                       const int8_t* a, const float* d_row_eff, float* y) {
    const __m256i T0    = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)(tab_avx + 0)));
    const __m256i T1    = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)(tab_avx + 16)));
    const __m256i glut2 = _mm256_broadcastsi128_si256(_mm_loadl_epi64((const __m128i*)gain_u8));
    const __m256i zero  = _mm256_setzero_si256();
    const __m256i one   = _mm256_set1_epi8(1);
    const __m256i m03   = _mm256_set1_epi8(3);
    const __m256i m04   = _mm256_set1_epi8(4);
    const __m256i m80   = _mm256_set1_epi8((char)0x80);

    for (int bg = 0; bg < M_ROWS / 16; bg++) {
        float acc_f[8][8] __attribute__((aligned(32))) = {{0}};
        for (int sb = 0; sb < NSB; sb++) {
            const vtq_st16_k50_t* t = &ST[bg * NSB + sb];
            __m256i isum[8]; for (int p = 0; p < 8; p++) isum[p] = zero;
            for (int g = 0; g < 16; g++) {
                const uint8_t* cb_cur = (const uint8_t*)t->c[g];
                const __m256i sv  = _mm256_loadu_si256((const __m256i*)t->usgn[g]);
                const __m256i ub2 = _mm256_loadu_si256((const __m256i*)t->ub2[g]);
                const __m256i a2  = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)(a + sb * 256 + 16 * g)));
                const __m256i geo = expand_gains_eo_h3(t->hdr[g], glut2);
                __m256i gsh = _mm256_set1_epi16((short)0x8000);
                __m256i u_cur = _mm256_loadu_si256((const __m256i*)t->u2[g][0]);
                #pragma GCC unroll 8
                for (int p = 0; p < 8; p++) {
                    if (p == 4) u_cur = _mm256_loadu_si256((const __m256i*)t->u2[g][1]);
                    __m256i v   = step_v_pair(cb_cur, p);
                    __m256i sb7 = _mm256_and_si256(p == 7 ? sv : _mm256_slli_epi16(sv, 7 - p), m80);
                    __m256i neg = _mm256_cmpgt_epi8(zero, sb7);
                    const int sh = 2 * (p & 3);
                    __m256i u01 = _mm256_and_si256(sh ? _mm256_srli_epi16(u_cur, sh) : u_cur, m03);
                    __m256i b2  = _mm256_and_si256(p <= 2 ? _mm256_slli_epi16(ub2, 2 - p) : _mm256_srli_epi16(ub2, p - 2), m04);
                    __m256i f   = _mm256_xor_si256(_mm256_or_si256(v, _mm256_or_si256(u01, b2)), neg);
                    __m256i wm  = _mm256_or_si256(_mm256_shuffle_epi8(T0, f), _mm256_shuffle_epi8(T1, _mm256_xor_si256(f, m80)));
                    __m256i prd = _mm256_maddubs_epi16(wm, _mm256_sign_epi8(a2, _mm256_or_si256(neg, one)));
                    __m256i gp  = _mm256_shuffle_epi8(geo, gsh);
                    gsh = _mm256_add_epi16(gsh, _mm256_set1_epi16(1));
                    isum[p] = _mm256_add_epi32(isum[p], _mm256_madd_epi16(prd, gp));
                }
            }
            flush_usb16(acc_f, isum, t->u_sb);
        }
        finish_y16(y + bg * 16, d_row_eff + bg * 16, acc_f);
    }
}

// 9. K = 5.0 HQ (`nl = 16, h = 3, AT2Z 32B+32B`, 5.219 bpw — 4x `vpshufb` bit-7 zeroing, 0 `vpblendvb`!)
__attribute__((noinline))
void vtq_avx2_k50_nl16(const vtq_st16_k50_t* ST, const uint8_t* tab_avx, const uint8_t* gain_u8,
                        const int8_t* a, const float* d_row_eff, float* y) {
    const __m256i TA0   = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)(tab_avx + 0)));
    const __m256i TA1   = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)(tab_avx + 16)));
    const __m256i TB0   = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)(tab_avx + 32)));
    const __m256i TB1   = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)(tab_avx + 48)));
    const __m256i glut2 = _mm256_broadcastsi128_si256(_mm_loadl_epi64((const __m128i*)gain_u8));
    const __m256i zero  = _mm256_setzero_si256();
    const __m256i one   = _mm256_set1_epi8(1);
    const __m256i m03   = _mm256_set1_epi8(3);
    const __m256i m04   = _mm256_set1_epi8(4);
    const __m256i m80   = _mm256_set1_epi8((char)0x80);

    for (int bg = 0; bg < M_ROWS / 16; bg++) {
        float acc_f[8][8] __attribute__((aligned(32))) = {{0}};
        for (int sb = 0; sb < NSB; sb++) {
            const vtq_st16_k50_t* t = &ST[bg * NSB + sb];
            __m256i isum[8]; for (int p = 0; p < 8; p++) isum[p] = zero;
            for (int g = 0; g < 16; g++) {
                const uint8_t* cb_cur = (const uint8_t*)t->c[g];
                const __m256i sv  = _mm256_loadu_si256((const __m256i*)t->usgn[g]);
                const __m256i ub2 = _mm256_loadu_si256((const __m256i*)t->ub2[g]);
                const __m256i a2  = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)(a + sb * 256 + 16 * g)));
                const __m256i geo = expand_gains_eo_h3(t->hdr[g], glut2);
                __m256i gsh = _mm256_set1_epi16((short)0x8000);
                __m256i u_cur = _mm256_loadu_si256((const __m256i*)t->u2[g][0]);
                #pragma GCC unroll 8
                for (int p = 0; p < 8; p++) {
                    if (p == 4) u_cur = _mm256_loadu_si256((const __m256i*)t->u2[g][1]);
                    __m256i v    = step_v_pair(cb_cur, p);
                    __m256i sb7  = _mm256_and_si256(p == 7 ? sv : _mm256_slli_epi16(sv, 7 - p), m80);
                    __m256i neg  = _mm256_cmpgt_epi8(zero, sb7);
                    const int sh = 2 * (p & 3);
                    __m256i u01  = _mm256_and_si256(sh ? _mm256_srli_epi16(u_cur, sh) : u_cur, m03);
                    __m256i b2   = _mm256_and_si256(p <= 2 ? _mm256_slli_epi16(ub2, 2 - p) : _mm256_srli_epi16(ub2, p - 2), m04);
                    __m256i fv   = _mm256_xor_si256(_mm256_or_si256(v, _mm256_or_si256(u01, b2)), neg);
                    __m256i wA   = _mm256_or_si256(_mm256_shuffle_epi8(TA0, fv), _mm256_shuffle_epi8(TA1, _mm256_xor_si256(fv, m80)));
                    __m256i idxB = _mm256_or_si256(_mm256_add_epi8(fv, fv), _mm256_and_si256(_mm256_srli_epi16(fv, 4), one));
                    __m256i wB   = _mm256_or_si256(_mm256_shuffle_epi8(TB0, idxB), _mm256_shuffle_epi8(TB1, _mm256_xor_si256(idxB, m80)));
                    __m256i wm   = _mm256_adds_epu8(wA, wB);
                    __m256i prd  = _mm256_maddubs_epi16(wm, _mm256_sign_epi8(a2, _mm256_or_si256(neg, one)));
                    __m256i gp   = _mm256_shuffle_epi8(geo, gsh);
                    gsh = _mm256_add_epi16(gsh, _mm256_set1_epi16(1));
                    isum[p] = _mm256_add_epi32(isum[p], _mm256_madd_epi16(prd, gp));
                }
            }
            flush_usb16(acc_f, isum, t->u_sb);
        }
        finish_y16(y + bg * 16, d_row_eff + bg * 16, acc_f);
    }
}

// 10. K = 6.0 (`nl = 16, h = 3, AT2Z 32B+32B`, 6.219 bpw — packed `u4[16][4][32]` + bit-7 zeroing, 0 `vpblendvb`!)
__attribute__((noinline))
void vtq_avx2_k60_nl16(const vtq_st16_k60_t* ST, const uint8_t* tab_avx, const uint8_t* gain_u8,
                        const int8_t* a, const float* d_row_eff, float* y) {
    const __m256i TA0   = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)(tab_avx + 0)));
    const __m256i TA1   = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)(tab_avx + 16)));
    const __m256i TB0   = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)(tab_avx + 32)));
    const __m256i TB1   = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)(tab_avx + 48)));
    const __m256i glut2 = _mm256_broadcastsi128_si256(_mm_loadl_epi64((const __m128i*)gain_u8));
    const __m256i zero  = _mm256_setzero_si256();
    const __m256i one   = _mm256_set1_epi8(1);
    const __m256i m0f   = _mm256_set1_epi8(0x0F);
    const __m256i m7f   = _mm256_set1_epi8(0x7F);
    const __m256i m80   = _mm256_set1_epi8((char)0x80);

    for (int bg = 0; bg < M_ROWS / 16; bg++) {
        float acc_f[8][8] __attribute__((aligned(32))) = {{0}};
        for (int sb = 0; sb < NSB; sb++) {
            const vtq_st16_k60_t* t = &ST[bg * NSB + sb];
            __m256i isum[8]; for (int p = 0; p < 8; p++) isum[p] = zero;
            for (int g = 0; g < 16; g++) {
                const uint8_t* cb_cur = (const uint8_t*)t->c[g];
                const __m256i sv  = _mm256_loadu_si256((const __m256i*)t->usgn[g]);
                const __m256i a2  = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)(a + sb * 256 + 16 * g)));
                const __m256i geo = expand_gains_eo_h3(t->hdr[g], glut2);
                __m256i gsh = _mm256_set1_epi16((short)0x8000);
                __m256i u_nib = zero;
                #pragma GCC unroll 8
                for (int p = 0; p < 8; p++) {
                    if ((p & 1) == 0) u_nib = _mm256_loadu_si256((const __m256i*)t->u4[g][p >> 1]);
                    __m256i v    = step_v_pair(cb_cur, p);
                    __m256i sb7  = _mm256_and_si256(p == 7 ? sv : _mm256_slli_epi16(sv, 7 - p), m80);
                    __m256i neg  = _mm256_cmpgt_epi8(zero, sb7);
                    __m256i ul   = _mm256_and_si256((p & 1) ? _mm256_srli_epi16(u_nib, 4) : u_nib, m0f);
                    __m256i fv   = _mm256_xor_si256(_mm256_or_si256(v, ul), neg);
                    __m256i wA   = _mm256_or_si256(_mm256_shuffle_epi8(TA0, fv), _mm256_shuffle_epi8(TA1, _mm256_xor_si256(fv, m80)));
                    __m256i idxB = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(fv, 2), m7f),
                                                   _mm256_and_si256(_mm256_add_epi8(fv, fv), m80));
                    __m256i wB   = _mm256_or_si256(_mm256_shuffle_epi8(TB0, idxB), _mm256_shuffle_epi8(TB1, _mm256_xor_si256(idxB, m80)));
                    __m256i wm   = _mm256_adds_epu8(wA, wB);
                    __m256i prd  = _mm256_maddubs_epi16(wm, _mm256_sign_epi8(a2, _mm256_or_si256(neg, one)));
                    __m256i gp   = _mm256_shuffle_epi8(geo, gsh);
                    gsh = _mm256_add_epi16(gsh, _mm256_set1_epi16(1));
                    isum[p] = _mm256_add_epi32(isum[p], _mm256_madd_epi16(prd, gp));
                }
            }
            flush_usb16(acc_f, isum, t->u_sb);
        }
        finish_y16(y + bg * 16, d_row_eff + bg * 16, acc_f);
    }
}

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static uint32_t xorshift32(uint32_t* s) {
    uint32_t x = *s;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    return (*s = x);
}

static void fill_rand_bytes(void* p, size_t n, uint32_t seed) {
    uint8_t* b = (uint8_t*)p;
    uint32_t s = seed;
    for (size_t i = 0; i < n; i += 4) {
        uint32_t v = xorshift32(&s);
        size_t rem = (n - i < 4) ? (n - i) : 4;
        memcpy(b + i, &v, rem);
    }
}

static double rel_l2_err(const float* y, const float* ref, int n) {
    double num = 0.0, den = 0.0;
    for (int i = 0; i < n; i++) {
        double d = (double)y[i] - (double)ref[i];
        num += d * d;
        den += (double)ref[i] * (double)ref[i];
    }
    return sqrt(num / (den + 1e-30));
}

int main(void) {
    const int nst = (M_ROWS / 16) * NSB;
    vtq_st16_k15_t* S15 = aligned_alloc(64, (size_t)nst * sizeof(*S15));
    vtq_st16_k20_t* S20 = aligned_alloc(64, (size_t)nst * sizeof(*S20));
    vtq_st16_k25_t* S25 = aligned_alloc(64, (size_t)nst * sizeof(*S25));
    vtq_st16_k30_t* S30 = aligned_alloc(64, (size_t)nst * sizeof(*S30));
    vtq_st16_k35_t* S35 = aligned_alloc(64, (size_t)nst * sizeof(*S35));
    vtq_st16_k40_t* S40 = aligned_alloc(64, (size_t)nst * sizeof(*S40));
    vtq_st16_k50_t* S50 = aligned_alloc(64, (size_t)nst * sizeof(*S50));
    vtq_st16_k60_t* S60 = aligned_alloc(64, (size_t)nst * sizeof(*S60));

    fill_rand_bytes(S15, (size_t)nst * sizeof(*S15), 0x1515A5A5u);
    fill_rand_bytes(S20, (size_t)nst * sizeof(*S20), 0x2020A5A5u);
    fill_rand_bytes(S25, (size_t)nst * sizeof(*S25), 0x2525A5A5u);
    fill_rand_bytes(S30, (size_t)nst * sizeof(*S30), 0x3030A5A5u);
    fill_rand_bytes(S35, (size_t)nst * sizeof(*S35), 0x3535A5A5u);
    fill_rand_bytes(S40, (size_t)nst * sizeof(*S40), 0x4040A5A5u);
    fill_rand_bytes(S50, (size_t)nst * sizeof(*S50), 0x5050A5A5u);
    fill_rand_bytes(S60, (size_t)nst * sizeof(*S60), 0x6060A5A5u);

    uint8_t tab_s8_canon[32], tab_u8_canon[64], tab_avx[64];
    uint8_t ta16[16], tb16[16], tab_at2_k4[64];
    uint8_t ta32[32], tb32[32], tab_at2z_k5[128], tab_at2z_k6[256];
    for (int i = 0; i < 32; i++) tab_s8_canon[i] = (uint8_t)(int8_t)((i < 16) ? -(127 - 7 * i) : (8 + 7 * (i - 16)));
    for (int i = 0; i < 64; i++) tab_u8_canon[i] = (uint8_t)(2 + ((i * 125) / 63));
    for (int i = 0; i < 16; i++) { ta16[i] = (uint8_t)(1 + 3 * i); tb16[i] = (uint8_t)(2 + 4 * i); }
    for (int um = 0; um < 4; um++) {
        for (int lb = 0; lb < 16; lb++) {
            tab_at2_k4[(um << 4) | lb] = (uint8_t)(ta16[um | ((lb & 3) << 2)] + tb16[um | ((lb >> 2) << 2)]);
        }
    }
    for (int i = 0; i < 32; i++) { ta32[i] = (uint8_t)(1 + (i * 60) / 31); tb32[i] = (uint8_t)(2 + (i * 62) / 31); }
    for (int um = 0; um < 8; um++) {
        for (int lb = 0; lb < 16; lb++) {
            tab_at2z_k5[(um << 4) | lb] = (uint8_t)(ta32[um | ((lb & 3) << 3)] + tb32[um | ((lb >> 2) << 3)]);
        }
    }
    for (int um = 0; um < 16; um++) {
        for (int lb = 0; lb < 16; lb++) {
            tab_at2z_k6[(um << 4) | lb] = (uint8_t)(ta32[um | ((lb & 1) << 4)] + tb32[(um >> 2) | ((lb >> 1) << 2)]);
        }
    }
    const uint8_t gain_u8[8] = {42, 49, 56, 63, 71, 79, 89, 101};

    int8_t* a     = aligned_alloc(64, K_COLS);
    float*  dr    = aligned_alloc(64, M_ROWS * sizeof(float));
    float*  y     = aligned_alloc(64, M_ROWS * sizeof(float));
    float*  y_ref = aligned_alloc(64, M_ROWS * sizeof(float));
    uint32_t sa = 0x12345678u;
    for (int i = 0; i < K_COLS; i++) a[i] = (int8_t)((int)(xorshift32(&sa) % 255u) - 127);
    for (int i = 0; i < M_ROWS; i++) dr[i] = (0.0005f + 0.0002f * (i & 7)) * (1.0f / 64.0f);

    const int iters = (M_ROWS * K_COLS >= 1000000) ? 400 : 1200;
    printf("=== VTQ 16x256 Super-Tile CPU AVX2 Benchmark (M=%d, K=%d, -mno-avx512f) ===\n", M_ROWS, K_COLS);

    #define BENCH_ONE(label, fmt_id, ub_b, nl_v, lut_m, ham_pos, tab_src, fn, st_ptr) do { \
        build_vtq_luts((ub_b), (nl_v), (lut_m), (ham_pos)); \
        if ((fmt_id) == 15 || (fmt_id) == 25) memcpy(tab_avx, (tab_src), 32); \
        else if ((lut_m) == 2) { memcpy(tab_avx, ta16, 16); memcpy(tab_avx + 16, tb16, 16); } \
        else if ((lut_m) == 3) { prepare_at2z_k5_for_avx2(ta32, tb32, tab_avx); } \
        else if ((lut_m) == 4) { memcpy(tab_avx, ta32, 32); memcpy(tab_avx + 32, tb32, 32); } \
        else permute_tab_for_avx2((tab_src), tab_avx, (ub_b), (nl_v)); \
        scalar_ref_vtq((st_ptr), sizeof(*(st_ptr)), (tab_src), gain_u8, a, dr, y_ref, (fmt_id)); \
        fn((st_ptr), tab_avx, gain_u8, a, dr, y); \
        double err = rel_l2_err(y, y_ref, M_ROWS); \
        for (int w = 0; w < 40; w++) { fn((st_ptr), tab_avx, gain_u8, a, dr, y); __asm__ volatile("" ::: "memory"); } \
        double t0 = now_s(); \
        for (int it = 0; it < iters; it++) { fn((st_ptr), tab_avx, gain_u8, a, dr, y); __asm__ volatile("" ::: "memory"); } \
        double dt = (now_s() - t0) / iters; \
        double gws = (double)M_ROWS * (double)K_COLS / dt * 1e-9; \
        double bpw = (double)sizeof(*(st_ptr)) * 8.0 / 4096.0 + 16.0 / K_COLS + 16.0 / M_ROWS; \
        printf("%-34s (%5.3f bpw): %6.2f GW/s | rel_err=%.2e %s\n", \
               (label), bpw, gws, err, (err < 1e-6) ? "PASS" : "FAIL!"); \
        if (err >= 1e-6) return 1; \
    } while (0)

    BENCH_ONE("VTQ K=1.5 nl=16 (Gosset s8)",   15, 0, 16, 0,  7, tab_s8_canon, vtq_avx2_k15,      S15);
    BENCH_ONE("VTQ K=2.0 nl=16 (16-ent)",      20, 0, 16, 0, -1, tab_u8_canon, vtq_avx2_k20,      S20);
    BENCH_ONE("VTQ K=2.5 nl=16 (Gosset)",      25, 0, 16, 0,  7, tab_u8_canon, vtq_avx2_k25,      S25);
    BENCH_ONE("VTQ K=3.0 nl=8  (16-ent fast)", 30, 1,  8, 0, -1, tab_u8_canon, vtq_avx2_k30_nl8,  S30);
    BENCH_ONE("VTQ K=3.0 nl=16 (32-ent HQ)",   31, 1, 16, 0, -1, tab_u8_canon, vtq_avx2_k30_nl16, S30);
    BENCH_ONE("VTQ K=3.5 nl=16 (Gosset AT2)",  35, 2, 16, 2,  0, tab_at2_k4,   vtq_avx2_k35,      S35);
    BENCH_ONE("VTQ K=4.0 nl=16 (AT2 16+16)",   40, 2, 16, 2, -1, tab_at2_k4,   vtq_avx2_k40,      S40);
    BENCH_ONE("VTQ K=5.0 nl=4  (32-ent fast)", 50, 3,  4, 0, -1, tab_u8_canon, vtq_avx2_k50_nl4,  S50);
    BENCH_ONE("VTQ K=5.0 nl=16 (AT2Z 32+32)",  51, 3, 16, 3, -1, tab_at2z_k5,  vtq_avx2_k50_nl16, S50);
    BENCH_ONE("VTQ K=6.0 nl=16 (AT2Z 32+32)",  60, 4, 16, 4, -1, tab_at2z_k6,  vtq_avx2_k60_nl16, S60);

    free(S15); free(S20); free(S25); free(S30); free(S35); free(S40); free(S50); free(S60);
    free(a); free(dr); free(y); free(y_ref);
    return 0;
}
