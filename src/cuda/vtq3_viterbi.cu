// VTQ v3 CUDA Encoder: Contiguous `L = 256` 1024-State Tail-Biting Shift-Register (`nu = 10`, 9 history bits)
// Joint `(gain, 1024-state)` Dynamic Programming (`G <= 8`) + Exact 16-Stage `Block16-LDLQ`
// with 16-Step Causal Lookahead & Monotone Cholesky-Loss Full-256 Tail-Biting Refinement.
//
// Build:
//   nvcc -O3 -std=c++17 -arch=sm_120 -shared -Xcompiler -fPIC src/cuda/vtq3_viterbi.cu -o src/cuda/libvtq3_viterbi_sm_120.so
//
// Key properties:
//   1. Optimal 1024-state (`nu = 10`) Ungerboeck GF(2) subset label map matching CPU AVX2 (`vtq3_avx2_bench.c`)
//      and GPU Joint-1024 (`vtq3_matvec_bench.cu`) across `NL in {4, 8, 16}`:
//        NL == 16 -> p10(s & 0x3B6) | (p10(s & 0x050) << 1) | (p10(s & 0x2CE) << 2) | ((s & 1) << 3)
//        NL == 8  -> p10(s & 0x050) | (p10(s & 0x2CE) << 1) | ((s & 1) << 2)
//        NL == 4  -> p10(s & 0x2CE) | ((s & 1) << 1)
//   2. Native little-endian `uint16_t` `c[16][16][2]` (`c[g][r][0] = cols 8..15`, `c[g][r][1] = cols 0..7`),
//      so `*(const uint16_t*)c[g][r] == (cols_0_7 << 8) | cols_8_15` directly on both x86-64 and CUDA!
//   3. Exact Bellman joint `(gain, 1024-state)` subblock transition (`subblock16_dp<NL, GOSSET>`, 128 threads = 1 CTA/row):
//      Evaluates all `gi = 0..G-1` (`G <= 8`) over the 16 columns of subblock `g` and packs
//      `(s_prev_12b, win_gi_4b, c16_msb, u0_16_msb)` into a single `uint64_t` per `(g, s_end)`!

#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>

#define VTQ3_INF 1.0e30f

__device__ __forceinline__ int shift_pred(int s_next, int top, int& c_bit) {
    c_bit = s_next & 1;
    return (top << 9) | (s_next >> 1);
}

__device__ __forceinline__ int p10_dev(int x) {
    return __popc((unsigned int)(x & 0x3FF)) & 1;
}

template <int NL>
__device__ __forceinline__ int shift_lb_t(int s) {
    if constexpr (NL == 16) {
        return (s & 1) | (p10_dev(s & 0x2CE) << 1) | (p10_dev(s & 0x050) << 2) | (p10_dev(s & 0x3B6) << 3);
    }
    if constexpr (NL == 8) {
        return (s & 1) | (p10_dev(s & 0x2CE) << 1) | (p10_dev(s & 0x39C) << 2);
    }
    return (s & 1) | (p10_dev(s & 0x2CE) << 1);
}

__device__ __forceinline__ float best_branch_err(float xt, float gain, const float* tab_lb, int U) {
    float best = VTQ3_INF;
    for (int u = 0; u < U; u++) {
        float d = xt - gain * tab_lb[u];
        float e = d * d;
        if (e < best) best = e;
    }
    return best;
}

__device__ __forceinline__ float best_branch_err_par(float xt, float gain, const float* tab_lb, int U, int par) {
    float best = VTQ3_INF;
    for (int u = par; u < U; u += 2) {
        float d = xt - gain * tab_lb[u];
        float e = d * d;
        if (e < best) best = e;
    }
    return best;
}

__device__ __forceinline__ int best_u_for_label(
    float xt, float gain, const float* tab_lb, int U, int par_mode, int req_par, float& best_val)
{
    float best = VTQ3_INF;
    int bu = par_mode ? req_par : 0;
    float bv = gain * tab_lb[bu];
    int step = par_mode ? 2 : 1;
    for (int u = bu; u < U; u += step) {
        float v = gain * tab_lb[u];
        float d = xt - v;
        float e = d * d;
        if (e < best) {
            best = e;
            bu = u;
            bv = v;
        }
    }
    best_val = bv;
    return bu;
}

// =================================================================================================
// Shared Workspace per 128-Thread CTA (`1024` states = `8` states per thread; trimmed per `GOSSET` for occupancy!)
// =================================================================================================
template <int NL, int GOSSET>
struct alignas(16) Sub16CtaSmem {
    float    base[1024];                                  //  4096 B: entering path metric at start of 16-col subblock
    float    best_end[1024];                              //  4096 B: min over gi of ending path metric
    float    dp0[1024];                                   //  4096 B
    float    dp1[(GOSSET == 2) ? 2 : 1024];               //  4096 B (unused by GOSSET == 2; 2 floats keeps 8B alignment!)
    alignas(8) float mid[(GOSSET == 0) ? 1 : 4][(GOSSET == 0) ? 2 : 1024]; // 16384 B (8B aligned for `uint64_t* s_cum`)
    uint32_t bp32[1024];                                  //  4096 B (step backpointers for current gi)
    uint32_t lo_pack[(GOSSET == 1) ? 1024 : 2];           //  4096 B (used only by GOSSET == 1)
    uint8_t  mwin[(GOSSET == 1 || GOSSET == 3) ? 1024 : 8]; // 1024 B
    float    err[2][NL][2];                               // <= 256 B
    float    red_v[4];
    int      red_s[4];
    int      min_s;
};

template <int NL, int GOSSET>
__device__ __forceinline__ int cta_argmin_base(Sub16CtaSmem<NL, GOSSET>& ws, int tid) {
    float best_v = VTQ3_INF;
    int best_s = 0;
    #pragma unroll
    for (int k = 0; k < 8; k++) {
        int s = tid + 128 * k;
        float v = ws.base[s];
        if (v < best_v) { best_v = v; best_s = s; }
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        float ov = __shfl_xor_sync(0xffffffffu, best_v, off);
        int os   = __shfl_xor_sync(0xffffffffu, best_s, off);
        if (ov < best_v || (ov == best_v && os < best_s)) { best_v = ov; best_s = os; }
    }
    if ((tid & 31) == 0) {
        ws.red_v[tid >> 5] = best_v;
        ws.red_s[tid >> 5] = best_s;
    }
    __syncthreads();
    if (tid == 0) {
        float bv = ws.red_v[0];
        int bs   = ws.red_s[0];
        for (int w = 1; w < 4; w++) {
            float ov = ws.red_v[w];
            int os   = ws.red_s[w];
            if (ov < bv || (ov == bv && os < bs)) { bv = ov; bs = os; }
        }
        ws.min_s = bs;
    }
    __syncthreads();
    return ws.min_s;
}

// Advance 16 columns `x16[0..15]`, `w16[0..15]` from `ws.base[0..1023]`, evaluating gains `gi_beg..gi_end-1`.
// Updates `ws.base[0..1023]` in-place and writes `out_trans[s_end] = s_prev_12b | (win_gi << 12) | (c16 << 16) | (u16 << 32)`!
template <int NL, int GOSSET>
__device__ __forceinline__ void subblock16_dp(
    const float* __restrict__ x16,
    const float* __restrict__ w16,
    const float* __restrict__ s_tab,
    const float* __restrict__ s_gains,
    int U, int gi_beg, int gi_end, int tid,
    const int p0[8], const int p1[8], const int lb_reg[8],
    Sub16CtaSmem<NL, GOSSET>& ws,
    uint64_t* __restrict__ out_trans)
{
    uint64_t best_tr[8];
    #pragma unroll
    for (int k = 0; k < 8; k++) {
        ws.best_end[tid + 128 * k] = VTQ3_INF;
        best_tr[k] = 0ull;
    }
    __syncthreads();

    for (int gi = gi_beg; gi < gi_end; gi++) {
        const float g_val = s_gains[gi];

        if constexpr (GOSSET == 0) {
            float* cur = ws.dp0;
            float* nxt = ws.dp1;
            uint32_t bp16[8] = {0, 0, 0, 0, 0, 0, 0, 0};
            #pragma unroll
            for (int k = 0; k < 8; k++) {
                int s = tid + 128 * k;
                cur[s] = ws.base[s];
            }
            __syncthreads();

            for (int j = 0; j < 16; j++) {
                if (tid < NL) {
                    ws.err[0][tid][0] = w16[j] * best_branch_err(x16[j], g_val, s_tab + tid * U, U);
                }
                __syncthreads();
                #pragma unroll
                for (int k = 0; k < 8; k++) {
                    int s = tid + 128 * k;
                    float v0 = cur[p0[k]];
                    float v1 = cur[p1[k]];
                    uint32_t top = (v1 < v0) ? 1u : 0u;
                    nxt[s] = (top ? v1 : v0) + ws.err[0][lb_reg[k]][0];
                    bp16[k] |= (top << j);
                }
                __syncthreads();
                float* tmp = cur; cur = nxt; nxt = tmp;
            }

            #pragma unroll
            for (int k = 0; k < 8; k++) ws.bp32[tid + 128 * k] = bp16[k];
            __syncthreads();

            #pragma unroll
            for (int k = 0; k < 8; k++) {
                int s_end = tid + 128 * k;
                float cost = cur[s_end];
                if (cost < ws.best_end[s_end]) {
                    ws.best_end[s_end] = cost;
                    int st_t = s_end;
                    uint32_t c16 = 0u;
                    #pragma unroll
                    for (int j = 15; j >= 0; j--) {
                        int top = (ws.bp32[st_t] >> j) & 1;
                        int cb = 0;
                        st_t = shift_pred(st_t, top, cb);
                        c16 |= (uint32_t(cb) << (15 - j));
                    }
                    best_tr[k] = uint64_t(st_t & 0xFFF)
                               | (uint64_t(gi & 0xF) << 12)
                               | (uint64_t(c16 & 0xFFFFu) << 16);
                }
            }
            __syncthreads();
        } else if constexpr (GOSSET == 1) {
            for (int hc = 0; hc < 2; hc++) {
                if (hc == 0) {
                    #pragma unroll
                    for (int k = 0; k < 8; k++) {
                        int s = tid + 128 * k;
                        ws.dp0[s] = ws.base[s];
                    }
                    __syncthreads();
                }

                uint32_t bp_m0[8] = {0, 0, 0, 0, 0, 0, 0, 0};
                uint32_t bp_m1[8] = {0, 0, 0, 0, 0, 0, 0, 0};

                for (int mode = 0; mode < 2; mode++) {
                    #pragma unroll
                    for (int k = 0; k < 8; k++) {
                        int s = tid + 128 * k;
                        ws.mid[0][s] = ws.dp0[s];
                        ws.mid[1][s] = VTQ3_INF;
                    }
                    __syncthreads();

                    for (int j = 0; j < 4; j++) {
                        const int c_ev = 8 * hc + 2 * j;
                        const int c_od = c_ev + 1;
                        for (int idx = tid; idx < 2 * NL * 2; idx += 128) {
                            int sp = idx / (NL * 2);
                            int rem = idx % (NL * 2);
                            int lb = rem >> 1, pr = rem & 1;
                            int cc = sp ? c_od : c_ev;
                            ws.err[sp][lb][pr] = w16[cc] * best_branch_err_par(x16[cc], g_val, s_tab + lb * U, U, pr);
                        }
                        __syncthreads();

                        float m_tmp[8][4];
                        #pragma unroll
                        for (int k = 0; k < 8; k++) {
                            int lb = lb_reg[k];
                            uint32_t ev_bits = 0u;
                            #pragma unroll
                            for (int pn = 0; pn < 2; pn++) {
                                #pragma unroll
                                for (int bj = 0; bj < 2; bj++) {
                                    int po = pn ^ bj;
                                    float v0 = ws.mid[po][p0[k]];
                                    float v1 = ws.mid[po][p1[k]];
                                    int top = (v1 < v0) ? 1 : 0;
                                    m_tmp[k][2 * pn + bj] = (top ? v1 : v0) + ws.err[0][lb][bj];
                                    ev_bits |= (uint32_t(top) << (2 * pn + bj));
                                }
                            }
                            if (mode == 0) bp_m0[k] |= (ev_bits << (8 * j));
                            else           bp_m1[k] |= (ev_bits << (8 * j));
                        }
                        __syncthreads();
                        #pragma unroll
                        for (int k = 0; k < 8; k++) {
                            int s = tid + 128 * k;
                            ws.mid[0][s] = m_tmp[k][0];
                            ws.mid[1][s] = m_tmp[k][1];
                            ws.mid[2][s] = m_tmp[k][2];
                            ws.mid[3][s] = m_tmp[k][3];
                        }
                        __syncthreads();

                        float d_tmp[8][2];
                        #pragma unroll
                        for (int k = 0; k < 8; k++) {
                            int lb = lb_reg[k];
                            uint32_t od_bits = 0u;
                            #pragma unroll
                            for (int pn = 0; pn < 2; pn++) {
                                float best = VTQ3_INF;
                                int best_bj = 0, best_top = 0;
                                #pragma unroll
                                for (int bj = 0; bj < 2; bj++) {
                                    float e_od = ws.err[1][lb][bj ^ mode];
                                    float v0 = ws.mid[2 * pn + bj][p0[k]] + e_od;
                                    if (v0 < best) { best = v0; best_bj = bj; best_top = 0; }
                                    float v1 = ws.mid[2 * pn + bj][p1[k]] + e_od;
                                    if (v1 < best) { best = v1; best_bj = bj; best_top = 1; }
                                }
                                d_tmp[k][pn] = best;
                                od_bits |= (uint32_t((best_top << 1) | best_bj) << (2 * pn));
                            }
                            if (mode == 0) bp_m0[k] |= (od_bits << (8 * j + 4));
                            else           bp_m1[k] |= (od_bits << (8 * j + 4));
                        }
                        __syncthreads();
                        #pragma unroll
                        for (int k = 0; k < 8; k++) {
                            int s = tid + 128 * k;
                            ws.mid[0][s] = d_tmp[k][0];
                            ws.mid[1][s] = d_tmp[k][1];
                        }
                        __syncthreads();
                    }

                    if (mode == 0) {
                        #pragma unroll
                        for (int k = 0; k < 8; k++) {
                            int s = tid + 128 * k;
                            ws.dp1[s] = ws.mid[0][s];
                        }
                        __syncthreads();
                    } else {
                        #pragma unroll
                        for (int k = 0; k < 8; k++) {
                            int s = tid + 128 * k;
                            float vm0 = ws.dp1[s];
                            float vm1 = ws.mid[0][s];
                            int win = (vm1 < vm0) ? 1 : 0;
                            ws.dp0[s]  = win ? vm1 : vm0;
                            ws.mwin[s] = (uint8_t)win;
                            ws.bp32[s] = bp_m0[k];
                            reinterpret_cast<uint32_t*>(ws.mid[2])[s] = bp_m1[k];
                        }
                        __syncthreads();
                    }
                }

                uint32_t* bp_mode1 = reinterpret_cast<uint32_t*>(ws.mid[2]);
                uint32_t pack_h[8];
                #pragma unroll
                for (int k = 0; k < 8; k++) {
                    int s_h = tid + 128 * k;
                    int mode = ws.mwin[s_h];
                    const uint32_t* bp_arr = mode ? bp_mode1 : ws.bp32;
                    int st_t = s_h;
                    int par  = 0;
                    uint32_t c8 = 0u, u8 = 0u;
                    #pragma unroll
                    for (int j = 3; j >= 0; j--) {
                        uint32_t b_od = (bp_arr[st_t] >> (8 * j)) & 0xFFu;
                        int od_pair = (b_od >> (4 + 2 * par)) & 3;
                        int bj = od_pair & 1, top_od = (od_pair >> 1) & 1;
                        int c_od = 0;
                        int st_mid = shift_pred(st_t, top_od, c_od);

                        uint32_t b_ev = (bp_arr[st_mid] >> (8 * j)) & 0xFFu;
                        int top_ev = (b_ev >> (2 * par + bj)) & 1;
                        int c_ev = 0;
                        int st_prv = shift_pred(st_mid, top_ev, c_ev);

                        c8 |= (uint32_t(c_ev) << (7 - 2 * j)) | (uint32_t(c_od) << (6 - 2 * j));
                        u8 |= (uint32_t(bj) << (7 - 2 * j))   | (uint32_t(bj ^ mode) << (6 - 2 * j));
                        par ^= bj;
                        st_t = st_prv;
                    }
                    pack_h[k] = uint32_t(st_t & 0xFFF) | (c8 << 12) | (u8 << 20);
                }
                __syncthreads();

                if (hc == 0) {
                    #pragma unroll
                    for (int k = 0; k < 8; k++) ws.lo_pack[tid + 128 * k] = pack_h[k];
                    __syncthreads();
                } else {
                    #pragma unroll
                    for (int k = 0; k < 8; k++) {
                        int s_end = tid + 128 * k;
                        float cost = ws.dp0[s_end];
                        if (cost < ws.best_end[s_end]) {
                            ws.best_end[s_end] = cost;
                            uint32_t hi = pack_h[k];
                            uint32_t lo = ws.lo_pack[hi & 0xFFFu];
                            uint32_t s_prv16 = lo & 0xFFFu;
                            uint32_t c16 = (((lo >> 12) & 0xFFu) << 8) | ((hi >> 12) & 0xFFu);
                            uint32_t u16 = (((lo >> 20) & 0xFFu) << 8) | ((hi >> 20) & 0xFFu);
                            best_tr[k] = uint64_t(s_prv16)
                                       | (uint64_t(gi & 0xF) << 12)
                                       | (uint64_t(c16) << 16)
                                       | (uint64_t(u16) << 32);
                        }
                    }
                    __syncthreads();
                }
            }
        } else {
            uint64_t* s_cum = reinterpret_cast<uint64_t*>(ws.mid[2]); // 1024 * 8 = 8192 B (`mid[2]` + `mid[3]`)
            #pragma unroll
            for (int k = 0; k < 8; k++) {
                int s = tid + 128 * k;
                ws.dp0[s] = ws.base[s];
                s_cum[s]  = uint64_t(s & 0xFFF);
            }
            __syncthreads();

            for (int q4 = 0; q4 < 4; q4++) {
                if constexpr (GOSSET == 2) {
                    #pragma unroll
                    for (int k = 0; k < 8; k++) {
                        int s = tid + 128 * k;
                        ws.mid[0][s] = ws.dp0[s];
                        ws.mid[1][s] = VTQ3_INF;
                    }
                    __syncthreads();
                    uint32_t bp_q[8] = {0, 0, 0, 0, 0, 0, 0, 0};
                    for (int t = 0; t < 4; t++) {
                        int cc = 4 * q4 + t;
                        for (int idx = tid; idx < NL * 2; idx += 128) {
                            ws.err[0][idx >> 1][idx & 1] = w16[cc] * best_branch_err_par(x16[cc], g_val, s_tab + (idx >> 1) * U, U, idx & 1);
                        }
                        __syncthreads();
                        float out_p[8][2];
                        #pragma unroll
                        for (int k = 0; k < 8; k++) {
                            int lb = lb_reg[k];
                            uint32_t sb = 0u;
                            #pragma unroll
                            for (int pn = 0; pn < 2; pn++) {
                                float best = VTQ3_INF; int bu0 = 0, btop = 0;
                                #pragma unroll
                                for (int u0 = 0; u0 < 2; u0++) {
                                    int po = pn ^ u0;
                                    float eb = ws.err[0][lb][u0];
                                    float v0 = ws.mid[po][p0[k]] + eb;
                                    if (v0 < best) { best = v0; bu0 = u0; btop = 0; }
                                    float v1 = ws.mid[po][p1[k]] + eb;
                                    if (v1 < best) { best = v1; bu0 = u0; btop = 1; }
                                }
                                out_p[k][pn] = best;
                                sb |= (uint32_t((btop << 1) | bu0) << (2 * pn));
                            }
                            bp_q[k] |= (sb << (4 * t));
                        }
                        __syncthreads();
                        #pragma unroll
                        for (int k = 0; k < 8; k++) {
                            int s = tid + 128 * k;
                            ws.mid[0][s] = out_p[k][0];
                            ws.mid[1][s] = out_p[k][1];
                        }
                        __syncthreads();
                    }
                    #pragma unroll
                    for (int k = 0; k < 8; k++) {
                        int s = tid + 128 * k;
                        ws.dp0[s] = ws.mid[0][s];
                        ws.bp32[s] = bp_q[k];
                    }
                    __syncthreads();
                    uint64_t nxt_cum[8];
                    #pragma unroll
                    for (int k = 0; k < 8; k++) {
                        int st_t = tid + 128 * k, par = 0;
                        uint32_t c4 = 0u, u4 = 0u;
                        for (int t = 3; t >= 0; t--) {
                            int pair = (ws.bp32[st_t] >> (4 * t + 2 * par)) & 3;
                            int u0 = pair & 1, top = (pair >> 1) & 1, cb = 0;
                            st_t = shift_pred(st_t, top, cb);
                            c4 |= (uint32_t(cb) << (3 - t));
                            u4 |= (uint32_t(u0) << (3 - t));
                            par ^= u0;
                        }
                        uint64_t prev_c = s_cum[st_t];
                        uint32_t s_p0 = uint32_t(prev_c & 0xFFFull);
                        uint32_t c_ac = (uint32_t((prev_c >> 16) & 0xFFFFull) << 4) | c4;
                        uint32_t u_ac = (uint32_t((prev_c >> 32) & 0xFFFFull) << 4) | u4;
                        nxt_cum[k] = uint64_t(s_p0) | (uint64_t(c_ac) << 16) | (uint64_t(u_ac) << 32);
                    }
                    __syncthreads();
                    #pragma unroll
                    for (int k = 0; k < 8; k++) s_cum[tid + 128 * k] = nxt_cum[k];
                    __syncthreads();
                } else {
                    uint32_t bp_pr[2][8] = {{0}, {0}};
                    for (int pr = 0; pr < 2; pr++) {
                        #pragma unroll
                        for (int k = 0; k < 8; k++) ws.mid[0][tid + 128 * k] = ws.dp0[tid + 128 * k];
                        __syncthreads();
                        for (int t = 0; t < 4; t++) {
                            int cc = 4 * q4 + t;
                            if (tid < NL) ws.err[0][tid][0] = w16[cc] * best_branch_err_par(x16[cc], g_val, s_tab + tid * U, U, pr);
                            __syncthreads();
                            float nxt_v[8];
                            #pragma unroll
                            for (int k = 0; k < 8; k++) {
                                float v0 = ws.mid[0][p0[k]], v1 = ws.mid[0][p1[k]];
                                uint32_t top = (v1 < v0) ? 1u : 0u;
                                nxt_v[k] = (top ? v1 : v0) + ws.err[0][lb_reg[k]][0];
                                bp_pr[pr][k] |= (top << t);
                            }
                            __syncthreads();
                            #pragma unroll
                            for (int k = 0; k < 8; k++) ws.mid[0][tid + 128 * k] = nxt_v[k];
                            __syncthreads();
                        }
                        if (pr == 0) {
                            #pragma unroll
                            for (int k = 0; k < 8; k++) ws.dp1[tid + 128 * k] = ws.mid[0][tid + 128 * k];
                            __syncthreads();
                        } else {
                            #pragma unroll
                            for (int k = 0; k < 8; k++) {
                                int s = tid + 128 * k;
                                float v0 = ws.dp1[s], v1 = ws.mid[0][s];
                                int win = (v1 < v0) ? 1 : 0;
                                ws.dp0[s]  = win ? v1 : v0;
                                ws.mwin[s] = (uint8_t)win;
                                ws.bp32[s] = bp_pr[0][k] | (bp_pr[1][k] << 8);
                            }
                            __syncthreads();
                        }
                    }
                    uint64_t nxt_cum[8];
                    #pragma unroll
                    for (int k = 0; k < 8; k++) {
                        int s_end_q = tid + 128 * k;
                        int pr = ws.mwin[s_end_q];
                        int st_t = s_end_q;
                        uint32_t c4 = 0u, u4 = pr ? 0xFu : 0x0u;
                        for (int t = 3; t >= 0; t--) {
                            int top = (ws.bp32[st_t] >> (8 * pr + t)) & 1, cb = 0;
                            st_t = shift_pred(st_t, top, cb);
                            c4 |= (uint32_t(cb) << (3 - t));
                        }
                        uint64_t prev_c = s_cum[st_t];
                        uint32_t s_p0 = uint32_t(prev_c & 0xFFFull);
                        uint32_t c_ac = (uint32_t((prev_c >> 16) & 0xFFFFull) << 4) | c4;
                        uint32_t u_ac = (uint32_t((prev_c >> 32) & 0xFFFFull) << 4) | u4;
                        nxt_cum[k] = uint64_t(s_p0) | (uint64_t(c_ac) << 16) | (uint64_t(u_ac) << 32);
                    }
                    __syncthreads();
                    #pragma unroll
                    for (int k = 0; k < 8; k++) s_cum[tid + 128 * k] = nxt_cum[k];
                    __syncthreads();
                }
            }
            #pragma unroll
            for (int k = 0; k < 8; k++) {
                int s_end = tid + 128 * k;
                float cost = ws.dp0[s_end];
                if (cost < ws.best_end[s_end]) {
                    ws.best_end[s_end] = cost;
                    best_tr[k] = s_cum[s_end] | (uint64_t(gi & 0xF) << 12);
                }
            }
            __syncthreads();
        }
    }

    #pragma unroll
    for (int k = 0; k < 8; k++) {
        int s = tid + 128 * k;
        ws.base[s] = ws.best_end[s];
        if (out_trans) out_trans[s] = best_tr[k];
    }
    __syncthreads();
}

// Reconstruct 16 columns of subblock `g` from `uint64_t tr` (`s_prev_12b | (gi<<12) | (c16<<16) | (u16<<32)`)
// Writes native little-endian `uint16_t` `c2_slot`: `[0] = cols 8..15`, `[1] = cols 0..7`!
template <int NL, int GOSSET>
__device__ __forceinline__ void emit_subblock16_from_trans(
    uint64_t tr,
    const float* __restrict__ x16,
    const float* __restrict__ s_tab,
    const float* __restrict__ s_gains,
    int U, int tid,
    float* __restrict__ rec16,
    uint8_t* __restrict__ g_sub_slot,
    uint8_t* __restrict__ c2_slot,
    uint8_t* __restrict__ u16_slot,
    uint8_t* __restrict__ lb16_slot)
{
    const uint32_t s_prv = uint32_t(tr & 0xFFFull);
    const uint32_t gi    = uint32_t((tr >> 12) & 0xFull);
    const uint32_t c16   = uint32_t((tr >> 16) & 0xFFFFull);
    const uint32_t u16p  = uint32_t((tr >> 32) & 0xFFFFull);
    const uint32_t w25   = ((s_prv & 0x1FFu) << 16) | c16;
    const float g_val    = s_gains[gi];

    if (tid == 0) {
        if (g_sub_slot) *g_sub_slot = (uint8_t)gi;
        if (c2_slot) {
            c2_slot[0] = (uint8_t)(c16 & 0xFFu);          // cols 8..15 (low byte of native uint16_t)
            c2_slot[1] = (uint8_t)((c16 >> 8) & 0xFFu);   // cols 0..7  (high byte of native uint16_t)
        }
    }
    if (tid < 16) {
        int st = (int)((w25 >> (15 - tid)) & 0x3FFu);
        int lb = shift_lb_t<NL>(st);
        int req_p = (GOSSET != 0) ? (int)((u16p >> (15 - tid)) & 1u) : 0;
        float bv = 0.0f;
        int bu = best_u_for_label(x16[tid], g_val, s_tab + lb * U, U, (GOSSET != 0) ? 1 : 0, req_p, bv);
        if (rec16)     rec16[tid]     = bv;
        if (u16_slot)  u16_slot[tid]  = (uint8_t)bu;
        if (lb16_slot) lb16_slot[tid] = (uint8_t)lb;
    }
    __syncthreads();
}

// =================================================================================================
// 1. Contiguous `L = 256` 2-Pass Tail-Biting Joint `(Gain, 1024-State)` Viterbi Kernel (`128` threads/CTA)
// =================================================================================================
template <int NL, int GOSSET>
__global__ void __launch_bounds__(128) vtq3_viterbi_256_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w,
    int w_stride,
    const float* __restrict__ tab,
    const float* __restrict__ gains,
    const uint8_t* __restrict__ g_sub_fix,
    uint64_t* __restrict__ ws_trans,        // [N, 16, 1024]
    float* __restrict__ rec,
    uint8_t* __restrict__ g_sub_out,
    uint8_t* __restrict__ c_out,
    uint8_t* __restrict__ u_out,
    uint8_t* __restrict__ lb_out,
    int N, int U, int G)
{
    __shared__ Sub16CtaSmem<NL, GOSSET> s_ws;
    __shared__ float s_tab[512];
    __shared__ float s_gains[8];
    __shared__ uint64_t s_win_tr[16];
    __shared__ int s_seam_sh;

    const int tid = threadIdx.x;
    const int n   = blockIdx.x;
    if (n >= N) return;

    for (int i = tid; i < NL * U; i += 128) s_tab[i] = tab[i];
    if (tid < G) s_gains[tid] = gains[tid];
    __syncthreads();

    const float* xn = x + (size_t)n * 256;
    const float* wn = w + (size_t)n * w_stride;
    const uint8_t* gfix_n = g_sub_fix ? (g_sub_fix + (size_t)n * 16) : nullptr;
    uint64_t* tr_n = ws_trans + (size_t)n * 16 * 1024;

    int p0[8], p1[8], lb_reg[8];
    #pragma unroll
    for (int k = 0; k < 8; k++) {
        int s = tid + 128 * k, dc = 0;
        p0[k] = shift_pred(s, 0, dc);
        p1[k] = shift_pred(s, 1, dc);
        lb_reg[k] = shift_lb_t<NL>(s);
    }

    int s_seam = 0;
    for (int pass = 0; pass < 2; pass++) {
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            int s = tid + 128 * k;
            s_ws.base[s] = (pass == 0 || s == s_seam) ? 0.0f : VTQ3_INF;
        }
        __syncthreads();

        for (int step_g = 0; step_g < 16; step_g++) {
            int g = (pass == 0) ? step_g : ((step_g + 8) & 15);
            int gi0 = gfix_n ? (int)gfix_n[g] : 0;
            int gi1 = gfix_n ? (gi0 + 1)      : G;
            subblock16_dp<NL, GOSSET>(
                xn + 16 * g, wn + 16 * g, s_tab, s_gains, U, gi0, gi1, tid,
                p0, p1, lb_reg, s_ws, tr_n + (size_t)g * 1024);
        }

        if (pass == 0) {
            int best_s = cta_argmin_base(s_ws, tid);
            if (tid == 0) {
                int st_cur = best_s;
                for (int g = 15; g >= 8; g--) {
                    st_cur = (int)(tr_n[(size_t)g * 1024 + st_cur] & 0xFFFull);
                }
                s_seam_sh = st_cur;
            }
            __syncthreads();
            s_seam = s_seam_sh;
        } else {
            if (tid == 0) {
                int st_cur = s_seam;
                for (int step_g = 15; step_g >= 0; step_g--) {
                    int g = (step_g + 8) & 15;
                    uint64_t tr = tr_n[(size_t)g * 1024 + st_cur];
                    s_win_tr[g] = tr;
                    st_cur = (int)(tr & 0xFFFull);
                }
            }
            __syncthreads();
        }
    }

    for (int g = 0; g < 16; g++) {
        emit_subblock16_from_trans<NL, GOSSET>(
            s_win_tr[g], xn + 16 * g, s_tab, s_gains, U, tid,
            rec       ? (rec       + (size_t)n * 256 + 16 * g) : nullptr,
            g_sub_out ? (g_sub_out + (size_t)n * 16  + g)      : nullptr,
            c_out     ? (c_out     + ((size_t)n * 16 + g) * 2) : nullptr,
            u_out     ? (u_out     + (size_t)n * 256 + 16 * g) : nullptr,
            lb_out    ? (lb_out    + (size_t)n * 256 + 16 * g) : nullptr);
    }
}

// =================================================================================================
// 2. Fused Single-Launch 16-Stage `Block16-LDLQ` Kernel (`1024` states, `128` threads/CTA = 1 row)
//    Fuses seam seeding (`{14, 15, 0}`) + all 16 causal lookahead stages (`g = 0..15`) in shared `s_w[256]`,
//    eliminating 16 kernel launches and all intermediate global memory reloads per 256-col block!
// =================================================================================================
template <int NL, int GOSSET>
__global__ void __launch_bounds__(128) vtq3_ldlq_fused_kernel(
    float* __restrict__ w_work,
    float* __restrict__ w_target_norm,
    const float* __restrict__ d_sb,
    const float* __restrict__ dw_blk,
    const float* __restrict__ U_blk,
    const float* __restrict__ U_blk_inv,
    const float* __restrict__ tab,
    const float* __restrict__ gains,
    uint64_t* __restrict__ ws_trans,    // [M, 16, 1024] (uses first 2 * 1024 slots per row)
    float* __restrict__ G_out,
    uint8_t* __restrict__ g_sub_out,
    uint8_t* __restrict__ c_out,
    uint8_t* __restrict__ u_out,
    int M, int U, int G)
{
    __shared__ Sub16CtaSmem<NL, GOSSET> s_ws;
    __shared__ float s_tab[512];
    __shared__ float s_gains[8];
    __shared__ float s_w[256];
    __shared__ float s_dw[256];
    __shared__ float s_Uinv[16][16];
    __shared__ float s_x16[16];
    __shared__ float s_rec16[16];
    __shared__ float s_err16[16];
    __shared__ int   s_win_sh;

    const int tid = threadIdx.x;
    const int r   = blockIdx.x;
    if (r >= M) return;

    for (int i = tid; i < NL * U; i += 128) s_tab[i] = tab[i];
    if (tid < G) s_gains[tid] = gains[tid];
    for (int i = tid; i < 256; i += 128) {
        s_w[i]  = w_work[(size_t)r * 256 + i];
        s_dw[i] = dw_blk[i];
    }
    __syncthreads();

    const float d     = d_sb[r];
    const float inv_d = 1.0f / fmaxf(d, 1e-10f);
    uint64_t* tr0 = ws_trans + (size_t)r * 16 * 1024;
    uint64_t* tr1 = tr0 + 1024;

    int p0[8], p1[8], lb_reg[8];
    #pragma unroll
    for (int k = 0; k < 8; k++) {
        int s = tid + 128 * k, dc = 0;
        p0[k] = shift_pred(s, 0, dc);
        p1[k] = shift_pred(s, 1, dc);
        lb_reg[k] = shift_lb_t<NL>(s);
        s_ws.base[s] = 0.0f;
    }
    __syncthreads();

    // Stage 0: Seed tail-biting seam across `{14, 15, 0}`
    const int g_seq[3] = {14, 15, 0};
    for (int si = 0; si < 3; si++) {
        int g = g_seq[si];
        if (tid < 16) s_x16[tid] = s_w[16 * g + tid] * inv_d;
        __syncthreads();
        subblock16_dp<NL, GOSSET>(
            s_x16, s_dw + 16 * g, s_tab, s_gains, U, 0, G, tid,
            p0, p1, lb_reg, s_ws, (si == 2) ? tr0 : nullptr);
    }

    int best_seed = cta_argmin_base(s_ws, tid);
    if (tid == 0) s_win_sh = (int)(tr0[best_seed] & 0xFFFull);
    __syncthreads();
    int s_init = s_win_sh;

    // 2-Round Post-Cholesky Seam Alignment:
    //   - Round 0 finds the true optimal ending state `s_end_free` at column 255 AFTER all 240 columns of causal
    //     Cholesky feedback (`g = 0..14`), avoiding forcing uncompensated columns `240..255` into a pre-Cholesky `s_init`!
    //   - If `s_end_free == s_init`, Round 0 is already valid tail-biting and exits immediately; otherwise Round 1
    //     re-runs causal Cholesky starting & ending at `s_init = s_end_free` so boundary adjustment happens at `g = 0`
    //     where 240 downstream columns absorb 100% of the Cholesky feedback!
    for (int rd = 0; rd < 2; rd++) {
        if (rd == 1) {
            for (int i = tid; i < 256; i += 128) s_w[i] = w_work[(size_t)r * 256 + i];
            __syncthreads();
        }
        int s_cur = s_init;

        for (int g = 0; g < 16; g++) {
            const int s0  = 16 * g;
            const int s1  = s0 + 16;
            const int rem = 256 - s1;

            #pragma unroll
            for (int k = 0; k < 8; k++) {
                int s = tid + 128 * k;
                s_ws.base[s] = (s == s_cur) ? 0.0f : VTQ3_INF;
            }
            if (rem > 0) {
                for (int i = tid; i < 256; i += 128) {
                    int kr = i >> 4, kc = i & 15;
                    s_Uinv[kr][kc] = U_blk_inv[(s0 + kr) * 256 + (s0 + kc)];
                }
            }
            if (tid < 16) {
                float xn = s_w[s0 + tid] * inv_d;
                s_x16[tid] = xn;
                w_target_norm[(size_t)r * 256 + s0 + tid] = xn;
            }
            __syncthreads();

            subblock16_dp<NL, GOSSET>(
                s_x16, s_dw + s0, s_tab, s_gains, U, 0, G, tid,
                p0, p1, lb_reg, s_ws, tr0);

            int s_win = s_init;
            if (g < 15) {
                if (tid < 16) s_x16[tid] = s_w[s1 + tid] * inv_d;
                __syncthreads();
                subblock16_dp<NL, GOSSET>(
                    s_x16, s_dw + s1, s_tab, s_gains, U, 0, G, tid,
                    p0, p1, lb_reg, s_ws, tr1);

                if (g == 14 && rd == 1) {
                    if (tid == 0) s_win_sh = (int)(tr1[s_init] & 0xFFFull);
                    __syncthreads();
                    s_win = s_win_sh;
                } else {
                    int best_s = cta_argmin_base(s_ws, tid);
                    if (tid == 0) s_win_sh = (int)(tr1[best_s] & 0xFFFull);
                    __syncthreads();
                    s_win = s_win_sh;
                }
                if (tid < 16) s_x16[tid] = w_target_norm[(size_t)r * 256 + s0 + tid];
                __syncthreads();
            } else if (rd == 0) {
                s_win = cta_argmin_base(s_ws, tid);
            }

            if (rd == 0 && g == 15) {
                if (s_win == s_init) {
                    uint64_t win_tr = tr0[s_win];
                    emit_subblock16_from_trans<NL, GOSSET>(
                        win_tr, s_x16, s_tab, s_gains, U, tid,
                        s_rec16,
                        g_sub_out + (size_t)r * 16 + g,
                        c_out ? (c_out + ((size_t)r * 16 + g) * 2) : nullptr,
                        u_out ? (u_out + (size_t)r * 256 + s0)     : nullptr,
                        nullptr);
                    if (tid < 16) G_out[(size_t)r * 256 + s0 + tid] = s_rec16[tid];
                    return;
                }
                s_init = s_win;
                break;
            }

            uint64_t win_tr = tr0[s_win];
            s_cur = s_win;

            emit_subblock16_from_trans<NL, GOSSET>(
                win_tr, s_x16, s_tab, s_gains, U, tid,
                s_rec16,
                g_sub_out + (size_t)r * 16 + g,
                c_out ? (c_out + ((size_t)r * 16 + g) * 2) : nullptr,
                u_out ? (u_out + (size_t)r * 256 + s0)     : nullptr,
                nullptr);

            if (tid < 16) {
                G_out[(size_t)r * 256 + s0 + tid] = s_rec16[tid];
            }
            if (rem > 0) {
                if (tid < 32) {
                    float diff_lane = (tid < 16) ? (s_w[s0 + tid] - d * s_rec16[tid]) : 0.0f;
                    float err_j = 0.0f;
                    #pragma unroll
                    for (int k = 0; k < 16; k++) {
                        float dk = __shfl_sync(0xffffffffu, diff_lane, k);
                        if (tid < 16) err_j = fmaf(dk, s_Uinv[k][tid], err_j);
                    }
                    if (tid < 16) s_err16[tid] = err_j;
                }
                __syncthreads();

                for (int col = s1 + tid; col < 256; col += 128) {
                    float acc = 0.0f;
                    #pragma unroll
                    for (int k = 0; k < 16; k++) {
                        acc = fmaf(s_err16[k], U_blk[(s0 + k) * 256 + col], acc);
                    }
                    s_w[col] -= acc;
                }
                __syncthreads();
            }
        }
    }
}

// Per-row exact Cholesky loss comparison `||(w_orig[r] - d_sb[r] * G[r]) @ U_blk_inv||_2^2` (`1` warp = 1 row)
__global__ void __launch_bounds__(256) vtq3_monotone_select_kernel(
    const float* __restrict__ w_orig,
    const float* __restrict__ d_sb,
    const float* __restrict__ U_blk_inv,
    float* __restrict__ G_best,
    uint8_t* __restrict__ g_sub_best,
    uint8_t* __restrict__ c_best,
    uint8_t* __restrict__ u_best,
    const float* __restrict__ G_cand,
    const uint8_t* __restrict__ g_sub_cand,
    const uint8_t* __restrict__ c_cand,
    const uint8_t* __restrict__ u_cand,
    int M)
{
    __shared__ float s_d0[8][256];
    __shared__ float s_d1[8][256];

    const int wid  = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int r    = blockIdx.x * 8 + wid;
    if (r >= M) return;

    const float d = d_sb[r];
    for (int i = lane; i < 256; i += 32) {
        float wo = w_orig[(size_t)r * 256 + i];
        s_d0[wid][i] = wo - d * G_best[(size_t)r * 256 + i];
        s_d1[wid][i] = wo - d * G_cand[(size_t)r * 256 + i];
    }
    __syncwarp();

    float l0 = 0.0f, l1 = 0.0f;
    for (int col = lane; col < 256; col += 32) {
        float a0 = 0.0f, a1 = 0.0f;
        for (int i = 0; i <= col; i++) {
            float u_ic = U_blk_inv[i * 256 + col];
            a0 = fmaf(s_d0[wid][i], u_ic, a0);
            a1 = fmaf(s_d1[wid][i], u_ic, a1);
        }
        l0 = fmaf(a0, a0, l0);
        l1 = fmaf(a1, a1, l1);
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        l0 += __shfl_xor_sync(0xffffffffu, l0, off);
        l1 += __shfl_xor_sync(0xffffffffu, l1, off);
    }

    if (l1 < l0) {
        for (int i = lane; i < 256; i += 32) {
            G_best[(size_t)r * 256 + i] = G_cand[(size_t)r * 256 + i];
            if (u_best && u_cand) u_best[(size_t)r * 256 + i] = u_cand[(size_t)r * 256 + i];
        }
        if (lane < 16) {
            g_sub_best[(size_t)r * 16 + lane] = g_sub_cand[(size_t)r * 16 + lane];
        }
        if (c_best && c_cand) {
            c_best[(size_t)r * 32 + lane] = c_cand[(size_t)r * 32 + lane];
        }
    }
}

// =================================================================================================
// Host Launchers & `extern "C"` Entry Points
// =================================================================================================
template <int NL, int GOSSET>
static int launch_viterbi_256_impl(
    const float* x, const float* w, int w_stride, const float* tab, const float* gains,
    const uint8_t* g_sub_fix, uint64_t* ws_trans, float* rec, uint8_t* g_sub_out,
    uint8_t* c_out, uint8_t* u_out, uint8_t* lb_out, int N, int U, int G, cudaStream_t st)
{
    if (N <= 0) return 0;
    if (G < 1 || G > 8 || NL * U > 512) return -4;
    vtq3_viterbi_256_kernel<NL, GOSSET><<<N, 128, 0, st>>>(
        x, w, w_stride, tab, gains, g_sub_fix, ws_trans, rec, g_sub_out, c_out, u_out, lb_out, N, U, G);
    return (int)cudaGetLastError();
}

static int dispatch_viterbi_256(
    const float* x, const float* w, int w_stride, const float* tab, const float* gains,
    const uint8_t* g_sub_fix, uint64_t* ws_trans, float* rec, uint8_t* g_sub_out,
    uint8_t* c_out, uint8_t* u_out, uint8_t* lb_out, int N, int nl, int U, int G, int gosset, cudaStream_t st)
{
    #define DISP(NL_C, GOS_C) launch_viterbi_256_impl<NL_C, GOS_C>( \
        x, w, w_stride, tab, gains, g_sub_fix, ws_trans, rec, g_sub_out, c_out, u_out, lb_out, N, U, G, st)
    if (gosset == 0) {
        if (nl == 4) return DISP(4, 0); if (nl == 8) return DISP(8, 0); if (nl == 16) return DISP(16, 0);
    } else if (gosset == 1) {
        if (nl == 4) return DISP(4, 1); if (nl == 8) return DISP(8, 1); if (nl == 16) return DISP(16, 1);
    } else if (gosset == 2) {
        if (nl == 4) return DISP(4, 2); if (nl == 8) return DISP(8, 2); if (nl == 16) return DISP(16, 2);
    } else if (gosset == 3) {
        if (nl == 4) return DISP(4, 3); if (nl == 8) return DISP(8, 3); if (nl == 16) return DISP(16, 3);
    }
    #undef DISP
    return -1;
}

template <int NL, int GOSSET>
static int launch_ldlq_256_impl(
    float* w_work, const float* d_sb, const float* dw_blk, const float* U_blk, const float* U_blk_inv,
    const float* tab, const float* gains,
    float* G_out, uint8_t* g_sub_out, uint8_t* c_out, uint8_t* u_out,
    float* ws_target, float* ws_orig, float* ws_G_cand, uint8_t* ws_gsub_cand,
    uint8_t* ws_c_cand, uint8_t* ws_u_cand, int32_t* ws_s_state, uint64_t* ws_trans,
    int M, int U, int G, cudaStream_t st)
{
    if (M <= 0) return 0;
    if (G < 1 || G > 8 || NL * U > 512) return -4;

    (void)ws_s_state;
    (void)ws_orig;
    vtq3_ldlq_fused_kernel<NL, GOSSET><<<M, 128, 0, st>>>(
        w_work, ws_target, d_sb, dw_blk, U_blk, U_blk_inv, tab, gains,
        ws_trans, G_out, g_sub_out, c_out, u_out, M, U, G);

    vtq3_viterbi_256_kernel<NL, GOSSET><<<M, 128, 0, st>>>(
        ws_target, dw_blk, 0, tab, gains, nullptr, ws_trans,
        ws_G_cand, ws_gsub_cand, c_out ? ws_c_cand : nullptr, u_out ? ws_u_cand : nullptr, nullptr,
        M, U, G);

    // Since `vtq3_ldlq_fused_kernel` keeps Cholesky updates in shared `s_w[256]` without mutating `w_work`,
    // `w_work` still holds the untouched original input (`ws_orig` copy eliminated too!).
    vtq3_monotone_select_kernel<<<(M + 7) / 8, 256, 0, st>>>(
        w_work, d_sb, U_blk_inv,
        G_out, g_sub_out, c_out, u_out,
        ws_G_cand, ws_gsub_cand, c_out ? ws_c_cand : nullptr, u_out ? ws_u_cand : nullptr, M);

    return (int)cudaGetLastError();
}

extern "C" int vtq3_viterbi_batch(
    const float* x, const float* w, int w_stride, const float* tab, const float* gains,
    const uint8_t* g_sub_fix, uint64_t* ws_trans,
    float* rec, uint8_t* g_sub_out, uint8_t* c_out, uint8_t* u_out, uint8_t* lb_out,
    int N, int nl, int U, int G, int gosset, void* stream)
{
    if (!ws_trans) return -2;
    return dispatch_viterbi_256(
        x, w, w_stride, tab, gains, g_sub_fix, ws_trans, rec, g_sub_out,
        c_out, u_out, lb_out, N, nl, U, G, gosset, (cudaStream_t)stream);
}

extern "C" int vtq3_ldlq_block256(
    float* w_work, const float* d_sb, const float* dw_blk, const float* U_blk, const float* U_blk_inv,
    const float* tab, const float* gains,
    float* G_out, uint8_t* g_sub_out, uint8_t* c_out, uint8_t* u_out,
    float* ws_target, float* ws_orig, float* ws_G_cand, uint8_t* ws_gsub_cand,
    uint8_t* ws_c_cand, uint8_t* ws_u_cand, int32_t* ws_s_state, uint64_t* ws_trans,
    int M, int nl, int U, int G, int gosset, void* stream)
{
    if (!ws_target || !ws_orig || !ws_G_cand || !ws_gsub_cand || !ws_s_state || !ws_trans) return -2;
    cudaStream_t st = (cudaStream_t)stream;
    #define DISP_L(NL_C, GOS_C) launch_ldlq_256_impl<NL_C, GOS_C>( \
        w_work, d_sb, dw_blk, U_blk, U_blk_inv, tab, gains, G_out, g_sub_out, c_out, u_out, \
        ws_target, ws_orig, ws_G_cand, ws_gsub_cand, ws_c_cand, ws_u_cand, ws_s_state, ws_trans, M, U, G, st)
    if (gosset == 0) {
        if (nl == 4) return DISP_L(4, 0); if (nl == 8) return DISP_L(8, 0); if (nl == 16) return DISP_L(16, 0);
    } else if (gosset == 1) {
        if (nl == 4) return DISP_L(4, 1); if (nl == 8) return DISP_L(8, 1); if (nl == 16) return DISP_L(16, 1);
    } else if (gosset == 2) {
        if (nl == 4) return DISP_L(4, 2); if (nl == 8) return DISP_L(8, 2); if (nl == 16) return DISP_L(16, 2);
    } else if (gosset == 3) {
        if (nl == 4) return DISP_L(4, 3); if (nl == 8) return DISP_L(8, 3); if (nl == 16) return DISP_L(16, 3);
    }
    #undef DISP_L
    return -1;
}
