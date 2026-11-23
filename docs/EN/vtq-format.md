# VTQ Format & Quantizer Specification (*Velri Trellis Quant*, `.vtq`)

This document specifies the primary **VTQ** weight quantization format (`MAGIC = b"VTQ\0"`, `VERSION = 2`), the reference decoder and binary container [`src/vtq_format.py`](../../src/vtq_format.py), the quantizer [`src/vtq.py`](../../src/vtq.py), the CPU AVX2 ([`src/cpu/vtq_avx2_bench.c`](../../src/cpu/vtq_avx2_bench.c)) and Blackwell `sm_120` GPU ([`src/cuda/vtq_matvec_bench.cu`](../../src/cuda/vtq_matvec_bench.cu)) decode kernels, and end-to-end speed, memory, and quality benchmarks (`KLD` vs `BF16`, paired bootstrap `B = 2000`) across **8 language models** at **`1.60–6.23 bits/weight`** (`bpw`).

---

## 1. Design Goals & Differences from Research Prototypes (`LTQ` and `VTQ v2`)

The central goal of **VTQ** is to **reduce decode instructions per weight and eliminate memory-subsystem bottlenecks** on both CPU (`AVX2`, compiled strictly with `-mno-avx512f`) and GPU (`FP32` and `DP4A INT8` `GEMV`/`GEMM` at `B = 1, 8, 32`), while preserving lattice-trellis coding fidelity and fast quantization speed.

1. **Unified Self-Contained `16 rows × 256 cols = 4,096 weights` Super-Tile (`816 .. 3,184 bytes`)**:
   - In the `VTQ v2` prototype, each row slice was split across 5 separate `SoA` regions (`codes`, `usgn`, `umid`, `lsb`, `gains`) plus a separate exponent tensor `e[M, K/256]`; fetching 6 disjoint pointers fragmented HBM/L2 cache-line transactions.
   - In **VTQ**, all bitplanes, gain indices, and linear superblock scales `uint8 u_sb[16]` are packed into **one contiguous C struct `vtq_st16_*`**, byte-for-byte identical across CPU, GPU, and `.vtq` files (`1` linear read stream, reaching up to **`1476.9 GB/s = 96.4%`** of peak HBM bandwidth on `RTX PRO 6000 Blackwell`).
2. **Zero Runtime Galois ALU on GPU (`Joint-1024` LUT in shared memory) and Native `uint16_t` Shift on CPU**: advancing to the next column on CPU takes a single `_mm256_srli_epi16` instruction; on GPU it takes a single shift `s = ((s << 1) | bit) & 0x3FF` with zero Ungerboeck parity-mask computations and zero cross-row warp shuffles.
3. **Full `nl = 16` Ungerboeck Subsets Across All 20 Active Menu Formats** (`1.5 .. 6.0 bpw`) using additive codebooks `AT2 (16B+16B)` at `K = 4` and bit-7-zeroed `AT2Z (32B+32B)` without `vpblendvb` at `K = 5, 6`. Optional fast CPU modes `VTQ_300_FAST` (`nl = 8`) and `VTQ_500_FAST` (`nl = 4`) remain supported.
4. **Linear 8-Bit Superblock Scale `uint8 u_sb[16]` (`× 1/128`)** instead of logarithmic exponents `2^(0.25·e)`: `0.45%` relative step around full scale instead of `18.9%`, with no `exp2f` in the decoder.
5. **Intra-Subblock `16 × 16` Tail-Biting Trellis**: each column subgroup `g` (columns `16g .. 16g+15`) across all 16 rows of a super-tile forms one closed `256`-step tail-biting trellis encoded with causal `Block16-LDLQ` (`tcq_ldlq_block256`).

---

## 2. Mathematical Matrix Reconstruction Model

Let a linear layer have weight matrix $W \in \mathbb{R}^{M \times K}$ with $16 \mid M$ and $256 \mid K$:

$$
\widehat{W}_{m, :} = \Bigl(\mathcal{R}_K^\top \bigl(\widetilde{W}_{m, :}\bigr)\Bigr) \odot \text{suh}, \qquad
\widetilde{W}_{m, 256 b + t} = d_{\text{row}}[m] \cdot \frac{u_{\text{sb}}[m, b]}{128} \cdot \frac{\text{gain}_{g[m, b, \lfloor t/16 \rfloor]}}{64} \cdot \frac{Q(\text{lab}_{m,b,t},\, u_{m,b,t})}{64},
$$

where:
- $b$ is the `256`-column superblock index and $t \in \{0, \dots, 255\}$ is the natural column index (`0` permutations);
- $d_{\text{row}} \in \mathbb{F}_{16}^M$ is the per-row scale (`fp16`), centered so that the row-wise maximum of `u_sb` lands near `224`;
- $u_{\text{sb}} \in \{1, \dots, 255\}$ is the linear superblock multiplier stored in the tail of each super-tile (`16` bytes per `4,096` weights);
- $\text{gain} \in \{1, \dots, 255\}^{2^h}$ is the `16`-weight subgroup gain bank ($h \in \{1, 2, 3\}$);
- $Q(\text{lab}, u) \in [-127, +127]$ is the codebook level (`qmax = 127`, `TA + TB ≤ 127` for additive codebooks);
- $\text{suh} \in \mathbb{F}_{16}^K$ and $\mathcal{R}_K = (H_{b_1} \otimes H_{b_2}) \otimes_{\text{outer}} \operatorname{DCT\text{-}II}_q$ (`rot_block = -1`) or block-wise `FWHT-256` (`rot_block = 256`).

---

## 3. `16 × 256` Super-Tile & Trellis Architecture (`ν = 10`, `1,024` States)

### 3.1. Intra-Subblock Tail-Biting Trellis

Coded bits are stored as `uint8_t c[16][16][2]`; `c[g][r][0]` holds columns `0..7` (MSB-first) and `c[g][r][1]` holds columns `8..15` of row `r` in subgroup `g` (in the CPU kernel, byte order within each pair is arranged for single-instruction little-endian `uint16_t` windowing).

The trellis for subgroup `g` traverses **all 16 rows sequentially** (`256` steps: `(r=0, j=0..15), (r=1, j=0..15), …, (r=15, j=0..15)`) and wraps around: the initial 9-bit history of row `0` comes from the final 9 coded bits of row `15` in the same subgroup `g`. Thus, the 16 subgroups of a super-tile form 16 independent trellises, requiring zero cross-thread state communication between rows on GPU.

At each step, a 10-bit state $s = (h \ll 1) \mid c_j$ determines the subset label via $\mathrm{GF}(2)$ parity masks:

$$
\begin{aligned}
\text{nl} = 16 &: \quad \text{lab}(s) = (s \mathbin{\&} 1) \;\Big|\; \bigl(\pi_{10}(s \mathbin{\&} \text{0x2CE}) \ll 1\bigr) \;\Big|\; \bigl(\pi_{10}(s \mathbin{\&} \text{0x050}) \ll 2\bigr) \;\Big|\; \bigl(\pi_{10}(s \mathbin{\&} \text{0x3B6}) \ll 3\bigr), \\
\text{nl} = 8  &: \quad \text{lab}(s) = (s \mathbin{\&} 1) \;\Big|\; \bigl(\pi_{10}(s \mathbin{\&} \text{0x2CE}) \ll 1\bigr) \;\Big|\; \bigl(\pi_{10}(s \mathbin{\&} \text{0x39C}) \ll 2\bigr), \\
\text{nl} = 4  &: \quad \text{lab}(s) = (s \mathbin{\&} 1) \;\Big|\; \bigl(\pi_{10}(s \mathbin{\&} \text{0x2CE}) \ll 1\bigr),
\end{aligned}
$$

where $\pi_{10}(x)$ is the 10-bit parity (Ungerboeck generators `[359, 40, 475]`, `[359, 462]`, `[359]`).

### 3.2. Super-Tile Layout (`pack_supertiles_vtq`)

| Field | Size (bytes) | Purpose |
| :--- | :---: | :--- |
| `c[16][16][2]` | `512 B` | Trellis coded bits |
| `usgn[16][16][2]` | `512 B` (`0 B` for `fmt=15,17`) | Sign half-axis bits for $K_{\text{eff}} \ge 2.0$ |
| `u_amp` | `0 .. 2048 B` | Uncoded amplitude planes: `Gosset[8,4,4]` (`256 B`), `D4[4,3,2]` (`384 B`), `D4_eq` (`128 B`), `1b` / `2b` / `4b` (`512 / 1024 / 2048 B`) |
| `gains` | `32 / 64 / 96 B` | $h$-bit gain indices per `16`-weight subgroup |
| `u_sb[16]` | `16 B` | Per-row linear `uint8` superblock scale (`× 1/128`) |

### 3.3. Hardware Codebooks (`hw_codebook`, `qmax = 127`, `tab / 64`)

| Type | Formats | CPU AVX2 Decode |
| :--- | :--- | :--- |
| `s8_32` | `K = 1.5, 1.75` | `2× vpshufb` over signed 32-byte table using bit-7 zeroing |
| `u8` | `K = 2.0 .. 3.0` (`16B` / `32B`), `_FAST` | `1–2× vpshufb` + sign |
| `at2_16_16` | `K = 3.25 .. 4.0` | `2× vpshufb + vpaddusb` (two 16-byte tables indexed by label nibbles) |
| `at2z_32_32` | `K = 4.25 .. 6.0` | `4× vpshufb + vpaddusb`; index bit 7 zeros `vpshufb` output without `vpblendvb` |

---

## 4. Decode Throughput

### 4.1. Inner-Loop Operation Comparison (per `32` weights)

| Aspect | `LTQ` Prototype | `VTQ v2` Prototype (`16×16 SoA`) | **`VTQ` (`16×256 Super-Tile`)** |
| :--- | :---: | :---: | :---: |
| Memory streams per tile | `3–4` | `6` (`5 SoA + e`) | **`1` contiguous struct** |
| State step & `lab` on GPU | `popc` / `shfl` + masks (`~9` ALU) | `__popc × 3` + shifts (`~11` ALU) | **`1` shift + `1× LDS s_joint`** |
| Cross-row trellis handoff (GPU `B=1`) | — | — | **`0` shuffles** |
| `256`-block scale | `fp16` per block | `exp2f(0.25 * e)` | **`u_sb * (1/128)` once per `256` cols** |

### 4.2. CPU AVX2 (`1` Thread, `-O3 -mavx2 -mfma -mno-avx512f`, `M=256, K=2048`)

Measurements from [`results/vtq_decode_speed.json`](../../results/vtq_decode_speed.json) (`AMD EPYC 9B45`, Colab G4) and [`results/vtq_local_5070ti_speed.json`](../../results/vtq_local_5070ti_speed.json) (`AMD Ryzen 9 9950X`). All kernels `PASS` (`rel_err = 7.0e-8 .. 8.2e-8`). Units: `GW/s`.

| Bitrate | `VTQ` Mode | `EPYC`: `LTQ` → `v2` → **`VTQ`** | `VTQ` vs `v2` (`EPYC`) | `9950X`: `LTQ` → `v2` → **`VTQ`** | `VTQ` vs `v2` (`9950X`) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| `1.60` | `K=1.5 Gosset84`, `s8_32` | — → — → **`18.82`** | new | — → — → **`24.04`** | new |
| `2.10` | `K=2.0`, `nl=16` | `19.02` → `21.70` → **`22.22`** | **`+2.4%`** | `24.96` → `23.17` → **`29.88`** | **`+29.0%`** |
| `2.60` | `K=2.5 Gosset84`, `nl=16` | `16.74` → `16.56` → **`18.13`** | **`+9.5%`** | `22.44` → `21.11` → **`24.21`** | **`+14.7%`** |
| `3.16` | `K=3.0`, `nl=16` / `FAST nl=8` | `15.67` → `16.99` → **`18.68 / 20.17`** | **`+9.9%`** | `20.33` → `20.09` → **`24.76 / 26.93`** | **`+23.2%`** |
| `3.66` | `K=3.5 Gosset84`, `AT2` | `14.42` → `13.14` → **`14.20`** | **`+8.1%`** | `20.03` → `17.27` → **`18.78`** | **`+8.7%`** |
| `4.22` | `K=4.0`, `AT2 16+16` | `14.66` (`nl=8`) → `14.85` → **`15.19`** | **`+2.3%`** | `19.40` (`nl=8`) → `19.23` → **`20.11`** | **`+4.6%`** |
| `5.22` | `K=5.0`, `AT2Z` / `FAST nl=4` | `14.23` (`nl=4`) → `10.69` → **`11.82 / 15.77`** | **`+10.6%`** | `19.06` (`nl=4`) → `14.42` → **`15.49 / 21.30`** | **`+7.4%`** |
| `6.22` | `K=6.0`, `AT2Z 32+32` | — → `10.12` → **`12.29`** | **`+21.4%`** | — → `11.95` → **`16.30`** | **`+36.4%`** |

### 4.3. Blackwell `sm_120` GPU (`2,048 MB` DRAM)

Practical HBM read roof: `RTX PRO 6000 Blackwell` (`188 SM`) — `1532 GB/s` (`fp16` GEMV `763.8 GW/s`); `RTX 5070 Ti` (`70 SM`) — `693 GB/s` (`fp16` GEMV `343.6 GW/s`).

| Bitrate | `PRO 6000`: `LTQ` → `v2` | **`VTQ` FP32 / DP4A `B=1`** | HBM Bandwidth | **`VTQ` `B=8 / B=32`** (`GW·tok/s`) | `5070 Ti`: `v2` → **`VTQ` DP4A** |
| :---: | :---: | :---: | :---: | :---: | :---: |
| `1.60` | — → — | **`3618.7 / 3802.3`** (`4.98× fp16`) | `758 GB/s` | — | — → **`1138.5`** |
| `2.10` | `1948.4` → `2337.9` | **`3650.3 / 3806.6`** (**`+62.8%`** vs v2) | `997 GB/s` | **`14,880 / 18,562`** | `816.3` → **`1117.8`** (`+36.9%`) |
| `2.60` | — → `1862.4` | **`3089.5 / 3378.1`** (**`+81.4%`**) | `1096 GB/s` | — | `606.2` → **`1002.0`** (`+65.3%`) |
| `3.16` | `1986.1` → `1636.9` | **`3084.6 / 3353.5`** (**`+104.9%`**) | `1324 GB/s` | **`11,268 / 17,003`** | `590.9` → **`1003.7`** (`+69.9%`) |
| `3.66` | — → `1399.3` | **`2454.4 / 2509.7`** (**`+79.4%`**) | `1148 GB/s` | — | `480.2` → **`776.1`** (`+61.6%`) |
| `4.22` | `1645.1` → `1512.8` | **`2701.8 / 2763.9`** (**`+82.7%`**) | **`1458 GB/s` (`95.2%`)** | **`12,947 / 15,451`** | `549.3` → **`942.0`** (`+71.5%`) |
| `5.22` | `1974.4` → `1261.5` | **`2235.2 / 2263.1`** (**`+79.4%`**) | **`1477 GB/s` (`96.4%`)** | — | `502.5` → **`833.2`** (`+65.8%`) |
| `6.22` | — → `1565.2` | **`1872.5 / 1875.6`** (**`+19.8%`**) | **`1459 GB/s` (`95.2%`)** | — | `463.3` → **`740.9`** (`+59.9%`) |

- Fused `Block-256 FWHT + INT8` activation quantization overhead (`K = 8192`): **`1.43 µs`** at `B = 1` and **`5.51 µs`** at `B = 32`.

---

## 5. VTQ Format Menu (`results/vtq_tables.json`)

All **22** formats pass the bit-exact self-test in [`src/vtq_format.py`](../../src/vtq_format.py) (`max_diff = 0.0e+00` across C reference, CUDA Viterbi, and Python). SNR and gap to $R(D)$ are measured on i.i.d. $\mathcal{N}(0,1)$ (`4,096 × 256` validation blocks). Italicized entries are optional fast CPU modes outside the default `VTQ_MENU` DP set.

| Name | Code | $K$ | `frac` | `nl` | $h$ | Codebook | Bytes / `16×256` | `bpw` | SNR (bits) | Gap to $R(D)$ | `distortion_ratio` |
| :--- | :-: | :-: | :---: | :-: | :-: | :---: | :---: | :---: | :---: | :---: | :---: |
| `VTQ_150` | `15` | 2 | `gosset84` | 16 | 1 | `s8_32` | `816` | `1.6016` | `1.3680` | `0.2336` | `3.35367` |
| `VTQ_175` | `17` | 2 | `d4` | 16 | 1 | `s8_32` | `944` | `1.8516` | `1.6622` | `0.1894` | `2.12358` |
| `VTQ_200` | `20` | 2 | — | 16 | 1 | `u8` | `1072` | `2.1016` | `1.9656` | `0.1359` | `1.34629` |
| `VTQ_225` | `22` | 3 | `d4_eq` | 16 | 1 | `u8` | `1200` | `2.3516` | `2.2095` | `0.1421` | `0.94317` |
| `VTQ_250` | `25` | 3 | `gosset84` | 16 | 1 | `u8` | `1328` | `2.6016` | `2.4647` | `0.1369` | `0.65376` |
| `VTQ_250_H2` | `24` | 3 | `gosset84` | 16 | 2 | `u8` | `1360` | `2.6641` | `2.5199` | `0.1441` | `0.60426` |
| `VTQ_275` | `26` | 3 | `d4` | 16 | 1 | `u8` | `1456` | `2.8516` | `2.7030` | `0.1485` | `0.46647` |
| `VTQ_275_H2` | `27` | 3 | `d4` | 16 | 2 | `u8` | `1488` | `2.9141` | `2.7649` | `0.1491` | `0.42761` |
| *`VTQ_300_FAST`* | `30` | 3 | — | 8 | 2 | `u8` | `1616` | `3.1641` | `3.0132` | `0.1509` | `0.30135` |
| `VTQ_300` | `31` | 3 | — | 16 | 2 | `u8` | `1616` | `3.1641` | `3.0275` | `0.1366` | `0.29543` |
| `VTQ_325` | `32` | 4 | `d4_eq` | 16 | 2 | `at2_16_16` | `1744` | `3.4141` | `3.2615` | `0.1525` | `0.21255` |
| `VTQ_350` | `35` | 4 | `gosset84` | 16 | 2 | `at2_16_16` | `1872` | `3.6641` | `3.5227` | `0.1414` | `0.14762` |
| `VTQ_375` | `37` | 4 | `d4` | 16 | 2 | `at2_16_16` | `2000` | `3.9141` | `3.7624` | `0.1517` | `0.10577` |
| `VTQ_400` | `40` | 4 | — | 16 | 3 | `at2_16_16` | `2160` | `4.2266` | `4.0592` | `0.1674` | `0.07009` |
| `VTQ_425` | `42` | 5 | `d4_eq` | 16 | 2 | `at2z_32_32` | `2256` | `4.4141` | `4.2680` | `0.1460` | `0.05240` |
| `VTQ_450` | `45` | 5 | `gosset84` | 16 | 2 | `at2z_32_32` | `2384` | `4.6641` | `4.5234` | `0.1407` | `0.03679` |
| `VTQ_475` | `47` | 5 | `d4` | 16 | 2 | `at2z_32_32` | `2512` | `4.9141` | `4.7632` | `0.1508` | `0.02642` |
| *`VTQ_500_FAST`* | `50` | 5 | — | 4 | 3 | `u8` | `2672` | `5.2266` | `5.0211` | `0.2055` | `0.01845` |
| `VTQ_500` | `51` | 5 | — | 16 | 3 | `at2z_32_32` | `2672` | `5.2266` | `5.0620` | `0.1646` | `0.01747` |
| `VTQ_550` | `55` | 6 | `gosset84` | 16 | 3 | `at2z_32_32` | `2928` | `5.7266` | `5.5559` | `0.1707` | `0.00881` |
| `VTQ_575` | `57` | 6 | `d4` | 16 | 3 | `at2z_32_32` | `3056` | `5.9766` | `5.7994` | `0.1771` | `0.00629` |
| `VTQ_600` | `60` | 6 | — | 16 | 3 | `at2z_32_32` | `3184` | `6.2266` | `6.0473` | `0.1793` | `0.00447` |

---

## 6. `.vtq` Binary Container (`VtqWriter` / `VtqFile`)

The container is `mmap`-ready (`MAGIC = b"VTQ\0"`, `VERSION = uint32_le(2)`), aligning the payload start and every tensor to `64` bytes. JSON metadata (`config`, `tokenizer`, `chat_template`, `quantization`, and the `tensors` index) is written as an **unbounded footer** at the end of the file in a single streaming pass with no fixed header reservation (`VtqFile` retains backward compatibility for reading `VERSION = 1` files):

```text
+-----------------------------------------------------------------------+
| Offset  0..3   (4 B)  : MAGIC         = b"VTQ\0"                      |
| Offset  4..7   (4 B)  : VERSION       = uint32_le(2)                  |
| Offset  8..15  (8 B)  : header_offset = uint64_le(footer_offset)      |
| Offset 16..23  (8 B)  : header_len    = uint64_le(len(json_bytes))    |
| Offset 24..31  (8 B)  : data_off      = uint64_le(64)  (% 64 == 0)    |
| Offset 32..63  (32 B) : reserved      = 0                             |
+-----------------------------------------------------------------------+
| Offset data_off .. header_offset - 1 : Tensors (64-byte aligned)      |
+-----------------------------------------------------------------------+
| Offset header_offset .. EOF          : UTF-8 JSON (metadata + index)  |
+-----------------------------------------------------------------------+
```

The `source_metadata(model_dir)` helper in `src/vtq_format.py` bundles `config.json`, `generation_config.json`, `tokenizer.json`, `tokenizer_config.json`, `preprocessor_config.json`, `video_preprocessor_config.json`, and `chat_template` for single-file model packaging. Because `u_sb[16]` is embedded inside each super-tile, a quantized matrix `<prefix>` emits **4** tensors:

1. `<prefix>.vtq.data` — `u8 [N_bytes]`: concatenated 64-byte-aligned super-tile arrays;
2. `<prefix>.vtq.slices` — `i32 [S, 5]`: `[r0, r1, fmt_idx, byte_offset, rot_block]` across mixed-precision row slices;
3. `<prefix>.vtq.d_row` — `f16 [M]`;
4. `<prefix>.vtq.suh` — `f16 [K]`.

**End-to-end verification**: all `196` linear matrices of `Llama-3.2-3B-Instruct @ 3.0000 bpw` serialized to `1,065,657,856` bytes and reloaded through `VtqFile` show **`0.000000` KLD drift** (`0.04084153 / 0.05497863` before and after disk round-trip). Independently, all 22 trained formats pass `VtqWriter → VtqFile` with `max_diff = 0.0e+00`.

---

## 7. VTQ Encoder & Resource Usage

1. **`Block16-LDLQ` over Intra-Subblock Trellises** (`ldlq_block256_cuda` in [`src/cuda/tcq_viterbi.cu`](../../src/cuda/tcq_viterbi.cu)): 16 causal Cholesky feedback stages per `256`-column block, 2-pass tail-biting Viterbi (`1,024` states, radix-4, `__ballot_sync`), and joint `(gain, state)` search with `n_cand = 4` when `2^h > 4`.
2. **Scales**: `d_row` is pre-centered so the row maximum of `u_sb` lands near `224` (`0` saturations at `255`), followed by monotonic Cholesky-weighted `u_sb` refinement.
3. **Global Refit**: closed-form `(c_row → fp16 d_row, fp16 suh)` pass under the full Hessian (`VTQ_REFIT_SWEEPS=1` by default).
4. **`VTQ_GS_SWEEPS`** (default `0`): optional Gauss–Seidel sweep with residual Viterbi (`1`) or full `LDLQ` re-encoding (`2`).
5. **`quantize_rings_vtq`** (embeddings, `lm_head`, `Gemma 4` PLE): 1D tail-biting Viterbi over rings of length `128 / 160 / 256` without zero-padding.
6. **`polish_model_scales_kld_checkpointed`**: checkpointed end-to-end `(d_row, suh)` optimization under teacher-forced KL divergence (`0` extra bits).
7. **Codebook Training** (`python -m src.vtq --train-tables`, `~2 min` on `RTX 5070 Ti`): joint trellis Lloyd–Max over `(tab, gain_u8)`, coordinate search `gain_u8 ± 1, 2`, and `distortion_ratio` calibration on `Llama-3.2-3B-Instruct`.

### 7.1. Full-Model Quantization Wall-Time (`quant_sec`) & Per-Matrix Peak VRAM (`RTX PRO 6000 Blackwell`)

| Model (Rig) | `VTQ v2` Prototype | **`VTQ`** | `EXL3` |
| :--- | :---: | :---: | :---: |
| `Llama-3.2-1B` @ `3.0` (`RTX 5070 Ti`) | `76.0 s` | **`78.9 s`** | — |
| `Llama-3.2-1B` (`G4`, `1.75–6.0 bpw`) | `32–50 s` | **`31–55 s`** | `109–160 s` |
| `Llama-3.2-3B` (`G4`) | `71–113 s` | **`80–127 s`** | `268–401 s` |
| `Qwen3.5-4B` (`G4`) | `112–142 s` | **`121–147 s`** | `168–336 s` |
| `Gemma-4-E4B` (`G4`) | `86–156 s` | **`92–139 s`** | `199–396 s` |
| `Qwen3-8B` / `Llama-3.1-8B` (`G4`) | `161–269 s` | **`234–422 s`** | — |
| `Qwen3.8-27B` (`G4`) | `600–660 s` | **`635–700 s`** | `~45 min` |
| `Qwen3-30B-A3B` (`G4`) | `599–630 s` | **`596–635 s`** | — |

Peak extra encoder VRAM above inputs (`W` and `H`) and single-matrix time (`RTX PRO 6000`):

| Matrix (`M × K`) | `VTQ v2` Prototype | **`VTQ`** |
| :--- | :---: | :---: |
| `3B down` (`3072 × 8192`), `3.0 bpw` | `1.80 GiB / 0.28 s` | **`1.09 GiB / 0.27 s`** |
| `8B gate/up` (`14336 × 4096`), `3.0` | `4.40 GiB / 0.28 s` | **`2.41 GiB / 0.28 s`** |
| `8B down` (`4096 × 14336`), `3.0` | `3.39 GiB / 0.62 s` | **`3.28 GiB / 0.62 s`** |
| `27B gate/up` (`17408 × 5120`), `3.0` | `4.55 GiB / 0.43 s` | **`3.66 GiB / 0.43 s`** |
| `27B down` (`5120 × 17408`), `3.0` | `5.12 GiB / 0.91 s` | **`4.85 GiB / 0.91 s`** |
| `27B down` (`5120 × 17408`), `4.5` | `5.11 GiB / 3.19 s` | **`4.85 GiB / 3.19 s`** |

**Isolated A/B with code emission (`PACK`) vs `98d8ef1`.** Single format `VTQ_300`, synthetic `W` and `H`, Colab G4. Peak is extra PyTorch-allocated VRAM above pre-call baseline; time is median of four interleaved runs after warmup (file write and decode excluded). Bit-exact reconstructed FP32 weights verified across all three shapes ([per-run measurements](../../results/vtq_g4_pack_shapes_ab.json)):

| `M × K` | Peak before → after, MiB | Encode before → after, ms |
| --- | ---: | ---: |
| `768 × 3072` | 153 → 139 | 51.3 → 50.9 |
| `8192 × 3072` | 1,307 → 1,086 | 136.3 → 133.3 |
| `3072 × 8192` | 1,340 → 1,215 | 283.4 → 276.7 |

[GPT-2 small end-to-end check](../../results/vtq_g4_gpt2_pack_ab.json): SHA-256 of codes, scales, and reconstructed FP32 weights matched across all 48 quantized matrices (`KLD` vs BF16 on 16 held-out WikiText-2 contexts of 128 tokens: `0.069489`; warm encode time `1.180 → 1.169 s`; peak extra per-matrix VRAM `153 → 139 MiB`).

---

## 8. End-to-End Quality Benchmarks Across 8 Models (`B = 2000`)

Mean teacher-forced `KL` divergence from `BF16` on held-out `nemotron_trace.json` (`nem`) and `multilingual_trace.json` (`mul`). Δ denotes relative mean difference (negative = `VTQ` better), with paired bootstrap win rate (`p`) in parentheses. All runs executed on `RTX PRO 6000 Blackwell` (Colab G4).

### 8.1. `Llama-3.2-1B-Instruct` ([`results/vtq_1b_sweep.json`](../../results/vtq_1b_sweep.json))

| `bpw` | `VTQ` (`nem / mul`) | Δ vs `VTQ v2` Prototype | Δ vs `EXL3` (`p = 1.000`) |
| :---: | :---: | :---: | :---: |
| `1.7500` | `0.49713 / 0.75092` | `+0.7% / +1.9%` | — |
| `2.0000` | `0.30866 / 0.43102` | `+2.1% / +1.4%` | — |
| `2.1000` | **`0.24646 / 0.34401`** | **`-4.9% / -5.5%`** (`1.000`) | — |
| `2.5000` | `0.12955 / 0.17878` (Polish **`0.12180 / 0.17243`**) | `+2.6% / +1.8%` (Polish **`-3.5% / -1.8%`**) | **`-16.0% / -16.9%`** (Polish **`-21.1% / -19.8%`**) |
| `3.0000` | `0.05871 / 0.08231` (Polish **`0.05728 / 0.08093`**) | `+0.5% / 0.0%` | **`-18.1% / -19.5%`** |
| `3.4375` | `0.03069 / 0.04456` (Polish **`0.03028 / 0.04404`**) | `-1.6% / +1.9%` | **`-25.0% / -19.8%`** |
| `4.0000` | `0.01451 / 0.02022` | `+1.2% /` **`-2.8%`** (`0.983`) | **`-18.2% / -20.5%`** |
| `4.5000` | `0.00753 / 0.01067` | `+0.9% / +0.6%` | **`-28.8% / -21.3%`** |
| `5.0000` | `0.00407 / 0.00572` | `-1.0% / +1.3%` | — |
| `6.0000` | `0.00174 / 0.00201` | `+2.0% / +0.7%` | — |

### 8.2. `Llama-3.2-3B-Instruct` ([`results/vtq_3b_sweep.json`](../../results/vtq_3b_sweep.json), [`results/vtq_3b_neutral_audit.json`](../../results/vtq_3b_neutral_audit.json))

| `bpw` | `VTQ` (`nem / mul`) | Δ vs `VTQ v2` Prototype | Δ vs `EXL3` (`p = 1.000`) |
| :---: | :---: | :---: | :---: |
| `1.7500` | `0.40900 / 0.58002` | `+2.5% / +5.4%` | — |
| `2.0000` | `0.23605 / 0.31906` (Polish **`0.21617 / 0.29708`**) | `-2.3% / +0.3%`; Polish **`-1.8%`** vs `VTQ2-Polish` on `nem` | — |
| `2.1000` | `0.19638 / 0.25746` (Polish **`0.18154 / 0.24171`**) | `+6.0% / +6.2%`; Polish `-2.0% / -0.2%` vs base `VTQ2` | — |
| `2.5000` | `0.08942 / 0.12070` (Polish **`0.08544 / 0.11666`**) | `+0.1% / +2.3%` | **`-4.9% / -6.0%`** (Polish **`-9.1% / -9.1%`**) |
| `3.0000` | **`0.04084 / 0.05498`** (Polish **`0.03956 / 0.05349`**) | **`-0.3% / -0.8%`**; Polish **`-1.2% / -1.7%`** vs `VTQ2-Polish` | **`-6.8% / -8.0%`** (Polish **`-9.8% / -10.5%`**) |
| `3.4375` | **`0.02106 / 0.02902`** (Polish **`0.02069 / 0.02860`**) | **`-4.3% / -2.0%`** (`0.999 / 0.887`) | **`-15.7% / -10.5%`** (Polish **`-17.1% / -11.8%`**) |
| `4.0000` | `0.01039 / 0.01345` | `+4.7% /` **`-1.1%`** | **`-6.8% / -11.2%`** |
| `4.5000` | `0.00535 / 0.00685` | `+3.1% /` **`-6.0%`** | **`-23.0% / -22.2%`** |
| `5.0000` | **`0.00295 / 0.00366`** | `0.0% / +0.7%` | — |
| `6.0000` | **`0.00123 / 0.00137`** | `+0.4% / +2.5%` | — |

- **Domain-neutral audit** (`WikiText-2 train → WikiText-2 test`): `VTQ-NeuCal` reaches **`0.05414`** at `3.0 bpw` (`-13.5%` vs `TCQPQ` `0.06260`) and **`0.01279`** at `4.0 bpw` (`-15.0%` vs `TCQPQ` `0.01504`).

### 8.3. `Llama-3.1-8B-Instruct` ($K_{\text{down}} = 14336 = 2048 \times 7$, [`results/vtq_llama31_8b.json`](../../results/vtq_llama31_8b.json))

| `bpw` | `VTQ` (`nem / mul`) | Δ vs `VTQ v2` (`p`) | Δ vs `TCQPQ` | Δ vs `EXL3` (`p = 1.000`) |
| :---: | :---: | :---: | :---: | :---: |
| `2.5000` | **`0.06749 / 0.08891`** (Polish **`0.06309 / 0.08509`**) | **`-2.1% / -2.2%`** (`0.92 / 0.95`) | **`-8.9% / -6.0%`** | — |
| `3.0000` | **`0.03014 / 0.04118`** (Polish **`0.02921 / 0.04028`**) | **`-2.1% / -1.6%`**; Polish **`-1.3% / -2.1%`** vs `VTQ2-Polish` | **`-6.9% / -3.8%`** | — |
| `3.4375` | **`0.01625 / 0.02223`** (Polish **`0.01589 / 0.02198`**) | **`-7.6% / -6.1%`** (`1.000 / 1.000`) | **`-3.8% / -4.0%`** | **`-12.6% / -10.7%`** (Polish **`-14.5% / -11.7%`**) |
| `4.0000` | **`0.00777 / 0.01025`** | **`-1.6% / -6.0%`** (`0.79 / 1.000`) | **`-4.1% / -8.7%`** | — |
| `4.5000` | **`0.00420 / 0.00541`** | **`-2.6% / -5.3%`** (`0.96 / 0.9995`) | **`-3.4% / -7.5%`** | **`-19.4% / -19.4%`** |

### 8.4. `Qwen3-8B` (`36` layers with `QK-Norm`, [`results/vtq_qwen3_8b.json`](../../results/vtq_qwen3_8b.json))

| `bpw` | `VTQ` (`nem / mul`) | Δ vs `VTQ v2` | Δ vs `TCQPQ` | Δ vs `EXL3` |
| :---: | :---: | :---: | :---: | :---: |
| `2.5000` | `0.09680 / 0.09521` (Polish **`0.08485 / 0.08598`**) | `+2.4% / +4.0%` | **`-8.0% / -4.8%`** (Polish **`-19.4% / -14.1%`**) | — |
| `3.0000` | `0.04270 / 0.04091` (Polish **`0.03992 / 0.03874`**) | `+2.5% / +7.4%` | `-0.7% / +4.5%` | **`-10.9% / -0.8%`** (Polish **`-16.7% / -6.1%`**) |
| `3.4375` | `0.02250 / 0.02036` (Polish **`0.02152 / 0.01966`**) | `+5.4% / +4.1%`; Polish `+0.8% / +0.5%` vs base `VTQ2` | `-1.7% / +1.8%` | **`-12.1% / -16.4%`** (`0.9995 / 1.000`) |
| `4.0000` | **`0.01046 / 0.00875`** | **`-2.1% / -1.8%`** | **`-5.1% / -7.8%`** | — |
| `4.5000` | `0.00565 / 0.00471` | `+5.5% / +4.1%` | **`-3.0% / -6.0%`** | **`-30.1% / -33.3%`** (`1.000`) |
| `5.0000` | **`0.00304`** / `0.00257` | **`-1.2%`** / `+7.2%` | **`-12.8% / -7.7%`** | — |

### 8.5. Hybrid `Qwen3.5-4B` (`Gated DeltaNet + Attention`, [`results/vtq_qwen35_4b.json`](../../results/vtq_qwen35_4b.json))

| Target `bpw` | `dp:VTQ` (`nem / mul`) | Δ vs `dp:VTQ2` (`p`) | Δ vs `EXL3` (`p = 1.000`) |
| :---: | :---: | :---: | :---: |
| `2.5` | `0.09497 /` **`0.13246`** | `+2.3% /` **`-4.4%`** (`1.000` on `mul`) | — |
| `3.0` | `0.04324 /` **`0.06285`** | `+1.9% /` **`-1.2%`** | **`-18.9% / -23.2%`** |
| `3.5` | `0.02092 /` **`0.03063`** | `+2.0% /` **`-1.3%`** | **`-21.9% / -25.0%`** |
| `4.0` | `0.01054 /` **`0.01550`** | `+0.5% /` **`-2.0%`** | **`-24.8% / -27.6%`** |
| `4.5` | `0.00557 /` **`0.00823`** | `+1.3% /` **`-0.4%`** | — |

### 8.6. `Gemma-4-E4B-it` — Linear Layers ([`results/vtq_gemma4_e4b.json`](../../results/vtq_gemma4_e4b.json)) & PLE `VtqRing` ([`results/vtq_gemma4_ple.json`](../../results/vtq_gemma4_ple.json))

**Linear layers (`TextModel + VTQ DP`)**:

| Target `bpw` | `dp:VTQ` (`nem / mul`) | Δ vs `dp:VTQ2` (`p`) | Δ vs `EXL3` (`p = 1.000`) |
| :---: | :---: | :---: | :---: |
| `2.5` | **`0.20454 / 0.23285`** | **`-0.1% / -3.6%`** (`0.49 / 0.99`) | — |
| `3.0` | `0.08745 / 0.10620` | `+1.1% / +2.2%` | **`-15.8% / -16.1%`** |
| `3.5` | **`0.03585 / 0.04303`** | **`-8.0% / -9.8%`** (`1.000 / 1.000`) | **`-34.0% / -33.9%`** |
| `4.0` | **`0.01726 / 0.02212`** | **`-14.7% / -11.8%`** (`1.000 / 1.000`) | **`-36.7% / -33.8%`** |
| `4.5` | **`0.00873`** / `0.01140` | **`-0.9%`** / `+5.2%` | — |

**Per-Layer Embeddings** (`embed_tokens_per_layer`, `262144 × 42 × 256`, `5.25 GiB` in BF16):

| Mode | `bpv` (`VTQ` / `v2`) | Size | `VTQ` (`nem / mul`) | `VTQ v2 rot` (`nem / mul`) | Time `VTQ` / `v2` |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `VTQ_200:rot` | `2.0952` / `2.1250` | `0.688 GiB` | `0.00890 /` **`0.01098`** | `0.00877 / 0.01123` | **`12.7 s`** / `22.0 s` |
| `VTQ_250:rot` | `2.5952` / `2.6250` | `0.852 GiB` | **`0.00416 / 0.00555`** (`-3.2% / -5.2%`) | `0.00430 / 0.00586` | **`82.3 s`** / `129.3 s` |
| `VTQ_300:rot` | `3.1577` / `3.1875` | `1.036 GiB` | **`0.00217 / 0.00274`** (`0.0% / -3.9%`) | `0.00217 / 0.00285` | **`30.1 s`** / `48.3 s` |
| `VTQ_400:rot` | `4.2202` / `4.2500` | `1.385 GiB` | `0.00101 / 0.00109` | `0.00099 / 0.00108` | **`35.4 s`** / `54.2 s` |

### 8.7. Hybrid 27B `Qwen3.8-27B` (`64` layers, $K_{\text{down}} = 17408 = 1024 \times 17$, [`results/vtq_qwen38_27b.json`](../../results/vtq_qwen38_27b.json))

| `q_bpw` | `dp:VTQ` (`nem / mul`) | Δ vs `dp:VTQ2` (`p`) | Δ vs `dp:TCQPQ` | External Baseline | Δ vs Baseline (`p`) | Quant. |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `2.9409` | `0.02339 / 0.02704` | `+4.4% / +0.6%` | **`-5.1% / -2.8%`** | `GSQ-RCO IQ3_XXS`: `0.04796 / 0.05488` | **`-51.2% / -50.7%`** (`1.000`) | `10.8 min` |
| `3.0043` | `0.02129 / 0.02540` | `+0.7% / +6.6%` | **`-10.7% / -2.1%`** | `EXL3 SC 3.00bpw H4`: `0.02601 / 0.02625` | **`-18.1% / -3.2%`** (`1.000 / 0.857`) | `10.6 min` |
| `3.4794` | `0.01198 / 0.01281` | `+0.9% / +3.3%` | `+1.0% /` **`-3.6%`** | `GSQ-RCO IQ3_S`: `0.02208 / 0.02442` | **`-45.7% / -47.6%`** (`1.000`) | `11.4 min` |
| `3.5038` | **`0.01092`** / `0.01220` | **`-4.3%`** / `+2.8%` (`0.91 / 0.15`) | **`-11.9% / -5.9%`** | `EXL3 3.50bpw`: `0.01860 / 0.01905` | **`-41.3% / -36.0%`** (`1.000`) | `11.7 min` |

### 8.8. MoE `Qwen3-30B-A3B-Instruct-2507` (`48` layers × `128` experts = `12,480` matrices, [`results/vtq_qwen3_30b.json`](../../results/vtq_qwen3_30b.json))

| `bpw` | `dp:VTQ` (`nem / mul`) | `PPL` (`nem / mul`) | Δ vs `dp:VTQ2` (`p`) | Δ vs `dp:TCQPQ` | Δ vs `EXL3` (`p = 1.000`) | Quant. |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `3.0` | `0.02342 / 0.02259` | `3.9260 / 3.3073` | `+4.3% / +3.1%` (`0.07 / 0.08`, CIs overlap) | `+6.1% / +6.9%` | **`-49.6% / -47.2%`** | `10.6 min` |
| `3.5` | `0.01365 / 0.01228` | `3.9014 / 3.2989` | — | — | — | `9.9 min` |
| `4.5` | `0.00633 / 0.00514` | `3.8962 / 3.2536` | `+2.0% / +4.1%` (`0.25 / 0.10`) | **`-6.4% / -8.4%`** | **`-19.5% / -19.8%`** | `10.6 min` |

---

## 9. Quick Start & Reproducibility

```bash
# 1. Bit-exact self-test of all 22 VTQ formats (C reference + CUDA Viterbi + .vtq container)
PYTHONPATH=. python3 -m src.vtq_format

# 2. Retrain codebooks and calibrate distortion_ratio (~2 min on RTX 5070 Ti)
PYTHONPATH=. python3 -m src.vtq --train-tables

# 3. CPU AVX2 and GPU sm_120 decode throughput benchmarks
PYTHONPATH=. python3 -m src.run_vtq_benchmarks --suite speed

# 4. Model evaluation suites: 1b, 3b, qwen3_8b, llama8b, qwen35, gemma4, qwen27, moe_30b
PYTHONPATH=. python3 -m src.run_vtq_benchmarks --suite 3b
```
