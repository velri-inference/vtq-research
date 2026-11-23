# End-to-End Format Comparison (`VTQ` vs `EXL3` vs `GGUF`)

All three formats (`VTQ`, `EXL3` `mul1`, and `GGUF` `imatrix`) are quantized from the same BF16 checkpoint on the **exact same neutral calibration set** (`wikitext-2-raw-v1` `train`, `64 × 2048 = 131,072` tokens) and evaluated end-to-end (including quantized `embed_tokens` and `lm_head`) on three strictly disjoint traces:
- **`wikitext2`** — held-out `wikitext-2-raw-v1` `test` split (`ppl_ref = 7.5537`),
- **`nemotron`** — English multi-turn chat dialogues (`ppl_ref = 3.9697`),
- **`multilingual`** — `ru / zh / de / fr / es` dialogues (`ppl_ref = 3.3358`).

Bitrates (`total_bpw`) are measured from the complete serialized text-model file size on disk (`total_text_bytes * 8 / total_text_params`).

---

## `Llama-3.1-8B-Instruct`

Raw JSON: [`results/paper_llama31_8b.json`](../../results/paper_llama31_8b.json)

![Llama-3.1-8B KLD and Perplexity Pareto](../figures/llama31_8b_kld_ppl.svg)

| Method | `total_bpw` | `linear_bpw` | Quant Time | KLD (`nem` / `mul` / `wt2`) | $\Delta\text{PPL}$ (`nem` / `mul` / `wt2`) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`VTQ:2.5`** | **`2.5072`** | `2.4235` | **`158.5 s`** | **`0.13003` / `0.19752` / `0.14959`** | **`+0.4336` / `+0.6293` / `+1.1839`** |
| `EXL3:2.5` | `2.4925` | `2.3353` | `527.5 s` | `0.15281` / `0.24938` / `0.16235` | `+0.4982` / `+0.8709` / `+1.2430` |
| `GGUF:IQ2_S` | `2.7481` | `2.4790` | `57.4 s`* | `0.24075` / `0.39132` / `0.37948` | `+0.9033` / `+1.3874` / `+3.2839` |
| **`VTQ:3.5`** | **`3.5072`** | `3.4188` | **`199.3 s`** | **`0.02670` / `0.04010` / `0.03206`** | **`+0.0781` / `+0.0947`** / `+0.2515` |
| `EXL3:3.5` | `3.4862` | `3.3281` | `526.5 s` | `0.03460` / `0.05234` / `0.03804` | `+0.1105` / `+0.1513` / **`+0.2503`** |
| `GGUF:IQ3_M` | `3.7706` | `3.5754` | `67.8 s`* | `0.05446` / `0.07111` / `0.08631` | `+0.1945` / `+0.2074` / `+0.6705` |
| **`VTQ:4.5`** | **`4.5072`** | `4.4141` | **`211.1 s`** | **`0.00674` / `0.01018` / `0.00863`** | **`+0.0223`** / `+0.0305` / `+0.0643` |
| `EXL3:4.5` | `4.4789` | `4.3197` | `347.3 s` | `0.00889` / `0.01380` / `0.01138` | `+0.0247` / **`+0.0303` / `+0.0582`** |
| `GGUF:Q4_K_S` | `4.6750` | `4.5361` | `31.5 s`* | `0.01671` / `0.02033` / `0.02425` | `+0.0549` / `+0.0556` / `+0.1795` |
| `GGUF:Q5_K_M` | `5.7114` | `5.6532` | `29.6 s`* | `0.00485` / `0.00563` / `0.00750` | `+0.0084` / `+0.0118` / `+0.0760` |
| **`VTQ:6.0`** | **`6.0072`** | `5.9659` | **`237.3 s`** | **`0.00156` / `0.00184` / `0.00192`** | **`+0.0053` / `+0.0046`** / `+0.0065` |
| `EXL3:6.0` | `5.9681` | `5.9579` | `350.6 s` | `0.00200` / `0.00259` / `0.00216` | `+0.0063` / `+0.0058` / **`+0.0054`** |
| `GGUF:Q6_K` | `6.5712` | `6.5625` | `15.7 s`* | `0.00248` / `0.00275` / `0.00358` | `+0.0071` / `+0.0060` / `+0.0245` |

*\*Excluding shared `llama-imatrix` pass (`310.7 s`).*
