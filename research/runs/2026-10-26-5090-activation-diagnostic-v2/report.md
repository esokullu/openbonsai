# Corrected activation diagnostic v2: completed, measurement only

The single corrected run completed, restored the exact final `rotated128x1` research state, passed all original/bypass/hard probe checks, captured both fixed windows on all 16 reused records, and sealed its measurements within the cumulative budget. **The later-window engineering selector passed on 16/16 records.** No training, optimizer, rescaling, new compression artifact, quality evaluation or successor experiment ran.

The main observation is a large discrepancy between targets produced from original versus quantized prefixes at window `[16,17)`, while the local hard window stays much closer to its conditional target. This supports investigating accumulated prefix drift. It does not identify a causal share of final error, isolate which earlier projection introduced the drift, or show that changing a training objective will recover quality. The [previous quality comparison](D:/openbonsai/runs/2026-10-26-5090-ultrachat-attempt2/report.md) remains unchanged.

**What was measured.** For each fixed input, `X0` and `Xq` are the window inputs after the original and hard-quantized prefixes. `T0` is the original window applied to `X0`; `Tq` is the original window applied to `Xq`; `Y` is the hard window applied to `Xq`. These are native hidden activations, not vocabulary logits or benchmark answers. Each window has 16 records, 7,203 active token positions and 29,503,488 valid scalar elements. Padding is excluded; records are pooled by valid-element count. Native oracle and repeat checks are excluded from those denominators.

| Window | Input drift RMS / RMS(X0) | Target gap RMS / RMS(T0) | Local error RMS / RMS(T0) | Total error RMS / RMS(T0) |
|---|---:|---:|---:|---:|
| `[0,1)` | 0.000000 | 0.000000 | 0.164182 | 0.164182 |
| `[16,17)` | 0.711287 | 0.761520 | 0.064267 | 0.760929 |

Here target gap is `Tq−T0`, local error is `Y−Tq`, and total error is `Y−T0`. The last three columns deliberately share the `T0` denominator. The separately reported pair metric `Y/Tq` is 0.076244 at window 16 because its denominator is RMS(Tq). Full RMS ratios, cosines, MSEs, per-record values and scalar-fit residuals are saved in the [independent metric interpretation](D:/openbonsai/runs/2026-10-26-5090-activation-diagnostic-v2/analysis/metric-interpretation.md) and its [JSON](D:/openbonsai/runs/2026-10-26-5090-activation-diagnostic-v2/analysis/metric-interpretation.json).

The exact decomposition includes the signed cross term:

`mean((Y−T0)^2) = mean((Y−Tq)^2) + mean((Tq−T0)^2) + 2 mean((Y−Tq)(Tq−T0))`.

| Window | Total MSE | Local MSE | Target-gap MSE | Signed cross mean |
|---|---:|---:|---:|---:|
| `[0,1)` | 0.000311311 | 0.000311311 | 0 | 0 |
| `[16,17)` | 0.196574567 | 0.001402227 | 0.196880042 | −0.001707702 |

At window 16, pooled gap RMS / total-error RMS is **1.000776694**; per-record minimum/median/maximum are **0.998704931 / 1.000907130 / 1.001816770**. Ratios near or above one are consistent with cancellation by the negative cross term. They are not percentages of error causally explained. The maximum absolute relative closure residual across records and pools is approximately `1.7140e-16`, below the predeclared `1e-12` limit.

All 16 later-window records are evaluable and satisfy the fixed rule: gap RMS at least half total-error RMS and above the noise floor, requiring 12/16 positives. Measured repeat/oracle discrepancy is exactly zero; the retained numerical floor gives threshold `1.4210854715202004e-13`. No denominator was zero or indeterminate. Window 0 supplies the negative control: `Xq=X0`, `Tq=T0`, and target gap/cross term are exactly zero.

The best nonnegative scalar on `Y` at window 16 is 0.797245, reducing descriptive relative RMSE against `T0` only from 0.760929 to 0.741553. This residual remains large. No scalar was applied to activations, weights or outputs, and no remedy was evaluated.

**Restoration and runtime gates.** The original checkpoint is `Qwen/Qwen3.5-9B`, revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`, original BF16 without task fine-tuning. The exact final learned FP32 state is the source pair's `rotated128x1/learned-fp32-state.pt`, SHA256 `4675c425ca82166a3f9275b5df358162f5b0a5ac8bb851d0fac8de962533274d`. Actual identities were obtained from authenticated inventories, not inferred from filenames. The pinned official snapshot's 15 files / 19,329,392,091 bytes were rehashed before execution.

All 764 loaded original tensors were recorded and checked against the expected scope: 200 selected projections plus 564 retained tensor descriptors, including separately described generated nonpersistent buffers. Saved retained values, shapes and dtypes matched. Restoration reproduced 200 wrappers, all 1,600 initial factor/statistic/sign fingerprints, 1,000 learned FP32 tensors, 200 saved sign tensors and the expected RNG states. Explicit signs were installed before group statistics. All original projection weight storages were reused. Group size 128, rank 64, rotation block size 4096, sigmoid factors and all other scientific settings remain unchanged. The complete original/restored tensor inventories match before and after measurement.

The five saved-probe comparisons were bitwise exact: original native; restored wrapper bypass; restored hard FP32; final hard FP32; final bypass. Each uses the same 31 IDs/mask and native eager/no-cache backend. Maximum and mean absolute logit differences were zero. The hard result matches the saved FP32 reference directly; packed FP16/export parity was not substituted. Embeddings, head, vision and small dynamics tensors retained their verified original precision. No optimizer moments or last-soft state were reconstructed.

The first two predetermined inputs passed 12 native original/conditional/hard oracle comparisons across the two windows and six repeat captures covering A-B-A and reverse order. All 32 counted captures kept batch 1, length 512 and native hidden width 4096. Hooks, modes, quantization flags and native recurrent/cache settings were restored. No kernel, input, sequence length or tolerance was changed to make a gate pass. The native reference backend's existing fallback implementations were retained.

**Timing and memory.** The one v2 model process ran **170.637 seconds**, from 20:39:17.954 to 20:42:08.591 on 26 October 2026, Europe/Istanbul (UTC+03). Its successful measurement seal was written at 169.536 seconds. V1's full failed activation cost of 23.503059 seconds is charged exactly once, giving **194.140074 seconds cumulative activation time**, below the original 2,700-second envelope. V2 had a conservative maximum of 2,676 seconds. V1's separately completed preparation took 299.838955 seconds and is reported once; v2 preparation time is zero.

| Timed operation group | Operations | Seconds |
|---|---:|---:|
| Original model loading | 1 | 15.205 |
| Exact FP32 state restoration | 1 | 20.758 |
| Saved model probes | 5 | 10.280 |
| Counted activation captures and statistics | 32 | 15.923 |
| Native oracle checks | 12 | 3.053 |
| Repeat captures | 6 | 2.899 |
| Streaming tensor hashes and preservation checks | 6 | 88.349 |
| Supervisor overhead and other checks | — | 14.170 |
| **V2 process total** | — | **170.637** |

The first-two-record projection was 433.194 seconds including a 300-second final-check reserve, comfortably within the remaining budget. Peak CUDA allocated/reserved memory across all 63 timed operations was **20,934,412,288 / 21,044,920,320 bytes (19.497 / 19.600 GiB)**. Minimum measured/conservative headroom was **11,319,377,920 bytes (10.542 GiB)**, above the 4 GiB requirement. These are diagnostic measurements with original weights and learned wrappers present; they are not packed-inference VRAM or speed benchmarks. Complete accounting is in [runtime-accounting.json](D:/openbonsai/runs/2026-10-26-5090-activation-diagnostic-v2/analysis/runtime-accounting.json).

Independent post-exit integrity/statistical audits and reporting are CPU-only and timed separately from the model process. The full independent preservation/outcome audit took **20.263 seconds**. They do not extend the model budget or run another model. The successful measurement seal, final native/hard probes, complete source-pair/v1 preservation verification and tensor checks all occurred inside the v2 model-process envelope.

**Exact v1 reuse and the guard correction.** The entire sealed v1 map inventory (340 files / 23,848,756 bytes) was authenticated before copying any inputs. The source-pair array inventory (2,034 files / 16,192,227,790 bytes) was fully authenticated, with its distinct schema validated. Thirty-three producer data/receipt files were copied byte-for-byte, including all raw traces, source/consumed records, ordered accepted IDs, source selection, generation settings, attempt history and completion receipts. V1 remains the producer; its runtime hash was not rewritten to v2's hash. The reuse receipt was fixed before the v2 source freeze, and the consumer receipt binds that freeze afterward, without circular hashes.

| Identity | SHA256 |
|---|---|
| Source pair final seal | `d7ae0ccf66b38f223366beb105beba32d651e5c2c1be166c21624eac7885bf8a` |
| V1 final seal | `c7397217fa4d6bd5889649194ac48a7ee12cb4d7cc0f8fcbf13fd81a017c75c4` |
| V1 producer runtime freeze | `08830dd39e3275b8686a189ce9f385621baefe3ed0ddf532851b4033601e1f8c` |
| V2 input-reuse receipt | `91c1195d4682ee8371ffc5553bc985e81207462414353db1dc92720515342a4e` |
| V2 consumer runtime freeze | `6186eadf00c73f69ca11544d3cdfaf0d7b0d555f736a93bc05153f8a3d71a067` |
| Raw traces, 118,848 bytes | `9c1c05142d948d8cb88e4770c3ac2695afb91b1f8534802ba4d81dd780b80d86` |
| Consumed records, 77,139 bytes | `443aaba5e5b41c641c27cdcd45f34fd5a2e770fa466321c9bbab84436ff40aaf` |
| Original calibration freeze | `6864b78e177fa8762da927789a5e53aa31f35c6bd1d0718b20af63c8dfe98016` |
| Original source-selection freeze | `eeb706f4cd47cdc266b7a6266b2bcff25f25c75167d3f0662286c29ba539a821` |

No trace was regenerated, replaced, selected anew or retokenized into measurement inputs. All 16 original accepted attempts and 7,203 active positions were reused. The original non-thinking/greedy preparation had 895 prompt and 7,023 generated tokens, four EOS completions and twelve capped continuations. Right truncation removed 715 tokens across twelve records; 989 padding positions were excluded from statistics. All EOS completions were retained. Disjointness evidence and accepted order were checked against authenticated producer receipts. These are search-validation inputs, not an untouched final confirmation set.

V1 failed because it compared the raw saved template hash to runtime text. V2 independently authenticates the saved 7,909-byte artifact (`a92e1dd97cb1cb175c9b70c0828e146bea4371c2643319b661b777e89811972e`), strictly decodes UTF-8, and converts only its 153 CRLF sequences to LF. The exact canonical/runtime text is 7,756 bytes (`a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715`). Bare CR, malformed UTF-8, BOMs, substantive changes and misdirected hashes/paths are rejected. Runtime text must already equal the pinned LF bytes. The pinned official tokenizer, original preparation settings, all 16 frozen prompt prefixes and exact saved probe passed CPU verification before model loading. Neither saved template nor producer receipt was edited.

**Source, tests and preservation.** All 89 frozen v1 checkout files remain byte-identical, including the 84 inherited source files, native backend, quantizer, trainer and earlier memory/chunking/tokenizer/stopping fixes. The measurement helper and common validation code are unchanged. Changes are limited to the template guard, explicit reuse provenance, a strict adapter for the specifically authenticated external v1 inventory receipt, cumulative timing and necessary path wiring/tests. Independent complete helper/driver review found no blocker. The combined CPU/native fixture suite passed **231 tests in 23.88 seconds**, including positive LF/CRLF cases and negative content, path, hash, ordering, mask, lineage and reset-budget checks. Draft preflight/test records are retained alongside the tightened final checks.

Environment matches v1: Python 3.12.11; PyTorch 2.14.1+cu130 / CUDA 13.0; Transformers 5.20.2; safetensors 0.8.0; NumPy 2.5.4; Accelerate 1.15.0; huggingface-hub 2.3.0; tokenizers 0.23.3; RTX 5090 sm120 with driver 596.21. Isolated imports and `PYTHONDONTWRITEBYTECODE=1` were enforced. Main HEAD remains `891d260af96a7ce61ccbcb23474015d975bb18dc`, with the original dirty diff/status and 39 source-file identities preserved. No frozen runtime code changed beneath the running process.

The in-process completion checks rehashed the entire source-pair and v1 inventories, all source/data bindings and restored tensors. The [independent post-exit audit](D:/openbonsai/runs/2026-10-26-5090-activation-diagnostic-v2/analysis/v2-outcome-audit.json) documents a separate CPU verification and its elapsed time. The original baseline, failed v1 attempt, prior checkpoints/logs and local fixes remain preserved. No code push, upload, monitor creation or cross-chat message was performed.

The process exited normally and the 5090 returned to idle. This is one completed activation diagnostic, not a recovered model or a new benchmark result. No fitting or follow-on run is scheduled. See [reproduction.md](D:/openbonsai/runs/2026-10-26-5090-activation-diagnostic-v2/reproduction.md), [the in-budget measurement seal](D:/openbonsai/runs/2026-10-26-5090-activation-diagnostic-v2/measurement/measurement-seal.json), and [the final report inventory](D:/openbonsai/runs/2026-10-26-5090-activation-diagnostic-v2/final-validation.json).
