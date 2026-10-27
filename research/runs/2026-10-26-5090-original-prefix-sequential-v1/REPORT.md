This attempt ended at the prescribed resource gate, before scientific training. The complete forecast was **11,970.646 seconds (3 h 19 min 31 s)** against a **10,800-second cap**. The supervisor stopped cleanly after **192.985 seconds**, with **4 discarded preflight updates and 0/4,096 scientific updates**. No retry, smaller recipe, new quality evaluation, or packed artifact was launched.

The isolated implementation and real 9B CUDA preflight passed. The separate actual-9B, 128-update window-0 historical control remains **unexecuted**; the native CPU equivalence test does not substitute for it. This attempt provides feasibility measurements, not evidence of quality recovery.

The planned experiment used the official, untouched `Qwen/Qwen3.5-9B` BF16 checkpoint at revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`. It would visit all 32 decoder windows in order, using the same 128 frozen records once per window: batch 1, length 512, window/stride 1, group 128, signed Hadamard blocks of 4096, rank-64 corrections, sigmoid factors, Huber delta 1, factor/LoRA learning rates 0.005/0.001, zero weight decay, gradient clipping 1, and gradient checkpointing. Each window would have 102 soft and 26 frozen-Jacobian hard updates with a fresh optimizer and the frozen seed/shuffle schedule.

The narrow trainer change preserves the current compressed-prefix input and native kwargs/masks, but captures its target independently with every quantizer bypassed through the original-weight prefix. The default conditional-target behavior remains available. The [actual trainer diff](D:/openbonsai/runs/2026-10-26-5090-original-prefix-sequential-v1/analysis/trainer-diff.patch), [14 named native tests](D:/openbonsai/runs/2026-10-26-5090-original-prefix-sequential-v1/analysis/sequential-trainer-native-validation.json), and [driver admission callsite](D:/openbonsai/runs/2026-10-26-5090-original-prefix-sequential-v1/tools/sequential_driver.py:255) are saved. All work used the isolated checkout; dirty main and previous artifacts were preserved.

The final CPU suite passed **896 tests with 2 skipped**. The real CUDA preflight exercised soft and hard backward/optimizer updates in recurrent window 0 and full-attention window 3. All four losses, gradients, FP32 optimizer moments, and parameter updates were valid. Learned state, snapshots, runtime/RNG state, the hard-initial logit probe, and all **764 original/retained tensors** were restored exactly. Fresh native and bypassed-wrapper logits matched the authenticated original probe, and the separately restored historical conditional128 probe matched exactly.

| Measured preflight result | Value |
|---|---:|
| Model-process elapsed time | 192.985 s |
| Completed measured phases | 93 |
| Peak allocated GPU memory | 21,708,011,520 B (20.217 GiB) |
| Peak reserved GPU memory | 22,693,281,792 B (21.135 GiB) |
| Minimum sampled/conservative headroom | 9,660,530,688 B (8.997 GiB) |
| Memory gate | All phases passed the 4 GiB minimum |

The machine was the local RTX 5090, driver 596.21, Python 3.12.11, PyTorch 2.14.1+cu130/CUDA 13.0, and Transformers 5.20.2. Execution used BF16 native computation, FP32 learned parameters, eager attention, no KV cache, TF32 disabled, and four CPU threads. Available RAM and disk exceeded 120 GiB and 460 GiB before launch. The native recurrent reference kernels were retained to preserve the runtime comparison.

The admission estimate includes all work, rather than extrapolating only optimizer steps. Cache timing uses the slower of two fixed records at each of six prefix-depth anchors, with separate recurrent/full-attention interpolation. Forward timing covers 513 positions, including the longest generation context. The four discarded updates include cold overhead; their conservative measured times were not replaced after seeing the admission result.

| Forecast component, before margin | Seconds |
|---|---:|
| All 32 window caches | 2,847.261 |
| 4,096 optimizer updates | 1,881.485 |
| Full before/after hard reconstruction passes | 1,455.967 |
| Window audit evidence | 11.375 |
| Both complete quality evaluations | 1,452.266 |
| Fresh original/packed loading and setup | 205.557 |
| Historical control, search boundaries, preservation | 269.211 |
| Complete artifact construction/save/reference verification | 820.452 |
| **Variable total** | **8,943.574** |

The admission total is `191.178 + 1.25 × 8,943.574 + 600 = 11,970.646 seconds`. The 600-second reserve is additional to export work. Both evaluations reserve all **1,024 possible generation tokens**, without assuming early EOS. Existing valid calibration traces were reused, so no new calibration generation was required. The [resource inputs](D:/openbonsai/runs/2026-10-26-5090-original-prefix-sequential-v1/preflight/resource-inputs.json) and [failed admission receipt](D:/openbonsai/runs/2026-10-26-5090-original-prefix-sequential-v1/preflight/resource-admission.json) retain every component.

The 128 calibration records have 59,855 consumed active positions: 26 original traces ended with EOS, 102 reached the generation cap, and 105 records were truncated to 512 positions, removing 7,049 tokens. They were generated with nonthinking greedy settings. The separate search set has 16 records and 7,203 active positions. Historical conditional128 search boundaries at windows 0/16/31 were captured, but there is no newly trained model for the matched comparison. No unused final-confirmation inputs were opened.

The following are **archived development results**, not new measurements from this stopped attempt. They use 2,048 scored WikiText tokens, 20 ARC-Easy questions, and four generation prompts; they must not be confused with the original pilot's 16,384-token/100-question/10-generation population. Scientific score-reuse eligibility remains pending the unstarted fresh scientific process.

| Representation | NLL | Perplexity | ARC-Easy | Empty / capped answers | Evaluation seconds | Evaluation peak allocated / reserved GiB |
|---|---:|---:|---:|---:|---:|---:|
| Archived untouched BF16 | 2.423777 | 11.288 | 19/20 | 0 / 2 | 34.278 | 17.954 / 18.191 |
| Archived conditional128, hard FP32 | 9.121631 | 9,151.112 | 8/20 | 4 / 0 | 34.019 | 19.599 / 19.869 |
| Archived conditional128, packed | 9.123808 | 9,171.055 | 8/20 | 4 / 0 | 18.230 | 19.460 / 19.846 |
| This original-prefix attempt | Unmeasured | Unmeasured | Unmeasured | Unmeasured | Unmeasured | Unmeasured |

The archived hard model differs from BF16 by **+6.697854 NLL and −55 percentage points ARC accuracy**; the packed model differs by +6.700031 and −55 points. Both compressed variants chose A for all 20 ARC questions, eight of which have A as the correct answer. Conditional128 generated only four EOS tokens, versus 341 BF16 tokens, so its shorter evaluation time indicates no useful speed advantage. [Exact archived answers and token IDs](D:/openbonsai/runs/2026-10-26-5090-original-prefix-sequential-v1/analysis/generations-draft.md) are preserved. The prior single-window original-target repair improved local reconstruction but still produced degenerate answers; neither result establishes recovery.

No new model size can be reported. For context, the authenticated conditional128 artifact occupies **6,576,831,354 bytes excluding tokenizer**, plus **19,998,761 tokenizer bytes**, for **6,596,830,115 deployable bytes**. Its equivalent tensor comparison is **6,576,245,389 / 18,819,627,488 bytes**, including every compressed projection and retained persistent tensor: **2.861759×, or 65.056453% reduction**. Unused MTP source tensors total 486,581,248 bytes and are excluded from that denominator. Cache duplicates, research state, and reconstructed inference weights are excluded from storage size.

| Archived retained floating component | Bytes |
|---|---:|
| Output head | 2,034,237,440 |
| Token embeddings | 2,034,237,440 |
| Vision encoder and its rotary buffers | 912,021,104 |
| Linear-attention gates | 12,582,912 |
| Linear-attention convolutions | 1,572,864 |
| Text normalization | 546,816 |
| Linear-attention dynamics | 3,072 |
| Text rotary buffers | 256 |
| **Total, including 400 generated buffer bytes** | **4,995,201,904** |

The same archived artifact contains 1,460,554,349 code-stream bytes, 108,003,328 scale bytes, 1,081,344 rotation-sign bytes, 11,404,864 row-index bytes, and no projection biases. These are historical storage measurements. BITCOS parity and file size do not establish quality or acceleration; the reference runtime reconstructs BF16 weights. The retained BF16 embedding/head alone exceed a 2 GB language-model target.

A **proposal only** is to retain the complete 128-record window-0 control and use 64 records once in each remaining window: 2,112 scientific updates. Linear scaling of only those windows' cache/update/reconstruction work gives approximately **8,188 seconds (2 h 16 min)**, keeping complete evaluations, export, controls, margin, and reserve. This is an unvalidated changed recipe, requires new authorization and admission, and was neither selected nor launched.

The [execution audit](D:/openbonsai/runs/2026-10-26-5090-original-prefix-sequential-v1/analysis/final-execution-audit.json) and [independent preservation audit](D:/openbonsai/runs/2026-10-26-5090-original-prefix-sequential-v1/analysis/final-preservation.json) passed. All 3,543 files across five sealed ancestors, all 182 frozen execution files, and 39 main source files match their recorded hashes. Main HEAD remains `891d260af96a7ce61ccbcb23474015d975bb18dc`; status and diff are unchanged. The worker exited and no successor remains. The source-freeze SHA256 is `1f16a6ed34b3450db6808551604c558784acfadc4c6051a3ac1d0946c7caa457`.

The four requested sealed development-evidence files were also transported in five ordered, verified chunks; their [transport receipt](D:/openbonsai/runs/2026-10-26-5090-original-prefix-sequential-v1/analysis/development-evidence-transport.json) records exact source hashes and round-trip checks. No model weights were transferred. Full phase tables, raw summaries, commands, environment evidence, source review, and test logs remain in this run directory. The [final inventory seal](D:/openbonsai/runs/2026-10-26-5090-original-prefix-sequential-v1/final-validation.json) binds this completed preflight-stop record.
