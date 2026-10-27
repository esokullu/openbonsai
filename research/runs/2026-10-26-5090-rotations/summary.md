# Completed pilot diagnosis: quality gate failed

The original failed pilot and all historical evidence are preserved. Diagnosis, CPU BITCOS repacking, controlled recovery tests, and the paired 16-sample × 8-epoch experiment have finished. **No 128-sample run is starting.**

Quantization-disabled wrappers preserve BF16 logits exactly. Severe degradation is already present in the evaluated hard ternary weights; FP16 scale casting changes NLL only slightly. Export and independent reload preserve their packed reference exactly. Thus successful serialization does not establish preserved quality.

Current comparison on the same **2,048 scored tokens, 20 ARC-Easy questions and four generation prompts**:

| Model | NLL | Perplexity | ARC-Easy | Useful complete answers |
|---|---:|---:|---:|---:|
| Original BF16 | 2.423777 | 11.288 | 19/20 | 2/4; other two relevant but truncated |
| Unrotated 16×8 packed | 14.437115 | 1,861,912.383 | 1/20 | 0/4 |
| Rotated 16×8 packed | 9.255346 | 10,460.334 | 8/20 | 0/4 |

Rotation improves NLL by **5.181769** and ARC by **35 percentage points** against its matched control. Nevertheless, every rotated answer loops `Here / This / This…` to the 128-token cap, while unrotated answers repeat `number`. The recovery condition remains unmet. Repeated use makes these diagnostics development data, not fresh confirmation.

The preserved original pilot used the larger **16,384-token / 100-question / 10-generation** workload: BF16 perplexity **9.18**, ARC **97/100**; compressed perplexity **327,505**, ARC **23/100**, ten whitespace-only capped outputs. These populations are distinct from the table above.

The requested **CPU BITCOS repack** saved **270,433,688 bytes** with exact codes, scales, biases, retained tensors, tokenizer assets and fixed-input logits:

| Failed pilot storage | Before | After |
|---|---:|---:|
| Model bytes excluding tokenizer | 6,831,783,487 | 6,561,349,799 |
| Supplementary tokenizer bytes | 19,998,761 | 19,998,761 |

Current complete deployable directories are **6,592,437,649 bytes unrotated** and **6,596,776,129 bytes rotated**, including tokenizer. Equivalent persistent BF16 tensor payload falls from **18.820 GB** to **6.572 / 6.576 GB**: approximately **65.08% / 65.06%** reduction. The retained 4.995 GB consists mainly of embeddings and output head (**4.068 GB combined**) plus vision (**0.912 GB**).

Both new conditions completed 4,096 finite, valid updates, all payload checks, and fresh-process reloads. Training peaks were about **20.2 GiB** allocated. Total measured drivers took **104.55 minutes**. Packed inference used a nonpersistent decoded BF16 cache: about **19.46 GiB** peak allocated versus **17.95 GiB BF16**, and was slower under this reference runtime. Storage savings are not VRAM or speed savings.

The final CPU suite passed **639 tests**, with two Windows privilege skips; native CUDA hybrid preflights passed. Preservation verification checked **1,379 unique historical files** and all **33 frozen runtime Python files** unchanged. No cloud GPU, push, upload, or additional training job was used.

[Full report, diagnosis, ablations, settings and limitations](D:/openbonsai/runs/2026-10-26-5090-rotations/report.md) · [Reproduction commands and archived source](D:/openbonsai/runs/2026-10-26-5090-rotations/reproduction.md) · [Exact size accounting](D:/openbonsai/runs/2026-10-26-5090-rotations/analysis/paired-size-accounting.md) · [Verbatim answers](D:/openbonsai/runs/2026-10-26-5090-rotations/analysis/quality-generations.md).
