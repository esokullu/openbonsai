# Completed small-pilot diagnosis

**Quality collapses when hard ternary weights are applied. The controlled scale recovery remains unusable, so the 128-sample run was not started.** The completed original pilot, failed artifact, calibration, evaluations, logs, and local fixes are preserved. All 231 original pilot file hashes and 35 archived source hashes still match.

These diagnostic results use the same fixed **2,048 scored WikiText tokens, 20 ARC-Easy questions, and four generation prompts** in every row. They are separate from the full pilot's 16,384 tokens / 100 questions / ten prompts.

| State | NLL | Perplexity | ARC correct |
|---|---:|---:|---:|
| Original BF16 | 2.423777 | 11.29 | 19/20 |
| Wrappers, quantization disabled | 2.423777 | 11.29 | 19/20 |
| Hard ternary, learned FP32 scales | 13.256495 | 571,772 | 1/20 |
| Same ternary, exported FP16 scales | 13.257692 | 572,457 | 1/20 |
| Independently reloaded pilot | 13.257692 | 572,457 | 1/20 |
| Fixed-code LS recovery, reloaded | 11.247840 | 76,714 | 4/20 |

Disabled wrappers preserve probe logits exactly. FP16 export adds only 0.001198 NLL; fresh reload adds zero and preserves probe logits exactly. The pilot did not save learned FP32 factors, so a separate deterministic diagnostic replay reconstructed them without replacing the original run. All 512 training losses/gradient norms and all 200 packed projection payloads match. Exact packed agreement cannot prove historical equality of the unsaved FP32 factors.

Sixteen forward ablations show distributed damage: quantizing only MLP gives NLL 10.742, only linear attention 10.242, and only the first four layers 6.381. Retaining any one tested broad projection family in BF16 does not recover quality. These use factors learned in the full quantized pipeline, not separately trained alternative policies.

The one predeclared recovery holds every ternary code fixed and analytically fits scales to original BF16 weights. It adds no samples, epochs, or held-out fitting. Relative weight RMSE improves from 0.5264 to 0.4752 and NLL falls 2.009853 nats, but ARC remains 20%. All four answers repeat unrelated `的/了` characters to the 128-token cap. Export/reload checks pass. The recovery gate fails; there is no evidence that more training alone will repair this recipe.

Thinking is consistently disabled. Three of 16 calibration traces hit the 512-generation-token cap; training removes 96 tokens from three traces (5,330 before truncation, 5,234 active afterward). These are limitations, not an established explanation for the collapse.

BITCOS was integrated in an isolated checkout at `997b9c3fa8788bf31b12b1cecb713655a7953361`, preserving the memory, chunking, tokenizer, and stopping fixes. CPU repacking of the original failed artifact passed exact checks for 200 projections, 564 retained tensors, tokenizer assets, configuration, and checksums. CUDA probe logits match exactly, maximum difference **0**; the original quality results carry over.

| Storage scope | Original packed bytes | BITCOS bytes |
|---|---:|---:|
| Model artifact, including metadata | 6,831,783,487 | 6,561,349,799 |
| Supplementary tokenizer | 19,998,761 | 19,998,761 |
| Complete deployable directory | 6,851,782,248 | 6,581,348,560 |

BITCOS saves **270,433,688 bytes: 3.9585% of model artifact bytes**, or 3.9469% including tokenizer. Codes shrink from 1,728,053,248 to 1,457,556,523 bytes; scales remain 108,003,328 bytes. There are no selected projection biases. Storage savings do not imply faster inference; the reference runtime reconstructs dense weights.

Retained tensors/buffers total **4,995,201,904 bytes**:

| Component | Bytes |
|---|---:|
| Untied output head | 2,034,237,440 |
| Token embeddings | 2,034,237,440 |
| Vision components | 912,021,104 |
| Recurrent gates | 12,582,912 |
| Recurrent convolutions | 1,572,864 |
| Text normalization | 546,816 |
| Recurrent dynamics | 3,072 |
| Text rotary buffer | 256 |

The preserved full pilot remains PPL **9.18 versus 327,505**, ARC **97/100 versus 23/100**. Its load-plus-evaluation peak allocated VRAM was **17.954 versus 6.789 GiB**, and evaluation time excluding load **58.17 versus 281.48 seconds**. The full report separates operational phases, diagnostic evaluation timings, and memory measurements.

Validation: **123 root CPU tests passed**; isolated BITCOS tests **230 passed, 2 skipped** for Windows symlink privileges. Existing CPU smoke and tiny native CUDA BF16 forward/backward/compression/export/reload evidence remains in the pilot. No code was pushed and no artifacts uploaded.

See [detailed measured report](report.md), [causal interpretation and limitations](diagnosis-interpretation.md), [reproduction commands](REPRODUCING.md), [recovery decision](recovery-decision.json), and [preserved full pilot report](../2026-10-26-5090-pilot/report.md). These preliminary results establish execution and serialization behavior, not lossless model quality.
