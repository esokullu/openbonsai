The historical diagnosis located the severe quality loss at hard ternary reconstruction, before FP16 scale export, independent reload, or BITCOS repacking. BITCOS subsequently reduced storage while preserving the failed representation exactly. These historical experiments used a different training recipe from the current paired rotation run and are not its matched control.

The **complete recorded pilot workload** comprised 16,384 scored WikiText tokens, 100 ARC-Easy questions, and ten generation prompts. Original BF16 achieved NLL **2.217540**, ARC **97/100**, and nonblank text in **10/10** prompts; the reloaded packed model achieved NLL **12.699258**, ARC **23/100**, and nonblank text in **0/10**. Its ten generations reached the 128-token cap with whitespace-only outputs. This was the complete local pilot workload, not the complete benchmark datasets. [Preserved pilot report](D:/openbonsai/runs/2026-10-26-5090-pilot/report.md)

The subsequent **diagnostic subset** used the first four frozen NLL segments (**2,048 scored tokens**), first **20 ARC questions**, and **four fixed generation prompts**, without resampling or retokenization. All stages below ran BF16 computation; “FP32” identifies the learned quantizer factors, not a full FP32 model.

| Diagnostic model state | NLL | ARC-Easy | Nonblank outputs | Reached cap without EOS |
|---|---:|---:|---:|---:|
| Original BF16 | 2.423777 | 19/20 | 4/4 | 2/4 |
| Wrappers installed, quantization disabled | 2.423777 | 19/20 | 4/4 | 2/4 |
| Hard ternary with FP32 learned factors | 13.256495 | 1/20 | 0/4 | 4/4 |
| Export representation with FP16 scales | 13.257692 | 1/20 | 0/4 | 4/4 |
| Original pilot artifact, independent reload | 13.257692 | 1/20 | 0/4 | 4/4 |

Disabled wrappers preserved the saved fixed-input logits exactly across the model with 200 replaced projections. Hard reconstruction added **10.832718 nats** of NLL; FP16 scale casting added only **0.001198**, and reload added zero. Export-reference and fresh-reload logits were bitwise equal. ARC is conditional option-label scoring; “nonblank” is not a correctness or intelligibility metric. [Stage results and controls](D:/openbonsai/runs/2026-10-26-5090-diagnosis/report.md), [wrapper parity](D:/openbonsai/runs/2026-10-26-5090-diagnosis/wrapper-parity.json), [reload parity](D:/openbonsai/runs/2026-10-26-5090-diagnosis/export-reload-logit-parity.json)

The original pilot had not saved its FP32 factors. A separate replay recovered factors whose exported codes/scales/biases matched all **200** saved projection payloads exactly. All **512** replay losses, gradient norms, and window/step/stage identities also matched the original event records. This reproduces the saved artifact and recorded training trajectory; it cannot establish equality of unavailable original FP32 factor tensors. [Payload agreement](D:/openbonsai/runs/2026-10-26-5090-diagnosis/replay-packed-agreement.json), [event agreement](D:/openbonsai/runs/2026-10-26-5090-diagnosis/replay-event-parity.json)

The family and depth controls reused those fitted factors and changed which projections were quantized at inference. They were **not separately retrained policies**, and generation was skipped for all 16 controls.

| Quantized subset | NLL | ARC-Easy |
|---|---:|---:|
| MLP only | 10.741537 | 8/20 |
| Recurrent/linear attention only | 10.241759 | 2/20 |
| Full attention only | 4.390061 | 8/20 |
| Everything selected except MLP | 13.149479 | 4/20 |
| Everything selected except recurrent/linear attention | 12.876343 | 8/20 |
| Everything selected except full attention | 10.651272 | 1/20 |
| First 4 decoder layers | 6.381065 | 7/20 |
| First 8 decoder layers | 8.830579 | 4/20 |
| First 16 decoder layers | 11.363617 | 2/20 |
| First 24 decoder layers | 11.255560 | 5/20 |

Individual controls implicated MLP `gate_proj` (NLL **6.270979**) and recurrent `out_proj` (**7.324950**) more strongly than `up_proj` (**3.346235**) or `in_proj_qkv` (**2.820819**) under this fitted state. Accumulated hidden-state cosine similarity fell from **0.9264** at layer 0 to **0.1465** at layer 31; final relative RMSE was **1.0336**. These interventions and accumulated drift demonstrate sensitivity, not a unique cause or proof that more calibration fixes it. [All ablations and definitions](D:/openbonsai/runs/2026-10-26-5090-diagnosis/report.md), [layer drift](D:/openbonsai/runs/2026-10-26-5090-diagnosis/hard-layer-drift.json)

Calibration used **16 original-model AYOT traces**, thinking disabled, greedy decoding, seed 5090, and EOS union **[248044, 248046]**. There were **557 prompt + 4,773 generated = 5,330 tokens**; three traces hit the 512-generation-token cap. Training at sequence length 512 retained **5,234 active tokens**, added **2,958 padding tokens**, and right-truncated **96 tokens from three records**. The historical recipe used one epoch, batch/window/stride 1, gradient checkpointing, and learning rate 0.001. Preparation used generation caching, while every compared evaluation explicitly used **no KV cache**, thinking disabled, and a 128-token generation cap. The recorded settings show no thinking-mode mismatch between preparation and evaluation. The small, truncated calibration set remains a limitation, not an established sole cause. The diagnostic inputs are separate from calibration, but their repeated use for diagnosis makes them development data for subsequent methods. [Preparation manifest](D:/openbonsai/runs/2026-10-26-5090-pilot/ayot.jsonl.manifest.json), [replay calibration audit](D:/openbonsai/runs/2026-10-26-5090-diagnosis/replay-training.json), [diagnostic input provenance](D:/openbonsai/runs/2026-10-26-5090-diagnosis/diagnostic-inputs.provenance.json)

One completed fixed-code least-squares scale control improved diagnostic packed NLL to **11.247840** and ARC to **4/20**, but all four answers remained repetitive and irrelevant. It failed its preregistered ARC/qualitative gate; no full-workload confirmation or 128-sample expansion followed that diagnosis. This was an explicit scale-fitting modification with fixed codes, not a reproduced CAT-Q quality result. [Recorded recovery decision](D:/openbonsai/runs/2026-10-26-5090-diagnosis/recovery-decision.json)

The **CPU BITCOS repack** used `--packing auto`; all 200 projections selected BITCOS. The completed audit verified manifest checksums, exact decoded ternary codes/scales/biases, all **564 retained tensors**, configuration/training metadata, and all tokenizer assets. A separate CUDA BF16/eager/no-KV check then matched the original saved five-token-input reference logits exactly (maximum error **0**). That fixed-input check was not a new quality evaluation. Repacking changed storage, not training or the previously measured quality. [CPU verification](D:/openbonsai/runs/2026-10-26-5090-diagnosis/bitcos-verification.json), [independent logit parity](D:/openbonsai/runs/2026-10-26-5090-diagnosis/bitcos-logit-parity.json)

| Historical storage scope | Original 2-bit bytes | BITCOS bytes |
|---|---:|---:|
| Packed code streams | 1,728,053,248 | 1,457,556,523 |
| FP16 scales | 108,003,328 | 108,003,328 |
| Retained tensors and buffers | 4,995,201,904 | 4,995,201,904 |
| Model artifact, including containers/config/manifest | 6,831,783,487 | 6,561,349,799 |
| Supplementary tokenizer assets | 19,998,761 | 19,998,761 |
| Complete model + tokenizer directory | 6,851,782,248 | 6,581,348,560 |

Savings were **270,433,688 bytes**: **3.9585%** of the original packed model artifact, or **3.9469%** of the original deployable directory. BITCOS stores **864,026,624 presence bytes** and **593,529,899 sign bytes**, plus unchanged scales. The three supplementary tokenizer files total 19,998,761 bytes and are already included in the complete-directory row. These historical artifact figures must not be substituted for the new paired mixed-format artifacts, which additionally account for explicit rotation signs and row indexes.

| Retained historical component | Bytes |
|---|---:|
| Output head | 2,034,237,440 |
| Token embeddings | 2,034,237,440 |
| Vision encoder, including vision rotary buffers | 912,021,104 |
| Recurrent gates | 12,582,912 |
| Recurrent convolution | 1,572,864 |
| Text normalization | 546,816 |
| Recurrent dynamics | 3,072 |
| Text rotary buffers | 256 |
| **Total** | **4,995,201,904** |

Vision comprises attention **286,903,296**, MLP **535,781,088**, normalization **248,832**, merger **80,238,080**, patch embedding **3,541,248**, position embedding **5,308,416**, and rotary buffers **144** bytes. The embedding and head alone total **4,068,474,880 bytes**. Storage savings do not imply faster inference: this historical runtime reconstructs dense weights and supplies no optimized low-bit kernel. [Detailed retained accounting](D:/openbonsai/runs/2026-10-26-5090-diagnosis/bitcos-verification.json)

For current cross-run accounting, use the refined denominators: **18,819,635,168 raw represented source bytes**, or **18,819,627,488 loaded persistent BF16 bytes** after the 48 FP32-to-BF16 casts. Exclude **486,581,248 unused MTP bytes** from both denominators and **400 constructor-generated rotary-buffer bytes** from strict source-tensor numerators. This clarification does not rewrite the historical reports. [Completed scope/cast audit](D:/openbonsai/runs/2026-10-26-5090-corrected/corrected16x1/artifact-accounting-audit.json)

This note reads completed small reports/JSON only. The prior preservation audit recorded all **231 pilot file hashes unchanged**, original HEAD `891d260af96a7ce61ccbcb23474015d975bb18dc`, and isolated BITCOS HEAD `997b9c3fa8788bf31b12b1cecb713655a7953361`. No payloads were rehashed or models loaded to prepare this note; prior runs and the current frozen checkout were not modified. [Historical preservation evidence](D:/openbonsai/runs/2026-10-26-5090-diagnosis/final-pilot-preservation.json)
