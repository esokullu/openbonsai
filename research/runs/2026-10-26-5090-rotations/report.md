# OpenBonsai Qwen3.5-9B: completed small-pilot diagnosis and paired recovery experiment

**The artifacts serialize correctly, but model quality has not recovered. Do not advance to 128 samples.** The failure is already present in the evaluated hard ternary state, before FP16 scale export or packing. The completed matched 16-sample × 8-epoch rotation experiment improves NLL and ARC substantially relative to its unrotated control, yet every compressed generation remains an irrelevant loop. These are preliminary results on repeatedly used development diagnostics, not evidence of lossless compression or a reproduction of the paper.

All work ran on the local RTX 5090. The failed original pilot, calibration traces, results, logs, and local fixes remain preserved. Both new conditions finished training, export, and independent-process reload/evaluation. No 128-sample run, full-recipe run, cloud job, push, or model upload was started. The GPU was released after the final reload at 10:59:36 UTC on 2026-10-26.

## Comparable quality results

The current fixed subset has **2,048 scored WikiText tokens, 20 ARC-Easy questions, and four instruction/reasoning/coding/Turkish prompts**. BF16 results are reused from the identical frozen protocol; new original-model probes in both conditions match the saved baseline logits exactly. Runtime for that baseline is historical. Table size uses equivalent persistent tensor scope, including all compressed payloads and retained source tensors, excluding file overhead/tokenizer/generated buffers.

| Model on fixed diagnostic subset | Equivalent tensor payload, decimal GB | NLL | Perplexity | ARC-Easy | Unfinished / useful complete answers | Evaluation peak allocated / reserved GiB | Resident evaluation seconds |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original BF16 | 18.819627488 | 2.423777 | 11.288 | 19/20 (95%) | 2/4; 2/4 useful | 17.954 / 18.191 | 34.278 |
| Unrotated 16×8, packed fresh reload | 6.571885396 | 14.437115 | 1,861,912.383 | 1/20 (5%) | 4/4; 0/4 useful | 19.454 / 19.750 | 284.163 |
| Rotated 16×8, packed fresh reload | 6.576191430 | 9.255346 | 10,460.334 | 8/20 (40%) | 4/4; 0/4 useful | 19.457 / 19.846 | 340.528 |

Rotated minus unrotated: **−5.181769 NLL, −1,851,452.049 perplexity, +35 percentage points ARC**. Rotated minus BF16: **+6.831569 NLL, +10,449.045 perplexity, −55 points ARC**. Twenty questions are too few for a broad capability claim. Evaluation times exclude loading/cache preparation; baseline generates 341 tokens versus 512 for each compressed stage, so these are not equal-token kernel benchmarks.

| Representation | Unrotated NLL / ARC | Rotated NLL / ARC |
|---|---:|---:|
| Hard ternary, learned FP32 state | 14.439949 / 1 of 20 | 9.257334 / 8 of 20 |
| Same ternary codes, FP16 scales/biases, same wrapper GEMM | 14.438327 / 1 of 20 | 9.253318 / 8 of 20 |
| Independently reloaded packed artifact | 14.437115 / 1 of 20 | 9.255346 / 8 of 20 |

“FP32” here describes quantizer factors/statistics/corrections, with BF16 model computation. FP16 casting changes NLL by −0.001623 and −0.004016 respectively. The packed runtime's row-chunked GEMMs introduce additional small BF16 rounding differences from wrapper GEMMs. Within that packed runtime, saved-reference versus fresh-reload logits are **bitwise exact, maximum error 0**, for both conditions. Cached versus uncached reference execution is also exact under the same row-64 calls. No quality collapse is introduced at serialization.

All 12 unrotated answers are exactly `" number"` repeated 128 times. All 12 rotated answers are exactly `"Here\n\n" + "This\n\n" × 63`, regardless of the question or representation. Neither reaches EOS. For example, the kitchen-organization instruction, arithmetic problem, Python coding request, and formal Turkish rewrite all receive the same loop. The BF16 kitchen and Turkish answers are useful and complete; its arithmetic and coding answers are relevant but truncated at the same 128-token limit. [Full verbatim answers](D:/openbonsai/runs/2026-10-26-5090-rotations/analysis/quality-generations.md), [manual review](D:/openbonsai/runs/2026-10-26-5090-rotations/analysis/manual-quality-review.md), [all quality metrics and absolute differences](D:/openbonsai/runs/2026-10-26-5090-rotations/analysis/quality-summary.md).

## Original pilot and localization of failure

The preserved original pilot completed the larger **16,384-token / 100-question / 10-generation** evaluation. BF16 NLL/PPL were **2.217540 / 9.184705**, ARC **97/100**; packed NLL/PPL were **12.699258 / 327,504.72**, ARC **23/100**, with ten whitespace-only outputs reaching the cap. Those values must not be mixed with the smaller diagnostic table above.

On the same small diagnostic inputs subsequently used for diagnosis:

| Original-pilot diagnostic state | NLL | ARC-Easy |
|---|---:|---:|
| Original BF16 | 2.423777 | 19/20 |
| All 200 wrappers installed, quantization disabled | 2.423777 | 19/20 |
| Hard ternary with recovered FP32 learned state | 13.256495 | 1/20 |
| Same hard representation with FP16 scales | 13.257692 | 1/20 |
| Original saved artifact independently reloaded | 13.257692 | 1/20 |

Disabled wrappers match the baseline logits exactly. The original pilot did not save its FP32 factors: a **separate diagnostic replay**, without overwriting the original, reproduced all 200 exported payloads and all 512 recorded training events exactly. This establishes agreement with the saved representation and recorded trajectory, not equality of unavailable original FP32 tensors.

Inference-only ablations using those fitted factors show distributed sensitivity: MLP-only NLL **10.741537**, recurrent/linear-attention-only **10.241759**, full-attention-only **4.390061**. MLP `gate_proj` (**6.270979**) and recurrent `out_proj` (**7.324950**) are more damaging than `up_proj` (**3.346235**) and `in_proj_qkv` (**2.820819**) under this state. Quantizing the first 4/8/16 layers yields NLL **6.381065 / 8.830579 / 11.363617**. These are not separately retrained policies and do not identify a unique cause. Their accumulating hidden-state drift supports a failure of end-to-end fidelity.

A small fixed-code least-squares scale recovery reached NLL **11.247840**, ARC **4/20**, but still generated irrelevant loops. A later corrected rank-64 **16×1** run reached packed NLL **11.191338**, ARC **1/20**, also repetitive. Its exp/MSE/LR0.001 recipe differs from the current sigmoid/Huber/LR0.005 pair, so it is not a matched rotation control. [Historical diagnosis, ablations, recovery, and preservation details](D:/openbonsai/runs/2026-10-26-5090-rotations/analysis/prior-diagnosis-and-repack.md).

## Calibration, method, and training evidence

OpenBonsai replaces selected decoder projections with groupwise learned ternary reconstructions and fits each window to its original-weight output on inputs from the already-quantized prefix. This experiment retained floating embeddings, output head, vision, small recurrent gates, convolutions, norms, and dynamics. **MLP `gate_proj` is quantized**; it is distinct from the small retained recurrent gates.

Both conditions start independently from official **Qwen/Qwen3.5-9B**, immutable revision **`c202236235762e1c871ad0ccb60c8ee5ba337b9a`**, with no task fine-tuning. They use the same 16 distinct original-model AYOT traces, 512-token sequences, batch/window/stride 1, eight epochs, group128, rank64, sigmoid factors, Huber delta1, factor LR0.005, correction LR0.001, seed5090, BF16 CUDA and gradient checkpointing. Per window: 102 soft +26 hard updates, frozen Jacobian at the last actual soft pre-update state (sharpness29.8828125). Each condition records **4,096 finite valid updates**, including **3,264 soft and 832 hard**, valid gradients and FP32 optimizer/snapshot state. Original-teacher bypass remains exact before and after optimization.

Rotation alone differs within the matched pair: normalized local Hadamard blocks4096 with deterministic per-projection signs, native ordering, FP32 complete-row weight transformation before correction/grouping. For row vectors, `R = D H`, with inputs `xR` and weights `WR`. It does not move transforms through nonlinearities, recurrence, RoPE, or norms. No persistent full rotated-weight copy is used.

All 32 local hard reconstruction MSEs improve in both conditions, while final-layer relative RMSE/cosine remain **1.09071 / 0.29393 unrotated**, **1.07104 / 0.38213 rotated**. Local reconstruction improvement therefore does not imply preserved end-to-end quality. On bounded pretraining samples, median per-module original-teacher input kurtosis changes **615.746→2.944** under rotation; median sampled weight kurtosis **3.253→3.058**. These describe sampled original-teacher inputs/weights, not trained student activation distributions or a proven causal explanation. [Training and distribution evidence](D:/openbonsai/runs/2026-10-26-5090-rotations/analysis/distribution-comparison.md), [method and fixes](D:/openbonsai/runs/2026-10-26-5090-rotations/analysis/method-and-fixes.md).

Calibration had **557 prompt +4,773 generated =5,330 source tokens**. Three traces hit the 512-new-token cap; sequence preparation retained **5,234 active tokens**, added **2,958 padding positions**, and truncated **96 tokens across three records**. Thinking was disabled in both calibration and evaluation; generation was greedy with EOS IDs **[248044,248046]**. There is no recorded thinking-setting mismatch. Small coverage and truncation are limitations, not an established sole cause. No additional traces were generated for this pair.

NLL uses shifted labels, masks padding in both source and target positions, and aggregates summed loss by scored-token count. WikiText uses `Salesforce/wikitext`, `wikitext-2-raw-v1`, test revision **`b08601e04326c79dfdd32d625aee71d232d685c3`**; ARC uses `allenai/ai2_arc`, ARC-Easy validation revision **`210d026faf9955653af8916fad021475a3f00453`**. ARC scores conditional answer-label likelihood. Ordered IDs, source row spans, tokenizer/template hashes, seed5090, tokenized prompts, and raw answers are saved. The same fixed four NLL blocks, 20 questions and four prompts were reused without retokenization; diagnostics use eager attention, no KV cache, greedy128-token outputs. These examples are separate from calibration, but repeated diagnosis makes them development data. [Frozen protocol and IDs](D:/openbonsai/runs/2026-10-26-5090-rotations/analysis/quality-summary.json).

## Storage: whole equivalent scope and actual deployable bytes

All 200 selected projections chose BITCOS after comparison of complete candidate file sizes. These are bytes, not disk allocation units. Research FP32 states, logs, caches, source archives and duplicate tokenizer staging are excluded.

| Current artifact component | Unrotated bytes | Rotated bytes |
|---|---:|---:|
| Compressed code streams | 1,457,275,700 | 1,460,500,390 |
| FP16 scales | 108,003,328 | 108,003,328 |
| Selected projection biases | 0 | 0 |
| Row sign indexes | 11,404,864 | 11,404,864 |
| Explicit rotation signs | 0 | 1,081,344 |
| Retained floating tensors/buffers | 4,995,201,904 | 4,995,201,904 |
| Container/manifest/config overhead | 553,092 | 585,538 |
| **Model artifact excluding tokenizer** | **6,572,438,888** | **6,576,777,368** |
| Supplementary tokenizer assets | 19,998,761 | 19,998,761 |
| **Complete deployable directory** | **6,592,437,649** | **6,596,776,129** |

The equivalent persistent BF16 source denominator is **18,819,627,488 bytes**. Against all compressed candidate tensors plus retained persistent source tensors, reduction is **65.079620% (2.863657×) unrotated** and **65.056740% (2.861782×) rotated**. Raw represented checkpoint tensors total **18,819,635,168 bytes**; 48 tiny FP32 source tensors cast to BF16 explain the 7,680-byte difference. **15 unused MTP tensors, 486,581,248 bytes**, are excluded from both equivalent-scope denominators. Four constructor-generated rotary buffers totaling **400 bytes** are included in files but excluded from strict source-tensor numerators. No duplicated Hugging Face snapshots are counted.

A separate constructed comparison adds identical saved config/tokenizer byte allowances to the loaded BF16 tensor denominator: **18,839,632,259 bytes** versus complete deployable directories, giving **65.007610% / 64.984581%** reductions. This is explicitly a tensor-plus-assets allowance, **not a measurement of a separately serialized equivalent BF16 directory**. [Exact audited arithmetic, payload hashes, and scope checks](D:/openbonsai/runs/2026-10-26-5090-rotations/analysis/paired-artifact-audit.json).

| Retained component, same for all these artifacts | Bytes |
|---|---:|
| Output head | 2,034,237,440 |
| Token embeddings | 2,034,237,440 |
| Vision encoder including rotary buffers | 912,021,104 |
| Small recurrent gates | 12,582,912 |
| Recurrent convolution | 1,572,864 |
| Text normalization | 546,816 |
| Recurrent dynamics | 3,072 |
| Text rotary buffers | 256 |
| **Total** | **4,995,201,904** |

The head and embeddings alone cost **4,068,474,880 bytes**. Vision further divides into attention286,903,296; MLP535,781,088; normalization248,832; merger80,238,080; patch embedding3,541,248; position embedding5,308,416; rotary buffers144 bytes. These retained components prevent this configuration from approaching a 2GB complete artifact.

### Separate CPU repack of the preserved failed pilot

BITCOS was integrated in an isolated checkout at **`997b9c3fa8788bf31b12b1cecb713655a7953361`**, preserving memory/chunking/tokenizer/stopping fixes. The requested `repack --packing auto` ran on CPU into a new directory. Exact decoded codes, scales, biases, all 564 retained tensors, tokenizer assets and relevant configuration were verified; a later independent CUDA check matched the saved fixed-input logits with **zero error**. Existing quality results therefore carry over; no full benchmark was repeated solely for repacking.

| Original failed pilot storage | Before BITCOS bytes | After BITCOS bytes |
|---|---:|---:|
| Model artifact excluding tokenizer | 6,831,783,487 | 6,561,349,799 |
| Supplementary tokenizer | 19,998,761 | 19,998,761 |
| Complete deployable directory | 6,851,782,248 | 6,581,348,560 |

Saved **270,433,688 bytes**, **3.9585%** of the prior packed model artifact, or **3.9469%** of its whole deployable directory. This operation changes storage only. [Executed CPU command](D:/openbonsai/runs/2026-10-26-5090-diagnosis/bitcos-commands.txt), [tensor/asset verification](D:/openbonsai/runs/2026-10-26-5090-diagnosis/bitcos-verification.json), [independent logit parity](D:/openbonsai/runs/2026-10-26-5090-diagnosis/bitcos-logit-parity.json).

## Measured cost and implementation validation

| Phase, seconds | Unrotated | Rotated |
|---|---:|---:|
| New calibration generation | 0 (reused) | 0 (reused) |
| Original BF16 load | 15.63 | 17.89 |
| Export-plan preparation / wrapper conversion | 24.34 / 1.12 | 25.65 / 3.99 |
| Training window cache capture | 116.96 | 324.06 |
| Optimization | 586.98 | 930.73 |
| Total trainer including probes/audits | 731.30 | 1,309.05 |
| Hard-FP32 / FP16-wrapper evaluation | 213.29 / 217.60 | 641.25 / 643.45 |
| Candidate construction / file writing / verification | 441.15 / 141.34 / 90.32 | 584.97 / 132.44 / 84.22 |
| Export-reference decoded cache preparation | 25.80 | 25.75 |
| Fresh cold packed load / decoded cache preparation | 70.53 / 26.69 | 67.28 / 26.10 |
| Fresh packed evaluation | 284.16 | 340.53 |
| **Combined real + reload process time** | **2,313.55 (38.56 min)** | **3,959.52 (65.99 min)** |

Trainer subphases overlap total training and must not be summed again. Combined process time includes additional setup, probes and Python overhead; preflights and final CPU audits are excluded. The prelaunch forecast was about35+55=90 minutes; actual drivers totaled **104.55 minutes**, within the broad1–2.5-hour allowance. Rotated wrapper evaluation and candidate construction were major costs, so training-step throughput alone would underestimate total runtime. [All phase timings and allocated/reserved peaks](D:/openbonsai/runs/2026-10-26-5090-rotations/analysis/runtime-summary.md).

Training peaked at **20.207 / 20.297 GiB** allocated/reserved unrotated and **20.221 / 20.297 GiB** rotated. Metrics files use the maximum synchronized trainer phase peaks because nested resets make the enclosing console counter incomplete. Fresh inference passed the full513-token memory gates under28GiB in both conditions. The packed runtime uses an optional nonpersistent **13,824,425,984-byte decoded BF16 weight cache**, retaining exact row64 GEMMs. It is excluded from artifact bytes and increases inference VRAM: packed evaluation is about19.46GiB allocated versus17.95GiB BF16. Cold packed load alone is about6.16GiB and is not a full inference-memory measure.

Generation throughput was **11.03 tokens/s BF16**, **1.89 unrotated packed**, **1.58 rotated packed**, under the recorded no-KV reference runtime and different stopping lengths. Causal-convolution and DeltaNet reference fallbacks were used consistently. No optimized low-bit kernel or speedup follows from file-size savings. A proposed row512 runtime was rejected because BF16 logits differed; the cache preserves row64 computation and passed exact cached/uncached checks before either matched run.

The isolated environment was Python3.12.11, PyTorch2.14.1+cu130, Transformers5.20.2, CUDA13.0, driver596.21, RTX5090 compute capability12.0 (32,607MiB reported). Final resources: approximately129.1GiB free RAM,485.4GiB free disk; RTX5090 returned to137MiB used. **639 CPU tests passed, two Windows symlink-privilege skips**. Native CUDA hybrid checks covered BF16 forward/backward, both quantization stages, actual projection shapes, rank64 FP32 gradients/moments/snapshots, teacher bypass, floating rotation placement, export, independent reload, tokenizer parity, and both EOS stop IDs. Full real-9B resident-prefix resource preflights preceded the paired runs.

Preserved fixes include ownership transfer without duplicate original weights, aligned complete-row FP32 chunks, correct low-rank and frozen-Jacobian state, padding-aware loss, hybrid-mask/checkpoint handling, bounded export and meta reload, tokenizer/generation config preservation, and EOS union stopping. Source-dtype reconstructed dense weights and transient concatenation still exist; this is bounded-intermediate memory work, not a claim of zero full-weight reconstruction. [Detailed fixes and preflights](D:/openbonsai/runs/2026-10-26-5090-rotations/analysis/method-and-fixes.md).

## Reproduction, preservation, and decision

Original workspace HEAD remains **`891d260af96a7ce61ccbcb23474015d975bb18dc`** with its preexisting fixes preserved. The paired isolated checkout is based on **`ff5bbf10e94d5d35282b9dfa48fa13069289fbfd`**, with the exact final integrated sources and patch archived before execution. Final runtime archive SHA256: **`ba4d28808cf1f7244e0ac9244eb4d0d6a4ce683b90f1f13741dc9caf0a0af87f`**. No source changes occurred beneath a running model process.

Final preservation verification passed **1,412 manifest-reference checks across1,379 unique historical files**, including all231 original-pilot files, diagnosis artifacts, earlier corrected artifacts, local-source snapshots and baseline anchors. All33 frozen runtime Python files remain unchanged. Each new artifact passed all769 listed file hashes plus770-file inventory,200 candidate checksums, retained descriptors/value checks, and fresh logit parity. [Preservation record](D:/openbonsai/runs/2026-10-26-5090-rotations/analysis/preservation-final.json), [commands, environment, snapshots and reproduction](D:/openbonsai/runs/2026-10-26-5090-rotations/reproduction.md).

**Recovery gate: failed.** Rotation provides useful numerical evidence of improvement within the matched recipe, but its prompt-independent loops and large remaining NLL gap preclude escalation. More samples or epochs are not established remedies. Any next investigation should remain a small controlled test of cumulative reconstruction error and the sensitive projection families, with a nondegenerate-generation gate and subsequently fresh confirmation data before increasing budget. Nothing further was launched. Text-only diagnostics also leave vision quality and the eventual BF16 task-fine-tuning → compression workflow unvalidated.
