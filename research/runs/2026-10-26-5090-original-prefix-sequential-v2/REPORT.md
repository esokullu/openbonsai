**The current controlled recovery completed, but useful model quality did not recover.** All 4,096 scientific updates, standalone export, and independent packed reload/evaluation finished. This original-prefix target experiment worsened held-out NLL relative to the matched conditional-prefix parent. Both new representations answer every ARC question “A”; all four generated answers are repetitive and unusable. The failure is already present before FP16 export or packing. No successor experiment was launched.

This is the isolated `2026-10-26-5090-original-prefix-sequential-v2` experiment, with 128 samples and **one epoch per window**, not a 128-sample/3-epoch escalation. It used the explicitly bounded 14,400-second model-process allowance. The original 16-sample artifact, calibration, evaluations, logs, local fixes, and all intervening sealed experiments remain preserved. V1 remains a separately sealed preflight stop; V2 did not reopen or overwrite it.

**Fixed development quality comparison.** Every row below uses the same 2,048 shifted, nonpadding, token-weighted NLL tokens, 20 ARC-Easy questions, and four manual prompts. Historical baseline and conditional-parent results were reused only after source/runtime/template and fresh original/bypass identity gates passed. Peak memory and elapsed time here cover evaluation only; loading, training, and export are separate below.

| Representation | NLL | Perplexity | ARC-Easy | Unfinished / 4 | Evaluation seconds | Peak allocated / reserved GiB |
|---|---:|---:|---:|---:|---:|---:|
| Untouched BF16 (saved) | 2.423777 | 11.29 | 19/20 | 2 | 34.278 | 17.954 / 18.191 |
| Historical conditional128 hard FP32 | 9.121631 | 9,151.11 | 8/20 | 0 | 34.019 | 19.599 / 19.869 |
| Historical conditional128 packed | 9.123808 | 9,171.06 | 8/20 | 0 | 18.230 | 19.460 / 19.846 |
| New original-prefix hard FP32 | 11.827692 | 136,993.93 | 8/20 | 4 | 637.354 | 19.599 / 19.869 |
| New original-prefix fresh packed | 11.849104 | 139,958.89 | 8/20 | 4 | 357.107 | 19.455 / 19.846 |

| New representation minus reference | Absolute ΔNLL | Absolute ΔPPL | ΔARC percentage points |
|---|---:|---:|---:|
| new_hard minus bf16 | +9.403915 | +136,982.64 | -55.0 |
| new_hard minus matched_conditional | +2.706061 | +127,842.82 | +0.0 |
| new_packed minus bf16 | +9.425327 | +139,947.60 | -55.0 |
| new_packed minus matched_conditional | +2.725296 | +130,787.84 | +0.0 |

The eight correct ARC answers are precisely the eight questions whose answer is A. Nonempty generation is not recovery: the hard-FP32 outputs repeat “Sight,” “Cyl,” and zeros; the independently reloaded outputs must also be read rather than judged by token count. Both generated 512 tokens, with four caps and no EOS. Historical conditional-parent outputs were four EOS-only empty answers. Timing differences therefore reflect different output lengths and reference implementations, not useful inference speedups. [All exact answers and token IDs](D:/openbonsai/runs/2026-10-26-5090-original-prefix-sequential-v2/analysis/v2-generations-draft.md).

The original full pilot used a different population: 16,384 tokens, 100 questions, and ten prompts. Its BF16/packed PPL was 9.184705/327,504.720532 and ARC was 97/100 versus 23/100. Those full-pilot scores are not interchangeable with the smaller diagnostic rows. The diagnostic population has been repeatedly inspected; the unused final-confirmation set was not opened.

**Where degradation appears.** The preserved original-pilot stage isolation used the same small diagnostic inputs. Its historical fitting recipe was 16 samples × one epoch, rank 0, no rotation, exponential factors, and MSE; this differs from the current rank-64/sigmoid/Huber recipe:

| Original-pilot stage | NLL | Perplexity | ARC-Easy |
|---|---:|---:|---:|
| Original BF16 | 2.423777 | 11.29 | 19/20 |
| Wrappers installed but quantization disabled | 2.423777 | 11.29 | 19/20 |
| Hard ternary, replayed FP32 factors | 13.256495 | 571,771.57 | 1/20 |
| Export representation with FP16 scales | 13.257692 | 572,456.68 | 1/20 |
| Original pilot packed artifact, fresh process | 13.257692 | 572,456.68 | 1/20 |

Disabled wrappers matched the original logits exactly. Hard ternary reconstruction caused the dominant loss; FP16 scales added ΔNLL +0.001198, and fresh reload added zero in this original-pilot comparison. The original FP32 factors had not been saved: the diagnostic replay reproduced all 512 recorded loss/gradient/stage events and all 200 exported payloads, but cannot prove identity of unavailable original FP32 tensors. Current V2 saves its final FP32 research state.

Forward ablations also localize broad sensitivity: only linear-attention projections gave NLL 10.241759; only MLP projections 10.741537; only linear-attention out_proj 7.324950, versus only in_proj_qkv 2.820819 and only in_proj_z 2.954871. Quantizing only the first four layers already raised NLL to 6.381065. These are controls using factors fitted in the full pipeline, not individually retrained alternative policies. They do not isolate a single causal component. [All 16 ablations, stage receipts, preserved fixes, and original repack evidence](D:/openbonsai/runs/2026-10-26-5090-original-prefix-sequential-v2/analysis/v2-historical-diagnosis-index.json).

The original 16-trace calibration also used greedy nonthinking generation with a 512-token cap: 557 prompt tokens and 4,773 generated tokens, three unfinished capped traces, 5,234 active training tokens and 2,958 padding tokens. Right truncation removed 96 tokens from three traces. These preserved counts are separate from the current UltraChat calibration described below. [Original calibration audit](D:/openbonsai/runs/2026-10-26-5090-diagnosis/REPORT.md:246).

**Current target-repair test.** V2 starts from fresh official BF16 weights. Student inputs come from the current compressed prefix; targets come from an independent original-weight prefix, with the same native masks and kwargs. Window 0 had to reproduce the historical 9B conditional control exactly before window 1. Its final FP32 state, shared update evidence, and native boundary passed. Four preflight updates were discarded with exact learned/runtime/RNG rollback; all 764 original/retained tensors and the original bypass probe stayed exact.

| Search window | Conditional-parent Y/T0 MSE | V2 Y/T0 MSE | Records improved / 16 |
|---:|---:|---:|---:|
| 0 | 0.000311311 | 0.000311311 | 0 |
| 16 | 0.196574567 | 0.104790041 | 16 |
| 31 | 18.120259124 | 6.858338666 | 16 |

All 32 cached hard-MSE diagnostics improved during fitting. Against common original targets on 16 separate search records, MSE fell 46.692% at window 16 and 62.151% at window 31, with all 16 records improving at both. Window 0 is unchanged. Nevertheless, NLL worsened and useful output did not recover. Better hidden-state reconstruction is insufficient evidence of language quality. Signed decomposition terms are descriptive, not causal shares. The independent training audit verifies 3,264 soft and 832 hard completed updates, all finite gradient/update evidence, FP32 moments, exact sample orders and schedules, and actual pre-update-102 frozen-Jacobian snapshots at sharpness 29.8828125. Ledger intentions are not substituted for completed steps. [Training interpretation](D:/openbonsai/runs/2026-10-26-5090-original-prefix-sequential-v2/analysis/v2-training-interpretation.json).

**Export and storage.** The current artifact uses automatic 2-bit/BITCOS selection; all 200 selected candidates use BITCOS. Fresh reload checked exact payloads for all 200 projections and 564 retained tensors, tokenizer/configuration consistency, and bitwise-exact stored-scale packed logits. The following fixed-probe differences separate scale rounding and arithmetic from serialization:

| Comparison | Bitwise exact | Max absolute logit difference | Mean absolute difference |
|---|---|---:|---:|
| fp32_to_fp16_scales_same_wrapper | False | 0.134765625 | 0.0203634035 |
| fp16_wrapper_to_packed_row64 | False | 0.25 | 0.0176641922 |
| decoded_cache_vs_uncached | True | 0 | 0 |
| fresh reload vs stored-scale packed reference | True | 0 | 0 |

V2 ran full quality passes for hard FP32 and fresh packed. Its pure FP16 same-wrapper check is a fixed logit probe only; no unmeasured FP16 NLL/ARC result is inferred.

| Current artifact scope | Bytes |
|---|---:|
| Equivalent original BF16 persistent tensors | 18,819,627,488 |
| Compressed code streams | 1,460,916,374 |
| FP16 scales | 108,003,328 |
| Projection biases | 0 |
| Rotation signs and row offsets | 12,486,208 |
| Retained floating tensors, including 400 generated buffer bytes | 4,995,201,904 |
| Complete model artifact, excluding supplemental tokenizer | 6,577,193,368 |
| Supplemental tokenizer assets | 19,998,761 |
| Total deployable directory | 6,597,192,129 |

The strict persistent-tensor comparison is 6,576,607,414 / 18,819,627,488 bytes: 2.861601× compression and 65.054529% reduction. This numerator includes compressed payloads and all persistent retained tensors. Complete file/directory bytes above additionally include containers/configuration/tokenizer. The equivalent raw checkpoint scope is 18,819,635,168 bytes; 48 retained FP32→BF16 casts account for the 7,680-byte difference. Unused original MTP tensors (486,581,248 bytes) are excluded from both equivalent scopes. Duplicate HF caches, research states, optimizer states, and the 13,824,425,984-byte decoded runtime cache are not deployable tensor payloads.

| Retained component | Bytes |
|---|---:|
| output_head | 2,034,237,440 |
| token_embeddings | 2,034,237,440 |
| text_normalization | 546,816 |
| linear_attention_dynamics | 3,072 |
| linear_attention_convolutions | 1,572,864 |
| linear_attention_gates | 12,582,912 |
| text_rotary_buffers | 256 |
| vision_encoder_including_vision_rotary_buffers | 912,021,104 |

The separate CPU repack of the original failed artifact reduced model files from 6,831,783,487 to 6,561,349,799 bytes (270,433,688 bytes saved). Supplemental tokenizer assets stayed 19,998,761 bytes; deployable totals were 6,851,782,248 and 6,581,348,560 bytes. Exact decoded codes/scales/biases, 564 retained tensors, tokenizer assets, and a separate BF16 CUDA fixed-logit probe passed. No full benchmark was repeated solely for repacking. BITCOS changes storage, not model quality. [Independent current artifact accounting](D:/openbonsai/runs/2026-10-26-5090-original-prefix-sequential-v2/analysis/sequential-artifact-accounting.json).

**Runtime, resources, and reproducibility.** The local RTX 5090 has 32,607 MiB VRAM; driver 596.21, Python 3.12.11, PyTorch 2.14.1+cu130, Transformers 5.20.2. Prelaunch free VRAM was 32,050 MiB, available RAM 129,906,483,200 bytes, and disk 503,312,109,568 bytes. Native Qwen3.5 CUDA/BF16 forward/backward ran throughout preflight and all real-model updates. The reference PyTorch linear-attention kernels are used; these timings do not describe optimized deployment kernels.

| Model process | Seconds |
|---|---:|
| preflight | 191.375 |
| fit | 5912.563 |
| reload | 473.719 |

V2 aggregate model-process time was 6577.657s (109.63 min), below 14,400s. The admission estimate was 11,609.088s including the unchanged 1.25× variable margin and 600s reserve. V1 separately consumed 192.985s with four discarded and zero scientific updates. Combined informational use is 6770.642s with eight discarded and 4,096 scientific updates; V1 was not charged to the V2 cap.

Across recorded V2 phases, peak allocated/reserved GPU memory was 20.217/21.135 GiB; minimum independently computed headroom was 8.997 GiB. All phase headroom gates passed.

| Fitting phase, excluding model setup/export | Seconds |
|---|---:|
| cache | 2775.068 |
| hard_evaluation_before | 215.834 |
| audit_boundary_before | 0.470 |
| optimization | 928.436 |
| freeze_window | 0.007 |
| hard_evaluation_after | 190.934 |
| audit_boundary_after | 0.453 |
| cleanup | 3.979 |

Optimization alone took 928.436s, or 4.412 updates/s; window caching took 2775.068s. Calibration generation added zero new time in V2 because authenticated traces were reused. Loading, conversion, search diagnostics, complete export, hard quality, and fresh loading/verification/evaluation are separately measured in the detailed phase table. CPU audits/reporting are outside the model-process cap. A larger training recipe is not justified by this result; no larger run or runtime forecast is presented as a recovery plan. [Detailed measured and projected phase tables](D:/openbonsai/runs/2026-10-26-5090-original-prefix-sequential-v2/analysis/v2-report-draft.md).

Source: original official `Qwen/Qwen3.5-9B` BF16, immutable revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`, no task fine-tuning. The same tokenizer/chat template is bound by SHA256 `a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715`. Science: 128×32×1 updates; sequence 512, batch/window/stride 1, group 128, signed Hadamard block 4096, rank 64, sigmoid factors, Huber δ1, factor/LoRA LR .005/.001, zero weight decay, clip 1, gradient checkpointing, fresh AdamW per window with zero decay, global initialization seed 5090 and per-window shuffle seed 5090+window. Compute is BF16, learned factors/moments FP32; eager attention, no KV cache, TF32 off, four CPU threads.

Calibration: 128 distinct UltraChat train_sft records at revision `8049631c405ae6576f93f445c6b8166f76f5505a`, greedy nonthinking original-model traces (SDPA/KV enabled during generation), max 512 new tokens. There were 26 EOS completions and 102 capped traces; 105 records were right-truncated to training length 512, removing 7,049 tokens and leaving 59,855 active positions per window. The 16 never-fit search records have 7,203 active positions, four EOS/12 capped traces, and 12 truncated records removing 715 tokens. Truncation is a limitation, not an established causal explanation.

Evaluation: WikiText-2 raw test revision `b08601e04326c79dfdd32d625aee71d232d685c3` (four 513-token blocks → 2,048 scored tokens); ARC-Easy validation revision `210d026faf9955653af8916fad021475a3f00453`, seed 5090, 20 fixed IDs; manual IDs 00/02/04/07 cover instructions, reasoning, code, and Turkish. Identical chat settings, greedy nonthinking decoding, 128 new tokens, EOS IDs 248044/248046. Labels are shifted, padding excluded, NLL aggregated by token count with unchanged FP32 CE/FP64 sums. [Exact inputs and IDs](D:/openbonsai/runs/2026-10-26-5090-original-prefix-sequential-v2/inputs/diagnostic-inputs.json).

The final CPU source gate passed 915 tests with two skipped. An earlier CPU-only attempt was rejected by the source-stability guard while the provenance helper was corrected; its logs are preserved, and no model process had started. V2 changed only run identity, explicit budget, prior-attempt reporting, and a narrow sealed pytest-cache inventory convention in operational helpers; scientific code matches V1. The post-run report helper received a tested numeric parsing fix for zero-padded manifest byte strings (19 synthetic checks). No running model code changed.

Earlier preserved implementation fixes cover frozen-weight storage transfer, bounded FP32 group statistics and hard reconstruction, FP32 factors/Jacobian snapshots, snapshot cleanup, explicit tokenizer list returns, combined EOS stopping, alias-safe CPU export, and meta-parameter reload with initialized buffers. BITCOS was integrated at a safe isolated boundary with these fixes retained. Main remains at `891d260af96a7ce61ccbcb23474015d975bb18dc` with all 39 local source changes preserved; no push or artifact upload occurred.

Post-exit audits rechecked all 182 frozen files, all 39 dirty-main source files, and six complete sealed ancestors (3,865 files, 17,549,350,928 bytes). The runtime freeze is SHA256 `717468370847ab1c3405589cf797f9d6fc26c450c57d0cce02290b980875025b`. Reproduction commands, isolated sources, environment/provenance receipts, raw step logs, final FP32 state, full packed artifact, calibration traces, and exact generated answers are retained under this run. [Experiment plan and command](D:/openbonsai/runs/2026-10-26-5090-original-prefix-sequential-v2/experiment-plan.json); [Execution audit](D:/openbonsai/runs/2026-10-26-5090-original-prefix-sequential-v2/analysis/final-execution-audit.json); [Preservation audit](D:/openbonsai/runs/2026-10-26-5090-original-prefix-sequential-v2/analysis/final-preservation.json).

The evidence establishes a completed local software pipeline and exact storage reload, alongside severe model-quality failure. It does not establish lossless compression, paper-quality reproduction, a unique failure mechanism, or that more samples/epochs will fix the model. Work stops after this report and seal.
