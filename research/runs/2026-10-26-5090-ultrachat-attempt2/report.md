This small-pilot UltraChat comparison completed both rotated arms and their independent packed reloads. Final packed quality on the repeated development subset is shown below; serialization correctness and compression accounting passed separately.

| Representation | Equivalent-scope tensor bytes | NLL | Perplexity | ARC | Evaluation peak allocated / reserved GiB | Evaluation seconds | Generated tokens |
|---|---|---|---|---|---|---|---|
| Original BF16 (historical) | 18,819,627,488 | 2.423777 | 11.288 | 19/20 | 17.954 / 18.191 | 34.278 | 341 |
| rotated16x8 (fresh packed) | 6,576,199,974 | 8.498620 | 4907.991 | 8/20 | 19.459 / 19.826 | 21.618 | 7 |
| rotated128x1 (fresh packed) | 6,576,245,389 | 9.123808 | 9171.055 | 8/20 | 19.460 / 19.846 | 18.230 | 4 |

The original byte figure is BF16-loaded persistent tensor storage, not a measured original deployment directory. Compressed figures use the same represented scope, including all candidate signs/scales/indexes and retained source tensors, excluding generated 400-byte buffers and file/tokenizer overhead. Packed runtime memory includes its decoded BF16 cache. Baseline evaluation timing is historical; different generated-token counts prevent an equal-work speed comparison.

128×1 minus 16×8: NLL +0.625188; ARC +0.0 percentage points. rotated16x8: No quality recovery:0/4 complete useful answers; two empty EOS-only replies and two nonanswers (Write a and #), despite4/4 EOS endings. rotated128x1: No quality recovery:0/4 complete useful answers; all four prompts receive immediate EOS-only empty replies.

Both arms use the same quantizer and 128 updates per window. This compares 128 distinct generated calibration records once against a frozen nested 16 repeated eight times, with almost matched active-token exposure. It is not a rotation ablation or a task fine-tuning comparison.

The quality protocol uses 2,048 scored NLL tokens, 20 ARC-Easy questions and four prompts, greedy nonthinking decoding with a 128-new-token cap, both EOS IDs, eager attention and no KV cache. These are repeatedly inspected development diagnostics, not a fresh confirmation set or the historical 16,384-token/100-question/10-prompt full evaluation. The historical BF16 baseline is reused only with exact current original-probe parity.

| Representation | NLL | Perplexity | ARC | Useful complete /4 | Relevant capped /4 | Empty /4 | Other failures /4 | EOS/cap outputs | Generated tokens |
|---|---|---|---|---|---|---|---|---|---|
| Original BF16 (historical) | 2.423777 | 11.288 | 19/20 | 2 | 2 | 0 | 0 | 2/2 | 341 |
| rotated16x8: Hard FP32 factors | 8.499410 | 4911.869 | 8/20 | 0 | 0 | 2 | 2 | 4/0 | 7 |
| rotated16x8: FP16 scales, same wrapper GEMM | 8.500027 | 4914.900 | 8/20 | 0 | 0 | 2 | 2 | 4/0 | 7 |
| rotated16x8: Packed, fresh reload, decoded cache | 8.498620 | 4907.991 | 8/20 | 0 | 0 | 2 | 2 | 4/0 | 7 |
| rotated128x1: Hard FP32 factors | 9.121631 | 9151.112 | 8/20 | 0 | 0 | 4 | 0 | 4/0 | 4 |
| rotated128x1: FP16 scales, same wrapper GEMM | 9.121127 | 9146.503 | 8/20 | 0 | 0 | 4 | 0 | 4/0 | 4 |
| rotated128x1: Packed, fresh reload, decoded cache | 9.123808 | 9171.055 | 8/20 | 0 | 0 | 4 | 0 | 4/0 | 4 |

| Representation | Predicted letters | Answer keys |
|---|---|---|
| Original BF16 (historical) | A=7, B=7, C=4, D=2 | A=8, B=7, C=4, D=1 |
| rotated16x8: Packed, fresh reload, decoded cache | A=20, B=0, C=0, D=0 | A=8, B=7, C=4, D=1 |
| rotated128x1: Packed, fresh reload, decoded cache | A=20, B=0, C=0, D=0 | A=8, B=7, C=4, D=1 |

rotated16x8 predicts A on all 20 questions: its 8/20 equals the 8 A-labeled answer keys. This is an answer-letter collapse, not broad retained QA competence. rotated128x1 predicts A on all 20 questions: its 8/20 equals the 8 A-labeled answer keys. This is an answer-letter collapse, not broad retained QA competence.

The 16×8 answer-letter collapse and fixed-input hidden-state comparisons are documented in [the failure analysis](<D:/openbonsai/runs/2026-10-26-5090-ultrachat-attempt2/analysis/rotated16x8-failure-analysis.md>). Its historical comparisons are descriptive and do not identify a causal explanation.

EOS and nonblank text alone do not establish usefulness. Counts above come from [manual inspection of all generations](<D:/openbonsai/runs/2026-10-26-5090-ultrachat-attempt2/analysis/manual-quality-review.md>). All four prompts and all seven representations' unaltered texts/IDs are in [the generation record](<D:/openbonsai/runs/2026-10-26-5090-ultrachat-attempt2/analysis/quality-generations.md>). Raw evaluation files: [Original BF16 (historical)](<D:/openbonsai/runs/2026-10-26-5090-ultrachat-attempt2/inputs/baseline-original.json>), [rotated16x8: Hard FP32 factors](<D:/openbonsai/runs/2026-10-26-5090-ultrachat-attempt2/rotated16x8/hard-fp32-evaluation.json>), [rotated16x8: FP16 scales, same wrapper GEMM](<D:/openbonsai/runs/2026-10-26-5090-ultrachat-attempt2/rotated16x8/exported-scales-evaluation.json>), [rotated16x8: Packed, fresh reload, decoded cache](<D:/openbonsai/runs/2026-10-26-5090-ultrachat-attempt2/rotated16x8/packed-evaluation.json>), [rotated128x1: Hard FP32 factors](<D:/openbonsai/runs/2026-10-26-5090-ultrachat-attempt2/rotated128x1/hard-fp32-evaluation.json>), [rotated128x1: FP16 scales, same wrapper GEMM](<D:/openbonsai/runs/2026-10-26-5090-ultrachat-attempt2/rotated128x1/exported-scales-evaluation.json>), [rotated128x1: Packed, fresh reload, decoded cache](<D:/openbonsai/runs/2026-10-26-5090-ultrachat-attempt2/rotated128x1/packed-evaluation.json>).

Calibration uses `HuggingFaceH4/ultrachat_200k` / `train_sft`, commit `8049631c405ae6576f93f445c6b8166f76f5505a`. The frozen selection inspected 5,000 rows, found 3,640 eligible first-user prompts and selected 128 primary plus 32 reserve prompts with seed 5090. Supplied assistant answers and later turns were discarded. Prompt eligibility used the unchanged nonthinking template length 20–160; it was not a quality filter.

Original BF16 generation accepted 128 distinct traces in 128 attempts; 0 reserves were used. It generated 58,089 tokens from 8,815 prompt tokens. 26 traces ended with EOS and 102 reached the 512-new-token cap. Right truncation of the complete prompt-plus-continuation to 512 positions removed 7,049 tokens from 105 records; right padding uses 248044 with mask zero. All retained prompt and continuation positions are active during training; right-padding positions are excluded.

| Calibration quantity | rotated16x8 | rotated128x1 |
|---|---|---|
| Distinct records | 16 | 128 |
| Epochs | 8 | 1 |
| Single-pass active positions | 7,482 | 59,855 |
| Active exposures per window | 59,856 | 59,855 |
| Prompt exposures per window | 9,632 | 8,815 |
| Continuation exposures per window | 50,224 | 51,040 |
| Padding exposures per window | 5,680 | 5,681 |
| Raw EOS / records | 3 | 26 |
| Raw cap / records | 13 | 102 |
| Truncated records | 13 | 105 |
| Truncated tokens | 927 | 7,049 |
| EOS removed by truncation | 0 | 3 |

The deterministic nested selector chose one record from each of 16 length bins of eight, minimizing the exact exposure difference and retaining parent order. Its independent optimum/tie-break audit passed. The repeated arm has 59,856 active exposures per window versus 59,855; absolute difference 1, or 0.001671%. Diversity, repetition and ordering differ; this is not a pure causal isolation of diversity.

Source records, all generated traces, masks and selection evidence: [source selection](<D:/openbonsai/runs/2026-10-26-5090-ultrachat-attempt2/data/source-selection.json>), [selected prompts](<D:/openbonsai/runs/2026-10-26-5090-ultrachat-attempt2/data/selected-prompts.jsonl>), [128 raw traces](<D:/openbonsai/runs/2026-10-26-5090-ultrachat-attempt2/data/ayot128.jsonl>), [16 raw traces](<D:/openbonsai/runs/2026-10-26-5090-ultrachat-attempt2/data/ayot16.jsonl>), [128 consumed masks](<D:/openbonsai/runs/2026-10-26-5090-ultrachat-attempt2/data/consumed128.jsonl>), [16 consumed masks](<D:/openbonsai/runs/2026-10-26-5090-ultrachat-attempt2/data/consumed16.jsonl>), [per-attempt traces](<D:/openbonsai/runs/2026-10-26-5090-ultrachat-attempt2/data/traces>), [attempt log](<D:/openbonsai/runs/2026-10-26-5090-ultrachat-attempt2/data/generation-attempts.jsonl>), [calibration audit](<D:/openbonsai/runs/2026-10-26-5090-ultrachat-attempt2/analysis/generation-calibration-audit.md>), [calibration freeze](<D:/openbonsai/runs/2026-10-26-5090-ultrachat-attempt2/data/calibration-freeze.json>).

Training used fresh official Qwen/Qwen3.5-9B BF16 sources per arm, group size 128, rank-64 correction, sigmoid factor mapping, fixed signed Hadamard block 4096/seed 5090, Huber δ=1, factor LR .005, LoRA LR .001, batch one and sequence 512. Each of 32 windows used 102 soft then 26 hard updates, retaining the last actual pre-update soft snapshot at k=29.8828125. The independent 512-update four-active-layer preflights retained all 32 layers/200 projections on GPU and discarded their learned states.

| Training check | rotated16x8 | rotated128x1 |
|---|---|---|
| Audited updates | 4096 | 4096 |
| Completed windows | 32 | 32 |
| Soft / hard updates | 3264 / 832 | 3264 / 832 |
| Training audit passed | True | True |
| Preflight projected training seconds | 1233.994 | 3310.642 |

Initial source weights, factors, statistics, rotation signs and CPU/CUDA RNG fingerprints match across arms. The paired audit verifies finite FP32 parameters/gradients/optimizer moments/snapshots and the declared schedule. Local reconstruction improvement does not imply downstream quality recovery. The first calibration activation sample can differ across arms; cross-arm activation-distribution differences are not a direct transform effect. Fixed NLL hidden-state probes are comparable. [Detailed training audit](<D:/openbonsai/runs/2026-10-26-5090-ultrachat-attempt2/analysis/paired-training-audit.json>).

Independent saved-checkpoint audits verify all 200 projections, 1,000 finite FP32 factor/LoRA tensors, 200 unchanged sign tensors, and all 160 window/family aggregates against their final training steps. They validate the saved real-process state; reload correctness is checked separately. [rotated16x8 learned-state audit](<D:/openbonsai/runs/2026-10-26-5090-ultrachat-attempt2/analysis/rotated16x8-real-completed-training-audit.md>) · [rotated128x1 learned-state audit](<D:/openbonsai/runs/2026-10-26-5090-ultrachat-attempt2/analysis/rotated128x1-real-completed-training-audit.md>).

| Representation comparison | rotated16x8 | rotated128x1 |
|---|---|---|
| fp32_to_fp16_scales_same_gemm | exact=False; max=0.125 | exact=False; max=0.125 |
| fp16_wrappers_to_packed_row_chunk_runtime | exact=False; max=0.09375 | exact=False; max=0.08984375 |
| fp32_to_exported_reference | exact=False; max=0.15625 | exact=False; max=0.109375 |

All 200 exported projection payloads and 564 retained tensors match the exporter reference. Independent streams verify candidate/internal checksums, 560 retained original/cast value hashes and all explicit rotation signs. Cached versus uncached probes and each fresh reload are bitwise exact. This does not claim that hard FP32, FP16-scale wrappers and row-chunk packed logits are identical; the table preserves their distinct rounding/runtime changes.

| Component, bytes | rotated16x8 | rotated128x1 |
|---|---|---|
| Legacy 2-bit code stream | 0 | 0 |
| BITCOS presence | 864,026,624 | 864,026,624 |
| BITCOS nonzero-code signs | 596,482,310 | 596,527,725 |
| FP16 scales | 108,003,328 | 108,003,328 |
| Bias | 0 | 0 |
| Explicit rotation signs | 1,081,344 | 1,081,344 |
| Row sign indexes | 11,404,864 | 11,404,864 |
| All candidate tensor payload | 1,580,998,470 | 1,581,043,885 |
| Retained linear_attention_convolutions | 1,572,864 | 1,572,864 |
| Retained linear_attention_dynamics | 3,072 | 3,072 |
| Retained linear_attention_gates | 12,582,912 | 12,582,912 |
| Retained output_head | 2,034,237,440 | 2,034,237,440 |
| Retained text_normalization | 546,816 | 546,816 |
| Retained text_rotary_buffers | 256 | 256 |
| Retained token_embeddings | 2,034,237,440 | 2,034,237,440 |
| Retained vision_encoder_including_vision_rotary_buffers | 912,021,104 | 912,021,104 |
| Candidate container prefix_bytes | 3,400 | 3,400 |
| Candidate container header_bytes | 178,728 | 178,728 |
| Candidate container checksum_bytes | 6,400 | 6,400 |
| Retained tensor total (includes generated 400 B) | 4,995,201,904 | 4,995,201,904 |
| Retained safetensors headers | 66,216 | 66,216 |
| Manifest | 324,810 | 324,811 |
| Model config | 3,951 | 3,951 |
| Generation config | 2,059 | 2,059 |
| All model containers/metadata/config | 585,564 | 585,565 |
| Model excluding tokenizer | 6,576,785,938 | 6,576,831,354 |
| Tokenizer assets | 19,998,761 | 19,998,761 |
| Complete deployable directory | 6,596,784,699 | 6,596,830,115 |
| Decoded BF16 runtime cache; not serialized | 13,824,425,984 | 13,824,425,984 |
| FP32 research factors; outside deployment | 648,019,968 | 648,019,968 |
| FP32 research LoRA; outside deployment | 641,728,512 | 641,728,512 |
| Research rotation signs; outside deployment | 1,081,344 | 1,081,344 |
| Research file container; outside deployment | 353,529 | 353,529 |
| FP32 research file total; outside deployment | 1,291,183,353 | 1,291,183,353 |
| Packed tensors + generated buffers + decoded cache (runtime payload) | 20,400,626,358 | 20,400,671,773 |

| Arm | Strict comparison | Compressed tensor bytes | Equivalent original bytes | Reduction | Factor |
|---|---|---|---|---|---|
| rotated16x8 | Raw represented source | 6,576,199,974 | 18,819,635,168 | 65.056709% | 2.861780× |
| rotated16x8 | BF16-loaded persistent source | 6,576,199,974 | 18,819,627,488 | 65.056694% | 2.861778× |
| rotated128x1 | Raw represented source | 6,576,245,389 | 18,819,635,168 | 65.056467% | 2.861760× |
| rotated128x1 | BF16-loaded persistent source | 6,576,245,389 | 18,819,627,488 | 65.056453% | 2.861759× |

The raw equivalent-source denominator is 18,819,635,168 bytes; actual BF16-loaded persistent tensors total 18,819,627,488 bytes. Forty-eight retained FP32→BF16 casts explain the 7,680-byte difference. The official checkpoint also has 486,581,248 MTP bytes unused by the native loader; these are excluded from both strict denominators. Generated rotary buffers (400 bytes) are excluded from strict tensor numerators. Code signs, rotation signs, scales, bias and indexes are included. Complete directory size includes containers/configs and tokenizer once; it is not the strict tensor numerator.

There is no endpoint prototype or endpoint quantization: additional prototype/quantized endpoint parameter bytes are zero. Retained token embeddings and output-head weights cost 2,034,237,440 bytes each and remain included, as does retained vision. Removing vision bytes from an illustration would not produce a validated standalone text-only artifact. The nonpersistent decoded cache adds 13,824,425,984 bytes at runtime; the FP32 research checkpoint is not required by deployed inference.

All payload checks are in [the streamed artifact audit](<D:/openbonsai/runs/2026-10-26-5090-ultrachat-attempt2/analysis/paired-artifact-audit.json>); the exact file partition, codec counts and container subcomponents remain in that JSON. [Attempt lineage accounting](<D:/openbonsai/runs/2026-10-26-5090-ultrachat-attempt2/analysis/attempt2-accounting.md>) separates disk and runtime scope.

| Arm | Timing scope | Preflight projection seconds | Measured seconds | Measured minus projection |
|---|---|---|---|---|
| rotated16x8 | Enclosing training wall time | 1233.994 | 1303.809 | +5.66% |
| rotated16x8 | cache | 250.264 | 323.431 | +29.24% |
| rotated16x8 | optimization | 930.247 | 926.978 | -0.35% |
| rotated16x8 | other | 53.483 | 53.296 | -0.35% |
| rotated128x1 | Enclosing training wall time | 3310.642 | 3965.794 | +19.79% |
| rotated128x1 | cache | 1975.430 | 2604.822 | +31.86% |
| rotated128x1 | optimization | 927.532 | 935.228 | +0.83% |
| rotated128x1 | other | 407.680 | 425.641 | +4.41% |

Measured 128×1/16×8 cache time is 8.053725×, while optimization is 1.008899×. The cache processes eight times as many distinct records per window; optimizer step counts remain equal. Preflight projection approximates deeper prefix work. The enclosing training timer exceeds the summed trainer phases by about 0.104 seconds in each arm; its total is not added to those phases again. These timings describe the observed implementation, not a quality mechanism.

| Arm | Final hidden cosine vs original | Final relative RMSE vs original | Final hidden RMS / teacher RMS |
|---|---|---|---|
| rotated16x8 | 0.411973 | 1.093587 | 1.016696 |
| rotated128x1 | 0.408845 | 1.086154 | 0.997847 |

On the same first NLL input, 128×1 has higher cosine in 4/32 layers and lower relative RMSE in 6/32. Final RMS is closer to the teacher while direction remains substantially altered. This fixed-input descriptive comparison does not explain downstream quality causally. [Full paired drift/timing evidence](<D:/openbonsai/runs/2026-10-26-5090-ultrachat-attempt2/analysis/paired-drift-and-timing.md>).

| Measured process scope | Seconds |
|---|---|
| Previous failed preparation (charged once) | 5.924203 |
| generation | 2218.500588 |
| preflight_rotated16x8 | 152.771101 |
| rotated16x8_real | 2300.718435 |
| rotated16x8_reload | 129.331333 |
| preflight_rotated128x1 | 229.729593 |
| rotated128x1_real | 4937.021908 |
| rotated128x1_reload | 124.337222 |
| Generation + both arm drivers, including prior debit | 9715.833690 |
| Same plus both resource preflights | 10098.334384 |

Shared preparation's internal elapsed time is 2217.661703 seconds; the separately sampled cumulative timer is 2223.594830 seconds and already includes the prior 5.924203 seconds. Initial total estimate was 3.4–4.0 hours; estimates are not substituted for measured process sums.

Generation preparation: loading 17.424 seconds, generation phase 2193.709 seconds, 26.480 generated tokens/s; maximum allocated/reserved 17.678/17.725 GiB.

| Representation | Evaluation seconds | Generated tokens | Peak allocated GiB | Peak reserved GiB |
|---|---|---|---|---|
| Original BF16 (historical) | 34.278 | 341 | 17.954 | 18.191 |
| rotated16x8: Hard FP32 factors | 37.240 | 7 | 19.599 | 19.869 |
| rotated16x8: FP16 scales, same wrapper GEMM | 37.677 | 7 | 19.599 | 19.869 |
| rotated16x8: Packed, fresh reload, decoded cache | 21.618 | 7 | 19.459 | 19.826 |
| rotated128x1: Hard FP32 factors | 34.019 | 4 | 19.599 | 19.869 |
| rotated128x1: FP16 scales, same wrapper GEMM | 34.256 | 4 | 19.599 | 19.869 |
| rotated128x1: Packed, fresh reload, decoded cache | 18.230 | 4 | 19.460 | 19.846 |

Different generation lengths/EOS stopping change total runtime substantially; these are not equal-token speed benchmarks. The BF16 baseline timing is historical, while the current original-model probe validates fixed-logit reuse. Cached packed execution retains BF16 weights and uses the same 64-row floating-point GEMMs, so storage reduction does not imply lower runtime VRAM or faster inference than BF16.

| Arm | Scope | Phase | Seconds | Peak allocated GiB | Peak reserved GiB |
|---|---|---|---|---|---|
| rotated16x8 | main | load_original_bf16 | 15.017 | 17.538 | 17.588 |
| rotated16x8 | main | baseline_probe_and_hidden_reference | 0.548 | 17.705 | 17.812 |
| rotated16x8 | main | prepare_original_export_plan | 24.791 | 17.569 | 17.619 |
| rotated16x8 | main | conversion_rank64 | 3.199 | 19.366 | 19.412 |
| rotated16x8 | main | initial_state_fingerprints | 1.363 | 19.183 | 19.207 |
| rotated16x8 | main | pretraining_distributions | 1.502 | 19.307 | 19.354 |
| rotated16x8 | main | training | 1303.809 | 20.221 | 20.297 |
| rotated16x8 | main | save_research_state_and_distributions | 3.021 | 19.219 | 19.244 |
| rotated16x8 | main | construct_hard_candidates | 596.268 | 19.434 | 19.492 |
| rotated16x8 | main | save_mixed_artifact | 135.856 | 19.214 | 19.238 |
| rotated16x8 | main | verify_export_and_install_reference | 88.080 | 19.218 | 19.240 |
| rotated16x8 | main | export-reference_decoded_cache_preparation | 26.057 | 19.168 | 19.266 |
| rotated16x8 | main | export-reference_full_nll_shape_memory_gate | 0.761 | 19.497 | 19.811 |
| rotated16x8 | main | verify_cached_reference_exact | 0.671 | 19.134 | 19.215 |
| rotated16x8 | fresh reload | load_mixed_without_decoded_cache_for_profiling | 69.893 | 6.168 | 6.254 |
| rotated16x8 | fresh reload | fresh-reload_decoded_cache_preparation | 26.691 | 19.099 | 19.221 |
| rotated16x8 | fresh reload | fresh-reload_full_nll_shape_memory_gate | 1.395 | 19.459 | 19.826 |
| rotated16x8 | fresh reload | exact_reference_parity | 0.667 | 19.096 | 19.207 |
| rotated16x8 | nested inside training | cache | 323.431 | 19.533 | 20.297 |
| rotated16x8 | nested inside training | hard_evaluation_before | 27.480 | 19.529 | 20.297 |
| rotated16x8 | nested inside training | audit_boundary_before | 0.454 | 19.564 | 20.297 |
| rotated16x8 | nested inside training | optimization | 926.978 | 20.221 | 20.297 |
| rotated16x8 | nested inside training | freeze_window | 0.007 | 19.383 | 20.297 |
| rotated16x8 | nested inside training | hard_evaluation_after | 24.421 | 19.605 | 20.297 |
| rotated16x8 | nested inside training | audit_boundary_after | 0.424 | 19.641 | 20.297 |
| rotated16x8 | nested inside training | cleanup | 0.510 | 19.297 | 20.297 |
| rotated16x8 | resident quality evaluation | hard_fp32 | 37.240 | 19.599 | 19.869 |
| rotated16x8 | resident quality evaluation | fp16_same_wrapper | 37.677 | 19.599 | 19.869 |
| rotated16x8 | resident quality evaluation | packed_fresh_reload | 21.618 | 19.459 | 19.826 |
| rotated128x1 | main | load_original_bf16 | 16.978 | 17.538 | 17.588 |
| rotated128x1 | main | baseline_probe_and_hidden_reference | 0.556 | 17.705 | 17.812 |
| rotated128x1 | main | prepare_original_export_plan | 24.979 | 17.569 | 17.619 |
| rotated128x1 | main | conversion_rank64 | 3.625 | 19.366 | 19.412 |
| rotated128x1 | main | initial_state_fingerprints | 1.428 | 19.183 | 19.207 |
| rotated128x1 | main | pretraining_distributions | 1.618 | 19.307 | 19.354 |
| rotated128x1 | main | training | 3965.794 | 20.219 | 20.297 |
| rotated128x1 | main | save_research_state_and_distributions | 3.081 | 19.219 | 19.244 |
| rotated128x1 | main | construct_hard_candidates | 576.950 | 19.434 | 19.492 |
| rotated128x1 | main | save_mixed_artifact | 133.677 | 19.214 | 19.238 |
| rotated128x1 | main | verify_export_and_install_reference | 88.328 | 19.218 | 19.240 |
| rotated128x1 | main | export-reference_decoded_cache_preparation | 25.659 | 19.168 | 19.266 |
| rotated128x1 | main | export-reference_full_nll_shape_memory_gate | 0.755 | 19.497 | 19.811 |
| rotated128x1 | main | verify_cached_reference_exact | 0.686 | 19.134 | 19.215 |
| rotated128x1 | fresh reload | load_mixed_without_decoded_cache_for_profiling | 68.238 | 6.168 | 6.273 |
| rotated128x1 | fresh reload | fresh-reload_decoded_cache_preparation | 26.588 | 19.100 | 19.240 |
| rotated128x1 | fresh reload | fresh-reload_full_nll_shape_memory_gate | 1.200 | 19.460 | 19.846 |
| rotated128x1 | fresh reload | exact_reference_parity | 0.663 | 19.096 | 19.227 |
| rotated128x1 | nested inside training | cache | 2604.822 | 19.533 | 20.297 |
| rotated128x1 | nested inside training | hard_evaluation_before | 223.278 | 19.528 | 20.297 |
| rotated128x1 | nested inside training | audit_boundary_before | 0.471 | 19.563 | 20.297 |
| rotated128x1 | nested inside training | optimization | 935.228 | 20.219 | 20.297 |
| rotated128x1 | nested inside training | freeze_window | 0.007 | 19.380 | 20.297 |
| rotated128x1 | nested inside training | hard_evaluation_after | 197.007 | 19.605 | 20.297 |
| rotated128x1 | nested inside training | audit_boundary_after | 0.441 | 19.641 | 20.297 |
| rotated128x1 | nested inside training | cleanup | 4.436 | 19.297 | 20.297 |
| rotated128x1 | resident quality evaluation | hard_fp32 | 34.019 | 19.599 | 19.869 |
| rotated128x1 | resident quality evaluation | fp16_same_wrapper | 34.256 | 19.599 | 19.869 |
| rotated128x1 | resident quality evaluation | packed_fresh_reload | 18.230 | 19.460 | 19.846 |

Main/reload process sums include setup, Python work and gaps. Nested trainer phases overlap training and must not be added again; resident evaluations are separately measured. Allocated/reserved peaks are absolute CUDA measurements, not incremental allocation or model-only tensor bytes. CPU audit/reporting work and human elapsed gaps are excluded from these process sums. The 90-minute preparation deadline is checked between steps/records; loading and individual CUDA calls are not exactly interruptible.

Attempt 1 stopped before loading weights or generating traces because the cached config omitted `_commit_hash`. Attempt 2 applies the reviewed metadata/provenance guard and necessary versioned output/gate/budget wiring. Contradictory non-null metadata is rejected; independently pinned path/hash/blob provenance is checked. The 25 core Python modules and training/evaluation math are byte-identical to the reviewed source. The prior failed process is charged exactly once; no endpoint prototype is integrated.

Base commit: `ff5bbf10e94d5d35282b9dfa48fa13069289fbfd`. Frozen runtime archive SHA256: `0a90d78f775d2c1e879c286646b4d545ec286231f97687e4bb99ea4f961f61ce`. Official model revision: `c202236235762e1c871ad0ccb60c8ee5ba337b9a`. Final source verification checks 84 frozen checkout files and 34 runtime Python hashes. The parent sealed manifest remains `5bd642801cf68c5250e5c56710929ac6c4da1a48e6e827da31607f509f2319d6`. [source review](<D:/openbonsai/runs/2026-10-26-5090-ultrachat-attempt2/independent-source-review.json>), [prior preservation evidence](<D:/openbonsai/runs/2026-10-26-5090-ultrachat-attempt2/independent-prior-verification.json>), [frozen source map](<D:/openbonsai/runs/2026-10-26-5090-ultrachat-attempt2/runtime-source-freeze.json>), [exact commands and reproduction](<D:/openbonsai/runs/2026-10-26-5090-ultrachat-attempt2/reproduction.md>).

The final historical-preservation audit independently rehashed 3,999 listed files / 44,381,745,395 bytes with no mismatches. Its scope is exactly the authenticated inventories, not unlisted files or unrelated branches. [Verification JSON](<D:/openbonsai/runs/2026-10-26-5090-ultrachat-attempt2/analysis/historical-preservation-final.json>) · [Preservation notes](<D:/openbonsai/runs/2026-10-26-5090-ultrachat-attempt2/analysis/historical-preservation-final.md>).

Earlier results are context only: [original diagnosis and repack](<D:/openbonsai/runs/2026-10-26-5090-rotations/analysis/prior-diagnosis-and-repack.md>); [prior rotation pair](<D:/openbonsai/runs/2026-10-26-5090-rotations/report.md>). Calibration, recipes or suite sizes differ; their results are not pooled into this matched comparison. No additional training, endpoint work, upload or push is triggered by this report.
