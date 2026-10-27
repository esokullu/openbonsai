The authorized full sequential fit, standalone export and fresh packed evaluation completed. This is a development diagnostic; numerical/serialization success does not establish useful model quality.

Status `completed_full_sequential_and_fresh_reload`. Supervisor-confirmed scientific updates 4096/4096; discarded preflight updates 4/4. Streamed completed training events 4096; audit `complete_verified`. Ledger intentions are reported separately.

V2 model-process time 6577.657s /14,400s; no prior debit. V1 consumed 192.985s and4 discarded/0 scientific updates. Informational two-attempt total 6770.642s, 8 discarded and 4096 scientific updates. This total is not charged twice.

Fresh official Qwen/Qwen3.5-9B BF16, no task fine-tuning:128 fixed records once per32 sequential windows,512 positions, batch1, rank64, group128, rotations, sigmoid factors, Huber1, factor/LoRA LR0.005/0.001. Each window has102 soft+26 hard updates and actual last-soft k29.8828125. The student prefix is current; targets come from an independent original-weight prefix. All59,855 active fitting positions are reused once per window. The earlier repair changed only window16 of a learned model; this recipe starts fresh throughout.

Numerical quality is the fixed development subset of2,048 shifted, nonpadding, token-weighted NLL tokens,20 ARC-Easy questions and4 generations. It uses unchanged FP32 CE with FP64 summation, eager/no-KV execution, greedy nonthinking generation, both EOS IDs248044/248046 and128-token limit. The original full pilot16,384/100/10 is a different population. No unused final-confirmation data is read.

| Representation | NLL | PPL | ARC | Empty / EOS-only / capped | Tokens | Eval seconds | Peak allocated / reserved GiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| Untouched BF16 (saved) | 2.423776811 | 11.288413 | 19/20 | 0 / 0 / 2 | 341 | 34.278 | 17.954 / 18.191 |
| Historical conditional128 hard FP32 | 9.121630685 | 9151.112039 | 8/20 | 4 / 4 / 0 | 4 | 34.019 | 19.599 / 19.869 |
| Historical conditional128 packed | 9.123807641 | 9171.055308 | 8/20 | 4 / 4 / 0 | 4 | 18.230 | 19.460 / 19.846 |
| New original-prefix hard FP32 | 11.827691902 | 136993.930601 | 8/20 | 0 / 0 / 4 | 512 | 637.354 | 19.599 / 19.869 |
| New original-prefix fresh packed | 11.849104019 | 139958.890526 | 8/20 | 0 / 0 / 4 | 512 | 357.107 | 19.455 / 19.846 |

Evaluation timings exclude loading/calibration/export. Token counts differ when outputs stop early; EOS-only speed is not useful throughput. Automatic empty/repetition/EOS counts do not replace reading all answers. Exact text and IDs are preserved in the generation draft.

| Comparison | ΔNLL | ΔPPL | ΔARC percentage points |
|---|---:|---:|---:|
| Historical conditional128 hard FP32 minus bf16 | +6.697853875 | +9139.823626 | -55.0 |
| Historical conditional128 packed minus bf16 | +6.700030831 | +9159.766895 | -55.0 |
| New original-prefix hard FP32 minus bf16 | +9.403915091 | +136982.642187 | -55.0 |
| New original-prefix hard FP32 minus matched_conditional | +2.706061216 | +127842.818561 | +0.0 |
| New original-prefix fresh packed minus bf16 | +9.425327209 | +139947.602112 | -55.0 |
| New original-prefix fresh packed minus matched_conditional | +2.725296378 | +130787.835217 | +0.0 |

Historical-score reuse passed strict source/runtime and fresh scientific original/bypass identity gates. Saved historical measurements remain distinct from new evaluations.
ARC predictions for Untouched BF16 (saved): `{"A": 7, "B": 7, "C": 4, "D": 2}`.
ARC predictions for Historical conditional128 hard FP32: `{"A": 20}`.
ARC predictions for Historical conditional128 packed: `{"A": 20}`.
ARC predictions for New original-prefix hard FP32: `{"A": 20}`.
ARC predictions for New original-prefix fresh packed: `{"A": 20}`.

| Complete-plan projected variable work | Seconds |
|---|---:|
| Fitting cache captures | 2852.827 |
| 4096 optimizer updates | 1897.784 |
| Before/after hard MSE diagnostics | 1255.665 |
| Per-window evidence | 9.810 |
| Two complete capped quality passes | 1346.677 |
| Fresh model loads/setups | 203.414 |
| Control/search/preservation | 269.002 |
| Complete artifact export/verification | 820.452 |

Projection including elapsed time,1.25× variable margin and600s reserve: 11609.088s; cap 14400s; fits=True.

Actual9B window0 historical control: passed exact selected final state, shared update evidence and same-input native boundary before window1.

Training audit: 4096 completed events; stages {'soft': 3264, 'hard': 832}; 32 completed window receipts; 0 observed audit violations. Full4096 finite gradient/update/snapshot/moment verification=True. Huber is the optimization objective; before/after hard MSE is a separate diagnostic. Family gradient/update ranges, first-step zeros, actual102/103 snapshots, all window losses and exposure checks are in the JSON.

| Window | Representation | Y/T0 MSE | Y/T0 relRMSE | Y/T0 cosine | Y/Tq relRMSE | Signed cross mean |
|---:|---|---:|---:|---:|---:|---:|
| 0 | conditional128 | 0.000311311 | 0.164182165 | 0.986454112 | 0.164182165 | +0.000000000 |
| 0 | original_prefix | 0.000311311 | 0.164182165 | 0.986454112 | 0.164182165 | +0.000000000 |

Window0: 0/16 improved and 0/16 worsened against the common T0 target. Every per-record change is retained.

| 16 | conditional128 | 0.196574567 | 0.760928664 | 0.670894394 | 0.076243948 | -0.001707702 |
| 16 | original_prefix | 0.104790041 | 0.555571919 | 0.831735228 | 0.174739080 | -0.009645913 |

Window16: 16/16 improved and 0/16 worsened against the common T0 target. Every per-record change is retained.

| 31 | conditional128 | 18.120259124 | 1.360481016 | 0.201413160 | 0.093858310 | -0.122788678 |
| 31 | original_prefix | 6.858338666 | 0.836988994 | 0.547380521 | 0.483423939 | -1.720103764 |

Window31: 16/16 improved and 0/16 worsened against the common T0 target. Every per-record change is retained.

FP64 activation diagnostics on16 search records. y/t0 and tq/t0 normalize byT0; y/tq normalizes byTq. Signed cross terms are descriptive, not causal shares. No new quality-support threshold is invented. These16 search records never fit parameters. Scalar rescaling is diagnostic only; no new support threshold is invented.

| Process / phase group | Seconds | Peak allocated / reserved GiB |
|---|---:|---:|
| preflight: load_pinned_tokenizer | 0.321 | 0.000 / 0.000 |
| preflight: verify_pinned_template_and_tokens | 0.609 | 0.000 / 0.000 |
| preflight: load_original_bf16 | 15.048 | 17.538 / 17.588 |
| preflight: hash_original_tensors | 19.761 | 17.538 / 17.588 |
| preflight: probe_original_native | 0.627 | 17.591 / 17.637 |
| preflight: prepare_original_export_plan | 25.105 | 17.569 / 17.637 |
| preflight: conversion_rank64 | 3.386 | 19.366 / 19.398 |
| preflight: initial_state_fingerprints | 1.382 | 19.183 / 19.398 |
| preflight: hash_original_after_conversion | 18.418 | 19.183 / 19.398 |
| preflight: probe_original_bypass | 0.269 | 19.205 / 19.412 |
| preflight: probe_fresh_hard_initial | 1.241 | 19.452 / 19.568 |
| preflight: preflight-learned-state-before | 1.408 | 19.183 / 19.568 |
| preflight: preflight-cache-window-0 | 0.045 | 19.303 / 19.568 |
| preflight: preflight-update-window-0-step-102 | 0.605 | 20.095 / 20.301 |
| preflight: preflight-update-window-0-step-103 | 0.277 | 20.217 / 20.301 |
| preflight: preflight-cache-window-3 | 0.310 | 19.520 / 20.301 |
| preflight: preflight-update-window-3-step-102 | 0.219 | 20.005 / 20.301 |
| preflight: preflight-update-window-3-step-103 | 0.309 | 20.121 / 20.301 |
| preflight: preflight-learned-state-after | 1.326 | 19.214 / 20.301 |
| preflight: hard-initial-probe-after-rollback | 1.345 | 19.483 / 20.301 |
| preflight: original-sources-after-rollback | 18.315 | 19.214 / 20.301 |
| preflight: anchor-cache-0-0 | 0.063 | 19.334 / 20.301 |
| preflight: anchor-hard-mse-0-0 | 0.187 | 19.516 / 20.301 |
| preflight: anchor-cache-0-1 | 0.026 | 19.334 / 20.301 |
| preflight: anchor-hard-mse-0-1 | 0.055 | 19.516 / 20.301 |
| preflight: anchor-cache-16-0 | 0.676 | 19.528 / 20.301 |
| preflight: anchor-cache-16-1 | 0.694 | 19.528 / 20.301 |
| preflight: anchor-cache-30-0 | 1.308 | 19.528 / 20.301 |
| preflight: anchor-cache-30-1 | 1.276 | 19.528 / 20.301 |
| preflight: anchor-cache-3-0 | 0.150 | 19.520 / 20.301 |
| preflight: anchor-hard-mse-3-0 | 0.051 | 19.524 / 20.301 |
| preflight: anchor-cache-3-1 | 0.153 | 19.520 / 20.301 |
| preflight: anchor-hard-mse-3-1 | 0.049 | 19.524 / 20.301 |
| preflight: anchor-cache-19-0 | 0.806 | 19.528 / 20.301 |
| preflight: anchor-cache-19-1 | 0.820 | 19.528 / 20.301 |
| preflight: anchor-cache-31-0 | 1.310 | 19.528 / 20.301 |
| preflight: anchor-cache-31-1 | 1.474 | 19.528 / 20.301 |
| preflight: resource-first-generation-prompt | 1.255 | 19.483 / 20.301 |
| preflight: resource-maximum-context | 1.253 | 19.528 / 20.301 |
| preflight: resource-maximum-context-nll | 1.363 | 20.046 / 21.135 |
| preflight: historical_state_file_binding | 0.949 | 19.214 / 21.135 |
| preflight: map_historical_state_cpu | 0.048 | 19.214 / 21.135 |
| preflight: validate_historical_sources_and_state | 14.379 | 19.214 / 21.135 |
| preflight: assign_historical_control_state | 0.343 | 19.214 / 21.135 |
| preflight: probe_historical_hard_restored | 1.201 | 19.483 / 21.135 |
| preflight: search-boundary | 39.739 | 19.528 / 21.135 |
| training: load_pinned_tokenizer | 0.333 | 0.000 / 0.000 |
| training: verify_pinned_template_and_tokens | 0.613 | 0.000 / 0.000 |
| training: load_original_bf16 | 14.767 | 17.538 / 17.588 |
| training: hash_original_tensors | 19.459 | 17.538 / 17.588 |
| training: probe_original_native | 0.649 | 17.591 / 17.637 |
| training: prepare_original_export_plan | 23.475 | 17.569 / 17.637 |
| training: conversion_rank64 | 3.809 | 19.366 / 19.398 |
| training: initial_state_fingerprints | 1.358 | 19.183 / 19.398 |
| training: hash_original_after_conversion | 18.704 | 19.183 / 19.398 |
| training: probe_original_bypass | 0.428 | 19.205 / 19.412 |
| training: probe_fresh_hard_initial | 1.338 | 19.452 / 19.568 |
| training: training:cache | 2775.068 | 19.533 / 20.285 |
| training: training:hard_evaluation_before | 215.834 | 19.528 / 20.285 |
| training: training:audit_boundary_before | 0.470 | 19.564 / 20.285 |
| training: training:optimization | 928.436 | 20.217 / 20.285 |
| training: training:freeze_window | 0.007 | 19.378 / 20.285 |
| training: training:hard_evaluation_after | 190.934 | 19.606 / 20.285 |
| training: training:audit_boundary_after | 0.453 | 19.641 / 20.285 |
| training: window0-control-parent-bindings | 0.997 | 19.294 / 20.285 |
| training: window0-control-map-parent-state | 0.049 | 19.294 / 20.285 |
| training: window0-control-source-model.language_model.layers.0.linear_attn.out_proj | 0.026 | 19.294 / 20.285 |
| training: window0-control-source-model.language_model.layers.0.linear_attn.in_proj_qkv | 0.044 | 19.294 / 20.285 |
| training: window0-control-source-model.language_model.layers.0.linear_attn.in_proj_z | 0.022 | 19.294 / 20.285 |
| training: window0-control-source-model.language_model.layers.0.mlp.gate_proj | 0.071 | 19.294 / 20.285 |
| training: window0-control-source-model.language_model.layers.0.mlp.up_proj | 0.072 | 19.294 / 20.285 |
| training: window0-control-source-model.language_model.layers.0.mlp.down_proj | 0.071 | 19.294 / 20.285 |
| training: window0-control-capture-native-input | 0.019 | 19.414 / 20.285 |
| training: window0-control-current-replay | 0.056 | 19.596 / 20.285 |
| training: window0-control-parent-replay | 0.052 | 19.596 / 20.285 |
| training: training:cleanup | 3.979 | 19.295 / 20.285 |
| training: save-final-fp32-research-state | 2.304 | 19.214 / 20.285 |
| training: original-sources-after-science | 19.599 | 19.214 / 20.285 |
| training: original-bypass-after-science | 0.175 | 19.236 / 20.285 |
| training: search-boundary | 40.746 | 19.528 / 20.285 |
| training: hard-fp32-nll | 5.117 | 19.599 / 19.869 |
| training: hard-fp32-benchmark | 26.825 | 19.495 / 19.602 |
| training: hard-fp32-generation | 605.252 | 19.496 / 19.604 |
| training: export-fp32-research-probe | 1.185 | 19.483 / 19.604 |
| training: export-fp16-scales-same-wrapper-probe | 1.239 | 19.483 / 19.604 |
| training: construct-candidate | 713.650 | 19.434 / 19.604 |
| training: save-full-mixed-artifact | 157.047 | 19.214 / 19.604 |
| training: verify-retained-artifact | 12.353 | 19.214 / 19.604 |
| training: verify-install-candidate | 62.337 | 19.218 / 19.604 |
| training: export-uncached-packed-probe | 21.112 | 6.287 / 19.604 |
| training: export-decoded-cache | 26.925 | 19.157 / 19.662 |
| training: export-full-nll-shape | 0.854 | 19.486 / 20.207 |
| training: export-cached-packed-probe | 0.730 | 19.122 / 20.207 |
| reload: cold-load-full-packed-artifact | 63.735 | 6.163 / 6.273 |
| reload: fresh-full-payload-verification | 11.746 | 6.163 / 6.273 |
| reload: fresh-decoded-cache | 27.392 | 19.060 / 19.240 |
| reload: fresh-full-nll-shape | 1.464 | 19.455 / 19.846 |
| reload: fresh-stored-scale-logit-parity | 0.729 | 19.091 / 19.850 |
| reload: packed-nll | 2.912 | 19.455 / 19.846 |
| reload: packed-benchmark | 13.607 | 19.128 / 19.248 |
| reload: packed-generation | 340.455 | 19.129 / 19.270 |

Disjoint phase durations are summed; memory peaks are maxima. Process wall time includes other administration. External CPU audits are separate.

| Stored scope (bytes) | Historical conditional128 | New original-prefix |
|---|---:|---:|
| Model artifact excluding tokenizer | 6,576,831,354 | 6577193368 |
| Tokenizer/configuration assets counted as tokenizer | 19,998,761 | 19998761 |
| Total deployable directory | 6,596,830,115 | 6597192129 |

Equivalent BF16-loaded persistent tensor denominator 18,819,627,488bytes; raw represented source 18,819,635,168bytes. Unused MTP tensors 486,581,248bytes are excluded. The48 retained FP32→BF16 casts differ by7,680bytes;4 generated nonpersistent buffers add400bytes.

Strict persistent payload numerator 6,576,607,414 / denominator 18,819,627,488: 2.861601×, 65.054529% reduction. Complete deployment bytes additionally include containers/configuration/tokenizer. Research states, optimizer state and nonpersistent decoded BF16 caches are excluded.

| Candidate tensor payload | Bytes |
|---|---:|
| presence | 864,026,624 |
| rotation_signs | 1,081,344 |
| row_sign_offsets | 11,404,864 |
| scales | 108,003,328 |
| signs | 596,889,750 |

| Retained floating-point component | Bytes |
|---|---:|
| Output head | 2,034,237,440 |
| Input embeddings | 2,034,237,440 |
| Text normalization | 546,816 |
| Linear-attention dynamics | 3,072 |
| Linear-attention convolution | 1,572,864 |
| Linear-attention gates | 12,582,912 |
| Text rotary tensors | 256 |
| Vision including its positional tensors | 912,021,104 |

| Fixed logit probe comparison | Bitwise exact | Max absolute difference | Mean absolute difference |
|---|---|---:|---:|
| fp32_to_fp16_scales_same_wrapper | False | 0.134765625 | 0.0203634035 |
| fp16_wrapper_to_packed_row64 | False | 0.25 | 0.0176641922 |
| decoded_cache_vs_uncached | True | 0 | 0 |
| fresh reload against stored-scale packed reference | True | 0 | 0 |

The FP16 same-wrapper row is a fixed probe only; no FP16 same-wrapper NLL/ARC/generation pass was run. FP32→FP16 scale rounding and FP16-wrapper→packed-row64 arithmetic differences are separate from exact same-representation fresh reload. File-size reduction is not inference-speed or VRAM reduction.

CPU gate: ================= 915 passed, 2 skipped in 116.91s (0:01:56) ==================. Source freeze 182files, SHA `717468370847ab1c3405589cf797f9d6fc26c450c57d0cce02290b980875025b`. Final execution and preservation audit receipts passed. The coordinator owns final report/seal creation. No new run, alternative recipe or final-confirmation evaluation is launched by this helper.
