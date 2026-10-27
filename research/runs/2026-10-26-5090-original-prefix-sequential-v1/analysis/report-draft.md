The experiment stopped before completing the authorized full calibration/export/evaluation sequence. This is a small development diagnostic; serialization and local reconstruction improvements do not establish recovered model quality.

Supervisor status: `stopped_before_fit_preflight_budget`. Confirmed scientific updates: 0/4096; confirmed discarded preflight updates: 4/4. Interrupted optimizer intents require the separate ledger audit. No retry or successor was authorized.

Aggregate model-process wall time: 192.985 /10,800s. Prior task debit is zero for this newly authorized experiment.

The recipe starts from official Qwen/Qwen3.5-9B BF16 revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`, without task fine-tuning. It uses128 fixed UltraChat records once in each of32 sequential windows (512 positions, batch1), rank64 rotations/LoRA, group128, sigmoid factors, Huber1, factor/LoRA LR0.005/0.001, and102 soft+26 hard updates per window. Actual last-soft sharpness is29.8828125. The student receives its current compressed prefix; the target comes from an independent original-weight prefix. The previous repair-v2 changed only window16 of the old learned model; this experiment starts fresh and applies the target change sequentially throughout the decoder.

The original128 traces are reused unchanged:59,855 active positions/window;26 traces ended in EOS and102 reached the generation cap;105 were truncated at512 consumed positions, removing7,049 tokens. Search16 is separate,7,203 active positions, and never enters fitting. No unused final-confirmation examples are accessed.

The prescribed numerical quality uses the same2,048 scored WikiText tokens,20 ARC-Easy questions and4 fixed generation prompts, with unchanged FP32 cross-entropy, token-count aggregation, eager attention, greedy nonthinking generation, no KV cache, both EOS IDs248044/248046 and128-token cap. This differs from the original full pilot16,384/100/10 population. Historical rows are saved measurements; fresh source/runtime/logit identity gates control their scientific reuse.

| Representation | NLL | PPL | ARC | Empty / EOS-only / capped | Generated tokens | Eval seconds | Eval peak allocated / reserved GiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| Untouched BF16 | 2.423777 | 11.288 | 19/20 | 0 / 0 / 2 | 341 | 34.278 | 17.954 / 18.191 |
| Historical conditional128 hard FP32 | 9.121631 | 9,151.112 | 8/20 | 4 / 4 / 0 | 4 | 34.019 | 19.599 / 19.869 |
| Historical conditional128 packed | 9.123808 | 9,171.055 | 8/20 | 4 / 4 / 0 | 4 | 18.230 | 19.460 / 19.846 |
| New original-prefix hard FP32 | unmeasured | unmeasured | unmeasured | unmeasured | unmeasured | unmeasured | unmeasured |
| New original-prefix fresh packed | unmeasured | unmeasured | unmeasured | unmeasured | unmeasured | unmeasured | unmeasured |

Evaluation runtimes exclude loading, training and export. Short EOS-only outputs can be faster while being useless; the table implies no speedup. Empty/EOS/cap statistics are automatic and do not count useful answers. All exact texts/token IDs are preserved in the generation draft and JSON.

| Comparison | ΔNLL | ΔPPL | ΔARC percentage points |
|---|---:|---:|---:|
| Historical conditional128 hard FP32 − BF16 | +6.697854 | +9139.824 | -55.0 |
| Historical conditional128 packed − BF16 | +6.700031 | +9159.767 | -55.0 |

The preflight projected 11,970.646s including its elapsed time,1.25× variable cost and600s reserve; fits_complete_plan=False. It reserves both full capped generation passes rather than assuming early EOS.

| Complete-plan projected variable work | Seconds |
|---|---:|
| Native fitting cache captures | 2847.261 |
| 4096 optimizer updates | 1881.485 |
| Before/after hard reconstruction diagnostics | 1455.967 |
| Per-window audit evidence | 11.375 |
| Two complete capped quality evaluations | 1452.266 |
| Fresh original and packed load/setup | 205.557 |
| Control, search-boundary and preservation work | 269.211 |
| Complete artifact construction/save/reference verification | 820.452 |
| Total before1.25× margin and600s reserve | 8943.574 |

The complete plan exceeded the10,800s cap by 1170.646s in this forecast. Stopping is a resource-gate result, not a measured quality result.

An **unselected, unauthorized proposal** would retain128 records and its exact historical control for window0, then use64 records for each remaining window once: 2112 scientific updates. Linear scaling gives 8188.446s under the current forecast assumptions. Half cache, optimizer and before/after reconstruction costs for windows1..31 only; keep window0, audit/control, load, full quality, export, preservation, margin and reserve unchanged. Linear estimate from this preflight, not measured or validated for a new recipe. Requires separate approval, new source/recipe freeze and resource admission. No64-record subset is selected here.

Preflight original/native/bypass and restored historical controls were recorded, but scientific reuse status remains `preflight_verified_pending_fresh_scientific_original` with eligible=False. A fresh scientific process never completed the required identity checks. All baseline/conditional128 quality values above are archived context only; none is a new quality measurement or evidence that quality stayed unchanged.

The four discarded native preflight updates reported hard-initial probe rollback=True, original/retained preservation=True, and learned/runtime rollback=True. These are distinct from the actual128-update scientific window0 historical control.

The mandatory actual9B window0 control was not completed; no pass is inferred from CPU tests.

The complete matched0/16/31 search-boundary comparison is unavailable.

| Process / measured phase group | Seconds | Peak allocated / reserved GiB |
|---|---:|---:|
| preflight: load_pinned_tokenizer | 0.326 | 0.000 / 0.000 |
| preflight: verify_pinned_template_and_tokens | 0.606 | 0.000 / 0.000 |
| preflight: load_original_bf16 | 14.946 | 17.538 / 17.588 |
| preflight: hash_original_tensors | 19.629 | 17.538 / 17.588 |
| preflight: probe_original_native | 0.540 | 17.591 / 17.637 |
| preflight: prepare_original_export_plan | 26.661 | 17.569 / 17.637 |
| preflight: conversion_rank64 | 4.221 | 19.366 / 19.398 |
| preflight: initial_state_fingerprints | 1.368 | 19.183 / 19.398 |
| preflight: hash_original_after_conversion | 18.482 | 19.183 / 19.398 |
| preflight: probe_original_bypass | 0.211 | 19.205 / 19.412 |
| preflight: probe_fresh_hard_initial | 1.330 | 19.452 / 19.568 |
| preflight: preflight-learned-state-before | 1.344 | 19.183 / 19.568 |
| preflight: preflight-cache-window-0 | 0.044 | 19.303 / 19.568 |
| preflight: preflight-update-window-0-step-102 | 0.602 | 20.095 / 20.301 |
| preflight: preflight-update-window-0-step-103 | 0.295 | 20.217 / 20.301 |
| preflight: preflight-cache-window-3 | 0.291 | 19.520 / 20.301 |
| preflight: preflight-update-window-3-step-102 | 0.202 | 20.005 / 20.301 |
| preflight: preflight-update-window-3-step-103 | 0.282 | 20.121 / 20.301 |
| preflight: preflight-learned-state-after | 1.347 | 19.214 / 20.301 |
| preflight: hard-initial-probe-after-rollback | 1.248 | 19.483 / 20.301 |
| preflight: original-sources-after-rollback | 18.302 | 19.214 / 20.301 |
| preflight: anchor-cache-0-0 | 0.063 | 19.334 / 20.301 |
| preflight: anchor-hard-mse-0-0 | 0.219 | 19.516 / 20.301 |
| preflight: anchor-cache-0-1 | 0.023 | 19.334 / 20.301 |
| preflight: anchor-hard-mse-0-1 | 0.052 | 19.516 / 20.301 |
| preflight: anchor-cache-16-0 | 0.685 | 19.528 / 20.301 |
| preflight: anchor-cache-16-1 | 0.684 | 19.528 / 20.301 |
| preflight: anchor-cache-30-0 | 1.332 | 19.528 / 20.301 |
| preflight: anchor-cache-30-1 | 1.275 | 19.528 / 20.301 |
| preflight: anchor-cache-3-0 | 0.150 | 19.520 / 20.301 |
| preflight: anchor-hard-mse-3-0 | 0.052 | 19.524 / 20.301 |
| preflight: anchor-cache-3-1 | 0.172 | 19.520 / 20.301 |
| preflight: anchor-hard-mse-3-1 | 0.055 | 19.524 / 20.301 |
| preflight: anchor-cache-19-0 | 0.863 | 19.528 / 20.301 |
| preflight: anchor-cache-19-1 | 0.823 | 19.528 / 20.301 |
| preflight: anchor-cache-31-0 | 1.332 | 19.528 / 20.301 |
| preflight: anchor-cache-31-1 | 1.320 | 19.528 / 20.301 |
| preflight: resource-first-generation-prompt | 1.259 | 19.483 / 20.301 |
| preflight: resource-maximum-context | 1.355 | 19.528 / 20.301 |
| preflight: resource-maximum-context-nll | 1.324 | 20.046 / 21.135 |
| preflight: historical_state_file_binding | 0.940 | 19.214 / 21.135 |
| preflight: map_historical_state_cpu | 0.055 | 19.214 / 21.135 |
| preflight: validate_historical_sources_and_state | 14.339 | 19.214 / 21.135 |
| preflight: assign_historical_control_state | 0.354 | 19.214 / 21.135 |
| preflight: probe_historical_hard_restored | 1.227 | 19.483 / 21.135 |
| preflight: search-boundary | 39.998 | 19.528 / 21.135 |

Phase times are disjoint recorded operations; their sum differs from process wall time. GPU peaks are maxima, not sums. The external post-exit CPU preservation audit is separate.

| Stored scope | Historical conditional128 | New original-prefix |
|---|---:|---:|
| Complete model artifact excluding tokenizer | 6,576,831,354 | unmeasured |
| Tokenizer assets | 19,998,761 | unmeasured |
| Total deployable directory | 6,596,830,115 | unmeasured |

Equivalent original BF16-loaded persistent tensors: 18,819,627,488bytes. Raw represented source tensors: 18,819,635,168bytes. Unused MTP checkpoint tensors: 486,581,248bytes, excluded from the equivalent model denominator. The48 retained FP32→BF16 casts change7,680bytes;4 generated nonpersistent buffers add400bytes. Duplicated caches and research state are excluded from deployable size.

No completed full deployable artifact was produced; temporary pre-conversion retained files do not count as an exported model.

The prior single-window repair improved local T0 reconstruction but kept ARC8/20 with all-A predictions and four degenerate “#” answers. That result motivated a different sequential target experiment; it did not establish quality recovery. Historical conditional128 emits four EOS-only empty answers. Current generated outputs must be read directly before interpreting any NLL improvement.

CPU validation: ================= 896 passed, 2 skipped in 116.22s (0:01:56) ==================. Receipt process time 117.562s. Frozen source inventory: 182files, SHA256 `1f16a6ed34b3450db6808551604c558784acfadc4c6051a3ac1d0946c7caa457`. Trainer SHA256 `a2971ea7a8ac1046aba113988c6ead72251f079d67ac2f2623044b3ab926192c`; driver SHA256 `39cb484751f6e698ec3b5abea00d45e38d8625349e533f6cacdcae95b96bcae9`.

Exact trainer diff: [trainer-diff.patch](D:/openbonsai/runs/2026-10-26-5090-original-prefix-sequential-v1/analysis/trainer-diff.patch). Source review: [sequential-source-review.json](D:/openbonsai/runs/2026-10-26-5090-original-prefix-sequential-v1/analysis/sequential-source-review.json). [Frozen runtime inventory](D:/openbonsai/runs/2026-10-26-5090-original-prefix-sequential-v1/runtime-source-freeze.json) and [experiment plan](D:/openbonsai/runs/2026-10-26-5090-original-prefix-sequential-v1/experiment-plan.json) bind the executed recipe.

The independent [execution audit](D:/openbonsai/runs/2026-10-26-5090-original-prefix-sequential-v1/analysis/final-execution-audit.json) and [preservation audit](D:/openbonsai/runs/2026-10-26-5090-original-prefix-sequential-v1/analysis/final-preservation.json) passed. The latter authenticates all five prior sealed runs, including the previous repair analysis cited here. The final report seal remains the coordinator's separate step. No successor, larger sample count, alternate method or unseen confirmation evaluation is launched by this reporter.

Observed preflight peaks: 21,708,011,520 allocated bytes and 22,693,281,792 reserved bytes; minimum measured/conservative free headroom 9,660,530,688 bytes across 93 phases. All four discarded updates had finite losses, valid gradients/FP32 moments and nonzero aggregate factor/LoRA movement. These checks do not substitute for the unexecuted full scientific run.
