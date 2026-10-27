# Teacher-target repair v1 — stopped before fitting

The single authorized attempt stopped at a faulty runtime-comparison guard after loading the original BF16 model. **Zero of four discarded preflight updates and zero of 256 scientific updates ran.** Neither repair arm was fitted, evaluated or exported. There is no new quality result or evidence of recovery.

The failed attempt, frozen source, logs and inputs are preserved in this directory. No retry, successor experiment, larger calibration run, push or upload was launched. All earlier checkpoints and results remain in their original directories.

## What failed

`tools/repair_driver.py:81` compared a live Python dictionary from `backend_evidence(model)` directly with a JSON-decoded reference. Native Qwen configuration dictionaries retain integer keys `0` and `1` in the top-level, text and vision `id2label` maps. JSON decodes those keys as strings. That representation difference makes direct dictionary equality fail.

The current `measurement/runtime.json` and sealed v2 `measurement/loaded-runtime.json` are **byte-for-byte identical**: 13,626 bytes, SHA256 `d4b711dde86c0fbcf364f0f641929717fb51a82cf9b686a2973156d41fbc037c`. CPU reconstruction reproduces the integer/string-key mismatch and exact equality after the JSON serialization boundary. The live pre-serialization dictionary was not retained; the diagnosis combines the saved identical files, executed source line and native CPU configuration reproduction.

This is an implementation failure in the new guard, not observed evidence of changed kernels or model quality. The frozen driver remains unchanged. `analysis/proposed_runtime_guard.patch` is an **unapplied proposal** to compare the current and reference runtime records in the same serialized representation. Negative CPU fixtures retain rejection of changed attention, precision, library-source hashes and configuration values. No real-model claim is made for that proposal.

## Executed scope and measurements

| Item | Measured result |
|---|---:|
| Model processes launched | 1; exit code 1 |
| Aggregate model-process duration, monotonic clock | 24.906 s / 3,600 s allowance |
| BF16 loading phase | 15.2804 s |
| Peak GPU allocated during loading | 18,831,105,024 bytes / 17.538 GiB |
| Peak GPU reserved during loading | 18,884,853,760 bytes / 17.588 GiB |
| Minimum sampled/conservative free GPU memory during loading | 13,611,565,056 bytes / 12.677 GiB |
| Discarded GPU preflight updates | 0 / 4 |
| Conditional-target arm updates | 0 / 128 |
| Original-prefix-target arm updates | 0 / 128 |
| New NLL / ARC / generation results | Not measured |
| Selected-state or packed delta artifact | Not produced |

The loading memory gate passed. Timing and memory for backward, Adam, caching, evaluation and export were not measured. A complete-pair runtime estimate therefore could not be made; loading time alone cannot establish feasibility. The unused 3,575.094 seconds were not treated as permission to retry.

The stop occurred before original/bypass/hard logit probes, final128 state restoration, search-boundary reproduction and the first two fitting-target captures. Previously saved baseline and final128 quality results consequently were **not adopted as fresh comparison results for this attempt**.

## Intended fixed comparison

The preregistered comparison was one decoder window `[16,17)`, with both arms starting from the same authenticated final128 learned FP32 state. Both students would receive the same quantized-prefix input `xq`; only the target would differ: `Tq = original_window(xq)` versus `T0 = original_window(x0)`. Only that window's factor and LoRA parameters were eligible to change.

Each arm was limited to the same 128 frozen calibration records, one epoch, batch 1, sequence 512, group 128, rank 64, rotation block 4096, sigmoid factors, frozen-Jacobian hard stage, Huber delta 1, gradient clip 1, weight decay 0, factor LR 0.005 and LoRA LR 0.001. Order was `random.Random(5106).shuffle(range(128))`, with 102 soft and 26 hard updates. The fixed 16-record search set contained 7,203 active positions and was never eligible for fitting. The unused final-confirmation set was not opened.

The primary engineering criterion required the T0 arm to reduce pooled T0 MSE by at least 5% versus both the common initial state and Tq control, improve at least 12/16 records versus control, and avoid any record regression above 10% versus either comparator. These gates are **unevaluated**. The fixed 2,048-token / 20-question / 4-generation development evaluation would still be required; an activation-boundary gain alone would not establish quality recovery.

The prior sealed activation diagnostic remains the available evidence: window16 had pooled target-gap MSE 0.1968800, conditional reconstruction MSE 0.00140223 and original-target MSE 0.1965746. It motivates investigating the teacher target, but does not show that this proposed repair works. See `../2026-10-26-5090-activation-diagnostic-v2/report.md` for the complete prior diagnostic and its limitations.

## Identity, preparation and preservation

- Official source: `Qwen/Qwen3.5-9B`, revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`; original BF16, no task fine-tuning.
- Common final128 state SHA256: `4675c425ca82166a3f9275b5df358162f5b0a5ac8bb851d0fac8de962533274d`.
- Source-pair seal: `d7ae0ccf66b38f223366beb105beba32d651e5c2c1be166c21624eac7885bf8a`.
- Diagnostic v2 seal: `a4c5e7b2058f298d01cc09f042417b64e4d4a0f32b4dbaf61d5c54bb42f840cc`.
- Repair runtime source freeze: `4a1e437db88f5d952a7bd12dd0a8c5d5d4f052a9372d31074b9c2a63d22f55e9`, binding 156 files.
- Main repository HEAD: `891d260af96a7ce61ccbcb23474015d975bb18dc`; all 39 recorded local source files, Git status and diff were preserved. All 89 inherited checkout files remain byte-identical to v2.
- Environment: Python 3.12.11, PyTorch 2.14.1+cu130 / CUDA 13.0, Transformers 5.20.2, RTX 5090, driver 596.21; native eager attention, no KV or decoded-suffix cache, TF32 disabled.

CPU authentication checked all sealed v2/v1/source-pair files and 15 official cache files, including the exact designated checkpoint paths and byte sizes; that separate CPU authentication took 36.9122 seconds. The integrated CPU suite passed 858 tests with two skips in 90.50 seconds (91.7877 seconds process time). After the final evidence/finite-padding changes, 41 relevant tests passed in 8.26 seconds (9.5051 seconds process time). These fixtures do not substitute for the unexecuted real-model backward and parity gates.

Post-exit preservation and resource evidence is recorded in `analysis/repair-outcome-audit.json`; exact inventory and hashes are in `final-validation.json`. Preparation, CPU tests and post-exit audits are reported separately from model-process time.

The source-preparation interval was about 1,069 seconds of wall time, including implementation, review and CPU tests; this is not CPU utilization or GPU time. The recorded `analysis/initial-isolation.json.created_unix` to supervisor-start interval is 1,068.8424 seconds. The post-exit preservation audit took 23.4461 seconds. Seven CPU tests for the unapplied guard proposal passed in 5.11 seconds. No subsequent model process ran.

## Review outcome

Keep this attempt sealed. The concrete next proposal is the narrow JSON-comparison correction and the same scientific recipe, with explicit accounting for this failed attempt's 24.906 seconds. It has not been selected or launched. No evidence from this attempt supports increasing calibration, epochs, window coverage or the total allowance.

Key evidence: `failure-fit.json`, `repair-supervisor.json`, `logs/fit.log`, `analysis/runtime-guard-diagnosis.json`, `analysis/proposed_runtime_guard.patch`, `analysis/repair-source-review.json`, `analysis/repair-cpu-tests.json`, and `reproduction.md`.
