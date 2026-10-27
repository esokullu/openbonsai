# Teacher-target repair v2 — local improvement, quality still failed

The single corrected attempt completed both training arms, their fixed development evaluations, selected-window exports and independent fresh-process verification. Changing the layer-16 teacher target reduced held-out activation error by **32.7%** and language-model NLL by **0.8425 nats** versus the conditional-target control. It did **not** recover useful model behavior: both arms choose A on every ARC question, and all four generation tasks remain degenerate. No successor experiment was launched.

## Fixed development comparison

These results use **2,048 scored WikiText tokens, 20 ARC-Easy questions and four prompts**. They are repeatedly inspected development diagnostics, not the original pilot's 16,384-token/100-question/10-prompt evaluation or an unused confirmation set. Original BF16 and the common initial state use authenticated earlier results, justified by fresh exact native/bypass/hard probes and unchanged evaluator/data/settings hashes. Both repair rows are newly measured hard-quantized research models with FP32 learned factors and LoRA parameters.

| State | NLL | Perplexity | ARC | Generated tokens | Useful complete answers | Evaluation allocated / reserved GiB | Evaluation seconds |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original BF16, reused | 2.423777 | 11.288 | 19/20 | 341 | 2/4 | 17.954 / 18.191 | 34.278 |
| Common final128 initial state, reused | 9.121631 | 9,151.112 | 8/20 | 4 | 0/4 | 19.599 / 19.869 | 34.019 |
| Conditional target Tq | 9.121692 | 9,151.670 | 8/20 | 4 | 0/4 | 19.599 / 19.869 | 33.538 |
| Original-prefix target T0 | 8.279187 | 3,940.989 | 8/20 | 8 | 0/4 | 19.599 / 19.869 | 38.683 |

T0 minus Tq: NLL **−0.842505**, perplexity **−5,210.681**, ARC **0 percentage points**. T0 remains **+5.855410 nats** and **−55 percentage points** from BF16. All compressed states predict A on all 20 questions; eight answer keys are A. This score does not demonstrate general QA competence.

The conditional arm emits `[248046]` (EOS only) for every prompt. The T0 arm emits `[2, 248046]`, decoding to `#` and EOS, for every prompt. Thus both have zero capped/unfinished outputs but zero useful answers. BF16 gives two useful completed answers and two relevant answers truncated at the 128-token limit. Different generated-token counts prevent interpreting these runtimes as equal-work inference speedups.

| Task | BF16 output summary | Conditional output | T0 output |
|---|---|---|---|
| Three shared-kitchen tips | Three numbered actionable tips | Empty | `#` |
| Chickens/rabbits reasoning | Relevant derivation, capped before completion | Empty | `#` |
| Balanced-parentheses Python | Relevant function draft, capped before completion | Empty | `#` |
| Formal Turkish rewrite | “Yarın katılım sağlayamayacağım; toplantıyı başka bir tarihe erteleyebilir miyiz?” | Empty | `#` |

Exact answers, token IDs, all option scores, stopping flags, per-sample losses and independent checks are in [paired-quality-boundary-analysis.md](analysis/paired-quality-boundary-analysis.md) and its JSON. Quality scoring retains shifted labels, nonpadding token weighting, FP32 cross-entropy and FP64 accumulation. Both arms use the same tokenizer/template, greedy nonthinking decoding, eager attention, no KV cache, a 128-new-token cap and EOS IDs 248044/248046.

The fixed source identities are `Salesforce/wikitext`, `wikitext-2-raw-v1`, test split, revision `b08601e04326c79dfdd32d625aee71d232d685c3`; and `allenai/ai2_arc`, ARC-Easy validation, revision `210d026faf9955653af8916fad021475a3f00453`, seed 5090. The four text blocks are IDs `wikitext-test-block-00` through `03`; generation IDs are `manual-heldout-00/02/04/07`. Exact ARC IDs and source rows are retained in the authenticated [diagnostic input file](../2026-10-26-5090-ultrachat-attempt2/inputs/diagnostic-inputs.json), SHA256 `330ffeeb77fa384492a6494e1e698edd400d7088456dbd0b7bc0fd366b7d838a`.

## What the controlled target change establishes

Both arms start from the same authenticated final128 learned state and receive the same quantized-prefix input `xq`, native arguments and masks. Only the target differs: `Tq = original_window(xq)` versus `T0 = original_window(x0)`. T0 comes from an independent native original-source pass with quantizers disabled. Only the six selected projections' factor and LoRA parameters in decoder window `[16,17)` are trainable. All original tensors, rotations, statistics and other layers remain fixed.

The comparison uses 16 fixed search records, never fitted, containing 7,203 active positions. Element-pooled statistics and every record's comparison are saved.

| Final state | MSE against T0 | Relative RMSE against T0 | Cosine with T0 | MSE against Tq |
|---|---:|---:|---:|---:|
| Common initial | 0.196574567 | 0.760929 | 0.670894 | 0.001402227 |
| Conditional target | 0.196725875 | 0.761221 | 0.670532 | 0.001184585 |
| Original-prefix target | 0.132229790 | 0.624087 | 0.781715 | 0.062466821 |

T0 MSE falls **32.7330% versus initial** and **32.7847% versus control**. All 16 records improve versus both; none regresses. The preregistered local criterion passes: at least 5% pooled improvement against both references, at least 12/16 improvements against control, and no record over 10% worse. The conditional control slightly worsens T0 MSE on all 16 records (+0.076972% pooled), despite improving its own Tq objective.

This supports the teacher-target mismatch hypothesis for this selected window and fixed cache. It does not establish that accumulated drift elsewhere is repaired, that the target change scales across all windows, or that quality is recovered. The signed cross-term decomposition is descriptive, not a standalone causal attribution.

Matched evaluation of the **same saved pre-update-102 parameters** separates hardening from optimization. Conditional-arm Tq MSE rises from 0.001221687 soft to 0.001334092 hard (+9.20%). T0-arm T0 MSE changes from 0.133738963 to 0.133728812 (−0.0076%). Comparing either saved-hard state with final-hard includes update 102 plus 26 hard-stage updates; it does not isolate the hard stage alone. No last-soft full-model quality evaluation was performed.

## Training validity and calibration limits

Each arm completed 128 updates: 102 soft and 26 hard, with batch 1, sequence 512, group 128, rank 64, rotation block 4096, sigmoid factors, frozen-Jacobian hard stage, Huber delta 1, gradient clip 1, weight decay 0, factor LR 0.005 and LoRA LR 0.001. Both use the identical seed-5106 shuffled order, fresh empty AdamW and common warm start. All 30 selected gradient tensors and 60 FP32 optimizer moments were finite; updates moved the intended parameter families. No clipping was triggered.

Four discarded real-model preflight updates passed, with exact state/runtime/RNG rollback and exact hard logits afterward. Independent CPU audit verified initial parameter equality, optimizer steps 0/101/102/128, the true pre-update-102 snapshot and sharpness 29.8828125. Separate per-arm start RNG hashes were not persisted; arm RNG reset is supported by the frozen snapshot/restore implementation and actual preflight reset evidence. Full details and coordinatewise Adam statistics are in [training-outcome-review.json](analysis/training-outcome-review.json).

No calibration was generated or changed in this attempt. The original 128 distinct UltraChat traces provide 59,855 consumed active positions. Their generation was greedy, nonthinking, SDPA with KV cache, capped at 512 new tokens: 58,089 generated tokens, 26 EOS completions and 102 capped traces. Right-truncation to the training sequence length removed 7,049 tokens from 105 records. The separate 16-record search set has four EOS completions, 12 capped traces and 715 tokens removed from 12 records. These truncation limits remain; this experiment does not test a different calibration recipe. The unused final-confirmation set was not opened.

Calibration and search prompts originate from `HuggingFaceH4/ultrachat_200k`, train_sft, revision `8049631c405ae6576f93f445c6b8166f76f5505a`. Frozen selection manifests retain the exact distinct IDs and generation seed 5090. The consumed fitting file SHA256 is `487a093fe330f35882ed04f969fa91e324b91630f5e17af003e02beebd23a699`; the consumed search SHA256 is `443aaba5e5b41c641c27cdcd45f34fd5a2e770fa466321c9bbab84436ff40aaf`. Neither input was retokenized for this repair.

## Serialization and storage scope

Both compact FP32 selected-state deltas and actual six-projection BITCOS/2-bit payloads passed independent fresh-process verification against the authenticated common base. Selected FP32 tensors, decoded ternary codes, FP16 scales and rotation signs match exactly; same-representation native logit probes match bitwise after reload. All retained/common-base tensors outside the selected parameters passed preservation checks.

The two delta directories occupy **93,260,793 bytes** (conditional) and **93,269,334 bytes** (T0), including selected-state data, packed payloads, probe references and manifests. Each requires the original BF16 checkpoint plus the authenticated final128 learned base; neither is a standalone model or a new whole-model deployment directory. No tokenizer assets or complete model copy were duplicated. These byte counts must not be compared against the entire original model as a compression ratio.

| Delta component, bytes | Conditional | T0 |
|---|---:|---:|
| BITCOS presence bits | 27,262,976 | 27,262,976 |
| Nonzero signs | 18,805,563 | 18,814,104 |
| FP16 scales | 3,407,872 | 3,407,872 |
| Rotation signs + row offsets | 393,264 | 393,264 |
| Packed container overhead | 6,004 | 6,004 |
| Six packed candidate files, subtotal | 49,875,679 | 49,884,220 |
| Compact FP32 selected-state file | 40,379,909 | 40,379,909 |
| Native probe reference file | 2,982,193 | 2,982,193 |
| Manifest | 23,012 | 23,012 |
| Complete delta directory | 93,260,793 | 93,269,334 |

FP32 learned factors, FP16-cast scales and the packed row-chunked computation are distinct representations. Their cross-representation logit differences are recorded separately from exact serialization parity. The new NLL/ARC table evaluates the FP32 research states; it is not a whole-packed-artifact quality benchmark. Existing full-model size, retained-component and BITCOS repack accounting remains in the preserved [original diagnosis](../2026-10-26-5090-diagnosis/report.md) and [source-pair report](../2026-10-26-5090-ultrachat-attempt2/report.md).

| Fixed-probe comparison, maximum / mean absolute logit difference | Conditional | T0 |
|---|---:|---:|
| FP32 learned factors → FP16 scales, same wrapper computation | 0.078125 / 0.0112052 | 0.078125 / 0.0119319 |
| FP16 wrapper → packed row-64 computation | 0.062500 / 0.0103653 | 0.078125 / 0.0130813 |
| Fresh reload versus saved reference, each of the three representations | 0 / 0 | 0 / 0 |

The nonzero first two rows are casting/computation changes; the zero third row establishes transport parity under identical settings. It does not establish BF16 quality preservation. Component arithmetic and hashes are in [delta-storage-parity-audit.json](analysis/delta-storage-parity-audit.json).

## Time, memory and implementation correction

The exclusive external supervisor measured **761.140 seconds** for this attempt: 526.140 seconds for fit/cache/evaluation/export, 117.141 seconds for conditional fresh verification and 117.859 seconds for T0 fresh verification. It debited the failed v1 process's **24.906 seconds once**, making **786.046 / 3,600 seconds** combined. All three children exited zero, with no concurrent model process or retry.

Measured components include 101.192 seconds for the remaining 126 paired fitting-cache records, 39.760/37.654 seconds for arm training, 33.538/38.683 seconds for quality evaluation, and 30.308/31.102 seconds for selected export. The preflight forecast was 3,597.684 seconds including the debit, all 1,024 possible generated tokens, both evaluations, two fresh verifications, 1.25 variable margin and 300 seconds reserve. Actual time was lower chiefly because generation terminated after degenerate one- or two-token outputs; the forecast did not assume that shortcut.

Across 424 recorded phases, peak allocated memory was **21,706,935,296 bytes (20.217 GiB)** and peak reserved memory **22,036,873,216 bytes (20.523 GiB)**. Minimum sampled/conservative headroom was **10,316,939,264 bytes (9.608 GiB)**; every observed memory gate passed. Storage savings and these reference-runtime memory/timing measurements are separate quantities.

The only model-driver correction from sealed repair v1 compares the current and reference backend records **after the existing strict JSON write/read**. Native configuration integer keys become strings in JSON; the failed v1 live-dictionary comparison rejected that representation change even though its two serialized runtime files were byte-identical. The correction preserves all fields and rejects changed kernels, precision, source hashes or nested config. It uses no lossy string coercion. Scientific helpers and all 89 inherited checkout files remain byte-identical to v1.

Administrative changes bind this new directory and prior debit. The first integrated CPU test run omitted the dedicated HF cache setting; its sole local-only configuration lookup failure and 909 passes/two skips are preserved. After correcting only the test launcher's `HF_HOME`, the final suite passed **910 tests, two skips**, in 91.12 seconds (92.436 seconds process time). Full CPU input authentication took 39.743 seconds; the post-exit preservation audit took 27.224 seconds. These are separate from model-process time.

## Identity, evidence and stopping decision

- Official BF16 source: `Qwen/Qwen3.5-9B`, revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`; no task fine-tuning.
- Common learned-state SHA256: `4675c425ca82166a3f9275b5df358162f5b0a5ac8bb851d0fac8de962533274d`.
- Runtime source freeze: `1641a407055610d63b42142a479f9fe7f43ec7349472fc8cc4392926ce48753e`, 163 files. Prior repair-v1 seal: `96bf138b9924d6f447a8133feaacc40024c36959a66d705d00e18585a718600c`.
- Main HEAD: `891d260af96a7ce61ccbcb23474015d975bb18dc`; all 39 recorded local source files, Git status and diff unchanged. All 2,804 files in the four authenticated ancestor inventories were rechecked after exit.
- Environment: Python 3.12.11; PyTorch 2.14.1+cu130 / CUDA 13.0; Transformers 5.20.2; RTX 5090; driver 596.21. Native kernels, precision, TF32-off settings and fallback implementations are unchanged.

Commands are recorded in [reproduction.md](reproduction.md); exact configuration in [experiment-plan.json](experiment-plan.json). Raw measurements, optimizer checkpoints, generated answers, per-record comparisons, payload manifests and fresh verifications remain in this directory. [repair-outcome-audit.json](analysis/repair-outcome-audit.json) verifies execution and preservation; the final inventory seal binds the completed report and evidence.

The local criterion passed, but the required nondegenerate-output condition did not. **Stop here.** No larger calibration recipe, wider-window intervention, grid, full 2,048-sample/60-epoch run, push or upload was launched. Any further experiment needs a separately reviewed scope; these results are preliminary evidence about a local target mismatch, not evidence of lossless compression or permission to increase training budget.
