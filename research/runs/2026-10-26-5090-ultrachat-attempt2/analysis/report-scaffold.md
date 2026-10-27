This is a provisional reporting scaffold for UltraChat attempt 2. Preparation is active; no final training, quality, artifact or cumulative-runtime result is asserted here. Replace pending cells only from completed saved evidence.

The controlled comparison is **rotated 16 samples × 8 epochs versus rotated 128 samples × 1 epoch**, with fresh original-model loads, identical initial factors/signs, and 128 updates per window. It compares calibration diversity and repetition under the frozen recipe; it does not isolate a rotation effect. The nested 16 are selected from the frozen 128 in parent order, with active-token exposure matched within 5%, rather than assumed exactly equal.

Attempt 1 stopped before weights or trace generation because cached configuration metadata omitted `_commit_hash`. The reviewed attempt 2 change validates the pinned cache's independent provenance and permits absent metadata while rejecting contradictory metadata. Quantization/training math, source tensors, generation settings, prompts and diagnostics remain unchanged. The prior failed process's **5.924203 seconds** is charged once to the 5,400-second preparation budget, leaving **5,394.075797 seconds** at this attempt's start. The recorded cumulative preparation timer already includes that debit.

Frozen runtime archive: `0a90d78f775d2c1e879c286646b4d545ec286231f97687e4bb99ea4f961f61ce`. Parent seal and independent review evidence are in [attempt-lineage.json](../attempt-lineage.json), [independent-prior-verification.json](../independent-prior-verification.json), and [independent-source-review.json](../independent-source-review.json). Integrated CPU validation passed 672 tests with two specific Windows symlink-privilege skips; this is software evidence, not a quality result.

| Evidence required | Rotated 16 × 8 | Rotated 128 × 1 |
|---|---|---|
| Calibration samples, epochs, active-token exposure and hashes | Pending | Pending |
| Resident 32-layer resource preflight: time and memory gate | Pending | Pending |
| Identical 200-module initial weights/factors/signs and RNG fingerprints | Pending paired audit | Pending paired audit |
| 4,096 updates: finite gradients, moments and frozen snapshots | Pending | Pending |
| Exact original/disabled-wrapper and fixed baseline probes | Pending | Pending |
| Exact packed payload, cached/uncached and fresh-reload probes | Pending | Pending |
| Hard FP32 NLL / ARC / generation text | Pending | Pending |
| FP16-exported-scale NLL / ARC / generation text | Pending | Pending |
| Independently reloaded packed NLL / ARC / generation text | Pending | Pending |

Quality uses the unchanged development subset: 2,048 scored NLL tokens, 20 ARC questions and four prompts with greedy generation capped at 128 new tokens. The historical BF16 baseline is NLL 2.4237768106 and ARC 19/20; its use requires this attempt's fixed baseline-probe parity. Preserve exact generated text, EOS/cap counts, and descriptive repetition measurements. Do not treat this repeatedly inspected subset as a held-out benchmark or compare its fractions directly with historical larger-suite results.

| Deployment bytes | Rotated 16 × 8 | Rotated 128 × 1 |
|---|---:|---:|
| Candidate presence/codes and nonzero-code signs | Pending | Pending |
| Candidate scales, bias, explicit rotation signs and row indexes (each separate) | Pending | Pending |
| Candidate/container/manifest/config overhead (separate from tensors) | Pending | Pending |
| Retained token embeddings | Pending verification | Pending verification |
| Retained output head | Pending verification | Pending verification |
| Retained vision, recurrent gates, convolutions, norms and dynamics (each separate) | Pending verification | Pending verification |
| Model files excluding tokenizer | Pending | Pending |
| Supplementary tokenizer assets | Pending | Pending |
| Complete deployable directory | Pending | Pending |
| Decoded BF16 runtime cache, excluded from disk total | Pending verification | Pending verification |
| FP32 research file, excluded from deployable directory | Pending | Pending |

The frozen equivalent scope contains 18,819,635,168 raw source tensor bytes or 18,819,627,488 BF16-loaded persistent bytes. Forty-eight retained FP32→BF16 casts account for 7,680 bytes. Unused MTP tensors (486,581,248 bytes) are excluded from both strict denominators. Generated nonpersistent rotary buffers (400 bytes) are excluded from strict compressed tensor numerators. Include all candidate signs, scales and row indexes. Report model-only versus supplementary tokenizer versus full deployable bytes separately; tokenizer assets are already in the full directory total.

There is **no separate endpoint prototype or endpoint quantization** in this run. Additional endpoint-prototype parameter bytes and quantized endpoint bytes are zero. Retained embedding and output-head weights are still present and must remain included: expected 2,034,237,440 bytes each, pending final artifact verification. Retained vision is also included; subtracting it for illustration would not create a validated standalone text-only artifact. The expected decoded cache is 13,824,425,984 BF16 bytes, runtime-only and nonpersistent; report measured cold-load/cache-preparation/steady phases separately.

The initial [runtime estimate](../runtime-estimate-before.json) is 3.4–4.0 hours and remains an estimate. Final measured reporting must separate shared generation (once), both resource preflights, each arm's main/reload process, and the one prior 5.924203-second failure. Never add nested trainer/evaluation phases to their enclosing process duration. The 90-minute preparation deadline is checked between steps/records; loading and individual CUDA calls are not exactly interruptible.

After completion and authorization, run the inherited training, quality, runtime and full payload-hash auditors, then the separate [attempt 2 accounting add-on](../summarize_attempt2_accounting.py). It preserves the copied tools and writes `analysis/attempt2-accounting.json` plus Markdown. The add-on requires completed process records and full streamed artifact verification; it cannot establish quality by itself. Activation distributions remain per arm because their first calibration records may differ; initial weight/sign fingerprints and the fixed NLL hidden probe have comparable input scopes.
