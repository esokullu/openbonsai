# Separate prefix drift from conditional reconstruction

The current decoder calibration normally fits each window using inputs from
the already-compressed prefix. Its local error can improve while the assembled
model becomes worse. The [original-prefix target option](teacher-target-control.md)
offers a controlled alternative. This diagnostic measures the difference on a
**single frozen model state**; the completed 9B measurement below supports a
bounded target comparison, without establishing a quality improvement.

This is read-only analysis. It does not select a target mode, optimize weights,
rescale activations or start another GPU experiment.

## Five quantities on the same tokens

For a decoder window with original weights `W` and hard ternary weights `Q`:

| Quantity | Meaning |
|---|---|
| `xq` | Window input from the hard-compressed prefix |
| `x0` | Window input from the original BF16 prefix |
| `tq = W(xq)` | Current conditional teacher target |
| `t0 = W(x0)` | Original-model target |
| `y = Q(xq)` | Current hard student output |

The two teacher paths bypass learned low-rank corrections and rotations exactly
where the original weights are used. The student replay uses its own captured
native masks and position inputs. No transforms are moved through gates,
nonlinearities or recurrent state. Passes stop at the selected window, before
the vocabulary head, and begin with empty KV/recurrent state.

The initial diagnostic is deliberately scoped to native dense Qwen3.5 with
original embeddings and recoverable original decoder weights. Packed-only
checkpoints cannot provide an original teacher. A frozen compressed checkpoint
must be restored as research wrappers around its exact original BF16 revision,
using its saved factors, corrections and rotation signs. Starting a new
initializer would answer a different question.

Match the completed run's loader, attention backend, dtype and exact rotation
signs, and validate saved hard-FP32 probe parity before collecting new records.
Do not silently substitute eager for SDPA, or packed FP16 scales for the saved
FP32 calibration state. Final compact factors do not contain the original
weights, optimizer moments or the actual last-soft pre-update snapshot.

## Measurements and interpretation

`openbonsai.activation_metrics.ActivationMetrics` computes per-record and pooled
metrics for `xq/x0`, `tq/t0`, `y/tq` and `y/t0`. Padding is excluded before
arithmetic. Reductions use bounded CPU FP64 chunks and are weighted by valid
hidden-state elements, not an unweighted average of sample errors.

For each pair, RMS ratio separates amplitude from direction; cosine and
relative RMSE describe the remaining distortion. A best nonnegative scalar
rescaling is reported only as a diagnostic. It is never applied to the model.
Undefined zero-norm ratios are explicit rather than hidden behind an arbitrary
epsilon.

The target decomposition includes its cross term:

```text
||y-t0||² = ||y-tq||² + ||tq-t0||² + 2 <y-tq, tq-t0>
```

The two errors can reinforce or cancel each other. Target-gap MSE is therefore
not a causal percentage of the final error. Likewise, low conditional MSE or a
better activation RMS alone does not establish better language quality.

`openbonsai.activation_capture.iter_activation_records` yields one detached CPU
record at a time and restores incoming model state before yielding. It does not
retain a corpus of hidden states or construct another 9B model. Callers must
consume and discard each record or explicitly provision disk storage. Source,
tokenizer, data-split and checkpoint provenance remain the caller's responsibility.
The measurement explicitly enables every supported decoder quantizer for the
hard student, even if the caller arrived with mixed enable flags; it restores
those incoming flags afterward.

The collector rejects packed/mixed modules, compressed embeddings, soft-phase
quantizers, rotation-only diagnostics, non-default RoPE and media/cache inputs.
It supports text tokens with explicit optional position IDs. These are scope
guards, not a general arbitrary-model instrumentation API. It requires finite
captured hidden states; the independent statistics API excludes padding before
its own finite-value checks and arithmetic.

## Completed RTX 5090 measurement: 26 October 2026

The corrected v2 run completed on `Qwen/Qwen3.5-9B`, original revision
`c202236235762e1c871ad0ccb60c8ee5ba337b9a`. It restored the completed diversity
pair's final `rotated128x1/learned-fp32-state.pt`, SHA256
`4675c425ca82166a3f9275b5df358162f5b0a5ac8bb851d0fac8de962533274d`.
This was the learned **hard FP32 research state**, not packed FP16 scales or a
new initialization. Collector source reference: `419d20e`.

Measurement used BF16 compute, the frozen native eager/no-cache backend, empty
recurrent state, batch 1, 512 positions and hidden width 4096. Group size 128,
rank 64, sigmoid factors and saved signed-Hadamard blocks of 4096 were unchanged.
Embeddings, head, vision and small dynamics tensors retained their verified
original precision. The scope was text-only; no vision quality was measured.

The 16 disjoint search records came from `HuggingFaceH4/ultrachat_200k`,
`train_sft`, revision `8049631c405ae6576f93f445c6b8166f76f5505a`, using original
BF16 nonthinking greedy continuations. V2 reused v1's frozen IDs, traces and
masks exactly, with no regeneration, replacement or error-based selection.
The 128 fitting traces were separate; the final confirmation set stayed unopened.
These 16 records now guide development and are **not final held-out evidence**.
Right truncation removed 715 tokens in 12 records; padding was excluded. Each
fixed window contains 7,203 valid positions, or **29,503,488 scalar elements**.
CPU FP64 statistics pool by those element counts, excluding oracle/repeat calls.

| Fixed window | Input drift / RMS(x0) | Target gap / RMS(t0) | Local error / RMS(t0) | Total error / RMS(t0) |
|---|---:|---:|---:|---:|
| `[0,1)` | 0 | 0 | 0.164182 | 0.164182 |
| `[16,17)` | 0.711287 | 0.761520 | 0.064267 | 0.760929 |

Here local error is `y-tq`, target gap is `tq-t0`, and total error is `y-t0`.
The last three columns share **RMS(t0)**. The separate `y/tq` relative RMSE is
0.076244 at window 16 because that pair instead divides by RMS(tq).

| Fixed window | Total MSE | Conditional MSE | Target-gap MSE | Signed cross-term mean |
|---|---:|---:|---:|---:|
| `[0,1)` | 0.000311310596 | 0.000311310596 | 0 | 0 |
| `[16,17)` | 0.196574566897 | 0.001402226641 | 0.196880041899 | -0.001707701643 |

The maximum relative decomposition residual across records/pools is approximately
`1.7140e-16`. Window 0 has exactly equal inputs and targets. At window
16, all **16/16** records pass the predeclared engineering selector: gap RMS
at least half total-error RMS, above repeat noise, requiring 12/16 positives.
There were no indeterminate denominators. Pooled gap/total RMS is 1.000776694;
the negative cross term explains why it can exceed one, without causal attribution.
The best scalar reduces `y/t0` relative RMSE only to 0.741553; none was applied.

Restoration checked 200 wrappers, 1,000 learned FP32 tensors and saved signs.
All five original/native-bypass/hard saved-probe comparisons were bitwise exact,
as were the first-two-record native oracle and A–B–A/reverse-order checks.
Source and learned tensors,
hooks, modes and flags were preserved. The v1 raw-CRLF/runtime-LF template guard
failure was corrected with separate raw/canonical identity checks, without
changing inputs. Its 23.503-second failed process remains charged: v2 took
170.637 seconds, **194.140 seconds cumulative**, below the 2,700-second cap.
Peak CUDA allocated/reserved memory was **19.497/19.600 GiB**, with minimum
measured/conservative headroom of 10.542 GiB. These are diagnostic costs, not
packed-inference speed or memory results. No training or quality evaluation ran.

The independently verified final archive inventory contains **226 files /
15,937,162 bytes**; `final-validation.json` SHA256 is
`a4c5e7b2058f298d01cc09f042417b64e4d4a0f32b4dbaf61d5c54bb42f840cc`.
Those bytes describe an experiment archive, not model size. The actual 9B
driver, restored-runtime adapter, data and receipts remain sealed remote
dependencies. The generic collector and fixture below do **not** reproduce
this run without them. Quality recovery and Bonsai equivalence remain unproven.

## Selected next control: results pending

The 5090 chat acknowledged the recipe and is preparing an isolated control;
no fit or quality result is claimed here. It specifies one warm-start pair at
`[16,17)`: 128 additional
updates per arm from the same final FP32 window state, with fresh optimizers and
identical reset annealing (102 soft / 26 hard). Only the target changes, `tq`
versus `t0`; student `xq`, native kwargs, masks, fitting order, prefix/suffix,
endpoints and all other settings stay fixed. The same 16 search records are
evaluation-only. Initial, true last-soft pre-update and final hard states must
be preserved; final factors cannot reconstruct the earlier Jacobian snapshot.

Support requires at least 5% lower pooled **common-t0 MSE** than both the
conditional arm and initial state, improvement on at least 12/16 records, and
no record more than 10% worse than either comparator. These are engineering
thresholds, not statistical significance. Both assembled arms must also receive
the same development evaluation: 2,048 NLL targets with legacy FP32 CE, 20 ARC
items with every option score, and four complete generation records. All-A,
EOS-only or other degenerate outputs remain failures; boundary improvement with
worse assembled NLL is not promotion. No full-model training follows automatically.

The recipe caps work at 256 accepted updates plus four discarded preflight
updates, 3,600 aggregate model-process seconds and at least 4 GiB headroom.
It requires budget projection before fitting, fresh-process FP32 restoration,
and separate packed-code/scale reload checks that distinguish scale casting
from packing. It does not authorize another grid or claim a recovered model.

## Offline software probe

```sh
python tools/probe_activation_diagnostic.py --output runs/activation-diagnostic
```

This constructs a random BF16 four-block native Qwen model, fits one small
rotated decoder, then inspects windows `[0,1)` and `[3,4)` on separate random
records. It verifies the first-window negative control, reverse-order replay,
and unchanged source, learned state and incoming flags. It does not download
weights or claim pretrained language quality. Use a new or empty output
directory.

The reviewed CPU run passed the full **1,439-test** suite, including 26 native
capture tests and 32 independent metric tests. All three search records have
exact reverse-order replay and unchanged state. Their 40 valid tokens produce:

| Window | Total error MSE | Conditional MSE | Target-gap MSE | Cross-term mean |
|---|---:|---:|---:|---:|
| `[0,1)` | 0.0000476532 | 0.0000476532 | 0 | 0 |
| `[3,4)` | 0.0003661582 | 0.0001857624 | 0.0002950222 | -0.0001146264 |

The FP64 decomposition closes exactly for this fixture. The later cross term
is negative, illustrating why the individual error terms cannot simply be
treated as additive causal contributions. These are random-model activation
measurements, not pretrained-quality improvements or a reproduction of the
separately sealed 9B run above.
[Validation record](../validation-activation-diagnostic.json).
