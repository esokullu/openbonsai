# Separate prefix drift from conditional reconstruction

The current decoder calibration normally fits each window using inputs from
the already-compressed prefix. Its local error can improve while the assembled
model becomes worse. The [original-prefix target option](teacher-target-control.md)
offers a controlled alternative, but a random fixture did not establish a
quality improvement. Before choosing another 9B training run, this diagnostic
measures the difference on a **single frozen model state**.

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

## Planned use after the frozen 5090 pair

Complete both current diversity arms and review their export, reload and quality
results first. If quality remains poor and correctness gates pass, predeclare
the 128-sample arm's saved state as the common reference. Keep endpoints and
vision handling unchanged.

Use window `[0,1)` as a negative control: with original embeddings its two
teacher targets must agree. Use fixed later window `[16,17)` to measure inherited
prefix drift. Do not scan windows and choose the most favorable result.

Freeze 16 additional disjoint records from the declared calibration source as
**search-validation**. They may guide the next experiment and are not a final
held-out evaluation. Keep benchmark answers and the unused confirmation set out
of this selection. Preflight memory and wall time on a bounded subset before
measuring all records.

The current proposed engineering selector is a later-window target-gap RMS at
least half the original-target error RMS on at least 12 of 16 records, above
repeat/bypass noise. This is not a statistical or quality threshold. Inspect
the cross term and record-level distribution before choosing one bounded
target-provenance control. If the gap is small, inspect optimizer movement and
schedule evidence before considering a budget control. No full-model rerun or
large allocation search follows automatically.

Any selected pair must share the same prefix, suffix, source, window
initialization, data order and evaluation. Compare both target errors on the
same search records; comparing each arm only to its own teacher is insufficient.
After a local improvement, measure the assembled model and actual generation
before promoting it. A successful diagnostic is evidence for a next experiment,
not evidence that OpenBonsai matches Bonsai.

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
measurements, not pretrained-quality improvements. CUDA memory/timing and the
completed 9B checkpoint restoration remain unvalidated in this stage.
[Validation record](../validation-activation-diagnostic.json).
