# Longer first-layer calibration: a local improvement

A controlled RTX 5090 experiment reduced first-layer hard reconstruction error
with a longer calibration schedule. **It did not measure recovered language
quality or produce a smaller deployable model.** The preceding
[full-model result](sequential-original-target-result.md) remains a failure of
useful generation and the approximately 2 GB language-artifact goal.

## What changed

Both arms started fresh from the same original Qwen3.5-9B checkpoint and seeded
initialization. They trained the same six window-0 projections, with signed
block-4096 Hadamard rotations, group128, rank64 corrections, sigmoid factors,
Huber loss and unchanged AdamW settings. Original embeddings supplied identical
inputs and original targets; the teacher-prefix distinction disappears here.

A used the existing 128 records once: 102 soft and 26 hard updates. B used the
same records over four shuffled epochs: 409 soft and 103 hard updates. This
changes repeated exposure, annealing duration and learning-rate trajectories
together; it does not isolate step count or sample diversity. B started with a
fresh optimizer and original parameters, rather than continuing A.

A reproduced the authenticated historical final FP32 state, shared update
evidence and same-input native layer output exactly. Both arms were measured
at their fixed final hard states on the same 128 calibration records and 16
disjoint search records. Search records were excluded from fitting and are
development data, not a fresh final confirmation set.

## Reconstruction results

Pooled MSE compares output hidden states with the original first-layer target,
using 59,855 valid calibration positions and 7,203 valid search positions.

| State | Calibration MSE | Search MSE | Search relative RMSE |
|---|---:|---:|---:|
| Untrained hard initializer | 0.002497925 | 0.002489267 | 0.464264 |
| A: 128 updates | 0.000309119 | 0.000311311 | 0.164182 |
| B: 512 updates | 0.000162908 | 0.000168010 | 0.120614 |

B reduced MSE versus A by **47.299% on calibration and 46.031% on search**.
Every calibration and search record improved; individual search reductions
ranged from 42.061% to 50.142%. The search RMSE reduction was 26.537%, not 46%.
Search cosine improved from 0.986454 to 0.992701 and output/target RMS ratio
from 0.979562 to 0.991235.

At each arm's actual last-soft pre-update state, a separate fixed-record probe
compared soft and hard computation without changing parameters. Hard MSE rose
3.840% for A and 6.329% for B. The hard-phase Jacobians retained the actual
last-soft snapshots. These modest probe discontinuities do not establish
transition behavior across all records or later layers.

## Optimizer observations

All eight predeclared post-update observations preserved parameters, gradients,
moments, snapshots, flags and RNG. At B's final step, epsilon remained material:

| Factor family | Fraction with sqrt(vhat) below epsilon | Direction-norm retention |
|---|---:|---:|
| mu | 95.644% | 0.111745 |
| scale | 97.017% | 0.081474 |
| threshold | 97.921% | 0.069595 |

The last column is `sqrt(sum(damped_direction²)/sum(epsilon_free_direction²))`
on the same positive-variance domain, using stored bias-corrected moments.
It is neither retained squared energy nor an actual learning-rate-scaled update.
Late-soft and first-hard observations also showed damping. This supports a
controlled optimizer investigation; it does not prove that a smaller epsilon
would improve training or that epsilon caused full-model failure.

Global weight statistics changed little despite the local gain: zero fraction
was 30.899% for A and 30.949% for B. The combined low-rank correction norm rose
from 0.280803 to 1.116008. These summaries do not isolate which learned component
caused the improvement or justify amplitude or sparsity adjustments.

## Execution and limits

The experiment completed 640 scientific updates and four discarded preflight
updates in **614.171 aggregate model-process seconds**. Peak allocated/reserved
VRAM was 21,709,584,896 / 21,781,020,672 bytes; minimum recorded headroom was
10,572,791,808 bytes. All three model processes exited and were reaped.

The admission audit found one omitted preservation pass in the original time
estimate. Including it raised the forecast from 1,293.717 to 1,319.779 seconds,
still below the unchanged 3,600-second cap. The running source was preserved;
future drivers must count all required calls rather than inherit that omission.

Both compact FP32 states restored in a fresh process with bitwise-identical
fixed first-layer outputs. This is compact-state parity, not a new packed
full-model export. No NLL, ARC, generation, endpoint-calibration or final-artifact
size claim follows from this experiment. The runtime still reconstructs
floating-point weights.

The [validation receipt](../validation-window0-calibration.json) records the
source, data, numeric results and final experiment identity. A local benefit
does not authorize an automatic longer full-model run or optimizer sweep.
