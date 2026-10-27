# Observing stored AdamW moments

`openbonsai.optimizer_diagnostics.summarize_adamw_coordinates` is an opt-in,
read-only observer. It does not install a training hook or change learning
rates, epsilon, losses, schedules, gradients, or optimizer state. The observations
describe stored moments; they do not establish a cause of quality loss or
recommend a training recipe.

Call it between completed optimizer updates, while no other operation mutates
the optimizer. For a diagnostic run, useful explicitly selected capture points
are **after update 1, after the last soft update, after the first hard update,
and after the final update**. A callback invoked before `optimizer.step()` sees
the preceding stored moments, even if gradients for the next update already
exist. The observer uses each parameter's stored step, not a supplied schedule
step or an upcoming update number.

```python
import json
from openbonsai.optimizer_diagnostics import summarize_adamw_coordinates

# Existing training code performs optimizer.step() before this observation.
named = dict(model.named_parameters())
members = {id(p) for group in optimizer.param_groups for p in group["params"]}
report = summarize_adamw_coordinates(
    optimizer,
    [(name, p) for name, p in named.items() if id(p) in members],
    chunk_elements=65536,
)
with open("moments-after-selected-update.json", "x") as output:
    json.dump(report, output, allow_nan=False, indent=2)
```

The caller chooses names and output ownership. A named subset is allowed, but
the report records both named and total optimizer parameter/element counts.
Duplicate names, duplicate named parameters, and duplicate membership anywhere
in the optimizer are rejected. Families are the final dotted name component,
such as `raw_scale` or `lora_A`; family and overall summaries pool coordinates,
not equally weighted parameter averages. Each coordinate uses its own group's
settings and parameter's stored step before aggregation.
Settings are read from the current stored groups; the observer does not
reconstruct historical beta, epsilon, or learning-rate schedules.

## Quantities and domains

For stored first and second moments `m`, `v`, completed-update step `t`, and
the owning group's `beta1`, `beta2`, and positive `eps`, the observer computes
in CPU FP64:

```text
mhat = m / (1 - beta1**t)
r    = sqrt(v / (1 - beta2**t))
u0   = mhat / r          only where r > 0
ue   = mhat / (r + eps)  on all observed coordinates
d    = r / (r + eps)
```

Per parameter, suffix family, and overall, the report includes:

- `r_over_epsilon`: minimum, maximum, RMS, and strict fractions below 1, 9,
  and 99, using all initialized coordinates. These ratios correspond to damping
  below 0.5, 0.9, and 0.99 respectively; they are observations, not decision rules.
- `mhat_rms`, all-coordinate `epsilon_direction.rms`, and positive-`r`-only
  `undamped_direction.rms`, with their respective counts.
- The minimum, maximum, mean, and RMS of `d` over all initialized coordinates.
- `direction_energy_attenuation.ratio`, defined as
  `sqrt(sum_{r>0}(ue**2) / sum_{r>0}(u0**2))`. Both sums use the **same** domain.
  This is the `u0**2`-weighted RMS of damping, not mean damping and not the ratio
  of RMS values computed over different domains.

This attenuation describes normalized adaptive **directions**. It excludes
learning-rate scaling, decoupled weight decay, and native kernel rounding.
Consequently a pooled value across groups with different learning rates is not
an update-energy attenuation measurement. Group settings and per-parameter
stored steps remain visible in the report.

Uninitialized state is counted separately and excluded from numerical summaries.
Initialized zero moments remain observed coordinates. Stored zeros do not prove
that a coordinate has never had a gradient or nonzero moment. The report counts
`m == 0, v == 0`, `m != 0, v == 0`, and `m == 0, v > 0` separately.
In particular, FP32 underflow can leave `v == 0, m != 0`: `ue` remains part of
the all-coordinate epsilon direction, but its epsilon-free comparator is
undefined and excluded from attenuation. A zero denominator energy produces
JSON `null`, as do unavailable statistics; it is never silently reported as
unit attenuation. Many zero-moment coordinates can lower unweighted mean damping
while contributing no direction energy.

## Supported state and bounds

The observer accepts the exact standard `torch.optim.AdamW` class with dense
materialized real FP32 parameters and matching contiguous FP32 moments without
gradients. An initialized state must contain exactly `step`, `exp_avg`, and
`exp_avg_sq`, with a positive integral stored step, finite moments, and
nonnegative second moments. Empty state is uninitialized. Epsilon must be finite
and positive; betas must lie in `[0, 1)`; learning rate and weight decay must be
finite and nonnegative. AMSGrad, differentiable optimization, complex or lower
precision moments, noncontiguous moments, partial/extended state schemas, and
nonfinite or unrepresentable diagnostic arithmetic are explicitly rejected.

The report records `foreach`, `fused`, `capturable`, and `maximize` settings; it
does not execute those kernels or reconstruct gradients from moments. Stored
moments already reflect the optimizer's preceding updates. The CPU oracle tests
cover standard and foreach AdamW updates within FP32 arithmetic tolerance, not
bitwise equivalence to every native kernel.

Each contiguous moment is sliced **before** conversion to CPU FP64. Temporary
numeric storage is proportional to `chunk_elements`, with a constant number of
chunk-sized arrays; no full moment or parameter copy, concatenation, or quantile
buffer is made. The JSON report scales with the number of named parameters and
families. Chunk boundaries can change the final floating-point reduction by
roundoff. Parameters and gradients are not read or copied, and no parameter,
gradient, moment, step, or RNG state is changed.

CPU observer validation does not measure CUDA synchronization or transfer
overhead. No CUDA performance, completed-run diagnosis, or model-quality result
is established by these tests. This observer is not connected to the default
trainer or CLI; using it in an experiment requires an explicit caller at the
chosen completed-update boundary.
