# Smaller Adam epsilon: a second local improvement

Changing AdamW epsilon from **1e-8 to 1e-9** reduced first-layer hard
reconstruction MSE by **27.187% on calibration and 24.495% on search** in a
controlled RTX 5090 experiment. All 128 calibration and 16 search records
improved. This establishes a benefit for this fixed first-window recipe;
**useful full-model quality and the approximately 2 GB goal remain unachieved**.

## What was controlled

C512 started fresh from the original Qwen3.5-9B checkpoint and was compared
with authenticated B512 from the [longer-calibration control](window0-calibration-result.md).
Both used the same six first-window projections, signed block-4096 Hadamard
rotations, group128, rank64 corrections, sigmoid factors, Huber objective,
initialization, record order and 512-update schedule: 409 soft and 103 hard
updates. The same 128 records were exposed over four epochs. Learning rates,
clipping and other optimizer settings were unchanged; epsilon changed in
every optimizer group, including LoRA.

The control reproduced B's saved state and native first-layer output exactly,
then restored the pristine initialization before C fitting. Original inputs,
targets, masks and native arguments matched across all 144 records. At window
zero, original and conditional teacher-prefix targets coincide. No search
record entered fitting or selected the final state. Search is reused development
data, not a new final confirmation set.

## Measured reconstruction

Pooled MSE uses summed FP64 squared error over 59,855 valid calibration
positions and 7,203 search positions, excluding padding. It is not a mean of
per-record means.

| Final hard state | Calibration MSE | Search MSE |
|---|---:|---:|
| B512, epsilon 1e-8 | 0.000162908354157672 | 0.00016801047914713087 |
| C512, epsilon 1e-9 | 0.00011861900800478741 | 0.00012685623910517702 |

Individual MSE reductions ranged from 22.812% to 38.255% on calibration and
23.657% to 25.718% on search, with no ties or worsening. These are MSE
reductions, not RMSE reductions or language-quality scores.

A separate same-parameter probe compared soft and hard output on one fixed
record at the actual pre-update-409 state. Its hardening penalty increased
from **6.329% for B to 11.430% for C**, but absolute hard MSE still fell from
0.00015580845748767202 to 0.00011568367065392189. The actual last-soft snapshot
and sharpness 29.9560546875 were preserved through the hard phase. Consecutive
training losses at steps 409 and 410 use different records and do not measure
this discontinuity.

Final raw-mu, scale and threshold direction-norm retention rose from
0.111745 / 0.081474 / 0.069595 to 0.380545 / 0.286356 / 0.255020.
The statistic is `sqrt(sum(damped_direction²) / sum(epsilon_free_direction²))`
over matching coordinates with `r > 0`, where `r = sqrt(vhat)` and `vhat` is
the bias-corrected second moment. It is neither retained squared
energy nor an actual learning-rate-scaled update. B and C have different late
moment histories, and the intervention affects all optimizer groups; this
does not assign the gain to a particular factor family or justify another sweep.

## Execution and artifact scope

The completed attempt performed **512 finite scientific updates plus two
discarded timing updates** in **498.609 aggregate model-process seconds**,
including preflight, fitting and fresh reload, within the unchanged 1,800-second
family cap. Peak allocated/reserved VRAM was 21,705,918,976 / 21,781,020,672
bytes; minimum recorded headroom was 10,572,791,808 bytes. All three owned
model processes exited and were reaped.

Fresh-process replay reproduced the compact state and fixed native output
bitwise, with zero maximum output error. The **54,047,955-byte compact state
requires the original weights**; it is not a standalone compressed model.
The sealed research inventory contains 405 files totaling 207,136,037 bytes,
excluding the final seal. No full-model NLL, ARC, generation, packed export,
endpoint calibration, artifact-size target or inference acceleration was tested.

The first attempt stopped at a worker guard before any model work. A narrow
Windows launcher-classification correction was validated natively before the
second attempt; the originally rejected PID was not recorded, so its identity
remains unknown. A later administrative sealer correction accepted the training
review's documented status string. Neither correction changed the scientific
recipe or reran completed model work; failed evidence was preserved.

## Provenance and next boundary

The [validation receipt](../validation-window0-epsilon.json) binds the remote
source freeze, numeric/training/execution reviews, final report and seal.
This publication reviews returned `read_thread` evidence; it does not claim
an independent local rehash of remote tensor payloads. The executed adapter
remains frozen with the remote run. Its historical BF16-cast source policy
differs from the newer preserve-checkpoint pipeline, so the current public CLI
is not presented as an exact reproduction of this experiment.

Epsilon 1e-9 is a **provisional decoder setting for this local recipe**, not a
new universal default. Head optimizer defaults remain unchanged. A separately
bounded head-only resource/correctness preflight is the next question; these
results authorize no automatic full-model run, endpoint fit or optimizer grid.
