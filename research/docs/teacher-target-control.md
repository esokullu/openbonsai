# Teacher target provenance control

The existing window objective trains a compressed block on inputs produced by
the already-compressed prefix, using original **window** weights as its target.
That measures the block's additional error on the student's current inputs.
It does not directly ask the block to correct error accumulated in earlier
blocks. This distinction is a research hypothesis for the failed 9B quality
results, not an established cause or a newly proven CAT-Q implementation bug.

An opt-in `--teacher-prefix original` control now uses the original model's
prefix for the teacher target. The default remains `quantized`; preserved
experiments and their running comparisons are unchanged.

## What the source establishes

At [SliderQuant commit bf2641e](https://github.com/deep-optimization/SliderQuant/tree/bf2641ece550d5f6edb0028c2ae8b0ee658bc9bf),
the ancestor implementation keeps separate original and quantized activation
streams:

- The [default loss flags](https://github.com/deep-optimization/SliderQuant/blob/bf2641ece550d5f6edb0028c2ae8b0ee658bc9bf/main.py#L130)
  enable the last base loss and disable the optional quantized-target loss.
- Its [base loss](https://github.com/deep-optimization/SliderQuant/blob/bf2641ece550d5f6edb0028c2ae8b0ee658bc9bf/quantize/sliderquant.py#L125)
  compares the student's output on `quant_input` with `fp_tar`.
- [Original targets](https://github.com/deep-optimization/SliderQuant/blob/bf2641ece550d5f6edb0028c2ae8b0ee658bc9bf/quantize/sliderquant.py#L591)
  use `windows_fp_inps`; [both streams advance separately](https://github.com/deep-optimization/SliderQuant/blob/bf2641ece550d5f6edb0028c2ae8b0ee658bc9bf/quantize/sliderquant.py#L692)
  as windows move. The original teacher bypass excludes learned corrections.

OpenBonsai's existing conditional target resembles that implementation's
optional `use_quant_tar_loss`, rather than its default base loss. The released
[CAT-Q code](https://github.com/IntelChina-AI/BitTern/tree/5a8fcd4f7e0366554b732d300a37e6ea467c3c35/projects/cat-q)
does not include the training loop. CAT-Q Equation 7 writes a shared input
symbol without resolving the origin of the cached teacher stream. ScaleQ's
reference to CAT-Q does not establish that it copied the ancestor's cache
semantics. This option is therefore source-informed, not a claim to reproduce
the missing CAT-Q or ScaleQ trainer.

## Precise behavior

For tokens `t`, original prefix `P`, current hard-compressed prefix `Pq`,
original window `W`, and trained window `Wq`:

| Mode | Student | Teacher target |
|---|---|---|
| `quantized` (unchanged default) | `Wq(Pq(t))` | `W(Pq(t))` |
| `original` (experimental) | `Wq(Pq(t))` | `W(P(t))` |

Student inputs and each native block's mask/rotary kwargs still come from the
same first pass. The second mode obtains only the target from an independent
original-weight pass. No rotation is moved across a norm, nonlinearity, gate,
or recurrent state. Both forwards stop at the window boundary; neither
materializes vocabulary logits. Original weights remain frozen, and the
teacher ignores learned factors and low-rank corrections.

The known bounded embedding prototype also uses its original-source bypass in
the teacher pass, while retaining the caller-selected embedding state for
student inputs. Packed/mixed embeddings and unknown replacements are rejected
before mutation because they cannot establish the required original source.
Known packed/mixed decoder weights are likewise rejected. This supports
teacher capture around a frozen endpoint; it does not make decoder-window
training a substitute for explicit embedding/head calibration.

The extra pass requires more computation, particularly for later windows.
The cache still holds one student input, one target and per-block kwargs per
record, on CPU or disk. It does not retain a second model or a second complete
hidden-state stream. Caller quantization flags and hooks are restored on
failure. The selected mode and a textual target definition appear in training
and export metadata. Hard-window MSE is measured against the selected target:
values from different modes are **not the same objective**.

The Python interface is `TrainingConfig(teacher_prefix="original")`, or
`cache_window(..., teacher_prefix="original")` for bounded diagnostics.
The standard training CLI accepts `--teacher-prefix original`; this flag alone
does not enable rotations. Rotated experiments use their existing module setup
and pass the same config to `train`. No existing packed artifact can recover
its original teacher weights by changing this flag.

## Measured local result

`tools/probe_teacher_targets.py` runs paired calibration on a random native
four-block hybrid Qwen fixture, BF16 source weights, block-64 rotations, group
128, rank 4 and sigmoid/Huber settings. Four random 16-token records fit each
arm for eight epochs; two different records check it. Each arm starts from
identical source weights, signs and trainable initialization. These are random
software fixtures, not pretrained language-quality data.

| Measured outcome | Conditional target | Original-prefix target |
|---|---:|---:|
| Validation hard-logit MSE to original model | 0.024494 | 0.024739 |
| Validation packed-logit MSE to original model | 0.024485 | 0.024720 |
| Actual complete fixture artifact bytes | 374,342 | 374,332 |
| Fresh-process packed reload logits | Exact | Exact |

The original-prefix control did **not** improve logit reconstruction in this
tiny test. Before fitting, target disagreement was zero at window 0 and
`3.0735e-4` at window 3. That demonstrates different target semantics and
accumulated prefix error, not useful pretrained recovery. Original-teacher
bypass was exact before and after fitting; source tensors were unchanged.
FP16 export rounding and runtime arithmetic caused a maximum BF16-logit
difference of 0.0078125 from the hard training path in both arms. Reload was
compared with the actual stored-scale runtime, not that pre-export path.
Execution still reconstructs floating weights; file size is not an acceleration
claim. The small size difference is not a meaningful compression gain.

Reproduce in a new directory:

```bash
python tools/probe_teacher_targets.py --output runs/teacher-target-control
```

Within each arm, the report evaluates both target definitions on the same
held-out student inputs. The trained prefixes can differ between arms.
It retains initialization/source hashes, training records, exact reload
receipts and actual bytes. CPU phase times are diagnostic only; the final
validation ran alongside the test suite and is not a throughput comparison.
See [the validation record](../validation-teacher-target.json)
for the tested source hashes and results.

## Completed pretrained single-window control

Both calibration-diversity arms completed without useful generation. The
[read-only diagnostic](activation-diagnostic.md) restored the final128 FP32
research state exactly and found a material target discrepancy on all 16 later-
window search records. Its negative-control window had identical teacher
outputs. This supports a target-provenance experiment, not a quality remedy.

The completed control gave both arms 128 additional updates at window `[16,17)`
from the same saved final128 window state, with fresh optimizers and identical
restarted schedules. The hard prefix/suffix, source, rotations, fitting records,
order, precision, loss and learning rates remain fixed. Only the target changes;
the student's quantized-prefix input and native kwargs are common. This is a
warm-start repair test, not resumed original optimization or a full-model rerun.

Both final hard arms were compared against the **common original target** on
the same 16 disjoint search records (7,203 active positions), then with identical
assembled development NLL, ARC and generation settings. Results below describe
the hard research model with FP32 factors/A/B, not a newly evaluated complete
packed artifact. The original and initial quality scores were reused with
authenticated source/probe/protocol bindings.

| State | Common-T0 boundary MSE | Development NLL | ARC | Useful generations |
|---|---:|---:|---:|---:|
| Original BF16 | — | 2.4237768106 | 19/20 | 2/4 complete, 2 relevant but capped |
| Initial final128 state | 0.196574566897 | 9.1216306852 | 8/20 | 0/4 |
| Conditional-target repair | 0.196725874517 | 9.1216917017 | 8/20 | 0/4 |
| Original-prefix-target repair | 0.132229789576 | 8.2791870161 | 8/20 | 0/4 |

Original-prefix targeting reduced common-T0 MSE by **32.73% versus initial**
and **32.78% versus conditional**, with all 16 records improving and no
regressions. It passed the predeclared local gate. NLL fell by 0.84250
nats/token against the conditional repair. However, every compressed ARC
prediction was A; eight reference answers happen to be A. Conditional outputs
were immediate EOS, while all four original-prefix outputs were exactly `#`
followed by EOS. **Useful model quality did not recover.** These are the repeated
2,048-token / 20-question / 4-generation development subsets, not full-pilot or
unused final-confirmation results.

At identical saved pre-update102 parameters, original-prefix common-T0 MSE was
0.133738962795 soft and 0.133728812382 hard. Conditional common-T0 MSE was
0.196779880091 soft and 0.196918048817 hard. This local measurement shows no
large hardening error jump. Final-versus-saved-hard improvement also includes
update102, so it cannot isolate the 26 hard updates. It is not a full-model
last-soft quality comparison or proof of optimal optimizer conditioning.

Both arms completed 102 soft and 26 hard scientific updates; four resource
preflight updates were discarded. Both fresh processes verified selected FP32
state and stored-code/scale/sign payload parity against the authenticated common
base. Six packed candidates cost 49,875,679 bytes (conditional) and 49,884,220
bytes (original-prefix). With their 23,012-byte manifest each, the totals are
49,898,691 and 49,907,232 bytes. These are **dependent window deltas**, not
standalone model sizes. Their complete research delta directories, including
selected state and probe references, occupy 93,260,793 and 93,269,334 bytes.
Codec and FP16-scale policies were unchanged; casting drift remains distinct
from lossless serialization.

The three v2 model processes took 761.140 seconds. Including a prior metadata-
guard failure with no updates, the cumulative time was 786.046 seconds, within
the 3,600-second cap. Across 424 measured phases, peak allocation was
21,706,935,296 bytes (20.216 GiB), peak reservation 22,036,873,216 bytes, and
minimum headroom 10,316,939,264 bytes. Short failed answers are not speed evidence.

The completed run is `D:/openbonsai/runs/2026-10-26-5090-teacher-target-repair-v2`.
Its reviewed report SHA256 is
`6a7e63a133756c45aec3ac25f9fc67d76a09df138b76ab4f91a70dcdf1a2edec`;
the final seal is
`8e9e9aed562eb9fe25508453dedc7dca782707447e3aa962e19898a0eff1f5e7`
(739 files / 1,264,382,456 research bytes). Prior artifacts and local fixes
passed preservation checks. These are reviewed remote measurements, not a
rerun of this documentation branch.

The result supports testing original-prefix targets sequentially from the
original BF16 initialization, rather than assuming a repair of one middle
window can recover the whole model. A controlled next pass must keep the same
128×1 data/update budget, wrapper names, rotations, endpoints and codec, and
validate the historical control at window0 where targets coincide. No quality
gain from that next experiment is established here. Endpoint coverage and the
approximately 2 GB language-model size target remain separate requirements.

The optimizer audit found no extra group-count division in the masked loss.
Default-scale synthetic factor gradients were well above Adam's epsilon;
smaller hypothetical gradients showed the expected epsilon damping. Real
attribution requires per-coordinate bias-corrected optimizer moments. This
control does not change epsilon, learning rates, loss scaling or the schedule.
