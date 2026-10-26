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

## Next pretrained test, only after the active pair finishes

1. Review both completed calibration-diversity arms first. Preserve their
   artifacts, local patches and evaluation protocol.
2. On a common frozen compressed prefix, measure target disagreement and both
   hard reconstruction errors at a later window. Window 0 with retained original
   embeddings cannot distinguish these modes. Use calibration/search-validation
   records, not benchmark answers.
3. If disagreement is material, run a bounded paired calibration of that same
   later window. Change target provenance alone: identical original checkpoint,
   prefix, records, signs, initialization, updates, annealing, learning rates,
   loss, tensor coverage and evaluation. First preflight memory and correctness.
4. Compare both target errors on disjoint records and the assembled model's
   behavior. Better training loss alone cannot justify a full rerun. Endpoint
   coverage and the complete approximately 2 GB language size target remain
   separate requirements.

The optimizer audit found no extra group-count division in the masked loss.
Default-scale synthetic factor gradients were well above Adam's epsilon;
smaller hypothetical gradients showed the expected epsilon damping. Real
attribution requires per-coordinate bias-corrected optimizer moments. This
control does not change epsilon, learning rates, loss scaling or the schedule.
