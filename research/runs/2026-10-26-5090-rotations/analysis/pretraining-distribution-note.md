# Pretraining distribution comparison

The saved pretraining samples show a large reduction in teacher-input coefficient concentration after rotation, with a smaller overall change in weight distributions. This note was prepared while rotated training was in progress and makes no post-training quality claim. The subsequently completed pair is reported separately in `distribution-comparison.md`.

All 200 module names, sampled weight rows, sampled teacher token rows, original-source weight statistics, original-teacher input statistics, and source/calibration/evaluation provenance match between conditions. Initial effective-weight statistics equal the pre-correction statistics in both conditions, consistent with the zero-initialized correction.

Each value below is the **median of 200 per-module sampled statistics**, not a pooled moment. `W`/`x` denote original coordinates; `WR`/`xR` denote the recorded rotated coordinates.

| Sampled basis | Kurtosis | Fraction with absolute value >6 RMS | Maximum absolute value / RMS |
|---|---:|---:|---:|
| Original weights W | 3.25305 | 0% | 4.94606 |
| Rotated weights WR | 3.05811 | 0% | 4.52804 |
| Original teacher inputs x | 615.74567 | 0.13428% | 52.69859 |
| Rotated teacher inputs xR | 2.94384 | 0% | 4.60438 |

Per-family kurtosis medians (U = unrotated basis, R = rotated basis):

| Projection family | Modules | Weight U → R | Teacher input U → R |
|---|---:|---:|---:|
| linear_attn.in_proj_qkv | 24 | 3.409 → 3.098 | 1770.612 → 2.141 |
| linear_attn.in_proj_z | 24 | 3.439 → 3.203 | 1770.612 → 2.136 |
| linear_attn.out_proj | 24 | 3.198 → 3.009 | 556.565 → 3.443 |
| mlp.down_proj | 32 | 3.109 → 3.004 | 550.523 → 3.799 |
| mlp.gate_proj | 32 | 3.300 → 3.119 | 227.089 → 2.920 |
| mlp.up_proj | 32 | 3.188 → 3.052 | 227.089 → 2.894 |
| self_attn.k_proj | 8 | 4.442 → 3.404 | 1280.695 → 2.402 |
| self_attn.o_proj | 8 | 3.293 → 3.013 | 48.339 → 3.527 |
| self_attn.q_proj | 8 | 6.408 → 6.549 | 1280.695 → 2.400 |
| self_attn.v_proj | 8 | 3.757 → 3.150 | 1280.695 → 2.401 |

The weight effect is not uniform: `self_attn.q_proj` has slightly higher sampled kurtosis after rotation (6.408 → 6.549), despite a lower median >6-RMS fraction (0.02899% → 0.01984%). Rotated `mlp.down_proj` teacher inputs retain a small nonzero median >6-RMS fraction, approximately 0.000572%. A zero aggregate median does not mean every module has no outliers.

Limits: weight samples contain at most eight evenly spaced rows per projection; activation samples contain at most 64 active token rows from the first calibration sample under the **unmodified original teacher prefix**. They are not trained student activations. Weight transforms use FP32, whereas the saved runtime-input transform reflects BF16 rounding. The table compares coefficient-distribution shape in labelled bases, not vector differences or norm errors across different coordinate systems. Outlier thresholds are each sample's own RMS. Lower concentration does not establish improved quantization quality, hidden-state fidelity, or task performance.

Read-only evidence: each condition's `weight-distributions-before.json`, `activation-distributions.json`, and `config.json`. No post-training distribution, hidden-state result, checkpoint, GPU execution, source edit, or final paired utility was used.
