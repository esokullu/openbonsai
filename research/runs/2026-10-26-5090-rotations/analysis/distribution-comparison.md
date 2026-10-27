# Paired distribution comparison

Both conditions completed. Source provenance, sampled row IDs and original-source/teacher statistics match. Values below are medians of per-module sample statistics, not pooled model moments.

| Sample view | Kurtosis | Fraction >6 RMS (%) | Max absolute / RMS |
|---|---:|---:|---:|
| Original source weights | 3.253 | 0 | 4.9461 |
| Unrotated weights before correction | 3.253 | 0 | 4.9461 |
| Unrotated weights after correction | 3.2528 | 0 | 4.9507 |
| Rotated weights before correction | 3.0581 | 0 | 4.528 |
| Rotated weights after correction | 3.0583 | 0 | 4.5279 |
| Original teacher inputs | 615.75 | 0.13428 | 52.699 |
| Rotated teacher inputs | 2.9438 | 0 | 4.6044 |

Per-family medians:

| Projection family | Final weight kurtosis U / R | Teacher runtime-input kurtosis U / R |
|---|---:|---:|
| linear_attn.in_proj_qkv | 3.4094 / 3.0985 | 1770.6 / 2.1414 |
| linear_attn.in_proj_z | 3.4394 / 3.2032 | 1770.6 / 2.1362 |
| linear_attn.out_proj | 3.1982 / 3.0089 | 556.57 / 3.4429 |
| mlp.down_proj | 3.1095 / 3.0042 | 550.52 / 3.7992 |
| mlp.gate_proj | 3.3004 / 3.1185 | 227.09 / 2.9202 |
| mlp.up_proj | 3.188 / 3.0516 | 227.09 / 2.8941 |
| self_attn.k_proj | 4.4416 / 3.4047 | 1280.7 / 2.4017 |
| self_attn.o_proj | 3.2931 / 3.0134 | 48.339 / 3.5265 |
| self_attn.q_proj | 6.4085 / 6.5452 | 1280.7 / 2.4002 |
| self_attn.v_proj | 3.7572 / 3.1507 | 1280.7 / 2.4006 |

Hidden-state drift on the first NLL record (zero-based layer; U / R):

| Layer | Relative RMSE U / R | Cosine U / R |
|---:|---:|---:|
| 0 | 0.23456 / 0.18862 | 0.97592 / 0.98305 |
| 3 | 0.52782 / 0.4772 | 0.88378 / 0.91684 |
| 7 | 0.61276 / 0.59351 | 0.79457 / 0.80581 |
| 11 | 0.64421 / 0.6133 | 0.76512 / 0.7905 |
| 15 | 0.74865 / 0.70011 | 0.69995 / 0.73024 |
| 19 | 1.0355 / 0.94921 | 0.39986 / 0.49824 |
| 23 | 0.98537 / 0.90555 | 0.45965 / 0.52762 |
| 27 | 0.99043 / 0.89861 | 0.45116 / 0.51416 |
| 31 | 1.0907 / 1.071 | 0.29393 / 0.38213 |

The JSON retains all 32 layers, block-kind summaries, per-family outlier fractions, full summary ranges and correction changes. No trained student activation distribution or projection-family hidden-state drift was captured.

Interpretation limits:

- Weight statistics use at most eight evenly spaced rows per projection, all input coordinates; they are not full-matrix or pooled model moments.
- Activation statistics use at most 64 active token rows of the first calibration sample under the unmodified original teacher prefix. They are not trained student activations.
- Rotated weights are transformed in FP32; recorded runtime inputs use the BF16 runtime transform. Their numerical precision differs.
- Kurtosis/outlier comparisons describe coefficient distributions in explicitly labelled bases. No vectors are subtracted or cosine-compared across original and rotated coordinates.
- Correction RMS ratios compare final effective weights with initial effective weights inside each condition's same coordinate system and same sampled rows.
- Summaries are unweighted distributions of per-module statistics; medians and ratios of medians are not pooled kurtosis, pooled outlier fractions, or energy-weighted norms.
- The >6*RMS fraction is the saved per-sample threshold statistic. Kurtosis follows the saved runner definition; neither is estimated anew from unavailable tensors.
- Hidden drift uses the first NLL record and common original model-output coordinates. Only layer-level probes exist; projection-family hidden drift is unavailable. Block-kind grouping is descriptive, not causal attribution.
- Changes in these diagnostics do not establish downstream quality or explain it causally. Both conditions use the same repeatedly reused development diagnostics.
