# Final paired training findings

Both conditions completed training, all diagnostic stages, export and independent fresh reload. Both training audits pass every recorded check: **8,192 updates total**, with 4,096 per condition (3,264 soft / 832 hard), all expected gradients present, finite FP32 parameters/moments/snapshots, and last actual soft snapshot sharpness 29.8828125. Teacher bypass is bitwise exact before/after training in both. Factors/B move on 128 steps per window and A on 127, consistent with zero-initialized B. The saved source, calibration, evaluation and training provenance matches.

| Measurement | Unrotated | Rotated |
|---|---:|---:|
| Training wall time | 731.303 s | 1309.048 s |
| Optimization | 586.979 s | 930.732 s |
| Window caching | 116.959 s | 324.059 s |
| Peak allocated / reserved | 20.207 / 20.297 GiB | 20.221 / 20.297 GiB |
| Local hard-MSE windows improved | 32/32 | 32/32 |
| Median local MSE after/before | 0.05389 | 0.12460 |
| Final-layer probe relative RMSE | 1.09071 | 1.07104 |
| Final-layer probe cosine | 0.29393 | 0.38213 |
| Hard-FP32 diagnostic NLL | 14.439949 | 9.257334 |
| Fresh-packed diagnostic NLL | 14.437115 | 9.255346 |
| ARC correct, each stage | 1/20 | 8/20 |

The phase audit independently reconciles all eight phase-time sums and memory maxima across 32 windows in both conditions. Global training peaks equal the largest recorded phase peaks. Rotated training takes 1.790× the time, with similar peak memory. These measurements include audit overhead.

The final distribution report verifies identical module selections, sampled rows, original source-weight and teacher-input statistics, and original hidden-probe references. Sampled median weight kurtosis changes from 3.25305 to 3.05811 under rotation, with little further change after correction. Teacher-input median kurtosis changes from 615.74567 to 2.94384; those are **original-teacher** samples, not trained student activations. Weight effects vary by family, including slightly higher query-projection kurtosis. On the single saved NLL hidden-state probe, rotation lowers relative RMSE and raises cosine at every layer, while substantial final-layer distortion remains.

Local MSE is conditional on each model's own quantized prefix, so these reductions are not identical-target comparisons and do not establish whole-model fidelity. Huber losses are a different quantity. Numerical improvements and completed execution do not establish useful generation; every generation sample at all three stages in both conditions reaches the 128-token cap without EOS, and text usefulness is assessed in the separate manual review.

For both conditions, all 200 projection payloads and 564 retained tensors verify exactly. Cached/uncached packed-runtime parity passes, and each independent fresh reload exactly matches its saved packed-runtime logit reference. This does not imply equality to FP32 wrappers: scale rounding and row-chunk computation have separately measured differences. The poor hard-FP32 scores already precede serialization.

Detailed evidence: `final-training-findings.json`, both `*-training-audit.json` files, `paired-training-audit.json`, both `*-training-interpretation.json` files, and `distribution-comparison.json/.md`. Method, source integration and historical-recipe limits are recorded in `method-and-fixes.md`. No checkpoint reads, GPU execution, frozen checkout edits or old-run changes were used for this final audit.
