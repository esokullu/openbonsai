Paired training cost and fixed-input hidden drift

Both training phases are complete. These comparisons do not assess final export, fresh reload or whole-model quality.

| Arm | Training actual / projected (s) | Cache actual / projected (s) | Optimization actual / projected (s) | Peak allocated / reserved (GiB) |
|---|---:|---:|---:|---:|
| rotated16x8 | 1303.809 / 1233.994 | 323.431 / 250.264 | 926.978 / 930.247 | 20.220553 / 20.296875 |
| rotated128x1 | 3965.794 / 3310.642 | 2604.822 / 1975.430 | 935.228 / 927.532 | 20.219088 / 20.296875 |

For 128×1, actual cache time was 131.86% of its projection (+31.86%), while optimization was 100.83%. Total training exceeded projection by 655.152 seconds (+19.79%). For 16×8, cache was 129.24% of projection and optimization 99.65%.

The actual 128×1/16×8 ratios were 8.0537 for cache, 1.0089 for optimization, and 7.9863 for other trainer phases. Both arms performed 128 optimizer updates per window; the latter phases include hard reconstruction evaluation over each arm's calibration set. Training time excludes loading, conversion, export and downstream evaluation.

| Layer | Relative RMSE: 16×8 / 128×1 | Cosine: 16×8 / 128×1 |
|---:|---:|---:|
| 0 | 0.177911 / 0.179426 | 0.984762 / 0.984612 |
| 7 | 0.577087 / 0.578596 | 0.818105 / 0.816951 |
| 15 | 0.689069 / 0.688876 | 0.739649 / 0.740026 |
| 23 | 0.900931 / 0.908681 | 0.536462 / 0.532436 |
| 31 | 1.093587 / 1.086154 | 0.411973 / 0.408845 |

On the same fixed NLL record, 128×1 has lower relative RMSE at 6/32 layers and higher cosine at 4/32. At the final layer, the teacher RMS is 3.671063, versus 3.732356 for 16×8 and 3.663158 for 128×1. The slightly lower final relative RMSE for 128×1 accompanies closer RMS and lower cosine; it does not show improved directional agreement.

These are correlated depth measurements on one input. Local reconstruction loss and this drift comparison do not establish whole-model usefulness or a causal explanation for generation behavior.

The companion JSON preserves every depth, phase values, projection assumptions and input hashes. Hashes for metrics.json describe bytes at capture; those producer documents may later gain export phases, so the exact consumed training objects are embedded. Other inputs are complete training, drift and preflight evidence.
