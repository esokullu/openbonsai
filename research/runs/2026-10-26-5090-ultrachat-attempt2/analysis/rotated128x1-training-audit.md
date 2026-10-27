# rotated128x1 training audit

Training audit: **PASS**. Observed 4096/4096 updates, 32/32 complete windows; stages {'soft': 3264, 'hard': 832}. Condition completion marker: True.

These are software and optimization measurements, not a quality gate.

| Window | Steps | Hard MSE before | Hard MSE after | Ratio | Soft Huber mean | Hard Huber mean | Correction/W (representative) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 128 | 0.00249792 | 0.000309119 | 0.12375 | 0.000258348 | 0.00016493 | 0.00431321 |
| 1 | 128 | 0.000252431 | 0.000131341 | 0.520307 | 7.96974e-05 | 6.16516e-05 | 3.24745e-05 |
| 2 | 128 | 0.000677181 | 0.000184539 | 0.27251 | 0.000128413 | 8.90159e-05 | 0.000459447 |
| 3 | 128 | 0.00205991 | 0.000181368 | 0.0880465 | 0.000168177 | 9.8837e-05 | 0.00124971 |
| 4 | 128 | 0.000790826 | 0.000185698 | 0.234816 | 0.00017056 | 9.33835e-05 | 0.000736625 |
| 5 | 128 | 0.00145753 | 0.00022072 | 0.151435 | 0.000207922 | 0.000109696 | 0.003263 |
| 6 | 128 | 0.00532579 | 0.000581248 | 0.109138 | 0.000631668 | 0.000297065 | 0.00518259 |
| 7 | 128 | 0.00913587 | 0.000913561 | 0.0999972 | 0.000908899 | 0.000516638 | 0.000768853 |
| 8 | 128 | 0.00529857 | 0.000584727 | 0.110355 | 0.000619024 | 0.000296678 | 0.00369385 |
| 9 | 128 | 0.00574467 | 0.000640704 | 0.11153 | 0.000694096 | 0.00035964 | 0.00321343 |
| 10 | 128 | 0.00702475 | 0.000855867 | 0.121836 | 0.000872512 | 0.000440117 | 0.00215452 |
| 11 | 128 | 0.00962405 | 0.00109839 | 0.11413 | 0.000995246 | 0.000608617 | 0.00254058 |
| 12 | 128 | 0.0051232 | 0.000733304 | 0.143134 | 0.000742988 | 0.000395789 | 0.00255809 |
| 13 | 128 | 0.00718231 | 0.000926583 | 0.129009 | 0.000998862 | 0.000489598 | 0.00370619 |
| 14 | 128 | 0.0118384 | 0.00147111 | 0.124266 | 0.001505 | 0.000753497 | 0.00358104 |
| 15 | 128 | 0.0203417 | 0.00211567 | 0.104007 | 0.00203977 | 0.0011298 | 0.00427438 |
| 16 | 128 | 0.0099072 | 0.00137164 | 0.138449 | 0.00139573 | 0.000712173 | 0.00303685 |
| 17 | 128 | 0.0125375 | 0.00170415 | 0.135924 | 0.00172367 | 0.000877507 | 0.00390199 |
| 18 | 128 | 0.0424588 | 0.00482983 | 0.113753 | 0.00483568 | 0.00242613 | 0.00527269 |
| 19 | 128 | 0.0847261 | 0.00897025 | 0.105873 | 0.00831845 | 0.00438078 | 0.0069983 |
| 20 | 128 | 0.0332662 | 0.00466104 | 0.140113 | 0.00422993 | 0.00249462 | 0.00402163 |
| 21 | 128 | 0.043097 | 0.00621592 | 0.144231 | 0.00567471 | 0.00326289 | 0.00435702 |
| 22 | 128 | 0.0691221 | 0.0120748 | 0.174689 | 0.0108029 | 0.00599665 | 0.00596005 |
| 23 | 128 | 0.144603 | 0.0136619 | 0.0944786 | 0.0120754 | 0.00725453 | 0.00673274 |
| 24 | 128 | 0.0457272 | 0.00489287 | 0.107001 | 0.00474534 | 0.00254502 | 0.00398297 |
| 25 | 128 | 0.0561569 | 0.00626179 | 0.111505 | 0.00579327 | 0.00328685 | 0.00313665 |
| 26 | 128 | 0.141527 | 0.0120952 | 0.0854622 | 0.00787845 | 0.0039278 | 0.00330622 |
| 27 | 128 | 0.301059 | 0.0255913 | 0.0850044 | 0.0225775 | 0.0133159 | 0.00575714 |
| 28 | 128 | 0.193096 | 0.0118221 | 0.061224 | 0.0118381 | 0.00643327 | 0.00276827 |
| 29 | 128 | 0.130244 | 0.00968265 | 0.0743423 | 0.0110419 | 0.00513098 | 0.00387275 |
| 30 | 128 | 0.306631 | 0.0215599 | 0.0703121 | 0.0237479 | 0.0106021 | 0.00388548 |
| 31 | 128 | 3.79008 | 0.108846 | 0.0287187 | 0.151173 | 0.0581786 | 0.00740804 |

The companion JSON retains per-window initial/final parameter norms, exact movement for zero-initialized factors/B, A update path lengths, gradient/update magnitudes, occupancy, correction norms, sampled distribution shifts, and hidden drift.

Limitations:

- Local hard MSE uses the quantized prefix and is not an end-to-end quality score.
- Hard MSE and Huber training losses are different objectives; step losses also use shuffled samples and changing sharpness.
- A update path length is not its net displacement. Aggregate norms cannot recover a nonzero-initialized tensor's exact net movement.
- Weight distributions sample at most eight rows per module. Activation distributions sample at most 64 active rows of the first calibration sample under the original teacher prefix.
- Representative boundary statistics cover only the first active projection; correction norms cover all active projections.
- For rotated wrappers, boundary original_rms denotes the FP32 transformed source. Orthogonal rotation preserves RMS up to numerical error.
- Hidden-state drift uses the saved first NLL record. No checkpoint was read, no model was executed, and no held-out fitting was performed by this audit.
- Cross-arm activation summaries may use different first calibration records. Only within-arm original-versus-transformed summaries isolate rotation on those sampled inputs; do not read cross-arm differences as a transform effect.
