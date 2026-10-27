# Independent activation metric interpretation

Completed measurement and its seal verified. Each fixed window contains 16 records and 7,203 active token positions; repeats/native oracles are excluded from that denominator. The engineering selector has 16/16 positives (requires 12), evaluable=True, passes=True.

| Window | Pair A/B (B is reference) | Pooled RMS ratio | Pooled cosine | Pooled relative RMSE | Best nonnegative scalar | Residual relative RMSE |
|---|---|---:|---:|---:|---:|---:|
| 0 | xq/x0 | 1 | 1 | 0 | 1 | 0 |
| 0 | tq/t0 | 1 | 1 | 0 | 1 | 0 |
| 0 | y/tq | 0.97956218 | 0.98645411 | 0.16418217 | 1.0070357 | 0.16403745 |
| 0 | y/t0 | 0.97956218 | 0.98645411 | 0.16418217 | 1.0070357 | 0.16403745 |
| 16 | xq/x0 | 0.89781046 | 0.72405826 | 0.71128719 | 0.80647118 | 0.68973882 |
| 16 | tq/t0 | 0.84291545 | 0.67064511 | 0.76151967 | 0.7956256 | 0.74177836 |
| 16 | y/tq | 0.9983397 | 0.99708998 | 0.076243948 | 0.99874819 | 0.076233705 |
| 16 | y/t0 | 0.84151596 | 0.67089439 | 0.76092866 | 0.797245 | 0.74155291 |

Relative RMSE in each pair uses that pair’s reference RMS. In particular, Y/Tq is normalized by Tq, while Y/T0 and Tq/T0 use T0. Their displayed magnitudes do not share a denominator.
At window 16, conditional error normalized by T0 is 0.064267201; the displayed Y/Tq relative RMSE is 0.076243948.

| Window | Total error MSE | Conditional error MSE | Target-gap MSE | Signed cross mean | Gap RMS / total-error RMS |
|---|---:|---:|---:|---:|---:|
| 0 | 0.0003113106 | 0.0003113106 | 0 | 0 | 0 |
| 16 | 0.19657457 | 0.0014022266 | 0.19688004 | -0.0017077016 | 1.0007767 |

Window 0 is the negative control: original and compressed prefixes coincide, so the target gap and cross term are exactly zero. Window 16 describes prefix-conditioned differences at the fixed later window. The decomposition closes independently from the saved scalar sums; the signed cross term prevents interpreting component energies as causal shares. A gap/total RMS ratio near or above 1 is consistent with cancellation by the signed cross term, not proof that the gap causally explains 100% of the error.

All pooled pair/decomposition metrics were recomputed from valid-element-weighted per-record statistics. Best-scalar residuals describe amplitude versus directional mismatch; no scalar correction was applied. The JSON preserves every record and min/median/max distributions.

Measured supervisor runtime: 170.637 s. Peak allocated/reserved across outer phases and activation operations: 19.497/19.600 GiB. Minimum observed/conservative headroom: 10.542 GiB.

These are action-selection measurements on reused search-validation inputs, not held-out quality evidence. No training, rescaling or parameter changes occurred; passing an engineering selector does not establish that a future correction will restore useful generation.
