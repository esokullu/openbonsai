# Measurement-only v2 completed

The single corrected activation diagnostic restored the exact final128 FP32 research state, passed all five original/bypass/hard saved-probe comparisons and completed both fixed windows on the same 16 reused records. The final measurement seal and complete preservation checks finished inside the budget. No preparation, fitting, optimizer, rescaling, quality evaluation or successor run occurred.

| Result | Measured |
|---|---:|
| Later-window selector | 16/16 positives; requires 12 |
| Window 16 target-gap RMS / RMS(T0) | 0.761520 |
| Window 16 local error RMS / RMS(T0) | 0.064267 |
| Window 16 total error RMS / RMS(T0) | 0.760929 |
| V2 model-process wall time | 170.637 s |
| Cumulative activation time including v1 failure | 194.140 s / 2,700 s |
| Peak allocated / reserved | 19.497 / 19.600 GiB |
| Minimum headroom | 10.542 GiB |
| Separate post-exit CPU preservation audit | 20.263 s |

Window 0 target gap and cross term are exactly zero. At window 16, pooled target-gap RMS / total-error RMS is 1.000777, with a negative signed cross term. The ratio is not a causal share of error. Best-scalar rescaling would still leave 0.741553 relative RMSE and was not applied.

The result supports inspecting accumulated prefix drift. It does not isolate its causal origin or establish a quality remedy. The previous degenerate quality results remain unchanged. All 143 frozen v2 files, both prior seals, raw/consumed inputs and main-checkout modifications are preserved. The process exited and no successor is scheduled.

[Full report](D:/openbonsai/runs/2026-10-26-5090-activation-diagnostic-v2/report.md) · [Independent metrics](D:/openbonsai/runs/2026-10-26-5090-activation-diagnostic-v2/analysis/metric-interpretation.md) · [Preservation audit](D:/openbonsai/runs/2026-10-26-5090-activation-diagnostic-v2/analysis/v2-outcome-audit.json) · [Final inventory](D:/openbonsai/runs/2026-10-26-5090-activation-diagnostic-v2/final-validation.json)
