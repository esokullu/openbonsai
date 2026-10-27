# rotated16x8 training audit

Training audit: **PASS**. Observed 4096/4096 updates, 32/32 complete windows; stages {'soft': 3264, 'hard': 832}. Condition completion marker: True.

These are software and optimization measurements, not a quality gate.

| Window | Steps | Hard MSE before | Hard MSE after | Ratio | Soft Huber mean | Hard Huber mean | Correction/W (representative) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 128 | 0.0025295 | 0.000313001 | 0.123741 | 0.000271055 | 0.000167515 | 0.00435311 |
| 1 | 128 | 0.000237439 | 0.000123753 | 0.521199 | 7.55667e-05 | 6.39123e-05 | 3.24055e-05 |
| 2 | 128 | 0.00065238 | 0.000182735 | 0.280106 | 0.000127649 | 9.08995e-05 | 0.000378554 |
| 3 | 128 | 0.00185119 | 0.000182603 | 0.098641 | 0.000158707 | 9.68886e-05 | 0.00115383 |
| 4 | 128 | 0.000720589 | 0.000186546 | 0.25888 | 0.00016315 | 9.72778e-05 | 0.000666072 |
| 5 | 128 | 0.00129771 | 0.00021662 | 0.166925 | 0.000196055 | 0.000114449 | 0.00320513 |
| 6 | 128 | 0.00525572 | 0.000587204 | 0.111727 | 0.000631036 | 0.000300036 | 0.00457123 |
| 7 | 128 | 0.00851515 | 0.000968223 | 0.113706 | 0.00095021 | 0.000534359 | 0.000824177 |
| 8 | 128 | 0.00519964 | 0.000581272 | 0.111791 | 0.000618578 | 0.000304268 | 0.00354724 |
| 9 | 128 | 0.00540265 | 0.000628907 | 0.116407 | 0.000683558 | 0.00033228 | 0.00311228 |
| 10 | 128 | 0.00680347 | 0.000861226 | 0.126586 | 0.000871148 | 0.000440454 | 0.00206044 |
| 11 | 128 | 0.0100481 | 0.00119744 | 0.119171 | 0.00107739 | 0.000651652 | 0.00293829 |
| 12 | 128 | 0.00498124 | 0.000747125 | 0.149988 | 0.000752761 | 0.000397372 | 0.00235799 |
| 13 | 128 | 0.00707636 | 0.000935377 | 0.132183 | 0.000998707 | 0.000494104 | 0.00374349 |
| 14 | 128 | 0.0120887 | 0.00149199 | 0.123419 | 0.00152675 | 0.000784633 | 0.00361344 |
| 15 | 128 | 0.0215579 | 0.00222704 | 0.103305 | 0.00214955 | 0.00119478 | 0.00526279 |
| 16 | 128 | 0.0100777 | 0.00143132 | 0.142029 | 0.00143194 | 0.000757848 | 0.00294713 |
| 17 | 128 | 0.0129103 | 0.00178628 | 0.138361 | 0.00179363 | 0.000933693 | 0.00399155 |
| 18 | 128 | 0.0424179 | 0.00494994 | 0.116695 | 0.00498429 | 0.00247064 | 0.005316 |
| 19 | 128 | 0.0839894 | 0.00921368 | 0.1097 | 0.00852465 | 0.00455819 | 0.00813204 |
| 20 | 128 | 0.0332485 | 0.00485548 | 0.146036 | 0.00445886 | 0.00252939 | 0.00413785 |
| 21 | 128 | 0.0419441 | 0.00609893 | 0.145406 | 0.0056386 | 0.00318181 | 0.00423826 |
| 22 | 128 | 0.0678048 | 0.0118008 | 0.17404 | 0.0106971 | 0.00587849 | 0.00573188 |
| 23 | 128 | 0.140606 | 0.0140155 | 0.0996788 | 0.0123908 | 0.00746943 | 0.0071801 |
| 24 | 128 | 0.0440616 | 0.0052251 | 0.118586 | 0.00495072 | 0.00274302 | 0.00384523 |
| 25 | 128 | 0.0551496 | 0.00630097 | 0.114252 | 0.00599753 | 0.00329026 | 0.00318658 |
| 26 | 128 | 0.124324 | 0.0116959 | 0.0940757 | 0.0080503 | 0.00416264 | 0.00331577 |
| 27 | 128 | 0.293085 | 0.0269403 | 0.0919198 | 0.0238569 | 0.0143289 | 0.00664014 |
| 28 | 128 | 0.198953 | 0.0125856 | 0.0632593 | 0.0126871 | 0.00658077 | 0.00287952 |
| 29 | 128 | 0.138349 | 0.0111894 | 0.0808783 | 0.0127339 | 0.00603076 | 0.00394727 |
| 30 | 128 | 0.308775 | 0.0233595 | 0.0756521 | 0.0260894 | 0.0120539 | 0.00383934 |
| 31 | 128 | 3.97157 | 0.123352 | 0.0310588 | 0.177291 | 0.0617306 | 0.0079517 |

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

Independent saved-state check: **PASS**. All 1,000 parameter tensors across 200 projections are finite FP32 on CPU; all 200 sign tensors match initialization. The payload SHA256 matches its saved metadata. All 160 window/family aggregates match the final optimizer-step records (relative tolerance1e-10, absolute1e-12). Optimizer and gradient tensors are verified through the complete step-event audit because they are not saved in this research checkpoint. The real process exited0; this operation does not audit fresh reload or evaluation results. Prior provisional evidence remains unchanged.
