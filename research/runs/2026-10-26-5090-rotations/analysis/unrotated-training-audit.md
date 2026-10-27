# unrotated training audit

Training audit: **PASS**. Observed 4096/4096 updates, 32/32 complete windows; stages {'soft': 3264, 'hard': 832}. Condition completion marker: True.

These are software and optimization measurements, not a quality gate.

| Window | Steps | Hard MSE before | Hard MSE after | Ratio | Soft Huber mean | Hard Huber mean | Correction/W (representative) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 128 | 0.00472271 | 0.000263096 | 0.0557087 | 0.000353763 | 0.000139252 | 0.00988226 |
| 1 | 128 | 0.00020268 | 0.000156352 | 0.771424 | 9.00957e-05 | 8.3606e-05 | 2.15193e-05 |
| 2 | 128 | 0.0010194 | 0.000312477 | 0.306529 | 0.000252482 | 0.000156166 | 0.000233512 |
| 3 | 128 | 0.0023182 | 0.00034867 | 0.150406 | 0.000309104 | 0.000193945 | 0.000498645 |
| 4 | 128 | 0.00100632 | 0.000273952 | 0.272232 | 0.000231192 | 0.00014867 | 0.00031016 |
| 5 | 128 | 0.00225714 | 0.000313067 | 0.1387 | 0.000310287 | 0.000167517 | 0.00187823 |
| 6 | 128 | 0.0076736 | 0.00277872 | 0.362115 | 0.00142582 | 0.000924049 | 0.00416131 |
| 7 | 128 | 0.0045631 | 0.00048697 | 0.106719 | 0.000727637 | 0.00031597 | 0.00191057 |
| 8 | 128 | 0.00315719 | 0.000404797 | 0.128214 | 0.000578425 | 0.000260582 | 0.002691 |
| 9 | 128 | 0.00431046 | 0.000476801 | 0.110615 | 0.000654534 | 0.000281908 | 0.00299066 |
| 10 | 128 | 0.00703549 | 0.00137616 | 0.195603 | 0.000889178 | 0.000346367 | 0.00293527 |
| 11 | 128 | 0.0112691 | 0.00064925 | 0.0576134 | 0.000997902 | 0.000370827 | 0.00278917 |
| 12 | 128 | 0.00903636 | 0.00058028 | 0.0642161 | 0.0009653 | 0.000365391 | 0.0039096 |
| 13 | 128 | 0.0123129 | 0.000731283 | 0.0593914 | 0.00123262 | 0.000441508 | 0.00459857 |
| 14 | 128 | 0.0165017 | 0.000978996 | 0.0593271 | 0.00171048 | 0.000598129 | 0.00457057 |
| 15 | 128 | 0.0353719 | 0.00108841 | 0.0307704 | 0.0020447 | 0.000696544 | 0.00355779 |
| 16 | 128 | 0.0150304 | 0.000782703 | 0.0520746 | 0.00141621 | 0.000510262 | 0.00407247 |
| 17 | 128 | 0.0176748 | 0.000833481 | 0.0471564 | 0.00159011 | 0.000510055 | 0.00450806 |
| 18 | 128 | 0.070096 | 0.00290467 | 0.0414385 | 0.00552067 | 0.00148866 | 0.00658295 |
| 19 | 128 | 0.124114 | 0.00481733 | 0.0388138 | 0.00731662 | 0.00190943 | 0.00555574 |
| 20 | 128 | 0.0379847 | 0.00100799 | 0.0265367 | 0.00228531 | 0.000633639 | 0.00411768 |
| 21 | 128 | 0.0397773 | 0.00100637 | 0.0253001 | 0.00242818 | 0.000602993 | 0.00469998 |
| 22 | 128 | 0.0596776 | 0.00294473 | 0.0493439 | 0.00427256 | 0.00106258 | 0.00567412 |
| 23 | 128 | 0.200654 | 0.00256532 | 0.0127848 | 0.00771939 | 0.00158148 | 0.00698681 |
| 24 | 128 | 0.0357841 | 0.000877645 | 0.0245261 | 0.00188971 | 0.000450787 | 0.00428107 |
| 25 | 128 | 0.0399615 | 0.00160397 | 0.040138 | 0.00185944 | 0.000472931 | 0.00406854 |
| 26 | 128 | 0.121501 | 0.0108001 | 0.0888892 | 0.00550781 | 0.00157304 | 0.00550401 |
| 27 | 128 | 0.648832 | 0.0137269 | 0.0211563 | 0.0270649 | 0.00539653 | 0.00716708 |
| 28 | 128 | 0.184519 | 0.00238047 | 0.0129009 | 0.0068567 | 0.00104245 | 0.00440677 |
| 29 | 128 | 0.162051 | 0.0018938 | 0.0116864 | 0.00712868 | 0.00108332 | 0.0048731 |
| 30 | 128 | 0.375725 | 0.00630719 | 0.0167867 | 0.0170786 | 0.00294723 | 0.00523273 |
| 31 | 128 | 2.22291 | 0.0322676 | 0.0145159 | 0.0813571 | 0.0133732 | 0.00505733 |

The companion JSON retains per-window initial/final parameter norms, exact movement for zero-initialized factors/B, A update path lengths, gradient/update magnitudes, occupancy, correction norms, sampled distribution shifts, and hidden drift.

Limitations:

- Local hard MSE uses the quantized prefix and is not an end-to-end quality score.
- Hard MSE and Huber training losses are different objectives; step losses also use shuffled samples and changing sharpness.
- A update path length is not its net displacement. Aggregate norms cannot recover a nonzero-initialized tensor's exact net movement.
- Weight distributions sample at most eight rows per module. Activation distributions sample at most 64 active rows of the first calibration sample under the original teacher prefix.
- Representative boundary statistics cover only the first active projection; correction norms cover all active projections.
- For rotated wrappers, boundary original_rms denotes the FP32 transformed source. Orthogonal rotation preserves RMS up to numerical error.
- Hidden-state drift uses the saved first NLL record. No checkpoint was read, no model was executed, and no held-out fitting was performed by this audit.
