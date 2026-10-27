# rotated training audit

Training audit: **PASS**. Observed 4096/4096 updates, 32/32 complete windows; stages {'soft': 3264, 'hard': 832}. Condition completion marker: True.

These are software and optimization measurements, not a quality gate.

| Window | Steps | Hard MSE before | Hard MSE after | Ratio | Soft Huber mean | Hard Huber mean | Correction/W (representative) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 128 | 0.00224789 | 0.000226585 | 0.100799 | 0.000230414 | 0.000125179 | 0.00406463 |
| 1 | 128 | 0.000190801 | 0.000106671 | 0.559069 | 6.87489e-05 | 5.73098e-05 | 3.11241e-05 |
| 2 | 128 | 0.000501138 | 0.000135317 | 0.270019 | 0.000106095 | 7.61781e-05 | 0.00016252 |
| 3 | 128 | 0.00181063 | 0.000142209 | 0.0785413 | 0.0001162 | 8.00366e-05 | 0.00105112 |
| 4 | 128 | 0.000629497 | 0.000160947 | 0.255675 | 0.000148414 | 8.8411e-05 | 0.000515838 |
| 5 | 128 | 0.00115392 | 0.00017385 | 0.15066 | 0.000168936 | 9.47979e-05 | 0.00357829 |
| 6 | 128 | 0.00562207 | 0.000453365 | 0.0806402 | 0.000625395 | 0.000255753 | 0.00577145 |
| 7 | 128 | 0.0078093 | 0.000785075 | 0.100531 | 0.000882545 | 0.000459169 | 0.000622296 |
| 8 | 128 | 0.00433688 | 0.000534544 | 0.123255 | 0.000611913 | 0.000312442 | 0.00311472 |
| 9 | 128 | 0.0045742 | 0.000571605 | 0.124963 | 0.000682307 | 0.000326684 | 0.00278288 |
| 10 | 128 | 0.00664438 | 0.000840116 | 0.12644 | 0.000933906 | 0.000440063 | 0.00183289 |
| 11 | 128 | 0.0115986 | 0.00158949 | 0.137041 | 0.00162372 | 0.000861043 | 0.00304185 |
| 12 | 128 | 0.00589892 | 0.000869493 | 0.147399 | 0.00102252 | 0.00050111 | 0.0027887 |
| 13 | 128 | 0.00732305 | 0.00105615 | 0.144223 | 0.00122396 | 0.000591925 | 0.00352873 |
| 14 | 128 | 0.0135568 | 0.00167846 | 0.123809 | 0.00182796 | 0.000917357 | 0.00377166 |
| 15 | 128 | 0.0279453 | 0.00347185 | 0.124237 | 0.00331214 | 0.00181845 | 0.00681163 |
| 16 | 128 | 0.0127641 | 0.00201821 | 0.158116 | 0.00206876 | 0.00108029 | 0.00344817 |
| 17 | 128 | 0.0170632 | 0.00251011 | 0.147107 | 0.00259563 | 0.00132323 | 0.00406644 |
| 18 | 128 | 0.0488866 | 0.00629461 | 0.128759 | 0.00640148 | 0.00311759 | 0.00489478 |
| 19 | 128 | 0.104252 | 0.0118247 | 0.113424 | 0.0109367 | 0.00578461 | 0.00863573 |
| 20 | 128 | 0.0328393 | 0.00462742 | 0.140911 | 0.004665 | 0.00247983 | 0.00389952 |
| 21 | 128 | 0.0416016 | 0.00557581 | 0.134029 | 0.00567343 | 0.00296069 | 0.00381297 |
| 22 | 128 | 0.0672194 | 0.0102348 | 0.15226 | 0.00978501 | 0.00505394 | 0.00444806 |
| 23 | 128 | 0.153863 | 0.0156959 | 0.102012 | 0.0141543 | 0.00810034 | 0.00634448 |
| 24 | 128 | 0.0400188 | 0.00493218 | 0.123247 | 0.00491503 | 0.00262121 | 0.00318847 |
| 25 | 128 | 0.0439721 | 0.0057826 | 0.131506 | 0.00562544 | 0.00303264 | 0.00302463 |
| 26 | 128 | 0.14657 | 0.014503 | 0.0989494 | 0.00908035 | 0.00439354 | 0.00355919 |
| 27 | 128 | 0.349605 | 0.0398412 | 0.11396 | 0.0336513 | 0.0197217 | 0.00678152 |
| 28 | 128 | 0.200014 | 0.0155137 | 0.0775629 | 0.0161439 | 0.00825364 | 0.00255995 |
| 29 | 128 | 0.13598 | 0.0136453 | 0.100348 | 0.0152466 | 0.00713317 | 0.00356964 |
| 30 | 128 | 0.357685 | 0.0320363 | 0.0895658 | 0.0360216 | 0.0161595 | 0.0036839 |
| 31 | 128 | 3.95168 | 0.158898 | 0.0402101 | 0.21205 | 0.0795604 | 0.00753007 |

The companion JSON retains per-window initial/final parameter norms, exact movement for zero-initialized factors/B, A update path lengths, gradient/update magnitudes, occupancy, correction norms, sampled distribution shifts, and hidden drift.

Limitations:

- Local hard MSE uses the quantized prefix and is not an end-to-end quality score.
- Hard MSE and Huber training losses are different objectives; step losses also use shuffled samples and changing sharpness.
- A update path length is not its net displacement. Aggregate norms cannot recover a nonzero-initialized tensor's exact net movement.
- Weight distributions sample at most eight rows per module. Activation distributions sample at most 64 active rows of the first calibration sample under the original teacher prefix.
- Representative boundary statistics cover only the first active projection; correction norms cover all active projections.
- For rotated wrappers, boundary original_rms denotes the FP32 transformed source. Orthogonal rotation preserves RMS up to numerical error.
- Hidden-state drift uses the saved first NLL record. No checkpoint was read, no model was executed, and no held-out fitting was performed by this audit.
