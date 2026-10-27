# Rotated 16×8: completed audit and interpretation

The rotated condition is complete and passes the JSON-only software audit: 4,096 contiguous updates over 32 windows/200 projections, comprising 3,264 soft and 832 hard updates. All required gradient tensors are present; parameters, gradients, updates, FP32 Adam moments and FP32 compact snapshots are finite. Every hard-stage snapshot retains the last actual soft pre-update sharpness 29.8828125, and snapshots clear after each window. Original-teacher bypass is bitwise exact before and after training. The refreshed audit has `condition_complete=true`; all diagnostic evaluations, export and independent fresh reload finished.

Raw factors and B have nonzero gradients/updates on all 128 steps per window; A has 127, as expected when B starts at zero. Median final raw-factor RMS is 0.05497 (mu), 0.04843 (scale), and 0.03813 (threshold), broadly similar in magnitude to unrotated. The largest global gradient norm is 0.14640, below clipping threshold 1; no step was clipped.

Timing and memory aggregation checks pass independently: each of eight trainer phases has exactly 32 window records; their elapsed-time sums and allocated/reserved maxima match the training report. The enclosing training peak equals the maximum recorded phase peak, rather than the last phase or a sum of peaks.

| Measured training quantity | Unrotated | Rotated |
|---|---:|---:|
| Total wall time | 731.303 s | 1309.048 s |
| Optimization | 586.979 s | 930.732 s |
| Window caching | 116.959 s | 324.059 s |
| Peak allocated | 20.207 GiB | 20.221 GiB |
| Peak reserved | 20.297 GiB | 20.297 GiB |

Rotated training took 1.790× the unrotated time. These synchronized measurements include audit overhead.

All 32 rotated windows reduce their local hard MSE; median after/before ratio is 0.12460 (range 0.04021–0.55907). Unrotated also reduces all 32, with median ratio 0.05389. These are not identical-target comparisons: each condition caches inputs from its own quantized prefix. Neither local reduction nor Huber step losses establish whole-model fidelity. Representative post-training hard/source RMS is 0.66419 median in rotated coordinates, versus 0.65349 within unrotated coordinates; this remains a first-projection-per-window observation, not a global weight norm. Rotated representative correction/source Frobenius ratio is 0.00356 median.

On the **same first NLL record**, with identical original-output RMS references, the rotated hard model has lower relative RMSE and higher cosine than unrotated at all 32 saved block outputs:

| Zero-based layer | Relative RMSE U / R | Cosine U / R |
|---:|---:|---:|
| 0 | 0.23456 / 0.18862 | 0.97592 / 0.98305 |
| 7 | 0.61276 / 0.59351 | 0.79457 / 0.80581 |
| 15 | 0.74865 / 0.70011 | 0.69995 / 0.73024 |
| 19 | 1.03546 / 0.94921 | 0.39986 / 0.49824 |
| 23 | 0.98537 / 0.90555 | 0.45965 / 0.52762 |
| 31 | 1.09071 / 1.07104 | 0.29393 / 0.38213 |

This is numerical improvement on a single saved hidden-state probe, with substantial remaining final-layer distortion. It is not projection-family attribution or a claim of useful generation.

All diagnostic stages completed on the same 2,048 scored tokens, 20 ARC questions and four generation prompts:

| Stage | NLL | ARC correct |
|---|---:|---:|
| Hard FP32 learned state | 9.257334 | 8/20 |
| FP16 scales, same wrapper computation | 9.253318 | 8/20 |
| Packed artifact, fresh-process reload | 9.255346 | 8/20 |

All four generation samples at every stage reached the 128-token cap without EOS. Completion/stopping metadata and numerical score improvements alone do not establish useful text; the separate manual generation review supplies that assessment.

All 200 projection payloads and 564 retained tensors verify exactly. Decoded-cache/uncached parity passes, and fresh-process logits match the saved packed-runtime reference bitwise, with maximum error 0. This is distinct from equality to hard-FP32 wrappers: isolated FP16-scale casting and the additional packed row-chunk computation each have measured maximum probe difference 0.125. Both final training audits and the strictly matched final distribution comparison are now complete.

Evidence and exact derived values are in `rotated-training-audit.json`, `rotated-training-interpretation.json`, `final-training-findings.json` and `distribution-comparison.json`, including all phase-aggregation and paired source/sample-matching checks. Only JSON/JSONL was read. No GPU execution, checkpoint reads, frozen source edits or monitoring were performed.
