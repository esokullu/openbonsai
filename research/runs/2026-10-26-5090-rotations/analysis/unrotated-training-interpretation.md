# Unrotated 16×8: completed audit and interpretation

The unrotated condition is complete: training, all three diagnostic evaluations, export and independent fresh reload finished. The refreshed training audit has `condition_complete=true` and still passes every check. Both matched conditions are now complete; paired findings are in `final-training-findings.md` and `distribution-comparison.md`. These findings concern this bounded experiment, not a general method verdict.

All 4,096 optimizer updates are present and contiguous across 32 windows/200 projections: 3,264 soft and 832 hard. Every update has all five parameter-family gradient tensors present, finite parameters/gradients/updates, FP32 parameters and Adam moments, and complete finite FP32 snapshots. Every hard step uses the last actual soft pre-update sharpness 29.8828125; snapshots clear after each window. The full report matches the per-step and per-window logs. Original-teacher bypass is bitwise exact before/after training.

Training took 731.303 s: optimization 586.979 s, caching 116.959 s, and hard evaluation before/after 26.197 s combined. CUDA peaks were 20.207 GiB allocated and 20.297 GiB reserved. The largest global gradient norm was 0.09042, so none of the updates reached the clipping threshold of 1.

Optimization was active. Each raw-factor family and B had nonzero gradients and updates on all 128 steps per window. A had nonzero gradients and updates on 127 steps, consistent with B starting at zero. All 200 final low-rank corrections have nonzero Frobenius norms. Median final raw-factor RMS across windows was 0.05426 for mu, 0.04785 for scale, and 0.03752 for threshold. The largest final absolute raw-scale value was 0.35920: under the implemented `2*sigmoid(raw)` mapping, all resulting scale multipliers lie within [0.82231, 1.17769]. These are bounds, not measured average multipliers or per-group alpha values. Exact net A displacement cannot be reconstructed from logged aggregate norms; the audit reports its update path length separately.

All 32 windows reduced local hard MSE. The median after/before ratio was 0.05389 (range 0.01169–0.77142). Huber losses are reported separately because they use a different objective, shuffled single samples, and a changing soft/hard forward. These strong local reductions did not preserve the original model's full hidden-state trajectory:

| Zero-based layer | Relative RMSE versus original | Cosine similarity |
|---:|---:|---:|
| 0 | 0.23456 | 0.97592 |
| 7 | 0.61276 | 0.79457 |
| 15 | 0.74865 | 0.69995 |
| 19 | 1.03546 | 0.39986 |
| 31 | 1.09071 | 0.29393 |

This probe covers the first NLL record. It directly shows large accumulated distortion despite improved reconstruction conditional on each already-quantized prefix; it does not isolate a single causal projection family.

The representative projection in each window retains a substantial amplitude mismatch: median hard/source RMS changes only from 0.65285 before training to 0.65349 after. Final ratios range 0.64381–0.65904. Its median correction/source Frobenius ratio is 0.00422 (maximum 0.00988); median zero-code fraction changes from 0.31476 to 0.31508. These are first-projection-per-window measurements, not statistics for all 200 matrices. Across sampled rows of all 200 matrices, effective/source RMS after/before remains near 1 (median 1.000019, range 0.999947–1.000698). Small corrections and modest factor movement therefore coexist with a persistent representative ternary amplitude mismatch. This is a plausible contributor, not proof that amplitude alone explains the failure.

The sampled original activation distributions are strongly heavy-tailed (median kurtosis 615.75), while sampled original weight kurtosis has median 3.253. Unrotated source/latent and original/runtime-transformed sample statistics are identical, as expected. These samples motivate the matched rotation comparison; they cannot establish its outcome.

All diagnostic stages completed on the same 2,048 NLL tokens, 20 ARC-Easy questions and four generation prompts:

| Stage | NLL | ARC correct |
|---|---:|---:|
| Hard FP32 learned state | 14.439949 | 1/20 |
| FP16 scales, same wrapper computation | 14.438327 | 1/20 |
| Packed artifact, fresh-process reload | 14.437115 | 1/20 |

All 200 exported projection payloads and 564 retained tensors verify exactly. Decoded-cache/uncached parity passes, and the independent fresh reload matches its saved packed-runtime logit reference bitwise (maximum error 0). This does not imply equality to the hard-FP32 wrappers: isolated FP16-scale casting has maximum probe error 0.125, and switching those wrappers to the packed row-chunk runtime adds a measured maximum difference of 0.0625. Their scopes remain distinct.

Severe degradation was already present before casting/export/packing, so serialization cannot be its original cause. The evidence also rejects missing gradients, skipped updates, nonfinite state, or a wrong recorded annealing schedule as explanations. It does not establish that a different budget, calibration set, objective, or the original research implementation would behave the same way.

Evidence: `unrotated-training-audit.json` contains the complete mechanical audit; `unrotated-training-interpretation.json` contains compact derived values and source paths. Only JSON/JSONL was read. No checkpoint, GPU execution, or frozen source file was touched.
