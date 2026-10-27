**Completed: one window-0 epsilon control, preliminary local evidence.** C512 (AdamW epsilon 1e-9) improved the final hard reconstruction MSE over authenticated B512 (1e-8): calibration by 27.1867% and search by 24.4950%. All 128 calibration and all 16 search records improved. The compact state reloaded exactly in a fresh original-model process. This is a first-layer result; it does not demonstrate restored language-model quality, lossless compression, or a deployable 2 GB model. No successor experiment was launched.

This corrected v2 attempt completed in **498.609 model-process seconds (8.310 minutes)**, including initialization, preflight, baseline replay, 512 fitting updates, all observations and metrics, saving, and fresh reload. The sealed failed v1 contributed zero model seconds, zero updates and zero observations. The family cap remains 1,800 seconds and 512 scientific plus two discarded updates; there was one corrected supervised attempt and no retry.

| State | Records | n | Hard MSE | Relative RMSE | Cosine | Output RMS | Target RMS |
|---|---|---|---|---|---|---|---|
| initial | calibration | 128 | 0.00249792472007 | 0.463974016 | 0.942535015 | 0.066850648 | 0.107719918 |
| initial | search | 16 | 0.00248926705469 | 0.464263748 | 0.942241362 | 0.066709015 | 0.107465972 |
| B | calibration | 128 | 0.000162908354158 | 0.118488354 | 0.992955715 | 0.106881799 | 0.107719918 |
| B | search | 16 | 0.000168010479147 | 0.120613859 | 0.992700582 | 0.106524013 | 0.107465972 |
| C | calibration | 128 | 0.000118619008005 | 0.101106978 | 0.994878416 | 0.106911414 | 0.107719918 |
| C | search | 16 | 0.000126856239105 | 0.104805710 | 0.994503372 | 0.106380535 | 0.107465972 |

Initial means the original seeded quantizer initialization evaluated in hard mode, not untouched BF16 output. All rows use the identical original first-layer target T0 and input/native kwargs. Calibration has **59,855 valid positions / 245,166,080 hidden elements**; search has **7,203 / 29,503,488**. Padding is excluded. MSE is pooled by valid hidden elements using bounded CPU FP64 reductions, rather than averaging record means. Relative RMSE is sqrt(total squared error / target squared norm). At window 0, cached input and conditional target equal the original input/target exactly.

C−B absolute MSE changes are −0.0000442893461529 on calibration and −0.0000411542400420 on search. Calibration per-record reductions span 22.8124–38.2550%; search reductions span 23.6573–25.7180%. Full per-record metrics and cache/data fingerprints for all 144 records are in [epsilon-results.json](D:/openbonsai/runs/2026-10-27-5090-window0-epsilon-control-v2/analysis/epsilon-results.json). The search records are disjoint from fitting but reused development evidence, not a fresh final confirmation set. The final state was frozen before search metrics; no checkpoint selection used search.

| Search ID prefix (full IDs in JSON) | B MSE | C MSE | Reduction vs B |
|---|---|---|---|
| f4d9c1d8988f | 0.00016138445 | 0.00012104524 | 24.9957% |
| b5400127dd7b | 0.000149015123 | 0.000113481228 | 23.8458% |
| 4c3a4d9c4bee | 0.000183474578 | 0.000139693282 | 23.8623% |
| b1fa1f22e9f7 | 0.00017248975 | 0.00012888707 | 25.2784% |
| ac7705b02cac | 0.000177090249 | 0.000131546266 | 25.7180% |
| 8f2626477def | 0.000143276864 | 0.000107041713 | 25.2903% |
| 53b03fd9ccfc | 0.000157731897 | 0.00012041674 | 23.6573% |
| a0313555b06d | 0.000166747457 | 0.00012526703 | 24.8762% |
| 96229087728c | 0.000149001428 | 0.000111614435 | 25.0917% |
| 85eed23277b5 | 0.000187022458 | 0.00013966085 | 25.3240% |
| 050b91f2111f | 0.000159876148 | 0.000121809553 | 23.8101% |
| 1f1cc714b368 | 0.00023364298 | 0.000177795216 | 23.9030% |
| b3ecd7445f54 | 0.000190831359 | 0.000144226098 | 24.4222% |
| 7c3532ec4bf9 | 0.000153622883 | 0.000115429349 | 24.8619% |
| 45cc6aac3d55 | 0.000168152499 | 0.000128140908 | 23.7948% |
| 92ed788cb828 | 0.000144824279 | 0.000109665046 | 24.2772% |

**Fixed scientific recipe and baseline eligibility.** Official Qwen/Qwen3.5-9B revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`, original BF16 checkpoint with the historical BF16-cast/native conditional wrapper policy. All 200 wrappers were constructed in the inherited seed/RNG order; only six window-0 projections trained. Original and unselected tensors stayed frozen and passed all preservation checks. The original model, tokenizer, chat template, native backend, rotations and data were inherited unchanged. B’s complete compact state and fixed native output replayed bitwise in the current preflight; pristine parameters, RNG, flags and snapshots were then restored exactly. C never warm-started from B.

C uses the same 128 fitting records, four fresh index lists shuffled by one evolving Random(5090), 512 updates (409 soft / 103 hard), batch 1, length 512, group 128, signed rotation block 4096, rank 64, sigmoid factor mapping, frozen-Jacobian hard stage, Huber delta 1 reduced over valid hidden elements, factor LR 0.005, LoRA LR 0.001, weight decay 0, clip threshold 1, and gradient checkpointing. Progress is s/512 and LR multiplier is 1−(s−1)/512. The only scientific intervention is epsilon 1e-9 applied to AdamW defaults and every group; baseline default remains 1e-8. All 512 completed steps have finite valid audits and matched optimizer-return observations. The two discarded timing steps used scheduled locations 409/410 but actual stored optimizer steps 1/2, followed by exact rollback.

**Calibration provenance and truncation.** No traces were generated in v2. The same UltraChat `HuggingFaceH4/ultrachat_200k`, revision `8049631c405ae6576f93f445c6b8166f76f5505a`, split `train_sft`, was reused. Generation was greedy with thinking disabled, max_new_tokens 512 and EOS IDs 248044/248046. Fitting traces contain 8,815 prompt and 58,089 generated tokens: 102 reached the generation cap, 105 were shortened for compression, 7,049 right-side tokens were dropped, and padding occupied 5,681 positions. Search traces contain 895 prompt and 7,023 generated tokens: 12 capped, 12 shortened, 715 dropped and 989 padded. Capping is recorded rather than assumed to be a completed answer. The consumed fitting/search SHA256s are `487a093fe330f35882ed04f969fa91e324b91630f5e17af003e02beebd23a699` / `443aaba5e5b41c641c27cdcd45f34fd5a2e770fa466321c9bbab84436ff40aaf`. The parent’s immutable 144-record, 1,228,577,670-byte cache was reused read-only after file/tensor authentication and fresh native recapture verification.

**Actual optimizer observations.** Four new scientific C observations at stored steps 1/409/410/512 are compared with four authenticated B observations. Each covers all 30 parameters / 10,092,544 coordinates, and preserves state, gradients, moments, snapshots, flags and RNG exactly. Two additional discarded preflight observations are timing evidence only. The published bounded CPU-FP64 observer is unchanged: commit `eab219ebfd0227a6069dca9e01c6ed996da7edc9`, SHA256 `eb2821b0ebc39e24ce211362c724cfe87d4a8834eb1e48c8f2b110552de3e942`.

| Arm | Stored step | mu retention | scale retention | threshold retention | LoRA A retention | LoRA B retention |
|---|---|---|---|---|---|---|
| B | 1 | 0.038575 | 0.007456 | 0.002058 | undefined | 0.024600 |
| B | 409 | 0.101642 | 0.074520 | 0.061339 | 0.118229 | 0.106455 |
| B | 410 | 0.094949 | 0.068074 | 0.055798 | 0.110310 | 0.101910 |
| B | 512 | 0.111745 | 0.081474 | 0.069595 | 0.150152 | 0.121357 |
| C | 1 | 0.172746 | 0.041251 | 0.009221 | undefined | 0.109835 |
| C | 409 | 0.378250 | 0.290980 | 0.257586 | 0.503065 | 0.539443 |
| C | 410 | 0.361232 | 0.270733 | 0.236672 | 0.487106 | 0.528882 |
| C | 512 | 0.380545 | 0.286356 | 0.255020 | 0.454715 | 0.531069 |

These are **direction-norm retention ratios**: sqrt(sum[mhat/(r+epsilon)]² / sum[mhat/r]²) on r>0. They exclude learning rate, weight decay and native rounding; they are not retained squared-energy fractions. Actual B and C moment trajectories differ, so this is not substitution of epsilon into identical moments. At step 1 all 2,097,152 LoRA-A coordinates have stored zero first/second moments and the comparator is undefined; the table preserves null rather than assigning zero or one. Parameter-level zero domains and group settings remain in the raw observations. Less damping and larger motion are observations, not standalone success criteria; clipping gradients does not cap Adam-normalized updates.

| Arm | Raw family | Final displacement D | Summed update path P | D/P | Hard-stage path |
|---|---|---|---|---|---|
| B | raw_mu | 38.821047 | 86.764740 | 0.447429 | 2.945063 |
| B | raw_scale | 38.491772 | 74.215808 | 0.518647 | 1.995029 |
| B | raw_threshold | 23.432214 | 54.034452 | 0.433653 | 1.851914 |
| C | raw_mu | 80.896910 | 209.564763 | 0.386023 | 8.755702 |
| C | raw_scale | 79.266109 | 172.644923 | 0.459128 | 6.308104 |
| C | raw_threshold | 53.767462 | 139.701522 | 0.384874 | 6.210344 |

D is displacement from the verified zero raw-factor initialization; P sums actual update norms. Hard-stage path is not hard-stage net displacement or a convergence estimate. Nonzero-origin LoRA A is excluded from this D/P table.

**Same-parameter boundary probe.** One fixed calibration record, at the true pre-update step-409 snapshot (sharpness 29.9560546875), with the same factors/A/B/input/target. The first hard step retained that snapshot and the observer did not refresh it. This is separate from pooled final metrics and losses on successive records.

| Arm | Soft MSE | Hard MSE | Hard−soft MSE | Hardening penalty |
|---|---|---|---|---|
| B | 0.000146533624332 | 0.000155808457488 | 9.27483315545e-06 | 6.329491% |
| C | 0.000103817293511 | 0.000115683670654 | 1.18663771434e-05 | 11.430058% |

C has lower absolute soft and hard errors on this probe, but its relative hardening penalty increased from 6.3295% to 11.4301%. The optimizer intervention did not eliminate the soft-to-hard gap.

| State | Negative | Zero | Positive | Hard BF16 weight RMS | Effective weight RMS | Source weight RMS | Aggregate ||BA/rank||F |
|---|---|---|---|---|---|---|---|
| initial | 34.562890% | 30.878192% | 34.558918% | 0.008581265 | 0.012946212 | 0.012946212 | 0.000000000 |
| B | 34.531690% | 30.949149% | 34.519161% | 0.008591830 | 0.012946577 | 0.012946212 | 1.116008168 |
| C | 34.412363% | 31.195802% | 34.391834% | 0.008657623 | 0.012953760 | 0.012946212 | 4.256517464 |

Weights cover six projections, 218,103,808 elements and 1,703,936 groups. Effective weights here are the inherited dense rotated source plus low-rank correction, before hard ternarization; they are not a deployed full-model accuracy measure. Full mapped-factor ranges and per-projection RMS/code counts are saved in the result JSON.

| Arm | Native mapped factor | Minimum across groups | Maximum across groups |
|---|---|---|---|
| B | mu | -0.0423578806 | 0.0371310078 |
| B | alpha | 0.00179952977 | 0.187580481 |
| B | threshold | 0.393593311 | 0.684629858 |
| C | mu | -0.0443114452 | 0.0317217521 |
| C | alpha | 0.00185437012 | 0.181017667 |
| C | threshold | 0.396269411 | 0.657591939 |

Separately, bounded FP32 CPU sigmoid(raw) diagnostics found **zero** near endpoints (≤0.01 or ≥0.99) and zero exact 0/1 values in all three raw-factor families for initial/B/C. C sigmoid ranges are mu 0.336446–0.650249, scale 0.311161–0.635969 and threshold 0.396269–0.657592. This diagnostic is distinct from native mapped factors and BA-dependent effective statistics; it does not establish that saturation cannot occur elsewhere.

**Runtime, admission and serialization.**

| Process | Measured seconds |
|---|---|
| preflight | 133.203 |
| fit | 265.656 |
| reload | 99.750 |
| Total | 498.609 |

Actual optimization (including its scheduled callbacks) took 127.214 seconds, about 4.025 updates/s. Final 128-record calibration metrics took 17.426 s and 16-record search metrics 2.225 s. Calibration generation cost was zero; immutable cache checks, loading, source hashing, observations, weight statistics, saving and reload are included in total model-process time. Measured complete-work admission projected 1326.952 s, using a 1.25 margin plus a separate 300 s reserve. All four actual standalone preservation calls were enumerated: B preflight, post-rollback preflight, final fit and fresh reload. The two remaining calls had explicit separate cost categories.

Peak CUDA allocated memory was **21,705,918,976 bytes (20.215212 GiB)**; peak reserved **21,781,020,672 bytes (20.285156 GiB)**. Conservative minimum headroom was **10,572,791,808 bytes (9.846680 GiB)**, above the 4 GiB gate. The prelaunch RTX 5090 had 32,050 MiB free, system free RAM 130,491,940,864 bytes and D: free disk 493,630,058,496 bytes. Recorded environment is Windows, Python 3.12.11, PyTorch 2.14.1+cu130, Transformers 5.20.2, NVIDIA driver 596.21. These are this native first-layer experiment's measurements, not inference-speed or deployable-size claims.

C's compact FP32 state is **54,047,955 bytes**, SHA256 `3ff5c9f1bd162185173051f8f7114e75546f71f995f86baab567746a98a24198`. It stores the six selected projections' factors, low-rank tensors, original statistics and signs; it requires the original checkpoint/runtime and is not a standalone compressed model. Fresh reload verified complete selected-state equality and bitwise fixed native output, plus original/unselected tensor preservation. All three owned model processes exited and were reaped. No full packed export, NLL, ARC, answer generation or full-model promotion was performed in this control.

**Execution correction and preserved lineage.** The failed epsilon-v1 source/final seal remains untouched. Its exact historical rejected PID was not recorded and remains unknown. The reviewed source could falsely match its own native Windows virtual-environment launcher while excluding only the interpreter PID. V2 adds one verified direct-launcher exception: actual parent-child relation, configured inspected executables, matching parsed/normalized script options and arguments, and creation relationship. Independent same-script peers still reject; missing/ambiguous inspection fails closed. Every matching process and exemption is persisted before the decision. Native tests demonstrated that unchanged kill/wait semantics exit both launcher and interpreter, so no cleanup repair was applied. A failed native development fixture exposed base-interpreter versus launcher argv behavior; its evidence was preserved and the narrow classifier correction then passed.

Validation: **22 new classifier/native checks + 56 focused supervisor checks passed**; 27 unaffected scientific/driver CPU tests were reused with exact source/test bindings. Nine saved-JSON collector tests passed. The 194-file v1 copy changed only the supervisor, provenance adapter, corresponding supervisor test and execution-lineage metadata; the classifier is a new operational helper. Scientific trainer, driver, quantizer, initialization, data and published observer are byte-identical to v1. Main remains at `891d260af96a7ce61ccbcb23474015d975bb18dc` with its 39-file dirty source boundary/status/diff preserved. All actually consumed ancestor references were revalidated; completed historical whole-archive audits were reused.

Scientific parent seal: `466a13bbb67aabede63b97019b014ffc639b4e4519a0bf28f3cdc206a5f5f1cf`. Failed v1 seal: `38b522a6900e532c27ae84328c922bbd0c3660bbf1940401ae2a48f80e777f5c`. V2 executed source freeze: **242 files**, SHA256 `d2facbb4905c8fbc7d9b55563d4768be5803940a2e0dec00b0dc94edb3435799`. [Commands](D:/openbonsai/runs/2026-10-27-5090-window0-epsilon-control-v2/COMMANDS.md), [fixed configuration](D:/openbonsai/runs/2026-10-27-5090-window0-epsilon-control-v2/experiment-plan.json), [supervision diff](D:/openbonsai/runs/2026-10-27-5090-window0-epsilon-control-v2/analysis/v1-to-v2-supervision.patch), [inherited epsilon trainer diff](D:/openbonsai/runs/2026-10-27-5090-window0-epsilon-control-v2/analysis/v1-evidence/epsilon-trainer.diff), [final preservation audit](D:/openbonsai/runs/2026-10-27-5090-window0-epsilon-control-v2/analysis/final-preservation.json).

The independent [numeric review](D:/openbonsai/runs/2026-10-27-5090-window0-epsilon-control-v2/analysis/epsilon-results-review.json), [training review](D:/openbonsai/runs/2026-10-27-5090-window0-epsilon-control-v2/analysis/training-results-review.json) and [source/execution review](D:/openbonsai/runs/2026-10-27-5090-window0-epsilon-control-v2/analysis/execution-review.json) accompany the final inventory/seal. Per-step logs, the actual observer records, timing phases, calibration traces and compact checkpoint remain in this directory. All historical failed artifacts and their quality results remain preserved. This local win supports the specific epsilon intervention in this fixed first-layer setting; it does not explain every source of the earlier full-model failure. Search reuse, one seed, truncation, a limited first-layer scope and a remaining hardening gap limit generalization. No additional experiment follows automatically.
