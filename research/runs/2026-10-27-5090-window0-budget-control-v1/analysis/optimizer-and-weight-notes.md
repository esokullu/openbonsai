**Optimizer observations and mapped-weight appendix — window-zero A128/B512**

This appendix uses saved JSON only. It adds no forward pass, tensor load, GPU work, optimizer update, or parameter change. A completed 128 updates (102 soft, 26 hard); B completed 512 (409 soft, 103 hard). All 640 scientific updates passed the finite/gradient/moment checks. The four discarded preflight updates remain separate. Both endpoints were frozen before comparative final metrics. The results concern one block, repeated exposure and annealing duration together; they do not establish language-quality recovery or an optimizer cause of earlier quality failures.

The full source-bound result is [budget-results.json](D:/openbonsai/runs/2026-10-27-5090-window0-budget-control-v1/analysis/budget-results.json); the compact arithmetic review is [budget-results-review.json](D:/openbonsai/runs/2026-10-27-5090-window0-budget-control-v1/analysis/budget-results-review.json). Every per-parameter observer field, optimizer group, coordinate-domain count and actual stored step is retained there and in the eight individual JSONs linked below.

**All eight after-update moment observations**

Each observation covers all 30 unique selected parameters (10,092,544 coordinates): six tensors per family, with 1,703,936 coordinates in each raw-factor family, 2,097,152 in LoRA A, and 2,883,584 in LoRA B. Every parameter has initialized FP32 moments; before/after value, identity, gradient, optimizer, snapshot, flag and RNG checks were exact. Each recorded stored step equals the actual completed step, not the upcoming step. The observer reads at most 65,536 coordinates per CPU reduction chunk.

Let m-hat and v-hat denote bias-corrected stored moments at that step, and r = sqrt(v-hat). Both optimizer groups use epsilon = 1e-8 and betas (0.9, 0.999). “Below 1” is the fraction of observed coordinates with r/epsilon < 1, including stored zeros. Direction-energy attenuation is sqrt(sum over r>0 of [m-hat/(r+epsilon)]² / sum over r>0 of [m-hat/r]²). Its denominator and domain are explicit; it excludes learning rate, weight decay and native update rounding. A null ratio is not a zero attenuation ratio.

| Arm / completed step | Family | RMS(r/epsilon) | Below 1 (%) | Direction-energy attenuation | Positive-r comparator coordinates | Stored m=v=0 coordinates |
|---|---|---:|---:|---:|---:|---:|
| A / 1 | raw_mu | 0.81679831 | 98.5046387 | 0.121670019 | 1,703,936 | 0 |
| A / 1 | raw_scale | 1.70421438 | 97.6696895 | 0.124663412 | 1,703,936 | 0 |
| A / 1 | raw_threshold | 0.255068589 | 99.9259949 | 0.0433556885 | 1,703,936 | 0 |
| A / 1 | lora_A | 0 | 100 | null | 0 | 2,097,152 |
| A / 1 | lora_B | 0.221390875 | 99.0603707 | 0.0894016403 | 2,883,584 | 0 |
| A / 102 | raw_mu | 1.16564216 | 95.4369178 | 0.138610738 | 1,703,936 | 0 |
| A / 102 | raw_scale | 1.79453497 | 96.3933505 | 0.103270119 | 1,703,936 | 0 |
| A / 102 | raw_threshold | 0.677532509 | 97.9265066 | 0.0869497819 | 1,703,936 | 0 |
| A / 102 | lora_A | 1.21673693 | 88.1612301 | 0.0764080607 | 2,097,152 | 0 |
| A / 102 | lora_B | 2.07811402 | 90.7817147 | 0.0798952565 | 2,883,584 | 0 |
| A / 103 | raw_mu | 1.17218456 | 95.3280522 | 0.126955785 | 1,703,936 | 0 |
| A / 103 | raw_scale | 1.78879222 | 96.3395339 | 0.0945407163 | 1,703,936 | 0 |
| A / 103 | raw_threshold | 0.679924808 | 97.856551 | 0.0781328014 | 1,703,936 | 0 |
| A / 103 | lora_A | 1.21856329 | 88.1411552 | 0.0664785044 | 2,097,152 | 0 |
| A / 103 | lora_B | 2.09089433 | 90.7757152 | 0.0664208897 | 2,883,584 | 0 |
| A / 128 | raw_mu | 1.14433141 | 94.8925899 | 0.146674117 | 1,703,936 | 0 |
| A / 128 | raw_scale | 1.62326171 | 96.2709867 | 0.108525917 | 1,703,936 | 0 |
| A / 128 | raw_threshold | 0.65142331 | 97.6446298 | 0.0933484164 | 1,703,936 | 0 |
| A / 128 | lora_A | 1.13557629 | 88.4846687 | 0.0897382399 | 2,097,152 | 0 |
| A / 128 | lora_B | 2.0058914 | 90.7625719 | 0.0908206799 | 2,883,584 | 0 |
| B / 1 | raw_mu | 0.609670619 | 99.95886 | 0.0385753305 | 1,703,936 | 0 |
| B / 1 | raw_scale | 0.0209592015 | 99.9972417 | 0.00745630299 | 1,703,934 | 2 |
| B / 1 | raw_threshold | 0.0031471075 | 100 | 0.00205824272 | 1,703,930 | 6 |
| B / 1 | lora_A | 0 | 100 | null | 0 | 2,097,152 |
| B / 1 | lora_B | 0.0582214116 | 99.9932376 | 0.024600388 | 2,883,584 | 0 |
| B / 409 | raw_mu | 0.955376132 | 95.5115098 | 0.101641767 | 1,703,936 | 0 |
| B / 409 | raw_scale | 0.851562395 | 96.7675429 | 0.0745199345 | 1,703,936 | 0 |
| B / 409 | raw_threshold | 0.499558114 | 97.7668175 | 0.0613389556 | 1,703,936 | 0 |
| B / 409 | lora_A | 1.42330288 | 85.5401993 | 0.118228549 | 2,097,152 | 0 |
| B / 409 | lora_B | 2.50604494 | 89.4119609 | 0.106455031 | 2,883,584 | 0 |
| B / 410 | raw_mu | 0.955642993 | 95.5076951 | 0.0949485513 | 1,703,936 | 0 |
| B / 410 | raw_scale | 0.850974033 | 96.767719 | 0.0680744428 | 1,703,936 | 0 |
| B / 410 | raw_threshold | 0.499600532 | 97.7659959 | 0.0557978441 | 1,703,936 | 0 |
| B / 410 | lora_A | 1.42204523 | 85.5453968 | 0.110309704 | 2,097,152 | 0 |
| B / 410 | lora_B | 2.50355509 | 89.4124118 | 0.101910144 | 2,883,584 | 0 |
| B / 512 | raw_mu | 0.896445665 | 95.6443787 | 0.111745427 | 1,703,936 | 0 |
| B / 512 | raw_scale | 0.769607716 | 97.0167307 | 0.0814738486 | 1,703,936 | 0 |
| B / 512 | raw_threshold | 0.46569197 | 97.9208139 | 0.0695949124 | 1,703,936 | 0 |
| B / 512 | lora_A | 1.27600359 | 86.4682674 | 0.150152483 | 2,097,152 | 0 |
| B / 512 | lora_B | 2.22959759 | 89.5832062 | 0.121356576 | 2,883,584 | 0 |

At step 1, LoRA A has zero stored first and second moments across all 2,097,152 coordinates in both arms. Its positive-r domain is empty and undamped energy is zero, so its attenuation ratio is null. B step 1 also has two stored-zero coordinates in raw_scale and six in raw_threshold. These are excluded from the positive-r comparator. No observation has stored nonzero first moment with zero second moment. Stored zeros alone do not establish a history of zero gradients, and they must not be relabeled as evidence of epsilon domination. Many nonzero coordinates have r below epsilon, but that descriptive fact does not establish a quality cause or justify changing epsilon.

A and B step 1 need not have identical moments: the same initialization and example encounter different progress values (1/128 versus 1/512). The arm comparison changes repeated exposure and the duration of annealing together. It is not a counterfactual test of epsilon, a comparison at identical sharpness, or a data-diversity-only experiment.

| Observation | Actual factor / LoRA learning rates | Observer synchronized time (s) | Full per-parameter evidence |
|---|---:|---:|---|
| A 1 | 0.005 / 0.001 | 0.9436508 | [JSON](D:/openbonsai/runs/2026-10-27-5090-window0-budget-control-v1/training/observer-A-1.json) |
| A 102 | 0.0010546875 / 0.0002109375 | 1.1156724 | [JSON](D:/openbonsai/runs/2026-10-27-5090-window0-budget-control-v1/training/observer-A-102.json) |
| A 103 | 0.001015625 / 0.000203125 | 0.9573553 | [JSON](D:/openbonsai/runs/2026-10-27-5090-window0-budget-control-v1/training/observer-A-103.json) |
| A 128 | 3.90625e-05 / 7.8125e-06 | 0.9478122 | [JSON](D:/openbonsai/runs/2026-10-27-5090-window0-budget-control-v1/training/observer-A-128.json) |
| B 1 | 0.005 / 0.001 | 1.0114213 | [JSON](D:/openbonsai/runs/2026-10-27-5090-window0-budget-control-v1/training/observer-B-1.json) |
| B 409 | 0.001015625 / 0.000203125 | 1.0023736 | [JSON](D:/openbonsai/runs/2026-10-27-5090-window0-budget-control-v1/training/observer-B-409.json) |
| B 410 | 0.00100585938 / 0.000201171875 | 0.9624226 | [JSON](D:/openbonsai/runs/2026-10-27-5090-window0-budget-control-v1/training/observer-B-410.json) |
| B 512 | 9.765625e-06 / 1.953125e-06 | 0.9384684 | [JSON](D:/openbonsai/runs/2026-10-27-5090-window0-budget-control-v1/training/observer-B-512.json) |

These observer timers are nested inside measured outer phases. They must not be added again to the 614.171-second aggregate model-process time. The normalized direction statistics above do not include the listed learning rates.

**Raw-factor displacement and path length**

All three raw-factor families begin at exactly zero. D is the L2 norm of the concatenated final raw-factor vector, therefore its displacement from that verified origin. P is the sum of the L2 norm of each actual completed update, aggregated over the same six tensors and 1,703,936 coordinates per family. Soft and hard P partition the path by the actual stage; they are not phase net displacement. All six D ≤ P checks hold. LoRA A has a nonzero initialized origin, so its final norm would not equal displacement and is deliberately excluded from this table. LoRA factors still appear in the moment observations and effective correction measurements.

| Arm | Raw family | D | Soft P | Hard P | Total P | D/P |
|---|---|---:|---:|---:|---:|---:|
| A | raw_mu | 25.0998573 | 40.476345 | 1.53057692 | 42.0069219 | 0.597517174 |
| A | raw_scale | 23.2126166 | 35.5339814 | 1.0543495 | 36.5883309 | 0.63442677 |
| A | raw_threshold | 16.2043817 | 26.8159643 | 0.99435351 | 27.8103178 | 0.582675174 |
| B | raw_mu | 38.8210471 | 83.8196764 | 2.94506326 | 86.7647397 | 0.447428843 |
| B | raw_scale | 38.4917721 | 72.2207795 | 1.9950289 | 74.2158084 | 0.518646538 |
| B | raw_threshold | 23.4322137 | 52.1825374 | 1.85191421 | 54.0344516 | 0.433653216 |

D/P describes cancellation along the saved path; it does not establish convergence, late-stage usefulness, or an epsilon mechanism. All path values come from completed update events, not intentions or a reconstructed gradient sequence.

**Mapped weights, ternary occupancy and effective rank-64 corrections**

Coverage is complete for the six selected window-zero projections: 218,103,808 valid weight elements and 1,703,936 groups of 128, at the original initializer and both endpoints. Padded coordinates are excluded from code occupancy. Actual mapped mu, alpha and threshold are computed from effective rotated weights; these are distinct from raw_mu/raw_scale/raw_threshold offsets. Source-original and rotated-source coordinates are distinguished. The effective correction norm is ||BA/64||F, computed through the inherited FP32 rank-squared Gram identity without forming a full BA matrix. Each saved row also retains mapped-factor minima, maxima, means and RMS, all code counts, all projection RMS values and bounded-reduction scopes.

| State | Zero code fraction (%) | Original-source RMS | Rotated-source RMS | Effective RMS | Hard FP32 RMS | Hard BF16-compute RMS | Combined ||BA/64||F |
|---|---:|---:|---:|---:|---:|---:|---:|
| initial | 30.8781917 | 0.0129462122 | 0.0129462122 | 0.0129462122 | 0.00858129412 | 0.008581265 | 0 |
| A | 30.8986072 | 0.0129462122 | 0.0129462122 | 0.0129462469 | 0.00858413479 | 0.0085841125 | 0.280803314 |
| B | 30.9491492 | 0.0129462122 | 0.0129462122 | 0.0129465774 | 0.00859181936 | 0.00859182978 | 1.11600817 |

Aggregate RMS is weighted by actual weight-element counts; the combined correction norm is the square root of the sum of the six projection norm squares. These are weight statistics, not activation-output RMS or a serialized-model size measurement. Hard FP32 and BF16 computation columns are separate precision representations.

In the following compact table, every triple is **initializer / A / B**; “alpha” is the mapped ternary scale. Full precision and ranges remain in the linked JSON.

| Projection (under model.language_model.layers.0) | Mapped mu RMS | Mapped alpha mean | Mapped threshold mean | Zero codes (%) | ||BA/64||F |
|---|---:|---:|---:|---:|---:|
| linear_attn.out_proj | 0.00136326779 / 0.00138792371 / 0.0014035879 | 0.011843959 / 0.0118535317 / 0.0118906282 | 0.5 / 0.500343174 / 0.500631038 | 30.8774412 / 30.9024036 / 31.0092509 | 0 / 0.272428453 / 0.45898357 |
| linear_attn.in_proj_qkv | 0.00151776661 / 0.00153858199 / 0.00159146314 | 0.0132739913 / 0.0132738923 / 0.0132837432 | 0.5 / 0.499944567 / 0.499809258 | 30.8794707 / 30.9006274 / 30.9444249 | 0 / 0.0176440086 / 0.741035104 |
| linear_attn.in_proj_z | 0.00148396163 / 0.00148461587 / 0.00148099132 | 0.013268181 / 0.0133110959 / 0.0133974033 | 0.5 / 0.499687488 / 0.499260887 | 30.8677852 / 30.9384763 / 31.0775936 | 0 / 0.045183979 / 0.258337259 |
| mlp.gate_proj | 0.000990360163 / 0.000991613538 / 0.000991605136 | 0.0089002906 / 0.00890675255 / 0.0089193935 | 0.5 / 0.499939676 / 0.499853019 | 30.8819095 / 30.8983783 / 30.934149 | 0 / 0.013482552 / 0.176120535 |
| mlp.up_proj | 0.000926834648 / 0.00092840095 / 0.000928021529 | 0.00837792754 / 0.00838374137 / 0.00839438153 | 0.5 / 0.499937912 / 0.499878291 | 30.8753451 / 30.8891733 / 30.921032 | 0 / 0.0452642255 / 0.567715526 |
| mlp.down_proj | 0.000967891365 / 0.000969717005 / 0.000969737049 | 0.00868789489 / 0.00869246884 / 0.00870282557 | 0.5 / 0.499976588 / 0.500058164 | 30.880187 / 30.8923682 / 30.9325675 | 0 / 0.00705031212 / 0.256159693 |

Weight records: [initializer](D:/openbonsai/runs/2026-10-27-5090-window0-budget-control-v1/preflight/initial-weight-metrics.json), [A](D:/openbonsai/runs/2026-10-27-5090-window0-budget-control-v1/training/A-final-weight-metrics.json), [B](D:/openbonsai/runs/2026-10-27-5090-window0-budget-control-v1/training/B-final-weight-metrics.json). Corrections and occupancy changed, but their size does not apportion causes of the reconstruction improvement.

**Actual-snapshot soft/hard boundary probe and outcome scope**

The same-parameter probe runs after the true last-soft training forward/backward and before its optimizer update. It keeps that actual Jacobian snapshot, current factors, gradients and RNG fixed, evaluates soft versus hard without gradient recording, and verifies the unchanged snapshot again at the first hard step. It measures one fixed calibration record; it does not substitute for final all-record hard metrics.

| Arm | Actual last-soft step / sharpness | Soft MSE versus T0 | Same-parameter hard MSE versus T0 | Increase (%) | Hard-vs-soft relative RMSE (soft-output denominator) |
|---|---:|---:|---:|---:|---:|
| A | 102 / 29.8828125 | 0.000289871386 | 0.000301003313 | 3.84029851 | 0.0369101472 |
| B | 409 / 29.9560547 | 0.000146533624 | 0.000155808457 | 6.3294914 | 0.031403683 |

The saved FP32 Huber objective is reported separately from these FP64 MSE statistics; it is not renamed MSE. The hard-vs-soft relative RMSE uses soft output as its reference, while y/T0 uses the common original target. Across the full final populations, B reduces pooled MSE versus A by 47.2992% on 128 calibration records and 46.0312% on 16 search records; all records improve. Search reductions range from 42.0612% to 50.1417%. This supports the measured first-block reuse/annealing-budget result only. No NLL, benchmark answers, generation quality, later-block training, optimizer intervention or full deployable artifact was evaluated here.

**Exact saved-evidence bindings**

| Evidence | Bytes | SHA-256 |
|---|---:|---|
| [Result JSON](D:/openbonsai/runs/2026-10-27-5090-window0-budget-control-v1/analysis/budget-results.json) | 3316093 | `1d69128886ab4375e8f386bc96d3b001f03fafe35c8c95fe6d6a8ef0966f4c83` |
| [Numeric review JSON](D:/openbonsai/runs/2026-10-27-5090-window0-budget-control-v1/analysis/budget-results-review.json) | 78469 | `0af63e7bfd9ed7dee88a4804ff26a4b52a14e5bed195fce1481041a7791e4d86` |
| [runtime-source-freeze.json](D:/openbonsai/runs/2026-10-27-5090-window0-budget-control-v1/runtime-source-freeze.json) | 29593 | `5f40a5af2f441c4813d7d99f5e776f1b37fd13adab3b8c44895b1f0320df2604` |
| [checkout/openbonsai/optimizer_diagnostics.py](D:/openbonsai/runs/2026-10-27-5090-window0-budget-control-v1/checkout/openbonsai/optimizer_diagnostics.py) | 15842 | `eb2821b0ebc39e24ce211362c724cfe87d4a8834eb1e48c8f2b110552de3e942` |
| [tools/budget_observer.py](D:/openbonsai/runs/2026-10-27-5090-window0-budget-control-v1/tools/budget_observer.py) | 13720 | `b833eb8c311419cbb7c1e4233af6227d66c5727a34b565cea83fe33de7b98283` |
| [tools/budget_metrics.py](D:/openbonsai/runs/2026-10-27-5090-window0-budget-control-v1/tools/budget_metrics.py) | 18998 | `a1e821170ac772a90f32fe04440610835d4865ec5eea014035cb4a26bbcb323d` |
| [A completed 1](D:/openbonsai/runs/2026-10-27-5090-window0-budget-control-v1/training/observer-A-1.json) | 326756 | `19e21e77966c8632200f25560cb40ad93d3802dede165ca4395c1782b405c4c1` |
| [A completed 102](D:/openbonsai/runs/2026-10-27-5090-window0-budget-control-v1/training/observer-A-102.json) | 328690 | `2cc953a0d059a118acaca6143bb4c8b3453ce09abc1be7a9e18812fd20c822d2` |
| [A completed 103](D:/openbonsai/runs/2026-10-27-5090-window0-budget-control-v1/training/observer-A-103.json) | 328697 | `c93e98489a9701f04ae25d3908c605f891d55aacddab9705d2133a33b5ce883c` |
| [A completed 128](D:/openbonsai/runs/2026-10-27-5090-window0-budget-control-v1/training/observer-A-128.json) | 328688 | `d46d7f8f1e0cddae0efbaf0f291cd50811aec27fb8665693c71e92712b64834b` |
| [B completed 1](D:/openbonsai/runs/2026-10-27-5090-window0-budget-control-v1/training/observer-B-1.json) | 326368 | `e592aaa78c89e6b64212f5fd08f7b115a6f850df7ecc45b9864e5a22b4abd4f8` |
| [B completed 409](D:/openbonsai/runs/2026-10-27-5090-window0-budget-control-v1/training/observer-B-409.json) | 328287 | `9f0a0cbb7380aa2288abad612a8a8de4c883dbfbc063fb5d87ea55b9b49ba655` |
| [B completed 410](D:/openbonsai/runs/2026-10-27-5090-window0-budget-control-v1/training/observer-B-410.json) | 328319 | `81ed1a56c37333cc6c69740fb3849bb5acdc279861cb653c3c9eabb8a52180a0` |
| [B completed 512](D:/openbonsai/runs/2026-10-27-5090-window0-budget-control-v1/training/observer-B-512.json) | 328334 | `2dcf1379beb222b6e50b5822cb3cebdae33ba8a4745a034d9f400f1dc20e4179` |

The result JSON additionally binds both completed-event streams, all metric files, snapshot receipts, source-preservation controls and fresh reload parity. The eight observation rows are scientific observations only; the four discarded preflight observations are retained separately in that result. Table rounding does not replace the saved full-precision values.
