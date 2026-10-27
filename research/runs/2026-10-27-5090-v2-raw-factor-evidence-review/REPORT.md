Saved-JSON evidence review of sealed original-prefix sequential V2. The final report and seal were already complete. No reviewed receipt or used source text changed; no evidence gate failed. The established ownership check found model PIDs 349832, 363024 and 365224 absent before and after this review. Earlier completed preservation, artifact, execution, data/order and snapshot audits were reused; no model payload or whole ancestor was rescanned.

The completed `event=="step"` rows match the ordered `(window, step, sample_indices)` of all 4,096 scientific ledger authorizations. Every window 0–31 has 128 completed rows: 102 soft and 26 hard, totaling 3,264 soft + 832 hard. The four discarded V2 preflight authorizations are excluded, with their completion already established by the bound execution audit. The ledger itself records intentions, not completion. `ordered-step-binding.json` retains source line numbers and ledger ordinals for every matched triple.

For every raw_mu/raw_scale/raw_threshold family in all 32 windows, the first-before L2 and nonzero count are exactly zero; element/tensor counts remain fixed within that window/family and all saved before/after/update statistics are finite. Counts differ between window types: 1,638,400 or 1,703,936 elements and seven or six tensors. LoRA A starts nonzero and is excluded from the zero-origin displacement calculation.

`D = last parameter_after_update.l2` is the final displacement from the verified zero raw-factor initialization. `P = sum(updates.l2)` is the actual update path length; `D/P` is null when P=0. The sum of update RMS values is also a path statistic. Soft and hard values below are path lengths, not net phase displacements.

| Window | Raw family | D | P | D/P | Sum update RMS | Soft P | Hard P | Soft % / hard % |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | raw_mu | 25.099857 | 42.006922 | 0.597517 | 0.032180606 | 40.476345 | 1.530577 | 96.356 / 3.644 |
| 0 | raw_scale | 23.212617 | 36.588331 | 0.634427 | 0.028029539 | 35.533981 | 1.054350 | 97.118 / 2.882 |
| 0 | raw_threshold | 16.204382 | 27.810318 | 0.582675 | 0.021304891 | 26.815964 | 0.994354 | 96.425 / 3.575 |
| 16 | raw_mu | 46.968260 | 103.098565 | 0.455567 | 0.078981610 | 97.986139 | 5.112426 | 95.041 / 4.959 |
| 16 | raw_scale | 41.158446 | 89.533069 | 0.459701 | 0.068589373 | 85.140584 | 4.392485 | 95.094 / 4.906 |
| 16 | raw_threshold | 32.907599 | 77.377002 | 0.425289 | 0.059276869 | 73.026244 | 4.350758 | 94.377 / 5.623 |
| 31 | raw_mu | 67.757054 | 146.194959 | 0.463471 | 0.114214811 | 140.463576 | 5.731383 | 96.080 / 3.920 |
| 31 | raw_scale | 67.108114 | 141.773458 | 0.473348 | 0.110760514 | 136.387958 | 5.385500 | 96.201 / 3.799 |
| 31 | raw_threshold | 64.855915 | 144.275803 | 0.449527 | 0.112715471 | 138.640431 | 5.635372 | 96.094 / 3.906 |

All 96 rows satisfy strict D ≤ P. There are zero strict excesses and zero tolerance violations. The explicit summary-rounding allowance is `1e-8 + 1e-6*max(abs(D),abs(P))`; strict excesses are separately retained even if within that allowance. This guards separately rounded FP32 norm summaries and is not a new scientific quality threshold.

| Family | D range | P range | D/P min / median / max | Sum update RMS range | Hard path fraction range |
|---|---:|---:|---:|---:|---:|
| raw_mu | 15.690867–67.757054 | 26.517415–146.194959 | 0.422237 / 0.454053 / 0.597517 | 0.020314425–0.114214811 | 3.644%–6.129% |
| raw_scale | 15.186927–67.108114 | 23.646377–141.773458 | 0.419858 / 0.452615 / 0.642252 | 0.018114984–0.110760514 | 2.882%–5.858% |
| raw_threshold | 10.355158–64.855915 | 17.670990–144.275803 | 0.399462 / 0.421395 / 0.592002 | 0.013537368–0.112715471 | 3.575%–6.880% |

These are raw-factor norm summaries. They do not identify phase net displacement, coordinatewise or projected-weight movement, or Adam-epsilon dominance. Neither LoRA A’s final norm nor a ratio of unrelated norms is treated as displacement.

The following table reproduces all 32 saved hard reconstruction MSEs and the individual-record Huber losses at executed soft step 102, hard step 103 and final step 128. The hard reconstruction MSE and per-step Huber objective are different quantities. In all 32 windows, steps 102 and 103 use different records and an optimizer update separates their parameter states; their loss difference is not a hardening jump. Exact sample indices and labels are in evidence.json.

| Window | Hard MSE before | Hard MSE after | Soft 102 Huber | Hard 103 Huber | Final 128 Huber |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.0024979247 | 0.000309119078 | 0.000168968574 | 0.000160038428 | 0.00017693211 |
| 1 | 0.00154559849 | 0.00100203929 | 0.000432528614 | 0.000534217863 | 0.000573629106 |
| 2 | 0.00275471575 | 0.00232264402 | 0.0011445866 | 0.00129428273 | 0.00145995803 |
| 3 | 0.0061277034 | 0.00431150588 | 0.0022358573 | 0.00194107112 | 0.0021281552 |
| 4 | 0.00750270848 | 0.00679411559 | 0.00366302021 | 0.00344567373 | 0.00301325624 |
| 5 | 0.0123542044 | 0.0103366626 | 0.00623673247 | 0.00546724489 | 0.00536380289 |
| 6 | 0.0223384291 | 0.016738672 | 0.00803688075 | 0.00791968778 | 0.00974460226 |
| 7 | 0.0265463598 | 0.0209461771 | 0.0104065984 | 0.0107008694 | 0.0105934422 |
| 8 | 0.0271806132 | 0.0256520308 | 0.0108710025 | 0.0124737257 | 0.0125783961 |
| 9 | 0.0328008895 | 0.0303597191 | 0.0141763426 | 0.0128692212 | 0.0147167332 |
| 10 | 0.0434978495 | 0.0378053578 | 0.0206472687 | 0.0176646225 | 0.0217051692 |
| 11 | 0.0474024628 | 0.0423164092 | 0.0188598335 | 0.0202298257 | 0.0202531405 |
| 12 | 0.0504618055 | 0.0485544252 | 0.0259224009 | 0.0270829704 | 0.0235206187 |
| 13 | 0.0584630947 | 0.0560604378 | 0.0278576314 | 0.0290300511 | 0.0310973972 |
| 14 | 0.0737733246 | 0.0687476114 | 0.0325899832 | 0.0346498825 | 0.030287873 |
| 15 | 0.0888179872 | 0.0824672078 | 0.0389880836 | 0.0369444899 | 0.0415803976 |
| 16 | 0.105803944 | 0.101656218 | 0.0504281111 | 0.0467930809 | 0.0506932363 |
| 17 | 0.139760712 | 0.133052946 | 0.0583595261 | 0.0656284094 | 0.0617730916 |
| 18 | 0.232792577 | 0.211199161 | 0.101651415 | 0.0927547067 | 0.0972908288 |
| 19 | 0.365547284 | 0.337568084 | 0.151676297 | 0.155376464 | 0.147777542 |
| 20 | 0.501453526 | 0.477102903 | 0.217849284 | 0.214420781 | 0.21887061 |
| 21 | 0.677612815 | 0.651353358 | 0.256454349 | 0.264199674 | 0.281862378 |
| 22 | 0.965735064 | 0.918018677 | 0.370252728 | 0.396692693 | 0.351902574 |
| 23 | 1.2118438 | 1.1746047 | 0.452910274 | 0.4768641 | 0.47246179 |
| 24 | 1.57584064 | 1.53240027 | 0.568608165 | 0.560997367 | 0.58171922 |
| 25 | 1.88707881 | 1.85592921 | 0.622969389 | 0.677832425 | 0.696822226 |
| 26 | 2.291207 | 2.23677141 | 0.706071854 | 0.717222273 | 0.770230293 |
| 27 | 2.7342336 | 2.62902729 | 0.877428949 | 0.793008208 | 0.799146652 |
| 28 | 3.33999215 | 3.26362944 | 0.919337511 | 0.976919413 | 0.933894157 |
| 29 | 4.0167279 | 3.93058209 | 1.09093904 | 1.07904959 | 1.13937211 |
| 30 | 5.06207975 | 4.88826666 | 1.21313119 | 1.18780875 | 1.25114763 |
| 31 | 7.07560283 | 6.4524465 | 1.49319184 | 1.50803983 | 1.3759501 |

At windows 0/16/31 the transition JSON files store raw-factor/snapshot fingerprints, progress and sharpness. The completed audit already verified that the frozen-Jacobian snapshot at first hard step 103 retains the actual pre-update-102 reference (sharpness 29.8828125), while current factors have received update 102. This is saved parameter-reference identity evidence, not a same-parameter soft/hard output or loss comparison. Such a quantitative hardening-jump measurement is unavailable.

Available saved JSON includes aggregate raw-factor norms, per-projection effective LoRA-correction Frobenius norms, and first-projection effective/hard weight RMS and code occupancy. The transition review records their limited scope. Mapped-factor distributions, coordinatewise effective-correction movement, and coordinatewise bias-corrected Adam moments/epsilon comparisons are unavailable in the reviewed JSON. No state tensor was opened to derive them.

The new review helper’s first invocation stopped before process checking or output writes because it expected a generic training-interpretation status. The sealed status is `passed_completed_training_receipt_audit_quality_not_recovered`; the helper was corrected to check that exact string. This was a review-script schema correction, not a failed scientific gate or change to the sealed run.

Exact source identities used below are checked against the existing final seal. Reused receipts retain their original scopes; this review does not claim a fresh hash scan of uninspected checkpoint files.

| Source absolute path | Bytes | SHA256 |
|---|---:|---|
| `D:/openbonsai/runs/2026-10-26-5090-original-prefix-sequential-v2/final-validation.json` | 321692 | `66e8712d99046fe80235a56b0a22fee2120fa0579afdc87a2bae7687aa72dc52` |
| `D:/openbonsai/runs/2026-10-26-5090-original-prefix-sequential-v2/REPORT.md` | 16779 | `e17dacfd00c23cb74b49999a466d6838fa3ecf7c4a32ac575e4f97d3cd7a299a` |
| `D:/openbonsai/runs/2026-10-26-5090-original-prefix-sequential-v2/analysis/final-report-review.json` | 77434 | `73727933d7aed899f3d602ab6d69673d8c9f916ff03e9cbb207447fbd38e07b6` |
| `D:/openbonsai/runs/2026-10-26-5090-original-prefix-sequential-v2/analysis/v2-training-interpretation.json` | 114801 | `c8422be058d679d7346543fa1026dce5a77974600115be0d376c30977272556c` |
| `D:/openbonsai/runs/2026-10-26-5090-original-prefix-sequential-v2/analysis/final-execution-audit.json` | 550064 | `b90147f2e6260bf5ec600cc35dd78592a82bb684246412d8f1e4eeab3822662b` |
| `D:/openbonsai/runs/2026-10-26-5090-original-prefix-sequential-v2/analysis/final-preservation.json` | 115019 | `faa3d5000672f9623356cf478aa5953f35c23b0548cf82f09a5eb9a9d407d674` |
| `D:/openbonsai/runs/2026-10-26-5090-original-prefix-sequential-v2/analysis/sequential-artifact-accounting.json` | 632701 | `0c357e3c2e663bf160ce20db4764197a08d44c59b488146ca3c8d0cb8980fb02` |
| `D:/openbonsai/runs/2026-10-26-5090-original-prefix-sequential-v2/sequential-supervisor.json` | 6383 | `122c8dfca2f0868a785afb6e6d90b97bb393c2247594c3032378b18498b13262` |
| `D:/openbonsai/runs/2026-10-26-5090-original-prefix-sequential-v2/analysis/audit_v2_preservation.py` | 10127 | `0c70c84f6a20388ac71334cc394cd9d356278af535f394df9da2f4cd5f41b58c` |
| `D:/openbonsai/runs/2026-10-26-5090-original-prefix-sequential-v2/optimizer-step-ledger.jsonl` | 925461 | `cd2ed85b327a9a2d58f58eac410d3b26632e3cc9ec626e157f33554a4a88b2f6` |
| `D:/openbonsai/runs/2026-10-26-5090-original-prefix-sequential-v2/training/training-events.jsonl` | 30091578 | `ed6be6f7100b2f3b4b3cc530000a52936701d803ff9ae61e4cb84b1f36f19e7c` |
| `D:/openbonsai/runs/2026-10-26-5090-original-prefix-sequential-v2/training/training-report.json` | 298529 | `c673d0840c86226895c942b1fee996ada509696c49779c8fd8095e48cabf8f02` |
| `D:\openbonsai\runs\2026-10-26-5090-original-prefix-sequential-v2\training\transitions\window-0-pre-update-102.json` | 17610 | `9484f30d29e1d7b08c280ce4289704829047c3af6c3414fd53f13b47405f3bf1` |
| `D:\openbonsai\runs\2026-10-26-5090-original-prefix-sequential-v2\training\transitions\window-0-pre-update-103.json` | 17616 | `61345cb3698344937487fd20360d28d1a57204bc04176fefb7da0155f49cfa37` |
| `D:\openbonsai\runs\2026-10-26-5090-original-prefix-sequential-v2\training\transitions\window-0-final-state.json` | 7105 | `a74238c7a58786ec8d979d7d092c6e5bb95b322417c663d62d47450d7ca78b1b` |
| `D:\openbonsai\runs\2026-10-26-5090-original-prefix-sequential-v2\training\transitions\window-16-pre-update-102.json` | 17617 | `53c5ab71c4336f0b4d161d808c05d82b6c2c4d52ea379b575f8a92b0b4906ebd` |
| `D:\openbonsai\runs\2026-10-26-5090-original-prefix-sequential-v2\training\transitions\window-16-pre-update-103.json` | 17623 | `51dea108902c69e770b3551bd8384fdf206b20f26b7ce0e3f3b52ba72e15b730` |
| `D:\openbonsai\runs\2026-10-26-5090-original-prefix-sequential-v2\training\transitions\window-16-final-state.json` | 7111 | `f4454e2459d74b1655387fec8e7834d643a47113d3bc5372752f854c658f3c7a` |
| `D:\openbonsai\runs\2026-10-26-5090-original-prefix-sequential-v2\training\transitions\window-31-pre-update-102.json` | 20490 | `0c78ceb72be3f7bfd867a57b2c450019e14a3f13f26cb15d72ebe70a463347c9` |
| `D:\openbonsai\runs\2026-10-26-5090-original-prefix-sequential-v2\training\transitions\window-31-pre-update-103.json` | 20497 | `7fe9e68f197ebc69b1afe716449f9330298d9374f8873bc312f4d7cb5f9cd74e` |
| `D:\openbonsai\runs\2026-10-26-5090-original-prefix-sequential-v2\training\transitions\window-31-final-state.json` | 8268 | `b2c718454b973e51aad0eb03e475a793fa1f3bf932b8058605f7b1b6f9436969` |

Compact result: [evidence.json](D:/openbonsai/runs/2026-10-27-5090-v2-raw-factor-evidence-review/evidence.json). All 96 family/window rows, ranges, 32 reconstruction/loss rows, source bindings, exact step-order proof reference, and transition availability are included. No new recipe, quality threshold, monitor or model action was chosen. Review ends here.
