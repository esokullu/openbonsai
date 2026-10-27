**Window-zero training-budget control — completed small local diagnostic, 27 October 2026 (Istanbul).**

B reduced pooled search hard-reconstruction MSE by **46.03%** relative to A, from **0.0003113106 to 0.0001680105**. All 16 search records improved; the individual reductions were 42.06–50.14%. Calibration MSE fell **47.30%**. The fixed A control reproduced the historical window-zero state exactly, and both compact final states reloaded with bitwise-identical selected-window outputs. Total model-process time was **614.171 seconds (10.24 minutes)**.

This supports a benefit of additional repeated exposure and a longer annealing schedule in the first layer. It does **not** establish full-model language-quality recovery: window zero has no compressed-prefix drift, and this control measured no NLL, ARC score, generation quality, inference speed, or deployable full-model size. The earlier severe full-model failures remain unresolved. No further training was launched.

The exact bounded authorization is in [request.txt](D:/openbonsai/runs/2026-10-27-5090-window0-budget-control-v1/analysis/request.txt). The fixed recipe and all commands are in [experiment-plan.json](D:/openbonsai/runs/2026-10-27-5090-window0-budget-control-v1/experiment-plan.json) and [budget-supervisor.json](D:/openbonsai/runs/2026-10-27-5090-window0-budget-control-v1/budget-supervisor.json). Existing artifacts and dirty main were preserved.

**Pooled hard reconstruction against the same original teacher T0.**

All six selected projections in native layer zero were evaluated on identical inputs/masks/native kwargs. Original-prefix and conditional targets were verified equal at this layer. The initializer below is actual hard ternary quantization at progress 1; dense bypass is a separate exact source check.

| State | Set | MSE | Relative RMSE | Cosine | Output RMS | Target RMS |
| --- | --- | --- | --- | --- | --- | --- |
| Original hard initializer | calibration | 0.0024979247 | 0.463974 | 0.942535 | 0.066850648 | 0.107719918 |
| Original hard initializer | search | 0.0024892671 | 0.464264 | 0.942241 | 0.066709015 | 0.107465972 |
| A: 128 updates, one pass | calibration | 0.0003091191 | 0.163218 | 0.986619 | 0.105463065 | 0.107719918 |
| A: 128 updates, one pass | search | 0.0003113106 | 0.164182 | 0.986454 | 0.105269602 | 0.107465972 |
| B: 512 updates, four passes | calibration | 0.0001629084 | 0.118488 | 0.992956 | 0.106881799 | 0.107719918 |
| B: 512 updates, four passes | search | 0.0001680105 | 0.120614 | 0.992701 | 0.106524013 | 0.107465972 |

Reductions use A's pooled MSE as the denominator. Absolute B−A changes are −0.0001462107 on calibration and −0.0001433001 on search. Metrics accumulate FP64 sums over valid hidden elements, excluding padding: calibration has 128 records / 59,855 positions / 245,166,080 elements; search has 16 records / 7,203 positions / 29,503,488 elements. Relative RMSE is sqrt(total squared error / total squared target); cosine is pooled dot/(pooled norms). These are hidden-activation errors, not language-model losses. Both final states were frozen before comparative final/search interpretation. The reused search set is development evidence, not untouched final confirmation.

Complete metrics, all source bindings and the independently checked arithmetic are saved in [analysis/budget-results.json](D:/openbonsai/runs/2026-10-27-5090-window0-budget-control-v1/analysis/budget-results.json) and [analysis/budget-results-review.json](D:/openbonsai/runs/2026-10-27-5090-window0-budget-control-v1/analysis/budget-results-review.json).

**Every search record, paired B versus A.** Full IDs, masks and original dataset row IDs are retained in the bound consumed records. The table uses unambiguous 12-character ID prefixes; the JSON contains full hashes.

| Record ID prefix | A MSE | B MSE | B−A MSE | Reduction |
| --- | --- | --- | --- | --- |
| f4d9c1d8988f | 0.0003101965 | 0.0001613844 | -0.0001488121 | 47.97% |
| b5400127dd7b | 0.0002910196 | 0.0001490151 | -0.0001420045 | 48.80% |
| 4c3a4d9c4bee | 0.0003258578 | 0.0001834746 | -0.0001423833 | 43.69% |
| b1fa1f22e9f7 | 0.0003459602 | 0.0001724897 | -0.0001734705 | 50.14% |
| ac7705b02cac | 0.0003479498 | 0.0001770902 | -0.0001708596 | 49.10% |
| 8f2626477def | 0.0002716208 | 0.0001432769 | -0.0001283440 | 47.25% |
| 53b03fd9ccfc | 0.0002881304 | 0.0001577319 | -0.0001303985 | 45.26% |
| a0313555b06d | 0.0002942470 | 0.0001667475 | -0.0001274995 | 43.33% |
| 96229087728c | 0.0002818429 | 0.0001490014 | -0.0001328415 | 47.13% |
| 85eed23277b5 | 0.0003473330 | 0.0001870225 | -0.0001603106 | 46.15% |
| 050b91f2111f | 0.0002930010 | 0.0001598761 | -0.0001331248 | 45.43% |
| 1f1cc714b368 | 0.0004134259 | 0.0002336430 | -0.0001797830 | 43.49% |
| b3ecd7445f54 | 0.0003293673 | 0.0001908314 | -0.0001385359 | 42.06% |
| 7c3532ec4bf9 | 0.0002878096 | 0.0001536229 | -0.0001341867 | 46.62% |
| 45cc6aac3d55 | 0.0003071132 | 0.0001681525 | -0.0001389607 | 45.25% |
| 92ed788cb828 | 0.0002741858 | 0.0001448243 | -0.0001293615 | 47.18% |

**Fixed method, source and data.**

Original official `Qwen/Qwen3.5-9B`, revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`, BF16, native multimodal conditional wrapper and original qualified module names; no task fine-tuning. The executed checkout/tools descend from the sealed original-prefix sequential v2 runtime, retaining its memory, row-chunk, FP32-state, tokenizer/EOS and native-Qwen fixes. All 200 wrappers were constructed in authenticated historical order/RNG; only the six layer-zero projections were trained. Other model tensors stayed exact.

The pair kept group 128, signed rotation block 4096/seed 5090, rank 64, sigmoid mapping, frozen-Jacobian hard rule, Huber delta 1 averaged over valid hidden elements, factor/LoRA base LR .005/.001, weight decay 0, clipping 1, batch 1, sequence length 512 and checkpointing. Standard AdamW realized betas (.9,.999), epsilon 1e−8, foreach=null, fused=null, amsgrad/capturable/differentiable/maximize=false. No optimizer/kernel choice was overridden for the observer.

A used 128 updates (102 soft, 26 hard); B independently used 512 (409 soft, 103 hard), with bitwise-identical initial selected tensors and RNG, fresh optimizer and original weights. For each arm, progress=s/T and LR multiplier=1−(s−1)/T; soft iff progress≤.8. B was never warm-started from A. Each arm used a fresh Random(5090), shuffling a fresh index list each epoch with evolving RNG. B's four orders differ, and A/B's first order matches the historical control. Valid-position presentations were 59,855 and 239,420, respectively; these are repeated exposure to the same 128 records.

Inherited calibration and disjoint search prompts come from `HuggingFaceH4/ultrachat_200k`, revision `8049631c405ae6576f93f445c6b8166f76f5505a`, `train_sft`. No new traces were generated. Saved generation settings used the same pinned original model/template, greedy decoding, thinking=false, EOS IDs 248044/248046, maximum 512 new tokens. The native diagnostic backend remains the sealed eager runtime; trace generation's recorded backend was SDPA.

| Set | Records | Prompt tokens | Generated tokens | Hit generation cap | Shortened to 512 | Right tokens dropped |
| --- | --- | --- | --- | --- | --- | --- |
| calibration | 128 | 8815 | 58089 | 102 | 105 | 7049 |
| search | 16 | 895 | 7023 | 12 | 12 | 715 |

A generation-cap flag means no terminal EOS was reached within the limit; semantic answer completeness was not newly judged. Calibration has 5,681 padding positions; search has 989. These truncations limit the represented response span and were held constant between arms. Two inherited selection IDs (indices 44/53) differ from prepared-prompt hashes; the actual source-prompt hashes, token IDs/masks and raw trace digests were authenticated without changing the data. The immutable 144-record CPU/disk cache was captured once, independently verified against native original inputs/T0/kwargs, and reused by both arms; capture stops before the vocabulary head.

The four epoch-order SHA256 digests (compact JSON integer list plus LF) are:

- `5a060191025e871784b07051151a2b885615cf0b819f9b5f749fbde2bc734849`
- `5ba4bb8e85de99f0689ad238c4d941d85d977a53d88513ac771fef867e1cf043`
- `076a2b3dfb0e028baf382ccc57732634563db965341d4d0835bbe7c47baee0e1`
- `03ebaa0687a2d226a75dd60f6efeb8a2de03d89f6e485fdee1c784bdb05462f9`

**Soft/hard transition and optimizer evidence.**

On predeclared calibration record 0, a no-gradient probe compared soft and hard forward passes with identical parameters, inputs and masks. The live frozen Jacobian remains the actual last-soft PRE-update snapshot; observer calls happen after optimizer updates. Comparing neighboring training losses would mix records and parameter states.

| Arm | Last-soft step / sharpness | Soft MSE | Hard MSE | Hard−soft | Relative increase |
| --- | --- | --- | --- | --- | --- |
| A | 102 / 29.8828125 | 0.0002898714 | 0.0003010033 | 0.0000111319 | 3.8403% |
| B | 409 / 29.9560546875 | 0.0001465336 | 0.0001558085 | 0.0000092748 | 6.3295% |

Both probes used 2,097,152 valid hidden elements. Snapshots were preserved into first-hard A103/B410, and probe flags, gradients and RNG were unchanged. Probe MSE is FP64 arithmetic; the separately saved Huber training numerator uses inherited FP32 arithmetic and should not be reconstructed as exactly half this MSE.

All 640 scientific updates had finite loss, gradients, FP32 learned state/moments and updates. No gradient tensor was missing; zero first-step LoRA-A gradients are expected from zero-initialized LoRA B. All eight actual-moment observations—A1/102/103/128 and B1/409/410/512—covered 30 parameters / 10,092,544 coordinates and passed exact nonmutation. Four separate timing updates used schedule locations A102/103 and B409/410 but actual stored optimizer steps 1/2 in each arm, followed by exact rollback. They are excluded from scientific counts.

[Optimizer and weight appendix](D:/openbonsai/runs/2026-10-27-5090-window0-budget-control-v1/analysis/optimizer-and-weight-notes.md) reports every family observation, actual-moment domains, raw-factor D/P and mapped-factor/code/RMS/correction measurements, with links to the complete per-parameter JSON. The verified standalone observer was imported from commit `eab219ebfd0227a6069dca9e01c6ed996da7edc9`; no commit-wide merge occurred.

Observed damping is a diagnostic of the stored AdamW moments, not proof that epsilon caused quality failure. Undefined comparators and stored zero moments are retained explicitly. Raw-factor D≤P passed for all six zero-origin family rows; path length is not phase net displacement or convergence, and LoRA A has a nonzero origin. Weighted zero-code occupancy was 30.8782% initially, 30.8986% for A, and 30.9491% for B. Hard BF16 weight RMS changed only from .00858127 to .00858411/.00859183, while source RMS was .01294621; activation quality cannot be inferred from weight RMS alone.

**Runtime, memory and admission.**

Executed locally on RTX 5090 (sm_120), NVIDIA driver 596.21; Python 3.12.11, Torch 2.14.1+cu130, CUDA 13.0, Transformers 5.20.2. Prelaunch GPU free memory was 32,051 MiB; system RAM was 268,198,324 KiB total / 127,431,484 KiB free, and disk free space was 495,200,010,240 bytes. No cloud GPU was used. One original model was resident at a time.

| Process | Wall seconds incl. exit | Peak allocated GiB | Peak reserved GiB | Minimum headroom GiB |
| --- | --- | --- | --- | --- |
| preflight | 137.797 | 20.2128 | 20.2852 | 9.8467 |
| fit | 380.671 | 20.2186 | 20.2852 | 9.8467 |
| reload | 95.703 | 19.4845 | 19.5879 | 10.5537 |

A/B optimization phases took 36.740/124.557 seconds (including nested required observers/boundary probes), or 3.484/4.111 completed updates/s. The one-time 144-record capture and independent cache verification took 2.464/3.291 seconds. Initial/A/B all-record calibration+search metrics took 19.570/19.905/19.435 seconds; weight diagnostics took 5.611/5.867/5.572 seconds. Fresh reload includes original loading, conversion, source hashing and both probes. All 91 metered outer phases passed the ≥4 GiB headroom gate; minimum conservative/observed headroom was 9.847 GiB. Nested observer/probe timings must not be added again to their enclosing optimization phases. Complete process wall time also includes imports and work outside measured phases.

The frozen admission estimate was 1,293.717 seconds, using completed setup + 1.25×remaining work + a separate 300-second reserve. Independent review found one omitted standalone preservation pass (20.849 seconds before margin): four calls were needed but that named bucket reserved three. Adding it without borrowing other slack gives **1,319.779 seconds (22.00 minutes)**. Conservative subsequent boundary checks also passed. This nominal accounting defect was recorded in append-only reviews; the running source and 3,600-second cap stayed unchanged. Actual complete model time was 614.171 seconds, with no retries, overlap, cap extension or successor. Correct this category count before any future authorized reuse of the supervisor.

**Correctness, preservation and saved state.**

Ninety unique focused CPU tests passed before launch; these cover selection/cache reuse, A parity, evolving B schedule, callback ordering, rollback, supervisor and observer/metric behavior. Eight saved-result summarizer tests also passed. The full previously completed suite was not repeated. The isolated trainer adapter adds optional window IDs, caller-owned cached records and an after-step callback; the default numeric path remains unchanged. New driver/helpers narrow orchestration, preserve authenticated initialization and add bounded observations. Only the verified observer file was imported from its published commit. The runtime source/config allowlist was frozen before model execution.

The A gate passed before B: final selected FP32 tensors, shared historical update fields, same-input native boundary output and rollback of temporary historical-state assignment were exact. In a separate fresh process, each saved compact state reproduced its own BF16 [1,512,4096] hard output bitwise with maximum absolute error 0. Original and unselected tensors were checked throughout. Both compact files are 54,047,955 bytes, with signs/statistics and source/config lineage; they require the original model and are not deployable compressed checkpoints.

Final source/input/main/worker audit: [analysis/final-preservation.json](D:/openbonsai/runs/2026-10-27-5090-window0-budget-control-v1/analysis/final-preservation.json). Execution audit: [analysis/execution-review.json](D:/openbonsai/runs/2026-10-27-5090-window0-budget-control-v1/analysis/execution-review.json). Budget correction: [analysis/budget-accounting-review.json](D:/openbonsai/runs/2026-10-27-5090-window0-budget-control-v1/analysis/budget-accounting-review.json). Main HEAD remains `891d260af96a7ce61ccbcb23474015d975bb18dc`; all 39 source-boundary files and the recorded Git status/diff were preserved. The 182 frozen experiment files and actually used historical references were revalidated. Prior whole-ancestor audit receipts were reused rather than rescanning entire archives. All three owned model children exited successfully and were reaped.

The initial raw GitHub observer fetch returned 404 after copying sources; that evidence is preserved. The exact file was then obtained through authenticated GitHub access and SHA-verified before the freeze. One CPU test-fixture deadline argument was corrected before the successful focused suite. Neither involved a model-process retry. No source changed underneath running model work.

Recorded launch command (already executed once; the supervisor intentionally rejects rerunning this directory):

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
& 'D:\openbonsai\.venv\Scripts\python.exe' -B 'D:\openbonsai\runs\2026-10-27-5090-window0-budget-control-v1\tools\budget_supervise.py'
```

**Selected exact SHA256 bindings.** The final inventory covers every file, including individual observer receipts, state metadata, logs, cached tensors and analysis. Full source/config bindings are in the runtime freeze and authentication JSON; all four order hashes appear above.

| File | Bytes | SHA256 |
| --- | --- | --- |
| [experiment-plan.json](D:/openbonsai/runs/2026-10-27-5090-window0-budget-control-v1/experiment-plan.json) | 2246 | `bdb993010ddd448796f4ac827e4efaaaa4bcbdb38e1824f84ce2fab1271779bd` |
| [runtime-source-freeze.json](D:/openbonsai/runs/2026-10-27-5090-window0-budget-control-v1/runtime-source-freeze.json) | 29593 | `5f40a5af2f441c4813d7d99f5e776f1b37fd13adab3b8c44895b1f0320df2604` |
| [analysis/budget-authentication.json](D:/openbonsai/runs/2026-10-27-5090-window0-budget-control-v1/analysis/budget-authentication.json) | 151030 | `3a5df8de03ed6d666098971b56be26b12d6a61f2934f8be67c45d4d6835685dc` |
| [checkout/openbonsai/trainer.py](D:/openbonsai/runs/2026-10-27-5090-window0-budget-control-v1/checkout/openbonsai/trainer.py) | 38641 | `ec43a0730768e403ff51e76c3629d3b3cc195fb02f159e4532a3fd859016d7df` |
| [checkout/openbonsai/optimizer_diagnostics.py](D:/openbonsai/runs/2026-10-27-5090-window0-budget-control-v1/checkout/openbonsai/optimizer_diagnostics.py) | 15842 | `eb2821b0ebc39e24ce211362c724cfe87d4a8834eb1e48c8f2b110552de3e942` |
| [inputs/calibration/ayot128.jsonl](D:/openbonsai/runs/2026-10-27-5090-window0-budget-control-v1/inputs/calibration/ayot128.jsonl) | 1014760 | `d77644080002e396c597e93b084d0e00596a69895281b73cd1d7952a62ce80f6` |
| [inputs/calibration/consumed128.jsonl](D:/openbonsai/runs/2026-10-27-5090-window0-budget-control-v1/inputs/calibration/consumed128.jsonl) | 612783 | `487a093fe330f35882ed04f969fa91e324b91630f5e17af003e02beebd23a699` |
| [data/search-consumed16.jsonl](D:/openbonsai/runs/2026-10-27-5090-window0-budget-control-v1/data/search-consumed16.jsonl) | 77139 | `443aaba5e5b41c641c27cdcd45f34fd5a2e770fa466321c9bbab84436ff40aaf` |
| [training/A-final-state.pt](D:/openbonsai/runs/2026-10-27-5090-window0-budget-control-v1/training/A-final-state.pt) | 54047955 | `5fa0f916dd67f85cb39da104e7f1ffe07a68fd42e02995146d1421cc4284c9cf` |
| [training/B-final-state.pt](D:/openbonsai/runs/2026-10-27-5090-window0-budget-control-v1/training/B-final-state.pt) | 54047955 | `7a9c4ee37a8d4bbb65f5518e36dbd509fc77468b06dd09de1575b3ccc7cbd6d8` |
| [reload/exact-parity.json](D:/openbonsai/runs/2026-10-27-5090-window0-budget-control-v1/reload/exact-parity.json) | 1131 | `9048dcd5efeb18f7ed961404508eb8463d873f36b8d75e78388236a19e81ec6d` |
| [analysis/budget-results.json](D:/openbonsai/runs/2026-10-27-5090-window0-budget-control-v1/analysis/budget-results.json) | 3316093 | `1d69128886ab4375e8f386bc96d3b001f03fafe35c8c95fe6d6a8ef0966f4c83` |
| [analysis/budget-results-review.json](D:/openbonsai/runs/2026-10-27-5090-window0-budget-control-v1/analysis/budget-results-review.json) | 78469 | `0af63e7bfd9ed7dee88a4804ff26a4b52a14e5bed195fce1481041a7791e4d86` |
| [analysis/final-preservation.json](D:/openbonsai/runs/2026-10-27-5090-window0-budget-control-v1/analysis/final-preservation.json) | 150599 | `95434375ee916e23a7514e46ff1d516bad9295ce42a66314c39e2dcb19d4e588` |

Parent source-freeze SHA256: `717468370847ab1c3405589cf797f9d6fc26c450c57d0cce02290b980875025b`; parent final-seal SHA256: `66e8712d99046fe80235a56b0a22fee2120fa0579afdc87a2bae7687aa72dc52`. Bound prior saved-factor evidence: `307654bd73af94168f4ee6c50bf65bc20821bf8484d0bd7692eaba34b74a7668`; independent review: `8736dba24c534441cc04e9758887090214631510f021e6d1a17e78ee4ed747c9`.

Historical context remains in the [original failure diagnosis and BITCOS repack report](D:/openbonsai/runs/2026-10-26-5090-diagnosis/REPORT.md) and [sealed full-model v2 report](D:/openbonsai/runs/2026-10-26-5090-original-prefix-sequential-v2/REPORT.md). This local result does not replace those language-quality scores or justify automatic escalation. The experiment ends after this fixed pair, report and seal.
