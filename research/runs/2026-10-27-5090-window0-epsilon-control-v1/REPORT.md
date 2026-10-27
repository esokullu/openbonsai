**C512 epsilon control — stopped before model launch; no C quality result.**

The one authorized supervisor invocation failed its worker-absence gate before spawning any model child. The exact saved error is `Prior or current owned model worker remains active`. The source gate passed, but preflight, B native replay, C fitting, metrics and fresh reload did not begin. There were **0 model-process seconds, 0 optimizer updates and 0 new scientific observations**. No C checkpoint was produced. The failed attempt is preserved; no retry, successor or epsilon sweep was launched.

The current evidence points to a Windows Python-launcher false positive in the new guard. A benign CPU ancestry inspection found a `.venv/Scripts/python.exe` launcher and a child base interpreter forwarding the same arguments. The guard newly matches `epsilon_supervise.py` command-line substrings but excludes only `os.getpid()`. With this launcher topology, the supervisor's own launcher can match as an apparent additional worker. The failed guard did not persist its matching process rows, so the exact historical rejected PID is **unproven**. The forensic receipt distinguishes the observed topology and source condition from that inference.

**Historical reference versus unavailable C.** These initial/B numbers are authenticated results from the sealed parent, not measurements repeated in this attempt.

| State | Calibration hard MSE | Search hard MSE | Search relative RMSE | Search cosine | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| Original hard initializer | 0.0024979247 | 0.0024892671 | 0.464264 | 0.942241 | Historical parent measurement |
| B512, epsilon 1e-8 | 0.0001629084 | 0.0001680105 | 0.120614 | 0.992701 | Historical parent measurement |
| C512, epsilon 1e-9 | Unavailable | Unavailable | Unavailable | Unavailable | No model process launched |

The parent comparisons use 128 calibration records / 59,855 valid positions / 245,166,080 hidden elements and 16 disjoint search records / 7,203 positions / 29,503,488 elements. Parent B's search output/target RMS is 0.106524013/0.107465972. Its four after-update observations at 1/409/410/512 remain historical references. **There is no C-versus-B quality, parameter-motion or optimizer-attenuation conclusion.** This failed attempt provides no additional evidence about full-model quality or deployable size.

**Fixed intended intervention and completed preparation.**

The source was copied from the actual sealed parent checkout/tools. It retains the original `Qwen/Qwen3.5-9B` revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`, historical BF16/native-wrapper policy, all-200-wrapper seeded construction order, group128, signed rotation block4096/seed5090, rank64, sigmoid mapping, frozen-Jacobian hard stage, Huber delta1, factor/LoRA LR .005/.001, decay0, clip1, batch1 and length512. The intended C schedule was four passes over the same 128 records: 409 soft and 103 hard updates, progress=s/512 and LR multiplier=1−(s−1)/512. C was to start from original initialization with fresh optimizer state; the saved B state was only a control reference.

The narrow trainer change adds `TrainingConfig.optimizer_epsilon`, defaults it to 1e-8, validates a finite positive value, supplies it to the AdamW constructor and reports it. The C adapter supplies 1e-9 in defaults and every group. Quantizer, native computation, memory bounds, BF16/FP32 behavior and published observer stayed unchanged. The C observer adapter changes the label while using the inherited B512 observation locations and the exact published helper. The trainer diff is saved for later reviewed publication; main and completed runs were not edited or published.

**77 unique focused CPU tests passed**: 19 trainer/preflight/observer tests, 50 supervisor/provenance tests and 8 driver/admission/label tests. Default-B512 CPU events, final learned state and window report matched the authenticated parent exactly. C's CPU fixtures verified epsilon, all 512 schedule positions, four orders, callbacks, snapshot timing and two-update rollback. These CPU checks did not cover the actual Windows supervisor launcher ancestry. During development, four preflight tests exposed forwarding an internal AdamW default as an unsupported constructor keyword; it was fixed before freezing and the affected tests passed. The original failed CPU log remains preserved. This was not a model-process retry.

The prelaunch authentication verified 159 copied files, specific parent B/state/results/observer inputs, actual immutable cache bindings and used historical initializer/template references. The source/config allowlist was frozen at **183 files** before the invocation. The fitting/raw data digests remain `487a093fe330f35882ed04f969fa91e324b91630f5e17af003e02beebd23a699` and `d77644080002e396c597e93b084d0e00596a69895281b73cd1d7952a62ce80f6`. No calibration text was generated or consumed by a model process in this attempt.

**Accounting, resources and execution.**

The parent accounting correction was incorporated before the freeze: preflight would complete `B-control-source-preservation` and `source-preservation-after-rollback`, then reserve exactly two remaining standalone checks, `final-source-preservation` in fit and `fresh-source-preservation` in reload. Each has its own future-cost bucket. Four scientific observer calls and one same-parameter boundary probe are budgeted separately, with their measured costs subtracted from the two timing-update bases. The independent review also verified complete 512-update, metric, state-I/O, cache and reload coverage. The 1.25 remaining-work margin and separate 300-second reserve stay intact.

The planned external cap was 1,800 aggregate model-process seconds, with at most 512 scientific and two discarded updates, one GPU owner and at least 4 GiB conservative headroom. **No measured C admission estimate was produced**, because the supervisor failed before preflight. The earlier 8–12-minute estimate was a parent-based planning estimate, not an actual C runtime measurement. Unused allowance does not authorize another attempt.

Prelaunch checks found RTX5090 driver596.21, 32,024 MiB free GPU memory, and 493,647,429,632 bytes free disk. The inherited environment is Python3.12.11, Torch2.14.1+cu130, CUDA13.0 and Transformers5.20.2. C's GPU allocated/reserved peaks, throughput, training loss, inference metrics and artifact size are **not measured**. Queries and CPU tests do not establish a new CUDA model result.

The supervisor started at `2026-10-26T22:36:42.282493+00:00` and ended at `2026-10-26T22:36:44.494552+00:00`. Its model-child list is empty, aggregate model time is exactly0 and all-owned-children-reaped is true. A separate terminal audit checks the current process list, parent/source/cache bindings and dirty-main preservation. Main HEAD remains `891d260af96a7ce61ccbcb23474015d975bb18dc`; its recorded 39-file source boundary, Git status and diff are preserved. Existing ancestor audits are reused, with no whole-archive rescan.

Executed command, once:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
& 'D:\openbonsai\.venv\Scripts\python.exe' -B 'D:\openbonsai\runs\2026-10-27-5090-window0-epsilon-control-v1\tools\epsilon_supervise.py'
```

**Remaining implementation issue and limits.**

A future reviewed correction should distinguish the supervisor's verified launcher ancestry from unrelated workers, preserve detection of unrelated concurrent supervisors/drivers, and record matching PIDs/parent PIDs/command lines on rejection. A focused regression should cover the two-process Windows launcher topology as well as an unrelated real worker. That correction was **not applied to this frozen failed attempt**, and no relaunch was made. The existing broad command-line substring condition and missing rejection evidence remain visible for review.

The bounded intervention is incomplete scientifically. Larger motion was never a success criterion; finite CPU fixtures are not model quality. Even a completed first-layer result would not establish later-layer behavior, full-model useful quality, or an approximately2GB language artifact. This attempt ends with its failure report, independent review and seal.

**Exact primary bindings.** The final inventory additionally covers logs, focused tests, forensic receipts and report reviews.

| File | Bytes | SHA256 |
| --- | ---: | --- |
| [runtime-source-freeze.json](D:/openbonsai/runs/2026-10-27-5090-window0-epsilon-control-v1/runtime-source-freeze.json) | 28993 | `94f08b87ab71faca2ed8fd9ff19be38ae66776e297801650495696d15cb9dc20` |
| [analysis/epsilon-authentication.json](D:/openbonsai/runs/2026-10-27-5090-window0-epsilon-control-v1/analysis/epsilon-authentication.json) | 214351 | `a4388347aa3577f47ad6affe7a3fbf68f1049d03b830db95ae06ba3abd1d548b` |
| [experiment-plan.json](D:/openbonsai/runs/2026-10-27-5090-window0-epsilon-control-v1/experiment-plan.json) | 3959 | `29303e07889eab91f8ac58cd07039c73f35a492a60189238ed25dd7de547ce29` |
| [epsilon-supervisor.json](D:/openbonsai/runs/2026-10-27-5090-window0-epsilon-control-v1/epsilon-supervisor.json) | 2188 | `1d5b0e350052bedc655421c88f3ea855f86a5f2a19769cb13f3b7fba5db458c9` |
| [analysis/focused-cpu-tests.json](D:/openbonsai/runs/2026-10-27-5090-window0-epsilon-control-v1/analysis/focused-cpu-tests.json) | 3469 | `37f722d226715fb60f6bdee2963fdbade4782f8ceeafef3f3d960573e22da019` |
| [analysis/source-review.json](D:/openbonsai/runs/2026-10-27-5090-window0-epsilon-control-v1/analysis/source-review.json) | 3410 | `a9998ae7317226033a9d3477b8331d17866e52c056c405753ea5107ae57339fb` |
| [analysis/epsilon-driver-independent-review.json](D:/openbonsai/runs/2026-10-27-5090-window0-epsilon-control-v1/analysis/epsilon-driver-independent-review.json) | 7115 | `514a2843f7db5a503d76b8896954fb17c70dde1aaa8a6c7e0247023e66badd10` |
| [analysis/epsilon-trainer.diff](D:/openbonsai/runs/2026-10-27-5090-window0-epsilon-control-v1/analysis/epsilon-trainer.diff) | 1593 | `0d58037876590b7d666356b8016fe6ebb9dc5daba5064ebd9fe1667bb7b1304b` |

Parent final seal: `466a13bbb67aabede63b97019b014ffc639b4e4519a0bf28f3cdc206a5f5f1cf`. Parent source freeze: `5f40a5af2f441c4813d7d99f5e776f1b37fd13adab3b8c44895b1f0320df2604`. Bound parent accounting review: `a5b75c24c0f7f6923e3717849edeee60479dee713a46bf1da536ef7a5a8cabd5`.

The [completed parent report](D:/openbonsai/runs/2026-10-27-5090-window0-budget-control-v1/REPORT.md) retains the actual B512 result. This attempt's [authorization](D:/openbonsai/runs/2026-10-27-5090-window0-epsilon-control-v1/analysis/request.txt) and [accounting clarification](D:/openbonsai/runs/2026-10-27-5090-window0-epsilon-control-v1/analysis/request-budget-clarification.txt) explicitly require preserving and reporting a failed attempt without retry.
