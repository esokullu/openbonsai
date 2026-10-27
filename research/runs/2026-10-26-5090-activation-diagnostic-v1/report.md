# Bounded activation diagnostic: stopped before restoration

The single authorized activation attempt failed at a template identity assertion after loading the original BF16 model. The driver compared raw CRLF-serialized template bytes with LF runtime text. Independent CPU checks establish identical normalized template text, rendered prompts and token IDs on the 16 fixed search prompts and saved probe. They do not establish model parity. No model restoration, original/bypass/hard logit check, activation capture, decomposition or selector ran. There was no GPU retry, fitting, successor experiment or automatic escalation.

The failure and partial outputs are preserved. This report completes failure accounting; it does **not** mark the scientific diagnostic complete. Previous quality conclusions remain unchanged; see the [sealed parent comparison](D:/openbonsai/runs/2026-10-26-5090-ultrachat-attempt2/report.md). This attempt supplies no new held-out quality result or causal explanation for degeneration.

| Phase | Status | Wall time | Peak CUDA allocated | Peak CUDA reserved |
|---|---|---:|---:|---:|
| Prepare 16 search-validation traces | Completed, exit 0 | 299.839 s | 17.656 GiB | 17.705 GiB |
| Activation process, including failed guard | Failed, exit 1 | 23.503 s | 17.538 GiB during load | 17.588 GiB during load |
| Original-model load inside activation process | Completed | 15.152 s | 17.538 GiB | 17.588 GiB |
| Restore / probes / activation capture / selector | Not executed | — | — | — |

Phase memory is PyTorch allocated/reserved, not whole-process or system GPU use. The activation load recorded at least 12.677 GiB free using both sampled and conservative accounting, exceeding the 4 GiB load gate. This says nothing about memory requirements of the unexecuted restoration/capture stages. GPU preparation and activation were separate sequential processes. [Preparation receipt](D:/openbonsai/runs/2026-10-26-5090-activation-diagnostic-v1/preparation-supervisor.json), [activation receipt](D:/openbonsai/runs/2026-10-26-5090-activation-diagnostic-v1/activation-supervisor.json), and [failure traceback](D:/openbonsai/runs/2026-10-26-5090-activation-diagnostic-v1/activation-failure.json) contain exact commands, environments, deadlines and timestamps.

**Search-validation inputs.** The 16 accepted records are the first 16 of the frozen 32 unused reserves from `HuggingFaceH4/ultrachat_200k`, `train_sft`, revision `8049631c405ae6576f93f445c6b8166f76f5505a`. Selection inherited the parent policy: first 5,000 source rows, first user message only, 20–160 templated prompt tokens, whitespace deduplication and seed 5090 shuffle. No supplied assistant answers or later turns were used. All 16 attempts were accepted without quality/length exclusions or replacements. Exact IDs, source row numbers and attempt order are saved in the [data audit](D:/openbonsai/runs/2026-10-26-5090-activation-diagnostic-v1/analysis/preparation-outcome-audit.json).

These inputs are for search/action selection, not an unused final confirmation set. The source audit verifies the 32 reserves were unattempted and disjoint from the enumerated prior fitting, search and evaluation registries. Independent completed-data checks found no exact prompt-ID, whitespace-normalized prompt, full prompt-token, full raw-token or full consumed-token matches against 144 prior fitting prompts. These checks do not exclude semantic topic overlap.

Generation used original `Qwen/Qwen3.5-9B` BF16 revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`, SDPA, cache enabled, non-thinking chat template, greedy decoding, maximum 512 new tokens, EOS IDs `[248044, 248046]` and seed 5090. There was no fine-tuning. Generation took 274.447 seconds, or 25.590 generated tokens/s; preparation loading took 17.496 seconds. The 299.839-second supervisor total also includes interpreter startup, authentication, writing and final validation.

| Trace accounting | Count |
|---|---:|
| Distinct samples / attempts / excluded attempts | 16 / 16 / 0 |
| Prompt / generated / raw tokens | 895 / 7,023 / 7,918 |
| Consumed active / padding positions | 7,203 / 989 |
| Consumed generated tokens | 6,308 |
| EOS completions / capped continuations | 4 / 12 |
| EOS-only continuations | 0 |
| Right-truncated records / discarded tokens | 12 / 715 |
| EOS retained / removed by consumption | 4 / 0 |

Consumption takes the first 512 prompt-plus-continuation IDs, then pads with 248044. Every retained prompt/continuation position has mask 1; padding has mask 0. All four EOS completions ended with 248046. Capped outputs are unfinished under this protocol; no semantic completion judgment was used. Raw traces: 118,848 bytes, SHA256 `9c1c05142d948d8cb88e4770c3ac2695afb91b1f8534802ba4d81dd780b80d86`. Consumed records: 77,139 bytes, SHA256 `443aaba5e5b41c641c27cdcd45f34fd5a2e770fa466321c9bbab84436ff40aaf`.

**Why the attempt stopped.** The assertion at [measure_activation.py:173](D:/openbonsai/runs/2026-10-26-5090-activation-diagnostic-v1/tools/measure_activation.py:173) compares different byte representations:

| Template representation | Bytes | SHA256 |
|---|---:|---|
| Parent saved file, 153 CRLF sequences | 7,909 | `a92e1dd97cb1cb175c9b70c0828e146bea4371c2643319b661b777e89811972e` |
| Official file / runtime template / parent after CRLF-to-LF only | 7,756 | `a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715` |

The raw parent file still matches its authenticated manifest. Two independent CPU investigations reproduced the cause. The independent check compared runtime, explicit raw CRLF and normalized LF templates: rendered prompt bytes and IDs agree for all 17 cases, and all 16 consumed prompt prefixes agree. [Independent review](D:/openbonsai/runs/2026-10-26-5090-activation-diagnostic-v1/analysis/template-failure-independent-review.md) documents the scope. No model weights or logits were loaded in that review; CUDA remained uninitialized.

A future driver should authenticate the raw saved asset against its immutable manifest separately from the runtime template text identity, using a declared canonical representation and checking frozen prompt IDs. This is a proposed correction only. The frozen driver was not edited and no replacement run was launched. This guard bug does not explain the earlier compressed-model quality failure.

**Isolation and validation.** This run copied the authenticated sealed attempt2 source based on `ff5bbf10e94d5d35282b9dfa48fa13069289fbfd`. Only the activation collector/statistics from reviewed commit `419d20ef43a0576f2f182842b88fbdb3c995aea0` were ported, alongside diagnostic drivers and tests. The sole production compatibility adaptation omits `teacher_prefix="quantized"` from a call to the inherited conditional-only `cache_window`; the trainer was not replaced. The statistics module is byte-identical to the reviewed commit. All 84 inherited checkout files, including all 25 inherited core modules and previous memory/chunking/tokenizer/stopping fixes, remained unchanged.

Before source freeze, review strengthened parent-receipt binding to the authenticated parent manifest and consumed-data binding to exact designated path, byte count, hash and accepted attempt order. The final combined CPU suite passed **171 tests in 11.86 seconds**, including strict restoration validation, native/replay/order fixtures, bounded FP64 statistics and invalid input bindings. These are CPU fixtures, not evidence of a completed real-model restore/capture. All setup failures and corrections are documented in [setup notes](D:/openbonsai/runs/2026-10-26-5090-activation-diagnostic-v1/analysis/setup-and-draft-validation-notes.json) and retained logs. The post-failure CPU diagnosis initially lacked UTF-8 mode; that failed read and its successful UTF-8 invocation are also retained. No runtime or dependency was changed after freeze.

The frozen plan would restore 1,000 learned FP32 tensors and 200 explicit saved sign arrays into 200 wrappers (group 128, rank 64, sigmoid, chunk 4096), without an optimizer or packed-scale substitution. The compact state file was authenticated by size/hash; its contents were **not restored into this loaded model**. Exact original, bypass and hard saved-probe equality was required before windows `[0,1)` and `[16,17)` could be measured. First-two-record native/oracle/A-B-A/reverse checks and memory/time gates preceded the full 16-record bounded FP64 decomposition. The selector required gap RMS at least half total error RMS on at least 12/16 inputs above measured noise; zero/near-noise denominators were unevaluable. None of these real-model gates executed. No selector value exists.

Environment: Windows, RTX 5090 (32,607 MiB, sm120), NVIDIA driver 596.21, Python 3.12.11, PyTorch 2.14.1+cu130 / CUDA 13.0, Transformers 5.20.2, safetensors 0.8.0, NumPy 2.5.4, Accelerate 1.15.0, huggingface-hub 2.3.0 and tokenizers 0.23.3. Installed package versions match the parent environment. Generation used SDPA; the unexecuted parity protocol selected eager attention. Full settings and source hashes are in the [runtime freeze](D:/openbonsai/runs/2026-10-26-5090-activation-diagnostic-v1/runtime-source-freeze.json).

**Preservation and remaining work.** The independent post-failure audit rehashed all 2,034 parent files (16,192,227,790 bytes), 39 main-checkout source files and 108 frozen diagnostic files. All remain unchanged. Main HEAD remains `891d260af96a7ce61ccbcb23474015d975bb18dc`, with the same dirty status and diff. Parent seal SHA256 is `d7ae0ccf66b38f223366beb105beba32d651e5c2c1be166c21624eac7885bf8a`; diagnostic runtime freeze SHA256 is `08830dd39e3275b8686a189ce9f385621baefe3ed0ddf532851b4033601e1f8c`. See [preservation evidence](D:/openbonsai/runs/2026-10-26-5090-activation-diagnostic-v1/analysis/failure-preservation.json).

Only input binding, original configuration provenance and original load telemetry exist in `measurement/`. Loaded tensor hashes, wrapper restoration, model probes, activation captures, post-state equality and a completed measurement seal remain pending. The final failure inventory seals reports, raw traces, source, tests, logs and partial outputs; its successful integrity check must not be read as successful measurement. Both model processes exited. No successor, training/grid/RCO experiment, model upload or code push was started. Review of the guard correction and outstanding gates is the next decision; this attempt remains stopped.

Exact commands and verification scope are in [reproduction.md](D:/openbonsai/runs/2026-10-26-5090-activation-diagnostic-v1/reproduction.md); the final inventory is [final-validation.json](D:/openbonsai/runs/2026-10-26-5090-activation-diagnostic-v1/final-validation.json).
