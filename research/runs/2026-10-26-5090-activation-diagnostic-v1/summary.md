# Activation diagnostic disposition

**Stopped at a required template guard; no activation result.** The one activation process exited 1 after loading original BF16 weights. A raw saved CRLF template hash was incorrectly compared with LF runtime text. Independent CPU checks found identical normalized text, rendered prompts and token IDs for all 16 search prompts plus the saved probe. This identifies a driver guard bug, not the cause of compressed-model quality failure.

| Phase | Result | Elapsed | Peak allocated / reserved |
|---|---|---:|---:|
| New search-validation traces | 16/16 accepted, no exclusions | 299.839 s | 17.656 / 17.705 GiB |
| Activation attempt | Failed before restoration | 23.503 s | 17.538 / 17.588 GiB during load |
| Restore / model probes / captures / selector | Not executed | — | — |

The traces contain 895 prompt and 7,023 generated tokens. Four reached EOS; twelve hit the 512-new-token cap. Consumption at 512 total tokens produced 7,203 active and 989 padding positions, truncating 715 tokens across twelve records. All four EOS markers were retained. Non-thinking, greedy generation and the pinned original BF16 model revision match the parent preparation protocol. These search inputs are not a final held-out confirmation set.

171 CPU tests passed. All 2,034 parent files, 39 main source files and 108 frozen diagnostic files passed the independent post-failure preservation audit. The frozen driver remains unchanged. No GPU retry, training or automatic escalation ran; both model processes exited. Earlier quality findings remain unchanged.

[Full report](D:/openbonsai/runs/2026-10-26-5090-activation-diagnostic-v1/report.md) · [Failure evidence](D:/openbonsai/runs/2026-10-26-5090-activation-diagnostic-v1/analysis/failure-preservation.json) · [Independent template diagnosis](D:/openbonsai/runs/2026-10-26-5090-activation-diagnostic-v1/analysis/template-failure-independent-review.md) · [Final failure inventory](D:/openbonsai/runs/2026-10-26-5090-activation-diagnostic-v1/final-validation.json)
