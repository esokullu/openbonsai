# Diversity control stopped at the source-data gate

**No generation or training was started.** The original calibration source is a hand-written list of exactly 16 prompts. There is no documented same-source pool supplying 112 additional records, so the explicit data gate fails. The proposed 128×1 model must wait for main-chat review of a changed data protocol.

## Provenance and available pool

- Source: [scripts/experiment_data.py:10](D:/openbonsai/scripts/experiment_data.py:10), literal `CALIBRATION` list. It has no dataset identifier, external revision, split, sampling procedure or larger source reservoir. Its file hash matches the historical preservation record.
- Frozen prompts: [calibration-prompts.jsonl](D:/openbonsai/runs/2026-10-26-5090-pilot/calibration-prompts.jsonl), IDs `cal-00` through `cal-15`. All 16 are distinct. Content-derived categories are **6 math, 6 coding and 4 general instruction**; the source did not store category labels.
- `calibration-prompts.first-attempt.jsonl` is byte-identical, SHA256 **`b6f14896b4f5b4282204236cf5e5755669b6f11979bf7cef77e1feb185843247`**. It provides zero new examples. Independent read-only inspection found the same 16-entry constant in 13 source/archive copies. Repository `examples/prompts.jsonl` contains two different generic examples, copied across checkouts; these are not a larger version of this calibration source.
- Consumed traces: [ayot.jsonl](D:/openbonsai/runs/2026-10-26-5090-pilot/ayot.jsonl), SHA256 **`09df0a26800a20e46288b5a775075c91dc5c0e98bc0bf81ac7906f304a0d820f`**. Each trace's prompt exactly matches the ordered source list. Trace records do not carry `cal-*` IDs, so linkage uses exact prompt text and position. The sealed rotation run's copy is identical.

**Available justified additional same-source prompts: 0. Required: 112.** Authoring new examples or taking an unrelated dataset would change the source definition. Neither has been done.

## Exact preparation and consumption recipe

The [generation manifest](D:/openbonsai/runs/2026-10-26-5090-pilot/ayot.jsonl.manifest.json), [measured preparation](D:/openbonsai/runs/2026-10-26-5090-pilot/prepare-metrics.json), and preparation wrapper establish:

| Setting | Observed original value |
|---|---|
| Model / immutable revision | `Qwen/Qwen3.5-9B` / `c202236235762e1c871ad0ccb60c8ee5ba337b9a` |
| Loader / precision | Native `AutoModelForImageTextToText`, CUDA BF16 |
| Template | Official pinned tokenizer; one user message; `add_generation_prompt=True`; thinking disabled |
| Decoding | Greedy, `do_sample=False`, one beam, one return sequence, seed 5090 |
| Temperature / top-p | Not passed during greedy generation |
| Limits | 512 maximum new tokens; no prompt-token limit |
| Stopping | EOS union `[248044,248046]`; tokenizer padding ID |
| Preparation attention / KV cache | **SDPA / `use_cache=True`**, recorded actual model defaults |
| Consumed sequence | Prompt plus generated tokens; right truncation/padding to 512; active mask 1, padding mask 0 |

Preparation's SDPA/cached settings differ from the common eager/no-KV evaluation settings. Any approved extension must reproduce the **preparation** recipe, not accidentally use the diagnostic decoding implementation.

The 16 traces contain **557 prompt + 4,773 generated = 5,330 tokens**. Thirteen end with EOS; three (`cal-03`, `cal-07`, `cal-08`) hit the new-token cap. Training consumes **5,234 active positions**, adds **2,958 padding positions**, and right-truncates **96 positions from those three records**. Prompt positions are included in the reconstruction mask. [Consumed-record audit](D:/openbonsai/runs/2026-10-26-5090-diversity-gate/data-gate.json) records each ID's lengths, hash and EOS/cap flags without loading the model or accessing benchmark answer files.

## Concrete proposed data plan — not authorized or executed

One possible revised protocol is an explicitly **new authored extension**. It can approximate the existing topics and style, but cannot be represented as sampling from a previously documented common distribution.

1. Retain the 16 original traces byte-for-byte. Define new source version `authored-calibration-extension-v1`, with author/reviewer provenance and unique IDs. Use only the original calibration prompts as the authoring reference; provide the authoring/review process no evaluation inputs, answers or model-quality results. Freeze task/difficulty/style criteria before writing candidates. Reject duplicates, paraphrases and mere numeric substitutions as insufficient diversity.
2. Predeclare at most **160 new candidate prompts**, with **60 math / 60 coding / 40 general instruction**. Freeze their text and hashes before generating any trace. Select exactly **112 new records: 42 math / 42 coding / 28 general**, producing total categories **48 / 48 / 32**. Selection may use only source eligibility, duplication checks and the predeclared consumed-length rule below, never answer quality or benchmark information.
3. Reproduce the original generation recipe above, once per distinct candidate, with the same model/tokenizer revision. Generate at most 160 new traces and stop at **90 minutes of GPU preparation**, preserving all completed records. Do not alter temperature, stopping, prompt treatment or output length to meet a budget.
4. Predeclare consumed-length bins **1–128, 129–256, 257–384, 385–512**. The original 16 have counts **3 / 2 / 5 / 6**. An eightfold marginal target is **24 / 16 / 40 / 48** across 128; the new 112 therefore require **21 / 14 / 35 / 42**. Category and length quotas, exact duplicate checks, and aggregate active-token tolerance must all pass. A deterministic selection rule may minimize distance to the active-token target within those quotas, resolving ties by frozen candidate IDs. This algorithm must be fixed before generation, with no semantic quality filtering.
5. Original 16×8 exposure is **41,872 active positions per window**. The permitted ±5% integer range is **39,779–43,965** for 128×1. The old 16 contribute 5,234; the exact new-112 target is **36,638**, mean **327.125**. If no valid subset exists within the frozen pool, stop for review; do not duplicate, synthesize filler tokens, change truncation or generate beyond the cap. Save both successful and excluded candidate provenance and disclose any token-budget difference and selection bias.
6. Only after main-chat approval of this revised source and a passed frozen-data audit would an isolated copy of the reviewed runner be adapted and tested for 128×1. The requested source-resident window preflight, <2-hour projected training bound, memory margin, one-candidate restriction, exact diagnostics, export/reload and stopping rules would remain mandatory. There would be no endpoint/RCO integration or automatic successor.

If strict common-source attribution is required, the cleaner alternative is an approved versioned **training/calibration-only** corpus, with a predeclared 128-record set and a nested 16-record control sampled from that same source. That would require a newly authorized matched control; comparing only against the current hand-written 16×8 result would confound diversity with source choice. No external dataset was selected or downloaded here.

## Preservation and stop state

All writes are confined to this new `2026-10-26-5090-diversity-gate` directory. The sealed rotation checkout and all prior experiment directories were left untouched. Its final manifest still hashes to **`d723275e9057ed27c8bb7e2b010c0e28ef5fa517ffb95294cac76830ccbb6c2f`**, recording 2,465 files. This gate checked that manifest anchor and the relevant small provenance files; it did not redundantly rehash all 2,465 payloads. The independent source reviewer made no writes.

No model was loaded, prompt authored, trace generated, training process started, source snapshot copied, branch pulled or GPU job queued. Process inspection found no active OpenBonsai Python job; the final RTX 5090 check reported 160 MiB used / 32,028 MiB free. **Main-chat review of the data-source change is the next required step under the explicit instruction to stop if the source is only a small hand-written list.**

[Reproducible CPU provenance auditor](D:/openbonsai/runs/2026-10-26-5090-diversity-gate/audit_source.py) · [Machine-readable gate result and source hashes](D:/openbonsai/runs/2026-10-26-5090-diversity-gate/data-gate.json).
