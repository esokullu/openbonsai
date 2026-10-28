# Qwen3-8B with the reasoning recipe (dense adapter)

Independent project, not affiliated with or endorsed by Prism ML. Bonsai is Prism ML's product.

The second real model through OpenBonsai, and the first through the generic `dense` adapter: `Qwen/Qwen3-8B` (revision `b968826`) with the `dense` adapter and the Qwen3.5-9B reasoning recipe (`examples/qwen3-8b-reasoning.yaml`, identical to this run's `config.yaml`). Every stage ran on one RTX 5090 (32 GB, Windows 11) with the package at `8c150da`: traces, calibrate, quantize, endpoints, distill, export with verification, eval. The adapter needed no change: the export passed all 11 checks, Prism's llama.cpp loaded the file, and scoring the GGUF read back gives the distilled state's numbers exactly.

Against its own BF16 model, the ternary Qwen3-8B keeps most of GSM8K (70/100 thinking off, 41/50 on, against 88/100 and 44/50) and loses about half of HumanEval with thinking off (22/50 against 41/50). With thinking on, HumanEval falls to 6/25 (BF16 22/25): 14 of 25 answers run to the 4,096-token cap, 12 of them looping. In Prism's runtime the GGUF scores perplexity 14.94 on WikiText-2 test, and on the same tokens it matches our PyTorch reference of the same weights: 98.8% top-1 agreement and perplexity 14.92 against 14.93.

Single run, small evals: read the caveats at the end.

## What ran

| | |
|---|---|
| Model | `Qwen/Qwen3-8B` at `b968826d9c46dd6066d109eabc6255188de91218`; `qwen3`, 36 layers, hidden 4,096, MLP 12,288, untied head, vocabulary 151,936 |
| Adapter | `dense` (auto from `model_type`), no code |
| Rotation | widths 4,096 and 12,288, Hadamard block 4,096, group 128, seeded signs per width |
| Traces | the 9B recipe's 1,520 prompts, thinking switches and caps; `traces.batch_tokens` 28,000 (the 9B used 14,000) |
| Calibration | 256 × 2,048 tokens: 64 WikiText-2, 64 UltraChat, 128 traces |
| Endpoints | head and embedding ternary (GPTQ) |
| Refine | off |
| Distillation | group scales only, 1,536 teacher sequences of 2,048 (256 WikiText-2, 640 UltraChat, 640 traces), top-64, 1 epoch, 192 steps |
| Export | Prism PTQ1_0 GGUF (architecture `qwen3`), 1,798,918,592 bytes, 8,190,427,136 ternary weights, 11/11 verification checks passed |
| Eval | `compare` (8 × 512 WikiText-2 test, seed 7, against this model's BF16), GSM8K and HumanEval with the 9B's subsets |
| Prism | llama.cpp fork at `c9c34bc`, CPU build, `llama-perplexity -c 512 --chunks 8 -t 8` |

The launcher (`launch.sh`) held `gpu.lock` from 17:20 on 28 October to 01:40 on 29 October: the run, then the BF16 reasoning baseline, then `compare` on the quantize, endpoints and exported-GGUF states. The CPU checks in Prism (`prism/cpu_checks.sh`) ran while the GPU eval was going. The run read code from a copy of the worktree identical to `8c150da`'s `openbonsai/`.

## Time on one RTX 5090

| Stage | Qwen3-8B | Qwen3.5-9B, same recipe |
|---|---|---|
| traces: 1,520 answers | 3 h 13 min for 90 of 94 batches (1.71M generated tokens, 140 tokens/s); the other 4 came from the first attempt's cache | 2.6 h (1.54M tokens, 167 tokens/s) |
| calibrate | 20 s | <1 min |
| quantize: sequential GPTQ, 252 linears | 19.5 min | 16 min |
| endpoints: head and embedding GPTQ | 2.4 min | 2 min |
| distill, teacher: BF16 top-64 on 1,536 × 2,048 tokens | 5.3 min | 6 min |
| distill, training: 192 steps of ~10.6 s | 34 min | 34 min |
| export and verification (CPU) | 2.7 min | 2 min |
| eval: compare | 35 s | ~1 min (with dev, extended) |
| eval: GSM8K + HumanEval, thinking off | 7.7 min | ~3 min |
| eval: GSM8K + HumanEval, thinking on | 78 min | ~25 min |

Without the restart below, the run takes about 4.5 h to the exported GGUF and about 6 h with its eval. The BF16 baseline (`eval --from bf16 --suite gsm8k,humaneval`) took another 70 min, and `compare` on the three other states 3 min.

Generation is slower than on the 9B. Qwen3-8B attends over the whole context in all 36 layers, while Qwen3.5-9B does so in 8 of its 32 layers (the rest are linear attention), so long thinking answers cost more per step. Its traces also run longer: 921 of the 1,520 answers reach their cap, against 684 for the 9B, mostly in code (OpenCodeInstruct 264 of 280, against 93; MBPP with thinking on 104 of 140, against 9). The Prism CPU checks ran during the first 25 minutes of the eval, and decode fell from about 10 to 8 steps per second: the same thinking-on suites took 62 minutes for the BF16 model, which ran alone. Looping answers add steps too, since a batch runs until its longest answer stops (27,100 decode steps for the ternary model's thinking-on GSM8K, 24,276 for BF16's).

**Disk.** The run directory takes 8.6 GB: quantize 3.5 GB, distill 2.0 GB, the GGUF 1.8 GB, the `compare` reference logits 1.2 GB, endpoints 0.3 GB. The Hugging Face cache holds 16 GB for the BF16 model.

### The VRAM spill and the restart

The first attempt started at 17:20 with the code before `8c150da`. Its thinking-on trace batches (16 rows, up to ~27K tokens in flight) generated at 19-25 tokens/s, with an ETA of 22 h. Windows' GPU counters showed the process at 31 GB of dedicated memory plus 6.6 GB of shared (system) memory, for about 21 GB of weights and KV cache. The KV cache grows a token per step, so PyTorch's caching allocator keeps cutting slightly larger blocks and holds on to the old ones. It releases them only on an out-of-memory error, and on Windows that never comes: the driver pages to system memory instead.

`8c150da` caps the process with `torch.cuda.set_per_process_memory_fraction` during trace and eval generation (as `refine` already did). With a cap set, the allocator frees its cache before it passes the cap, and a batch that really does not fit raises out-of-memory, which the traces stage handles by splitting the batch. The run was stopped at 18:34 and restarted at 18:39, keeping `gpu.lock`. The 4 finished batches came from the trace cache, and the rest ran at 110-160 tokens/s per thinking batch, with shared memory at about 0.2 GB for the remainder of the run. This probably explains why the 9B reasoning run needed `batch_tokens` 14,000: it ran before the cap existed.

## Against its own BF16 model: `compare`

8 × 512 WikiText-2 test tokens (4,088 scored), full vocabulary. NLL in nats per token; EAR is the expected acceptance rate of the model as a speculative draft for its BF16 original.

| State | NLL | ΔNLL | Top-1 agreement | EAR | KL |
|---|---|---|---|---|---|
| BF16 | 2.502 | | | | |
| quantize: decoder ternary, head and embedding BF16 | 3.277 | +0.776 | 0.583 | 0.565 | 1.135 |
| endpoints: all ternary | 3.425 | +0.923 | 0.557 | 0.526 | 1.318 |
| **distill** | **2.819** | **+0.317** | **0.683** | **0.686** | **0.650** |
| the exported GGUF, read back (`--from <gguf>`) | 2.819 | +0.317 | 0.683 | 0.686 | 0.650 |

The GGUF read back matches the distilled state to every printed digit. For the 9B reasoning model (`research/runs/2026-10-27-reasoning-mix-v1/eval/distill.compare.json`), the same suite gives ΔNLL +0.464, top-1 0.696, EAR 0.706 and KL 0.554. The two models tokenize differently (151,936 and 248,320 tokens), so only the rough size compares.

## Reasoning

Greedy decoding with the model's chat template. Thinking off: the frozen 100 GSM8K and 50 HumanEval problems, 512-token cap, batch 16. Thinking on: the first 50 and 25 of the same subsets, 4,096-token cap, batch 8. The 9B rows come from the README (`research/runs/2026-10-27-reasoning-v1/` and `research/runs/2026-10-27-reasoning-mix-v1/eval/`).

| Model | GSM8K off | GSM8K on | HumanEval off | HumanEval on |
|---|---|---|---|---|
| Qwen3-8B BF16 | 88/100 | 44/50 | 41/50 | 22/25 |
| **Qwen3-8B ternary (this run)** | **70/100** | **41/50** | **22/50** | **6/25** |
| Qwen3.5-9B BF16 | 77/100 | 36/50 | 44/50 | 22/25 |
| Qwen3.5-9B ternary, same recipe (the Hugging Face model) | 69/100 | 25/50 | 19/50 | 10/25 |

How often the 4,096-token cap cut an answer off with thinking on, and how many of those cut-off answers loop (their last 300 characters begin with text that already appeared earlier):

| | GSM8K on: capped / looping | HumanEval on: capped / looping |
|---|---|---|
| Qwen3-8B BF16 | 5 / 1 | 3 / 1 |
| Qwen3-8B ternary | 7 / 7 | 14 / 12 |

With thinking off, 3 of 100 GSM8K answers hit the 512-token cap for each model, and 1 of 50 (ternary) and 0 of 50 (BF16) HumanEval answers.

- **Math holds up better than code.** The ternary model keeps 80% of BF16's GSM8K with thinking off and 93% with thinking on, but 54% of HumanEval with thinking off. The 9B showed the same pattern with the same recipe (90%, 69% and 43%).
- **HumanEval with thinking on fails by looping.** 14 of 25 answers run out of tokens mid-thought, 12 of them repeating earlier passages verbatim, and an answer cut off mid-thought has no code to test. All 7 capped GSM8K answers loop too. BF16 loops once in each suite. On the 9B the same recipe left 5 of 25 HumanEval answers capped.
- **The code traces are mostly unfinished.** With thinking on and a 1,536-token cap, Qwen3-8B finished thinking in 48 of 280 OpenCodeInstruct answers and 54 of 140 MBPP answers; the 9B finished 215 and 138. So most of the 8B's code calibration and distillation data is reasoning without a final program. We have not tested whether this causes the code gap; a longer trace cap for the code sources is the obvious experiment.

## In Prism's runtime

**Perplexity.** Upstream `wiki.test.raw`, 8 chunks of 512 tokens, CPU, 8 threads: **14.94 ± 1.01** (`prism/prism_ppl_qwen3-8b_wiki.test.log`). On the copy rebuilt from the Hugging Face parquet, which drops the blank lines (`wiki.test.join.raw`, `prism/make_wiki.py`): 14.35 ± 0.98. We have no BF16 Qwen3-8B perplexity in llama.cpp. The 9B reasoning model scores 13.61 on the same file, but with a different tokenizer, so the two do not compare.

**Runtime parity on the real weights** (`runtime_parity.py`, output in `prism/runtime_parity.json`). `llama-perplexity --kl-divergence-base` writes Prism's full-vocabulary log-probs for the 8 chunks, and an FP32 PyTorch model with the distilled state's dequantized weights scores the same tokens on the CPU. Over the 2,040 scored positions (the second half of each chunk):

| | |
|---|---|
| top-1 agreement | 98.8% |
| perplexity of the actual next tokens | 14.920 (Prism), 14.934 (PyTorch) |
| mean \|log-prob difference\| where Prism's values are in range | 0.023 nats (max 0.59) |
| KL(PyTorch ‖ Prism), floored tokens lumped into one bucket: a lower bound | ≥ 0.0002 nats (max 0.005) |
| KL(PyTorch ‖ Prism), Prism renormalized: biased high | 0.008 nats (max 0.017) |
| KL(Prism renormalized ‖ PyTorch), the estimator of the 9B synthetic check | 0.044 nats (max 0.31) |
| time | Prism 347 s, PyTorch 220 s |

Prism quantizes activations to 8 bits, so the two do not match exactly.

**Why three KL numbers.** Prism stores each log-prob as a uint16 over a 16-nat window below the position's top log-prob. Every token below that window is stored at the floor, so the decoded probabilities sum to slightly more than one, and the plain KL comes out slightly negative here (−0.0005). Lumping the floored tokens of each position into one bucket gives a lower bound, because merging categories cannot raise a KL (up to the uint16 rounding). Renormalizing gives the floored tokens more than their true mass, so that estimate runs high. The third row is the estimator of the 9B check in the README (`research/runs/gguf-test/runtime-parity/ref_vs_llama.py`, mean KL 0.0028). On a confident real model it is dominated by the floor: about 150,000 floored tokens per position, each at e^-16 of the top token's probability. The 9B check ran on the exporter's synthetic test state, whose perplexity is about 13,700: its distributions are nearly flat, so almost nothing is floored. Its 0.0028 and this run's 0.044 therefore measure different things. Compare the two checks by top-1 agreement and perplexity instead.

## What the run exposed, and the fixes

- **VRAM spill during generation on Windows** (above): fixed in `8c150da` (`vram_cap` around trace and eval generation, with tests).
- **Launcher**: the first launcher read a stale `run FAILED` line in the 0.8B queue's log as that queue's end, and was killed before it took the lock. `1e7f819` waits for the log's last line instead.
- **Prism perplexity file**: the README's earlier 9B perplexities had come from the parquet-rebuilt test file. `962ac07` re-measured them on upstream `wiki.test.raw`.
- **Runtime-parity KL**: the plain KL against Prism's floored log-probs can go negative. `runtime_parity.py` reports the bracketing estimates above.
- **The dense adapter itself**: no change needed for Qwen3-8B.

## Caveats

- **One run, small evals.** One HumanEval problem is 2 points with thinking off and 4 with thinking on; one GSM8K problem with thinking on is 2. There is no variance estimate.
- **No sensitivity scan.** What stays in high precision is the dense adapter's default (norms; head and embedding ternary). `docs/adding-a-model.md` recommends a scan for a new model, and this run skipped it.
- **The official comparison is missing.** Prism publishes a first-generation Ternary Bonsai 8B built on Qwen3-8B (`prism-ml/Ternary-Bonsai-8B-gguf`). We have not downloaded or scored it, so this run has no "official model" row.
- **WikiText is partly in-domain**, as for the 9B: calibration and distillation include WikiText-2 train, and `compare` and Prism's perplexity use WikiText-2 test.
- **The thinking-on scores are capped.** BF16 still has 5 of 50 GSM8K and 3 of 25 HumanEval answers cut off at 4,096 tokens.

## Files

- `launch.sh`, `launcher.log`: the GPU job and its log; `run.yaml`: the resolved config.
- `logs/`: one log per stage; `traces/summary.json`: per-source trace counts, caps and timing.
- `eval/`: `compare` for each state, GSM8K and HumanEval for `distill` and `bf16`, with each problem's completion and result. The per-batch generation cache and the cached BF16 reference logits are not committed.
- `export/`: `DONE.json`, the export and verification report, and Prism's perplexity line.
- `prism/`: `cpu_checks.sh` (the Prism perplexity runs and the parity check), their logs, and `runtime_parity.json`. Prism's raw log-prob file (590 MB) is not committed. The `prism_ppl_9b_*.log` files are the 9B perplexities re-measured on the upstream test file, and `make_wiki.py` rebuilds the parquet copy.
- `runtime_parity.py`: the parity check.
