# Evaluation

`openbonsai eval --run DIR --from SOURCE --suite dev,extended,compare,teacher,gsm8k,humaneval` writes one JSON file per suite to `DIR/eval/<source>.<suite>.json` (the reasoning suites: one per thinking mode, `DIR/eval/<source>.<suite>.think-<off|on>.json`).

`SOURCE` is a stage of the run (`bf16`, `quantize`, `endpoints`, `distill`) or a file: a run state, a legacy research `.pt`, a latent checkpoint's `ternary.safetensors`, or any PTQ1_0 GGUF, ours or the official Ternary Bonsai. For a GGUF, `--run` may be left out (it defaults to `<file>.eval`); the config or `--model` only supplies the model's configuration, its tokenizer and the BF16 reference:

```
openbonsai eval --config examples/qwen3_5-9b.yaml --from Ternary-Bonsai-2-9B-PTQ1_0.gguf --suite dev,extended,compare
```

When you report a compressed model, report it next to its own BF16 original on the same suites, and say which data it was compressed with (see [Keeping test data out](#keeping-test-data-out)).

| Suite | Data | What it reports |
|---|---|---|
| `dev` | Frozen Qwen3.5 manifest: 2,048 WikiText-2 test tokens, 20 ARC-Easy questions, 4 generations | `nll`, `arc_correct`, generations |
| `extended` | 16,384 WikiText-2 test tokens, 100 ARC-Easy questions, 16 held-out chat traces | `wiki16k_nll`, `arc100`, `chat_nll` |
| `compare` | `eval.compare`: `n` held-out blocks of `seq_len` tokens (WikiText-2 test by default), scored against the model's own BF16 original | NLL of both, `top1_agreement`, EAR and KL (below) |
| `teacher` | The first `eval.teacher.rows` rows of this run's distillation teacher shards (stored BF16 top-k) | Top-k EAR bounds and KL only; no BF16 model is loaded |
| `gsm8k` | 100 frozen GSM8K test problems, greedy chat generation | `score` (correct / 100), `hit_cap`, per-problem records |
| `humaneval` | 50 frozen HumanEval problems, greedy chat generation, tests run in a subprocess | `score` (pass@1, passed / 50), passed/failed/timeout, `hit_cap`, per-problem records |

`dev` and `extended` only accept the Qwen3.5 tokenizer. `compare` works for any model, so use it for your own fine-tune. The teacher shards are the data distillation trains on, so `teacher` is in-sample for a distilled model; use `compare` for held-out numbers. `gsm8k` and `humaneval` work for any model with a chat template (next section).

## Frozen inputs

Every scored input is pinned, so numbers from different runs, machines and backends can be compared:

| Suite | Inputs | Pinned by |
|---|---|---|
| `dev` | `openbonsai/eval/assets/bonsai-dev/frozen-inputs.json`: 4 WikiText-2 test windows of 513 tokens (2,048 scored tokens), 20 ARC-Easy validation questions as A/B/C/D letter choices, 4 generation prompts (greedy, 128 tokens) | file SHA256 `9936a538...`, canonical `frozen_sha256` `8021d9cc...`, tokenizer Qwen/Qwen3.5-9B at `c202236` |
| `extended` | `legacy-eval-inputs.json`: 32 WikiText-2 test windows of 513 tokens (16,384 scored tokens) and 100 ARC-Easy letter questions; `chat-traces16.jsonl`: 16 greedy Qwen3.5-9B answers to UltraChat first-turn prompts | SHA256 of both files in `openbonsai/eval/suites.py` |
| `gsm8k`, `humaneval` | `openbonsai/eval/assets/reasoning/`: 100 GSM8K and 50 HumanEval test problems, chosen by a seeded SHA256-rank selector | file SHA256 and `content_sha256`; dataset repo, revision and file SHA256 |
| `compare` | `eval.compare.n` blocks of `seq_len` tokens (default 8 × 512, seed 7) from WikiText-2 *test* at the pinned revision, or your `eval.compare.sources` | the config hash; built at run time, not a frozen file |

- **Checked on load.** Every frozen file is checked against its hashes when it is loaded. The `dev` and `extended` token IDs must match the tokenizer exactly, raw pieces included, or scoring is refused.
- **Windows.** `.gitattributes` keeps the bytes unchanged on Windows checkouts.
- **Probabilities.** The `dev` suite takes FP32 logits and normalizes them in FP64. ARC is scored by the likelihood of the answer letter, not of the answer text.
- **Not a fresh held-out set.** The `dev` inputs are reused development data (role `search_validation`): we looked at them throughout development.
- **Details.** `openbonsai/eval/assets/bonsai-dev/README.md` and `openbonsai/eval/assets/reasoning/README.md`.

## Keeping test data out

The pipeline fits the model on three kinds of data: the calibration blocks (quantize, endpoints), the distillation teacher's data, and the prompts the traces answer. Here is what keeps evaluation data out of them, and what does not:

- **GSM8K and HumanEval test problems: enforced.** Every data source and every trace prompt is checked. A source that names a test split, a local copy of a test file, or text that contains a test problem is refused, and the stage stops. Eval checks the stored tokens again before it scores a stage (details in the next section). Your own `kind: jsonl` data goes through the same check.
- **WikiText-2: separated by split, not enforced.** The default calibration and distillation data read WikiText-2 *train*; `dev`, `extended` and `compare` score WikiText-2 *test*. The articles differ, but the source is the same, so the WikiText numbers of a model compressed with the default data are partly in-domain. Say so when you report them. To remove the effect, leave WikiText out of `calibrate.sources` and `distill.teacher.sources` and rerun.
- **ARC.** Not in any default data source.
- **Chat traces.** `chat-traces16.jsonl` holds the answers to 16 UltraChat prompts from rows below 5,000. The calibration and distillation UltraChat data start at row 6,000. The traces stage is different: the example config samples its 160 prompts from rows 0 to 5,999, and that sample includes 2 of the 16 (source rows 859 and 3307). A run with those defaults is therefore partly in-sample on `chat_nll`. The measured 9B results did not have this problem: their traces came from a fixed set disjoint from the 16. To avoid it, pass `--set 'traces.prompts.rows=[5000,6000]'`. Either way, the default data contains other answers generated by the same model, so `chat_nll` is in-distribution by design.
- **`teacher`.** In-sample for a distilled model, by definition.
- **Anything else is up to you.** No other check exists. Keep your own evaluation sets out of your calibration data.

## Reasoning suites (`gsm8k`, `humaneval`)

```
openbonsai eval --run runs/x --from distill --suite gsm8k,humaneval --set 'eval.reasoning.thinking=[off,on]'
openbonsai eval --run runs/x --from bf16 --suite gsm8k --set eval.reasoning.gsm8k.n=20     # a quick look
openbonsai eval --run runs/x --from model.gguf --suite humaneval --set eval.reasoning.thinking=on \
    --set eval.reasoning.max_new_tokens_thinking=2048     # shorter thinking
```

**Data.** Fixed TEST subsets are frozen in `openbonsai/eval/assets/reasoning/`: 100 GSM8K (`main` test) and 50 HumanEval problems, picked by a seeded SHA256-rank selector from datasets pinned by repo and revision (see that folder's README). Each manifest records the dataset, revision, file SHA256, seed, selector rule, row ids and its own content SHA256; the loader checks them. Evaluation needs no download. `eval.reasoning.<suite>.n` takes the first `n` of a subset, which is what the selector picks for that `n`. A larger `n` reads the dataset from the Hugging Face cache or hub (HF_HUB_OFFLINE works), or from a local parquet given as `eval.reasoning.<suite>.path`. Those extra problems are not frozen.

These are test data (see [Keeping test data out](#keeping-test-data-out)). The calibrate, distillation-teacher and traces stages refuse sources that name these test splits, local copies of the test files, and any data stream or trace prompt that contains a test problem: every problem of both splits (1,319 GSM8K, 164 HumanEval) when they are in the Hugging Face cache, else the frozen subsets. Before it scores `quantize`, `endpoints` or `distill`, eval decodes the run's stored `calib/tokens.npy` (and, for `distill`, `distill/teacher/input_ids.npy`) in build order and refuses them if they contain a problem it is about to score. That covers data built before the guard existed, or without the full splits (a larger `n`). A clean check is remembered in `eval/heldout-checked.json`. Future math or code calibration must come from training splits (for example `openai/gsm8k` `main/train-00000-of-00001.parquet`).

**Generation.** Decoding is plain greedy (argmax) for every model. A checkpoint's `generation_config.json` is not applied: `generate` would otherwise take its `repetition_penalty`, `no_repeat_ngram_size`, `bad_words_ids`, `suppress_tokens` or `min_length`, which a model built from a GGUF does not have, so a BF16 and a GGUF run of one model would decode differently. Prompts use the model's own chat template (`enable_thinking` for Qwen3.5). A template that prints today's date (Llama 3.2, Mistral Small) sees a fixed 26 Jul 2024, so prompts and cache keys do not change from day to day. Generation stops at the adapter's EOS ids. Prompts run in left-padded batches of `eval.reasoning.batch` (default 16) with the KV cache. A thinking mode the template has no switch for (`on` for Llama, Mistral, Qwen2) fails before the model loads, and `openbonsai run` checks it before the first stage.

**Reproducibility.** Scores reproduce at the same `batch`. In BF16, left padding changes the prefill arithmetic (padded and unpadded logits differ by up to about 6e-3), so a row's greedy tokens can depend on which rows share its batch. The generation cache keys each batch by its prompts, the weights' identity, the decoding, and the runtime: torch and transformers versions, the GPU, and whether transformers bound the FLA / causal-conv1d kernels or the torch paths. Each result file records them (`decoding`, `runtime`). To measure the batch effect on your GPU, set `eval.reasoning.batch_check: N`: the `N` most padded rows (shortest prompts) of the first newly generated batch are generated again one at a time, and the result's `batch_check` says which ones match (the first differing token otherwise).

| `eval.reasoning` | Default | Meaning |
|---|---|---|
| `thinking` | `[off]` | `off`, `on` or `[off, on]`; one result file per mode |
| `max_new_tokens` | 512 | cap with thinking off |
| `max_new_tokens_thinking` | 4096 | cap with thinking on (lower it if thinking is too slow) |
| `batch` | 16 | prompts per generate call |
| `batch_check` | 0 | regenerate this many rows of the first new batch alone and compare (above) |
| `dense` | true | a ternary model (distill, GGUF, `--install ternary`) generates with its dense dequantized weights, the same weights `--install dense` writes; much faster than the ternary modules |
| `cache` | true | keep generations per batch in `eval/generations/`, keyed by the weights' identity; an interrupted run resumes and a rerun rescores without generating |
| `gsm8k.n`, `humaneval.n` | 100, 50 | subset sizes |
| `humaneval.timeout`, `humaneval.workers` | 10 s, 4 | per-program timeout, programs run at once |
| `humaneval.memory_mb` | 2048 | memory limit of each program and the processes it starts (Linux, Windows) |

The log shows tokens/s, decode steps/s and an ETA after every batch. A row that reaches the cap counts in `hit_cap`. With thinking on, the answer is the text after the last `</think>`. A generation that never closes its thinking has no answer and counts in `unfinished_thinking`.

**Scoring.**
- GSM8K: exact match on the final number. The answer is the last `\boxed{...}` holding a number, else the number after the last `####`, else the answer after the last "answer is" or "answer:", else the last number. For the marker, the clause up to the end of its sentence or line counts: the number after its last `=` when it states an equation (`Answer: 16 - 3 - 4 = 9` gives 9), else its first number (`The answer is 18 (6 boxes of 3)` gives 18). Commas, `$`, a `+` sign and trailing zeros are normalized (`1,080.00` = `1080`). A run of more than 64 digits (a looping generation) is not a number, so it scores wrong instead of stopping the suite. Each record states the method used (`methods` counts them).
- HumanEval pass@1: the code comes from the last fenced block that defines the entry point, else the first fenced block, else the raw text. Code indented as a whole (a fence under a markdown list item) is dedented. The problem's prompt goes first, so a full function keeps the prompt's imports and helpers, and a continued body completes the stub. The model's own top-level statements are then cleaned as EvalPlus does: only imports, function and class definitions, and the module-level assignments those definitions read are kept. Its example calls, `print`s, its own asserts, `input()` and `if __name__ == "__main__":` blocks never run, and `from __future__` imports move to the top. Each record counts what was dropped (`dropped_statements`); code that does not parse is run as written. The official `test` and `check(entry_point)` follow. Statuses are `passed`, `failed` and `timeout`.

**HumanEval execution is not a security sandbox.** Each program runs in a separate `python -I -X utf8` process (never in the evaluator's process; UTF-8 output on every OS), in a fresh temporary directory, with stdin closed, a reduced environment and no GPU. Its output goes to files, not memory, and it has limits: the timeout, 8 MiB of output, and `humaneval.memory_mb` of memory (address space through `prlimit` on Linux; committed memory of the whole process tree through a Job Object on Windows; no memory limit on macOS). When the program ends or breaks a limit, it and every process it started are killed: its process group on POSIX, its Job Object on Windows (the program starts suspended and joins the job before it runs). A pass needs exit code 0 and an end marker printed after the test, so code that exits early fails. That protects the evaluation from crashes, hangs, output floods, runaway allocations, leftover processes and `exit()`. It does not stop model code from touching your files or the network. The suite has no network intent, but nothing enforces it, so run untrusted models in a container or VM.

**Cost.** HF `generate` for Qwen3.5-9B BF16 on an RTX 5090 decoded 26.5 tokens/s at batch 1 (AYOT traces, 512-token cap, no FLA kernels). A batch of 16 costs about the same per decode step, but every batch runs until its longest row stops. Expect a few minutes for both suites with thinking off, and roughly 30–60 minutes with thinking on at 4096 tokens, where most batches run to the cap. Each batch logs its steps/s; total time is about the number of batches × the cap ÷ steps/s.

## Teacher-vs-student metrics

Here `p` is the BF16 original's next-token distribution and `q` the compressed model's, at one position. Each metric is computed per position in FP64 and averaged over all scored positions.

| Key | Definition | Needs |
|---|---|---|
| `ear_full` | Expected acceptance rate: `sum_v min(p(v), q(v))` over the whole vocabulary | BF16 logits at every position |
| `kl_full` | `KL(p ‖ q)` over the whole vocabulary | BF16 logits at every position |
| `ear_top64_lb` | `sum` of `min(p_k, q_k)` over the BF16 top-64 tokens | Stored BF16 top-64 |
| `ear_top64_ub` | `ear_top64_lb + min(p_tail, q_tail)`, where tail means the mass outside the top-64 | Stored BF16 top-64 |
| `kl_top64` | KL over the top-64 plus one tail bucket (the distillation objective's form) | Stored BF16 top-64 |
| `top1_agreement` | Fraction of positions where both models have the same argmax | — |

The `64` in the key is `eval.compare.topk` (or the shards' `distill.teacher.topk`). Each result file also states `topk` and whether full-vocabulary metrics were computed (`full_vocab`).

**What EAR means.** Speculative sampling drafts a token from `q` and accepts it with probability `min(1, p/q)`. One drafted token is accepted with probability `sum_v min(p(v), q(v))`, which equals `1 - TV(p, q)`. So `ear_full` is the per-token acceptance rate at temperature 1 when the compressed model drafts for the BF16 model. It is symmetric in `p` and `q`, and it is 1 only for identical distributions. Under greedy decoding the analogous quantity is `top1_agreement`.

**Bounds.** The top-k metrics bracket or bound the full ones:
- `ear_top64_lb <= ear_full <= ear_top64_ub`
- `kl_top64 <= kl_full`, because merging outcomes never increases KL.

Both are exact when the stored log-probabilities are exact. The `compare` reference stores FP64 values. Distillation's teacher shards store fp16, so for the `teacher` suite the bounds hold only up to that rounding. If the gap between `lb` and `ub` is small, the cheap numbers are as informative as the full ones.

**Cost.** `compare` computes the full metrics when `eval.compare.full: true`, which is the default. The BF16 reference pass then keeps its raw logits on disk in `eval/compare-reference-<key>.logits/`, one file per block, in the model's dtype. The size is `n × (seq_len − 1) × vocab` values: about 2 GB for Qwen3.5-9B (vocabulary 248,320) at the defaults `n: 8`, `seq_len: 512`. Later evaluations of other sources (quantize, endpoints, distill, a GGUF) reuse the cache without loading the BF16 model again. Positions are processed 64 at a time, so peak memory stays at a few `[64, vocab]` FP64 slabs. Set `eval.compare.full: false` to keep only the top-k metrics.

## Scoring in Prism's runtime

`openbonsai eval --from model.gguf` builds the model from the file in PyTorch: the same codes, scales and F32 tensors, computed with PyTorch's arithmetic. To score the file in Prism's own llama.cpp runtime instead:

- **The `dev` suite, on exactly the same inputs.** `tools/prism/evaluate_prism_manifest.py` drives a small C++ worker built against Prism at the pinned commit. Build it with `python tools/prism/build_prism_eval.py --source <Prism checkout> --build-dir <dir>`. The runner applies the same token-ID and raw-piece checks and the same FP64 normalization:
  ```
  python tools/prism/evaluate_prism_manifest.py --worker <prism-eval-worker> --model model.gguf --output out/dev
  ```
  The official Ternary Bonsai 2 9B scored dev NLL 3.192 and ARC 16/20 this way, against 3.191 and 17/20 in PyTorch.
- **Perplexity.** `llama-perplexity -m model.gguf -f wiki.test.raw -c 512 --chunks 8`, with the WikiText-2 test set as raw text. This uses llama.cpp's own tokenization and windows, so it is not comparable with the `dev` NLL. Compare files with each other under the same command.
- **Full log-probabilities.** `python tools/prism/compare_logits.py --bin <Prism build>/bin --gguf model.gguf --text text.txt` reads back llama-perplexity's full-vocabulary log-probabilities (`--kl-divergence-base`). Use it to compare the runtime with the PyTorch model of the same state. `tests/test_dense.py` does this for tiny Llama and Qwen3 GGUFs when `OPENBONSAI_PRISM_BIN` points to a Prism build.

Prism's CPU matrix products quantize the activations to 8 bits (Q8_0 operands), so its numbers differ slightly from PyTorch's.

## Comparing with published numbers

Before you put another method's numbers next to ours, check what they measure.

**EAR.** SLQ's reported EAR is not comparable with `ear_full`, or with either top-k bound:
- **Tokens.** It uses only BF16's top-10 tokens, renormalized over those 10. Any mass either model puts outside them is ignored. A model can score near 1 there while its full-vocabulary EAR is much lower, so it is neither a lower nor an upper bound on `ear_full`.
- **Data.** It is measured on calibration data, the data the quantizer was fit on. Our `compare` blocks are held out.

When comparing against another method, report `ear_full` (or the `ear_top64_lb`–`ear_top64_ub` interval) on held-out text, and state the data and the vocabulary coverage of their number.

**Perplexity and NLL.** They depend on the tokenizer, the window length, the stride, whether the first tokens of each window are scored, and the split. Our `dev` NLL scores positions 1–512 of four independent 513-token windows. Numbers from another harness are comparable only after you rescore both models in one harness.
