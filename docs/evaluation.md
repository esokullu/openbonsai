# Evaluation

`openbonsai eval --run DIR --from SOURCE --suite dev,extended,compare,teacher,gsm8k,humaneval` writes one JSON file per suite to `DIR/eval/<source>.<suite>.json` (the reasoning suites: one per thinking mode, `DIR/eval/<source>.<suite>.think-<off|on>.json`).

| Suite | Data | What it reports |
|---|---|---|
| `dev` | Frozen Qwen3.5 manifest: 2,048 WikiText-2 test tokens, 20 ARC-Easy questions, 4 generations | `nll`, `arc_correct`, generations |
| `extended` | 16,384 WikiText-2 test tokens, 100 ARC-Easy questions, 16 held-out chat traces | `wiki16k_nll`, `arc100`, `chat_nll` |
| `compare` | `eval.compare`: `n` held-out blocks of `seq_len` tokens (WikiText-2 test by default), scored against the model's own BF16 original | NLL of both, `top1_agreement`, EAR and KL (below) |
| `teacher` | The first `eval.teacher.rows` rows of this run's distillation teacher shards (stored BF16 top-k) | Top-k EAR bounds and KL only; no BF16 model is loaded |
| `gsm8k` | 100 frozen GSM8K test problems, greedy chat generation | `score` (e.g. `71/100`), `hit_cap`, per-problem records |
| `humaneval` | 50 frozen HumanEval problems, greedy chat generation, tests run in a subprocess | `score` (pass@1, e.g. `30/50`), passed/failed/timeout, `hit_cap`, per-problem records |

`dev` and `extended` only accept the Qwen3.5 tokenizer. `compare` works for any model, so use it for your own fine-tune. The teacher shards are the data distillation trains on, so `teacher` is in-sample for a distilled model; use `compare` for held-out numbers. `gsm8k` and `humaneval` work for any model with a chat template (next section).

## Reasoning suites (`gsm8k`, `humaneval`)

```
openbonsai eval --run runs/x --from distill --suite gsm8k,humaneval --set 'eval.reasoning.thinking=[off,on]'
openbonsai eval --run runs/x --from bf16 --suite gsm8k --set eval.reasoning.gsm8k.n=20     # a quick look
openbonsai eval --run runs/x --from model.gguf --suite humaneval --set eval.reasoning.thinking=on \
    --set eval.reasoning.max_new_tokens_thinking=2048     # shorter thinking
```

**Data.** Fixed TEST subsets are frozen in `openbonsai/eval/assets/reasoning/`: 100 GSM8K (`main` test) and 50 HumanEval problems, picked by a seeded SHA256-rank selector from datasets pinned by repo and revision (see that folder's README). Each manifest records the dataset, revision, file SHA256, seed, selector rule, row ids and its own content SHA256; the loader checks them. Evaluation needs no download. `eval.reasoning.<suite>.n` takes the first `n` of a subset, which is what the selector picks for that `n`. A larger `n` reads the dataset from the Hugging Face cache or hub (HF_HUB_OFFLINE works), or from a local parquet given as `eval.reasoning.<suite>.path`. Those extra problems are not frozen.

These are test data. The calibrate, distillation-teacher and traces stages refuse sources that name these test splits, local copies of the test files, and any data stream or trace prompt that contains a test problem: every problem of both splits (1,319 GSM8K, 164 HumanEval) when they are in the Hugging Face cache, else the frozen subsets. Before it scores `quantize`, `endpoints` or `distill`, eval decodes the run's stored `calib/tokens.npy` (and, for `distill`, `distill/teacher/input_ids.npy`) in build order and refuses them if they contain a problem it is about to score. That covers data built before the guard existed, or without the full splits (a larger `n`). A clean check is remembered in `eval/heldout-checked.json`. Future math or code calibration must come from training splits (for example `openai/gsm8k` `main/train-00000-of-00001.parquet`).

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

## Comparing with published numbers

SLQ's reported EAR is not comparable with `ear_full`, or with either top-k bound:
- **Tokens.** It uses only BF16's top-10 tokens, renormalized over those 10. Any mass either model puts outside them is ignored. A model can score near 1 there while its full-vocabulary EAR is much lower, so it is neither a lower nor an upper bound on `ear_full`.
- **Data.** It is measured on calibration data, the data the quantizer was fit on. Our `compare` blocks are held out.

When comparing against another method, report `ear_full` (or the `ear_top64_lb`–`ear_top64_ub` interval) on held-out text, and state the data and the vocabulary coverage of their number.
