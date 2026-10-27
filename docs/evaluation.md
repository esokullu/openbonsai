# Evaluation

`openbonsai eval --run DIR --from SOURCE --suite dev,extended,compare,teacher` writes one JSON file per suite to `DIR/eval/<source>.<suite>.json`.

| Suite | Data | What it reports |
|---|---|---|
| `dev` | Frozen Qwen3.5 manifest: 2,048 WikiText-2 test tokens, 20 ARC-Easy questions, 4 generations | `nll`, `arc_correct`, generations |
| `extended` | 16,384 WikiText-2 test tokens, 100 ARC-Easy questions, 16 held-out chat traces | `wiki16k_nll`, `arc100`, `chat_nll` |
| `compare` | `eval.compare`: `n` held-out blocks of `seq_len` tokens (WikiText-2 test by default), scored against the model's own BF16 original | NLL of both, `top1_agreement`, EAR and KL (below) |
| `teacher` | The first `eval.teacher.rows` rows of this run's distillation teacher shards (stored BF16 top-k) | Top-k EAR bounds and KL only; no BF16 model is loaded |

`dev` and `extended` only accept the Qwen3.5 tokenizer. `compare` works for any model, so use it for your own fine-tune. The teacher shards are the data distillation trains on, so `teacher` is in-sample for a distilled model; use `compare` for held-out numbers.

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
