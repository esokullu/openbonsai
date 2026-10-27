# UltraChat rotated16×8 quality audit (final)

**The completed representations do not produce useful answers on any of the four prompts.** All four reach EOS, but the outputs are empty, empty, `Write a`, and `#`. Successful EOS termination does not establish quality.

This is the fixed diagnostic subset: 2,048 scored WikiText tokens, 20 ARC-Easy questions, and four held-out prompts. The original full pilot evaluated 16,384 tokens, 100 questions, and ten prompts. The 128×1 paired arm remains pending; its authorization and recipe are unchanged.

| Representation | NLL | PPL | ARC | ΔNLL vs BF16 | ΔARC pp | EOS /4 | Useful complete /4 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original BF16 (historical) | 2.423776811 | 11.288413 | 19/20 | +0.000000000 | +0.0 | 2 | 2 |
| Hard FP32 factors | 8.499409803 | 4911.869014 | 8/20 | +6.075632992 | -55.0 | 4 | 0 |
| FP16 scales, same wrapper GEMM | 8.500026628 | 4914.899710 | 8/20 | +6.076249817 | -55.0 | 4 | 0 |
| Packed, fresh reload, decoded cache | 8.498620065 | 4907.991458 | 8/20 | +6.074843255 | -55.0 | 4 | 0 |

NLL was rechecked from the saved per-block loss sums and scored-token counts; ARC from the saved option likelihoods and answers. Input hashes, IDs/order, masks, tokenizer/chat-template hash, model revision, nonthinking/greedy settings, 128-token cap, EOS IDs, evaluator/helper hashes and environment match the BF16 record. The current original model matches the stored BF16 logit probe exactly. Historical BF16 timing is reused; no new baseline evaluation was launched.

The FP16 stage changes scales/bias precision while retaining wrapper code selection and GEMM. Packed execution additionally changes to the row-chunk runtime, so these numerical differences are reported separately from fresh serialization parity. The exact exported projection/retained payload and cached/uncached parity checks passed.

| Stage | Evaluation seconds | Generated tokens | Peak allocated GiB | Peak reserved GiB |
|---|---:|---:|---:|---:|
| Original BF16 (historical) | 34.278 | 341 | 17.954 | 18.191 |
| Hard FP32 factors | 37.240 | 7 | 19.599 | 19.869 |
| FP16 scales, same wrapper GEMM | 37.677 | 7 | 19.599 | 19.869 |
| Packed, fresh reload, decoded cache | 21.618 | 7 | 19.459 | 19.826 |

Fresh-process packed reload is bitwise exact against the saved packed-runtime reference (maximum absolute logit error 0). FP16 minus hard-FP32 NLL is +0.000616825; packed minus same-GEMM FP16 NLL is -0.001406562. All three representations produce identical generated token IDs and texts for all four prompts.

Current compressed answers total only 7 generated tokens versus 341 in the BF16 record; shorter end-to-end evaluation time is not evidence of inference speedup.
Timing and peaks cover evaluation with its representation already resident, excluding loading, training, export and cache warmup. Research wrappers and deployment caches have different memory scopes; full phase records are in the JSON.

## Full saved answer review

### manual-heldout-00 — instruction

Prompt: Give exactly three numbered tips for keeping a shared kitchen clean. Each tip must be one sentence.

**Original BF16 (historical)** — 61 tokens; EOS=True; cap=False.

Exact saved text (JSON string, preserving newlines/whitespace):

```json
"1. Wash and sanitize all dishes and utensils immediately after use to prevent cross-contamination.\n2. Wipe down countertops and appliances with disinfectant before and after every cooking session.\n3. Take out the trash and empty the recycling bin daily to avoid lingering odors and pests."
```

Judgment: Complete and useful: exactly three numbered, one-sentence kitchen tips.

**Hard FP32 factors** — 1 tokens; EOS=True; cap=False.

Exact saved text (JSON string, preserving newlines/whitespace):

```json
""
```

Judgment: Empty response; provides none of the three requested numbered kitchen tips.

Exact generated IDs: `[248046]`.

**FP16 scales, same wrapper GEMM** — 1 tokens; EOS=True; cap=False.

Exact saved text (JSON string, preserving newlines/whitespace):

```json
""
```

Judgment: Empty response; provides none of the three requested numbered kitchen tips.

Exact generated IDs: `[248046]`.

**Packed, fresh reload, decoded cache** — 1 tokens; EOS=True; cap=False.

Exact saved text (JSON string, preserving newlines/whitespace):

```json
""
```

Judgment: Empty response; provides none of the three requested numbered kitchen tips.

Exact generated IDs: `[248046]`.

### manual-heldout-02 — reasoning

Prompt: A farmer has chickens and rabbits. There are 10 heads and 28 legs. How many of each animal are there? Explain briefly.

**Original BF16 (historical)** — 128 tokens; EOS=False; cap=True.

Exact saved text (JSON string, preserving newlines/whitespace):

```json
"To solve this, we can use a simple logical deduction based on the number of legs:\n\n1.  **Assume all 10 animals are chickens.** Since chickens have 2 legs, this would result in $10 \\times 2 = 20$ legs.\n2.  **Compare with the actual leg count.** The farmer has 28 legs, which is $28 - 20 = 8$ more legs than our assumption.\n3.  **Account for the difference.** Each time we swap a chicken for a rabbit, the leg count increases by 2 (since rabbits have"
```

Judgment: Relevant partial reasoning, cut at 128 tokens before stating the answer (6 chickens and 4 rabbits).

**Hard FP32 factors** — 1 tokens; EOS=True; cap=False.

Exact saved text (JSON string, preserving newlines/whitespace):

```json
""
```

Judgment: Empty response; no animal counts or explanation.

Exact generated IDs: `[248046]`.

**FP16 scales, same wrapper GEMM** — 1 tokens; EOS=True; cap=False.

Exact saved text (JSON string, preserving newlines/whitespace):

```json
""
```

Judgment: Empty response; no animal counts or explanation.

Exact generated IDs: `[248046]`.

**Packed, fresh reload, decoded cache** — 1 tokens; EOS=True; cap=False.

Exact saved text (JSON string, preserving newlines/whitespace):

```json
""
```

Judgment: Empty response; no animal counts or explanation.

Exact generated IDs: `[248046]`.

### manual-heldout-04 — coding

Prompt: Write a Python function is_balanced(text) that checks whether parentheses are balanced, ignoring other characters. Include an empty-string case.

**Original BF16 (historical)** — 128 tokens; EOS=False; cap=True.

Exact saved text (JSON string, preserving newlines/whitespace):

```json
"```python\ndef is_balanced(text: str) -> bool:\n    \"\"\"\n    Check whether parentheses in the given text are balanced.\n    Only '(' and ')' are considered; all other characters are ignored.\n    An empty string is considered balanced.\n\n    Examples:\n        is_balanced(\"()\") -> True\n        is_balanced(\"(]\") -> False\n        is_balanced(\"((()))\") -> True\n        is_balanced(\"(()\") -> False\n        is_balanced(\"\") -> True\n        is_balanced(\"a(b)c\") -> True\n    \"\"\"\n    count = 0\n   "
```

Judgment: Relevant function signature, documentation and examples, cut at 128 tokens before the checking logic; not a usable function.

**Hard FP32 factors** — 3 tokens; EOS=True; cap=False.

Exact saved text (JSON string, preserving newlines/whitespace):

```json
"Write a"
```

Judgment: Fragment of the request; no function, logic, or empty-string case.

Exact generated IDs: `[7734, 264, 248046]`.

**FP16 scales, same wrapper GEMM** — 3 tokens; EOS=True; cap=False.

Exact saved text (JSON string, preserving newlines/whitespace):

```json
"Write a"
```

Judgment: Fragment of the request; no function, logic, or empty-string case.

Exact generated IDs: `[7734, 264, 248046]`.

**Packed, fresh reload, decoded cache** — 3 tokens; EOS=True; cap=False.

Exact saved text (JSON string, preserving newlines/whitespace):

```json
"Write a"
```

Judgment: Fragment of the request; no function, logic, or empty-string case.

Exact generated IDs: `[7734, 264, 248046]`.

### manual-heldout-07 — turkish

Prompt: Aşağıdaki cümleyi daha resmi bir Türkçeyle yeniden yaz: Yarın gelemiyorum, toplantıyı başka güne alsak olur mu?

**Original BF16 (historical)** — 24 tokens; EOS=True; cap=False.

Exact saved text (JSON string, preserving newlines/whitespace):

```json
"Yarın katılım sağlayamayacağım; toplantıyı başka bir tarihe erteleyebilir miyiz?"
```

Judgment: Complete, relevant formal Turkish rewrite preserving the request.

**Hard FP32 factors** — 2 tokens; EOS=True; cap=False.

Exact saved text (JSON string, preserving newlines/whitespace):

```json
"#"
```

Judgment: A hash character; no Turkish sentence or formal rewrite.

Exact generated IDs: `[2, 248046]`.

**FP16 scales, same wrapper GEMM** — 2 tokens; EOS=True; cap=False.

Exact saved text (JSON string, preserving newlines/whitespace):

```json
"#"
```

Judgment: A hash character; no Turkish sentence or formal rewrite.

Exact generated IDs: `[2, 248046]`.

**Packed, fresh reload, decoded cache** — 2 tokens; EOS=True; cap=False.

Exact saved text (JSON string, preserving newlines/whitespace):

```json
"#"
```

Judgment: A hash character; no Turkish sentence or formal rewrite.

Exact generated IDs: `[2, 248046]`.

The two completed useful BF16 answers are the kitchen instructions and formal Turkish rewrite. BF16 reasoning and coding are relevant but truncated at 128 tokens; they are not counted as complete useful answers. This small qualitative review supplements NLL and ARC and makes no population-wide claim.
