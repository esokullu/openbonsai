# Final manual review of paired diagnostic generations

Rotation improves the diagnostic scores, but neither compressed condition recovers useful answers on the four fixed prompts. Every unrotated output repeats ` number`; every rotated output repeats `This` after `Here`. This remains true with hard FP32 factors, pure FP16 scales and the independently reloaded packed artifacts. The baseline gives two complete useful responses and two relevant responses truncated by the same 128-token cap.

Both conditions and all six evaluations completed. This note reviews all 24 compressed outputs plus the four baseline outputs; these are the same four prompts observed repeatedly, not 28 independent benchmark examples. The observations below are manual judgments of specific answers, not a general quality estimate.

| Evaluation | NLL | PPL | ARC correct | Unfinished | Useful complete / relevant truncated / degenerate |
|---|---:|---:|---:|---:|---|
| Original BF16 (historical) | 2.423776811 | 11.288413 | 19/20 | 2/4 | 2 / 2 / 0 |
| unrotated16x8: Hard FP32 factors | 14.439949335 | 1,867,197.748316 | 1/20 | 4/4 | 0 / 0 / 4 |
| unrotated16x8: FP16 scales, same wrapper GEMM | 14.438326641 | 1,864,170.315635 | 1/20 | 4/4 | 0 / 0 / 4 |
| unrotated16x8: Packed, fresh reload, decoded cache | 14.437114680 | 1,861,912.383044 | 1/20 | 4/4 | 0 / 0 / 4 |
| rotated16x8: Hard FP32 factors | 9.257334059 | 10,481.153923 | 8/20 | 4/4 | 0 / 0 / 4 |
| rotated16x8: FP16 scales, same wrapper GEMM | 9.253318076 | 10,439.146194 | 8/20 | 4/4 | 0 / 0 / 4 |
| rotated16x8: Packed, fresh reload, decoded cache | 9.255345656 | 10,460.333869 | 8/20 | 4/4 | 0 / 0 / 4 |

| Final packed rotated minus reference | Delta NLL | Delta PPL | Delta ARC percentage points |
|---|---:|---:|---:|
| unrotated16x8: Packed, fresh reload, decoded cache | -5.181769025 | -1,851,452.049175 | +35.0 |
| Original BF16 (historical) | +6.831568845 | +10,449.045456 | -55.0 |

The final packed rotation comparison adds seven correct ARC selections (8/20 versus 1/20), but remains eleven below BF16 (19/20). ARC ranks fixed answer-label likelihoods, so its improvement does not establish useful free-form generation. The remaining NLL gap and actual repeated answers prevent a quality-recovery claim.

## Exact generated sequences across representations

For every unrotated prompt in each representation, the full sequence is `[1324] * 128` and the recorded text is `" number" * 128`, with the leading spaces preserved. All 12 unrotated outputs reach the cap without EOS.

For every rotated prompt in each representation, the full sequence is `[8160, 271] + [1919, 271] * 63` and the recorded text is `"Here\n\n" + "This\n\n" * 63`. All 12 rotated outputs reach the cap without EOS. Token 8160 appears once, 271 appears 64 times and 1919 appears 63 times. Word/newline alternation means the longest identical-token run is one; that automated statistic alone misses this repetitive failure.

Within each condition, every generated ID and both recorded text fields match across hard FP32, pure FP16 and packed fresh reload for each prompt. This output equality does not imply equality of logits or NLL. The complete raw outputs/IDs and settings appear in [quality-generations.md](D:/openbonsai/runs/2026-10-26-5090-rotations/analysis/quality-generations.md) and `quality-summary.json`.

## Prompt-by-prompt judgments

### manual-heldout-00 (instruction)

Original prompt:

<pre>Give exactly three numbered tips for keeping a shared kitchen clean. Each tip must be one sentence.</pre>

Baseline captured answer:

<pre>1. Wash and sanitize all dishes and utensils immediately after use to prevent cross-contamination.
2. Wipe down countertops and appliances with disinfectant before and after every cooking session.
3. Take out the trash and empty the recycling bin daily to avoid lingering odors and pests.</pre>

Complete useful response: exactly three numbered, single-sentence kitchen-cleaning tips.

Unrotated: exact `" number" * 128`. Rotated: exact `"Here\n\n" + "This\n\n" * 63`. Neither condition supplies numbered tips or cleaning advice; all three representations fail the task.

### manual-heldout-02 (reasoning)

Original prompt:

<pre>A farmer has chickens and rabbits. There are 10 heads and 28 legs. How many of each animal are there? Explain briefly.</pre>

Baseline captured answer:

<pre>To solve this, we can use a simple logical deduction based on the number of legs:

1.  **Assume all 10 animals are chickens.** Since chickens have 2 legs, this would result in $10 \times 2 = 20$ legs.
2.  **Compare with the actual leg count.** The farmer has 28 legs, which is $28 - 20 = 8$ more legs than our assumption.
3.  **Account for the difference.** Each time we swap a chicken for a rabbit, the leg count increases by 2 (since rabbits have</pre>

Relevant, correct partial arithmetic, but caps before stating six chickens and four rabbits. It is not a completed answer.

Unrotated: exact `" number" * 128`. Rotated: exact `"Here\n\n" + "This\n\n" * 63`. Neither condition gives animal counts or an explanation; all three representations fail the task.

### manual-heldout-04 (coding)

Original prompt:

<pre>Write a Python function is_balanced(text) that checks whether parentheses are balanced, ignoring other characters. Include an empty-string case.</pre>

Baseline captured answer:

<pre>```python
def is_balanced(text: str) -&gt; bool:
    &quot;&quot;&quot;
    Check whether parentheses in the given text are balanced.
    Only &#x27;(&#x27; and &#x27;)&#x27; are considered; all other characters are ignored.
    An empty string is considered balanced.

    Examples:
        is_balanced(&quot;()&quot;) -&gt; True
        is_balanced(&quot;(]&quot;) -&gt; False
        is_balanced(&quot;((()))&quot;) -&gt; True
        is_balanced(&quot;(()&quot;) -&gt; False
        is_balanced(&quot;&quot;) -&gt; True
        is_balanced(&quot;a(b)c&quot;) -&gt; True
    &quot;&quot;&quot;
    count = 0
   </pre>

Relevant docstring/examples including an empty-string case, but caps after count = 0 and indentation; no loop/return or working balanced-parentheses implementation. No code was executed.

Unrotated: exact `" number" * 128`. Rotated: exact `"Here\n\n" + "This\n\n" * 63`. Neither condition emits a Python function, parentheses logic or empty-string case; all three representations fail the task.

### manual-heldout-07 (turkish)

Original prompt:

<pre>Aşağıdaki cümleyi daha resmi bir Türkçeyle yeniden yaz: Yarın gelemiyorum, toplantıyı başka güne alsak olur mu?</pre>

Baseline captured answer:

<pre>Yarın katılım sağlayamayacağım; toplantıyı başka bir tarihe erteleyebilir miyiz?</pre>

Complete useful formal Turkish rewrite retaining inability to attend tomorrow and a request to postpone the meeting.

Unrotated: exact `" number" * 128`. Rotated: exact `"Here\n\n" + "This\n\n" * 63`. Neither condition produces a Turkish rewrite or preserves the requested meaning; all three representations fail the task.

## Protocol, aggregation and parity verification

- The final utility verified all six completed stages against the identical frozen model revision, token IDs/order, template hash, evaluator/helper hashes, environment, seed and decoding settings. Input SHA-256: `330ffeeb77fa384492a6494e1e698edd400d7088456dbd0b7bc0fd366b7d838a`.
- NLL totals were checked from all four rows: exactly 2,048 shifted, mask-valid scored tokens per evaluation, token-weighted loss and exp(NLL) perplexity. ARC was checked from all 20 per-question likelihood vectors, label token IDs, deterministic argmax and answer keys. EOS/cap totals were recomputed from generated IDs for every output.
- Both current-run baseline probes match the recorded original BF16 logits with identical IDs/masks. The original teacher bypass matches before/after training in both conditions. These saved GPU-runner results were checked; this CPU-only report did not reload the logit tensors.
- Fresh reload is bitwise exact against each saved packed-runtime reference, maximum logit error 0. Both exports verified all 200 projection payloads and 564 retained tensors; cached versus uncached packed probes are exact.
- FP16 scale rounding and wrapper-to-row-64 GEMM changes are separately measured. Their logits differ from the original FP32-wrapper stage, so successful packed reload must not be described as FP32-to-packed exactness or preserved model quality.
- Full small-file checks and evidence hashes are in `final-quality-verification.json`; manual per-prompt judgments/counts are in `manual-quality-review.json`.

## Limits

This is the repeatedly used development subset (2,048 NLL tokens, 20 ARC-Easy questions, four generations), not the original full 16,384-token/100-question/10-generation suite or fresh confirmation. The historical corrected16x1 recipe differs from this matched pair and is reported separately in the quality summary. No unseen prompt quality, lossless compression, inference speedup, or benefit from a future larger budget follows from these observations. Both conditions stop at the authorized paired experiment.

Runtime and memory in the automated summary cover resident evaluation only; training, loading and decoded-cache preparation are separate. Hard/FP16 research wrappers and cached packed deployment have different resident-state scopes. Baseline timing is historical. No GPU, checkpoint read, frozen-checkout change or old-run write was performed for this review.
