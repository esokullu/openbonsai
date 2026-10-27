# Provisional manual review: unrotated16x8 generations

Reviewed 2026-10-26T09:53:55.298776+00:00. This note covers the historical untouched BF16 baseline and all three completed unrotated evaluations: hard-FP32, pure FP16 scales and packed fresh reload on the same four frozen prompts. The rotated condition is still pending and is not judged here.

All twelve unrotated outputs (four per representation) are exactly token ID **1324 repeated 128 times**, with the complete recorded decoded string equal to **`" number" * 128`**, including the leading space before every word. All twelve reach the cap without EOS. This is already present with FP32 factors before export. Reading every output against its prompt, none of the four answers in any unrotated stage provides a useful task response. This is a direct review of these specific outputs, not a general accuracy estimate.

The baseline supplies two complete useful responses (kitchen and Turkish); its other two responses are relevant but incomplete at the same 128-token limit. The reasoning response does not reach the answer, and the code response does not finish an implementation. Baseline truncation is therefore visible in this comparison rather than silently treated as success.

| Evaluation | NLL (2048 tokens) | PPL | ARC | EOS / cap | Manual assessment of these four outputs |
|---|---:|---:|---:|---:|---|
| Historical BF16 | 2.423776811 | 11.288413 | 19/20 | 2 / 2 | 2 complete useful; 2 relevant incomplete |
| Unrotated hard FP32 | 14.439949335 | 1,867,197.748316 | 1/20 | 0 / 4 | 0 useful task responses; 4 repetitive failures |
| Unrotated pure FP16 scales | 14.438326641 | 1,864,170.315635 | 1/20 | 0 / 4 | 0 useful task responses; 4 repetitive failures |
| Unrotated packed fresh reload | 14.437114680 | 1,861,912.383044 | 1/20 | 0 / 4 | 0 useful task responses; 4 repetitive failures |

## Numerical differences and serialization boundary

| Left minus right | Delta NLL | Delta PPL | Delta ARC percentage points |
|---|---:|---:|---:|
| unrotated16x8: FP16 scales, same wrapper GEMM minus unrotated16x8: Hard FP32 factors | -0.001622693 | -3,027.432681 | +0.0 |
| unrotated16x8: Packed, fresh reload, decoded cache minus unrotated16x8: FP16 scales, same wrapper GEMM | -0.001211961 | -2,257.932591 | +0.0 |

Packed versus historical BF16: delta NLL +12.013337870, delta PPL +1,861,901.094631, delta ARC -90.0 percentage points. These small representation-to-representation NLL differences do not restore task usefulness.

Fresh reload matches the saved packed-runtime reference logits bitwise (maximum absolute error 0). All 200 projection payloads and 564 retained tensors pass export checks, and decoded-cache versus uncached inference matches on the fixed probe. The pure FP16-scale wrapper differs from the row-64 packed GEMM runtime (fixed-probe maximum absolute logit difference 0.0625); its NLL difference is therefore not described as a serialization error or solely as FP16 rounding. The severe quality failure already exists with hard FP32 factors.

| Resident evaluation | Seconds | Peak allocated GiB | Peak reserved GiB |
|---|---:|---:|---:|
| Original BF16 (historical) | 34.278 | 17.954 | 18.191 |
| unrotated16x8: Hard FP32 factors | 213.287 | 19.598 | 19.875 |
| unrotated16x8: FP16 scales, same wrapper GEMM | 217.598 | 19.598 | 19.875 |
| unrotated16x8: Packed, fresh reload, decoded cache | 284.163 | 19.454 | 19.750 |

Runtime covers evaluation only, excluding loading and cache preparation. Hard/FP16 research wrappers retain the original model plus factors/A/B; fresh packed inference includes the decoded BF16 weight cache. The baseline timing is historical. These scopes differ and no speedup is inferred.

## Per-prompt review

### manual-heldout-00 (instruction)

Original prompt:

<pre>Give exactly three numbered tips for keeping a shared kitchen clean. Each tip must be one sentence.</pre>

Baseline recorded answer (complete captured text):

<pre>1. Wash and sanitize all dishes and utensils immediately after use to prevent cross-contamination.
2. Wipe down countertops and appliances with disinfectant before and after every cooking session.
3. Take out the trash and empty the recycling bin daily to avoid lingering odors and pests.</pre>

Baseline: 61 generated tokens; EOS=True; cap=False. Relevant and useful complete response: exactly three numbered kitchen-cleaning tips, one sentence each. It terminates with EOS.

All three unrotated stages have the same exact output: `[1324] * 128`; decoded text `" number" * 128`. Representative first ten tokens only (the remaining 118 are identical):

<pre> number number number number number number number number number number</pre>

Manual judgment for all three stages: No numbered tips or cleaning advice; the repeated word is not a useful response.

### manual-heldout-02 (reasoning)

Original prompt:

<pre>A farmer has chickens and rabbits. There are 10 heads and 28 legs. How many of each animal are there? Explain briefly.</pre>

Baseline recorded answer (complete captured text):

<pre>To solve this, we can use a simple logical deduction based on the number of legs:

1.  **Assume all 10 animals are chickens.** Since chickens have 2 legs, this would result in $10 \times 2 = 20$ legs.
2.  **Compare with the actual leg count.** The farmer has 28 legs, which is $28 - 20 = 8$ more legs than our assumption.
3.  **Account for the difference.** Each time we swap a chicken for a rabbit, the leg count increases by 2 (since rabbits have</pre>

Baseline: 128 generated tokens; EOS=False; cap=True. Relevant, intelligible partial reasoning. The assumptions and 20-versus-28 leg arithmetic are correct, but the 128-token cap interrupts the explanation before it states the requested counts (six chickens and four rabbits). This is not a completed answer.

All three unrotated stages have the same exact output: `[1324] * 128`; decoded text `" number" * 128`. Representative first ten tokens only (the remaining 118 are identical):

<pre> number number number number number number number number number number</pre>

Manual judgment for all three stages: No animal counts or explanation; the repeated word is not a useful response.

### manual-heldout-04 (coding)

Original prompt:

<pre>Write a Python function is_balanced(text) that checks whether parentheses are balanced, ignoring other characters. Include an empty-string case.</pre>

Baseline recorded answer (complete captured text):

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

Baseline: 128 generated tokens; EOS=False; cap=True. Relevant partial code/docstring, with an empty-string example, but generation stops after count = 0 and indentation. There is no loop or return, so the captured output is not a usable balanced-parentheses implementation. No generated code was executed.

All three unrotated stages have the same exact output: `[1324] * 128`; decoded text `" number" * 128`. Representative first ten tokens only (the remaining 118 are identical):

<pre> number number number number number number number number number number</pre>

Manual judgment for all three stages: No Python function or empty-string case; the repeated word is not a useful response.

### manual-heldout-07 (turkish)

Original prompt:

<pre>Aşağıdaki cümleyi daha resmi bir Türkçeyle yeniden yaz: Yarın gelemiyorum, toplantıyı başka güne alsak olur mu?</pre>

Baseline recorded answer (complete captured text):

<pre>Yarın katılım sağlayamayacağım; toplantıyı başka bir tarihe erteleyebilir miyiz?</pre>

Baseline: 24 generated tokens; EOS=True; cap=False. Relevant and useful complete formal Turkish rewrite that preserves inability to attend tomorrow and requests postponement. It terminates with EOS.

All three unrotated stages have the same exact output: `[1324] * 128`; decoded text `" number" * 128`. Representative first ten tokens only (the remaining 118 are identical):

<pre> number number number number number number number number number number</pre>

Manual judgment for all three stages: No Turkish rewrite or preserved request; the repeated English word is not a useful response.

## Scope and provenance

- Frozen input SHA-256: `330ffeeb77fa384492a6494e1e698edd400d7088456dbd0b7bc0fd366b7d838a`.
- Identical prompt IDs, input tokenization, chat template, seed5090, greedy decoding, thinking disabled, no KV cache, 128-token limit, and EOS IDs248044/248046. The utility checked the saved protocol and aggregate calculations.
- The baseline output is historical; current-run original-model parity is separately recorded by the runner. This note does not re-run generation.
- These four prompts are a repeatedly used diagnostic subset, not a fresh confirmation set or the full 10-example qualitative suite. Manual judgments are descriptive, not a blinded rubric or statistical quality estimate.
- The repetitive FP32 outputs show that this observed failure precedes FP16 scale serialization. They do not by themselves identify the underlying optimization/quantization cause. Packed export/reload correctness now passes on the fixed checks; this preserves the failed model behavior. The rotated condition remains pending.
- Exact generated IDs, all raw recorded texts and EOS/cap flags are retained in `quality-summary-progress.json` and the original evaluation JSONs. No GPU, model/checkpoint read, or frozen-source edit was performed.

Reviewed result file hashes:

- `D:\openbonsai\runs\2026-10-26-5090-rotations\inputs\baseline-original.json`: `a673bd24dd81eda919eaaa14437529c464f8794c3e3a335295a73bf76d4356ad`.
- `D:\openbonsai\runs\2026-10-26-5090-rotations\unrotated16x8\hard-fp32-evaluation.json`: `48350f5fe9c8ea8d392592ab882e7742b6606111eb8dd4339a0f286a58992384`.
- `D:\openbonsai\runs\2026-10-26-5090-rotations\unrotated16x8\exported-scales-evaluation.json`: `1368f281e6590a8484501b84816a8cde90bb3dadd9fae813728693dd98a331bd`.
- `D:\openbonsai\runs\2026-10-26-5090-rotations\unrotated16x8\packed-evaluation.json`: `72db0ea8594d22e7e60404f2181a8d08681f8303ad51eda3cadd81c540a076bf`.
- `D:\openbonsai\runs\2026-10-26-5090-rotations\unrotated16x8\export-verification.json`: `6f5737414e0ddd863e1b9c06c03e17231e747c1e298ac339b26c38b9ec2c6511`.
- `D:\openbonsai\runs\2026-10-26-5090-rotations\unrotated16x8\fresh-reload-parity.json`: `b8c8af2b7e7e6143e3d81921c22c65a0c11778e5ce2b1aa84375e4bba17a00d5`.
- `D:\openbonsai\runs\2026-10-26-5090-rotations\unrotated16x8\completed.json`: `11a254e93f17d54cf811fff5b15c7b34f9b9f138be8c57c42cca1afff4536b58`.
