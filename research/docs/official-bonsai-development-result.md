# Official Bonsai result on the frozen development inputs

On 26 October 2026, the official [**Ternary-Bonsai-2-9B PTQ1_0** checkpoint](https://huggingface.co/prism-ml/Ternary-Bonsai-2-9B-gguf/tree/5b2abb02362efbed46bbe204014b02837778b2c2) completed our frozen development suite on macOS CPU: **mean token NLL 3.1921478146079645**, **perplexity 24.340650545056054**, and **16/20 ARC-Easy label choices correct**. Four generation prompts also completed under their original 128-token limits.

This measures the official checkpoint together with its pinned runtime. It is an actual small development result, not a vendor benchmark, fresh held-out confirmation, or evidence that OpenBonsai already preserves the same quality. A numerical BF16/OpenBonsai comparison remains pending rescoring with the common probability reduction.

## What was scored

| Component | Fixed workload | Result |
|---|---|---|
| WikiText-2 test slices | Four independent 513-token inputs; positions 1–512 scored in each | 2,048 targets, NLL sum 6537.518724317111 |
| ARC-Easy validation subset | 20 existing questions; four single-token A/B/C/D continuations each | 16/20 correct (80%) |
| Greedy generation | Four existing prompts; cap 128 each; EOS 248044/248046 | Two EOS stops; two length stops |

All **88 text records** passed native tokenizer checks for exact IDs and raw byte pieces before scoring. The original IDs, causal positions, choice order, answers, generation caps and EOS were preserved. Explicit NFC normalization changed no records. NLL text in the converted manifest is reconstructed from authenticated token slices and tokenizer pieces; this does not independently authenticate the original dataset text. The frozen role is `search_validation`: historical `heldout` record names do not make these repeatedly inspected inputs new confirmation data.

Choice scoring measures the likelihood of the **option letter**, not the full answer text. No winning choice was tied. The smallest top-two logprob margin was 0.20588111877441395. An all-A predictor scores 8/20 on this particular subset. The small sample is not a benchmark-wide accuracy estimate.

## Generation assessment

| Task | Tokens / stop | Observed response |
|---|---|---|
| Three numbered kitchen tips | 69 / EOS | Three relevant tips, one sentence each; satisfies the requested format. |
| Ten heads and 28 legs | 128 / length | Correct equations, but no final animal counts before the cap; incomplete answer. |
| Parenthesis-balancing Python function | 128 / length | Complete, syntactically valid function body, including correct empty-string behavior; unclosed Markdown fence and no explicit empty-string example. |
| Formal Turkish rewrite | 99 / EOS | Provides two formal alternatives. The second preserves the intended request naturally; the first has awkward phrasing. |

The Python body passed **14 hand-authored, posthoc smoke cases**, including empty input, nesting, unmatched parentheses, ignored text and Unicode. An independent review checked the exact recorded code, receipt hashes, syntax and case expectations. These tests are narrow behavioral evidence, not a preregistered coding benchmark or complete formatting success. No cap was extended to finish an answer.

## Runtime, storage and completion

The run used Prism commit `c9c34bc9c4042c13a31598ebf43461cd0ad8f00d`, the C-API worker and the unchanged `evaluate_manifest` / `PrismBackend` implementation. Settings were four CPU threads, zero GPU layers, context 513 (internally allocated 768), batch and microbatch 64, F32 KV cache, no Flash Attention and no MTP. Independent requests reset hybrid memory; greedy generation recomputes the full prefix. FP32 logits use the `fp32-logits-fp64-logsumexp-v1` reduction. PTQ1_0 CPU matrix products still use Q8_0 activation operands; F32 logits/cache do not imply all-F32 arithmetic.

The complete official language GGUF is **1,997,486,976 bytes**, including embedded tokenizer and container data. It covers 202 ternary matrices, including embedding and output head, with retained small F32 tensors. It excludes vision and MTP. Earlier OpenBonsai deployment sizes retained BF16 endpoints and vision, so their total sizes are not equivalent coverage comparisons. See the [measured inventory](bonsai-reference-preflight.md) and [storage budget](storage-budget.md).

Whole-process time was **3746.9061732499395 seconds** (about 62.45 minutes), including startup, preflight, scoring, generation and shutdown, within a 10,800-second deadline. The supervisor recorded exit code 0 and no remaining owned process-group members. The sampled peak sum of runner and descendant RSS was **2,751,627,264 bytes**; it can count shared pages twice and is not an exact physical-memory peak. Concurrent local work means this timing is operational evidence, not an isolated speed benchmark.

The admission forecast allowed all 512 possible generated tokens, with a 1.25 margin and 300-second reserve; it did not assume early EOS. Three resource probes were excluded from quality metrics. All 449 recorded calls completed, including 84 quality scoring calls and four generations. Independent summation of the returned logprobs agreed with the reported mean NLL within 2e-15. The completed archive contains 463 run files totaling 3,811,993 bytes, separate from model weights. Post-run full GGUF, executable, evaluator source, scratch runner and manifest hashes still matched; the pinned Prism tracked source was clean. The seal SHA256 is `a57291a8e4b27e9066403b25b02bbd43d14edccd2e7b50ee515ebd33d3b23d2f`.

## Reproduction and limits

**The measured run used the reviewed scratch script `run_bonsai_development.py`, SHA256 `b6fdb42611473cba51e079b8d2dbf6381025dc8fc8b2eb9f5ba814a5f35394ae`. It did not execute the later portable `tools/evaluate_prism_manifest.py`.** Both use the same evaluator/backend, but a future portable-tool run is a separate execution. The [matched evaluation guide](matched-evaluation.md) documents the portable workflow and required local inputs.

The [validation receipt](../validation-official-bonsai-development.json) binds the original transported input/helper files, converted manifest, tokenizer, official checkpoint, runtime, executable, scratch runner and completed result receipts. The canonical frozen-manifest SHA256 is `8021d9cc8f54943a3b816af74cbc41d588173d385f51b73a681e979393ef5fdf`. The exact [frozen development inputs and answer keys](../research/evaluation/bonsai-development/frozen-inputs.json) are included for reproducible handoff. Their NLL text is reconstructed from token pieces, not the original raw WikiText source. Model weights, tokenizer assets, original transport files and the full run archive remain outside this publication. Use the packaged manifest for an exact rerun; downloading the named dataset versions alone does not establish the same selection, rendering or IDs.

Legacy 5090 NLL used FP32 cross-entropy, and legacy ARC used FP32 log-softmax. This run normalizes FP32 logits in FP64. Moreover, the official CPU quantized kernels differ from native CUDA execution; ARC used 80 independent option calls instead of 20 shared-prefix forwards. The saved scalar scores cannot retroactively remove those differences. BF16 and OpenBonsai must be rescored on the exact frozen inputs with the common scorer, with their original-dtype policy and backend declared, before a quantitative quality-retention table is justified. No losslessness, fine-tune equivalence, Windows/CUDA parity or successful OpenBonsai endpoint calibration is established here.
