# Provisional manual review: rotated hard-FP32 and pure FP16 scales

Reviewed 2026-10-26T10:39:53.940937+00:00. Both completed rotated hard-FP32 and pure FP16-scale stages are judged here; packed fresh reload remains pending. This is the fixed 2,048-token / 20-question / four-generation diagnostic subset.

Rotation improves the measured hard-FP32 NLL and ARC score relative to the matched unrotated model, but does not recover useful generation on these four prompts. Each stage produces the same four identical outputs (eight reviewed outputs total): `Here` followed by a blank line, then `This` followed by a blank line 63 times. All eight reach 128 tokens without EOS. All four responses in each representation fail task relevance and usefulness on direct reading; these are specific qualitative observations rather than a general quality estimate.

| Evaluation | NLL | PPL | ARC correct | ARC % | Unfinished |
|---|---:|---:|---:|---:|---:|
| Original BF16 (historical) | 2.423776811 | 11.288413 | 19/20 | 95.0 | 2/4 |
| unrotated16x8: Hard FP32 factors | 14.439949335 | 1,867,197.748316 | 1/20 | 5.0 | 4/4 |
| rotated16x8: Hard FP32 factors | 9.257334059 | 10,481.153923 | 8/20 | 40.0 | 4/4 |

| Rotated hard FP32 minus reference | Delta NLL | Delta PPL | Delta ARC percentage points |
|---|---:|---:|---:|
| unrotated16x8: Hard FP32 factors | -5.182615276 | -1,856,716.594393 | +35.0 |
| Original BF16 (historical) | +6.833557248 | +10,469.865510 | -55.0 |

The ARC improvement is seven more correctly selected answer labels (8/20 versus 1/20); it is still eleven fewer than BF16 (19/20). ARC is likelihood ranking of fixed answer labels, so it can improve even while free-form generation remains degenerate. The large remaining BF16 gap and failed outputs prevent a claim of quality recovery.

## Pure FP16-scale comparison

I reviewed all four completed FP16-scale responses against their original prompts. Every generated token ID and both recorded text fields match hard FP32 exactly on every prompt. Thus the same task-specific failures below apply to both representations: no kitchen tips, no animal counts/explanation, no Python function, and no Turkish rewrite. This establishes identical captured output behavior on these examples; it does not mean the underlying logits or likelihoods are identical.

| Completed rotated representation | NLL | PPL | ARC | EOS / cap | Useful task responses (manual) |
|---|---:|---:|---:|---:|---:|
| rotated16x8: Hard FP32 factors | 9.257334059 | 10,481.153923 | 8/20 | 0 / 4 | 0/4 |
| rotated16x8: FP16 scales, same wrapper GEMM | 9.253318076 | 10,439.146194 | 8/20 | 0 / 4 | 0/4 |

| Rotated pure FP16 scales minus reference | Delta NLL | Delta PPL | Delta ARC pp |
|---|---:|---:|---:|
| rotated16x8: Hard FP32 factors | -0.004015983 | -42.007728 | +0.0 |
| unrotated16x8: FP16 scales, same wrapper GEMM | -5.185008565 | -1,853,731.169440 | +35.0 |
| Original BF16 (historical) | +6.829541265 | +10,427.857781 | -55.0 |

The pure FP16 evaluation took 643.455 seconds with peak allocated/reserved VRAM 19.599/19.869 GiB. It retains the same wrapper GEMM and original code selection; this comparison isolates the intended scale/bias precision change before the separate packed runtime.

The smaller NLL after FP16 rounding is a measured numerical difference, not qualitative recovery. ARC remains 8/20 and all four generated answers remain unusable for their prompts. Rotated packed evaluation is excluded until completed.

## Exact recorded repetition

All four token sequences in each of the two representations are exactly:

```python
[8160, 271] + [1919, 271] * 63
```

All eight recorded decoded strings, including the versions retaining special tokens, are exactly:

```python
"Here\n\n" + "This\n\n" * 63
```

The token-frequency counts are 8160: 1, 271: 64 and 1919: 63; none is an EOS token. The longest identical-token run is only one because the repetition alternates word/newline tokens. This illustrates why the utility’s single-token-run statistic alone is not a semantic or general repetition detector.

Representative beginning, with the remaining repetitions omitted from this excerpt only:

<pre>Here

This

This

This

This

</pre>

Full unaltered recorded strings and all 128 token IDs for each example are preserved in `quality-summary-progress.json` and the source evaluation JSON.

## Per-prompt manual review

### manual-heldout-00 (instruction)

<pre>Give exactly three numbered tips for keeping a shared kitchen clean. Each tip must be one sentence.</pre>

No numbered list, sentences of advice or kitchen-cleaning content. It does not satisfy the instruction and is not useful for the requested task.

Recorded output: exact sequence shown above; 128 tokens, EOS=False, cap=True. Generation time 152.901 seconds.

Pure FP16-scale output comparison: full 128-token sequence and both recorded text fields exactly equal hard FP32 on this prompt; EOS=False, cap=True. The same manual task judgment applies.

### manual-heldout-02 (reasoning)

<pre>A farmer has chickens and rabbits. There are 10 heads and 28 legs. How many of each animal are there? Explain briefly.</pre>

No chicken/rabbit counts or mathematical explanation. It never answers the question; the expected counts are six chickens and four rabbits. It is not useful for the requested task.

Recorded output: exact sequence shown above; 128 tokens, EOS=False, cap=True. Generation time 152.795 seconds.

Pure FP16-scale output comparison: full 128-token sequence and both recorded text fields exactly equal hard FP32 on this prompt; EOS=False, cap=True. The same manual task judgment applies.

### manual-heldout-04 (coding)

<pre>Write a Python function is_balanced(text) that checks whether parentheses are balanced, ignoring other characters. Include an empty-string case.</pre>

No Python function, balanced-parentheses logic or empty-string case. The output contains no usable implementation. No code execution is necessary to identify this absence.

Recorded output: exact sequence shown above; 128 tokens, EOS=False, cap=True. Generation time 153.115 seconds.

Pure FP16-scale output comparison: full 128-token sequence and both recorded text fields exactly equal hard FP32 on this prompt; EOS=False, cap=True. The same manual task judgment applies.

### manual-heldout-07 (turkish)

<pre>Aşağıdaki cümleyi daha resmi bir Türkçeyle yeniden yaz: Yarın gelemiyorum, toplantıyı başka güne alsak olur mu?</pre>

No Turkish rewrite, attendance statement or postponement request. The repeated English fragments do not preserve the requested meaning and are not useful for this task.

Recorded output: exact sequence shown above; 128 tokens, EOS=False, cap=True. Generation time 153.422 seconds.

Pure FP16-scale output comparison: full 128-token sequence and both recorded text fields exactly equal hard FP32 on this prompt; EOS=False, cap=True. The same manual task judgment applies.

## Timing and limits

Rotated hard-FP32 evaluation took 641.245 seconds, versus 213.287 for unrotated and historical BF16 34.278. Rotated evaluation peak allocated/reserved VRAM was 19.599/19.869 GiB. These are already-resident evaluation phases, excluding loading and export; the research wrappers retain original weights and factors/A/B. No speedup is implied.

The baseline kitchen and Turkish responses are complete and useful on direct review. Its reasoning and coding responses are relevant but truncate at the same 128-token cap before the final answer/implementation. Thus baseline completion limitations remain explicit.

The repeated diagnostic prompts have been used during prior investigation. Four manual examples and 20 ARC items are small, development-set evidence, not a fresh full-suite confirmation or a blinded quality study. This review makes no assumption that more training would repair the failure. Only completed hard-FP32 and pure FP16-scale rotated results enter this note; packed fresh-reload results remain pending.

## Provenance

- Input SHA-256: `330ffeeb77fa384492a6494e1e698edd400d7088456dbd0b7bc0fd366b7d838a`.
- Rotated hard-FP32 result SHA-256: `10d26bc5dbe1049013f40c249c54d03e79e0937c33ce929a314496cf1ac128d2`.
- Same frozen prompt token IDs, model revision, chat template, seed 5090, greedy/nonthinking settings, no KV cache, 128-token limit and EOS IDs 248044/248046 were checked by the stdlib progress utility.
- No GPU, model/checkpoint read, frozen-source change or old-run write was performed.

- Rotated pure FP16-scale result SHA-256: `fb81e891583b5747366466e20a2d4754ab8157518c994781c5e00098c9c81cee`.
