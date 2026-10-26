# Broad-coverage storage budget

Broad ternary coverage could approach a 2 decimal GB language artifact with
the current representation. This is a **component-size projection**, not a
new exported model or evidence that the quantized model has useful quality.

The pinned Qwen3.5-9B scope contains 200 decoder matrices plus the untied
embedding and output head: 8,946,450,432 weights. The two endpoints each have
shape `[248320, 4096]`. The existing decoder artifact still retains both
endpoints in BF16, costing 4,068,474,880 bytes by themselves.

## Conditional broad-coverage calculation

Assume all 202 matrices can use the learned ternary representation, group size
128, one FP16 scale per group, and the current module-local rotations. Use
flat base-3 packing for this calculation; it requires no assumption about the
learned zero density. Costs are summed separately for each matrix, including
the final partial code byte.

| Component | Bytes |
|---|---:|
| Base-3 codes: sum of `ceil(weight_count / 5)` | 1,789,290,192 |
| FP16 group scales: sum of `2 * ceil(weight_count / 128)` | 139,788,288 |
| Current int8 rotation signs | 1,089,536 |
| Retained small language tensors, from the existing artifact scope | 14,705,920 |
| Existing tokenizer files | 19,998,761 |
| **Component subtotal** | **1,964,872,697** |

This leaves **35,127,303 bytes** below 2,000,000,000 for containers, configuration,
manifest, indexing where required, and any precision exceptions. It is not a
guaranteed final size: the selected complete artifact must be serialized and
counted. In particular, this is not the size of a current text-only export.

Rank-64 low-rank corrections do **not** add a deployment sidecar. The quantizer
adds `B @ A / rank` before deriving the final ternary codes and scales; export
stores those codes and scales, not the BF16 master or learned A/B matrices.
Research checkpoints and calibration memory have different accounting.

BITCOS uses presence and nonzero-sign streams, so its cost depends on the actual
candidate's zeros. Use it only when the complete serialized candidate, including
any row index and metadata, is smaller. At low sparsity, base-3 supplies the
zero-density-independent alternative used above. Neither codec changes the
already learned ternary values; FP16 scale casting is a separate approximation.

## Scope and decision limits

The baseline contains 912,021,104 bytes of vision tensor payload, or 912,058,304
bytes in its vision files including headers. Those bytes already overlap the
baseline's retained total. They are excluded from the language subtotal above
and must not be counted twice. Retaining that vision payload gives about
2.877 GB before the remaining container/configuration overhead.

The existing 9B artifact preserves its native multimodal configuration. The
adapter also supports Qwen's native text-only model class, but converting this
checkpoint into a validated text-only artifact or separate vision package
remains unimplemented and unverified. Deleting vision files from the existing
artifact violates its retained-tensor contract. Explicit source/module mapping,
native output parity and reload tests are required before a 2 GB language
deployment can be claimed.

The size calculation supports continuing to investigate broad ternary quality.
It does not support replacing that objective with many 4/6-bit matrices: the
remaining budget is small. Any RCO exceptions must use actual candidate and
whole-artifact bytes, with the budget fixed before evaluating the assignment.

See [the rotation placement audit](rotations-audit.md),
[the packing implementation](../openbonsai/packing.py),
[endpoint export](../openbonsai/endpoint_export.py), and
[the measured 5090 results](5090-runtime.md). The measured decoder-only artifacts
remain about 6.597 GB with vision and BF16 endpoints; the broad-coverage quality
and projected deployment above have not been demonstrated.
