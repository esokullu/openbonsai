# Bonsai rotation contract and OpenBonsai prototype

Audited 2026-10-26 against the official small-model whitepaper, the 9B MLX
checkpoint at `f4e28a59531731c41b6dfac88cf271bd58f0a287`, and Prism's llama.cpp
release `prism-b10826-c9c34bc`. This distinguishes the released inference
contract from choices for an independent training implementation.

## Exact placement and order

Let `H` be a block-diagonal **normalized** Sylvester Walsh-Hadamard matrix
(`H.T = H`, `H @ H = I`) and `S` a diagonal matrix of explicit +/-1 signs.
For PyTorch row vectors, define `R = S @ H`:

```text
rotate(x)     = (x * signs) @ H       = x @ R
inverse(z)    = (z @ H) * signs      = z @ R.T
W_rot         = (W * signs) @ H      = W @ R
linear(x)     = rotate(x) @ W_rot.T + bias = x @ W.T + bias
E_rot         = (E * signs) @ H
embedding(id) = inverse(E_rot[id])    = E[id]
```

The paper writes the corresponding column-vector input transform `H @ S`.
This is the transpose of the row-vector `S @ H`, not a contradictory ordering.
Signs multiply **before** Hadamard for both activation and weight rows; inverse
embedding restoration multiplies signs **after** Hadamard. Bias remains in the
original output coordinates. Hadamard twice without the sign order reversal
does not generally invert the signed transform.

Direct pinned source evidence:

- [MLX `fwht`, lines 18-30](https://huggingface.co/prism-ml/Ternary-Bonsai-2-9B-mlx-2bit/blob/f4e28a59531731c41b6dfac88cf271bd58f0a287/runtime/runtime.py#L18): cast to FP32, forward signs, normalized transform, inverse signs, cast back.
- [MLX packed embedding/linear, lines 44-79](https://huggingface.co/prism-ml/Ternary-Bonsai-2-9B-mlx-2bit/blob/f4e28a59531731c41b6dfac88cf271bd58f0a287/runtime/runtime.py#L44): decode only requested embedding rows then inverse-transform; transform linear inputs then packed matrix multiplication.
- [llama.cpp linear graph, lines 1597-1628](https://github.com/PrismML-Eng/llama.cpp/blob/prism-b10826-c9c34bc/src/llama-graph.cpp#L1597): optional feature permutation, signs, Hadamard, matrix multiplication.
- [llama.cpp embedding graph, lines 2451-2462](https://github.com/PrismML-Eng/llama.cpp/blob/prism-b10826-c9c34bc/src/llama-graph.cpp#L2451): Hadamard after row lookup, then signs.

These are module-local transforms. They cancel around each original linear
operation; they do not rotate residual streams across nonlinearities, RMSNorm,
RoPE, attention products, sigmoid gates, or recurrent state updates. Quantizing
`W_rot` breaks exact equivalence and requires measurement/reconstruction.

## Coverage and architecture specifics

The pinned [Hadamard manifest](https://huggingface.co/prism-ml/Ternary-Bonsai-2-9B-mlx-2bit/blob/f4e28a59531731c41b6dfac88cf271bd58f0a287/hadamard.json)
records block size 4096, input-last-dimension axis, explicit signs for widths
4096 and 12288, 201 forward-transform weights, and one inverse-transform
embedding. All 202 [module records](https://huggingface.co/prism-ml/Ternary-Bonsai-2-9B-mlx-2bit/blob/f4e28a59531731c41b6dfac88cf271bd58f0a287/config.json)
use block 4096. The MLP down projection's 12288-wide input uses three blocks.
The sign vector spans the entire input width; it need not repeat every block.

The forward set contains LM head and all 200 large decoder projections:
MLP up/gate/down; full attention q/k/v/o; DeltaNet in_proj_qkv/in_proj_z/out_proj.
The fused q_proj gate is carried as output rows; input rotation does not split
or change that gate. DeltaNet in_proj_a/in_proj_b, recurrence parameters,
normalization, and convolution remain outside the rotated ternary set. The
vision tower remains unrotated. Input embedding and LM head are separate in
this 9B checkpoint; avoid assuming tied weights.

The GGUF-to-MLX loader also reorders DeltaNet value heads and matching gate,
state, convolution/output coordinates. See [runtime lines 186-207 and
242-247](https://huggingface.co/prism-ml/Ternary-Bonsai-2-9B-mlx-2bit/blob/f4e28a59531731c41b6dfac88cf271bd58f0a287/runtime/runtime.py#L186).
This is a backend layout conversion. A native Transformers prototype that
keeps native module ordering should not copy those permutations blindly or
reorder activations twice.

## What the sources do not establish

The published manifest specifies final fixed signs and placement; it does not
provide an open BF16-to-ternary training pipeline or prove that random signs
plus a few reconstruction epochs reproduce Bonsai quality. The [small-model
whitepaper, section 2](https://github.com/PrismML-Eng/Bonsai-demo/blob/main/bonsai-2-small-whitepaper.pdf)
identifies fixed rotations, group-128 ternary weights, and the 9B block width.
Sign generation/selection, full compression training, and an independently
reproducible 9B recovery recipe are not established by the runtime contract.

Therefore deterministic seeded signs, smaller block sizes for fixtures,
ScaleQ/CAT-Q reconstruction, low-rank corrections in rotated coordinates,
precision choices, and later RCO allocation are OpenBonsai experimental
choices. Record them explicitly. Repacking fixed ternary codes with BITCOS
is lossless; the preceding model quantization is not.

## Implementation and validation boundaries

First prove the float transformation is equivalent before any quantization:
compare individual modules with bias and full native model logits, covering
both attention types and the embedding/head. Use an explicit dense H oracle
on small fixtures to catch orientation/normalization mistakes; inverse-only
round trips can conceal matching wrong conventions. Report BF16 activation
rounding separately from FP32 algebra. A frozen BF16 model whose rotated
weights are immediately rounded back to BF16 acquires extra rounding even
before ternarization; do not call that path exactly lossless.

During reconstruction, the original teacher uses original coordinates and
weights. The student linear sees rotated inputs and quantized rotated weights;
its output is already in original coordinates, so existing block teacher
targets remain comparable. The embedding student must inverse-transform
selected rows before sending them into the native decoder. Never let a
global teacher-disable flag apply rotation to original, unrotated weights.

Keep the original BF16 weights available during training or as a reloadable
source. In deployment, packed rotated weights plus signs and block metadata
suffice: dense original master copies are unnecessary. Embedding inference
can decode and inverse-transform only requested rows, as the official runtime
does. A 248320 x 4096 FP32 embedding reconstruction would itself allocate
about 4.07 GB; avoid recreating the whole table for each lookup. Materialize
weight transformations in bounded rows for export, and verify fresh reload
logits against the exact quantized student. Packed inference correctness does
not establish a speedup without dedicated kernels and measurement.
