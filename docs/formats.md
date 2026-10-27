# Formats

This page describes three formats:
- **Prism's PTQ1_0 GGUF**: what `openbonsai export` writes, and what Prism's llama.cpp fork runs.
- **The latent checkpoint**: what the export writes when Prism cannot run the architecture.
- **The run state**: how the framework stores codes, scales and signs between stages.

Prism's side was checked against PrismML-Eng/llama.cpp at commit `c9c34bc9c4042c13a31598ebf43461cd0ad8f00d` (`ggml-common.h`, `ggml-quants.c`, `llama-model.cpp`) and against the official `Ternary-Bonsai-2-9B-PTQ1_0.gguf`. The code: `openbonsai/formats/ptq1_0.py` (block codec), `openbonsai/formats/gguf.py` (container), `openbonsai/adapters/qwen3_5_gguf.py` and `openbonsai/adapters/dense.py` (layouts), `openbonsai/stages/export.py` (writing and verification).

## Rotation

Take a linear layer y = W x with W of shape [out, in], and these definitions:
- s ∈ {−1, +1}^in is the sign vector for width `in`, and S = diag(s).
- H_B is the normalized Sylvester-Hadamard matrix of size B, in natural order (H_B = H_Bᵀ, H_B H_B = I).
- H = blockdiag(H_B, …, H_B) covers the whole width; B is the block size.
- R = S·H.

| Quantity | Formula |
|---|---|
| Stored weight (codes times scales, rotated basis) | Q ≈ W R = W·S·H |
| Runtime activation | x′ = Rᵀ x = H (s ⊙ x): signs first, then the Hadamard transform |
| Runtime output | y = Q x′ = Q H (s ⊙ x) |
| Equivalent original-basis weight | W_q = Q Rᵀ = Q H S |

Since R Rᵀ = S H H S = I, y = W x exactly when Q = W R. In row-vector form, as the code writes it: x R = (x ⊙ s) H, and W_rot = W R.

**Signs.** There is one sign vector per input width, shared by every tensor of that width. For Qwen3.5-9B that means two vectors, for widths 4,096 and 12,288. Prism reads them from the file, so any ±1 vector works. OpenBonsai draws them with `seeded_signs(seed=width, width)`, which uses PyTorch's CPU generator (`rotation.signs: {mode: seeded, seed: width}`). It always stores them explicitly.

**Block.** The block is one size per model: the largest power of two up to `rotation.max_block` (4,096) that divides every rotated input width. Every rotated width must be a multiple of 128. For Qwen3.5-9B the block is 4,096. Prism's runtime accepts any power of two. It has run real models only at 4,096, and tiny random models at 128 in our tests; other blocks, such as 2,048 for Llama 3.1 8B, are untested on real weights. The export records a note for any block other than 4,096.

**Embedding (inverse lookup).** The token embedding E [vocab, hidden] is stored in the rotated basis too: row t holds z_t ≈ e_t R. The runtime looks up the row and rotates it back: e_t = z_t Rᵀ = s ⊙ (H z_t). The file lists it under `prism.hadamard.inverse_weight_names`, not under `weight_names`.

**What is not rotated.** Norms, biases, convolutions, the Gated DeltaNet gates and every other F32 tensor stay in the original basis. The rotation is never moved through a norm, an activation function, RoPE or a recurrent state: each rotated matrix is undone at its own input. A high-precision (BF16) embedding or head is outside `prism.hadamard` and is not rotated.

## The PTQ1_0 block

PTQ1_0 is GGML type 143. The file type is `LLAMA_FTYPE_MOSTLY_PTQ1_0` = 143. (gguf-py's `LlamaFileType` says 129; the C++ side uses 143.)

A row of a PTQ1_0 tensor is a sequence of blocks. Each block holds 128 consecutive weights of the row, in the rotated input order, in 28 bytes:

```
struct block_ptq1_0 {      // 28 bytes, little-endian
    uint8_t   qs[24];      // 120 weights, five base-3 digits per byte
    uint8_t   qh[2];       // 8 weights, four digits per byte
    ggml_half d;           // fp16 scale of the whole block
};
```

That is 1.75 bits per weight: 1.625 for the codes and 0.125 for the scale. A tensor's `ne[0]` (its input width) must be a multiple of 128.

**Digits.** Each weight is a code c ∈ {−1, 0, +1}, stored as the digit d = c + 1 ∈ {0, 1, 2}. A byte packs its digits most significant first. With q = Σₙ dₙ · 3^(4−n) (0 ≤ q ≤ 242), the byte stores ⌈q · 256 / 243⌉: the value scaled to fill the byte. Which weights each byte holds:

| Bytes | Weights (n = digit position, most significant first) |
|---|---|
| `qs[m]`, m = 0..15 | m + 16n, n = 0..4 (weights 0..79) |
| `qs[16 + m]`, m = 0..7 | 80 + m + 8n, n = 0..4 (weights 80..119) |
| `qh[h]`, h = 0..1 | 120 + h + 2n, n = 0..3 (weights 120..127); q is multiplied by 3 first, so its four digits fill the top four positions |

**Decoding.** Digit n of byte b is `((uint8_t)(b * 3^n) * 3) >> 8`. The multiplication wraps to 8 bits before the multiply by 3, as the C code does. The weight is (digit − 1) · d.

**Canonical form.** Different bytes can encode the same function, so the export normalizes every group before packing:
1. A scale with the fp16 sign bit set, −0.0 included, negates its group's codes and clears the bit.
2. A zero scale zeroes its group's codes, because they are dead.

These are the bytes Prism's reference quantizer (`quantize_row_ptq1_0_ref`) writes for the same function, with one exception: an all-zero group with a nonzero scale keeps its scale and encodes as `qs = 128`, `qh = 127`, where Prism's quantizer would write d = 0. Both decode to zeros. Codes outside {−1, 0, 1} and non-finite scales are refused.

## The GGUF file

GGUF version 3, alignment 32 (no `general.alignment` key). OpenBonsai has its own reader and writer (`formats/gguf.py`); it does not need gguf-py. The writer streams each tensor, so a 2 GB file never sits in memory. Shapes are in GGUF order: `ne[0]` is the contiguous input width, so a [out, in] matrix has `ne = [in, out]`.

**Tensor order.** `output.weight` (absent for a tied table), `output_norm.weight`, `rope_freqs.weight` (llama3 rope only), `token_embd.weight`, then `blk.0` … `blk.N-1`. Inside a block, names are ASCII-sorted, as llama.cpp's `weight_name_comparer` sorts them.

### `prism.hadamard.*` metadata

| Key | Type | Value |
|---|---|---|
| `prism.hadamard.version` | u32 | 1 |
| `prism.hadamard.block_size` | u32 | the block, e.g. 4096 |
| `prism.hadamard.transform` | str | `normalized-sylvester-walsh-hadamard` |
| `prism.hadamard.axis` | str | `input-last-dimension` |
| `prism.hadamard.sign_mode` | str | `explicit` |
| `prism.hadamard.weight_names` | arr[str] | the folded tensors: `output.weight` first, then the decoder weights in sorted Hugging Face name order |
| `prism.hadamard.sign_widths` | arr[i32] | the rotated input widths, ascending, e.g. [4096, 12288] |
| `prism.hadamard.sign_values` | arr[i32] | the sign vectors concatenated in `sign_widths` order (±1) |
| `prism.hadamard.inverse_weight_names` | arr[str] | the inverse-lookup tensors: `["token_embd.weight"]` |
| `prism.hadamard.gdn_v_grouped` | bool | `qwen35` only: true, see below |

The loader enforces these rules:
- Every PTQ1_0 tensor is in exactly one of `weight_names` and `inverse_weight_names`.
- The sign table covers every PTQ1_0 input width.
- In version 1, a tied output (one table that is both embedding and head) cannot be part of `prism.hadamard`.
- Prism (c9c34bc) accepts `prism.hadamard` only for the architectures `llama`, `qwen3`, `qwen3moe`, `qwen35`, `qwen35moe`, `qwen3next` and `dspark`. It refuses every other architecture at load. That is why Qwen2 (`qwen2`) exports a latent checkpoint.

The rest of the metadata is what Prism's converter writes for the architecture: the hyperparameters, `general.*`, the tokenizer (`tokenizer.ggml.*`, byte-level BPE) and the chat template. For the 9B config, the KV keys, their order and their types equal the official file, and so do the tensor layout and the data offsets. Only `general.name` and `prism.hadamard.sign_values` differ. For a fine-tune, the tokenizer and descriptive keys are the fine-tune's own.

### Qwen3.5 (`qwen35`)

| Hugging Face (`model.language_model.` + …) | GGUF | Type | Stored as |
|---|---|---|---|
| `lm_head` | `output.weight` | PTQ1_0 | folded |
| `norm.weight` | `output_norm.weight` | F32 | 1 + w |
| `embed_tokens` | `token_embd.weight` | PTQ1_0 | inverse lookup |
| `layers.N.input_layernorm.weight` | `blk.N.attn_norm.weight` | F32 | 1 + w |
| `layers.N.post_attention_layernorm.weight` | `blk.N.post_attention_norm.weight` | F32 | 1 + w |
| `layers.N.mlp.{gate,up,down}_proj` | `blk.N.ffn_{gate,up,down}.weight` | PTQ1_0 | folded |
| `layers.N.linear_attn.in_proj_qkv` | `blk.N.attn_qkv.weight` | PTQ1_0 | folded; V rows tiled |
| `layers.N.linear_attn.in_proj_z` | `blk.N.attn_gate.weight` | PTQ1_0 | folded; rows tiled |
| `layers.N.linear_attn.out_proj` | `blk.N.ssm_out.weight` | PTQ1_0 | folded; columns stay grouped |
| `layers.N.linear_attn.in_proj_a`, `in_proj_b` | `blk.N.ssm_alpha.weight`, `blk.N.ssm_beta.weight` | F32 | copy; rows tiled |
| `layers.N.linear_attn.A_log` | `blk.N.ssm_a` | F32 | −exp(A_log); tiled |
| `layers.N.linear_attn.dt_bias` | `blk.N.ssm_dt.bias` | F32 | copy; tiled |
| `layers.N.linear_attn.conv1d.weight` [C, 1, K] | `blk.N.ssm_conv1d.weight` | F32 | squeezed to [C, K]; channels as `attn_qkv` |
| `layers.N.linear_attn.norm.weight` | `blk.N.ssm_norm.weight` | F32 | w (this norm is not zero-centered) |
| `layers.N.self_attn.q_proj` (query and output gate, 2 × heads × head_dim rows) | `blk.N.attn_q.weight` | PTQ1_0 | folded |
| `layers.N.self_attn.{k,v,o}_proj` | `blk.N.attn_{k,v,output}.weight` | PTQ1_0 | folded |
| `layers.N.self_attn.{q,k}_norm.weight` | `blk.N.attn_{q,k}_norm.weight` | F32 | 1 + w |

The F32 transforms run in FP32, as Prism's converter does. With `distill.train.small`, the distilled values replace the checkpoint's, then go through the same transforms.

**Gated DeltaNet head order.** Hugging Face groups the value heads by key head: with n_k key heads and rep = n_v / n_k value heads per key head, grouped index g = k · rep + r. The GGUF tiles them instead: tiled index t = r · n_k + k. So GGUF row t holds Hugging Face row g, per head and per element of the head. That permutes the rows of:
- the V part of `in_proj_qkv`, after its 2 · n_k · d_k query and key rows;
- `in_proj_z`;
- the matching `conv1d` channels;
- `in_proj_a` and `in_proj_b`, `A_log` and `dt_bias` (one row per head).

Rows can be permuted freely, because they are outputs and the rotation acts on the inputs. Columns cannot: a column permutation does not commute with the rotation, so the input columns of `out_proj` (`ssm_out`) stay in the grouped order the codes were rounded in. The file says so with `prism.hadamard.gdn_v_grouped = true`, which Prism's runtime reads. The exporter states each permutation twice: once in the layout (`gguf_tensors`) and once as arithmetic on the tensor name (`gguf_source_rows`). Verification checks one against the other, so a wrong permutation fails at export, before anything runs in Prism.

### Dense models (`llama`, `qwen3`)

- **Names.** `blk.N.attn_{q,k,v,output}`, `blk.N.ffn_{gate,up,down}`, `blk.N.attn_norm`, `blk.N.ffn_norm`, plus `blk.N.attn_{q,k}_norm` for Qwen3. Norms are stored as w (copy).
- **RoPE permutation.** llama.cpp's `llama` architecture uses interleaved (NORM) rope, so Prism's converter permutes the q and k rows, and their biases, per head from the rotate-half order. Head h's GGUF row 2a + b holds Hugging Face row h · d + b · d/2 + a. `qwen3` uses NEOX rope and needs no permutation.
- **llama3 rope** adds the converter's `rope_freqs.weight`, a generated F32 constant.
- **Biases** are F32, outside `prism.hadamard`.
- **High-precision endpoints.** An untied BF16 head or embedding is written in the checkpoint's float type, outside `prism.hadamard`. A tied table is written once as `token_embd.weight`, with no `output.weight`, and llama.cpp reuses it as the head.

## Verification

`openbonsai export` writes to `<out>.tmp`, fsyncs it, verifies it, and renames it only if every check passes. A report goes to `<out>.export.json`. The checks:

1. Structure: GGUF v3, alignment 32, the planned tensor names, order, types, shapes and offsets, and a file size equal to the data start plus the data.
2. Every KV reads back equal to what was written. For a config with a published reference header (the 9B), the layout and the runtime KVs also equal the reference.
3. `weight_names` and `inverse_weight_names` cover every PTQ1_0 tensor exactly once, and the sign table covers every width.
4. Every PTQ1_0 block decodes to the permuted, canonical state.
5. Every F32 tensor is bit-identical to the transformed source.
6. A function check in FP64 on sample rows of a few layers (0, 3 and the last, for the 9B) and both endpoints. The runtime's computation, W_gguf · H(s ⊙ x) with the file's signs (and s ⊙ (H z) for the embedding), is compared with the fake-quant (C ⊙ s) Rᵀ x of the matching source rows. The worst relative error on the 9B was 1.1e-15.

`openbonsai inspect FILE.gguf` prints the tensor type counts and the metadata.

## The latent checkpoint

When Prism cannot load the architecture (Qwen2, a SentencePiece Mistral, an unsupported rope or bias), `export.format: auto` writes a folder instead. `export.format: ptq1_0-gguf` would refuse before any stage runs. The folder is `<run>/export/OpenBonsai-<model>-ternary/`:

| File | Contents |
|---|---|
| `model.safetensors`, `config.json`, generation config, tokenizer | loadable by plain transformers; every ternary tensor holds its dequantized weight (C ⊙ s) Rᵀ in the model's dtype |
| `ternary.safetensors` | the exact codes, fp16 scales, per-width signs and block, as a run state (below) |
| `small.safetensors` | with `distill.train.small`: the distilled high-precision tensors (FP32 masters and the dtype each was computed in) |
| `openbonsai.json` | what the folder is, why it is not a GGUF, the Hadamard block note, the distilled tensors and the SHA256 of every file |

`openbonsai eval --from <folder>/ternary.safetensors` installs exactly the distilled model, `small.safetensors` included. A GGUF can be written from it once Prism supports the architecture.

## The run state

The stages pass the ternary model between them as safetensors files (`openbonsai/formats/state.py`). Nothing in them depends on time or host, so equal inputs give byte-identical files.

**State** (`quantize/state.safetensors`, `endpoints/state.safetensors`, the per-layer `quantize/layers/NNNN.safetensors`):

| Tensor | Type | Shape |
|---|---|---|
| `codes::<key>` | uint8, 2-bit packed (or int8) | [⌈rows · cols / 4⌉] (or [rows, cols]) |
| `scales::<key>` | fp16 | [rows, cols / group] |
| `signs::<width>` | int8 | [width] |

- **Keys** are module names relative to the text model: `layers.{i}.{sub}` (for example `layers.3.mlp.down_proj`), `lm_head` and `embed_tokens`.
- **Codes** are in the rotated basis, row-major.
- **2-bit packing.** Four codes per byte, least significant first: 0 → `00`, +1 → `01`, −1 → `10`. `11` is invalid, and the padding of the last byte must be zero.
- **Metadata.** The safetensors header holds one entry, `openbonsai`, a canonical JSON object: `format` = `openbonsai.ternary-state.v1`, `block`, `group`, `codes` (`2bit` or `int8`), `shapes` per key, and `meta`.

**Distilled scales** (`distill/scales.safetensors`, format `openbonsai.ternary-scales.v1`):
- `scales::<key>` (fp16) replaces the state's scales.
- With `distill.train.small`, `small::<name>` holds an FP32 master of each trained high-precision tensor, and the metadata's `small_dtypes` records the dtype each was computed in.
- The metadata also records the digests of the quantize and endpoints outputs it was trained on.

**Signs** (`signs.safetensors` in the run directory, format `openbonsai.signs.v1`) hold `signs::<width>` and the block. The file is rewritten only when `rotation.signs` changes.

**Legacy states.** The research scripts' `.pt` files (`ternary_state.pt`, `endpoint_state.pt`) load read-only for evaluation. Only a legacy state with one sign vector per width converts to this format; per-module signs cannot be expressed in Prism's layout.

### The run directory

```
runs/<name>/
  run.yaml                 the resolved config (defaults, YAML file, --set overrides)
  signs.safetensors        sign vectors per width
  digests.json             cached SHA256 of input files
  traces/traces.jsonl      the model's own answers (prompt + generated ids, with source, row, thinking, hit_cap)
  traces/summary.json      counts per prompt source; traces/cache/ holds each generated batch (resume)
  calib/tokens.npy         calibration blocks, int32 [N, seq_len], and sources.json
  quantize/state.safetensors   (layers/NNNN.safetensors while running)
  endpoints/state.safetensors  the ternary head and embedding (only these two keys)
  distill/teacher/         input_ids.npy and topk_NNNNNN.npz shards (int32 ids, fp16 log-probs)
  distill/scales.safetensors   (plus checkpoint.pt, the resumable training state, and train.jsonl, its log)
  export/<name>.gguf       and <name>.gguf.export.json (or the latent folder)
  eval/<source>.<suite>.json
  logs/<stage>.log
```

A stage is complete when `<stage>/DONE.json` exists. That file records the stage's config hash and the SHA256 of its output. A stage's hash covers the config sections it reads, the contents of the files it names, and its upstream stages' hashes and output digests. So a rerun skips a stage that is current, and reruns a stage whose config, input files or upstream output changed. A stage refuses to run on a missing, disabled or stale upstream output.
