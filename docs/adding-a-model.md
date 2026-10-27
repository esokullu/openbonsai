# Adding a model

The pipeline (traces, calibrate, quantize, endpoints, distill, export, eval) is architecture-agnostic. Everything it needs to know about a model comes from a `ModelAdapter` (`openbonsai/adapters/base.py`), picked by the config's `model_type` (`openbonsai/adapters/registry.py`). There are two adapters today:

| Adapter | Model types | File |
|---|---|---|
| `qwen3_5` | `qwen3_5`, `qwen3_5_text` (Gated DeltaNet + full-attention hybrid) | `adapters/qwen3_5.py`, `adapters/qwen3_5_gguf.py` |
| `dense` | `llama`, `mistral`, `qwen2`, `qwen3` (pre-norm attention + SwiGLU) | `adapters/dense.py` |

Which case you are in:

- **A dense model whose `model_type` is in the table:** nothing to write. Run it.
- **A dense model with a new `model_type`** (same seven Linears per layer): add one family row to `FAMILIES` in `adapters/dense.py` and one tiny config to the tests.
- **A new architecture** (other Linears, other layer kinds, a recurrent block): write an adapter, about 150 lines plus its GGUF layout if Prism can load it.

Status: only Qwen3.5-9B has been compressed and measured end to end. The dense adapter is tested on tiny random Llama, Mistral, Qwen2 and Qwen3 models on CPU, including loading tiny GGUFs in Prism's runtime; no real dense model has been run yet. Whatever the adapter, run the [sensitivity scan](#choosing-the-high-precision-tensors-a-sensitivity-scan) on the real model before trusting its `keep` list.

## What an adapter provides

| Member | What it says | Dense adapter | Qwen3.5 adapter |
|---|---|---|---|
| `load(path, ...)` | `(model, tokenizer)`, eval mode, grads off | `AutoModelForCausalLM` | `Qwen3_5ForConditionalGeneration` (vision tower dropped) or `Qwen3_5ForCausalLM` |
| `text_model(model)` | the module with `.embed_tokens`, `.layers`, `.norm` | `model.model` | `model.model.language_model` (or `model.model`) |
| `head(model)` | the LM head | `model.lm_head` | `model.lm_head` |
| `layer_plan(i)` | `LayerPlan(kind, groups, keep)` (below) | `[q,k,v] -> [o] -> [gate,up] -> [down]`, keep nothing | two plans, by layer type |
| `zero_centered_norm` | the norm is `(1 + w) * x_hat` instead of `w * x_hat`; sets the embedding's GPTQ weighting | `False` for all four families | `True` |
| `tied` | embedding and head share one table | from the config | from the config |
| `add_bos`, `bos_id(tok)` | every calibration block and eval sequence starts with BOS | Llama, Mistral | no |
| `chat_kwargs(thinking=)`, `think_end` | the chat template's thinking switch | `enable_thinking` for Qwen3 | `enable_thinking` |
| `eos_ids(tok, model)` | generation stops | generation config + tokenizer EOS | same |
| `checkpoint_name(key)` | checkpoint tensor name of a text-model key | `model.` + key | `model.language_model.` + key |
| `gguf_arch`, `gguf_unsupported()`, `gguf_tensors()`, `gguf_kvs()` | the Prism PTQ1_0 GGUF layout, or why there is none | per family | `qwen35` |
| `empty_model()` | the model on the meta device (for `eval --from X.gguf`) | yes | yes |

The adapter never declares rotation geometry. The core derives it from the planned Linears (`core/geometry.py`):

- **Block:** the largest power of two ≤ 4096 that divides every rotated input width.
- **Signs:** one sign vector per width.
- **Refusal:** a width that is not a multiple of 128 is refused.

Prism's runtime builds its rotation for any power-of-two block, but it has only been run with block 4096 (the official Ternary-Bonsai-2-9B and our 9B export). Quantize logs a note for any other block, and the export records it. Some examples:

| Model shape | Rotated widths | Block |
|---|---|---|
| Qwen3.5-9B | 4096, 12288 | 4096 |
| Qwen3.5-4B | 2560, 4096, 9216 | 512 |
| Llama 3.1 8B, Mistral 7B | 4096, 14336 | 2048 |
| Llama 3.2 3B | 3072, 8192 | 1024 |
| Qwen3-0.6B | 1024, 2048, 3072 | 1024 |
| Qwen2.5-7B | 3584, 18944 | 512 |
| hidden 2560, MLP 10496 | 2560, 10496 | 256 |

### The layer plan

`LayerPlan.groups` is the sequential GPTQ order inside a layer. Each group is a tuple of Linears that read the same input: one forward pass collects the Hessian for the whole group. The groups go in data-flow order, so a later group's Hessian sees the earlier groups already quantized. For a standard attention block:

```python
PLAN = (("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"),   # all read the attention norm's output
        ("self_attn.o_proj",),                                           # reads the attention output
        ("mlp.gate_proj", "mlp.up_proj"),                                # both read the MLP norm's output
        ("mlp.down_proj",))                                              # reads act(gate) * up
```

`LayerPlan.keep` lists the Linears that stay high precision. Every `nn.Linear` in a layer must be in exactly one group or in `keep`. `planned_linears` enforces this, and `tests/test_dense.py::test_every_linear_of_every_registered_adapter_is_planned_or_kept` checks it for every registered model type. A Linear's bias always stays high precision: it is kept inside `TernaryLinear` in the checkpoint dtype and written as F32 in a GGUF.

Non-Linear tensors (norms, Qwen3's `q_norm`/`k_norm`, Qwen3.5's conv1d, `A_log`, `dt_bias`) are never quantized. With `distill.train.small` they are trained through FP32 masters.

### Embedding, head and tied tables

`endpoints.head` and `endpoints.embedding` take `ternary`, `bf16` or `auto` (the default).

- **Head:** the LM head gets GPTQ with the Hessian of the final hidden states.
- **Embedding:** the token embedding gets GPTQ with `H = diag(gain²)`. `gain` is layer 0's entry-norm weight, or `1 + w` for zero-centered norms. This is why the norm convention matters.
- **Tied tables** (Qwen3 0.6B/1.7B/4B, Llama 3.2 1B/3B):
  - `auto` keeps the shared table in high precision, and no GPTQ runs on it.
  - Asking for `ternary` is refused: one joint GPTQ for a table that is both the embedding and the head is not implemented yet.
  - In a GGUF the table is `token_embd.weight` in the checkpoint's float type (BF16), with no `output.weight`. llama.cpp then reuses the embedding as the head. The table is outside `prism.hadamard`, so it is not rotated.

## A new dense model type: one family row

```python
Family("llama", "llama", add_bos=True, thinking=False, rope_permute=True, pre="llama-bpe",
       biases=(("attention_bias", ATTN), ("mlp_bias", MLP))),
```

| Field | Meaning |
|---|---|
| `gguf_arch` | the llama.cpp architecture Prism's converter writes for this `model_type` (`conversion/__init__.py` in Prism) |
| `prism_hadamard` (derived) | whether Prism's loader accepts `prism.hadamard` metadata for that arch. Prism (c9c34bc, `llama-model.cpp`) verifies only `llama`, `qwen3`, `qwen3moe`, `qwen35`, `qwen35moe`, `qwen3next` and `dspark`; every other arch is refused at load. No GGUF means a latent export |
| `add_bos` | blocks and eval sequences start with BOS; also the GGUF's `add_bos_token` when the tokenizer config does not set it |
| `thinking` | the chat template understands `enable_thinking` (asking for thinking otherwise is an error) |
| `rope_permute` | llama.cpp's `llama` arch uses interleaved (NORM) rope, so Prism's converter permutes q/k rows (and biases) per head from rotate-half order. `qwen3` uses NEOX rope: no permutation |
| `pre` | `tokenizer.ggml.pre` when the tokenizer's probe hash is not one Prism's converter knows (`KNOWN_PRE`) |
| `zero_centered_norm`, `qk_norm` | norm convention; per-head q/k norms |
| `biases`, `gguf_bias_slots` | which Linears have a bias under which config flag; which of those the arch's GGUF loader has a slot for |

Then add a tiny config to `CONFIGS` in `tests/test_dense.py`, and an end-to-end case if it exports differently.

The current table, and what each family exports to:

| `model_type` | Examples | GGUF arch | Prism loads `prism.hadamard` | Export | BOS | Thinking | Norm | Notes |
|---|---|---|---|---|---|---|---|---|
| `llama` | Llama 2/3.x; Llama-architecture checkpoints such as Nanbeige4-style models published as `LlamaForCausalLM` | `llama` | yes | PTQ1_0 GGUF | yes | no | `w` | q/k rows permuted; llama3 rope writes `rope_freqs.weight`; q/k/v/o and MLP biases have slots |
| `mistral` | Mistral 7B, Mistral-Nemo | `llama` | yes | PTQ1_0 GGUF with a byte-level BPE tokenizer (Nemo); latent with a SentencePiece one (7B v0.x) | yes | no | `w` | the `llama` arch has no sliding window: the GGUF matches the model only within `sliding_window` tokens |
| `qwen2` | Qwen2, Qwen2.5 | `qwen2` | **no** | latent checkpoint | no | no | `w` | q/k/v biases |
| `qwen3` | Qwen3 (0.6B-4B tied) | `qwen3` | yes | PTQ1_0 GGUF | no | yes | `w` | per-head q/k norms; a tied table stays BF16 |
| `qwen3_5` (own adapter) | Qwen3.5 | `qwen35` | yes | PTQ1_0 GGUF, untied only | no | yes | `1 + w` | tied sizes (2B, 4B) cannot be written: Prism's version 1 metadata forbids a tied output |

`NanbeigeForCausalLM` (the looped Nanbeige4.2) is Prism's `nanbeige` arch. It is not hadamard-verified and needs remote code, so it is not covered.

The GGUF export of a dense model also needs:

- a byte-level BPE `tokenizer.json`;
- rope type `default`, `linear` or `llama3`;
- no sliding-window layers (Qwen3);
- a slot for every bias.

`adapter.gguf_unsupported(model_dir)` names whatever is missing. `export.format: auto`, the default, then writes the latent checkpoint instead and logs why. An explicit `export.format: ptq1_0-gguf` is refused before any stage runs.

### The latent checkpoint

When no PTQ1_0 GGUF can be written, `export` writes `<run>/export/OpenBonsai-<model>-ternary/`:

- `model.safetensors`, `config.json`, the generation config and the tokenizer. Plain transformers loads the folder. Every ternary tensor holds its dequantized weight `(C * s) R^T` in the model dtype; the other tensors are the checkpoint's (or the distilled ones).
- `ternary.safetensors`: the exact codes, fp16 scales, per-width signs and block. `openbonsai eval --from <folder>/ternary.safetensors --install ternary` installs exactly what distillation evaluated, and a GGUF can be written from it once Prism loads the architecture.
- `small.safetensors` (with `distill.train.small: true`): the distilled high-precision tensors (norms, `A_log`, ...) as FP32 masters with the dtype each was computed in. `eval --from <folder>/ternary.safetensors` applies them, so it scores the distilled model rather than the original checkpoint's tensors.
- `openbonsai.json`: what the folder is, why it is not a GGUF, the Hadamard block note, the distilled tensors and the SHA256 of every file.

Verification re-reads every saved tensor, the state and the distilled tensors bit for bit.

`export --out DIR` names the checkpoint folder itself, not a parent folder. The folder is built next to `DIR` and moved into place. An existing `DIR` is replaced only when it is empty or a previous latent checkpoint (its `openbonsai.json` says so); any other folder, or a file, is refused and left untouched. `openbonsai run` checks `export.out` before the first stage.

## A new architecture

Write `openbonsai/adapters/<name>.py`, subclass `ModelAdapter`, set `name` and `model_types`, and `register()` it in `registry.py`. The minimum:

```python
class MyAdapter(ModelAdapter):
    name = "mine"
    model_types = ("mine",)
    zero_centered_norm = False
    modeling_module = "transformers.models.mine.modeling_mine"   # imported with CUDA-only kernels hidden on the CPU

    def load(self, path, *, revision=None, device="cpu", dtype=torch.bfloat16, attn="sdpa",
             local_files_only=False, **options):
        self.prepare_device(device)
        model = MyForCausalLM.from_pretrained(path, revision=revision, dtype=dtype, device_map=str(device),
                                              attn_implementation=attn, local_files_only=local_files_only)
        model.eval(); model.requires_grad_(False)
        self.text_model(model).config.use_cache = False
        return model, self.load_tokenizer(path, revision=revision, local_files_only=local_files_only)

    def text_model(self, model):
        return model.model

    def checkpoint_name(self, key):
        return key if key.startswith("lm_head.") else "model." + key

    def layer_plan(self, i):
        return LayerPlan(kind="dense", groups=PLAN, keep=("mlp.router",))
```

Without `gguf_arch` the export writes the latent checkpoint. For a Prism PTQ1_0 GGUF, also implement:

- `gguf_tensors()`: the file's tensors as `TensorMap` entries (GGUF name, type, `ne`, source key, F32 transform, row permutation, fold or inverse role).
- `gguf_kvs()`: the metadata, with the hyperparameter KVs Prism's converter writes for the arch and the `prism.hadamard.*` block.
- `gguf_source_rows()` / `gguf_f32_op()`: the same row order and transforms stated independently, by arithmetic from the tensor's name. Verification checks the layout against this rule, so a wrong permutation fails before anything runs in Prism.
- `empty_model()` for `eval --from X.gguf`.

Check the arch is in Prism's `prism.hadamard` switch, and that every folded weight is a `build_lora_mm` input in the arch's graph (that is what the switch certifies). See `adapters/dense.py` and `adapters/qwen3_5_gguf.py` for the two layouts.

## Choosing the high-precision tensors: a sensitivity scan

An adapter's `keep` list, and whether the endpoints stay BF16, should come from measurement, not guesses. A sensitivity scan rounds one part of the model at a time, leaves everything else in BF16, and scores the result. It needs no training, and each row takes about a minute on one GPU for a 9B model.

### The recipe

1. **Get a quantized state** with the method you will ship: `openbonsai run --model <model> --config <config> --out runs/scan --until quantize`. (A cheaper first pass rounds to nearest with rotation and no calibration, as `research/experiments/quick_eval.py` does. It overstates the damage, but it gives a quick first ranking.)
2. **Score the BF16 model once** on a fixed held-out set: `compare` blocks for any model, or the `dev` suite for the Qwen3.5 tokenizer.
3. **Only one family.** For each family of linear layers (`mlp.gate_proj`, `linear_attn.out_proj`, ...), load the BF16 model, write the rounded weights of that family only, and score. Do the same per layer type (all attention layers, all MLPs) and per depth (the first N layers, the last N, layer 0 alone).
4. **Everything but one family** (the complement): round all families except one. If keeping one family barely helps, that family is not the problem, however bad its "only" row looked.
5. **Per layer, for the worst family.** Round that family in one layer at a time. A single outlier layer is common: in Qwen3.5-9B it is layer 0's Gated DeltaNet `out_proj`.
6. **Write down each family's share of the decoder's parameters** next to its NLL delta.

With the package, after a run has finished `quantize` (steps 2 and 3; the filters below the code give the other rows):

```python
from openbonsai.config import DEFAULTS, check_keys, deep_merge
from openbonsai.eval import suites as S
from openbonsai.rundir import RunDir
from openbonsai.stages.common import StageContext, apply_dense, free
from openbonsai.stages.evaluate import compare_blocks

run = RunDir("runs/scan")
ctx = StageContext(check_keys(deep_merge(DEFAULTS, run.load_config())), run)
adapter, state = ctx.adapter(), ctx.state_through("quantize")
model, tok = ctx.load_model()
blocks = compare_blocks(ctx, tok)
reference = S.compare_reference(model, blocks, topk=64)
del model
families = sorted({key.split(".", 2)[2] for key in state.tensors if key.startswith("layers.")})
for family in families:                                   # only <family>
    model, _ = ctx.load_model()
    apply_dense(adapter, model, [r for r in state.records() if r[0].endswith("." + family)])
    r = S.compare(model, blocks, reference)
    print(family, round(r["delta_nll"], 4), round(r["top1_agreement"], 3), round(r["kl_top64"], 4))
    del model
    free(ctx.device)
```

- **One whole layer:** filter on `r[0].startswith(f"layers.{i}.")`.
- **One family in one layer:** `r[0] == f"layers.{i}.{family}"`.
- **Complement:** use `not r[0].endswith("." + family)`.

The state comes from sequential GPTQ, so each family's codes were fitted on inputs from an already-rounded prefix. Read the scan as the sensitivity of that state's families, not of independently rounded ones.

### Decision rules

Keep a linear layer in high precision only if **all three** hold:

1. **The runtime format has a float slot for it.** A kept Linear must be stored as F32 (or another float type) in the target GGUF. Prism stores Qwen3.5's `in_proj_a`/`in_proj_b` as F32 (`ssm_alpha`/`ssm_beta`), so they can be kept. A family Prism expects as PTQ1_0 cannot be kept without a new layout, and a new layout means a file Prism's released runtime may not load.
2. **It is small.** Keeping it should cost well under 1% of the ternary size. Qwen3.5's `in_proj_a`/`in_proj_b` are about 0.1% of the decoder.
3. **Its damage is out of proportion to its size,** in both the "only" and the complement scans.

Everything else stays ternary, including large families with large deltas. Keeping them would cost most of the compression; they are where GPTQ, endpoint calibration and scale distillation have to do their work. For a sensitive family, first try what the pipeline can change: more or better calibration data, a scan of the GPTQ state instead of the rounded-to-nearest one, and distillation.

For the endpoints:
- Score `endpoints.head` and `endpoints.embedding` set to `bf16` and to `ternary` separately; both are config settings, so no code is needed.
- A 2 GB-class file needs both ternary, because the embedding and head of a large-vocabulary model are as big as the ternary decoder.
- A tied table stays high precision (`auto`).

Accept a decision only on the end-to-end numbers: `compare` (and your own task) after distillation. In our runs, local error was a poor predictor of the final model.

### What the scans found on Qwen3.5-9B

Rounded to nearest with rotation, no calibration (`research/experiments/out/`; BF16 dev NLL 2.423, ARC-Easy 19/20):

| Rounded alone | Dev NLL | ARC /20 |
|---|---|---|
| Gated DeltaNet `out_proj`, all layers | 4.253 | 13 |
| `gate_proj` + `up_proj` | 3.082 | 18 |
| `down_proj` | 2.638 | 19 |
| `in_proj_qkv` / `in_proj_z` | 2.511 / 2.487 | 17 / 20 |
| `in_proj_a` + `in_proj_b` | 2.412 | 19 |
| Layer 0's `out_proj` only | 4.013 | 15 |
| `out_proj` of layer 1, 2, 4, 8, 16, 24 or 30 only | 2.417 to 2.445 | 19 to 20 |
| Everything except `out_proj`, `in_proj_a`, `in_proj_b` | 6.647 | 6 |

The earlier learned-factor quantizer (`research/runs/2026-10-26-5090-diagnosis/report.md`; NLL deltas against BF16) put the same two families first, `out_proj` and `gate_proj`:

| Rounded alone | Dev NLL delta | ARC /20 |
|---|---|---|
| `up_proj` | +0.92 | 16 |
| `gate_proj` | +3.85 | 13 |
| `down_proj` | +1.58 | 18 |
| `in_proj_qkv` | +0.40 | 16 |
| `in_proj_z` | +0.53 | 10 |
| `out_proj` (Gated DeltaNet output) | +4.90 | 9 |
| All full-attention layers | +1.97 | 8 |

What we did with it:
- `in_proj_a`/`in_proj_b` stay F32. Rounding them costs nothing measurable, but they are tiny and Prism's format stores them as F32 anyway.
- `out_proj`, and layer 0's in particular, is the most sensitive family. Prism's format has no float slot for it, so it stays ternary, and GPTQ plus distillation handle it: the whole model reaches dev NLL 2.892.
- Keeping all of `out_proj` (the complement row) still leaves 6.6, so no single family is the problem under plain rounding: the damage is spread out, and calibration has to fix it.

## Worked example 1: Qwen3.5 (hybrid)

Qwen3.5 interleaves Gated DeltaNet layers (`linear_attention`) with full-attention layers, one every fourth layer. So its adapter has two plans:

```python
"linear_attention": (("linear_attn.in_proj_qkv", "linear_attn.in_proj_z"), ("linear_attn.out_proj",),
                     ("mlp.gate_proj", "mlp.up_proj"), ("mlp.down_proj",)),
"full_attention":   (("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"), ("self_attn.o_proj",),
                     ("mlp.gate_proj", "mlp.up_proj"), ("mlp.down_proj",)),
keep (linear_attention): ("linear_attn.in_proj_a", "linear_attn.in_proj_b")
```

- **Groups:** `in_proj_qkv` and `in_proj_z` read the same normed input. `out_proj` reads the gated recurrent output.
- **Keep:** `in_proj_a`/`in_proj_b` produce the per-head decay and beta gates. They are tiny, and Prism stores them F32 (`ssm_alpha`/`ssm_beta`).
- **Untouched:** conv1d, `A_log` and `dt_bias` are not Linear.
- **Norms** are zero-centered (`(1 + w)`), so the embedding's GPTQ weight is `(1 + w)²` of layer 0's input norm.
- **The GGUF layout** (`adapters/qwen3_5_gguf.py`) is the hard part:
  - Value heads are grouped by key head in HF and tiled in GGUF, so `in_proj_qkv` V rows, `in_proj_z` rows, conv1d channels, `A_log`, `dt_bias` and the alpha/beta rows are permuted.
  - `out_proj` columns stay grouped (`prism.hadamard.gdn_v_grouped`).
  - F32 tensors get `1 + w`, `-exp(A_log)` and a conv squeeze.
  - The independent row rule in `gguf_source_rows` restates the permutation as arithmetic.
  - For the 9B config the file layout equals the official Ternary-Bonsai-2-9B header.

## Worked example 2: a dense Llama/Nanbeige-style model

A Llama-architecture checkpoint (`model_type: llama`, also how Nanbeige4-style models are published) needs no code:

```
openbonsai run --model <dir or HF id> --config examples/qwen3_5-9b.yaml --out runs/my-llama \
    --set 'eval.suites=[compare]' --set export.format=auto --set model.revision=null
```

- `dev` and `extended` need the Qwen3.5 tokenizer, so use `compare`, which scores against the model's own BF16.
- The 9B example pins Qwen's commit, so `model.revision=null` (or your own commit) is needed for a Hugging Face id. A local folder ignores it.
- The 9B example asks for ternary endpoints and a GGUF explicitly. For a tied model, also pass `--set endpoints.head=auto --set endpoints.embedding=auto`.

What happens:

- **Plan:** `[q,k,v] -> [o] -> [gate,up] -> [down]` in every layer, nothing kept. Biases, if the config has `attention_bias` or `mlp_bias`, stay high precision.
- **Geometry:** from the widths, e.g. block 2048 for Llama 3.1 8B (14336 = 2048 x 7). The export notes that Prism has been verified at block 4096 only.
- **BOS:** every calibration block starts with BOS (`add_bos`). The chat template renders traces and chat calibration data.
- **Endpoints:** with an untied head (Llama 3.1 8B), head and embedding are ternary. The embedding's GPTQ weight is `w²` of layer 0's input norm. With a tied table (Llama 3.2 1B/3B) the table stays BF16.
- **Export:** a `llama` PTQ1_0 GGUF.
  - q/k rows (and biases) are permuted per head as Prism's converter does. Llama 3.x's llama3 rope adds the converter's `rope_freqs.weight`.
  - Every decoder Linear and the ternary head are in `prism.hadamard.weight_names`, and a ternary embedding is the inverse-lookup table.
  - The tokenizer must be byte-level BPE (Llama 3 is). Unknown pre-tokenizer hashes get the family default `llama-bpe`, with a note to check tokenization in Prism.

Checked on CPU with Prism's own runtime: tiny random Llama (untied, biases, llama3 rope) and Qwen3 (tied, BF16 table) GGUFs load in Prism's `llama-perplexity`. Its full-vocabulary log-probs match the FP32 torch model of the same state within 0.06 / 0.015 nats at most (Prism quantizes activations to 8 bits). An export without the q/k permutation is off by about 1 nat. Run it yourself with `OPENBONSAI_PRISM_BIN=<prism build>/bin pytest tests/test_dense.py -k prism`.

## Checklist

1. A tiny config in `CONFIGS` (`tests/test_dense.py`): the conformance test then checks the plan covers every Linear, the biases match the family rule, and geometry derives.
2. An end-to-end case with a tiny random model (`tests/tiny_models.py:save_tiny_dense` shows how) running `openbonsai run --config examples/tiny-dense.yaml` in under 60 s, exporting a GGUF or a latent checkpoint.
3. For a GGUF:
   - the layout's permutations against an independent statement of Prism's converter (`test_llama_rope_permutation_matches_prisms_converter_and_the_independent_rule`);
   - `eval --from X.gguf` reproducing the distilled model's numbers exactly;
   - if you have a Prism build, the runtime check above.
4. A sensitivity scan on the real model before you settle `keep`.
