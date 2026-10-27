# P2 export parity: package exporter vs research/experiments/export_gguf.py

Date: 2026-10-27. Branch `framework/v1` at 049eee2 (P2 commits 4155f24..049eee2).
CPU only (`CUDA_VISIBLE_DEVICES=-1`; the scripts assert `torch.cuda.device_count() == 0`).
torch 2.14.1+cu130, transformers 5.20.2, numpy 2.5.4.

## Result

| Check | Inputs | Result |
|---|---|---|
| Byte parity, synthetic 9B state (`tests/test_export_parity.py`, slow) | `D:/openbonsai/runs/gguf-test/synthetic` (RTN shared-sign state with edge cases, e2e scales) | **identical**: both 1,997,486,976 B, SHA-256 `f41f302d7e4dc1d42d5f057207bd16c8b63b060dd63ebb8c1bfe15389868a1cb`; 1 passed in 172 s |
| Byte parity, real shared-sign 9B state | GPTQ shared-256 decoder + shared-256 endpoints + e2e-kd-shared-v1 scales | **identical** to `D:/openbonsai/runs/2026-10-27-gguf-shared-v1/OpenBonsai-Qwen3.5-9B-PTQ1_0.gguf` (the research file Prism's CPU runtime loads, 97/97 checks): SHA-256 `39a319bdc549255aef49cc84446615869923fedfeaf44d77a4dc23b696e7cbc0` |
| Package verification of both files | | 14/14 checks each (below) |
| Our GGUF reader vs the sibling session's decoder (`bonsai-start/gguf_ptq1.py`, read-only import) | the official `Ternary-Bonsai-2-9B-PTQ1_0.gguf` (SHA-256 `3561df7a...`) | **all equal**: 202 PTQ1_0 tensors (codes and fp16 scale bits in HF row order) and 225 F32 tensors (HF conventions) |
| Same | the synthetic package export above | **all equal**: 202 + 225 tensors |

## What the synthetic state exercises

`make_synthetic_state.py --edge-cases`: 13,736 negative-scale groups (normalized by negating the
group's codes), 1,616 zero-scale groups, 808 of them with nonzero codes (`-0.0` scales after a sign
flip; the codes are cleared, as `quantize_row_ptq1_0_ref` writes them). Both exporters report the
same counts; 8,946,450,432 ternary weights. The real state has none of these edge cases.

## Package verification (both 9B files)

1. GGUF v3, alignment 32; 427 tensors with the planned names, order, types, `ne` and offsets; file size = data start + data.
2. 49 KVs read back equal the KVs written.
3. `weight_names` (201) + `inverse_weight_names` (1) cover the 202 PTQ1_0 tensors once; the sign table covers widths 4096 and 12288; `general.file_type` = 143.
4. Against the official header (the config is the 9B one, so the reference applies): KV keys, order and types identical; tensor layout and offsets identical; data section starts at 11,039,616 like the official file; the only KV value differences are `general.name` and `prism.hadamard.sign_values` (by design).
5. Every PTQ1_0 block decoded equals the permuted, canonical state (8.9e9 weights, 74 s).
6. All 225 F32 tensors are bit-identical to the transformed snapshot tensors.
7. FP64 function check on sample rows of 22 tensors (layers 0, 3, 31 and both endpoints): the runtime's `W_gguf @ H(s * x)` (file signs, popcount Hadamard) vs the fake-quant `(C * s) R^T x` of the matching source rows; worst relative error 1.1e-15.

The package derives the whole layout and every KV from the adapter and the model config; the
official header (`openbonsai/formats/assets/ternary-bonsai-2-9b.header.json`) is only compared
against. Timing on this machine: package write 41 s, verification about 75 s (the research
exporter: write 40 s).

## How to reproduce

```
cd D:/openbonsai/work/framework-v1
OPENBONSAI_SLOW=1 CUDA_VISIBLE_DEVICES=-1 OPENBONSAI_PARITY_OUT=<dir with 4 GB free> \
  D:/openbonsai/.venv/Scripts/python.exe -m pytest tests/test_export_parity.py -s
```

The test runs the research exporter's own source in a subprocess (with its
`CUDA_VISIBLE_DEVICES=""` line replaced by `-1`, since an empty value unsets the variable on
Windows), exports the same state with `openbonsai.stages.export.export_gguf`, and compares the
files byte for byte. The real-state comparison and the decoder cross-checks were run from
scratch scripts (`p2_real_parity.py`, `p2_sibling_check.py` in the session scratchpad); their
inputs are listed above.
