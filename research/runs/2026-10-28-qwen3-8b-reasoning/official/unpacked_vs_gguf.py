"""Is the unpacked F16 checkpoint the GGUF's model? Compares, tensor by tensor, the weight count and the number of
zero weights that f16_check.py found in the unpacked checkpoint with the PQ2_0 code counts of the GGUF
(pq2_codes.py). It does not compare signs or scales.

    python -I unpacked_vs_gguf.py unpacked_f16_check.json pq2_codes.json
"""
import json
import re
import sys

NAMES = {"attn_q": "self_attn.q_proj", "attn_k": "self_attn.k_proj", "attn_v": "self_attn.v_proj",
         "attn_output": "self_attn.o_proj", "ffn_gate": "mlp.gate_proj", "ffn_up": "mlp.up_proj",
         "ffn_down": "mlp.down_proj"}


def hf_name(name: str) -> str:
    if name == "output.weight":
        return "lm_head"
    if name == "token_embd.weight":
        return "model.embed_tokens"
    m = re.match(r"blk\.(\d+)\.(\w+)\.weight", name)
    return f"model.layers.{m[1]}.{NAMES[m[2]]}"


unpacked = json.load(open(sys.argv[1], encoding="utf-8"))["ternary_per_module"]
gguf = json.load(open(sys.argv[2], encoding="utf-8"))["per_tensor"]
same = 0
for name, codes in gguf.items():          # codes: counts of -1, 0, +1, +2
    r = unpacked[hf_name(name)]
    n = r["shape"][0] * r["shape"][1]
    zeros = round(r["zero_fraction"] * n)
    if n == sum(codes) and zeros == codes[1]:
        same += 1
    else:
        print(f"differs: {name}: unpacked {n} weights, {zeros} zero; GGUF {sum(codes)}, {codes[1]}")
print(f"{same} of {len(gguf)} tensors: the same weight count and zero count")
sys.exit(0 if same == len(gguf) == len(unpacked) else 1)
