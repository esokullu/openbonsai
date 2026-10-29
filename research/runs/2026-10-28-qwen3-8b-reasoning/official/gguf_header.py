"""Dump a GGUF header (KVs, tensor types and shapes) without decoding tensor data. Usage: gguf_header.py FILE OUT.json"""
import json
import struct
import sys
from collections import Counter

SC = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


def main(path, out):
    with open(path, "rb") as f:
        def rd(fmt):
            return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]

        def rstr():
            return f.read(rd("<Q")).decode("utf-8", "replace")

        def rval(t):
            if t == 8:
                return rstr()
            if t == 9:
                et, n = rd("<I"), rd("<Q")
                return [rval(et) for _ in range(n)]
            return rd(SC[t])

        assert f.read(4) == b"GGUF"
        version, nt, nkv = rd("<I"), rd("<Q"), rd("<Q")
        kv = {}
        for _ in range(nkv):
            k = rstr()
            v = rval(rd("<I"))
            kv[k] = v if not isinstance(v, (list, str)) or len(v) <= 16 else f"<{type(v).__name__} of {len(v)}>"
        tensors = []
        for _ in range(nt):
            name = rstr()
            ne = [rd("<Q") for _ in range(rd("<I"))]
            gtype, off = rd("<I"), rd("<Q")
            tensors.append({"name": name, "type": gtype, "ne": ne})
    types = Counter(t["type"] for t in tensors)
    by_role = {}
    for t in tensors:
        role = t["name"].split(".")[-2] if t["name"].startswith("blk.") else t["name"]
        by_role.setdefault(role, set()).add(t["type"])
    res = {"version": version, "tensors": nt, "types": {str(k): v for k, v in sorted(types.items())},
           "type_by_role": {k: sorted(v) for k, v in sorted(by_role.items())},
           "non_block_tensors": [t for t in tensors if not t["name"].startswith("blk.")],
           "blk0": [t for t in tensors if t["name"].startswith("blk.0.")], "kv": kv}
    with open(out, "w", encoding="utf-8") as o:
        json.dump(res, o, indent=1)
    print(json.dumps({k: res[k] for k in ("version", "tensors", "types", "type_by_role")}))
    print(json.dumps({k: v for k, v in kv.items() if not k.startswith("tokenizer.")}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
