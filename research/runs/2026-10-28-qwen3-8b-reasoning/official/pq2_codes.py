"""Count the 2-bit codes of every PQ2_0 (type 142) tensor in a GGUF: q=0 -> -1, 1 -> 0, 2 -> +1, 3 -> +2.
Usage: pq2_codes.py FILE OUT.json"""
import json
import struct
import sys

import numpy as np

SC = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
BLOCK, QK = 34, 128


def header(f):
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
    rd("<I")
    nt, nkv = rd("<Q"), rd("<Q")
    kv = {}
    for _ in range(nkv):
        k = rstr()
        kv[k] = rval(rd("<I"))
    tensors = []
    for _ in range(nt):
        name = rstr()
        ne = [rd("<Q") for _ in range(rd("<I"))]
        tensors.append((name, rd("<I"), ne, rd("<Q")))
    align = int(kv.get("general.alignment", 32))
    start = (f.tell() + align - 1) // align * align
    return tensors, start


def main(path, out):
    with open(path, "rb") as f:
        tensors, start = header(f)
    mm = np.memmap(path, np.uint8, "r")
    total = np.zeros(4, np.int64)
    per = {}
    for name, gtype, ne, off in tensors:
        if gtype != 142:
            continue
        nblocks = int(np.prod(ne)) // QK
        blocks = mm[start + off:start + off + nblocks * BLOCK].reshape(nblocks, BLOCK)
        counts = np.zeros(4, np.int64)
        for a in range(0, nblocks, 1 << 20):
            qs = np.asarray(blocks[a:a + (1 << 20), 2:])
            for shift in (0, 2, 4, 6):
                counts += np.bincount(((qs >> shift) & 3).ravel(), minlength=4)
        per[name] = counts.tolist()
        total += counts
    n = int(total.sum())
    res = {"tensors": len(per), "weights": n, "codes": {"-1": int(total[0]), "0": int(total[1]), "+1": int(total[2]),
                                                       "+2 (unused for ternary)": int(total[3])},
           "zero_fraction": float(total[1] / n), "per_tensor": per}
    with open(out, "w", encoding="utf-8") as o:
        json.dump(res, o, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "per_tensor"}))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
