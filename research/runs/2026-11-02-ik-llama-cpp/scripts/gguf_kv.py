"""Print the KV metadata of a GGUF file (arrays summarized). Usage: python -I gguf_kv.py FILE [KEY ...]"""
import struct
import sys

SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


def read_str(f):
    (n,) = struct.unpack("<Q", f.read(8))
    return f.read(n).decode("utf-8", errors="replace")


def read_val(f, t):
    if t in SCALAR:
        fmt = SCALAR[t]
        return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]
    if t == 8:
        return read_str(f)
    if t == 9:
        (at,) = struct.unpack("<I", f.read(4))
        (n,) = struct.unpack("<Q", f.read(8))
        return [read_val(f, at) for _ in range(n)]
    raise ValueError(f"unknown GGUF type {t}")


def main():
    path, keys = sys.argv[1], set(sys.argv[2:])
    with open(path, "rb") as f:
        assert f.read(4) == b"GGUF"
        version, n_tensors, n_kv = struct.unpack("<IQQ", f.read(20))
        print(f"version {version}, tensors {n_tensors}, kvs {n_kv}")
        for _ in range(n_kv):
            key = read_str(f)
            (t,) = struct.unpack("<I", f.read(4))
            val = read_val(f, t)
            if keys and key not in keys:
                continue
            if isinstance(val, list) and not keys:
                head = val[:4]
                print(f"{key} = array[{len(val)}] {head}{' ...' if len(val) > 4 else ''}")
            elif isinstance(val, str) and len(val) > 200 and not keys:
                print(f"{key} = str[{len(val)}] {val[:120]!r} ...")
            else:
                print(f"{key} = {val!r}" if not isinstance(val, list) else f"{key} = array[{len(val)}] {val[:8]}")


if __name__ == "__main__":
    main()
