"""Verify and inventory the official language-only Bonsai PTQ1_0 GGUF.

Uses the clean pinned Prism reader without executing a model or downloading
files. Tensor bytes include packed scales. The remaining container bytes include
metadata, tokenizer, shared signs and alignment; they are not weight payload.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys

REPOSITORY = 'prism-ml/Ternary-Bonsai-2-9B-gguf'
REVISION = '5b2abb02362efbed46bbe204014b02837778b2c2'
FILENAME = 'Ternary-Bonsai-2-9B-PTQ1_0.gguf'
SHA256 = '3561df7ac7d26a71ffa991ac2db740b10841c3a350aa2676ccd3ff5d428b4e8b'
BYTES = 1997486976
RUNTIME = 'c9c34bc9c4042c13a31598ebf43461cd0ad8f00d'


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(4 << 20), b''):
            result.update(block)
    return result.hexdigest()


def inspect(gguf, prism_source):
    path, source = Path(gguf).resolve(), Path(prism_source).resolve()
    if Path(gguf).is_symlink() or not path.is_file():
        raise ValueError('Expected a materialized local GGUF')
    before = path.stat()
    if before.st_size != BYTES or sha256(path) != SHA256:
        raise ValueError('Official GGUF size or SHA256 mismatch')
    revision = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain'], text=True).strip()
    if revision != RUNTIME or dirty:
        raise ValueError('Use a clean checkout of the pinned Prism runtime')
    # CLI runs in a fresh interpreter; refuse an unrelated previously imported reader.
    if 'gguf' in sys.modules:
        raise RuntimeError('Run the inventory in a fresh Python process')
    sys.path.insert(0, str(source / 'gguf-py'))
    from gguf import GGUFReader
    reader = GGUFReader(path, 'r')
    tensors = [{'name': t.name, 'type': t.tensor_type.name,
                'shape_ggml_order': t.shape.tolist(), 'elements': t.n_elements,
                'bytes': t.n_bytes, 'offset': t.data_offset} for t in reader.tensors]
    by_type = defaultdict(lambda: {'tensors': 0, 'elements': 0, 'bytes': 0})
    retained = defaultdict(lambda: {'tensors': 0, 'bytes': 0})
    for tensor in tensors:
        group = by_type[tensor['type']]
        group['tensors'] += 1
        for key in ('elements', 'bytes'):
            group[key] += tensor[key]
        if tensor['type'] == 'F32':
            name = tensor['name']
            family = name.split('.', 2)[-1] if name.startswith('blk.') else name
            retained[family]['tensors'] += 1
            retained[family]['bytes'] += tensor['bytes']
    fields = {}
    for key, field in reader.fields.items():
        if not (key.startswith('prism.') or key in ('general.architecture', 'general.file_type',
                'qwen35.block_count', 'qwen35.context_length', 'tokenizer.ggml.eos_token_id',
                'tokenizer.ggml.bos_token_id', 'tokenizer.ggml.add_bos_token')):
            continue
        value = field.contents()
        if isinstance(value, list) and len(value) > 16:
            encoded = json.dumps(value, separators=(',', ':'), ensure_ascii=False).encode('utf-8')
            fields[key] = {'entries': len(value), 'canonical_value_sha256': hashlib.sha256(encoded).hexdigest()}
        else:
            fields[key] = value
    after = path.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
        raise RuntimeError('GGUF changed during inspection')
    total = sum(t['bytes'] for t in tensors)
    return {
        'format': 'openbonsai.official-bonsai-inventory.v1',
        'source': {'repository': REPOSITORY, 'revision': REVISION, 'filename': FILENAME,
                   'sha256': SHA256, 'bytes': BYTES},
        'reader_source_sha': revision, 'inspection_script_sha256': sha256(__file__),
        'scope': 'Language-only file with embedded tokenizer; vision and optional MTP excluded',
        'quality_claim': False, 'inference_executed': False,
        'tensor_count': len(tensors), 'by_type': dict(by_type),
        'tensor_payload_bytes': total, 'container_metadata_alignment_bytes': BYTES - total,
        'ternary_matrix_count': sum(t['type'] == 'PTQ1_0' for t in tensors),
        'ternary_endpoints': [t for t in tensors if t['name'] in ('token_embd.weight', 'output.weight')],
        'retained_f32_families': dict(retained), 'selected_metadata': fields, 'tensors': tensors,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gguf', required=True)
    parser.add_argument('--prism-source', required=True)
    parser.add_argument('--output', required=True, help='New JSON file; existing files are refused')
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError('Use a new output file')
    report = inspect(args.gguf, args.prism_source)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({key: report[key] for key in ('tensor_count', 'by_type', 'container_metadata_alignment_bytes')}, indent=2))


if __name__ == '__main__':
    main()
