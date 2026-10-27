"""Prepare a random native hybrid Qwen fixture and convert it with pinned Prism.

Only local tokenizer assets are accepted. No pretrained weights are loaded or
network request is made. F32 weights isolate adapter/runtime correctness from
BF16 export rounding and ternary quality. This fixture is not a useful model.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PRISM_SHA = 'c9c34bc9c4042c13a31598ebf43461cd0ad8f00d'


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        while part := handle.read(1024 * 1024):
            digest.update(part)
    return digest.hexdigest()


def prepare(output, tokenizer_dir, prism_source, *, seed=5090):
    import torch
    from transformers import AutoTokenizer
    from openbonsai.adapter import tiny_model, model_from_config

    output, tokenizer_dir, prism_source = map(Path, (output, tokenizer_dir, prism_source))
    source_sha = subprocess.check_output(
        ['git', '-C', str(prism_source), 'rev-parse', 'HEAD'], text=True).strip()
    if source_sha != PRISM_SHA:
        raise ValueError('Prism converter must be at the reviewed commit')
    # Include untracked changes: modified conversion dependencies must not be
    # attributed to the pinned source. Build products should live outside it.
    if subprocess.check_output(['git', '-C', str(prism_source), 'status', '--porcelain'], text=True).strip():
        raise ValueError('Prism source must be clean before fixture conversion')
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError('Use a new/empty output directory')
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True, trust_remote_code=False)
    # Use the supported architecture fixture with Qwen3.5's padded vocabulary.
    # Actual Qwen tokenizer has fewer entries than the output dimension; unused
    # rows still participate in the probability normalizer.
    config = tiny_model(seed=seed).config
    config.vocab_size = 248320
    config.eos_token_id = [248044, 248046]
    config.pad_token_id = 248044
    config.max_position_embeddings = 256
    if max(tokenizer.get_vocab().values()) >= config.vocab_size:
        raise ValueError('Tokenizer IDs exceed fixture vocabulary')
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        model = model_from_config(config).float().eval()
    model.requires_grad_(False)
    native = output / 'native-f32'
    model.save_pretrained(native, safe_serialization=True)
    tokenizer_files = {}
    for name in ('tokenizer.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt',
                 'chat_template.jinja', 'special_tokens_map.json'):
        source = tokenizer_dir / name
        if source.is_file():
            shutil.copyfile(source, native / name)
            tokenizer_files[name] = {'bytes': source.stat().st_size, 'sha256': file_sha256(source)}
    gguf = output / 'native-f32.gguf'
    command = [sys.executable, str(prism_source / 'convert_hf_to_gguf.py'), str(native),
               '--outfile', str(gguf), '--outtype', 'f32', '--no-mtp']
    # The random native fixture has no MTP parameters. The explicit flag avoids
    # upstream's default assumption that one or more MTP blocks are present.
    with (output / 'conversion.log').open('wb') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    report = {'format': 'openbonsai.random-eval-fixture.v1', 'seed': seed,
              'quality_claim': False, 'pretrained_weights_loaded': False,
              'dtype': 'float32', 'architecture': 'Qwen3_5ForCausalLM',
              'layer_types': config.layer_types, 'vocab_size': config.vocab_size,
              'tokenizer_entries': len(tokenizer), 'context_length': 256,
              'prism_source_sha': source_sha,
              'tokenizer_files': tokenizer_files,
              'files': {str(p.relative_to(output)): {'bytes': p.stat().st_size, 'sha256': file_sha256(p)}
                        for p in sorted(output.rglob('*')) if p.is_file()}}
    (output / 'fixture.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--tokenizer-dir', required=True)
    parser.add_argument('--prism-source', required=True)
    parser.add_argument('--seed', type=int, default=5090)
    args = parser.parse_args()
    report = prepare(args.output, args.tokenizer_dir, args.prism_source, seed=args.seed)
    print(json.dumps({k: v for k, v in report.items() if k not in ('files', 'tokenizer_files')}, indent=2))


if __name__ == '__main__':
    main()
