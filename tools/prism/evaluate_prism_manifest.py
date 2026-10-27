"""Score a frozen manifest with a PTQ1_0 GGUF in Prism's llama.cpp (optional tool).

Runs the pinned C++ worker (build it with build_prism_eval.py against a Prism
checkout at the pinned commit; the worker reports that commit and the backend
checks it) through openbonsai.eval.backends.PrismBackend
and the frozen-manifest contract in openbonsai.eval.manifest: every frozen
token ID and raw piece must match before scoring; NLL uses FP64 normalization
of the worker's FP32 logits. Use it to score the official Ternary Bonsai GGUF
or an exported model on exactly the inputs the torch ``dev`` suite uses.

    python tools/prism/evaluate_prism_manifest.py --worker build/prism_eval_worker \\
        --model Ternary-Bonsai-2-9B.gguf --output out/bonsai-dev

The default manifest is the package's dev suite (SHA-pinned). Nothing is
downloaded; the report is written as ``<output>/report.json``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RUNTIME_SHA = 'c9c34bc9c4042c13a31598ebf43461cd0ad8f00d'


def context_length(manifest) -> int:
    """Smallest context that holds every scored input and every generation cap."""
    lengths = [len(r['input_ids']) for r in manifest['nll']]
    lengths += [len(o['input_ids']) for q in manifest['choices'] for o in q['options']]
    lengths += [len(g['input_ids']) + g['max_new_tokens'] for g in manifest['generations']]
    return max(lengths)


def load(args):
    from openbonsai.eval import manifest as evaluation
    from openbonsai.eval.suites import DEV_MANIFEST, DEV_SHA256
    if args.manifest is None:
        return evaluation.load_manifest(DEV_MANIFEST, expected_file_sha256=DEV_SHA256)
    return evaluation.load_manifest(args.manifest, expected_file_sha256=args.expected_manifest_sha256)


def run(args, backend_factory=None) -> dict:
    from openbonsai.eval import manifest as evaluation
    from openbonsai.formats.atomic import atomic_write_json
    if backend_factory is None:
        from openbonsai.eval.backends import PrismBackend as backend_factory
    manifest = load(args)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'report.json').exists() and not args.overwrite:
        raise FileExistsError(f'{output / "report.json"} exists; pass --overwrite to replace it')
    stderr = output / 'worker-stderr.log'
    if stderr.exists():
        stderr.unlink()
    backend = backend_factory(
        args.worker, args.model, context_length=args.context_length or context_length(manifest),
        stderr_path=stderr, threads=args.threads,
        gpu_layers=args.gpu_layers, runtime_sha=RUNTIME_SHA,
        expected_executable_sha256=args.expected_worker_sha256, expected_model_sha256=args.expected_model_sha256)
    try:
        report = evaluation.evaluate_manifest(backend, manifest)
    finally:
        backend.close()
    atomic_write_json(output / 'report.json', report)
    return report


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--worker', type=Path, required=True, help='prism_eval_worker executable')
    parser.add_argument('--model', type=Path, required=True, help='PTQ1_0 GGUF')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, help='frozen manifest (default: the package dev suite)')
    parser.add_argument('--expected-manifest-sha256')
    parser.add_argument('--expected-model-sha256', help='hash the GGUF and refuse a mismatch')
    parser.add_argument('--expected-worker-sha256', help='hash the worker and refuse a mismatch')
    parser.add_argument('--context-length', type=int, help='default: derived from the manifest')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--gpu-layers', type=int, default=0)
    parser.add_argument('--overwrite', action='store_true')
    return parser


def main(argv=None, backend_factory=None) -> int:
    args = build_parser().parse_args(argv)
    report = run(args, backend_factory)
    print(json.dumps({'output': str(args.output), **report['aggregate']}, indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
