"""Explain a tiny CPU cross-runtime difference without changing official baselines.

Accepts only a hash-verified random F32 fixture from prepare_eval_fixture.py.
Measures Prism afresh, then temporarily changes ONLY the HF recurrent Q/K
epsilon placement in this process. The original function is restored in finally
and its original full-native scores must reproduce exactly. No downloads,
GPU execution, checkpoint writes, runtime edits, or 9B quality claims.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
IDS = [10, 20, 30, 40, 50]
POSITIONS = [1, 4]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def verify_fixture(fixture, runtime_sha):
    """Bound resource use before any model allocation; verify the fixture ledger."""
    if fixture.is_symlink() or not fixture.is_dir():
        raise ValueError('Fixture must be a materialized local directory')
    receipt_path = fixture / 'fixture.json'
    if receipt_path.is_symlink() or not receipt_path.is_file():
        raise ValueError('A prepared fixture receipt is required')
    receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
    expected = {'format': 'openbonsai.random-eval-fixture.v1', 'dtype': 'float32',
                'architecture': 'Qwen3_5ForCausalLM', 'prism_source_sha': runtime_sha}
    if (any(receipt.get(key) != value for key, value in expected.items())
            or receipt.get('pretrained_weights_loaded') is not False
            or receipt.get('quality_claim') is not False
            or type(receipt.get('seed')) is not int):
        raise ValueError('Only the prepared random F32 fixture is supported; no pretrained/9B models')
    files = receipt.get('files')
    required = {'native-f32/model.safetensors', 'native-f32/config.json', 'native-f32.gguf'}
    if not isinstance(files, dict) or not required.issubset(files):
        raise ValueError('Incomplete fixture file ledger')
    for path in fixture.rglob('*'):
        if path.is_symlink():
            raise ValueError('Fixture symlinks are unsupported')
    actual_files = {path.relative_to(fixture).as_posix() for path in fixture.rglob('*')
                    if path.is_file() and path != receipt_path}
    if actual_files != set(files):
        raise ValueError('Fixture files differ from the recorded ledger')
    # Reject large inputs before hashing or loading them. This probe is only a
    # tiny software fixture, and has no option to relax these bounds.
    if ((fixture / 'native-f32/model.safetensors').stat().st_size > 300 * 1024**2
            or (fixture / 'native-f32.gguf').stat().st_size > 320 * 1024**2):
        raise ValueError('Fixture exceeds the tiny CPU probe size bound')
    for name, details in files.items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Invalid fixture ledger path')
        path = fixture / relative
        if (not isinstance(details, dict) or type(details.get('bytes')) is not int
                or path.stat().st_size != details['bytes'] or sha256(path) != details.get('sha256')):
            raise ValueError(f'Fixture file changed: {name}')
    config = json.loads((fixture / 'native-f32/config.json').read_text(encoding='utf-8'))
    bounds = {'hidden_size': 128, 'num_hidden_layers': 4, 'intermediate_size': 256,
              'max_position_embeddings': 256, 'vocab_size': 248320}
    if (config.get('model_type') != 'qwen3_5_text' or config.get('dtype') != 'float32'
            or any(type(config.get(key)) is not int or not 1 <= config[key] <= maximum
                   for key, maximum in bounds.items())
            or config.get('rms_norm_eps') != 1e-6
            or config.get('layer_types') != receipt.get('layer_types')
            or set(config.get('layer_types', [])) != {'linear_attention', 'full_attention'}
            or config['vocab_size'] != receipt.get('vocab_size')
            or config['max_position_embeddings'] != receipt.get('context_length')):
        raise ValueError('Unsupported random F32 hybrid fixture configuration')
    from safetensors import safe_open
    with safe_open(fixture / 'native-f32/model.safetensors', framework='pt', device='cpu') as tensors:
        if not tensors.keys() or any(tensors.get_slice(key).get_dtype() != 'F32' for key in tensors.keys()):
            raise ValueError('The native fixture must contain only F32 tensors')
    return receipt, config


def run(fixture, worker, prism_source, output):
    import torch
    import transformers
    from transformers import AutoModelForCausalLM
    import transformers.models.qwen3_5.modeling_qwen3_5 as qwen
    from openbonsai.eval_backends import PrismBackend, PRISM_RUNTIME_SHA
    from openbonsai.evaluation import REDUCTION, reduce_target_logprobs

    fixture, worker, prism_source, output = map(Path, (fixture, worker, prism_source, output))
    receipt, config = verify_fixture(fixture, PRISM_RUNTIME_SHA)
    # Never overwrite an earlier proof, even an empty output directory.
    output.mkdir(parents=True, exist_ok=False)
    before_threads = torch.get_num_threads()
    original_function = qwen.l2norm
    hf_source_path = Path(qwen.__file__)
    hf_source_sha = sha256(hf_source_path)
    receipt_sha = sha256(fixture / 'fixture.json')

    def full_native_scores(model):
        # Independent full model.forward, not the backend's hidden/head adapter.
        with torch.no_grad():
            logits = model(input_ids=torch.tensor([IDS], dtype=torch.long, device='cpu'),
                           use_cache=False, return_dict=True).logits
            if logits.dtype != torch.float32 or logits.device.type != 'cpu':
                raise AssertionError('Expected native CPU F32 logits')
            return reduce_target_logprobs(logits[0, [j - 1 for j in POSITIONS]], [IDS[j] for j in POSITIONS])

    try:
        torch.set_num_threads(4)
        with PrismBackend(worker, fixture / 'native-f32.gguf',
                          context_length=config['max_position_embeddings'], threads=4, gpu_layers=0,
                          runtime_source_dir=prism_source, stderr_path=output / 'prism.stderr',
                          expected_model_sha256=receipt['files']['native-f32.gguf']['sha256'],
                          request_timeout=60.) as prism:
            prism_result = prism.score_tokens(IDS, POSITIONS)
            prism_metadata = prism.metadata()
        model = AutoModelForCausalLM.from_pretrained(
            fixture / 'native-f32', local_files_only=True, trust_remote_code=False,
            dtype=torch.float32, attn_implementation='eager').eval()
        if any(parameter.device.type != 'cpu' or parameter.dtype != torch.float32
               for parameter in model.parameters()):
            raise AssertionError('This probe only permits a CPU F32 native fixture')
        unmodified = full_native_scores(model)
        try:
            def diagnostic_clamp(x, dim=-1, eps=1e-6):
                # Only epsilon placement changes. Keep the native F32 sum.
                return x / torch.sqrt((x * x).sum(dim=dim, keepdim=True)).clamp_min(eps)
            qwen.l2norm = diagnostic_clamp
            patched = full_native_scores(model)
        finally:
            qwen.l2norm = original_function
        restored = full_native_scores(model)
        if qwen.l2norm is not original_function or restored != unmodified:
            raise AssertionError('Original function/scores did not restore exactly')
        if sha256(hf_source_path) != hf_source_sha:
            raise AssertionError('Installed HF source changed')
        verify_fixture(fixture, PRISM_RUNTIME_SHA)
        if sha256(fixture / 'fixture.json') != receipt_sha:
            raise AssertionError('Fixture receipt changed')
        prism_scores = prism_result['target_logprobs']
        before = [native - official for native, official in zip(unmodified, prism_scores)]
        after = [native - official for native, official in zip(patched, prism_scores)]
        maximum_before, maximum_after = max(map(abs, before)), max(map(abs, after))
        report = {
            'format': 'openbonsai.normalization-diagnostic.v1',
            'scope': 'CPU random F32 hybrid fixture; two fixed target positions',
            'diagnostic_only': True, 'quality_claim': False,
            'official_baselines_unchanged': True, 'gpu_layers': 0, 'threads': 4,
            'pretrained_or_ternary_weights_loaded': False,
            'fixture': str(fixture.absolute()), 'fixture_receipt_sha256': receipt_sha,
            'model_safetensors': receipt['files']['native-f32/model.safetensors'],
            'model_config': receipt['files']['native-f32/config.json'],
            'model_gguf': receipt['files']['native-f32.gguf'],
            'script_sha256': sha256(Path(__file__)),
            'torch_version': torch.__version__, 'transformers_version': transformers.__version__,
            'hf_source_path': str(hf_source_path), 'hf_source_sha256_unchanged': hf_source_sha,
            'hf_original_l2norm_source': inspect.getsource(original_function),
            'attention_implementation': model.config._attn_implementation,
            'input_ids': IDS, 'score_positions': POSITIONS, 'reduction': REDUCTION,
            'prism': prism_metadata, 'prism_result_measured_this_run': prism_result,
            'unmodified_native_scores': unmodified, 'unmodified_native_signed_delta': before,
            'diagnostic_clamp_scores': patched, 'diagnostic_clamp_signed_delta': after,
            'max_absolute_drift_before': maximum_before,
            'max_absolute_drift_after': maximum_after,
            'reduction_factor': maximum_before / maximum_after if maximum_after else None,
            'restored_native_scores': restored, 'restored_function_identity': True,
            'formulas': {'official_hf': 'x * rsqrt(sum(x*x) + 1e-6)',
                         'official_prism_cpu': 'x / max(sqrt(sum(x*x)), rms_norm_eps)',
                         'diagnostic_hf': 'x / max(sqrt(sum(x*x)), 1e-6); native F32 sum retained'},
            'sources': [
                f'https://github.com/PrismML-Eng/llama.cpp/blob/{PRISM_RUNTIME_SHA}/src/models/qwen35.cpp#L551-L560',
                f'https://github.com/PrismML-Eng/llama.cpp/blob/{PRISM_RUNTIME_SHA}/ggml/src/ggml-cpu/ops.cpp#L4177-L4223'],
            'interpretation': 'The temporary diagnostic is NOT an official baseline or quality improvement. '
                'Keep official HF and Prism unmodified and report their measured backend discrepancy. '
                'These two tiny-fixture positions cannot establish pretrained 9B or ternary equivalence.',
        }
        (output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        return report
    finally:
        qwen.l2norm = original_function
        torch.set_num_threads(before_threads)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture', required=True, help='Local random eval fixture directory with fixture.json')
    parser.add_argument('--worker', required=True, help='Pinned compiled prism-eval-worker executable')
    parser.add_argument('--prism-source', required=True, help='Clean pinned Prism source directory')
    parser.add_argument('--output', required=True, help='New output directory; existing paths are refused')
    args = parser.parse_args()
    report = run(args.fixture, args.worker, args.prism_source, args.output)
    print(json.dumps({key: report[key] for key in ('max_absolute_drift_before', 'max_absolute_drift_after',
                                                 'reduction_factor', 'official_baselines_unchanged')}))


if __name__ == '__main__':
    main()
