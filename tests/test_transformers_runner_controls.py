"""Source/lifecycle boundary tests; CUDA observations are fake, never GPU work."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('native_runner_controls', ROOT / 'tools/evaluate_transformers_manifest.py')
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def test_hash_rejection_precedes_tokenizer_and_model_loading(tmp_path, monkeypatch):
    from openbonsai import adapter, mixed_export
    import transformers
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'test').write_bytes(b'not a model')
    calls = []
    monkeypatch.setattr(adapter, 'load_base_model', lambda *a, **k: calls.append('original'))
    monkeypatch.setattr(mixed_export, 'load_mixed', lambda *a, **k: calls.append('mixed'))
    monkeypatch.setattr(transformers.AutoTokenizer, 'from_pretrained', lambda *a, **k: calls.append('tokenizer'))
    args = SimpleNamespace(kind='original', device='cpu', cache_policy='cold', model=source,
                           tokenizer=source, expected_model_sha256='0' * 64,
                           expected_tokenizer_sha256='0' * 64)
    with pytest.raises(ValueError, match='Model directory SHA'):
        runner.load_backend(args, {'minimum_free_cuda_bytes': 0}, tmp_path, time.monotonic() + 20)
    assert calls == []


def test_cuda_zero_headroom_is_rejected_before_any_source_or_device_call(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(runner, 'check_source_inputs', lambda _: calls.append('source'))
    with pytest.raises(ValueError, match='at least 1 GiB'):
        runner.load_backend(SimpleNamespace(device='cuda:0'), {'minimum_free_cuda_bytes': 0},
                            tmp_path, time.monotonic() + 20)
    assert not calls


@pytest.mark.parametrize('change', [{'kind': 'legacy'}, {'device': 'auto'}, {'cache_policy': 'auto'},
                                   {'kind': 'original', 'cache_policy': 'warm'}])
def test_unsupported_loading_policy_fails_before_source_read(change, monkeypatch):
    import openbonsai.eval_backends as backends
    calls = []
    monkeypatch.setattr(backends, 'local_directory_identity', lambda _: calls.append('source'))
    settings = dict(kind='original', device='cpu', cache_policy='cold')
    settings.update(change)
    with pytest.raises(ValueError):
        runner.check_source_inputs(SimpleNamespace(**settings))
    assert not calls


def test_output_cannot_be_nested_in_model_source(tmp_path):
    source = tmp_path / 'model'
    source.mkdir()
    args = SimpleNamespace(max_seconds=10, output=source / 'new-results', model=source,
                           tokenizer=source, manifest=tmp_path / 'manifest', profile=tmp_path / 'profile')
    with pytest.raises(ValueError, match='inside an input source'):
        runner.supervisor_main(args)
    assert not args.output.exists()


class FakeCUDA:
    def __init__(self, before=10, after=10, initial_reserved=2, peak_reserved=3):
        self.free = iter([before, after])
        self.initial_reserved, self.peak_reserved = initial_reserved, peak_reserved
        self.resets = 0

    def synchronize(self, _):
        pass

    def mem_get_info(self, _):
        return next(self.free), 100

    def memory_reserved(self, _):
        return self.initial_reserved

    def reset_peak_memory_stats(self, _):
        self.resets += 1

    def max_memory_reserved(self, _):
        return self.peak_reserved

    def max_memory_allocated(self, _):
        return self.peak_reserved - 1


def fake_meter(tmp_path, monkeypatch, cuda):
    fake = SimpleNamespace(cuda=cuda, device=lambda _: SimpleNamespace(type='cuda', __str__=lambda: 'cuda:0'))
    monkeypatch.setitem(sys.modules, 'torch', fake)
    return runner.RuntimeMeter(tmp_path / 'phases', 'cuda:0', time.monotonic() + 10, 4)


def test_headroom_meter_preserves_failed_before_operation_receipt(tmp_path, monkeypatch):
    cuda = FakeCUDA(before=3)
    meter = fake_meter(tmp_path, monkeypatch, cuda)
    calls = []
    with pytest.raises(ValueError, match='initial CUDA headroom'):
        meter('test', lambda: calls.append('model'))
    receipt = json.loads((tmp_path / 'phases/0001.json').read_text())
    assert not calls and cuda.resets == 0
    assert receipt['status'] == 'failed' and receipt['initial_free_bytes'] == 3


def test_headroom_gate_uses_allocator_peak_not_only_ending_free(tmp_path, monkeypatch):
    cuda = FakeCUDA(before=10, after=9, initial_reserved=2, peak_reserved=10)
    meter = fake_meter(tmp_path, monkeypatch, cuda)
    calls = []
    with pytest.raises(ValueError, match='headroom gate failed'):
        meter('test', lambda: calls.append('completed'))
    receipt = json.loads((tmp_path / 'phases/0001.json').read_text())
    assert calls == ['completed'] and cuda.resets == 1
    assert receipt['conservative_free_bytes'] == 2
    assert receipt['status'] == 'failed'


def test_headroom_meter_failure_retains_original_error_and_peaks(tmp_path, monkeypatch):
    meter = fake_meter(tmp_path, monkeypatch, FakeCUDA())
    def fail():
        raise RuntimeError('fake OOM')
    with pytest.raises(RuntimeError, match='fake OOM'):
        meter('test', fail)
    receipt = json.loads((tmp_path / 'phases/0001.json').read_text())
    assert receipt['status'] == 'failed' and receipt['peak_reserved_bytes'] == 3
    assert 'fake OOM' in receipt['error']


def test_driver_package_query_is_bounded_and_has_no_shell(monkeypatch):
    calls = []
    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        return SimpleNamespace(stdout='580.95.05\n580.95.05\n')
    monkeypatch.setattr(runner.subprocess, 'run', run)
    result = runner.driver_package_identity()
    assert result['available'] and result['versions'] == ['580.95.05']
    argv, kwargs = calls[0]
    assert argv == ['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader']
    assert kwargs['timeout'] == 3 and kwargs.get('shell', False) is False


def test_missing_driver_query_is_explicit_not_runtime_version(monkeypatch):
    def fail(*args, **kwargs):
        raise subprocess.TimeoutExpired('nvidia-smi', 3)
    monkeypatch.setattr(runner.subprocess, 'run', fail)
    result = runner.driver_package_identity()
    assert not result['available'] and result['versions'] == []
    assert 'TimeoutExpired' in result['error']


def test_execution_metadata_does_not_alias_mutable_caller_settings():
    backend = SimpleNamespace(metadata=lambda: {'source_identity': {'sha256': 'a' * 64}})
    execution = {'cache_bytes': 5, 'mixed_modules': {'test': {'row_chunk_size': 64}}}
    wrapped = runner.ExecutionBackend(backend, execution)
    execution['cache_bytes'] = 6
    first = wrapped.metadata()
    first['native_execution']['mixed_modules']['test']['row_chunk_size'] = 100
    assert wrapped.metadata()['native_execution'] == {'cache_bytes': 5, 'mixed_modules': {'test': {'row_chunk_size': 64}}}


def test_cli_has_no_implicit_model_launch():
    result = subprocess.run([sys.executable, str(ROOT / 'tools/evaluate_transformers_manifest.py')],
                            capture_output=True, text=True, timeout=10)
    assert result.returncode != 0 and '--run and all explicit' in result.stderr
