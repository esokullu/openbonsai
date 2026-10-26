"""Evaluate one existing native original/mixed model on a frozen manifest.

No downloads, training, export, requantization, module renaming or research-state
restoration. The original source loads in BF16; a mixed source keeps its stored
payload dtypes, signs, native names and row chunks. Each attempt owns one child.
"""
from __future__ import annotations

import argparse
import copy
import ctypes
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
PROFILE_FORMAT = 'openbonsai.transformers-profile.v1'
MAX_SECONDS = 10800.0
FACTOR = 1.25
RESERVE = 300.0


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            value.update(block)
    return value.hexdigest()


def encode(value):
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False,
                       allow_nan=False) + '\n').encode('utf-8')


def write_new(path, value):
    with Path(path).open('xb') as handle:
        handle.write(encode(value))


def update_owned(path, value):
    temporary = Path(str(path) + '.tmp')
    with temporary.open('xb') as handle:
        handle.write(encode(value))
    os.replace(temporary, path)


def check_deadline(deadline, clock=time.monotonic):
    remaining = deadline - clock()
    if not math.isfinite(remaining) or remaining <= 0:
        raise TimeoutError('Whole-process monotonic deadline exhausted')
    return remaining


def sha_string(value):
    require(isinstance(value, str) and len(value) == 64
            and all(c in '0123456789abcdef' for c in value), 'Expected lowercase SHA256')
    return value


def integer(value, minimum, maximum, label):
    require(type(value) is int and minimum <= value <= maximum, 'Invalid ' + label)
    return value


def exact_keys(value, names, label):
    require(isinstance(value, dict) and set(value) == set(names), 'Invalid ' + label + ' fields')


def read_pinned_json(path, expected):
    data = Path(path).read_bytes()
    require(hashlib.sha256(data).hexdigest() == sha_string(expected), 'File SHA mismatch: ' + str(path))
    return json.loads(data)


def score_shape(item, logits_chunk_size):
    """Native _hidden consumes ALL input IDs, including trailing unscored IDs."""
    rows = len(item['score_positions'])
    return {'input_tokens': len(item['input_ids']), 'output_rows': rows,
            'maximum_head_rows': min(rows, logits_chunk_size)}


def dominates(probe, other):
    return all(probe[key] >= other[key] for key in probe)


def build_plan(manifest, profile, *, evaluation=None):
    if evaluation is None:
        from openbonsai import evaluation
    exact_keys(profile, {'format', 'manifest_sha256', 'context_length', 'vocab_size',
                         'threads', 'logits_chunk_size', 'minimum_free_cuda_bytes',
                         'resource_probes'}, 'profile')
    require(profile['format'] == PROFILE_FORMAT, 'Unsupported profile format')
    require(sha_string(profile['manifest_sha256']) == manifest.get('frozen_sha256'),
            'Profile belongs to a different frozen manifest')
    context = integer(profile['context_length'], 2, 1048576, 'context_length')
    vocab = integer(profile['vocab_size'], 2, 1048576, 'vocab_size')
    chunk = integer(profile['logits_chunk_size'], 1, 64, 'logits_chunk_size')
    integer(profile['threads'], 1, 4, 'threads')
    integer(profile['minimum_free_cuda_bytes'], 0, 64 << 30, 'minimum_free_cuda_bytes')
    evaluation.validate_manifest(manifest, context_length=context, vocab_size=vocab)
    require(manifest['role'] in ('search_validation', 'evaluation'), 'Calibration-role inputs are forbidden')
    require(manifest['nll'] and manifest['choices'] and manifest['generations'],
            'The bounded profile requires NLL, choices and generations')
    probes = profile['resource_probes']
    exact_keys(probes, {'nll', 'choice', 'generation'}, 'resource probes')
    exact_keys(probes['nll'], {'record_index'}, 'NLL probe')
    ni = integer(probes['nll']['record_index'], 0, len(manifest['nll']) - 1, 'NLL index')
    nll = manifest['nll'][ni]
    exact_keys(probes['choice'], {'question_index', 'option_index'}, 'choice probe')
    qi = integer(probes['choice']['question_index'], 0, len(manifest['choices']) - 1, 'question index')
    question = manifest['choices'][qi]
    oi = integer(probes['choice']['option_index'], 0, len(question['options']) - 1, 'option index')
    choice = question['options'][oi]
    options = [option for row in manifest['choices'] for option in row['options']]
    for label, selected, records in [('NLL', nll, manifest['nll']), ('Choice', choice, options)]:
        require(all(dominates(score_shape(selected, chunk), score_shape(row, chunk)) for row in records),
                label + ' probe does not dominate every native input/head shape')
    generation = probes['generation']
    exact_keys(generation, {'text_index', 'token_start', 'token_end', 'target_position'}, 'generation probe')
    ti = integer(generation['text_index'], 0, len(manifest['texts']) - 1, 'generation text index')
    source = manifest['texts'][ti]
    start = integer(generation['token_start'], 0, len(source['input_ids']) - 2, 'generation slice start')
    end = integer(generation['token_end'], start + 2, len(source['input_ids']), 'generation slice end')
    ids = source['input_ids'][start:end]
    target = integer(generation['target_position'], 1, len(ids) - 1, 'generation target')
    require(target == len(ids) - 1, 'Generation probe target must be the last source token')
    evaluation.validate_score_positions(ids, [target], context_length=context, vocab_size=vocab)
    maximum_forward = max(len(row['input_ids']) + row['max_new_tokens'] - 1 for row in manifest['generations'])
    maximum_guard = max(len(row['input_ids']) + row['max_new_tokens'] for row in manifest['generations'])
    require(len(ids) >= maximum_forward, 'Generation probe is shorter than a capped native forward')
    referenced = {row['text_id'] for row in manifest['nll'] + manifest['generations']}
    referenced.update(row['text_id'] for row in options)
    require(source['id'] in referenced, 'Generation probe source is not an existing evaluation input')
    workload = {'nll_records': len(manifest['nll']),
                'nll_targets': sum(len(row['score_positions']) for row in manifest['nll']),
                'choice_questions': len(manifest['choices']),
                'choice_labelled': sum('answer_id' in row for row in manifest['choices']),
                'choice_option_calls': len(options),
                'choice_targets': sum(len(row['score_positions']) for row in options),
                'generation_records': len(manifest['generations']),
                'generation_tokens_maximum': sum(row['max_new_tokens'] for row in manifest['generations']),
                'generation_forward_positions_maximum': maximum_forward,
                'generation_context_guard_maximum': maximum_guard,
                'tokenization_texts': len(manifest['texts'])}
    resource = [
        {'name': 'nll', 'input_ids': nll['input_ids'], 'score_positions': nll['score_positions'],
         'identity': {'record_index': ni, 'record_id': nll['id']}},
        {'name': 'choice', 'input_ids': choice['input_ids'], 'score_positions': choice['score_positions'],
         'identity': {'question_index': qi, 'option_index': oi, 'question_id': question['id'], 'option_id': choice['id']}},
        {'name': 'generation_last_logit', 'input_ids': ids, 'score_positions': [target],
         'identity': {**generation, 'source_text_id': source['id']}}]
    for probe in resource:
        probe['shape'] = score_shape(probe, chunk)
    return {'workload': workload, 'probes': resource, 'context_length': context,
            'vocab_size': vocab, 'threads': profile['threads'], 'logits_chunk_size': chunk,
            'minimum_free_cuda_bytes': profile['minimum_free_cuda_bytes'],
            'shape_semantics': 'native hidden forward consumes all input IDs; bounded full-vocabulary head positions',
            'probe_scope': 'resource-only existing-token inputs; excluded from quality aggregation',
            'forecast_limit': 'shape dominance plus margin is an estimate, not a runtime guarantee'}


def forecast(plan, seconds, remaining):
    for key in ('nll', 'choice', 'generation_last_logit', 'tokenization'):
        value = seconds.get(key)
        require(type(value) in (int, float) and math.isfinite(value) and value > 0, 'Invalid measured duration ' + key)
    require(math.isfinite(remaining) and remaining > 0, 'Invalid remaining budget')
    work = plan['workload']
    components = {'nll': work['nll_records'] * seconds['nll'],
                  'choice_independent_options': work['choice_option_calls'] * seconds['choice'],
                  'generation_full_cap': work['generation_tokens_maximum'] * seconds['generation_last_logit'],
                  'repeated_full_tokenization': seconds['tokenization']}
    total = FACTOR * math.fsum(components.values()) + RESERVE
    return {'components_seconds': components, 'measured_probe_seconds': seconds,
            'factor': FACTOR, 'reserve_seconds': RESERVE, 'projected_remaining_seconds': total,
            'remaining_seconds_at_admission': remaining, 'admitted': total <= remaining,
            'early_eos_assumed': False, 'workload': work, 'forecast_is_not_a_guarantee': True}


def jsonable(value):
    if isinstance(value, bytes):
        return {'bytes_hex': value.hex()}
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


class RecordedBackend:
    def __init__(self, backend, output, deadline, manifest_sha, clock=time.monotonic, measure=None):
        self.backend, self.output, self.deadline = backend, Path(output), deadline
        self.manifest_sha, self.clock, self.measure = manifest_sha, clock, measure
        self.output.mkdir(exist_ok=False)
        self.phase, self.calls, self.last_receipt = 'tokenization-gate', 0, None

    def call(self, operation, request, function):
        remaining = check_deadline(self.deadline, self.clock)
        self.calls += 1
        path = self.output / f'{self.calls:04d}-{operation}.json'
        row = {'index': self.calls, 'phase': self.phase, 'operation': operation,
               'request': copy.deepcopy(request), 'status': 'started',
               'manifest_sha256': self.manifest_sha, 'remaining_before_seconds': remaining}
        write_new(path, row)
        started = self.clock()
        try:
            result = (function() if self.measure is None else self.measure(
                f'{self.phase}-{self.calls:04d}-{operation}', function))
            check_deadline(self.deadline, self.clock)
            row.update(status='complete', result=jsonable(result))
            return result
        except BaseException as error:
            row.update(status='failed', error=f'{type(error).__name__}: {error}')
            raise
        finally:
            row['seconds'] = self.clock() - started
            row['remaining_after_seconds'] = max(0., self.deadline - self.clock())
            update_owned(path, row)
            self.last_receipt = row

    def metadata(self):
        return self.call('metadata', {}, self.backend.metadata)

    def tokenize(self, text, parse_special=False):
        return self.call('tokenize', {'text': text, 'parse_special': parse_special, 'add_special': False},
                         lambda: self.backend.tokenize(text, parse_special))

    def token_pieces(self, ids):
        return self.call('token_pieces', {'ids': ids}, lambda: self.backend.token_pieces(ids))

    def score_tokens(self, ids, positions):
        return self.call('score', {'input_ids': ids, 'score_positions': positions},
                         lambda: self.backend.score_tokens(ids, positions))

    def generate(self, ids, maximum, eos):
        return self.call('generate', {'input_ids': ids, 'max_new_tokens': maximum, 'eos_ids': eos},
                         lambda: self.backend.generate(ids, maximum, eos))


def evaluate_plan(manifest, profile, backend, output, deadline, *, clock=time.monotonic,
                  evaluation=None, measure=None):
    if evaluation is None:
        from openbonsai import evaluation
    plan = build_plan(manifest, profile, evaluation=evaluation)
    output = Path(output)
    write_new(output / 'derived-plan.json', plan)
    wrapped = RecordedBackend(backend, output / 'calls', deadline, manifest['frozen_sha256'], clock, measure)
    started = clock()
    verified = evaluation.verify_tokenization(wrapped, manifest)
    tokenization_seconds = clock() - started
    require(verified['texts'] == plan['workload']['tokenization_texts']
            and verified['exact_ids'] and verified['exact_raw_pieces'], 'Incomplete native tokenization gate')
    write_new(output / 'native-tokenization-gate.json', dict(status='passed', seconds=tokenization_seconds, **verified))
    wrapped.phase = 'resource-preflight-not-quality'
    seconds, measurements = {'tokenization': tokenization_seconds}, []
    for probe in plan['probes']:
        result = wrapped.score_tokens(probe['input_ids'], probe['score_positions'])
        evaluation.validate_score_result(result, probe['input_ids'], probe['score_positions'])
        seconds[probe['name']] = wrapped.last_receipt['seconds']
        measurements.append({**probe, 'seconds': seconds[probe['name']], 'call_index': wrapped.calls,
                             'purpose': 'resource-only; excluded from final quality aggregation'})
    write_new(output / 'resource-preflight.json', {'status': 'complete', 'probes': measurements})
    admission = forecast(plan, seconds, check_deadline(deadline, clock))
    write_new(output / 'admission.json', admission)
    if not admission['admitted']:
        raise RuntimeError('Full fixed manifest does not fit remaining budget; no data/caps changed')
    wrapped.phase = 'full-fixed-evaluation-suite'
    report = evaluation.evaluate_manifest(wrapped, manifest)
    work = plan['workload']
    require(report['aggregate']['scored_tokens'] == work['nll_targets']
            and report['aggregate']['choice_labelled'] == work['choice_labelled']
            and report['aggregate']['generations'] == work['generation_records']
            and len(report['nll']) == work['nll_records'] and len(report['choices']) == work['choice_questions'],
            'Incomplete final evaluation')
    write_new(output / 'evaluation.json', report)
    return {'status': 'complete', 'calls': wrapped.calls, 'aggregate': report['aggregate'],
            'admission': admission, 'role': manifest['role'], 'fresh_heldout_confirmation': False,
            'legacy_fp32_cross_entropy_compatible': False,
            'scope': 'frozen native BF16/stored-payload reference; no cross-kernel equivalence claim'}


class RuntimeMeter:
    """Synchronized per-operation CUDA peaks with conservative allocator headroom.

    Before/after free memory plus allocator peaks are measured; other processes'
    transient allocations are not continuously observed. This is not a RAM cap.
    """
    def __init__(self, output, device, deadline, minimum_free):
        import torch
        self.torch, self.device, self.deadline = torch, torch.device(device), deadline
        self.minimum_free, self.index = minimum_free, 0
        self.output = Path(output)
        self.output.mkdir(exist_ok=False)

    def __call__(self, label, function):
        check_deadline(self.deadline)
        torch, device = self.torch, self.device
        cuda = device.type == 'cuda'
        self.index += 1
        path = self.output / f'{self.index:04d}.json'
        row = {'label': label, 'status': 'started', 'device': str(device),
               'minimum_required_free_cuda_bytes': self.minimum_free if cuda else None,
               'headroom_method': 'before/after device free and peak allocator-reserved bound; not continuous device sampling'}
        write_new(path, row)
        started = time.monotonic()
        begun = False
        try:
            if cuda:
                torch.cuda.synchronize(device)
                free, total = map(int, torch.cuda.mem_get_info(device))
                row.update(initial_free_bytes=free, total_device_bytes=total,
                           initial_reserved_bytes=int(torch.cuda.memory_reserved(device)))
                require(free >= self.minimum_free, 'Insufficient initial CUDA headroom')
                torch.cuda.reset_peak_memory_stats(device)
                begun = True
            value = function()
            if cuda:
                torch.cuda.synchronize(device)
                after, _ = torch.cuda.mem_get_info(device)
                peak_reserved = int(torch.cuda.max_memory_reserved(device))
                conservative = row['initial_free_bytes'] + row['initial_reserved_bytes'] - peak_reserved
                row.update(ending_free_bytes=int(after), peak_reserved_bytes=peak_reserved,
                           peak_allocated_bytes=int(torch.cuda.max_memory_allocated(device)),
                           conservative_free_bytes=conservative,
                           minimum_observed_or_conservative_free_bytes=min(row['initial_free_bytes'], int(after), conservative))
                require(row['minimum_observed_or_conservative_free_bytes'] >= self.minimum_free,
                        'CUDA headroom gate failed')
            check_deadline(self.deadline)
            row['status'] = 'complete'
            return value
        except BaseException as error:
            row.update(status='failed', error=f'{type(error).__name__}: {error}')
            if cuda and begun:
                try:
                    row.update(peak_allocated_bytes=int(torch.cuda.max_memory_allocated(device)),
                               peak_reserved_bytes=int(torch.cuda.max_memory_reserved(device)))
                except BaseException:
                    pass
            raise
        finally:
            row['seconds'] = time.monotonic() - started
            update_owned(path, row)


class ExecutionBackend:
    """Add immutable source/runtime provenance to the unchanged backend."""
    def __init__(self, backend, execution):
        self.backend, self.execution = backend, copy.deepcopy(execution)

    def metadata(self):
        return {**self.backend.metadata(), 'native_execution': copy.deepcopy(self.execution)}

    def tokenize(self, *args, **kwargs):
        return self.backend.tokenize(*args, **kwargs)

    def token_pieces(self, *args, **kwargs):
        return self.backend.token_pieces(*args, **kwargs)

    def score_tokens(self, *args, **kwargs):
        return self.backend.score_tokens(*args, **kwargs)

    def generate(self, *args, **kwargs):
        return self.backend.generate(*args, **kwargs)

    def close(self):
        self.backend.close()


def check_source_inputs(args):
    from openbonsai.eval_backends import local_directory_identity
    require(args.kind in ('original', 'mixed'), 'kind must be original or mixed')
    require(args.device in ('cpu', 'cuda:0'), 'device must be cpu or cuda:0')
    require(args.cache_policy in ('cold', 'warm'), 'cache-policy must be cold or warm')
    require(args.kind == 'mixed' or args.cache_policy == 'cold', 'Original models have no decoded-weight cache')
    model = local_directory_identity(args.model)
    tokenizer = model if Path(args.model).absolute() == Path(args.tokenizer).absolute() else local_directory_identity(args.tokenizer)
    require(model['sha256'] == sha_string(args.expected_model_sha256), 'Model directory SHA mismatch')
    require(tokenizer['sha256'] == sha_string(args.expected_tokenizer_sha256), 'Tokenizer directory SHA mismatch')
    return model, tokenizer


def cuda_driver_identity():
    """Read the installed driver API version without a shell/network call.

    This fixed driver-library query runs only inside the supervised model child.
    Failure is explicit provenance, never an invented version or runtime alias.
    """
    try:
        library = (ctypes.WinDLL('nvcuda.dll') if os.name == 'nt' else
                   ctypes.CDLL('libcuda.so.1'))
        query = library.cuDriverGetVersion
        query.argtypes = [ctypes.POINTER(ctypes.c_int)]
        query.restype = ctypes.c_int
        value = ctypes.c_int()
        status = int(query(ctypes.byref(value)))
        if status != 0 or value.value <= 0:
            return {'available': False, 'query_status': status, 'cuda_api_version': None}
        return {'available': True, 'query_status': status, 'cuda_api_version': value.value,
                'major': value.value // 1000, 'minor': value.value % 1000 // 10,
                'method': 'local CUDA driver cuDriverGetVersion'}
    except (OSError, AttributeError) as error:
        return {'available': False, 'cuda_api_version': None, 'error': f'{type(error).__name__}: {error}'}


def driver_package_identity():
    """Bounded local NVIDIA driver-package query, distinct from CUDA API version."""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=3,
                                check=True)
        values = sorted(set(line.strip() for line in result.stdout.splitlines() if line.strip()))
        require(values and all(re.fullmatch(r'[0-9]+(?:\.[0-9]+){1,3}', x) for x in values),
                'Unexpected NVIDIA driver version response')
        return {'available': True, 'versions': values, 'method': 'nvidia-smi driver_version; 3s timeout'}
    except (OSError, subprocess.SubprocessError, ValueError) as error:
        return {'available': False, 'versions': [], 'error': f'{type(error).__name__}: {error}'}


def load_backend(args, plan, output, deadline, *, measure=None):
    """Hash before loading. Return one backend plus immutable execution receipt."""
    import torch
    import transformers
    from transformers import AutoTokenizer
    from openbonsai.adapter import load_base_model, _text_module
    from openbonsai.eval_backends import TransformersBackend
    from openbonsai.mixed_export import load_mixed, FORMAT as MIXED_FORMAT
    from openbonsai.mixed_formats import MixedLinear, MixedEmbedding
    measure = measure or (lambda _label, operation: operation())
    output = Path(output)
    check_deadline(deadline)
    if args.device != 'cpu':
        require(plan['minimum_free_cuda_bytes'] >= 1 << 30, 'CUDA profile requires at least 1 GiB headroom')
    source, tokens = measure('source-file-identities', lambda: check_source_inputs(args))
    mixed_manifest = None
    if args.kind == 'mixed':
        mixed_manifest = json.loads((Path(args.model) / 'manifest.json').read_bytes())
        require(mixed_manifest.get('format') == MIXED_FORMAT, 'Expected complete mixed format; legacy packed artifacts are a different loader')
        require(mixed_manifest.get('modules') and all(x.get('compute_dtype') == 'bfloat16'
                for x in mixed_manifest['modules'].values()), 'Mixed stored compute dtype must be BF16')
    torch.set_num_threads(plan['threads'])
    tokenizer = measure('load-local-tokenizer', lambda: AutoTokenizer.from_pretrained(
        str(args.tokenizer), local_files_only=True, trust_remote_code=False))
    model = measure('load-one-native-model', lambda: (
        load_mixed(args.model, device=args.device, cache_decoded_weights=False) if args.kind == 'mixed' else
        load_base_model(str(args.model), device=args.device, dtype='bfloat16', local_files_only=True)))
    prefix, text = _text_module(model)
    require(text.config._attn_implementation == 'eager' and text.config.use_cache is False,
            'Native eager/no-cache configuration required')
    require(not any(p.requires_grad for p in model.parameters()), 'Model must be frozen')
    require(not any(m.training for m in model.modules()), 'Model must be in eval mode')
    require(not any('forward' in vars(m) or hasattr(m, '_hf_hook') or m._forward_hooks
                    or m._forward_pre_hooks for m in model.modules()), 'Custom forward/hooks/offload are unsupported')
    require(model.config.get_text_config().vocab_size == plan['vocab_size'], 'Native vocabulary differs from profile')
    native = TransformersBackend(model, tokenizer, context_length=plan['context_length'],
        source_identity={**source, 'model_kind': args.kind}, tokenizer_identity=tokens,
        logits_chunk_size=plan['logits_chunk_size'])
    modules = {}
    for name, module in model.named_modules():
        if isinstance(module, (MixedLinear, MixedEmbedding)):
            candidate = module._candidate()
            modules[name] = {'kind': candidate.module_kind, 'encoding': candidate.encoding,
                             'shape': list(candidate.shape), 'row_chunk_size': module.row_chunk_size,
                             'compute_dtype': str(module._compute_reference.dtype),
                             'payload_dtypes': {k: str(v.dtype) for k, v in candidate.tensors.items()},
                             'rotation': copy.deepcopy(candidate.metadata.get('rotation'))}
            require(module._compute_reference.dtype == torch.bfloat16, 'Mixed runtime computation must remain BF16')
            if isinstance(module, MixedLinear):
                require(module.decoded_weight_cache_bytes == 0, 'Mixed loader must start cold')
    require((bool(modules)) == (args.kind == 'mixed'), 'Unexpected mixed module scope')
    if mixed_manifest is not None:
        require(set(modules) == set(mixed_manifest['modules']), 'Loaded mixed module coverage differs')
    # File checking is repeated after load to reject a changing source tree.
    measure('post-load-source-identities', lambda: require(check_source_inputs(args) == (source, tokens), 'Sources changed while loading'))
    execution = {'model_kind': args.kind, 'device': args.device, 'native_model_class': type(model).__name__,
                 'native_text_prefix': prefix, 'attention_backend': 'eager', 'use_cache': False,
                 'original_load_dtype': 'bfloat16' if args.kind == 'original' else None,
                 'mixed_compute_dtype': 'bfloat16' if args.kind == 'mixed' else None,
                 'cache_policy': args.cache_policy, 'mixed_modules': modules,
                 'artifact_manifest_sha256': digest(Path(args.model) / 'manifest.json') if args.kind == 'mixed' else None,
                 'native_generation_eos_ids': copy.deepcopy(model.generation_config.eos_token_id),
                 'generation_eos_policy': 'use each frozen manifest EOS list without config mutation',
                 'python': platform.python_version(), 'platform': platform.platform(),
                 'torch': torch.__version__, 'transformers': transformers.__version__,
                 'cuda_runtime': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
                 'cuda_available_for_run': args.device != 'cpu',
                 'cuda_driver': (cuda_driver_identity() if args.device != 'cpu' else
                                 {'available': None, 'cuda_api_version': None, 'reason': 'CPU run; not queried'}),
                 'nvidia_driver_package': (driver_package_identity() if args.device != 'cpu' else
                                           {'available': None, 'versions': [], 'reason': 'CPU run; not queried'}),
                 'tf32': False,
                 'bf16_reduced_precision_reduction': torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
                 'fp16_reduced_precision_reduction': torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
                 'native_source_sha256': digest(inspect.getsourcefile(type(model))),
                 'headroom_policy_bytes': plan['minimum_free_cuda_bytes'] if args.device != 'cpu' else None,
                 'cache_bytes': 0, 'cache_is_nonpersistent_extra_memory': True}
    if args.device != 'cpu':
        props = torch.cuda.get_device_properties(args.device)
        execution['cuda_device'] = {'name': props.name, 'total_memory': props.total_memory,
                                    'compute_capability': [props.major, props.minor]}
    write_new(output / 'loaded-source.json', {'source': source, 'tokenizer': tokens, 'execution': execution})
    check_deadline(deadline)
    return native, execution


def warm_cache(backend, execution, plan, output, deadline, *, measure=None):
    """Preserve stored rows/rotations/GEMM chunks; verify cold/warm probe logits."""
    import torch
    from openbonsai.mixed_formats import MixedLinear
    measure = measure or (lambda _label, operation: operation())
    require(execution['model_kind'] == 'mixed' and execution['cache_policy'] == 'warm', 'Warm policy requires mixed model')
    probe = plan['probes'][0]
    def probe_logits():
        with torch.no_grad(), backend._precision():
            hidden = backend._hidden(probe['input_ids'])
            return backend._logits(hidden, [probe['score_positions'][-1] - 1])
    before = measure('cold-cache-logit-probe', probe_logits)
    rows = []
    for name, module in backend.model.named_modules():
        if isinstance(module, MixedLinear):
            check_deadline(deadline)
            measure('warm-decoded-cache:' + name, module.prepare_decoded_weight_cache)
            rows.append({'name': name, 'bytes': module.decoded_weight_cache_bytes})
    after = measure('warm-cache-logit-probe', probe_logits)
    require(torch.equal(before, after), 'Cached and uncached native logits differ')
    execution['cache_bytes'] = sum(row['bytes'] for row in rows)
    receipt = {'status': 'passed', 'all_probe_logits_bitwise_equal': True,
               'probe_input_ids': probe['input_ids'], 'prediction_position': probe['score_positions'][-1] - 1,
               'modules': rows, 'cache_bytes': execution['cache_bytes'],
               'serialized_payload_modified': False, 'low_bit_kernel_acceleration': False}
    write_new(Path(output) / 'cache-warmup.json', receipt)
    return receipt


def code_identity():
    paths = sorted((ROOT / 'openbonsai').rglob('*.py')) + [Path(__file__).resolve()]
    return {str(path.relative_to(ROOT)): digest(path) for path in paths}


def child_main(job_path):
    job = json.loads(Path(job_path).read_bytes())
    require(os.getppid() == job['supervisor_pid'], 'Child must run under its owning supervisor')
    require(code_identity() == job['code_sha256'], 'Source code changed after freeze')
    args, output = argparse.Namespace(**job['arguments']), Path(job['output'])
    deadline = job['deadline_monotonic']
    backend = None
    status = {'status': 'starting', 'models_per_child': 1, 'fresh_heldout_confirmation': False}
    write_new(output / 'child-status.json', status)
    try:
        check_deadline(deadline)
        manifest = read_pinned_json(args.manifest, args.expected_manifest_sha256)
        profile = read_pinned_json(args.profile, args.expected_profile_sha256)
        plan = build_plan(manifest, profile)
        meter = RuntimeMeter(output / 'phases', args.device, deadline, plan['minimum_free_cuda_bytes'])
        backend, execution = load_backend(args, plan, output, deadline, measure=meter)
        # No scoring, including cache probes, may precede exact full tokenization.
        from openbonsai.evaluation import verify_tokenization
        token_gate = meter('pre-cache-tokenization-gate', lambda: verify_tokenization(backend, manifest))
        write_new(output / 'pre-cache-tokenization-gate.json', token_gate)
        if args.cache_policy == 'warm':
            warm_cache(backend, execution, plan, output, deadline, measure=meter)
        write_new(output / 'execution.json', execution)
        wrapped = ExecutionBackend(backend, execution)
        status.update(evaluate_plan(manifest, profile, wrapped, output, deadline, measure=meter))
        expected_source = {key: value for key, value in backend._source.items() if key != 'model_kind'}
        meter('final-source-preservation', lambda: require(check_source_inputs(args) ==
              (expected_source, backend._tokenizer_identity), 'Source identity differs'))
    except BaseException as error:
        status.update(status='failed', error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc())
    finally:
        if backend is not None:
            backend.close()
        status['elapsed_from_supervisor_start_seconds'] = time.monotonic() - job['started_monotonic']
        update_owned(output / 'child-status.json', status)
    return 0 if status['status'] == 'complete' else 1


def supervisor_main(args):
    require(type(args.max_seconds) in (int, float) and math.isfinite(args.max_seconds)
            and 0 < args.max_seconds <= MAX_SECONDS, 'max-seconds must be in (0,10800]')
    started = time.monotonic()
    deadline = started + args.max_seconds
    output = Path(args.output).absolute()
    require(not output.exists() and not output.is_symlink(), 'Refusing existing output')
    for name in ('model', 'tokenizer', 'manifest', 'profile'):
        path = Path(getattr(args, name)).resolve()
        require(path != output.resolve() and not output.resolve().is_relative_to(path), 'Output cannot be inside an input source')
    output.mkdir(parents=True, exist_ok=False)
    try:
        manifest = read_pinned_json(args.manifest, args.expected_manifest_sha256)
        profile = read_pinned_json(args.profile, args.expected_profile_sha256)
        plan = build_plan(manifest, profile)
        check_source_inputs(args)
        check_deadline(deadline)
        write_new(output / 'frozen-inputs.json', manifest)
        write_new(output / 'profile.json', profile)
        arguments = vars(args).copy()
        for name in ('model', 'tokenizer', 'manifest', 'profile', 'output'):
            arguments[name] = str(Path(arguments[name]).absolute())
        arguments.pop('child_job', None)
        job = {'schema_version': 1, 'supervisor_pid': os.getpid(), 'output': str(output),
               'arguments': arguments, 'started_monotonic': started, 'deadline_monotonic': deadline,
               'maximum_whole_process_seconds': args.max_seconds, 'code_sha256': code_identity(),
               'manifest_sha256': manifest['frozen_sha256'], 'models_per_child': 1,
               'workload': plan['workload'], 'role': manifest['role']}
        write_new(output / 'job.json', job)
        from openbonsai.evaluation_supervisor import supervise
        env = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                   PYTHONDONTWRITEBYTECODE='1', TOKENIZERS_PARALLELISM='false',
                   OMP_NUM_THREADS=str(plan['threads']), MKL_NUM_THREADS=str(plan['threads']))
        if args.device == 'cpu':
            env['CUDA_VISIBLE_DEVICES'] = '-1'
        result = supervise([sys.executable, str(Path(__file__).resolve()), '--child-job', str(output / 'job.json')],
                           output, deadline, env=env)
        if result['status'] == 'complete':
            require(json.loads((output / 'child-status.json').read_bytes())['status'] == 'complete'
                    and (output / 'evaluation.json').is_file(), 'Missing complete evaluation receipt')
        result.update(whole_process_elapsed_seconds=time.monotonic() - started,
                      maximum_whole_process_seconds=args.max_seconds,
                      manifest_sha256=manifest['frozen_sha256'], model_kind=args.kind,
                      fresh_heldout_confirmation=False, legacy_fp32_cross_entropy_compatible=False)
        write_new(output / 'result.json', result)
        print(json.dumps({'status': result['status'], 'output': str(output)}))
        return 0 if result['status'] == 'complete' else 1
    except BaseException as error:
        write_new(output / 'failure.json', {'status': 'failed', 'error': f'{type(error).__name__}: {error}',
                  'traceback': traceback.format_exc(), 'whole_process_elapsed_seconds': time.monotonic() - started})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--child-job', type=Path, help=argparse.SUPPRESS)
    for name in ('manifest', 'profile', 'model', 'tokenizer', 'output'):
        parser.add_argument('--' + name, type=Path)
    for name in ('manifest', 'profile', 'model', 'tokenizer'):
        parser.add_argument('--expected-' + name + '-sha256')
    parser.add_argument('--kind', choices=('original', 'mixed'))
    parser.add_argument('--device', choices=('cpu', 'cuda:0'), default='cpu')
    parser.add_argument('--cache-policy', choices=('cold', 'warm'), default='cold')
    parser.add_argument('--max-seconds', type=float)
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT))
    if args.child_job:
        return child_main(args.child_job)
    required = ('manifest', 'profile', 'model', 'tokenizer', 'output', 'kind', 'max_seconds',
                'expected_manifest_sha256', 'expected_profile_sha256', 'expected_model_sha256',
                'expected_tokenizer_sha256')
    if not args.run or any(getattr(args, name) is None for name in required):
        parser.error('--run and all explicit paths, hashes, model kind and deadline are required')
    return supervisor_main(args)


if __name__ == '__main__':
    raise SystemExit(main())
