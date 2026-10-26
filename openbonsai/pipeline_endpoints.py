"""Explicit original-cache and bounded endpoint stages for a sealed local job.

Each fit publishes a complete ternary candidate, never an optimizer/resume
checkpoint. Head fitting uses original-decoder cached inputs; embedding fitting
uses every original table row. Neither is a joint assembled-quality claim.
"""
from __future__ import annotations

import copy
from dataclasses import fields
import gc
import hashlib
import json
import math
from pathlib import Path

import torch
from torch import nn

from .adapter import _text_module
from .embedding_calibration import (EmbeddingTrainingConfig, fit_embedding_table,
                                    require_complete_calibration)
from .embedding_quantizer import BoundedTernaryEmbedding
from .endpoint_export import export_embedding_candidate, export_head_candidate
from .endpoint_quantizer import BoundedTernaryHead
from .head_calibration import (HeadTrainingConfig, _record_inputs,
                               cache_original_head_inputs, evaluate_head_from_cache,
                               fit_head_from_cache, load_head_input_cache)
from . import pipeline_sources as sources
from .rotations import fixed_signs

CACHE_STAGE_FORMAT = 'openbonsai.pipeline-head-cache-stage.v1'
_QUANTIZATION_KEYS = {'group_size', 'row_chunk_size', 'block_size', 'lora_rank',
    'factor_mapping', 'hard_phase', 'soft_fraction', 'max_sharpness',
    'rotation_seed', 'initialization_seed'}


def _operation(operation, label, function):
    return function() if operation is None else operation(label, function)


def _event(callback, stage, **event):
    if callback is not None:
        callback(copy.deepcopy({'stage': stage, **event}))


def _exact(value, keys, label):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError(f'{label} requires exactly {sorted(keys)}')


def _recipe(context, stage):
    recipe = copy.deepcopy(context.job[stage])
    _exact(recipe, {'quantization', 'training', 'packing'} | ({'cache'} if stage == 'head' else set()), stage)
    quantization = recipe['quantization']
    keys = _QUANTIZATION_KEYS | ({'rotation_mode'} if stage == 'head'
                                 and isinstance(quantization, dict) and 'rotation_mode' in quantization else set())
    _exact(quantization, keys, stage + '.quantization')
    sources._rotation_mode(stage, quantization)
    for name in ('rotation_seed', 'initialization_seed'):
        if type(quantization[name]) is not int or not 0 <= quantization[name] < 2**63:
            raise ValueError(f'{name} must be an integer in [0,2**63)')
    if recipe['packing'] not in {'auto', '2bit', 'bitcos', 'base3'}:
        raise ValueError('Unsupported endpoint packing')
    cls = HeadTrainingConfig if stage == 'head' else EmbeddingTrainingConfig
    _exact(recipe['training'], {field.name for field in fields(cls)}, stage + '.training')
    training = cls(**recipe['training'])
    training.validate()
    if stage == 'head':
        cache = recipe['cache']
        _exact(cache, {'max_positions', 'shard_positions'}, 'head.cache')
        for key, value in cache.items():
            if key == 'max_positions' and value is None:
                continue
            if type(value) is not int or value < 1:
                raise ValueError(f'head.cache.{key} must be positive or max_positions=null')
    return recipe, training


def _file_tree(directory):
    result = {}
    directory = Path(directory)
    if directory.is_symlink():
        raise ValueError('Cache directory must not be a symlink')
    for path in sorted(directory.rglob('*')):
        if path.is_symlink():
            raise ValueError('Cache symlinks are unsupported')
        if path.is_file():
            digest = hashlib.sha256()
            with path.open('rb') as handle:
                for block in iter(lambda: handle.read(1 << 20), b''):
                    digest.update(block)
            result[path.relative_to(directory).as_posix()] = {
                'bytes': path.stat().st_size, 'sha256': digest.hexdigest()}
    return result


def _records(context, role):
    if role not in sources.ROLES or role not in context.records_by_role:
        raise ValueError('Cache role is absent from the sealed job')
    receipt = copy.deepcopy(context.record_receipts_by_role[role])
    if receipt != context.job['traces_by_role'][role]:
        raise ValueError('In-memory record receipt differs from sealed job')
    records = copy.deepcopy(context.records_by_role[role])
    hashes = []
    for record in records:
        # Current source adapter deliberately exposes only normalized IDs/masks.
        _exact(record, {'input_ids', 'attention_mask'}, 'normalized trace record')
        ids, mask = record['input_ids'], record['attention_mask']
        if (not isinstance(ids, torch.Tensor) or not isinstance(mask, torch.Tensor)
                or ids.device.type != 'cpu' or mask.device.type != 'cpu'
                or ids.dtype != torch.int64 or mask.dtype != torch.int64
                or ids.ndim != 2 or ids.shape[0] != 1 or mask.shape != ids.shape):
            raise ValueError('Normalized records must be CPU int64 [1,sequence] tensors')
        hashes.append(sources.json_sha256({'input_ids': ids[0].tolist(),
                                          'attention_mask': mask[0].tolist()}))
    if hashes != receipt['record_sha256s'] or len(records) != receipt['num_records']:
        raise ValueError('In-memory normalized records differ from sealed ordered input')
    return records, receipt


def _cache_record_binding(records, vocabulary, maximum):
    expected, positions, seen = [], 0, 0
    for record in records:
        _, _, coordinates, info = _record_inputs(record, vocabulary)
        seen += 1
        if not len(coordinates):
            continue
        count = len(coordinates) if maximum is None else min(len(coordinates), maximum - positions)
        expected.append({**info, 'positions': count})
        positions += count
        if maximum is not None and positions >= maximum:
            break
    return {'records': expected, 'positions': positions, 'records_seen': seen}


def _check_cache_records(cache, records, maximum):
    expected = _cache_record_binding(records, cache.manifest['vocabulary'], maximum)
    actual = {key: cache.manifest[key] for key in ('positions', 'records_seen')}
    actual['records'] = [{key: value for key, value in record.items() if key != 'shards'}
                         for record in cache.manifest['records']]
    if actual != expected:
        raise ValueError('Cache rows do not match exact ordered source records/position limit')
    return expected


def _cache_binding(context, role, recipe, record_receipt):
    return {'job_sha256': context.job_sha256, 'source_provenance': copy.deepcopy(context.source_provenance),
            'code_sha256': context.job['code']['sha256'], 'role': role,
            'record_receipt_sha256': record_receipt['sha256'],
            'head_config_sha256': sources.json_sha256(recipe),
            'cache_configuration': recipe['cache']}


def cache_head_stage(context, role, output, device='cpu', *, operation=None, progress_callback=None):
    """Publish original final-normalized hidden inputs with exact role binding."""
    output = sources.validate_stage_output(context, output)
    _operation(operation, 'verify-context', lambda: sources.verify_context(context))
    recipe, _ = _recipe(context, 'head')
    records, record_receipt = _records(context, role)
    output.mkdir(parents=True, exist_ok=False)
    model = None
    try:
        model = _operation(operation, 'load-original-for-head-cache', lambda: sources.load_original(context, device=device))
        def inputs():
            for index, record in enumerate(records):
                _event(progress_callback, 'cache-head', role=role, record=index, records=len(records))
                yield record
        cache = _operation(operation, 'cache-original-head-inputs', lambda: cache_original_head_inputs(
            model, inputs(), output / 'cache', context.source_provenance, role=role, **recipe['cache']))
        binding = _check_cache_records(cache, records, recipe['cache']['max_positions'])
        if cache.manifest['source_provenance'] != context.source_provenance:
            raise ValueError('Cache original source provenance differs')
        del model
        model = None
        gc.collect()
        _event(progress_callback, 'cache-head', role=role, phase='before-publication', positions=cache.positions)
        _operation(operation, 'verify-context-before-publication', lambda: sources.verify_context(context))
        receipt = {'format': CACHE_STAGE_FORMAT, 'status': 'complete',
                   **_cache_binding(context, role, recipe, record_receipt),
                   'cache_files': _file_tree(output / 'cache'), 'record_binding': binding,
                   'capture': cache.manifest['capture'], 'quality_claim': False}
        receipt['sha256'] = sources.json_sha256(receipt)
        with (output / 'stage.json').open('xb') as handle:
            handle.write(sources.canonical_json(receipt) + b'\n')
        return receipt
    finally:
        del model
        gc.collect()


def load_head_cache_stage(context, directory, *, role):
    """Reject partial, stale, foreign-role or altered cached-stage outputs."""
    sources.verify_context(context)
    recipe, _ = _recipe(context, 'head')
    records, record_receipt = _records(context, role)
    root = Path(directory)
    if root.is_symlink() or (root / 'stage.json').is_symlink():
        raise ValueError('Cache-stage symlinks are unsupported')
    receipt = json.loads((root / 'stage.json').read_bytes())
    body = {key: value for key, value in receipt.items() if key != 'sha256'}
    if (receipt.get('sha256') != sources.json_sha256(body) or receipt.get('format') != CACHE_STAGE_FORMAT
            or receipt.get('status') != 'complete'):
        raise ValueError('Invalid or incomplete head-cache stage receipt')
    if any(receipt.get(key) != value for key, value in _cache_binding(context, role, recipe, record_receipt).items()):
        raise ValueError('Cache-stage job/source/code/role/records/config binding differs')
    if {p.name for p in root.iterdir()} != {'cache', 'stage.json'}:
        raise ValueError('Unaccounted cache-stage files')
    if _file_tree(root / 'cache') != receipt['cache_files']:
        raise ValueError('Cache-stage files changed')
    cache = load_head_input_cache(root / 'cache')
    if cache.manifest['role'] != role or cache.manifest['source_provenance'] != context.source_provenance:
        raise ValueError('Internal cache role/source binding differs')
    if _check_cache_records(cache, records, recipe['cache']['max_positions']) != receipt['record_binding']:
        raise ValueError('Cache-stage record accounting differs')
    return cache, receipt


def _endpoint_source(context, stage, device, operation):
    original = _operation(operation, 'load-original-cpu', lambda: sources.load_original(context, device='cpu'))
    prefix, text = _text_module(original)
    head, embedding = original.lm_head, original.get_input_embeddings()
    if (getattr(text.config, 'tie_word_embeddings', False)
            or head.weight.untyped_storage().data_ptr() == embedding.weight.untyped_storage().data_ptr()):
        raise ValueError('Tied source endpoints are unsupported')
    name = 'lm_head' if stage == 'head' else prefix + '.embed_tokens'
    endpoint = original.get_submodule(name)
    expected = nn.Linear if stage == 'head' else nn.Embedding
    if type(endpoint) is not expected or endpoint.weight.dtype != torch.bfloat16 or endpoint.weight.requires_grad:
        raise ValueError('Endpoint must be an original frozen native BF16 module')
    # No module container retains the decoder when the endpoint reaches GPU.
    del original, text, head, embedding
    gc.collect()
    endpoint = _operation(operation, 'move-only-' + stage, lambda: endpoint.to(device=device))
    return name, endpoint


def _construct(endpoint, stage, name, recipe):
    quantization = copy.deepcopy(recipe['quantization'])
    mode = sources._rotation_mode(stage, quantization)
    quantization.pop('rotation_mode', None)
    rotation_seed, initialization_seed = (quantization.pop(key) for key in ('rotation_seed', 'initialization_seed'))
    signs = (None if mode == 'none' else
             fixed_signs(endpoint.weight.shape[1], seed=rotation_seed, name=name))
    cls = BoundedTernaryHead if stage == 'head' else BoundedTernaryEmbedding
    device = endpoint.weight.device
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(initialization_seed)
        operator = cls(endpoint, signs=signs, **quantization)
    return operator


def _provenance(context, stage, name, operator, report, **extra):
    return sources.candidate_provenance(context, stage, name,
        source_weight_sha256=operator.source_weight_sha256,
        training_report_sha256=sources.json_sha256(report),
        rotation_seed=context.job[stage]['quantization']['rotation_seed'],
        initialization_seed=context.job[stage]['quantization']['initialization_seed'], **extra)


def _head_state_sha256(head):
    """Bind compact final inference state with bounded copies, excluding source.

    Source weight/bias have independent unchanged-byte guards. Derivative-only
    snapshots are excluded because a completed candidate has no training state.
    """
    config = {name: getattr(head, name) for name in (
        'in_features', 'out_features', 'group_size', 'row_chunk_size', 'block_size',
        'lora_rank', 'factor_mapping', 'hard_phase', 'soft_fraction', 'max_sharpness',
        'progress', 'quant_enabled', 'rotation_only')}
    digest = hashlib.sha256(sources.canonical_json(config))
    tensors = {**dict(head.named_parameters()), **dict(head.named_buffers())}
    for name, value in sorted(tensors.items()):
        if name in {'weight', 'bias', '_device_probe'} or name.startswith('_hard_snapshot_'):
            continue
        digest.update(sources.canonical_json({'name': name, 'dtype': str(value.dtype), 'shape': list(value.shape)}))
        values = value.detach() if value.ndim else value.detach().reshape(1)
        rows = max(1, (1 << 20) // max(1, math.prod(values.shape[1:])))
        for start in range(0, values.shape[0], rows):
            digest.update(values[start:start + rows].cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _publish(context, stage, name, operator, report, output, recipe, operation, callback, **extra):
    _event(callback, 'fit-' + stage, phase='before-export')
    operator.assert_source_unchanged(verify_hash=True)
    operator.validate_parameters()
    if stage == 'head' and _head_state_sha256(operator) != report['final_hard_state_sha256']:
        raise ValueError('Final head state changed after fitting')
    provenance = _provenance(context, stage, name, operator, report, **extra)
    exporter = export_head_candidate if stage == 'head' else export_embedding_candidate
    candidate = _operation(operation, 'export-' + stage + '-candidate', lambda: exporter(
        operator, packing=recipe['packing'], provenance=provenance))
    _event(callback, 'fit-' + stage, phase='before-publication')
    operator.assert_source_unchanged(verify_hash=True)
    if stage == 'head' and _head_state_sha256(operator) != report['final_hard_state_sha256']:
        raise ValueError('Final head state changed during export/publication callback')
    if stage == 'embedding':
        if require_complete_calibration(operator) != candidate.metadata['provenance']['calibration_certificate']:
            raise ValueError('Embedding state changed after export')
    _operation(operation, 'verify-context-before-publication', lambda: sources.verify_context(context))
    return _operation(operation, 'publish-' + stage, lambda: sources.publish_candidates(
        context, stage, {name: candidate}, output, report=report))


def fit_head_stage(context, calibration_cache, output, search_cache=None, evaluation_cache=None,
                   device='cpu', *, operation=None, progress_callback=None):
    """Fit one bounded head against original cached full-vocabulary teacher KL."""
    output = sources.validate_stage_output(context, output)
    _operation(operation, 'verify-context', lambda: sources.verify_context(context))
    recipe, training = _recipe(context, 'head')
    caches, receipts = {}, {}
    for role, path in [('calibration', calibration_cache), ('search_validation', search_cache), ('evaluation', evaluation_cache)]:
        if path is not None:
            caches[role], receipts[role] = _operation(operation, 'verify-head-cache-' + role,
                lambda path=path, role=role: load_head_cache_stage(context, path, role=role))
    total = training.epochs * math.ceil(len(caches['calibration']) / training.batch_size)
    if not 0 < sum(step / total <= recipe['quantization']['soft_fraction'] for step in range(1, total + 1)) < total:
        raise ValueError('Head stage needs genuine soft and hard updates')
    name, endpoint = _endpoint_source(context, 'head', device, operation)
    operator = None
    try:
        operator = _operation(operation, 'construct-bounded-head', lambda: _construct(endpoint, 'head', name, recipe))
        del endpoint
        fitted = _operation(operation, 'fit-head', lambda: fit_head_from_cache(operator, caches['calibration'], training,
            source_provenance=context.source_provenance, validation_cache=caches.get('search_validation'),
            evaluation_cache=caches.get('evaluation'), progress_callback=lambda event: _event(
                progress_callback, 'fit-head', phase='update', **event)))
        report = copy.deepcopy(fitted.report)
        del fitted  # No second compact CPU parameter copy persists into export.
        if len(report['steps']) != total or operator.progress != 1.:
            raise RuntimeError('Head fitting did not complete its declared update budget')
        operator.validate_parameters()
        report['final_hard_state_sha256'] = _head_state_sha256(operator)
        report['state_digest_scope'] = 'configuration/factors/LoRA/statistics/signs; source checked separately; no derivative snapshots'
        report['cache_stage_sha256s'] = {role: receipt['sha256'] for role, receipt in receipts.items()}
        report['evaluation_cache_used_for_scoring'] = False
        if 'search_validation' in caches:
            report['search_validation'] = _operation(operation, 'evaluate-head-search', lambda: evaluate_head_from_cache(
                operator, caches['search_validation'], source_provenance=context.source_provenance,
                token_microbatch=training.token_microbatch, vocabulary_chunk=training.vocabulary_chunk))
        return _publish(context, 'head', name, operator, report, output, recipe, operation,
                        progress_callback, cache_stage_sha256s=report['cache_stage_sha256s'])
    finally:
        del operator
        gc.collect()


def fit_embedding_stage(context, output, device='cpu', *, operation=None, progress_callback=None):
    """Fit every embedding row and export its live complete-coverage certificate."""
    output = sources.validate_stage_output(context, output)
    _operation(operation, 'verify-context', lambda: sources.verify_context(context))
    recipe, training = _recipe(context, 'embedding')
    name, endpoint = _endpoint_source(context, 'embedding', device, operation)
    operator = None
    try:
        operator = _operation(operation, 'construct-bounded-embedding', lambda: _construct(endpoint, 'embedding', name, recipe))
        del endpoint
        fitted = _operation(operation, 'fit-embedding-table', lambda: fit_embedding_table(operator, training,
            source_provenance=context.source_provenance, progress_callback=lambda event: _event(
                progress_callback, 'fit-embedding', event_type='update', **event)))
        report = copy.deepcopy(fitted.report)
        report['text_records_used_for_fitting'] = 0
        report['coverage_certificate'] = require_complete_calibration(operator)
        return _publish(context, 'embedding', name, operator, report, output, recipe, operation, progress_callback)
    finally:
        del operator
        gc.collect()
