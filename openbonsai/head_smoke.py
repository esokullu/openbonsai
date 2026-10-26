"""Paired native random-fixture head calibration, export and fresh reload.

Run: python -m openbonsai.head_smoke --output runs/head-smoke
This is a software integration check, not evidence of pretrained 9B quality.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import torch

from .adapter import tiny_model
from .endpoint_export import export_head_candidate
from .endpoint_quantizer import BoundedTernaryHead
from .head_calibration import (HeadTrainingConfig, cache_original_head_inputs,
                               evaluate_head_from_cache, fit_head_from_cache)
from .mixed_export import load_mixed, prepare_mixed_export, save_mixed
from .mixed_formats import MixedLinear
from .rotations import fixed_signs


def _model_hash(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _records(start):
    return [{'input_ids': torch.arange(start+i*12, start+i*12+8).reshape(1, 8),
             'attention_mask': torch.ones(1, 8, dtype=torch.long)} for i in range(2)]


def _logits(model, records):
    with torch.no_grad():
        return torch.cat([model(**record).logits.cpu() for record in records])


class _HardHeadReference(torch.nn.Module):
    def __init__(self, head):
        super().__init__()
        self.head = head

    def forward(self, hidden):
        return torch.cat([self.head.forward_rows(hidden, start, min(start+self.head.row_chunk_size,
                                                                    self.head.out_features))
                          for start in range(0, self.head.out_features, self.head.row_chunk_size)], dim=-1)


def _nll(logits, records):
    ids = torch.cat([r['input_ids'] for r in records])
    mask = torch.cat([r['attention_mask'] for r in records]).bool()
    valid = mask[:, :-1] & mask[:, 1:]
    losses = torch.nn.functional.cross_entropy(logits[:, :-1].float().flatten(0, 1),
                                               ids[:, 1:].flatten(), reduction='none').reshape_as(valid)
    return float(losses[valid].mean())


def _reload_worker(artifact, inputs, output):
    torch.set_num_threads(2)
    records = torch.load(inputs, map_location='cpu', weights_only=True)
    model = load_mixed(artifact)
    torch.save({'logits': _logits(model, records),
                'eos_token_id': model.generation_config.eos_token_id}, output)


def run(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    original = tiny_model(seed=5100, multimodal=True).to(torch.bfloat16)
    original.generation_config.eos_token_id = [2, 3]
    initial_hash = _model_hash(original)
    provenance = {'source_checkpoint': 'random-native-Qwen3.5-head-fixture',
                  'source_content_sha256': initial_hash}
    fitting_records, validation_records = _records(4), _records(80)
    fitting = cache_original_head_inputs(original, fitting_records, output / 'calibration-cache',
                                         source_provenance=provenance, role='calibration', shard_positions=4)
    validation = cache_original_head_inputs(original, validation_records, output / 'validation-cache',
                                            source_provenance=provenance, role='search_validation', shard_positions=4)
    reference = _logits(original, validation_records)
    inputs_file = output / 'reload-inputs.pt'
    torch.save(validation_records, inputs_file)
    report = {'scope': 'random native hybrid fixture; no pretrained quality or CUDA memory claim',
              'source': provenance, 'conditions': {},
              'coverage': 'lm_head only; decoder, embedding, vision and small tensors original BF16',
              'calibration_records': 2, 'validation_records': 2,
              'tokenizer_included': False,
              'runtime': 'floating-point weight reconstruction; no ternary kernel'}
    config = HeadTrainingConfig(epochs=2, batch_size=1, token_microbatch=4,
                                vocabulary_chunk=31, learning_rate=.005,
                                lora_learning_rate=.001, seed=5090)
    for rotated in (False, True):
        name = 'rotated' if rotated else 'unrotated'
        torch.manual_seed(5090)
        head = BoundedTernaryHead(original.lm_head, group_size=128, row_chunk_size=31,
                                  lora_rank=4, factor_mapping='sigmoid',
                                  signs=fixed_signs(128, seed=5090, name='lm_head') if rotated else None,
                                  block_size=64)
        started = time.perf_counter()
        result = fit_head_from_cache(head, fitting, config, source_provenance=provenance,
                                     validation_cache=validation)
        training_seconds = time.perf_counter() - started
        held_out = evaluate_head_from_cache(head, validation, source_provenance=provenance,
                                             token_microbatch=4, vocabulary_chunk=31)
        candidate = export_head_candidate(head, packing='auto', provenance=provenance)
        artifact = output / name
        with prepare_mixed_export(original, ['lm_head'], metadata=provenance) as plan:
            manifest = save_mixed(plan, {'lm_head': candidate}, artifact, budget_bytes=10**9)
        packed = copy.deepcopy(original)
        packed.lm_head = MixedLinear(candidate)
        packed_logits = _logits(packed, validation_records)
        hard = copy.deepcopy(original)
        hard.lm_head = _HardHeadReference(head)
        hard_logits = _logits(hard, validation_records)
        same_chunks = copy.deepcopy(original)
        same_chunks.lm_head = MixedLinear(candidate, row_chunk_size=head.row_chunk_size)
        same_chunk_logits = _logits(same_chunks, validation_records)
        reload_file = output / f'{name}-fresh-reload.pt'
        subprocess.run([sys.executable, '-m', 'openbonsai.head_smoke', '--reload-worker',
                        str(artifact.resolve()), str(inputs_file.resolve()), str(reload_file.resolve())],
                       check=True, timeout=120)
        reloaded = torch.load(reload_file, map_location='cpu', weights_only=True)
        torch.testing.assert_close(reloaded['logits'], packed_logits, atol=0, rtol=0)
        if reloaded['eos_token_id'] != [2, 3]:
            raise AssertionError('fresh reload lost multi-EOS configuration')
        if _model_hash(original) != initial_hash:
            raise AssertionError('head calibration changed original model')
        torch.testing.assert_close(_logits(original, validation_records), reference, atol=0, rtol=0)
        torch.save(result.trainable_state, output / f'{name}-trainable-state.pt')
        report['conditions'][name] = {
            'training': result.report, 'validation': held_out,
            'training_seconds': training_seconds,
            'candidate_encoding': candidate.encoding,
            'artifact_bytes': int(manifest['storage']['artifact_bytes']),
            'candidate_bytes': int(manifest['storage']['candidate_bytes']),
            'retained_vision_file_bytes': int(manifest['storage']['vision_file_bytes']),
            'random_fixture_nll': {'original': _nll(reference, validation_records),
                                   'hard_fp32_scales_bf16_compute': _nll(hard_logits, validation_records),
                                   'exported_fp16_scales_bf16_compute': _nll(same_chunk_logits, validation_records)},
            'scale_export_max_logit_difference_same_chunks': float((hard_logits.float()-same_chunk_logits.float()).abs().max()),
            'runtime_chunk_change_max_logit_difference': float((same_chunk_logits.float()-packed_logits.float()).abs().max()),
            'fresh_process_max_logit_difference': 0.0,
            'original_model_unchanged': True, 'multi_eos_preserved': True,
        }
    (output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='runs/head-smoke')
    parser.add_argument('--reload-worker', nargs=3, metavar=('ARTIFACT', 'INPUTS', 'OUTPUT'))
    args = parser.parse_args()
    if args.reload_worker:
        _reload_worker(*args.reload_worker)
    else:
        report = run(args.output)
        print(json.dumps({name: {'validation_kl': condition['validation']['kl'],
                                'artifact_bytes': condition['artifact_bytes'],
                                'fresh_process_max_logit_difference': 0.0}
                          for name, condition in report['conditions'].items()}, indent=2))


if __name__ == '__main__':
    main()
