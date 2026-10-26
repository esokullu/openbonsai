"""Embedding-only table fitting and native hybrid-model fresh-process reload.

Run: python -m openbonsai.embedding_smoke --output runs/embedding-smoke
Random CPU fixture only: no pretrained quality, CUDA feasibility or speed claim.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
import time

import torch
from transformers import LogitsProcessorList

from .adapter import tiny_model
from .embedding_calibration import EmbeddingTrainingConfig, fit_embedding_table
from .embedding_quantizer import BoundedTernaryEmbedding
from .endpoint_export import export_embedding_candidate
from .head_smoke import _logits, _model_hash, _nll
from .mixed_export import load_mixed, prepare_mixed_export, save_mixed
from .mixed_formats import MixedEmbedding, candidate_nbytes
from .rotations import fixed_signs


class _ForceEOS:
    def __init__(self, prompt_length, eos):
        self.prompt_length, self.eos = prompt_length, eos

    def __call__(self, ids, scores):
        scores.fill_(float('-inf'))
        scores[:, 7 if ids.shape[-1] == self.prompt_length else self.eos] = 0
        return scores


def _forced_generations(model, ids):
    """Exercise both stop tokens; forced outputs are not quality evidence."""
    with torch.no_grad():
        return {str(eos): model.generate(
            input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False,
            do_sample=False, max_new_tokens=5,
            logits_processor=LogitsProcessorList([_ForceEOS(ids.shape[-1], eos)]),
        ).cpu().tolist() for eos in (2, 3)}


def _reload_worker(artifact, inputs, output):
    torch.set_num_threads(2)
    records = torch.load(inputs, map_location='cpu', weights_only=True)
    model = load_mixed(artifact)
    torch.save({'logits': _logits(model, records),
                'eos_token_id': model.generation_config.eos_token_id,
                'forced_generations': _forced_generations(model, records[0]['input_ids'][:, :3])}, output)


def run(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    original = tiny_model(seed=5101, multimodal=True).to(torch.bfloat16)
    original.generation_config.eos_token_id = [2, 3]
    original.generation_config.pad_token_id = 0
    source = original.get_input_embeddings()
    module_name = next(name for name, module in original.named_modules() if module is source)
    initial_hash = _model_hash(original)
    provenance = {'source_checkpoint': 'random-native-Qwen3.5-embedding-fixture',
                  'source_content_sha256': initial_hash}
    # None of these text inputs is passed to table fitting or used for selection.
    records = [{'input_ids': torch.tensor([[4, 7, 7, 80, 2, 100, 255, 5]]),
                'attention_mask': torch.ones((1, 8), dtype=torch.long)},
               {'input_ids': torch.tensor([[90, 120, 3, 120, 200, 201, 5, 9]]),
                'attention_mask': torch.ones((1, 8), dtype=torch.long)}]
    reference = _logits(original, records)
    inputs_file = output / 'reload-inputs.pt'
    torch.save(records, inputs_file)
    report = {'scope': 'random native hybrid CPU fixture; no pretrained quality or CUDA memory claim',
              'source': provenance, 'module': module_name, 'conditions': {},
              'coverage': 'embedding only; original decoder/head/vision retained identically',
              'fitting_inputs': 'all original BF16 embedding rows; no text, token labels or evaluation answers',
              'evaluation': 'two unused random-fixture text records; not language-quality evidence',
              'tokenizer_included': False, 'seed': 5090,
              'runtime': 'floating-point row reconstruction and explicit transforms; no low-bit kernel'}
    config = EmbeddingTrainingConfig(steps=4, factor_lr=.005, lora_lr=.001)
    for rotated in (False, True):
        name = 'rotated' if rotated else 'unrotated'
        torch.manual_seed(5090)
        embedding = BoundedTernaryEmbedding(
            source, group_size=32, row_chunk_size=31, lora_rank=4,
            signs=fixed_signs(source.embedding_dim, seed=5090, name=module_name) if rotated else None,
            block_size=64)
        hard = copy.deepcopy(original)
        hard.set_submodule(module_name, embedding)
        embedding.quant_enabled = False
        torch.testing.assert_close(_logits(hard, records), reference, atol=0, rtol=0)
        embedding.quant_enabled = True
        started = time.perf_counter()
        fit = fit_embedding_table(embedding, config, source_provenance=provenance)
        training_seconds = time.perf_counter() - started
        candidate = export_embedding_candidate(embedding, packing='auto', provenance=provenance)
        hard_logits = _logits(hard, records)
        packed = copy.deepcopy(original)
        packed.set_submodule(module_name, MixedEmbedding(candidate, padding_idx=source.padding_idx))
        packed_logits = _logits(packed, records)
        expected_generation = _forced_generations(packed, records[0]['input_ids'][:, :3])
        for eos in (2, 3):
            if expected_generation[str(eos)] != [[4, 7, 7, 7, eos]]:
                raise AssertionError('Native generation did not stop on a configured EOS')
        artifact = output / name
        with prepare_mixed_export(original, [module_name], metadata=provenance) as plan:
            budget = plan.fixed_bytes + candidate_nbytes(candidate)
            manifest = save_mixed(plan, {module_name: candidate}, artifact, budget_bytes=budget)
        actual_bytes = sum(p.stat().st_size for p in artifact.rglob('*') if p.is_file())
        if actual_bytes != budget or actual_bytes != int(manifest['storage']['artifact_bytes']):
            raise AssertionError('Embedding artifact differs from its exact predeclared byte budget')
        reload_file = output / f'{name}-fresh-reload.pt'
        subprocess.run([sys.executable, '-m', 'openbonsai.embedding_smoke', '--reload-worker',
                        str(artifact.resolve()), str(inputs_file.resolve()), str(reload_file.resolve())],
                       check=True, timeout=120)
        reloaded = torch.load(reload_file, map_location='cpu', weights_only=True)
        torch.testing.assert_close(reloaded['logits'], packed_logits, atol=0, rtol=0)
        if reloaded['eos_token_id'] != [2, 3] or reloaded['forced_generations'] != expected_generation:
            raise AssertionError('Fresh embedding artifact changed EOS generation behavior')
        embedding.quant_enabled = False
        torch.testing.assert_close(_logits(hard, records), reference, atol=0, rtol=0)
        embedding.quant_enabled = True
        if _model_hash(original) != initial_hash:
            raise AssertionError('Embedding calibration changed the original model')
        torch.testing.assert_close(_logits(original, records), reference, atol=0, rtol=0)
        report['conditions'][name] = {
            'training': fit.report, 'training_seconds': training_seconds,
            'candidate_encoding': candidate.encoding,
            'artifact_bytes': actual_bytes, 'candidate_bytes': candidate_nbytes(candidate),
            'retained_vision_file_bytes': int(manifest['storage']['vision_file_bytes']),
            'random_fixture_nll': {'original': _nll(reference, records),
                                   'hard_fp32_scales_bf16_compute': _nll(hard_logits, records),
                                   'exported_fp16_scales_bf16_compute': _nll(packed_logits, records)},
            'scale_export_max_logit_difference': float((hard_logits.float()-packed_logits.float()).abs().max()),
            'fresh_process_max_logit_difference': 0.,
            'original_model_unchanged': True, 'original_bypass_exact_before_and_after': True,
            'multi_eos_preserved': True, 'forced_generations_are_quality_evidence': False,
        }
    (output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='runs/embedding-smoke')
    parser.add_argument('--reload-worker', nargs=3, metavar=('ARTIFACT', 'INPUTS', 'OUTPUT'))
    args = parser.parse_args()
    if args.reload_worker:
        _reload_worker(*args.reload_worker)
    else:
        report = run(args.output)
        print(json.dumps({name: {'table_mse': condition['training']['final_hard_reconstruction']['mse'],
                                'artifact_bytes': condition['artifact_bytes'],
                                'fresh_process_max_logit_difference': 0.}
                          for name, condition in report['conditions'].items()}, indent=2))


if __name__ == '__main__':
    main()
