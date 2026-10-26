"""Memory, complete-file packing, and deployable generation regression checks."""
import copy
from pathlib import Path

import pytest
import torch
from transformers import AutoTokenizer, LogitsProcessorList, PreTrainedTokenizerFast
from tokenizers import Tokenizer
from tokenizers.models import WordLevel

from openbonsai.adapter import tiny_model, replace_projections
from openbonsai.candidates import source_weight_sha256
from openbonsai import mixed_export as me, mixed_formats as mf
from openbonsai.packing import pack_grouped_weight
from openbonsai.rotations import replace_rotated_projections, fixed_signs


@pytest.mark.parametrize("zero_fraction", [0., .4, .95])
def test_rotation_auto_compares_complete_candidate_bytes(zero_fraction):
    generator = torch.Generator().manual_seed(7)
    codes = torch.randint(0, 2, (7, 32), generator=generator, dtype=torch.int8) * 2 - 1
    codes[torch.rand(codes.shape, generator=generator) < zero_fraction] = 0
    state = pack_grouped_weight(codes, torch.full((28,), .125), shape=codes.shape, group_size=8)
    kwargs = dict(rotation_signs=fixed_signs(32, seed=29), rotation_block_size=8)
    candidates = [mf.ternary_candidate(state, packing=codec, **kwargs) for codec in ("2bit", "bitcos")]
    chosen = mf.ternary_candidate(state, packing="auto", **kwargs)
    expected = min(candidates, key=mf.candidate_nbytes)
    assert chosen.encoding == expected.encoding
    assert mf.candidate_nbytes(chosen) == min(map(mf.candidate_nbytes, candidates))
    if chosen.encoding == "ternary-bitcos":
        assert mf.candidate_nbytes(chosen) < mf.candidate_nbytes(candidates[0])
    assert torch.equal(chosen.tensors["rotation_signs"], kwargs["rotation_signs"])
    assert chosen.tensors["rotation_signs"].dtype == torch.int8
    assert torch.equal(chosen.dequantize(), candidates[0].dequantize())


def test_candidate_streamed_save_load_without_full_blob_or_reencoding(tmp_path, monkeypatch):
    weight = torch.arange(256 * 128).reshape(256, 128).bfloat16()
    candidate = mf.bf16_candidate(weight)
    candidate.tensors["weight"] = weight.T.contiguous().T
    expected_blob = mf.encode_candidate(candidate)
    target = tmp_path / "candidate.obc"
    assert mf.save_candidate(candidate, target) == len(expected_blob)
    assert target.read_bytes() == expected_blob
    def forbidden(*args, **kwargs):
        raise AssertionError("full serialized blob copy/re-encoding is unnecessary")
    monkeypatch.setattr(Path, "read_bytes", forbidden)
    monkeypatch.setattr(mf, "encode_candidate", forbidden)
    restored = mf.load_candidate(target)
    assert torch.equal(restored.tensors["weight"], weight)
    runtime = mf.MixedLinear(restored, copy_payload=False, row_chunk_size=512)
    assert runtime.payload_weight.data_ptr() == restored.tensors["weight"].data_ptr()


class ForceContinuationThenEOS:
    def __init__(self, prompt_length, eos):
        self.prompt_length, self.eos = prompt_length, eos

    def __call__(self, ids, scores):
        scores.fill_(float("-inf"))
        scores[:, 7 if ids.shape[-1] == self.prompt_length else self.eos] = 0
        return scores


@pytest.mark.parametrize("rotated", [False, True])
@pytest.mark.parametrize("chunk", [64, 512])
def test_mixed_native_meta_reload_chunk_option_assets_and_both_eos(tmp_path, monkeypatch, rotated, chunk):
    torch.set_num_threads(4)
    model = tiny_model(seed=23, multimodal=True).to(torch.bfloat16)
    model.generation_config.eos_token_id = [2, 3]
    model.generation_config.pad_token_id = 0
    prefix = "model.language_model"
    names = [prefix + ".layers.0.mlp.down_proj", prefix + ".layers.3.self_attn.o_proj"]
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel(
        {"<pad>": 0, "<bos>": 1, "<end>": 2, "<turn-end>": 3, "hello": 4, "<unk>": 5}, unk_token="<unk>")),
        bos_token="<bos>", eos_token="<turn-end>", pad_token="<pad>", unk_token="<unk>")
    tokenizer.chat_template = "{% for message in messages %}{{ message['content'] }}{% endfor %}"
    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer.save_pretrained(tokenizer_dir)
    source_hashes = {name: source_weight_sha256(model.get_submodule(name).weight) for name in names}
    selected_ptrs = {model.get_submodule(name).weight.data_ptr() for name in names}
    writer = me._write_tensor_file
    def retained_only(directory, filename, tensors, files):
        assert not any(t.data_ptr() in selected_ptrs for t in tensors.values())
        return writer(directory, filename, tensors, files)
    monkeypatch.setattr(me, "_write_tensor_file", retained_only)
    with me.prepare_mixed_export(model, names, metadata={"source_checkpoint": "tiny-original-BF16"},
                                 tokenizer_dir=tokenizer_dir, row_chunk_size=chunk) as plan:
        assert all(plan.manifest["modules"][n]["source_weight_sha256"] == source_hashes[n] for n in names)
        if rotated:
            replace_rotated_projections(model, block_size=64, lora_rank=3)
        else:
            replace_projections(model, lora_rank=3)
        assignment = {}
        for name in names:
            module = model.get_submodule(name)
            provenance = {"source_weight_sha256": source_hashes[name]}
            assignment[name] = (module.mixed_candidate(provenance=provenance) if rotated else
                                mf.ternary_candidate(module.packed_state(), provenance=provenance))
        # Plan captures every unselected original tensor before wrapping. Build
        # the reference from a fresh identical original plus selected payloads.
        expected = tiny_model(seed=23, multimodal=True).to(torch.bfloat16)
        expected.generation_config = copy.deepcopy(model.generation_config)
        for name, candidate in assignment.items():
            expected.set_submodule(name, mf.MixedLinear(candidate, row_chunk_size=chunk))
        budget = plan.fixed_bytes + sum(map(mf.candidate_nbytes, assignment.values()))
        target = tmp_path / "packed"
        manifest = me.save_mixed(plan, assignment, target, budget_bytes=budget)
    assert int(manifest["storage"]["artifact_bytes"]) == budget
    assert int(manifest["runtime_options"]["row_chunk_size"]) == chunk
    factory = me.model_from_config
    created = []
    def on_meta(config):
        native = factory(config)
        assert all(parameter.is_meta for parameter in native.parameters())
        assert all(not buffer.is_meta for buffer in native.buffers())
        created.append(True)
        return native
    monkeypatch.setattr(me, "model_from_config", on_meta)
    loaded = me.load_mixed(target)
    assert created and not any(t.is_meta for t in loaded.parameters())
    assert all(loaded.get_submodule(name).row_chunk_size == chunk for name in names)
    overridden = me.load_mixed(target, row_chunk_size=64)
    assert all(overridden.get_submodule(name).row_chunk_size == 64 for name in names)
    ids = torch.tensor([[4, 7, 9]])
    with torch.no_grad():
        wanted = expected(input_ids=ids, use_cache=False).logits
        assert torch.equal(loaded(input_ids=ids, use_cache=False).logits, wanted)
        assert torch.equal(overridden(input_ids=ids, use_cache=False).logits, wanted)
        for name in names:
            assert torch.equal(loaded.get_submodule(name).weight, expected.get_submodule(name).weight)
        for eos in (2, 3):
            for instance in (expected, loaded):
                generated = instance.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                    use_cache=False, do_sample=False, max_new_tokens=5,
                    logits_processor=LogitsProcessorList([ForceContinuationThenEOS(ids.shape[1], eos)]))
                assert generated.tolist() == [[4, 7, 9, 7, eos]]
    restored_tokenizer = AutoTokenizer.from_pretrained(target / "tokenizer", local_files_only=True)
    assert restored_tokenizer.eos_token_id == 3
    assert restored_tokenizer.chat_template == tokenizer.chat_template
    for path in tokenizer_dir.iterdir():
        assert (target / "tokenizer" / path.name).read_bytes() == path.read_bytes()
    assert loaded.generation_config.eos_token_id == [2, 3]


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_plan_rejects_invalid_row_chunk_before_allocating(value):
    with pytest.raises(ValueError, match="row_chunk_size"):
        me.prepare_mixed_export(None, [], metadata={}, row_chunk_size=value)
