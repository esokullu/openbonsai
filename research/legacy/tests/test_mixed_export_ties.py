"""Independent candidate allocation must not silently sever source aliases."""
import pytest
import torch
from torch import nn

from openbonsai.adapter import tiny_model
from openbonsai import mixed_export as me


def native(multimodal=False):
    model = tiny_model(seed=619, multimodal=multimodal).bfloat16()
    prefix = 'model.language_model' if multimodal else 'model'
    return model, prefix + '.embed_tokens', prefix + '.layers.0.mlp.down_proj'


def no_staging(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('Source alias validation must precede temporary directories and writes')
    monkeypatch.setattr(me.tempfile, 'TemporaryDirectory', forbidden)
    monkeypatch.setattr(me, '_write_tensor_file', forbidden)


@pytest.mark.parametrize('multimodal', [False, True])
@pytest.mark.parametrize('selected', ['embedding', 'head', 'both'])
@pytest.mark.parametrize('alias', ['same_parameter', 'distinct_parameter', 'offset_view', 'disjoint_view'])
def test_rejects_selected_endpoint_alias_to_selected_or_retained_tensor(monkeypatch, multimodal, selected, alias):
    model, embedding_name, _ = native(multimodal)
    embedding = model.get_submodule(embedding_name)
    if alias == 'same_parameter':
        model.lm_head.weight = embedding.weight
    elif alias == 'distinct_parameter':
        model.lm_head.weight = nn.Parameter(embedding.weight.detach(), requires_grad=False)
        assert model.lm_head.weight is not embedding.weight
    else:
        rows, width = embedding.weight.shape
        offset = 1 if alias == 'offset_view' else rows
        storage = torch.zeros(rows + offset, width, dtype=torch.bfloat16)
        embedding.weight = nn.Parameter(storage[:rows], requires_grad=False)
        model.lm_head.weight = nn.Parameter(storage[offset:], requires_grad=False)
        assert embedding.weight.data_ptr() != model.lm_head.weight.data_ptr()
    assert (embedding.weight.untyped_storage().data_ptr()
            == model.lm_head.weight.untyped_storage().data_ptr())
    names = {'embedding': [embedding_name], 'head': ['lm_head'],
             'both': [embedding_name, 'lm_head']}[selected]
    no_staging(monkeypatch)
    with pytest.raises(ValueError, match='Tied or aliased.*shares storage.*shared-storage views'):
        me.prepare_mixed_export(model, names, metadata={'source_checkpoint': 'native-fixture'})


def test_rejects_alias_to_nonpersistent_retained_buffer(monkeypatch):
    model, _, projection_name = native()
    weight = model.get_submodule(projection_name).weight
    model.register_buffer('retained_alias', weight[1:], persistent=False)
    assert model.retained_alias.data_ptr() != weight.data_ptr()
    no_staging(monkeypatch)
    with pytest.raises(ValueError, match='Tied or aliased.*retained_alias'):
        me.prepare_mixed_export(model, [projection_name], metadata={'source_checkpoint': 'native-fixture'})


@pytest.mark.parametrize('selected', ['embedding', 'head'])
@pytest.mark.parametrize('config_location', ['text_root', 'multimodal_root', 'multimodal_text'])
def test_declared_endpoint_tying_is_rejected_even_after_manual_untying(monkeypatch, selected, config_location):
    model, embedding_name, _ = native(config_location != 'text_root')
    config = model.config.text_config if config_location == 'multimodal_text' else model.config
    config.tie_word_embeddings = True
    assert (model.get_submodule(embedding_name).weight.untyped_storage().data_ptr()
            != model.lm_head.weight.untyped_storage().data_ptr())
    no_staging(monkeypatch)
    with pytest.raises(ValueError, match='Tied embedding/head configuration'):
        me.prepare_mixed_export(model, [embedding_name if selected == 'embedding' else 'lm_head'],
                                metadata={'source_checkpoint': 'native-fixture'})


@pytest.mark.parametrize('location', ['selected_weight', 'retained_parameter', 'retained_buffer'])
def test_requires_materialized_source_before_any_files(monkeypatch, location):
    model, embedding_name, _ = native()
    if location == 'selected_weight':
        embedding = model.get_submodule(embedding_name)
        embedding.weight = nn.Parameter(torch.empty_like(embedding.weight, device='meta'), requires_grad=False)
    elif location == 'retained_parameter':
        model.lm_head.weight = nn.Parameter(torch.empty_like(model.lm_head.weight, device='meta'), requires_grad=False)
    else:
        model.register_buffer('unmaterialized', torch.empty(0, device='meta'), persistent=False)
    no_staging(monkeypatch)
    with pytest.raises(ValueError, match='materialized source.*meta'):
        me.prepare_mixed_export(model, [embedding_name], metadata={'source_checkpoint': 'native-fixture'})


@pytest.mark.parametrize('multimodal', [False, True])
def test_untied_endpoints_and_empty_views_do_not_produce_false_aliases(tmp_path, multimodal):
    model, embedding_name, projection_name = native(multimodal)
    model.register_buffer('empty_alias', model.get_submodule(embedding_name).weight[:0], persistent=False)
    names = [embedding_name, 'lm_head', projection_name]
    with me.prepare_mixed_export(model, names, metadata={'source_checkpoint': 'native-fixture'},
                                 work_dir=tmp_path) as plan:
        assert set(plan.manifest['modules']) == set(names)
        assert plan.manifest['retained_tensors']['empty_alias']['tensor_bytes'] == 0
        assert int(plan.manifest['storage']['fixed_bytes']) == plan.fixed_bytes
        directory = plan.directory
    assert not directory.exists()


def test_decoder_candidate_can_leave_tied_endpoints_unselected(tmp_path):
    model, embedding_name, projection_name = native()
    model.config.tie_word_embeddings = True
    model.lm_head.weight = model.get_submodule(embedding_name).weight
    with me.prepare_mixed_export(model, [projection_name], metadata={'source_checkpoint': 'native-fixture'},
                                 work_dir=tmp_path) as plan:
        assert set(plan.manifest['modules']) == {projection_name}
        assert embedding_name + '.weight' in plan.manifest['retained_tensors']
        assert 'lm_head.weight' in plan.manifest['retained_tensors']
