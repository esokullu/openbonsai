"""ModelAdapter registry, plan conformance and derived geometry."""
import pytest
import torch
from transformers import Qwen3_5ForCausalLM

from openbonsai.adapters import LayerPlan, PlanError, Qwen35Adapter, adapter_for_config, load_adapter
from openbonsai.core.geometry import UnsupportedGeometry, derive_geometry
from tiny_models import tiny_qwen3_5, tiny_text_config


def test_registry_picks_qwen3_5_for_text_and_multimodal_configs():
    assert isinstance(adapter_for_config(tiny_text_config()), Qwen35Adapter)
    assert isinstance(adapter_for_config(tiny_qwen3_5(multimodal=True).config), Qwen35Adapter)
    with pytest.raises(ValueError, match="no adapter"):
        adapter_for_config(type("C", (), {"model_type": "gpt2"})())


@pytest.mark.parametrize("multimodal", [False, True])
def test_every_linear_is_planned_or_kept(multimodal):
    model = tiny_qwen3_5(multimodal=multimodal)
    adapter = adapter_for_config(model.config)
    seen = 0
    for i in range(adapter.n_layers):
        plan = adapter.layer_plan(i)
        planned = adapter.planned_linears(model, i)
        assert list(planned) == list(plan.quantized)
        seen += len(planned)
        if plan.kind == "linear_attention":
            assert plan.keep == ("linear_attn.in_proj_a", "linear_attn.in_proj_b")
            assert plan.groups[0] == ("linear_attn.in_proj_qkv", "linear_attn.in_proj_z")
        else:
            assert plan.keep == () and plan.groups[0] == ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj")
    assert seen == 3 * 6 + 7
    assert adapter.text_model(model) is (model.model.language_model if multimodal else model.model)


def test_unplanned_linear_fails_conformance(monkeypatch):
    model = tiny_qwen3_5()
    adapter = adapter_for_config(model.config)
    plan = adapter.layer_plan(0)
    monkeypatch.setattr(adapter, "layer_plan", lambda i: LayerPlan(plan.kind, plan.groups, ()))
    with pytest.raises(PlanError, match="in_proj_a"):
        adapter.planned_linears(model, 0)


def test_embedding_gain_follows_the_zero_centered_norm():
    model = tiny_qwen3_5(random_norms=True)
    adapter = adapter_for_config(model.config)
    w = model.model.layers[0].input_layernorm.weight.float()
    assert torch.equal(adapter.embedding_gain(model), 1.0 + w)
    assert adapter.chat_kwargs(thinking=False) == {"enable_thinking": False}
    assert adapter.bos_id(None) is None and adapter.gguf_arch == "qwen35" and not adapter.tied


def _meta_geometry(**text):
    cfg = tiny_text_config(**text)
    with torch.device("meta"):
        model = Qwen3_5ForCausalLM(cfg)
    adapter = adapter_for_config(cfg)
    return derive_geometry(adapter.quantized_widths(model), adapter.hidden_size), adapter


_SIZES = {
    "9B": (dict(hidden_size=4096, intermediate_size=12288, linear_num_value_heads=32, linear_value_head_dim=128,
                linear_num_key_heads=16, linear_key_head_dim=128, num_attention_heads=16, head_dim=256,
                num_key_value_heads=4), (4096, 12288), 4096, False),
    "2B": (dict(hidden_size=2048, intermediate_size=6144, linear_num_value_heads=16, linear_value_head_dim=128,
                linear_num_key_heads=16, linear_key_head_dim=128, num_attention_heads=8, head_dim=256,
                num_key_value_heads=2, tie_word_embeddings=True), (2048, 6144), 2048, True),
    "4B": (dict(hidden_size=2560, intermediate_size=9216, linear_num_value_heads=32, linear_value_head_dim=128,
                linear_num_key_heads=16, linear_key_head_dim=128, num_attention_heads=16, head_dim=256,
                num_key_value_heads=4, tie_word_embeddings=True), (2560, 4096, 9216), 512, True),
    "27B": (dict(hidden_size=5120, intermediate_size=17408, linear_num_value_heads=48, linear_value_head_dim=128,
                 linear_num_key_heads=16, linear_key_head_dim=128, num_attention_heads=24, head_dim=256,
                 num_key_value_heads=4), (5120, 6144, 17408), 1024, False),
}


@pytest.mark.parametrize("name", list(_SIZES))
def test_geometry_is_derived_from_the_planned_linears(name):
    text, widths, block, tied = _SIZES[name]
    geo, adapter = _meta_geometry(**text)
    assert geo.widths == widths and geo.block == block
    assert adapter.tied is tied


def test_unsupported_hidden_size_is_refused():
    with pytest.raises(UnsupportedGeometry, match="multiples"):
        _meta_geometry(hidden_size=576, intermediate_size=1536, linear_num_value_heads=4, linear_value_head_dim=128,
                       num_attention_heads=4, head_dim=128)


def test_load_text_and_multimodal_checkpoints(tiny_dir, tiny_mm_dir):
    for path, cls in ((tiny_dir, "Qwen3_5ForCausalLM"), (tiny_mm_dir, "Qwen3_5ForConditionalGeneration")):
        adapter = load_adapter(path)
        model, tok = adapter.load(path, device="cpu", dtype=torch.bfloat16)
        assert type(model).__name__ == cls and not model.training
        assert all(not p.requires_grad for p in model.parameters())
        if cls.endswith("ConditionalGeneration"):
            assert model.model.visual is None
        ids = torch.tensor([tok("hello world")["input_ids"]])
        with torch.no_grad():
            logits = model(input_ids=ids, use_cache=False).logits
        assert logits.shape == (1, ids.shape[1], 320) and torch.isfinite(logits).all()
        assert adapter.eos_ids(tok, model) == [tok.convert_tokens_to_ids("<|im_end|>")]
