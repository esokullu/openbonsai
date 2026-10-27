"""AYOT trace generation: prompt parsing, batched greedy generation, resumable parts."""
import json
import warnings

import pytest
import torch

from openbonsai.stages.traces import generate_batch, read_prompts


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def test_prompt_reader_accepts_prompt_question_and_messages(tmp_path):
    path = write_jsonl(tmp_path / "p.jsonl", [{"prompt": "a"}, {"question": "b"},
                                              {"messages": [{"role": "system", "content": "s"},
                                                            {"role": "user", "content": "c"}]}])
    prompts = read_prompts(path)
    assert prompts[0] == [{"role": "user", "content": "a"}] and prompts[1][0]["content"] == "b"
    assert prompts[2][1]["content"] == "c" and len(read_prompts(path, 1)) == 1


@pytest.mark.parametrize("row,match", [({"prompt": "a", "question": "b"}, "conflicting"), ({"prompt": " "}, "nonempty"),
                                       ({"messages": []}, "messages"), (["x"], "object")])
def test_prompt_reader_rejects_bad_records(tmp_path, row, match):
    with pytest.raises(ValueError, match=match):
        read_prompts(write_jsonl(tmp_path / "p.jsonl", [row]))


def test_batched_generation_keeps_prompt_and_stops_at_eos(tiny_dir):
    from openbonsai.adapters import load_adapter
    adapter = load_adapter(tiny_dir)
    model, tok = adapter.load(tiny_dir, device="cpu", dtype=torch.bfloat16)
    eos = adapter.eos_ids(tok, model)
    prompts = [[{"role": "user", "content": "Short."}], [{"role": "user", "content": "A rather longer question?"}]]
    recs = generate_batch(model, tok, prompts, chat_kwargs=adapter.chat_kwargs(thinking=False), eos_ids=eos,
                          max_new_tokens=12, device=torch.device("cpu"))
    for msgs, rec in zip(prompts, recs):
        prompt_ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=False,
                                             enable_thinking=False)
        assert rec["input_ids"][:rec["prompt_token_count"]] == prompt_ids
        gen = rec["input_ids"][rec["prompt_token_count"]:]
        assert len(gen) == rec["generated_token_count"] <= 12 and rec["attention_mask"] == [1] * len(rec["input_ids"])
        assert rec["hit_generation_limit"] == (gen[-1] not in eos)
        assert not any(t in eos for t in gen[:-1])
    # The longest prompt has no padding, so it matches unbatched greedy decoding.
    alone = generate_batch(model, tok, prompts[1:], chat_kwargs=adapter.chat_kwargs(thinking=False), eos_ids=eos,
                           max_new_tokens=12, device=torch.device("cpu"))
    assert alone[0]["input_ids"] == recs[1]["input_ids"]


def test_trace_stage_resumes_by_batch(make_ctx, tmp_path):
    from openbonsai.cli import execute
    ctx = make_ctx(tmp_path / "run", "traces.batch=1")
    assert execute(ctx, "traces") == "ran"
    d = ctx.run.stage_dir("traces")
    parts = sorted((d / "parts").glob("*.jsonl"))
    assert len(parts) == 2
    first = (d / "traces.jsonl").read_bytes()
    # Interrupted after batch 0: the rerun resumes and regenerates only batch 1.
    parts[1].unlink()
    ctx.run.clear_done("traces")
    stamp = parts[0].stat().st_mtime_ns
    assert execute(ctx, "traces") == "ran"
    assert (d / "traces.jsonl").read_bytes() == first and parts[0].stat().st_mtime_ns == stamp
    # --force regenerates every batch.
    msgs = []
    log = ctx.log
    ctx.log = lambda stage: (lambda m: (msgs.append(m), log(stage)(m)))
    assert execute(ctx, "traces", force=True) == "ran"
    assert sum(m.startswith("batch ") for m in msgs) == 2 and (d / "traces.jsonl").read_bytes() == first
    summary = json.loads((d / "summary.json").read_text())
    assert summary["records"] == 2 and summary["max_new_tokens"] == 24


@pytest.mark.parametrize("multimodal", [False, True])
@pytest.mark.parametrize("attn", ["sdpa", "eager"])
def test_left_padded_rows_match_unbatched_generation_in_fp32(tmp_path, multimodal, attn):
    """Batching pads short prompts on the left; in FP32 the padded row decodes as if alone."""
    from openbonsai.adapters import load_adapter
    from tiny_models import save_tiny_qwen3_5
    d = save_tiny_qwen3_5(tmp_path / "tiny", multimodal=multimodal)
    adapter = load_adapter(d)
    model, tok = adapter.load(d, device="cpu", dtype=torch.float32, attn=attn)
    prompts = [[{"role": "user", "content": "Hi."}],
               [{"role": "user", "content": "A considerably longer question about many different things?"}]]
    kw = dict(chat_kwargs=adapter.chat_kwargs(thinking=False), eos_ids=[0], max_new_tokens=16,
              device=torch.device("cpu"))
    batched = generate_batch(model, tok, prompts, **kw)
    alone = generate_batch(model, tok, prompts[:1], **kw)
    assert batched[0]["input_ids"] == alone[0]["input_ids"]
    ids = [tok.apply_chat_template(p, tokenize=True, add_generation_prompt=True, return_dict=False,
                                   **adapter.chat_kwargs(thinking=False)) for p in prompts]
    width = max(map(len, ids))
    x = torch.tensor([[tok.pad_token_id] * (width - len(i)) + i for i in ids])
    m = torch.tensor([[0] * (width - len(i)) + [1] * len(i) for i in ids])
    with torch.no_grad():
        padded = model(input_ids=x, attention_mask=m, use_cache=False).logits[0, -1]
        single = model(input_ids=torch.tensor([ids[0]]), use_cache=False).logits[0, -1]
    assert width > len(ids[0]) and float((padded - single).abs().max()) < 1e-4


def test_greedy_generation_ignores_the_checkpoints_generation_config(tmp_path):
    """generate() fills unset options from model.generation_config (a checkpoint's generation_config.json:
    Qwen2.5-Instruct ships repetition_penalty 1.05); greedy decoding must be plain argmax for every model."""
    from openbonsai.adapters import load_adapter
    from openbonsai.stages.traces import generate_encoded, greedy_config
    from tiny_models import save_tiny_qwen3_5
    d = save_tiny_qwen3_5(tmp_path / "tiny")
    adapter = load_adapter(d)
    model, tok = adapter.load(d, device="cpu", dtype=torch.float32)
    rows = [tok.apply_chat_template([{"role": "user", "content": c}], tokenize=True, add_generation_prompt=True,
                                    return_dict=False, enable_thinking=False) for c in ("Hi.", "Count to ten.")]
    kw = dict(eos_ids=[10_000], max_new_tokens=24, device=torch.device("cpu"), pad_id=tok.pad_token_id)
    plain = generate_encoded(model, rows, **kw)
    # The argmax tokens, step by step, without generate().
    ids = torch.tensor([rows[1]])
    with torch.no_grad():
        for _ in range(24):
            ids = torch.cat([ids, model(input_ids=ids, use_cache=False).logits[:, -1:].argmax(-1)], dim=1)
    assert plain[1][0] == ids[0, len(rows[1]):].tolist()
    saved = model.generation_config
    for key, value in {"repetition_penalty": 1.3, "no_repeat_ngram_size": 2, "min_length": 200,
                       "suppress_tokens": [plain[1][0][0]], "bad_words_ids": [[plain[1][0][1]]]}.items():
        setattr(model.generation_config, key, value)
    # The settings do change what generate() makes when it is left to them ...
    width = max(map(len, rows))
    x = torch.tensor([[tok.pad_token_id] * (width - len(r)) + r for r in rows])
    m = torch.tensor([[0] * (width - len(r)) + [1] * len(r) for r in rows])
    with torch.no_grad(), warnings.catch_warnings():
        warnings.simplefilter("ignore")      # min_length above the cap, on purpose
        styled = model.generate(input_ids=x, attention_mask=m, max_new_tokens=24, do_sample=False,
                                eos_token_id=[10_000], pad_token_id=tok.pad_token_id)
    assert styled[1, width:].tolist() != plain[1][0]
    # ... but not greedy generation, and the model keeps its own config.
    assert generate_encoded(model, rows, **kw) == plain
    assert model.generation_config is saved and model.generation_config.repetition_penalty == 1.3
    config = greedy_config(max_new_tokens=5, eos_ids=[1, 2], pad_id=0)
    assert (config.do_sample, config.num_beams, config.eos_token_id) == (False, 1, [1, 2])


def test_chat_templates_see_a_fixed_date():
    """Llama 3.2's template prints strftime_now("%d %b %Y") unless date_string is given; prompts (and the
    generation cache) would change every day."""
    from openbonsai.adapters.base import CHAT_DATE, template_vars
    from openbonsai.stages.calibrate import chat_text
    from openbonsai.stages.traces import encode_chat
    from tiny_models import byte_tokenizer
    tok = byte_tokenizer()
    msgs = [{"role": "user", "content": "Hi"}]
    for template in ("{%- if not date_string is defined %}{%- if strftime_now is defined %}"
                     "{%- set date_string = strftime_now('%d %b %Y') %}{%- else %}"
                     "{%- set date_string = '26 Jul 2024' %}{%- endif %}{%- endif %}"
                     "Today Date: {{ date_string }}|{{ messages[0]['content'] }}",
                     "{%- set today = strftime_now('%Y-%m-%d') %}Date {{ today }}|{{ messages[0]['content'] }}"):
        tok.chat_template = template
        text = tok.decode(encode_chat(tok, msgs, {}))
        assert text in ("Today Date: 26 Jul 2024|Hi", "Date 2024-07-26|Hi"), text
        assert chat_text(tok, msgs, {}) == text
    assert CHAT_DATE == (2024, 7, 26) and template_vars({"enable_thinking": True})["enable_thinking"] is True


def test_a_thinking_mode_the_template_cannot_do_fails_before_loading(tmp_path):
    from openbonsai.adapters import adapter_for_config
    from openbonsai.config import ConfigError, load_config
    from openbonsai.stages.evaluate import check_thinking_modes
    from openbonsai.stages.traces import checked_chat_kwargs
    from tiny_models import tiny_dense_config
    llama = adapter_for_config(tiny_dense_config("llama"))
    with pytest.raises(ConfigError, match="traces.thinking"):
        checked_chat_kwargs(llama, True, "traces.thinking")
    assert checked_chat_kwargs(llama, False, "traces.thinking") == {}
    with pytest.raises(ConfigError, match="eval.reasoning.thinking on"):
        check_thinking_modes(load_config(None, ["eval.reasoning.thinking=[off,on]"]), llama)
    assert check_thinking_modes(load_config(None, ["eval.reasoning.thinking=off"]), llama) == ["off"]
    with pytest.raises(ConfigError, match="maybe"):
        check_thinking_modes(load_config(None, ["eval.reasoning.thinking=maybe"]), llama)
