"""AYOT trace generation: prompt parsing, batched greedy generation, resumable parts."""
import json

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
    parts[1].unlink()
    assert execute(ctx, "traces", force=True) == "ran"
    assert (d / "traces.jsonl").read_bytes() == first
    summary = json.loads((d / "summary.json").read_text())
    assert summary["records"] == 2 and summary["max_new_tokens"] == 24
