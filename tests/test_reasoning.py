"""GSM8K/HumanEval reasoning suites: answer extraction, the HumanEval executor, frozen manifests,
the test-data guard, dense generation weights and a tiny-model end-to-end run."""
import hashlib
import json
import os
from pathlib import Path

import pytest
import torch

from openbonsai.eval import heldout as H
from openbonsai.eval import reasoning as R
from openbonsai.eval import scoring as SC

ROOT = Path(__file__).resolve().parents[1]


# -- GSM8K answers ------------------------------------------------------------------

@pytest.mark.parametrize("text, value, method", [
    (r"so the total is \boxed{18}.", "18", "boxed"),
    (r"$\boxed{\$1,080}$", "1080", "boxed"),
    (r"\boxed{1{,}234.50}", "1234.5", "boxed"),
    (r"\boxed{\frac{3}{4}}", "0.75", "boxed"),
    (r"\boxed{\dfrac{12}{4}}", "3", "boxed"),
    (r"\boxed{18 \text{ eggs}}", "18", "boxed"),
    (r"\boxed{5 + 3 = 8}", "8", "boxed"),
    (r"I will put it in \boxed{} ... \boxed{42} and later \boxed{}", "42", "boxed"),
    (r"\boxed{-3}", "-3", "boxed"),
    (r"First \boxed{7}, corrected: \boxed{9}", "9", "boxed"),
    ("She makes 9 * 2 = $18 every day.\n#### 18", "18", "####"),
    ("#### 1,000.00", "1000", "####"),
    ("#### +5", "5", "####"),
    ("The answer is 18.", "18", "answer is"),
    ("The final answer is: $1,234", "1234", "answer is"),
    ("Final Answer: -7 degrees", "-7", "answer is"),
    ("So the answer is −5.", "-5", "answer is"),
    ("The answer is 12 apples, not 13.", "12", "answer is"),
    ("Step 1: 5 eggs. Step 2: 10 eggs. So 20 eggs in total.", "20", "last number"),
    ("x = 5-3 = 2", "2", "last number"),
    ("between 1,2,3 and 4", "4", "last number"),
    ("It costs 18.00 dollars", "18", "last number"),
    ("no digits at all", None, "none"),
    ("", None, "none"),
])
def test_gsm8k_answer_extraction(text, value, method):
    assert SC.extract_answer(text) == (value, method)


def test_marker_precedence_boxed_over_hash_over_answer_is_over_last_number():
    assert SC.extract_answer(r"The answer is 3. #### 4 \boxed{5} then 6") == ("5", "boxed")
    assert SC.extract_answer("The answer is 3. #### 4 then 6") == ("4", "####")
    assert SC.extract_answer("The answer is 3, then 6") == ("3", "answer is")


@pytest.mark.parametrize("raw, norm", [("18", "18"), ("18.0", "18"), ("18.", "18"), ("1,080", "1080"),
                                       ("$1,080.00", "1080"), ("+5", "5"), ("-0", "0"), ("-0.50", "-0.5"),
                                       ("−12", "-12"), ("007", "7"), ("abc", None), ("", None),
                                       ("inf", None), ("nan", None)])
def test_normalize_number(raw, norm):
    assert SC.normalize_number(raw) == norm


def test_gsm8k_reference_parsing():
    assert SC.gsm8k_reference("He sells 9 * 2 = <<9*2=18>>18.\n#### 1,018") == "1018"
    with pytest.raises(ValueError):
        SC.gsm8k_reference("no marker")


# -- HumanEval code extraction and execution ------------------------------------------

PROMPT = 'from typing import List\n\n\ndef add(xs: List[int]) -> int:\n    """Sum the list."""\n'
TEST = "def check(candidate):\n    assert candidate([1, 2, 3]) == 6\n    assert candidate([]) == 0\n"


def program_for(completion):
    code, how = SC.extract_code(completion, "add")
    program, mode = SC.build_program(PROMPT, code, TEST, "add")
    return program, how, mode


@pytest.mark.parametrize("completion, how, mode", [
    ("Here it is:\n```python\ndef add(xs):\n    return sum(xs)\n```\nDone.", "fenced", "function"),
    ("```python\ndef add(xs):\n    return sum(xs)\n\n\nif __name__ == '__main__':\n    print(input())\n```",
     "fenced", "function"),
    ("```py\ndef add(xs):\n    total = 0\n    for x in xs:\n        total += x\n    return total", "fenced", "function"),
    ("```\n    return sum(xs)\n```", "fenced", "continuation"),
    ("```python\nreturn sum(xs)\n```", "fenced", "body"),
    ("    return sum(xs)\n", "raw", "continuation"),
    ("Sure, here you go:\nfrom typing import List\ndef add(xs: List[int]) -> int:\n    return sum(xs)\n", "raw",
     "function"),
    ("```python\ndef add(xs):\n    return 0\n```\nBetter:\n```python\ndef add(xs):\n    return sum(xs)\n```",
     "fenced", "function"),
    ("```python\nimport math\n\ndef helper(xs):\n    return math.fsum(xs)\n\ndef add(xs):\n    return int(helper(xs))\n```",
     "fenced", "function"),
])
def test_code_extraction_and_passing_programs(completion, how, mode):
    program, got_how, got_mode = program_for(completion)
    assert (got_how, got_mode) == (how, mode)
    assert program.startswith(PROMPT) and program.rstrip().endswith("check(add)")
    assert "__main__" not in program
    assert SC.run_program(program, timeout=10)["status"] == "passed"


def test_executor_pass_fail_timeout_and_early_exit():
    assert SC.run_program("def f(x):\n    return x + 1\n\nassert f(1) == 2\n")["status"] == "passed"
    failed = SC.run_program("def f(x):\n    return x\n\nassert f(1) == 2, 'wrong'\n")
    assert failed["status"] == "failed" and failed["returncode"] != 0 and "AssertionError" in failed["stderr"]
    assert SC.run_program("def f(:\n")["status"] == "failed"                       # syntax error
    hang = SC.run_program("while True:\n    pass\n", timeout=1.0)
    assert hang["status"] == "timeout" and hang["seconds"] < 8
    # An exit before the test finishes is not a pass, even with exit code 0.
    assert SC.run_program("import sys\nsys.exit(0)\nassert False\n")["status"] == "failed"
    assert SC.run_program("import os\nos._exit(0)\n")["status"] == "failed"
    assert SC.run_program("x = input()\n")["status"] == "failed"                  # stdin is closed


def test_executor_runs_in_a_separate_isolated_process(tmp_path):
    probe = (f"import os, sys\n"
             f"assert os.getpid() != {os.getpid()}\n"
             f"assert os.path.abspath(os.getcwd()) != {str(Path.cwd().resolve())!r}\n"
             f"assert sorted(os.listdir('.')) == ['program.py']\n"
             f"assert sys.flags.isolated == 1\n"
             f"assert os.environ.get('CUDA_VISIBLE_DEVICES') == '-1'\n"
             f"open('scratch.txt', 'w').write('x')\n")
    assert SC.run_program(probe)["status"] == "passed"
    assert not (Path.cwd() / "scratch.txt").exists()
    results = SC.run_programs(["assert True\n", "assert False\n", "assert True\n"], workers=3)
    assert [r["status"] for r in results] == ["passed", "failed", "passed"]


def test_humaneval_canonical_solutions_pass_and_wrong_ones_fail():
    _, problems = R.load_problems("humaneval", 4)
    for p in problems:
        program, mode = SC.build_program(p["prompt"], p["canonical_solution"], p["test"], p["entry_point"])
        assert mode == "continuation"
        whole = p["prompt"] + p["canonical_solution"]
        fenced = f"Sure.\n```python\n{whole}```\n"
        code, how = SC.extract_code(fenced, p["entry_point"])
        program2, mode2 = SC.build_program(p["prompt"], code, p["test"], p["entry_point"])
        assert (how, mode2) == ("fenced", "function")
        wrong, _ = SC.build_program(p["prompt"], "    return None\n", p["test"], p["entry_point"])
        assert [r["status"] for r in SC.run_programs([program, program2, wrong], workers=3)] == \
            ["passed", "passed", "failed"], p["id"]


# -- frozen manifests -----------------------------------------------------------------

@pytest.mark.parametrize("name, n", [("gsm8k", 100), ("humaneval", 50)])
def test_frozen_manifests_are_pinned_and_follow_the_selector(name, n):
    f = R.FROZEN[name]
    data = (R.ASSETS / f["file"]).read_bytes()
    assert hashlib.sha256(data).hexdigest() == f["sha256"] and f["n"] == n
    m = R.load_frozen(name)
    assert R.manifest_bytes(m) == data                       # canonical serialization round-trips
    assert m["content_sha256"] == R.content_sha256(m)
    assert m["row_ids"] == R.select_rows(name, n) == [p["row"] for p in m["problems"]]
    assert len(set(m["row_ids"])) == n
    assert m["selector"] == {"algorithm": "sha256-rank/1", "rule": R.SELECTOR_RULE, "seed": R.SEED, "n": n}
    d = R.DATASETS[name]
    assert m["dataset"] == {k: d[k] for k in ("repo", "config", "split", "revision", "file", "rows", "sha256")}
    # A smaller n is the selector's own choice for that n: a prefix of the frozen order.
    assert R.select_rows(name, 10) == m["row_ids"][:10]
    info, first = R.load_problems(name, 7)
    assert [p["row"] for p in first] == m["row_ids"][:7] and info["frozen"] and info["n"] == 7
    if name == "gsm8k":
        assert all(SC.normalize_number(p["answer"]) == p["answer"] for p in m["problems"])
        assert all(p["id"] == f"gsm8k/test/{p['row']}" for p in m["problems"])
    else:
        assert all(p["id"] == f"HumanEval/{p['row']}" for p in m["problems"])
        assert all(f"def {p['entry_point']}(" in p["prompt"] and "def check(" in p["test"] for p in m["problems"])


def test_manifest_building_is_deterministic_and_tamper_evident():
    rows = [{"question": f"Q{i}: how many?", "answer": f"steps\n#### {i * 3}"} for i in range(1319)]
    a = R.manifest_bytes(R.build_manifest("gsm8k", 12, rows=rows))
    b = R.manifest_bytes(R.build_manifest("gsm8k", 12, rows=list(rows)))
    assert a == b
    m = json.loads(a)
    assert m["row_ids"] == R.select_rows("gsm8k", 12) and m["problems"][0]["answer"] == str(m["row_ids"][0] * 3)
    other_seed = R.build_manifest("gsm8k", 12, rows=rows, seed=R.SEED + 1)
    assert other_seed["row_ids"] != m["row_ids"]
    m["problems"][0]["answer"] = "999999"
    with pytest.raises(ValueError, match="content SHA256"):
        R.verify_manifest(m)
    with pytest.raises(ValueError, match="rows"):
        R.build_manifest("gsm8k", 12, rows=rows[:-1])


def test_frozen_manifests_rebuild_byte_for_byte_from_the_cached_datasets():
    """Needs both pinned parquet files in the local HF cache (HF_HOME); skipped otherwise."""
    try:
        paths = {name: R.dataset_file(name, local_files_only=True) for name in R.SUITES}
    except Exception as error:   # not in this machine's HF cache
        pytest.skip(f"dataset not cached: {type(error).__name__}")
    result = R.check_frozen(paths=paths)
    assert all(r["identical"] and r["sha256"] == r["pinned"] for r in result.values()), result


def test_problem_counts_and_thinking_modes():
    with pytest.raises(ValueError):
        R.load_problems("gsm8k", 0)
    with pytest.raises(FileNotFoundError):   # beyond the frozen 50: needs the dataset
        R.load_problems("humaneval", 51, path=ROOT / "missing.parquet")
    assert R.thinking_modes([False, True]) == ["off", "on"]
    assert R.thinking_modes("on") == ["on"] and R.thinking_modes(["off", "off"]) == ["off"]
    with pytest.raises(ValueError):
        R.thinking_modes(["maybe"])
    from openbonsai.config import load_config
    cfg = load_config(None, ["eval.reasoning.thinking=[off,on]", "eval.reasoning.gsm8k.n=20"])
    assert R.thinking_modes(cfg["eval"]["reasoning"]["thinking"]) == ["off", "on"]
    assert R.split_thinking("plan</think>\n\nanswer 4", "</think>", True) == ("\n\nanswer 4", True)
    assert R.split_thinking("still thinking 4", "</think>", True) == ("", False)
    assert R.split_thinking("answer 4", "</think>", False) == ("answer 4", True)
    assert R.split_thinking("answer 4", None, True) == ("answer 4", True)


# -- test problems never reach calibration or trace prompts -----------------------------

def test_guard_finds_test_problems_in_any_whitespace():
    gsm = R.load_frozen("gsm8k")["problems"][3]
    he = R.load_frozen("humaneval")["problems"][5]
    chat = "<|im_start|>user\n" + "  \n".join(gsm["question"].split(" ")) + "<|im_end|>"
    assert H.contained([chat, "clean text"]) == [(0, gsm["id"])]
    assert H.contained(["notes:\n" + he["prompt"] + "    pass\n"]) == [(0, he["id"])]
    with pytest.raises(H.ContaminationError, match="training splits"):
        H.check_texts(["fine", chat], "unit")
    H.check_texts(["The river turned west below the old mill."], "unit")


def test_guard_refuses_test_split_sources(tmp_path, monkeypatch):
    for src in ({"kind": "jsonl", "repo": "openai/gsm8k", "file": "main/test-00000-of-00001.parquet"},
                {"kind": "jsonl", "repo": "openai/openai_humaneval", "file": "openai_humaneval/test-00000-of-00001.parquet"},
                {"kind": "jsonl", "repo": "openai/gsm8k"}):
        with pytest.raises(H.ContaminationError):
            H.check_source(src, "unit")
    H.check_source({"kind": "jsonl", "repo": "openai/gsm8k", "file": "main/train-00000-of-00001.parquet"}, "unit")
    # A local copy of a test parquet is recognized by its bytes.
    copy = tmp_path / "renamed.parquet"
    copy.write_bytes(b"pretend parquet bytes")
    monkeypatch.setitem(R.DATASETS, "gsm8k", {**R.DATASETS["gsm8k"], "bytes": copy.stat().st_size,
                                               "sha256": hashlib.sha256(copy.read_bytes()).hexdigest()})
    with pytest.raises(H.ContaminationError):
        H.check_source({"kind": "jsonl", "path": str(copy)}, "unit", path=copy)


def test_calibration_and_trace_prompts_refuse_test_problems(tiny_dir, make_ctx, tmp_path):
    from openbonsai.adapters import load_adapter
    from openbonsai.stages import calibrate, traces
    tok = load_adapter(tiny_dir).load_tokenizer(tiny_dir)
    he = R.load_frozen("humaneval")["problems"][0]
    gsm = R.load_frozen("gsm8k")["problems"][0]
    data = tmp_path / "data.jsonl"
    rows = [{"text": "Plain calibration text about rivers and mills. " * 4},
            {"messages": [{"role": "user", "content": "Finish this:\n" + he["prompt"]},
                          {"role": "assistant", "content": he["canonical_solution"] + " filler" * 20}]}]
    data.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    clean = tmp_path / "clean.jsonl"
    clean.write_text(json.dumps(rows[0]) + "\n", encoding="utf-8")
    calibrate.build(tok, [{"kind": "jsonl", "path": str(clean), "n": 2}], seq_len=64)
    # The problem spans many 64-token blocks; the source's blocks are checked as one stream.
    n_blocks = len(tok(calibrate.chat_text(tok, rows[1]["messages"], {}), add_special_tokens=False)["input_ids"]) // 64 + 3
    with pytest.raises(H.ContaminationError, match=he["id"].replace("/", ".")):
        calibrate.build(tok, [{"kind": "jsonl", "path": str(data), "n": n_blocks}], seq_len=64)
    with pytest.raises(H.ContaminationError, match="calibration source"):
        calibrate.build(tok, [{"kind": "wikitext2", "n": 1, "repo": "openai/gsm8k",
                               "file": "main/test-00000-of-00001.parquet"}], seq_len=64)
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text(json.dumps({"prompt": "Hi there."}) + "\n" + json.dumps({"prompt": gsm["question"]}) + "\n",
                       encoding="utf-8")
    ctx = make_ctx(tmp_path / "run", f"traces.prompts.path={prompts.as_posix()}")
    with pytest.raises(H.ContaminationError, match="traces.prompts"):
        traces.load_prompts(ctx)


def test_shipped_calibration_and_prompt_data_hold_no_test_problems():
    from openbonsai.stages import calibrate
    assert not {d["repo"] for d in R.DATASETS.values()} & {h["repo"] for h in calibrate.HUB.values()}
    texts = []
    for f in [ROOT / "examples" / "prompts.jsonl", *sorted((ROOT / "examples" / "tiny-data").glob("*.jsonl"))]:
        texts += f.read_text(encoding="utf-8").splitlines()
    assert len(texts) > 5 and H.contained(texts) == []


# -- dense generation weights and the end-to-end run -------------------------------------

def test_densify_writes_the_weights_apply_dense_writes(calibrated):
    from openbonsai.cli import execute
    from openbonsai.core.ternary import TernaryLinear
    from openbonsai.stages.common import apply_dense, densify, has_ternary, install_ternary
    ctx = calibrated
    execute(ctx, "quantize")
    execute(ctx, "endpoints")
    state = ctx.state_through("endpoints")
    adapter = ctx.adapter()
    dense, _ = ctx.load_model()
    apply_dense(adapter, dense, state.records())
    tern, _ = ctx.load_model()
    install_ternary(adapter, tern, state.records(), ctx.device)
    assert has_ternary(tern) and any(isinstance(m, TernaryLinear) for m in tern.modules())
    assert densify(tern, ctx.dtype) == len(state.tensors) and not has_ternary(tern)
    a, b = dense.state_dict(), tern.state_dict()
    assert set(a) == set(b) and all(torch.equal(a[k], b[k]) for k in a)
    ids = torch.tensor([[5, 9, 33, 70, 101]])
    assert torch.equal(dense(input_ids=ids).logits, tern(input_ids=ids).logits)


def test_reasoning_suites_end_to_end_on_the_tiny_model(calibrated):
    from openbonsai.cli import execute
    ctx = calibrated
    for key, value in (("thinking", [False, True]), ("max_new_tokens", 6), ("max_new_tokens_thinking", 5),
                       ("batch", 2)):
        ctx.cfg["eval"]["reasoning"][key] = value
    ctx.cfg["eval"]["reasoning"]["gsm8k"]["n"] = 3
    ctx.cfg["eval"]["reasoning"]["humaneval"]["n"] = 2
    execute(ctx, "quantize")
    msgs = []
    log = ctx.log
    ctx.log = lambda stage: (lambda m: (msgs.append(m), log(stage)(m)))
    execute(ctx, "eval", source="quantize", suites=["humaneval", "gsm8k"], install_mode="ternary")
    assert any("ternary modules replaced" in m for m in msgs)
    assert any("tok/s" in m and "ETA" in m for m in msgs)
    out = ctx.run.stage_dir("eval")
    first = {}
    for suite, n in (("gsm8k", 3), ("humaneval", 2)):
        for mode, cap in (("off", 6), ("on", 5)):
            r = json.loads((out / f"quantize.{suite}.think-{mode}.json").read_text(encoding="utf-8"))
            first[suite, mode] = r
            assert r["source"] == "quantize" and r["weights"] == "dense" and r["thinking"] == mode
            assert r["total"] == n == len(r["problems"]) and r["max_new_tokens"] == cap
            assert r["batches"] == (n + 1) // 2 and r["cached_batches"] == 0
            assert 0 <= r["hit_cap"] <= n and r["generated_tokens"] <= n * cap
            assert r["manifest"]["frozen"] and r["manifest"]["n"] == n
            assert r["chat_kwargs"] == {"enable_thinking": mode == "on"}
            for p in r["problems"]:
                assert 1 <= p["generated_tokens"] <= cap and (not p["hit_cap"] or p["generated_tokens"] == cap)
                assert ("finished_thinking" in p) == (mode == "on")
            if suite == "gsm8k":
                correct = sum(p["correct"] for p in r["problems"])
                assert r["score"] == f"{correct}/{n}" and r["correct"] == correct
                assert sum(r["methods"].values()) == n
            else:
                assert r["passed"] + r["failed"] + r["timeout"] == n and r["score"] == f"{r['passed']}/{n}"
                assert all(p["status"] in ("passed", "failed", "timeout") for p in r["problems"])
            if mode == "on":
                assert r["unfinished_thinking"] == sum(not p["finished_thinking"] for p in r["problems"])
    # A rerun takes every generation from the cache and scores identically.
    execute(ctx, "eval", source="quantize", suites=["gsm8k", "humaneval"], install_mode="ternary")
    for (suite, mode), r in first.items():
        again = json.loads((out / f"quantize.{suite}.think-{mode}.json").read_text(encoding="utf-8"))
        assert again["cached_batches"] == again["batches"] and again["tokens_per_second"] is None
        strip = ("seconds", "returncode")
        assert [{k: v for k, v in p.items() if k not in strip} for p in again["problems"]] == \
            [{k: v for k, v in p.items() if k not in strip} for p in r["problems"]]
