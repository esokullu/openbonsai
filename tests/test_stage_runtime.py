"""Phase accounting and after-call rejection, without CUDA claims."""
import json
import time

import pytest

from openbonsai.stage_runtime import StageMeter, check_deadline, write_json


def test_cpu_meter_records_success_and_failure_and_preserves_outputs(tmp_path):
    meter = StageMeter(tmp_path / "phases", "cpu", time.monotonic() + 30, 0)
    assert meter("small-work", lambda: 3) == 3
    def fail():
        raise RuntimeError("measured failure")
    with pytest.raises(RuntimeError, match="measured failure"):
        meter("bad-work", fail)
    rows = [json.loads(p.read_bytes()) for p in sorted((tmp_path / "phases").glob("*.json"))]
    assert [r["status"] for r in rows] == ["complete", "failed"]
    assert all(r["seconds"] >= 0 for r in rows)
    assert rows[0]["minimum_free_cuda_bytes"] is None
    with pytest.raises(FileExistsError):
        StageMeter(tmp_path / "phases", "cpu", time.monotonic() + 30, 0)


def test_post_call_deadline_can_reject_an_already_written_payload(tmp_path):
    meter = StageMeter(tmp_path / "phases", "cpu", time.monotonic() + 30, 0)
    def publish_then_exhaust():
        write_json(tmp_path / "payload.json", {"status": "complete"})
        meter.deadline = time.monotonic() - 1
    with pytest.raises(TimeoutError):
        meter("publish", publish_then_exhaust)
    assert (tmp_path / "payload.json").is_file()
    assert json.loads((tmp_path / "phases" / "000001.json").read_bytes())["status"] == "failed"


@pytest.mark.parametrize("deadline", [float("nan"), float("inf"), -1.])
def test_invalid_deadline(deadline):
    with pytest.raises(TimeoutError):
        check_deadline(deadline)


def test_new_json_does_not_overwrite_and_replace_is_owned_explicit(tmp_path):
    path = tmp_path / "receipt.json"
    write_json(path, {"a": 1})
    with pytest.raises(FileExistsError):
        write_json(path, {"a": 2})
    write_json(path, {"a": 3}, replace=True)
    assert json.loads(path.read_bytes()) == {"a": 3}


@pytest.mark.parametrize("margin", [-1, True, 1.5])
def test_invalid_margin_rejected_before_output(tmp_path, margin):
    with pytest.raises(ValueError):
        StageMeter(tmp_path / "phases", "cpu", time.monotonic() + 30, margin)
    assert not (tmp_path / "phases").exists()


def test_cuda_margin_rejected_without_device_access(tmp_path):
    with pytest.raises(ValueError, match="1 GiB"):
        StageMeter(tmp_path / "phases", "cuda:0", time.monotonic() + 30, 0)
    assert not (tmp_path / "phases").exists()
