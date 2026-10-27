"""Scoring for the reasoning suites: GSM8K final-number exact match, HumanEval pass@1.

GSM8K: ``extract_answer`` takes the model's final number, preferring an
explicit marker: the last ``\\boxed{...}`` holding a number, else the number
after the last ``####``, else the number after the last "answer is" /
"answer:", else the last number in the text. ``normalize_number`` drops
thousands separators, ``$``, a ``+`` sign and trailing zeros (``1,080.00`` ->
``1080``, ``-0`` -> ``0``), so the match is on the value, not its spelling.

HumanEval: ``extract_code`` takes the code from fenced blocks (the last one
that defines the entry point, else the first) or the raw text; ``build_program``
prepends the problem's prompt (so a continued function body completes the
stub, and a full redefinition keeps the prompt's imports and helpers) and
appends the official ``test`` and ``check(entry_point)``. ``run_program``
executes it.

Execution is NOT a security sandbox. Every program runs in a separate Python
process (``python -I``: no user site, no PYTHON* variables, script directory
not on sys.path) in a fresh temporary directory, with stdin closed, a reduced
environment (temp and home pointed at that directory, no GPU) and a wall-clock
timeout after which the process (its process group on POSIX) is killed. That
protects the evaluator from crashes, hangs and ``exit()`` in model code, which
never runs in the evaluator's process. It does not stop the code from reading
or writing your files or opening network connections: the suite has no
network intent, and nothing enforces that. Evaluate untrusted models in a
container or VM.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal, InvalidOperation
import os
from pathlib import Path
import re
import secrets
import signal
import subprocess
import sys
import tempfile
import time

# -- GSM8K -----------------------------------------------------------------

# A number not glued to a preceding word, digit or dot; thousands groups of exactly 3 digits.
_NUMBER = re.compile(r"(?<![\w.])[-+\u2212]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")
_FRAC = re.compile(r"\\[dt]?frac\s*\{\s*([^{}]+?)\s*\}\s*\{\s*([^{}]+?)\s*\}")
_ANSWER_IS = re.compile(r"answer\s*(?:is\b|:)", re.I)


def normalize_number(text: str) -> str | None:
    """Canonical string of a number (``"1,080.00"`` -> ``"1080"``, ``"-0.50"`` -> ``"-0.5"``); None if not one."""
    s = str(text).strip().replace(",", "").replace("$", "").replace("\u2212", "-").replace(" ", "")
    s = s.rstrip(".")
    if s.startswith("+"):
        s = s[1:]
    try:
        value = Decimal(s)
    except InvalidOperation:
        return None
    if not value.is_finite():
        return None
    if value == value.to_integral_value():
        return str(int(value))
    return format(value.normalize(), "f")


def _numbers(text: str) -> list[str]:
    return [m.group(0) for m in _NUMBER.finditer(text)]


def _braced(text: str, start: int) -> tuple[str, int] | None:
    """Content of the brace group opening at ``text[start] == '{'`` (nesting allowed), and its end."""
    depth = 0
    for k in range(start, len(text)):
        if text[k] == "{":
            depth += 1
        elif text[k] == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1:k], k + 1
    return None


def boxed_contents(text: str) -> list[str]:
    """Contents of every ``\\boxed{...}`` / ``\\fbox{...}``, in order."""
    out = []
    for m in re.finditer(r"\\(?:boxed|fbox)\s*(?=\{)", text):
        got = _braced(text, m.end())
        if got is not None:
            out.append(got[0])
    return out


def _latex_number(content: str) -> str | None:
    """The value of a boxed answer: a ``\\frac{a}{b}``, else its last number."""
    c = content.replace("{,}", ",").replace("\\,", "").replace("\\!", "").replace("\\$", "$")
    frac = _FRAC.search(c)
    if frac:
        a, b = normalize_number(frac.group(1)), normalize_number(frac.group(2))
        if a is not None and b is not None and Decimal(b) != 0:
            return normalize_number(str(Decimal(a) / Decimal(b)))
    nums = _numbers(c)
    return normalize_number(nums[-1]) if nums else None


def extract_answer(text: str) -> tuple[str | None, str]:
    """(normalized final number or None, method): boxed, ####, answer is, last number, none."""
    for content in reversed(boxed_contents(text)):
        value = _latex_number(content)
        if value is not None:
            return value, "boxed"
    if "####" in text:
        nums = _numbers(text.rsplit("####", 1)[1])
        if nums:
            return normalize_number(nums[0]), "####"
    hits = list(_ANSWER_IS.finditer(text))
    if hits:
        tail = text[hits[-1].end():hits[-1].end() + 200]
        nums = _numbers(tail.replace("\\$", "$"))
        if nums:
            return normalize_number(nums[0]), "answer is"
    nums = _numbers(text)
    if nums:
        return normalize_number(nums[-1]), "last number"
    return None, "none"


def gsm8k_reference(solution: str) -> str:
    """The reference answer of a GSM8K solution: the number after ``####``."""
    if "####" not in solution:
        raise ValueError("GSM8K solution without ####")
    value = normalize_number(solution.rsplit("####", 1)[1])
    if value is None:
        raise ValueError(f"GSM8K reference is not a number: {solution.rsplit('####', 1)[1]!r}")
    return value


# -- HumanEval ---------------------------------------------------------------

_FENCE = re.compile(r"```[ \t]*([A-Za-z0-9_+.-]*)[^\n]*\n(.*?)(?:```|\Z)", re.S)
_CODE_LANGS = {"", "python", "py", "python3", "py3"}
_MAIN = re.compile(r"""^if\s+__name__\s*==\s*['"]__main__['"]\s*:""")


def _defines(code: str, name: str) -> bool:
    return re.search(rf"^def\s+{re.escape(name)}\s*\(", code, re.M) is not None


def strip_main_block(code: str) -> str:
    """Drop a top-level ``if __name__ == "__main__":`` block (example usage that would run, or wait on input)."""
    lines, out, skipping = code.split("\n"), [], False
    for line in lines:
        if _MAIN.match(line):
            skipping = True
            continue
        if skipping and (not line.strip() or line[:1] in (" ", "\t")):
            continue
        skipping = False
        out.append(line)
    return "\n".join(out)


_CODE_START = re.compile(r"^(?:from\s+\S+\s+import\s|import\s|def\s|class\s|async\s+def\s|@)", re.M)


def extract_code(text: str, entry_point: str) -> tuple[str, str]:
    """(code, how): from the last fenced block that defines ``entry_point``, else the first fenced
    Python block, else the raw text (from its first top-level import/def when it defines the entry
    point, so a line of prose before unfenced code is dropped). A fence the generation cap cut off
    still counts."""
    blocks = [m.group(2) for m in _FENCE.finditer(text) if m.group(1).lower() in _CODE_LANGS]
    defining = [b for b in blocks if _defines(b, entry_point)]
    if defining:
        return defining[-1], "fenced"
    if blocks:
        return blocks[0], "fenced"
    start = _CODE_START.search(text) if _defines(text, entry_point) else None
    return (text[start.start():] if start else text), "raw"


def build_program(prompt: str, code: str, test: str, entry_point: str) -> tuple[str, str]:
    """(program, mode). Modes: ``function`` (the code defines the entry point; it follows the prompt,
    so the prompt's imports and helpers stay), ``continuation`` (indented code continues the prompt's
    function body), ``body`` (an unindented body without definitions; indented under the prompt)."""
    code = strip_main_block(code.replace("\r\n", "\n")).rstrip() + "\n"
    head = prompt if prompt.endswith("\n") else prompt + "\n"
    first = next((ln for ln in code.split("\n") if ln.strip()), "")
    if _defines(code, entry_point):
        body, mode = head + "\n" + code, "function"
    elif first[:1] in (" ", "\t"):
        body, mode = head + code, "continuation"
    elif not re.search(r"^(def|class|async\s+def)\s", code, re.M):
        body, mode = head + "".join(("    " + ln if ln.strip() else ln) + "\n" for ln in code.rstrip("\n").split("\n")), "body"
    else:
        body, mode = head + "\n" + code, "module"
    return f"{body}\n\n{test}\n\ncheck({entry_point})\n", mode


def _child_env(tmp: str) -> dict:
    keep = ("PATH", "SYSTEMROOT", "WINDIR", "COMSPEC", "PATHEXT", "LANG", "LC_ALL")
    env = {k: os.environ[k] for k in keep if k in os.environ}
    env.update({"TMP": tmp, "TEMP": tmp, "TMPDIR": tmp, "HOME": tmp, "USERPROFILE": tmp,
                "CUDA_VISIBLE_DEVICES": "-1", "OMP_NUM_THREADS": "1", "PYTHONIOENCODING": "utf-8"})
    return env


def _kill(proc: subprocess.Popen) -> None:
    try:
        if os.name == "posix":
            os.killpg(proc.pid, signal.SIGKILL)
        else:
            proc.kill()
    except (ProcessLookupError, PermissionError, OSError):
        pass


def _tail(text: str, limit: int = 2000) -> str:
    return text if len(text) <= limit else "..." + text[-limit:]


def run_program(source: str, *, timeout: float = 10.0, python: str | None = None) -> dict:
    """Run ``source`` in a separate ``python -I`` process in a fresh temporary directory.

    Returns {"status": "passed" | "failed" | "timeout", "returncode", "seconds", "stderr"}.
    ``passed`` needs exit code 0 AND a random end marker printed after the program's last
    line, so code that exits early (``sys.exit(0)``, ``os._exit``) fails. Not a security
    sandbox (see the module docstring)."""
    marker = secrets.token_hex(16)
    with tempfile.TemporaryDirectory(prefix="openbonsai-exec-", ignore_cleanup_errors=True) as tmp:
        path = Path(tmp) / "program.py"
        path.write_text(f"{source}\n\nprint({marker!r}, flush=True)\n", encoding="utf-8")
        kwargs = {"cwd": tmp, "stdin": subprocess.DEVNULL, "stdout": subprocess.PIPE, "stderr": subprocess.PIPE,
                  "env": _child_env(tmp)}
        if os.name == "posix":
            kwargs["start_new_session"] = True
        else:
            kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        t0 = time.monotonic()
        proc = subprocess.Popen([python or sys.executable, "-I", path.name], **kwargs)   # cwd=tmp
        try:
            out, err = proc.communicate(timeout=timeout)
            timed_out = False
        except subprocess.TimeoutExpired:
            _kill(proc)
            timed_out = True
            try:
                out, err = proc.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                out, err = b"", b""
        seconds = time.monotonic() - t0
        message = err.decode("utf-8", errors="replace").replace("\r\n", "\n")
        for prefix in {str(Path(tmp)), os.path.realpath(tmp)}:   # tracebacks name the absolute script path
            message = message.replace(prefix + os.sep, "")
    if timed_out:
        status = "timeout"
    elif proc.returncode == 0 and marker in out.decode("utf-8", errors="replace"):
        status = "passed"
    else:
        status = "failed"
        if proc.returncode == 0:
            message += "\n[openbonsai] the program exited before the test finished"
    return {"status": status, "returncode": proc.returncode, "seconds": round(seconds, 3), "stderr": _tail(message)}


def run_programs(sources: list[str], *, timeout: float = 10.0, workers: int = 4) -> list[dict]:
    """``run_program`` for each source, ``workers`` processes at a time; results in input order."""
    if not sources:
        return []
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        return list(pool.map(lambda s: run_program(s, timeout=timeout), sources))
