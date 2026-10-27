"""The package and its tests stand alone: nothing imports research/."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IMPORT = re.compile(r"^\s*(from\s+research(\.\w+)*\s+import\b|import\s+research(\.\w+)*\s*(,|as\b|$))"
                    r"|sys\.path\.(insert|append)\([^)]*research", re.M)


def test_the_import_pattern():
    for line in ("from research.experiments import x", "import research", "import research.catq as c",
                 "sys.path.insert(0, str(ROOT / 'research'))"):
        assert IMPORT.search(line), line
    for line in ("import research/.", "x = 'research'", "# from research import nothing"):
        assert not IMPORT.search(line), line


def test_package_tests_and_tools_never_import_research():
    files = [*ROOT.joinpath("openbonsai").rglob("*.py"), *ROOT.joinpath("tests").rglob("*.py"),
             *ROOT.joinpath("tools").rglob("*.py")]
    offenders = [str(f.relative_to(ROOT)) for f in files
                 if f.name != "test_layout.py" and IMPORT.search(f.read_text(encoding="utf-8"))]
    assert len(files) > 20 and offenders == []


# A Windows drive path (D:/ or C:\) or an .exe name; in decoded Python strings also a
# backslash-separated path (in raw text files a backslash is usually an escape, as in JSON).
DRIVE_OR_EXE = re.compile(r"(?<![A-Za-z0-9])[A-Za-z]:[\\/]|\.exe\b", re.I)
BACKSLASH_PATH = re.compile(r"[A-Za-z0-9_.-]\\[A-Za-z0-9_][A-Za-z0-9_.-]+\\")   # x\dir\ (not regex \.\.)


def windows_path(text: str, decoded: bool) -> bool:
    return bool(DRIVE_OR_EXE.search(text) or (decoded and BACKSLASH_PATH.search(text)))


def test_the_windows_path_patterns():
    for text in ("D:/openbonsai/runs", r"C:\Users\x", "Scripts\\python\\", "python.exe"):
        assert windows_path(text, decoded=True), text
    for text in ("https://github.com/x", "runs/x.gguf", "line\nnext\t", "a.execute", "${VENV:-/x}"):
        assert not windows_path(text, decoded=True), text
    assert not windows_path(r'"a\nb\nc"', decoded=False) and not windows_path(r"^(.*)\.\.\.<(\d+)", decoded=True)


def test_package_examples_and_scripts_have_no_windows_paths():
    """The package runs on Linux: no drive letters, backslash paths or .exe names in its
    string literals (Python comments do not count), assets, examples or setup scripts."""
    import ast
    offenders = []
    for f in sorted(ROOT.joinpath("openbonsai").rglob("*.py")):
        for node in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and windows_path(node.value, True):
                offenders.append(f"{f.relative_to(ROOT)}:{node.lineno}")
    texts = [p for p in ROOT.joinpath("openbonsai").rglob("*") if p.suffix in (".json", ".jsonl", ".md", ".yaml")]
    texts += [p for d in ("examples", "scripts") for p in ROOT.joinpath(d).rglob("*") if p.is_file()]
    for f in texts:
        text = f.read_text(encoding="utf-8")
        if f.suffix == ".json":
            # Scan the decoded strings: in raw JSON, Python code such as "if c in d:\n" (the
            # HumanEval manifest) reads as the drive path d:\ because of the \n escape.
            text = "\n".join(_json_strings(json.loads(text)))
        if windows_path(text, decoded=False):
            offenders.append(str(f.relative_to(ROOT)))
    assert any(f.name == "setup_linux.sh" for f in texts) and offenders == []


def _json_strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for k, v in value.items():
            yield k
            yield from _json_strings(v)
    elif isinstance(value, list):
        for v in value:
            yield from _json_strings(v)


def test_json_assets_are_scanned_as_decoded_strings():
    code = json.dumps({"canonical_solution": "    for c in s:\n        if c in d:\n            pass\n"})
    assert windows_path(code, decoded=False)                    # the raw-text false positive
    assert not windows_path("\n".join(_json_strings(json.loads(code))), decoded=False)
    assert windows_path("\n".join(_json_strings(json.loads(json.dumps({"p": "D:\\runs\\x"})))), decoded=False)
