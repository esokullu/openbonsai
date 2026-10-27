"""The package and its tests stand alone: nothing imports research/."""
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
