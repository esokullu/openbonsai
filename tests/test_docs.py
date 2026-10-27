"""The README and docs describe the CLI that exists, and carry the launch lines.

Every ``openbonsai <command> ...`` written in a code span or code block of the
README, docs/, research/README.md or the example configs' comments is parsed
against the real argparse parser (built, never run): the command must exist,
every ``--flag`` must belong to it, a value given to an option with choices
must be one of them, ``--suite`` names must be known suites, and every
``--set key=value`` must be accepted by the config schema. Relative links in
the same files must point to existing files and headings.
"""
from __future__ import annotations

import argparse
import re
import shlex
from pathlib import Path

import pytest

from openbonsai.cli import build_parser
from openbonsai.config import ConfigError, load_config

ROOT = Path(__file__).resolve().parents[1]
MARKDOWN = [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md")), ROOT / "research" / "README.md"]
YAML = sorted((ROOT / "examples").glob("*.yaml"))

PREVIEW = "Research preview"
NON_AFFILIATION = "Independent project, not affiliated with or endorsed by Prism ML. Bonsai is Prism ML's product."

FENCE = re.compile(r"^```[^\n]*\n(.*?)^```", re.M | re.S)
INLINE = re.compile(r"`([^`\n]+)`")
COMMAND = re.compile(r"(?<![\w./-])openbonsai[ \t]+(?P<args>[^\n#]*)")
SET_ANYWHERE = re.compile(r"--set[ \t]+('[^']*'|\"[^\"]*\"|\S+)")
LINK = re.compile(r"(?<!!)\[[^\]]*\]\(([^)\s]+)\)")


def _rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _code_texts(path: Path) -> list[str]:
    """The code a file shows: fenced blocks and inline code spans (Markdown), comments (YAML).
    Shell line continuations are joined."""
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".yaml":
        return [line.split("#", 1)[1] for line in text.splitlines() if "#" in line]
    blocks = [re.sub(r"\\\n[ \t]*", " ", b) for b in FENCE.findall(text)]
    prose = FENCE.sub("", text)
    return blocks + INLINE.findall(prose)


def _commands(path: Path) -> list[str]:
    out = []
    for code in _code_texts(path):
        for line in code.splitlines():
            for m in COMMAND.finditer(line):
                out.append(m.group("args").strip())
    return out


def _tokens(args: str) -> list[str]:
    return shlex.split(args.replace("\\", "/"))


def _subparsers(parser: argparse.ArgumentParser) -> dict[str, argparse.ArgumentParser]:
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return dict(action.choices)
    raise AssertionError("the CLI has no subcommands")


def _options(parser: argparse.ArgumentParser) -> dict[str, argparse.Action]:
    return {opt: a for a in parser._actions for opt in a.option_strings}


def _check_set(assignment: str) -> str | None:
    """Why ``--set assignment`` is refused by the config schema (None: accepted). ``<placeholders>``
    become a plain value; an upper-case key (``SECTION.KEY=VALUE``, the CLI's metavar) is a placeholder."""
    if assignment.split("=", 1)[0].isupper():
        return None
    if "<" in assignment:
        assignment = re.sub(r"<[^>]*>", "x", assignment)
    try:
        load_config(None, [assignment])
    except ConfigError as error:
        return str(error)
    return None


def _command_problems(args: str) -> list[str]:
    from openbonsai.eval.suites import SUITES
    parser = build_parser()
    commands = _subparsers(parser)
    tokens = _tokens(args)
    if not tokens:
        return []
    if tokens[0].startswith("-"):
        top = _options(parser)
        return [f"unknown top-level option {t}" for t in tokens if t.startswith("-") and t.split("=")[0] not in top]
    cmd = tokens[0]
    if cmd not in commands:
        return [f"unknown command {cmd!r} (known: {sorted(commands)})"]
    options = _options(commands[cmd])
    problems = []
    rest = tokens[1:]
    for i, tok in enumerate(rest):
        if not tok.startswith("-") or tok == "-":
            continue
        flag, _, inline = tok.partition("=")
        action = options.get(flag)
        if action is None:
            problems.append(f"{cmd}: unknown option {flag}")
            continue
        takes_value = action.nargs != 0 and not isinstance(
            action, (argparse._StoreTrueAction, argparse._StoreFalseAction, argparse._StoreConstAction))
        value = inline or (rest[i + 1] if takes_value and i + 1 < len(rest) else None)
        if not takes_value or value is None or value.startswith("<"):
            continue
        if action.choices is not None and value not in action.choices:
            problems.append(f"{cmd} {flag}: {value!r} is not one of {list(action.choices)}")
        if flag == "--suite":
            unknown = [s for s in value.split(",") if s not in SUITES]
            if unknown:
                problems.append(f"{cmd} --suite: unknown suites {unknown} (known: {list(SUITES)})")
        if flag == "--set":
            error = _check_set(value)
            if error:
                problems.append(f"{cmd} --set {value}: {error}")
    return problems


def _documented_commands() -> list[tuple[str, str]]:
    return [(_rel(path), args) for path in MARKDOWN + YAML for args in _commands(path)]


def test_docs_mention_the_main_commands():
    """The extractor finds commands at all (a broken regex would make the next test pass vacuously)."""
    found = {_tokens(args)[0] for _, args in _documented_commands() if _tokens(args)}
    assert {"run", "eval", "export", "inspect"} <= found, found


@pytest.mark.parametrize("where,args", _documented_commands())
def test_documented_command_exists_in_the_cli(where, args):
    problems = _command_problems(args)
    assert not problems, f"{where}: openbonsai {args}\n" + "\n".join(problems)


def _set_assignments() -> list[tuple[str, str]]:
    out = []
    for path in MARKDOWN + YAML:
        for code in _code_texts(path):
            for m in SET_ANYWHERE.finditer(code):
                value = m.group(1)
                if value[:1] in "'\"":
                    value = value[1:-1]
                if "=" in value:
                    out.append((_rel(path), value))
    return out


@pytest.mark.parametrize("where,assignment", _set_assignments())
def test_documented_set_overrides_are_valid_config(where, assignment):
    error = _check_set(assignment)
    assert error is None, f"{where}: --set {assignment}: {error}"


def test_the_checker_rejects_what_does_not_exist():
    assert _command_problems("run --run runs/x")                    # stages take --run; run takes --out
    assert "unknown command" in _command_problems("frobnicate --run x")[0]
    assert _command_problems("export --run x --suite dev")          # eval's option, not export's
    assert _command_problems("run --out x --until quantise")       # not a stage
    assert _command_problems("eval --run x --suite dev,wiki")      # not a suite
    assert _command_problems("eval --run x --set eval.sutes=[dev]")
    assert not _command_problems("run --out x --until quantize --set 'eval.suites=[compare]'")
    assert not _command_problems("eval --from <file>.gguf --suite gsm8k,humaneval --set eval.reasoning.gsm8k.n=20")


def _slug(heading: str) -> str:
    """GitHub's anchor for a Markdown heading."""
    text = re.sub(r"[`*_]|<[^>]+>", "", heading.strip().lower())
    text = re.sub(r"[^\w\- ]", "", text)
    return text.replace(" ", "-")


def _anchors(path: Path) -> set[str]:
    text = FENCE.sub("", path.read_text(encoding="utf-8"))
    return {_slug(m.group(1)) for m in re.finditer(r"^#{1,6}[ \t]+(.+?)[ \t]*#*$", text, re.M)}


def _links() -> list[tuple[str, str]]:
    out = []
    for path in MARKDOWN:
        text = FENCE.sub("", path.read_text(encoding="utf-8"))
        for target in LINK.findall(text):
            if not re.match(r"^[a-z][a-z0-9+.-]*:", target):   # skip http:, https:, mailto:
                out.append((_rel(path), target))
    return out


@pytest.mark.parametrize("where,target", _links())
def test_relative_links_resolve(where, target):
    file_part, _, anchor = target.partition("#")
    source = ROOT / where
    dest = (source.parent / file_part).resolve() if file_part else source
    assert dest.exists(), f"{where}: link to missing {target}"
    if anchor and dest.suffix == ".md":
        assert anchor in _anchors(dest), f"{where}: no heading for #{anchor} in {_rel(dest)}"


def test_readme_carries_the_research_preview_and_non_affiliation_lines():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert PREVIEW in readme
    assert NON_AFFILIATION in readme


def test_website_carries_the_same_lines_and_the_relabeled_video():
    page = (ROOT / "web" / "index.html").read_text(encoding="utf-8")
    assert PREVIEW in page
    assert NON_AFFILIATION in page
    assert "Background: how ScaleQ works" in page


# Results of the experiment that started from Prism's released weights (dev NLL 2.706 with distilled
# scales, 2.713 with LoRA). They are not from-scratch results, so they may appear only under a heading
# or bold label that names that experiment, and never in docs/ or on the website.
BONSAI_START = re.compile(r"(?<![\d.])2\.(?:706|713)(?!\d)")
BONSAI_START_LABEL = "Prism's weights"


def _label_above(lines: list[str], i: int) -> str | None:
    """The nearest heading (``#``) or bold label line (``**...**``) at or above line ``i``."""
    for line in reversed(lines[:i + 1]):
        s = line.strip()
        if s.startswith("#") or (s.startswith("**") and s.rstrip(":").endswith("**")):
            return s
    return None


def test_bonsai_start_numbers_stay_in_their_own_attributed_section():
    for path in MARKDOWN + [ROOT / "web" / "index.html", ROOT / "FRAMEWORK_PLAN.md"]:
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines()
        for i, line in enumerate(lines):
            if not BONSAI_START.search(line):
                continue
            where = f"{_rel(path)}:{i + 1}"
            assert path.name == "README.md", f"{where}: a Prism-start result outside the README files"
            label = _label_above(lines, i)
            assert label and BONSAI_START_LABEL in label, f"{where}: a Prism-start result under {label!r}"


def test_the_label_check_finds_the_section():
    lines = ["## Results", "dev 2.892", "### Starting from Prism's weights (separate experiment)", "2.706"]
    assert _label_above(lines, 1) == "## Results"
    assert BONSAI_START_LABEL in _label_above(lines, 3)
    assert BONSAI_START.search("went to 2.706 and") and not BONSAI_START.search("12.7060")
