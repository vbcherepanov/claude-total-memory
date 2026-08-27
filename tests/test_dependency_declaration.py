"""`requirements.txt` and `pyproject.toml` must not drift apart.

The project ships through two independent paths and they resolve different
dependency lists:

    install.sh / Docker  ->  requirements.txt
    pip / uvx / npx / brew / the wheel  ->  pyproject [project.dependencies]

Anything listed only in `requirements.txt` is missing from every packaged
install — and because the code degrades instead of crashing, nobody finds out.
This has now happened four times: `migrations/`, `vocabularies/`, `filters/`
(all fixed by packaging them) and `fastembed`.

`fastembed` is the one that changed behaviour rather than crashing. It is the
*default* embedding backend, so without it the server silently falls back to
sentence-transformers with a different, English-only model — the two install
paths retrieve differently. (Measured: the fallback is actually *lighter*,
565 MB against 921 MB, because the models differ in size. The memory problem
users reported was torch being imported eagerly, which is a separate fix — so
this is a consistency guard, not a performance one.)

So the two lists are compared here, with an explicit allow-list for packages
that are genuinely optional (guarded import, documented fallback).
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# Genuinely optional: imported behind a try/except with a working fallback.
# Adding a name here is a claim that the product still works without it.
OPTIONAL = {
    "apscheduler",  # reflection scheduler — falls back to manual drain
    "watchdog",     # file watcher — falls back to polling
}


def _name(spec: str) -> str:
    return re.split(r"[<>=\[!~;]", spec, maxsplit=1)[0].strip().lower()


def _requirements() -> set[str]:
    out = set()
    for line in (ROOT / "requirements.txt").read_text().splitlines():
        line = line.split("#")[0].strip()
        if line and not line.startswith("-"):
            out.add(_name(line))
    return out


def _pyproject() -> set[str]:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text())
    return {_name(d) for d in data["project"]["dependencies"]}


def test_nothing_required_is_missing_from_the_wheel_metadata():
    missing = _requirements() - _pyproject() - OPTIONAL
    assert not missing, (
        f"{sorted(missing)} are in requirements.txt but not in pyproject "
        "dependencies — install.sh and Docker users get them, every pip / uvx / "
        "npx / brew user does not. Either declare them, or add them to OPTIONAL "
        "here and make sure the import is guarded with a real fallback."
    )


def test_the_default_embedding_backend_is_a_hard_dependency():
    """Without fastembed the server falls back to sentence-transformers+torch."""
    assert "fastembed" in _pyproject(), (
        "fastembed is the default embedding path; if it is optional the "
        "advertised memory footprint is not what users get"
    )


def test_the_http_transport_dependencies_are_declared():
    """MCP_TRANSPORT=http is a documented feature, not an accident of mcp[cli]."""
    deps = _pyproject()
    assert {"starlette", "uvicorn"} <= deps


@pytest.mark.parametrize("package", sorted(OPTIONAL))
def test_optional_packages_really_are_optional(package: str):
    """An entry in OPTIONAL must be imported defensively somewhere in src/."""
    hits = [
        path
        for path in (ROOT / "src").rglob("*.py")
        if package in path.read_text(errors="replace").lower()
    ]
    assert hits, f"{package} is marked optional but src/ never mentions it"
    guarded = any(
        re.search(
            rf"try:[^\n]*\n(?:.*\n)*?\s*(?:from|import)\s+{package}",
            path.read_text(errors="replace"),
            re.IGNORECASE,
        )
        for path in hits
    )
    assert guarded, (
        f"{package} is in OPTIONAL but its import in {[str(h.relative_to(ROOT)) for h in hits]} "
        "is not inside a try/except — it is a hard dependency in practice"
    )
