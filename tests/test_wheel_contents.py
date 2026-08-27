"""Regression tests for runtime assets in built distributions.

`src/` resolves several sibling directories as
``Path(__file__).resolve().parent.parent / <dir>``. Anything not declared in
`pyproject.toml` vanishes from the wheel, and the code that reads it degrades
*silently* rather than failing — the failure mode is a feature that quietly
does nothing on every pip / uvx / npx install while working perfectly in a git
checkout. That has now happened three times:

  - `migrations/` — the server crashed on first migration (fixed in #12)
  - `vocabularies/` — canonical tag normalisation fell back to an empty vocab
  - `filters/` — every `memory_save(filter=...)` became a no-op

So the wheel is built and inspected here rather than trusted.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from zipfile import ZipFile

import pytest

from build import ProjectBuilder

ROOT = Path(__file__).resolve().parents[1]

# (directory, glob, why the runtime needs it)
RUNTIME_ASSETS = [
    ("migrations", "*.sql", "schema migrations applied on first start"),
    ("vocabularies", "*.txt", "canonical tag vocabulary"),
    ("filters", "*.toml", "memory_save(filter=...) configs"),
]


@pytest.fixture(scope="module")
def wheel_names(tmp_path_factory) -> set[str]:
    """Build the wheel once and return its member list."""
    tmp_path = tmp_path_factory.mktemp("wheel-build")
    source = tmp_path / "source"
    shutil.copytree(
        ROOT,
        source,
        ignore=shutil.ignore_patterns(
            ".git", ".venv", "build", "dist", "*.egg-info", "__pycache__",
            "benchmarks", "evals", "node_modules",
        ),
    )
    wheel_path = Path(ProjectBuilder(source).build("wheel", tmp_path / "wheel"))
    with ZipFile(wheel_path) as wheel:
        return set(wheel.namelist())


def test_wheel_contains_the_server(wheel_names: set[str]) -> None:
    assert "src/server.py" in wheel_names


@pytest.mark.parametrize(
    ("directory", "pattern", "why"),
    RUNTIME_ASSETS,
    ids=[d for d, _, _ in RUNTIME_ASSETS],
)
def test_wheel_contains_runtime_assets(
    wheel_names: set[str], directory: str, pattern: str, why: str
) -> None:
    expected = {f"{directory}/{p.name}" for p in (ROOT / directory).glob(pattern)}
    assert expected, f"the source tree has no {directory}/{pattern}"
    missing = expected - wheel_names
    assert not missing, (
        f"{directory}/ is not packaged ({why}); missing {sorted(missing)[:3]}"
    )


def test_every_repo_root_dir_the_runtime_reads_is_declared() -> None:
    """Catch a *new* sibling directory before it silently goes missing.

    Greps `src/` for the `parent.parent / "<dir>"` idiom and checks each hit is
    either packaged or explicitly known to be dev-only.
    """
    import re

    dev_only = {"benchmarks", "docs", "evals", "telegram"}
    packaged = {d for d, _, _ in RUNTIME_ASSETS}

    referenced: set[str] = set()
    for path in ROOT.joinpath("src").rglob("*.py"):
        for match in re.finditer(
            r'parent\.parent\s*/\s*"([a-z_]+)"', path.read_text(errors="replace")
        ):
            referenced.add(match.group(1))

    unknown = referenced - packaged - dev_only
    assert not unknown, (
        f"src/ reads {sorted(unknown)} from the repo root, but they are neither "
        "packaged nor marked dev-only — add them to RUNTIME_ASSETS and to "
        "pyproject.toml, or to dev_only here"
    )
