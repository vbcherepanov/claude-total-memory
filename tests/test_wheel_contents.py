"""Regression tests for runtime assets in built distributions."""

import shutil
from pathlib import Path
from zipfile import ZipFile

from build import ProjectBuilder


def test_wheel_contains_all_database_migrations(tmp_path: Path) -> None:
    """Every source migration must ship where server.py resolves it."""
    root = Path(__file__).resolve().parents[1]
    source = tmp_path / "source"
    shutil.copytree(
        root,
        source,
        ignore=shutil.ignore_patterns(
            ".git", ".venv", "build", "dist", "*.egg-info", "__pycache__"
        ),
    )

    wheel_dir = tmp_path / "wheel"
    wheel_path = Path(ProjectBuilder(source).build("wheel", wheel_dir))
    expected = {
        f"migrations/{path.name}" for path in (root / "migrations").glob("*.sql")
    }

    assert expected, "the source tree must contain SQL migrations"
    with ZipFile(wheel_path) as wheel:
        names = set(wheel.namelist())

    assert expected <= names
    assert "src/server.py" in names
