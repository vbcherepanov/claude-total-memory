"""Verify builtin TOML filter configs load and behave as documented."""

from __future__ import annotations

from pathlib import Path


FILTERS_DIR = Path(__file__).parent.parent / "filters"


def _load(name: str):
    from content_filter import load_filter_config

    return load_filter_config(FILTERS_DIR / f"{name}.toml")


def test_all_builtin_filters_loadable():
    """Every TOML file under filters/ should parse without errors."""
    from content_filter import load_filter_config

    toml_files = sorted(FILTERS_DIR.glob("*.toml"))
    assert toml_files, "no builtin filters found"
    for p in toml_files:
        cfg = load_filter_config(p)
        assert cfg.get("name"), f"{p.name} missing 'name'"
        assert cfg.get("safety") in {"strict", "semantic", "off"}, f"{p.name} bad safety"
        assert "stages" in cfg, f"{p.name} missing [stages]"


def test_pytest_filter_removes_pass_lines():
    from content_filter import run_pipeline

    cfg = _load("pytest")
    text = (
        "============== test session starts ==============\n"
        "platform darwin\n"
        "collected 5 items\n"
        "test_a.py::test_one PASSED\n"
        "test_a.py::test_two FAILED\n"
        "test_a.py:42: AssertionError\n"
        "====== 1 failed, 1 passed in 0.05s ======"
    )
    out = run_pipeline(text, cfg["stages"], safety=cfg["safety"])
    assert "PASSED" not in out
    assert "FAILED" in out
    assert "AssertionError" in out
    assert "platform darwin" not in out


def test_cargo_filter_keeps_errors_drops_progress():
    from content_filter import run_pipeline

    cfg = _load("cargo")
    text = (
        "   Compiling foo v0.1.0\n"
        "   Compiling bar v0.2.0\n"
        "error[E0308]: mismatched types\n"
        "  --> src/main.rs:10:5\n"
        "   |\n"
        "10 |     let x: i32 = \"hello\";\n"
        "   |     ^^^^^^^^^^^\n"
        "   Finished dev [unoptimized] target(s) in 2.34s"
    )
    out = run_pipeline(text, cfg["stages"], safety=cfg["safety"])
    assert "Compiling" not in out
    assert "Finished" not in out
    assert "error[E0308]" in out
    assert "src/main.rs:10:5" in out  # path preserved


def test_generic_logs_filter_keeps_critical():
    from content_filter import run_pipeline

    cfg = _load("generic_logs")
    text = (
        "DEBUG connection opened\n"
        "INFO processed record 42\n"
        "WARN slow query detected\n"
        "ERROR database timeout\n"
        "FATAL crash in /var/log/x.log\n"
    )
    out = run_pipeline(text, cfg["stages"], safety=cfg["safety"])
    assert "DEBUG" not in out
    assert "INFO" not in out
    assert "WARN" in out
    assert "ERROR" in out
    assert "FATAL" in out
    assert "/var/log/x.log" in out  # path preserved even by whitelist
