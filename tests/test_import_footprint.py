"""Importing the server must not drag in the fallback ML stack.

`chromadb` and `sentence_transformers` are fallback paths — when fastembed is
healthy neither is used. They were imported at module scope anyway, and
`sentence_transformers` pulls in torch, so every user paid for them: a bare
`import server` measured 558 MB resident, and a serving process 1367 MB.
Deferring both to first use took that to 116 MB and 972 MB.

Reported by d.snezhinskiy against a client install sitting at ~1.5 GB.

These tests run the import in a subprocess, because once torch is in
`sys.modules` for the pytest process there is no way to un-import it.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# Modules that must not be loaded by `import server` alone. Each is a fallback
# whose cost is only worth paying when it is actually reached.
FORBIDDEN_ON_IMPORT = ("torch", "sentence_transformers", "chromadb")


def _probe(body: str, tmp_path: Path) -> dict:
    """Run `body` in a fresh interpreter with an isolated memory dir."""
    script = textwrap.dedent(
        f"""
        import json, os, sys
        sys.path.insert(0, {str(ROOT / "src")!r})
        os.environ["TAM_MEMORY_DIR"] = {str(tmp_path)!r}
        os.environ["MEMORY_QUIET"] = "1"
        {body}
        """
    )
    # A clean environment: by the time the suite reaches this file, an earlier
    # test has already imported `paths` in the pytest process and pinned
    # FASTEMBED_CACHE_PATH, which the child would otherwise inherit.
    env = {k: v for k, v in __import__("os").environ.items()
           if k not in ("FASTEMBED_CACHE_PATH", "TAM_MEMORY_DIR",
                        "CLAUDE_MEMORY_DIR", "TAM_MODEL_CACHE")}
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, cwd=str(ROOT), timeout=600, env=env,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    return __import__("json").loads(result.stdout.strip().splitlines()[-1])


def test_importing_the_server_does_not_load_the_fallback_stack(tmp_path):
    loaded = _probe(
        """
        import server  # noqa: F401
        print(json.dumps({m: (m in sys.modules) for m in %r}))
        """ % (FORBIDDEN_ON_IMPORT,),
        tmp_path,
    )
    offenders = [m for m, present in loaded.items() if present]
    assert not offenders, (
        f"`import server` loaded {offenders}; these are fallback paths and must "
        "stay behind a deferred import (see HAS_CHROMA / HAS_ST in server.py)"
    )


def test_availability_flags_are_still_accurate(tmp_path):
    """find_spec must agree with what an actual import would find."""
    result = _probe(
        """
        import importlib.util, server
        print(json.dumps({
            "has_chroma": server.HAS_CHROMA,
            "has_st": server.HAS_ST,
            "chroma_installed": importlib.util.find_spec("chromadb") is not None,
            "st_installed": importlib.util.find_spec("sentence_transformers") is not None,
        }))
        """,
        tmp_path,
    )
    assert result["has_chroma"] == result["chroma_installed"]
    assert result["has_st"] == result["st_installed"]


def test_the_deferred_import_still_works_when_reached(tmp_path):
    """Deferring must not turn the fallback into a dead branch."""
    result = _probe(
        """
        import server
        ok = False
        if server.HAS_ST:
            from sentence_transformers import SentenceTransformer  # noqa: F401
            ok = True
        print(json.dumps({"st_importable": ok, "torch": "torch" in sys.modules}))
        """,
        tmp_path,
    )
    if result["st_importable"]:
        assert result["torch"], "importing sentence_transformers should pull torch"


def test_the_model_cache_is_not_moved_by_default(tmp_path):
    """Pinning is opt-in on purpose.

    Moving the cache orphans models the user already downloaded: every install
    would re-fetch ~500 MB once, and every offline test run would fail. Users
    who hit the macOS tmp purge opt in with TAM_MODEL_CACHE.
    """
    result = _probe(
        """
        from paths import memory_dir
        memory_dir()
        print(json.dumps({"cache": os.environ.get("FASTEMBED_CACHE_PATH")}))
        """,
        tmp_path,
    )
    assert result["cache"] is None


def test_tam_model_cache_pins_the_cache_when_set(tmp_path):
    result = _probe(
        """
        os.environ["TAM_MODEL_CACHE"] = "/tmp/tam-models-override"
        from paths import memory_dir
        memory_dir()
        print(json.dumps({"cache": os.environ.get("FASTEMBED_CACHE_PATH")}))
        """,
        tmp_path,
    )
    assert result["cache"] == "/tmp/tam-models-override"


def test_the_pinned_cache_is_never_inside_the_memory_dir(tmp_path):
    """Models are machine-wide artifacts — a throwaway memory dir (a test, a
    benchmark, a second profile) must reuse them, not re-download."""
    result = _probe(
        """
        from paths import memory_dir, model_cache_dir
        p = memory_dir()
        print(json.dumps({"memory": str(p), "cache": str(model_cache_dir())}))
        """,
        tmp_path,
    )
    assert result["memory"] == str(tmp_path)
    assert not result["cache"].startswith(str(tmp_path))


def test_an_explicit_fastembed_cache_always_wins(tmp_path):
    result = _probe(
        """
        os.environ["FASTEMBED_CACHE_PATH"] = "/tmp/my-own-cache"
        os.environ["TAM_MODEL_CACHE"] = "/tmp/ignored"
        from paths import memory_dir
        memory_dir()
        print(json.dumps({"cache": os.environ["FASTEMBED_CACHE_PATH"]}))
        """,
        tmp_path,
    )
    assert result["cache"] == "/tmp/my-own-cache"


def test_a_failed_fastembed_init_explains_the_memory_cost(tmp_path):
    """The fallback used to be one log line; the client read it as random RAM."""
    import inspect, sys as _s
    _s.path.insert(0, str(ROOT / "src"))
    import embed_provider

    src = inspect.getsource(embed_provider.FastEmbedProvider._ensure_model)
    assert "TAM_MODEL_CACHE" in src
    assert "RSS" in src or "MB" in src


@pytest.mark.parametrize("module", FORBIDDEN_ON_IMPORT)
def test_no_module_reintroduces_the_eager_import(module: str):
    """Guard the source itself — a stray top-level import is easy to re-add."""
    source = (ROOT / "src" / "server.py").read_text()
    header = source[: source.index("class Store")]
    for line in header.splitlines():
        stripped = line.strip()
        if stripped.startswith("#") or "noqa" in stripped:
            continue
        assert not stripped.startswith(f"import {module}"), line
        assert not stripped.startswith(f"from {module} import"), line
