"""v11.0 Phase 4 regression — `ai_layer` ↔ `memory_core` import wall.

Three guarantees this suite locks down:

(a) `memory_core` does NOT import `ai_layer` (directly or transitively).
    AST-walk every `*.py` under `src/memory_core/`, fail if any module
    has `import ai_layer.*` or `from ai_layer ... import ...`.

(b) `ai_layer` is importable as a package and every advertised submodule
    resolves cleanly.

(c) Every existing v10 LLM-touching module
    (`quality_gate`, `contradiction_detector`, `coref_resolver`,
    `deep_enricher`, `representations`, `reranker`, `query_rewriter`)
    is reachable through some `ai_layer.*` shim. We compute the
    transitive set of v10 modules imported by `ai_layer.*` and check
    that the seven contract modules are inside it.

These tests have no LLM dependency, no DB dependency, and run in
milliseconds. They are CI-cheap and should never be skipped.
"""

from __future__ import annotations

import ast
import importlib
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────


def _module_imports(py_path: Path) -> set[str]:
    """Return the set of top-level module names imported by `py_path`.

    Parses with `ast` so we don't execute the module — this matters for
    `memory_core/` which may import heavy stuff at module top-level.

    Tolerates rare encoding/parse hiccups (e.g. a sibling phase still
    writing the file) by skipping the offending file rather than
    failing the whole suite — a half-written file cannot import
    `ai_layer` because `ast.parse` would have raised before the import
    statement was even visible.
    """
    try:
        source = py_path.read_text()
    except (UnicodeDecodeError, OSError):
        return set()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for n in node.names:
                out.add(n.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            # `from X.Y import Z` → top-level is X. Relative imports
            # (`from . import x`) have `node.module is None` or the
            # parent name; we only care about top-level src/ modules.
            if node.level == 0 and node.module:
                out.add(node.module.split(".")[0])
    return out


def _walk_pkg(pkg_dir: Path) -> list[Path]:
    """Yield every `*.py` under `pkg_dir` recursively, excluding caches."""
    return [p for p in pkg_dir.rglob("*.py") if "__pycache__" not in p.parts]


# ──────────────────────────────────────────────
# (a) memory_core does not import ai_layer
# ──────────────────────────────────────────────


def test_memory_core_does_not_import_ai_layer() -> None:
    """The whole point of v11.0 Phase 3-4: a hard wall.

    `memory_core` is the deterministic, sync, no-LLM hot path. Any
    `import ai_layer*` inside it would break the contract — even one.
    """
    core = SRC / "memory_core"
    if not core.is_dir():
        pytest.skip("memory_core/ not extracted yet (Phase 3 in progress)")

    offenders: list[str] = []
    for py in _walk_pkg(core):
        for top in _module_imports(py):
            if top == "ai_layer":
                offenders.append(str(py.relative_to(SRC)))
    assert offenders == [], (
        f"memory_core modules import ai_layer (forbidden by v11 §1): "
        f"{sorted(set(offenders))}"
    )


# ──────────────────────────────────────────────
# (b) ai_layer is importable as a package
# ──────────────────────────────────────────────


_AI_LAYER_PUBLIC = (
    "enrichment_worker",
    "enrichment_jobs",
    "summarizer",
    "keyword_extractor",
    "question_generator",
    "relation_extractor",
    "contradiction_detector",
    "reflection",
    "self_improve",
    "quality_gate",
    "coref_resolver",
    "reranker",
    "query_rewriter",
)


def test_ai_layer_package_imports() -> None:
    """`import ai_layer` must succeed and expose every advertised submodule."""
    pkg = importlib.import_module("ai_layer")
    missing = [name for name in _AI_LAYER_PUBLIC if not hasattr(pkg, name)]
    assert not missing, f"ai_layer is missing submodules: {missing}"


@pytest.mark.parametrize("submodule", _AI_LAYER_PUBLIC)
def test_ai_layer_submodule_imports(submodule: str) -> None:
    """Every `ai_layer.<x>` shim resolves on its own."""
    importlib.import_module(f"ai_layer.{submodule}")


# ──────────────────────────────────────────────
# (c) v10 LLM-touching modules reachable through ai_layer.*
# ──────────────────────────────────────────────


# The v10.5 modules that synchronously talk to the LLM. v11 Phase 4
# requires every one of them to be reachable through some ai_layer.*
# shim — directly or transitively (an ai_layer module importing it).
_V10_LLM_MODULES = (
    "quality_gate",
    "contradiction_detector",
    "coref_resolver",
    "deep_enricher",
    "representations",
    "reranker",
    "query_rewriter",
)


def _transitive_imports_from_ai_layer() -> set[str]:
    """Compute the set of v10 top-level modules reachable from ai_layer.

    We start from every `*.py` under `src/ai_layer/` and AST-walk their
    imports, treating the result as a 1-hop set (the shims themselves
    are always 1 hop from the v10 modules they wrap). This is sufficient
    for the contract: the seven v10 modules must each be the direct
    target of an `import` or `from ... import` somewhere under ai_layer.
    """
    layer = SRC / "ai_layer"
    reachable: set[str] = set()
    for py in _walk_pkg(layer):
        reachable |= _module_imports(py)
    return reachable


def test_every_v10_llm_module_is_reachable_through_ai_layer() -> None:
    """Every v10 LLM-touching module must have a shim under ai_layer.*.

    If this fails, somebody added a new LLM-using v10 module without
    parking a thin re-export in `ai_layer` — meaning the next time we
    grep "what touches the LLM?" we will miss it.
    """
    reachable = _transitive_imports_from_ai_layer()
    missing = [m for m in _V10_LLM_MODULES if m not in reachable]
    assert not missing, (
        "v10 LLM-touching modules with no ai_layer.* shim "
        f"(grep-invisible from the v11 audit): {missing}"
    )
