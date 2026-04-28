"""Re-export of `src/auto_self_improve.py`.

The self-improvement helper logs errors and fixes into the memory DB
from bash hooks (memory-trigger.sh). It does NOT call the LLM directly,
but `check_patterns` is consumed by the reflection agent and other LLM
analysers — so the module rides along under `ai_layer` to keep the
"audit trail of LLM-touching code" cleanly inside one directory.

The CLI entrypoint `main` is preserved so `python -m ai_layer.self_improve`
can be wired up later without breaking the existing
`python auto_self_improve.py` invocation used by hooks.
"""

from __future__ import annotations

from auto_self_improve import (  # noqa: F401  (re-exports)
    check_patterns,
    get_db,
    log_error,
    log_fix,
    main,
)

__all__ = [
    "check_patterns",
    "get_db",
    "log_error",
    "log_fix",
    "main",
]
