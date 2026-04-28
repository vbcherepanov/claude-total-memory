"""v11 W2-G — Background workers.

The `workers` package owns long-running off-hot-path jobs (currently:
the idle-project consolidation daemon). Workers MAY import from
`memory_core` but the reverse is forbidden — this is enforced by the
v11 layer-separation regression test.

The daemon entrypoint lives in :mod:`workers.consolidation_daemon`;
project-activity tracking lives in :mod:`workers.project_activity`.
"""

from __future__ import annotations

from workers.consolidation_daemon import (
    ConsolidationStats,
    consolidate_project,
    run_daemon,
)
from workers.project_activity import (
    get_active_project,
    is_active,
    list_idle_projects,
    touch,
)

__all__ = [
    "ConsolidationStats",
    "consolidate_project",
    "run_daemon",
    "get_active_project",
    "is_active",
    "list_idle_projects",
    "touch",
]
