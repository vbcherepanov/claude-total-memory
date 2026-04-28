"""Deterministic temporal reasoning for memory recall.

Three concerns, kept apart:

* :mod:`allen` — Allen's interval algebra: 13 base relations and a
  composition table. Pure logic over intervals.
* :mod:`normalizer` — relative phrase parser ("yesterday", "вчера",
  "in March") that grounds language to absolute datetimes given an
  anchor.
* :mod:`arithmetic` — calendar-aware duration math and human-readable
  formatting for English and Russian.

The package is stdlib-only. It does not import from server.py, recall.py,
or the existing temporal_* modules — those rely on it, not the other way
around. ``Interval`` here is intentionally separate from
``temporal_filter.DateRange``: this one models open/closed boundaries so
Allen relations can distinguish ``meets`` from ``overlaps`` correctly.
"""

from .allen import (
    AllenRelation,
    Interval,
    NotSupportedComposition,
    compose,
    relation,
    before,
    after,
    meets,
    met_by,
    overlaps,
    overlapped_by,
    starts,
    started_by,
    during,
    contains,
    finishes,
    finished_by,
    equals,
)
from .arithmetic import (
    days_between,
    duration_between,
    format_human,
    months_between,
    weeks_between,
    years_between,
)
from .normalizer import NormalizedDate, normalize

__all__ = [
    # allen
    "AllenRelation",
    "Interval",
    "NotSupportedComposition",
    "compose",
    "relation",
    "before",
    "after",
    "meets",
    "met_by",
    "overlaps",
    "overlapped_by",
    "starts",
    "started_by",
    "during",
    "contains",
    "finishes",
    "finished_by",
    "equals",
    # arithmetic
    "duration_between",
    "days_between",
    "weeks_between",
    "months_between",
    "years_between",
    "format_human",
    # normalizer
    "NormalizedDate",
    "normalize",
]
