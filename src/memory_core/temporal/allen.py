"""Allen's interval algebra: 13 base relations + composition table.

Reference: Allen, J.F. (1983) "Maintaining Knowledge About Temporal
Intervals", CACM 26(11). The 13 mutually exclusive, jointly exhaustive
relations between two proper intervals A and B are:

    before (b)        A finishes strictly before B starts
    after (bi)        inverse of before
    meets (m)         A.end == B.start (touching, no gap, no overlap)
    met_by (mi)       inverse of meets
    overlaps (o)      A starts before B and they intersect non-trivially
    overlapped_by(oi) inverse of overlaps
    starts (s)        A.start == B.start, A.end < B.end
    started_by (si)   inverse of starts
    during (d)        A is strictly inside B
    contains (di)     inverse of during
    finishes (f)      A.end == B.end, A.start > B.start
    finished_by (fi)  inverse of finishes
    equals (eq)       A.start == B.start AND A.end == B.end

Boundary handling
-----------------
Real-world intervals come in flavours: closed-closed [a, b], closed-open
[a, b), open-closed (a, b], open-open (a, b). The half-open closed-open
variant (the default here) is the one the rest of the codebase uses
(``DateRange`` semantics + LoCoMo session windows). For Allen ``meets``
to be distinguishable from ``overlaps`` we need to know whether a single
shared endpoint counts as overlap or merely touching.

Convention used by :func:`relation`:
    * Two intervals "touch at a point" iff one's end equals the other's
      start AND at most one of those endpoints is closed. That collapses
      to ``meets``/``met_by``.
    * If both endpoints at a shared instant are closed, there is a
      single shared moment → counted as overlap (overlaps/overlapped_by
      depending on direction).
    * If both endpoints at a shared instant are open, they neither meet
      nor overlap — they fall back to ``before``/``after``.

This is the standard "point-as-overlap" convention; it makes the
relations partition the space cleanly for any boundary configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Final


class AllenRelation(str, Enum):
    """The 13 base relations of Allen's interval algebra.

    Stored as their canonical short codes so they round-trip cleanly
    through JSON / logs.
    """

    BEFORE = "b"
    AFTER = "bi"
    MEETS = "m"
    MET_BY = "mi"
    OVERLAPS = "o"
    OVERLAPPED_BY = "oi"
    STARTS = "s"
    STARTED_BY = "si"
    DURING = "d"
    CONTAINS = "di"
    FINISHES = "f"
    FINISHED_BY = "fi"
    EQUALS = "eq"

    @property
    def inverse(self) -> "AllenRelation":
        return _INVERSES[self]


_INVERSES: Final[dict[AllenRelation, AllenRelation]] = {
    AllenRelation.BEFORE: AllenRelation.AFTER,
    AllenRelation.AFTER: AllenRelation.BEFORE,
    AllenRelation.MEETS: AllenRelation.MET_BY,
    AllenRelation.MET_BY: AllenRelation.MEETS,
    AllenRelation.OVERLAPS: AllenRelation.OVERLAPPED_BY,
    AllenRelation.OVERLAPPED_BY: AllenRelation.OVERLAPS,
    AllenRelation.STARTS: AllenRelation.STARTED_BY,
    AllenRelation.STARTED_BY: AllenRelation.STARTS,
    AllenRelation.DURING: AllenRelation.CONTAINS,
    AllenRelation.CONTAINS: AllenRelation.DURING,
    AllenRelation.FINISHES: AllenRelation.FINISHED_BY,
    AllenRelation.FINISHED_BY: AllenRelation.FINISHES,
    AllenRelation.EQUALS: AllenRelation.EQUALS,
}


class NotSupportedComposition(Exception):
    """Raised by :func:`compose` for relation pairs not in the curated table.

    The full Allen composition table maps to disjunctions of base
    relations (e.g. ``overlaps ∘ during`` ≡ ``{o, d, s}``). We only
    expose pairs that yield a single base relation deterministically;
    the rest raise this so callers don't silently get a wrong answer.
    """


@dataclass(frozen=True)
class Interval:
    """A time interval with explicit boundary semantics.

    Defaults to the half-open ``[start, end)`` convention which matches
    how the rest of the codebase models date ranges (see
    ``temporal_filter.DateRange``). Pass ``closed_end=True`` for a
    classical fully-closed interval.

    A "point interval" (``start == end``) is allowed only when both
    boundaries are closed; otherwise the interval would be empty and
    Allen relations are undefined for empty sets.
    """

    start: datetime
    end: datetime
    closed_start: bool = True
    closed_end: bool = False
    label: str | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        if self.start > self.end:
            raise ValueError(
                f"Interval start ({self.start.isoformat()}) is after "
                f"end ({self.end.isoformat()})"
            )
        if self.start == self.end and not (self.closed_start and self.closed_end):
            raise ValueError(
                "Degenerate (start == end) interval requires both "
                "endpoints to be closed; otherwise it is empty."
            )

    # --- predicates that look at boundary type at a shared instant ----- #

    def starts_inclusive(self) -> bool:
        return self.closed_start

    def ends_inclusive(self) -> bool:
        return self.closed_end


# Convenience aliases for the rest of the module.
_ClosedClosed = "[a, b]"
_HalfOpen = "[a, b)"


# --------------------------------------------------------------------- #
# Helpers                                                               #
# --------------------------------------------------------------------- #


def _shared_instant_overlaps(a_inclusive: bool, b_inclusive: bool) -> bool:
    """Decide whether a single shared endpoint counts as overlap.

    Both endpoints closed → the single instant is in both intervals
    (overlap by one point). Otherwise → they merely touch (meets) or
    don't intersect at all.
    """
    return a_inclusive and b_inclusive


def _touching(end_inclusive: bool, start_inclusive: bool) -> bool:
    """A.end == B.start and exactly one side is open → meets/met_by."""
    return end_inclusive ^ start_inclusive


# --------------------------------------------------------------------- #
# 13 base relations                                                     #
# --------------------------------------------------------------------- #


def before(a: Interval, b: Interval) -> bool:
    """A ends strictly before B begins."""
    if a.end < b.start:
        return True
    if a.end == b.start:
        # No overlap and not "meets" → both endpoints open at the seam.
        return not a.ends_inclusive() and not b.starts_inclusive()
    return False


def after(a: Interval, b: Interval) -> bool:
    return before(b, a)


def meets(a: Interval, b: Interval) -> bool:
    """A.end == B.start and exactly one of those endpoints is open."""
    if a.end != b.start:
        return False
    return _touching(a.ends_inclusive(), b.starts_inclusive())


def met_by(a: Interval, b: Interval) -> bool:
    return meets(b, a)


def overlaps(a: Interval, b: Interval) -> bool:
    """A starts before B, A ends inside B (strict, non-zero overlap).

    Edge case: A.end == B.start with both endpoints closed counts as a
    one-point overlap → still ``overlaps`` per the convention above.
    """
    if a.start >= b.start:
        return False
    if a.end <= b.start:
        # A.end < B.start → before; A.end == B.start needs the closed
        # case to count as overlap.
        if a.end == b.start and _shared_instant_overlaps(
            a.ends_inclusive(), b.starts_inclusive()
        ):
            return True
        return False
    if a.end >= b.end:
        return False
    return True


def overlapped_by(a: Interval, b: Interval) -> bool:
    return overlaps(b, a)


def starts(a: Interval, b: Interval) -> bool:
    """A and B start at the same instant; A ends strictly before B."""
    if a.start != b.start or a.starts_inclusive() != b.starts_inclusive():
        return False
    return a.end < b.end


def started_by(a: Interval, b: Interval) -> bool:
    return starts(b, a)


def during(a: Interval, b: Interval) -> bool:
    """A is strictly inside B (no shared endpoints)."""
    return a.start > b.start and a.end < b.end


def contains(a: Interval, b: Interval) -> bool:
    return during(b, a)


def finishes(a: Interval, b: Interval) -> bool:
    """A and B end at the same instant; A starts strictly after B."""
    if a.end != b.end or a.ends_inclusive() != b.ends_inclusive():
        return False
    return a.start > b.start


def finished_by(a: Interval, b: Interval) -> bool:
    return finishes(b, a)


def equals(a: Interval, b: Interval) -> bool:
    """Same start, same end, same boundary inclusivity on both sides."""
    return (
        a.start == b.start
        and a.end == b.end
        and a.starts_inclusive() == b.starts_inclusive()
        and a.ends_inclusive() == b.ends_inclusive()
    )


# Order matters: most-specific relations first. The 13 relations are
# mutually exclusive on proper intervals, but evaluating ``equals``
# before ``starts``/``finishes`` keeps things readable.
_PREDICATES: Final[tuple[tuple[AllenRelation, callable], ...]] = (
    (AllenRelation.EQUALS, equals),
    (AllenRelation.STARTS, starts),
    (AllenRelation.STARTED_BY, started_by),
    (AllenRelation.FINISHES, finishes),
    (AllenRelation.FINISHED_BY, finished_by),
    (AllenRelation.DURING, during),
    (AllenRelation.CONTAINS, contains),
    (AllenRelation.MEETS, meets),
    (AllenRelation.MET_BY, met_by),
    (AllenRelation.OVERLAPS, overlaps),
    (AllenRelation.OVERLAPPED_BY, overlapped_by),
    (AllenRelation.BEFORE, before),
    (AllenRelation.AFTER, after),
)


def relation(a: Interval, b: Interval) -> AllenRelation:
    """Classify the relation between A and B as one of the 13 base relations.

    Raises :class:`ValueError` if no relation matches — that should be
    impossible for proper intervals and indicates a bug or a degenerate
    input that slipped past :class:`Interval` validation.
    """
    for rel, pred in _PREDICATES:
        if pred(a, b):
            return rel
    raise ValueError(
        "No Allen relation matched. This is a bug; intervals: "
        f"a=[{a.start} .. {a.end}] b=[{b.start} .. {b.end}]"
    )


# --------------------------------------------------------------------- #
# Composition table                                                     #
# --------------------------------------------------------------------- #
#
# r(A, C) given r1(A, B) and r2(B, C). The full table maps to
# disjunctions of base relations; we expose only pairs that compose to
# a single deterministic base relation. That is sufficient for the
# memory-system use case: "if X happened before Y and Y happened before
# Z, then X happened before Z" — chained reasoning, not full constraint
# satisfaction.
#
# Pairs not listed raise :class:`NotSupportedComposition` so callers
# fall back to the LLM (or a richer solver) instead of getting a wrong
# answer. We deliberately enumerate rather than compute: composition is
# tricky enough that an explicit, audited table beats clever code.

_COMPOSITION: Final[dict[tuple[AllenRelation, AllenRelation], AllenRelation]] = {
    # ── transitive chains around BEFORE ───────────────────────────── #
    (AllenRelation.BEFORE, AllenRelation.BEFORE): AllenRelation.BEFORE,
    (AllenRelation.BEFORE, AllenRelation.MEETS): AllenRelation.BEFORE,
    (AllenRelation.BEFORE, AllenRelation.OVERLAPS): AllenRelation.BEFORE,
    (AllenRelation.BEFORE, AllenRelation.STARTS): AllenRelation.BEFORE,
    (AllenRelation.BEFORE, AllenRelation.DURING): AllenRelation.BEFORE,
    (AllenRelation.BEFORE, AllenRelation.FINISHES): AllenRelation.BEFORE,
    (AllenRelation.MEETS, AllenRelation.BEFORE): AllenRelation.BEFORE,
    (AllenRelation.OVERLAPS, AllenRelation.BEFORE): AllenRelation.BEFORE,
    (AllenRelation.STARTS, AllenRelation.BEFORE): AllenRelation.BEFORE,
    (AllenRelation.DURING, AllenRelation.BEFORE): AllenRelation.BEFORE,
    (AllenRelation.FINISHES, AllenRelation.BEFORE): AllenRelation.BEFORE,
    # ── transitive chains around AFTER ────────────────────────────── #
    (AllenRelation.AFTER, AllenRelation.AFTER): AllenRelation.AFTER,
    (AllenRelation.AFTER, AllenRelation.MET_BY): AllenRelation.AFTER,
    (AllenRelation.AFTER, AllenRelation.OVERLAPPED_BY): AllenRelation.AFTER,
    (AllenRelation.AFTER, AllenRelation.STARTED_BY): AllenRelation.AFTER,
    (AllenRelation.AFTER, AllenRelation.CONTAINS): AllenRelation.AFTER,
    (AllenRelation.AFTER, AllenRelation.FINISHED_BY): AllenRelation.AFTER,
    (AllenRelation.MET_BY, AllenRelation.AFTER): AllenRelation.AFTER,
    (AllenRelation.OVERLAPPED_BY, AllenRelation.AFTER): AllenRelation.AFTER,
    (AllenRelation.STARTED_BY, AllenRelation.AFTER): AllenRelation.AFTER,
    (AllenRelation.CONTAINS, AllenRelation.AFTER): AllenRelation.AFTER,
    (AllenRelation.FINISHED_BY, AllenRelation.AFTER): AllenRelation.AFTER,
    # ── MEETS chains that stay deterministic ─────────────────────── #
    (AllenRelation.MEETS, AllenRelation.MEETS): AllenRelation.BEFORE,
    (AllenRelation.MET_BY, AllenRelation.MET_BY): AllenRelation.AFTER,
    (AllenRelation.MEETS, AllenRelation.DURING): AllenRelation.OVERLAPS,
    (AllenRelation.MET_BY, AllenRelation.CONTAINS): AllenRelation.OVERLAPPED_BY,
    (AllenRelation.MEETS, AllenRelation.FINISHES): AllenRelation.MEETS,
    (AllenRelation.MET_BY, AllenRelation.STARTS): AllenRelation.MET_BY,
    # ── DURING / CONTAINS deterministic chains ───────────────────── #
    (AllenRelation.DURING, AllenRelation.DURING): AllenRelation.DURING,
    (AllenRelation.CONTAINS, AllenRelation.CONTAINS): AllenRelation.CONTAINS,
    (AllenRelation.DURING, AllenRelation.STARTS): AllenRelation.DURING,
    (AllenRelation.DURING, AllenRelation.FINISHES): AllenRelation.DURING,
    (AllenRelation.STARTS, AllenRelation.DURING): AllenRelation.DURING,
    (AllenRelation.FINISHES, AllenRelation.DURING): AllenRelation.DURING,
    (AllenRelation.STARTS, AllenRelation.CONTAINS): AllenRelation.CONTAINS,
    (AllenRelation.FINISHES, AllenRelation.CONTAINS): AllenRelation.CONTAINS,
    # ── STARTS / FINISHES self-chains ────────────────────────────── #
    (AllenRelation.STARTS, AllenRelation.STARTS): AllenRelation.STARTS,
    (AllenRelation.STARTED_BY, AllenRelation.STARTED_BY): AllenRelation.STARTED_BY,
    (AllenRelation.FINISHES, AllenRelation.FINISHES): AllenRelation.FINISHES,
    (AllenRelation.FINISHED_BY, AllenRelation.FINISHED_BY): AllenRelation.FINISHED_BY,
    # ── EQUALS is the identity element for every relation ────────── #
    # (built dynamically below to avoid 26 repetitive entries)
}


def _install_equals_identity() -> None:
    """``eq ∘ r == r`` and ``r ∘ eq == r`` for every base relation."""
    for r in AllenRelation:
        _COMPOSITION[(AllenRelation.EQUALS, r)] = r
        _COMPOSITION[(r, AllenRelation.EQUALS)] = r


_install_equals_identity()


def compose(r1: AllenRelation, r2: AllenRelation) -> AllenRelation:
    """Return the deterministic composition r(A, C) given r1(A, B), r2(B, C).

    Raises :class:`NotSupportedComposition` for pairs whose true result
    is a disjunction (e.g. ``overlaps ∘ during`` could be any of
    {overlaps, during, starts}). Callers that need full constraint
    propagation should escalate to a SAT/CSP solver — out of scope for
    this module.
    """
    try:
        return _COMPOSITION[(r1, r2)]
    except KeyError as exc:
        raise NotSupportedComposition(
            f"Composition {r1.name} ∘ {r2.name} maps to a disjunction "
            "of base relations and is not supported by this table. "
            "Use a constraint solver if you need it."
        ) from exc


def supported_compositions() -> list[tuple[AllenRelation, AllenRelation, AllenRelation]]:
    """Enumerate every (r1, r2, result) triple in the composition table.

    Useful for tests and documentation. Sorted so output is stable.
    """
    triples = [(r1, r2, res) for (r1, r2), res in _COMPOSITION.items()]
    triples.sort(key=lambda t: (t[0].value, t[1].value))
    return triples
