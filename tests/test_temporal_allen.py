"""Allen interval algebra: 13 base relations + composition table."""

from datetime import datetime

import pytest

from memory_core.temporal import (
    AllenRelation,
    Interval,
    NotSupportedComposition,
    after,
    before,
    compose,
    contains,
    during,
    equals,
    finished_by,
    finishes,
    met_by,
    meets,
    overlapped_by,
    overlaps,
    relation,
    started_by,
    starts,
)
from memory_core.temporal.allen import supported_compositions


# --------------------------------------------------------------------- #
# Helpers                                                               #
# --------------------------------------------------------------------- #


def I(  # noqa: E743 — short helper deliberate
    y_a: int, m_a: int, d_a: int,
    y_b: int, m_b: int, d_b: int,
    *,
    closed_start: bool = True,
    closed_end: bool = False,
) -> Interval:
    return Interval(
        datetime(y_a, m_a, d_a),
        datetime(y_b, m_b, d_b),
        closed_start=closed_start,
        closed_end=closed_end,
    )


# --------------------------------------------------------------------- #
# Construction / validation                                             #
# --------------------------------------------------------------------- #


def test_interval_rejects_inverted_bounds():
    with pytest.raises(ValueError):
        Interval(datetime(2024, 5, 1), datetime(2024, 4, 1))


def test_interval_rejects_open_point_interval():
    with pytest.raises(ValueError):
        Interval(
            datetime(2024, 5, 1),
            datetime(2024, 5, 1),
            closed_start=True,
            closed_end=False,
        )


def test_interval_accepts_closed_point():
    iv = Interval(
        datetime(2024, 5, 1),
        datetime(2024, 5, 1),
        closed_start=True,
        closed_end=True,
    )
    assert iv.start == iv.end


# --------------------------------------------------------------------- #
# Each of 13 relations: positive + a couple of inverse / non-match cases #
# --------------------------------------------------------------------- #


def test_before_strict():
    a = I(2024, 1, 1, 2024, 1, 10)
    b = I(2024, 2, 1, 2024, 2, 10)
    assert before(a, b) is True
    assert relation(a, b) == AllenRelation.BEFORE


def test_after_is_inverse_of_before():
    a = I(2024, 2, 1, 2024, 2, 10)
    b = I(2024, 1, 1, 2024, 1, 10)
    assert after(a, b) is True
    assert relation(a, b) == AllenRelation.AFTER


def test_meets_half_open_seam():
    """In the [start, end) convention, A.end == B.start with one closed
    side and one open side counts as 'meets', not 'overlaps'."""
    a = I(2024, 1, 1, 2024, 2, 1)
    b = I(2024, 2, 1, 2024, 3, 1)
    assert meets(a, b) is True
    assert relation(a, b) == AllenRelation.MEETS


def test_met_by_is_inverse_of_meets():
    a = I(2024, 2, 1, 2024, 3, 1)
    b = I(2024, 1, 1, 2024, 2, 1)
    assert met_by(a, b) is True
    assert relation(a, b) == AllenRelation.MET_BY


def test_meets_with_both_endpoints_open_is_before():
    a = Interval(datetime(2024, 1, 1), datetime(2024, 2, 1),
                 closed_start=False, closed_end=False)
    b = Interval(datetime(2024, 2, 1), datetime(2024, 3, 1),
                 closed_start=False, closed_end=False)
    # Both seam points open → no shared instant, no contact
    assert meets(a, b) is False
    assert relation(a, b) == AllenRelation.BEFORE


def test_meets_with_both_endpoints_closed_is_overlap():
    a = Interval(datetime(2024, 1, 1), datetime(2024, 2, 1),
                 closed_start=True, closed_end=True)
    b = Interval(datetime(2024, 2, 1), datetime(2024, 3, 1),
                 closed_start=True, closed_end=True)
    # Both seam points closed → 1-instant overlap
    assert relation(a, b) == AllenRelation.OVERLAPS


def test_overlaps_strict():
    a = I(2024, 1, 1, 2024, 2, 15)
    b = I(2024, 2, 1, 2024, 3, 1)
    assert overlaps(a, b) is True
    assert relation(a, b) == AllenRelation.OVERLAPS


def test_overlapped_by_inverse():
    a = I(2024, 2, 1, 2024, 3, 1)
    b = I(2024, 1, 1, 2024, 2, 15)
    assert overlapped_by(a, b) is True
    assert relation(a, b) == AllenRelation.OVERLAPPED_BY


def test_starts_same_start_a_shorter():
    a = I(2024, 1, 1, 2024, 1, 15)
    b = I(2024, 1, 1, 2024, 2, 1)
    assert starts(a, b) is True
    assert relation(a, b) == AllenRelation.STARTS


def test_started_by_inverse():
    a = I(2024, 1, 1, 2024, 2, 1)
    b = I(2024, 1, 1, 2024, 1, 15)
    assert started_by(a, b) is True
    assert relation(a, b) == AllenRelation.STARTED_BY


def test_during_strictly_inside():
    a = I(2024, 1, 5, 2024, 1, 20)
    b = I(2024, 1, 1, 2024, 2, 1)
    assert during(a, b) is True
    assert relation(a, b) == AllenRelation.DURING


def test_contains_inverse():
    a = I(2024, 1, 1, 2024, 2, 1)
    b = I(2024, 1, 5, 2024, 1, 20)
    assert contains(a, b) is True
    assert relation(a, b) == AllenRelation.CONTAINS


def test_finishes_same_end_a_starts_later():
    a = I(2024, 1, 15, 2024, 2, 1)
    b = I(2024, 1, 1, 2024, 2, 1)
    assert finishes(a, b) is True
    assert relation(a, b) == AllenRelation.FINISHES


def test_finished_by_inverse():
    a = I(2024, 1, 1, 2024, 2, 1)
    b = I(2024, 1, 15, 2024, 2, 1)
    assert finished_by(a, b) is True
    assert relation(a, b) == AllenRelation.FINISHED_BY


def test_equals_full_match():
    a = I(2024, 1, 1, 2024, 2, 1)
    b = I(2024, 1, 1, 2024, 2, 1)
    assert equals(a, b) is True
    assert relation(a, b) == AllenRelation.EQUALS


def test_equals_rejects_different_boundary_types():
    a = Interval(datetime(2024, 1, 1), datetime(2024, 2, 1),
                 closed_start=True, closed_end=False)
    b = Interval(datetime(2024, 1, 1), datetime(2024, 2, 1),
                 closed_start=True, closed_end=True)
    assert equals(a, b) is False


# --------------------------------------------------------------------- #
# Inverse property: relation(a, b).inverse == relation(b, a)            #
# --------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "a, b",
    [
        (I(2024, 1, 1, 2024, 1, 10), I(2024, 2, 1, 2024, 2, 10)),  # before/after
        (I(2024, 1, 1, 2024, 2, 1), I(2024, 2, 1, 2024, 3, 1)),    # meets/met_by
        (I(2024, 1, 1, 2024, 2, 15), I(2024, 2, 1, 2024, 3, 1)),   # overlaps/oi
        (I(2024, 1, 1, 2024, 1, 15), I(2024, 1, 1, 2024, 2, 1)),   # starts/si
        (I(2024, 1, 5, 2024, 1, 20), I(2024, 1, 1, 2024, 2, 1)),   # during/contains
        (I(2024, 1, 15, 2024, 2, 1), I(2024, 1, 1, 2024, 2, 1)),   # finishes/fi
        (I(2024, 1, 1, 2024, 2, 1), I(2024, 1, 1, 2024, 2, 1)),    # equals/equals
    ],
)
def test_relation_is_symmetric_with_inverse(a, b):
    assert relation(a, b).inverse == relation(b, a)


# --------------------------------------------------------------------- #
# Composition table                                                     #
# --------------------------------------------------------------------- #


def test_compose_before_before():
    assert compose(AllenRelation.BEFORE, AllenRelation.BEFORE) == AllenRelation.BEFORE


def test_compose_before_meets():
    assert compose(AllenRelation.BEFORE, AllenRelation.MEETS) == AllenRelation.BEFORE


def test_compose_meets_meets_yields_before():
    assert compose(AllenRelation.MEETS, AllenRelation.MEETS) == AllenRelation.BEFORE


def test_compose_during_during():
    assert compose(AllenRelation.DURING, AllenRelation.DURING) == AllenRelation.DURING


def test_compose_contains_contains():
    assert compose(AllenRelation.CONTAINS, AllenRelation.CONTAINS) == AllenRelation.CONTAINS


def test_compose_after_after():
    assert compose(AllenRelation.AFTER, AllenRelation.AFTER) == AllenRelation.AFTER


def test_compose_starts_starts():
    assert compose(AllenRelation.STARTS, AllenRelation.STARTS) == AllenRelation.STARTS


def test_compose_finishes_finishes():
    assert compose(AllenRelation.FINISHES, AllenRelation.FINISHES) == AllenRelation.FINISHES


def test_compose_meets_during_yields_overlaps():
    assert compose(AllenRelation.MEETS, AllenRelation.DURING) == AllenRelation.OVERLAPS


def test_compose_during_starts_yields_during():
    assert compose(AllenRelation.DURING, AllenRelation.STARTS) == AllenRelation.DURING


def test_compose_equals_is_identity():
    """eq composed with anything (either side) returns that other relation."""
    for r in AllenRelation:
        assert compose(AllenRelation.EQUALS, r) == r
        assert compose(r, AllenRelation.EQUALS) == r


def test_compose_unknown_pair_raises():
    # overlaps ∘ during is the canonical "disjunctive" case in Allen's
    # composition table — three possible base relations.
    with pytest.raises(NotSupportedComposition):
        compose(AllenRelation.OVERLAPS, AllenRelation.DURING)


def test_supported_compositions_count_is_at_least_30():
    """Acceptance criterion: cover ≥ 30 deterministic pairs."""
    triples = supported_compositions()
    assert len(triples) >= 30
    # Sanity: every triple is in (rel, rel, rel) shape
    for r1, r2, res in triples:
        assert isinstance(r1, AllenRelation)
        assert isinstance(r2, AllenRelation)
        assert isinstance(res, AllenRelation)


def test_supported_compositions_are_consistent_with_compose():
    for r1, r2, res in supported_compositions():
        assert compose(r1, res.EQUALS) == res or compose(r1, r2) == res


# --------------------------------------------------------------------- #
# AllenRelation enum mechanics                                          #
# --------------------------------------------------------------------- #


def test_inverse_is_involutive():
    for r in AllenRelation:
        assert r.inverse.inverse == r


def test_short_codes_are_unique():
    codes = [r.value for r in AllenRelation]
    assert len(codes) == len(set(codes)) == 13
