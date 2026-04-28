"""Date-phrase normalisation: English + Russian, edge cases."""

from datetime import datetime

import pytest

from memory_core.temporal import NormalizedDate, normalize


# A stable anchor in the middle of the year, on a Tuesday, well clear
# of month boundaries — mid-2024 means leap year quirks are reachable
# but don't dominate.
ANCHOR = datetime(2024, 6, 11, 14, 30)  # Tuesday, June 11 2024 14:30


# --------------------------------------------------------------------- #
# Trivial / passthrough                                                 #
# --------------------------------------------------------------------- #


def test_empty_returns_none():
    assert normalize("", ANCHOR) is None
    assert normalize("   ", ANCHOR) is None


def test_unknown_phrase_returns_none():
    assert normalize("kalamazoo flux capacitor", ANCHOR) is None


def test_iso_date_passthrough_keeps_full_confidence():
    r = normalize("2023-04-15", ANCHOR)
    assert r is not None
    assert r.iso == "2023-04-15T00:00:00"
    assert r.confidence == 1.0
    assert r.kind == "day"


def test_iso_with_time_keeps_exact_kind():
    r = normalize("2023-04-15T08:30:00", ANCHOR)
    assert r is not None
    assert r.iso == "2023-04-15T08:30:00"
    assert r.kind == "exact"


def test_iso_invalid_calendar_returns_none():
    # 2023-02-30 doesn't exist
    assert normalize("2023-02-30", ANCHOR) is None


# --------------------------------------------------------------------- #
# Simple offsets — bilingual                                            #
# --------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "phrase,delta_days",
    [
        ("today", 0), ("сегодня", 0),
        ("yesterday", -1), ("вчера", -1),
        ("tomorrow", 1), ("завтра", 1),
        ("day before yesterday", -2), ("позавчера", -2),
        ("day after tomorrow", 2), ("послезавтра", 2),
    ],
)
def test_simple_offsets(phrase, delta_days):
    r = normalize(phrase, ANCHOR)
    assert r is not None, phrase
    expected = (ANCHOR.replace(hour=0, minute=0, second=0, microsecond=0)
                .replace(day=ANCHOR.day + delta_days)).isoformat()
    assert r.iso == expected
    assert r.kind == "day"


# --------------------------------------------------------------------- #
# "<n> <unit> ago" — bilingual, calendar-aware                          #
# --------------------------------------------------------------------- #


def test_ago_three_days_en():
    r = normalize("3 days ago", ANCHOR)
    assert r is not None
    assert r.iso == "2024-06-08T00:00:00"
    assert r.kind == "day"


def test_ago_three_days_ru():
    r = normalize("3 дня назад", ANCHOR)
    assert r is not None
    assert r.iso == "2024-06-08T00:00:00"


def test_ago_word_form_en():
    r = normalize("two weeks ago", ANCHOR)
    assert r is not None
    assert r.iso == "2024-05-28T00:00:00"
    assert r.kind == "week"


def test_ago_word_form_ru():
    r = normalize("две недели назад", ANCHOR)
    assert r is not None
    assert r.iso == "2024-05-28T00:00:00"
    assert r.kind == "week"


def test_ago_one_month_calendar_aware():
    r = normalize("1 month ago", ANCHOR)
    assert r is not None
    # June 11 → May 11
    assert r.iso == "2024-05-11T00:00:00"
    assert r.kind == "month"


def test_ago_one_year_leap_to_nonleap_clamps_feb29():
    """Feb 29 2024 - 1 year should clamp to Feb 28 2023."""
    r = normalize("1 year ago", datetime(2024, 2, 29))
    assert r is not None
    assert r.iso == "2023-02-28T00:00:00"
    assert r.kind == "year"


def test_ago_anchor_at_month_boundary_jan31_minus_one_month():
    """Jan 31 - 1 month → Dec 31 (no clamp needed, Dec has 31 days)."""
    r = normalize("1 month ago", datetime(2024, 1, 31))
    assert r is not None
    assert r.iso == "2023-12-31T00:00:00"


def test_ago_anchor_mar31_minus_one_month_clamps_to_feb29():
    """Mar 31 2024 - 1 month → Feb 29 2024 (leap year clamp)."""
    r = normalize("1 month ago", datetime(2024, 3, 31))
    assert r is not None
    assert r.iso == "2024-02-29T00:00:00"


def test_ago_mismatched_unit_returns_none():
    # English unit + Russian preposition (not a real phrase) should fail
    assert normalize("3 weeks назад", ANCHOR) is None


# --------------------------------------------------------------------- #
# Relative periods: last / next / this week|month|year                  #
# --------------------------------------------------------------------- #


def test_last_week_en():
    r = normalize("last week", ANCHOR)
    assert r is not None
    # ANCHOR is Tuesday, June 11 → start of "last week" = Monday June 3
    assert r.iso == "2024-06-03T00:00:00"
    assert r.kind == "week"


def test_last_week_ru_prepositional():
    r = normalize("на прошлой неделе", ANCHOR)
    assert r is not None
    assert r.iso == "2024-06-03T00:00:00"


def test_this_week_returns_monday_of_current_week():
    r = normalize("this week", ANCHOR)
    assert r is not None
    assert r.iso == "2024-06-10T00:00:00"


def test_next_month_jumps_to_first_of_next_month():
    r = normalize("next month", ANCHOR)
    assert r is not None
    assert r.iso == "2024-07-01T00:00:00"


def test_next_year_ru():
    r = normalize("в следующем году", ANCHOR)
    assert r is not None
    assert r.iso == "2025-01-01T00:00:00"
    assert r.kind == "year"


def test_last_month_at_jan_wraps_to_december_previous_year():
    r = normalize("last month", datetime(2024, 1, 15))
    assert r is not None
    assert r.iso == "2023-12-01T00:00:00"


# --------------------------------------------------------------------- #
# Months                                                                #
# --------------------------------------------------------------------- #


def test_in_march_with_anchor_after_march_picks_next_year():
    """Anchor June 11 2024: 'in March' is closer to March 2024 than 2025
    (June - March 2024 = 3 months, 2025 - June = 9 months) → 2024."""
    r = normalize("in March", ANCHOR)
    assert r is not None
    assert r.iso == "2024-03-01T00:00:00"
    assert r.kind == "month"
    assert r.confidence == 0.7  # year was inferred


def test_in_october_picks_same_year_when_closest():
    r = normalize("in October", ANCHOR)
    assert r is not None
    assert r.iso == "2024-10-01T00:00:00"


def test_in_december_with_january_anchor_picks_previous_december():
    """Jan 5 2024: Dec 2023 is 1 month away, Dec 2024 is 11 months away."""
    r = normalize("in December", datetime(2024, 1, 5))
    assert r is not None
    assert r.iso == "2023-12-01T00:00:00"


def test_explicit_year_in_march_2026_full_confidence():
    r = normalize("in March 2026", ANCHOR)
    assert r is not None
    assert r.iso == "2026-03-01T00:00:00"
    assert r.confidence == 1.0


def test_ru_month_prepositional():
    r = normalize("в марте", ANCHOR)
    assert r is not None
    assert r.iso == "2024-03-01T00:00:00"


def test_ru_month_with_year():
    r = normalize("в марте 2026", ANCHOR)
    assert r is not None
    assert r.iso == "2026-03-01T00:00:00"
    assert r.confidence == 1.0


def test_short_month_name_en():
    r = normalize("Sep 2023", ANCHOR)
    assert r is not None
    assert r.iso == "2023-09-01T00:00:00"


# --------------------------------------------------------------------- #
# Relative weekdays                                                     #
# --------------------------------------------------------------------- #


def test_next_tuesday_from_tuesday_jumps_seven_days():
    """ANCHOR is Tuesday June 11. 'next Tuesday' = June 18."""
    r = normalize("next Tuesday", ANCHOR)
    assert r is not None
    assert r.iso == "2024-06-18T00:00:00"
    assert r.kind == "day"


def test_next_friday_from_tuesday():
    r = normalize("next Friday", ANCHOR)
    assert r is not None
    # June 11 is Tuesday → Friday is June 14
    assert r.iso == "2024-06-14T00:00:00"


def test_last_friday_from_tuesday():
    r = normalize("last Friday", ANCHOR)
    assert r is not None
    # Most recent Friday before Tuesday June 11 = June 7
    assert r.iso == "2024-06-07T00:00:00"


def test_this_thursday_returns_thursday_of_same_week():
    r = normalize("this Thursday", ANCHOR)
    assert r is not None
    assert r.iso == "2024-06-13T00:00:00"


def test_ru_next_weekday():
    r = normalize("в следующий вторник", ANCHOR)
    assert r is not None
    assert r.iso == "2024-06-18T00:00:00"


def test_ru_last_weekday():
    r = normalize("в прошлую пятницу", ANCHOR)
    assert r is not None
    assert r.iso == "2024-06-07T00:00:00"


# --------------------------------------------------------------------- #
# Lang routing                                                          #
# --------------------------------------------------------------------- #


def test_explicit_lang_en_ignores_russian_phrase():
    """With lang='en' a Russian phrase shouldn't accidentally match an
    English pattern."""
    assert normalize("вчера", ANCHOR, lang="en") is None


def test_explicit_lang_ru_ignores_english_phrase():
    assert normalize("yesterday", ANCHOR, lang="ru") is None


def test_invalid_lang_raises():
    with pytest.raises(ValueError):
        normalize("today", ANCHOR, lang="fr")


def test_returns_normalized_date_dataclass():
    r = normalize("yesterday", ANCHOR)
    assert isinstance(r, NormalizedDate)
    assert r.original == "yesterday"
    assert r.anchor == ANCHOR
