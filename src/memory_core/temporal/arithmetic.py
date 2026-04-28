"""Calendar-aware temporal arithmetic with bilingual human formatting.

Why a dedicated module: ``timedelta`` is fine for "how many seconds /
days" but useless for "how many calendar months from Jan 31 to Feb 28".
LoCoMo's temporal questions ("how long after the wedding did they move
in", "сколько месяцев прошло между релизами") need calendar-aware
counting that respects month/year length.

Conventions
-----------
* ``days_between`` and ``weeks_between`` are floor-division on whole
  days / 7-day periods. They never go fractional.
* ``months_between`` follows the ISO/Postgres-like rule: count whole
  calendar-month boundaries that have been crossed. ``Jan 31 → Feb 28``
  in a non-leap year is ``1`` month (the day-of-month "rolls back" to
  the last valid day of the target month). ``Jan 31 → Feb 27`` is
  ``0`` months.
* ``years_between`` works the same way on the year axis: ``Feb 29 2020
  → Feb 28 2021`` is ``1`` year.
* ``duration_between`` always returns a non-negative ``timedelta``;
  direction is intentionally lost, callers that care ask for ordering.
* For interval inputs we anchor on ``start`` for "moment-style" math
  (length of the gap between two events) — it is the only choice that
  makes ``duration_between(point, interval) == duration_between(point,
  interval.start)`` symmetric with the datetime case.
"""

from __future__ import annotations

import calendar
from datetime import datetime, timedelta
from typing import Final, Union

from .allen import Interval


# Type alias: anything you can take a "moment" of for these helpers.
TemporalLike = Union[datetime, Interval]


def _as_datetime(x: TemporalLike) -> datetime:
    """Anchor an Interval on its start; pass a datetime through."""
    if isinstance(x, Interval):
        return x.start
    if isinstance(x, datetime):
        return x
    raise TypeError(
        f"Expected datetime or Interval, got {type(x).__name__}"
    )


def duration_between(a: TemporalLike, b: TemporalLike) -> timedelta:
    """Absolute distance between two moments. Always non-negative."""
    da, db = _as_datetime(a), _as_datetime(b)
    return abs(db - da)


def days_between(a: TemporalLike, b: TemporalLike) -> int:
    """Whole calendar days between A and B (floor on the absolute delta)."""
    return duration_between(a, b).days


def weeks_between(a: TemporalLike, b: TemporalLike) -> int:
    """Whole 7-day periods between A and B."""
    return days_between(a, b) // 7


def _normalize_pair(a: TemporalLike, b: TemporalLike) -> tuple[datetime, datetime]:
    """Return (earlier, later) so callers can speak of forward differences."""
    da, db = _as_datetime(a), _as_datetime(b)
    return (da, db) if da <= db else (db, da)


def _last_day_of_month(year: int, month: int) -> int:
    return calendar.monthrange(year, month)[1]


def _clamp_day(year: int, month: int, day: int) -> int:
    """Clamp ``day`` to the last valid day of (year, month).

    Used so that "one month after Jan 31" lands on Feb 28/29 rather
    than overflowing into March.
    """
    return min(day, _last_day_of_month(year, month))


def months_between(a: TemporalLike, b: TemporalLike) -> int:
    """Whole calendar months between A and B (sign-stripped).

    A "whole month" is counted only once we cross the same day-of-month
    in the target month. Day-of-month is clamped to the target month's
    last valid day, so:

    * ``2024-01-31 → 2024-02-29`` ⇒ ``1`` (Feb 29 is "Jan 31's slot").
    * ``2024-01-31 → 2024-02-28`` ⇒ ``0`` in a leap year (we haven't
      reached the clamped target day yet — Feb 29 exists).
    * ``2023-01-31 → 2023-02-28`` ⇒ ``1`` in a non-leap year.

    Hours/minutes/seconds are honoured: ``2024-01-15 12:00 →
    2024-02-15 11:59`` is ``0`` months, ``... → 2024-02-15 12:00`` is
    ``1`` month.
    """
    earlier, later = _normalize_pair(a, b)

    months = (later.year - earlier.year) * 12 + (later.month - earlier.month)
    if months == 0:
        return 0

    # Build the candidate "anniversary" instant in the later year/month.
    anchor_year = earlier.year + (earlier.month - 1 + months) // 12
    anchor_month = (earlier.month - 1 + months) % 12 + 1
    anchor_day = _clamp_day(anchor_year, anchor_month, earlier.day)
    anchor = earlier.replace(
        year=anchor_year, month=anchor_month, day=anchor_day
    )
    if later < anchor:
        months -= 1
    return months


def years_between(a: TemporalLike, b: TemporalLike) -> int:
    """Whole calendar years (uses the same anniversary rule as months)."""
    earlier, later = _normalize_pair(a, b)
    years = later.year - earlier.year
    if years == 0:
        return 0

    anchor_day = _clamp_day(later.year, earlier.month, earlier.day)
    anchor = earlier.replace(year=later.year, month=earlier.month, day=anchor_day)
    if later < anchor:
        years -= 1
    return years


# --------------------------------------------------------------------- #
# Human-readable formatting                                             #
# --------------------------------------------------------------------- #
#
# Formatting is deliberately conservative: we surface at most two units
# (e.g. "1 month 4 days") because longer chains read like an MM:SS:MS
# stopwatch, not natural language. Sub-day precision drops to "less
# than a minute" rather than scientific notation.

_SECONDS_PER_DAY: Final[int] = 86400
_SECONDS_PER_HOUR: Final[int] = 3600
_SECONDS_PER_MINUTE: Final[int] = 60
_DAYS_PER_MONTH_AVG: Final[float] = 30.4375  # only for >365d → years/months split
_DAYS_PER_YEAR: Final[int] = 365


# Russian noun plural forms: one / few (2-4) / many (0, 5-20, ...)
# Selection rule per ru.wikipedia.org/wiki/Множественное_число:
#   if n % 100 in 11..14 → many
#   elif n % 10 == 1     → one
#   elif n % 10 in 2..4  → few
#   else                 → many
def _ru_plural(n: int, one: str, few: str, many: str) -> str:
    n = abs(n)
    mod100 = n % 100
    if 11 <= mod100 <= 14:
        return many
    mod10 = n % 10
    if mod10 == 1:
        return one
    if 2 <= mod10 <= 4:
        return few
    return many


_RU_FORMS: Final[dict[str, tuple[str, str, str]]] = {
    "year":   ("год",     "года",    "лет"),
    "month":  ("месяц",   "месяца",  "месяцев"),
    "week":   ("неделя",  "недели",  "недель"),
    "day":    ("день",    "дня",     "дней"),
    "hour":   ("час",     "часа",    "часов"),
    "minute": ("минута",  "минуты",  "минут"),
    "second": ("секунда", "секунды", "секунд"),
}


def _ru_unit(n: int, key: str) -> str:
    one, few, many = _RU_FORMS[key]
    return _ru_plural(n, one, few, many)


def _en_unit(n: int, key: str) -> str:
    return key if abs(n) == 1 else f"{key}s"


def _split_for_human(td: timedelta) -> list[tuple[int, str]]:
    """Break a timedelta into at most two coarsest non-zero units.

    Decision tree (longest unit wins; second unit only if it adds
    information):

    * ≥ 1 year  → ``years`` and ``months`` (calendar-approx)
    * ≥ 30 days → ``months`` and ``days`` (calendar-approx)
    * ≥ 7 days  → ``weeks`` and ``days``
    * ≥ 1 day   → ``days`` and ``hours``
    * ≥ 1 hour  → ``hours`` and ``minutes``
    * ≥ 1 min   → ``minutes`` only
    * else      → ``seconds`` (with a "less than a minute" floor at 0s)
    """
    total = abs(int(td.total_seconds()))
    if total == 0:
        return [(0, "second")]

    days = total // _SECONDS_PER_DAY
    rem_seconds = total - days * _SECONDS_PER_DAY

    if days >= _DAYS_PER_YEAR:
        # Approximate months/years from the day count: this branch is
        # only reachable from format_human when the caller passes a
        # raw timedelta (no calendar context). For calendar-aware
        # output, caller should hand us months_between/years_between
        # results upstream.
        years = days // _DAYS_PER_YEAR
        leftover_days = days - years * _DAYS_PER_YEAR
        months = int(leftover_days / _DAYS_PER_MONTH_AVG)
        parts: list[tuple[int, str]] = [(years, "year")]
        if months:
            parts.append((months, "month"))
        return parts

    if days >= 30:
        months = int(days / _DAYS_PER_MONTH_AVG)
        leftover_days = days - int(months * _DAYS_PER_MONTH_AVG)
        parts = [(months, "month")]
        if leftover_days:
            parts.append((leftover_days, "day"))
        return parts

    if days >= 7:
        weeks = days // 7
        leftover_days = days - weeks * 7
        parts = [(weeks, "week")]
        if leftover_days:
            parts.append((leftover_days, "day"))
        return parts

    if days >= 1:
        hours = rem_seconds // _SECONDS_PER_HOUR
        parts = [(days, "day")]
        if hours:
            parts.append((hours, "hour"))
        return parts

    hours = rem_seconds // _SECONDS_PER_HOUR
    if hours >= 1:
        minutes = (rem_seconds - hours * _SECONDS_PER_HOUR) // _SECONDS_PER_MINUTE
        parts = [(hours, "hour")]
        if minutes:
            parts.append((minutes, "minute"))
        return parts

    minutes = rem_seconds // _SECONDS_PER_MINUTE
    if minutes >= 1:
        return [(minutes, "minute")]

    return [(rem_seconds, "second")]


def format_human(td: timedelta, lang: str = "en") -> str:
    """Render a duration in compact natural language.

    * Picks at most two coarsest non-zero units.
    * Pluralisation: English regular -s; Russian three-form rule
      (``день / дня / дней``).
    * Negative durations are reported in absolute terms; if you need
      direction, format the sign yourself.
    * ``timedelta(0)`` → ``"0 seconds"`` / ``"0 секунд"``.
    """
    if lang not in ("en", "ru"):
        raise ValueError(f"Unsupported language: {lang!r}. Use 'en' or 'ru'.")

    parts = _split_for_human(td)
    rendered = []
    for value, key in parts:
        unit = _en_unit(value, key) if lang == "en" else _ru_unit(value, key)
        rendered.append(f"{value} {unit}")
    return " ".join(rendered)
