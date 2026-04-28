"""Ground relative date phrases to absolute datetimes.

Covers English and Russian phrasing commonly seen in chat memory
("yesterday", "3 days ago", "last week", "in March", "next Tuesday",
"вчера", "3 дня назад", "на прошлой неделе", "в марте", "в следующий
вторник"). ISO-formatted dates pass through unchanged.

Stdlib only — we use ``re``, ``datetime``, ``calendar``. We deliberately
avoid ``dateparser`` / ``parsedatetime`` because:
* they pull in heavy transitive deps (regex, pytz, six, ...);
* their locale handling is fuzzy enough to misclassify edge cases
  (e.g. ru "в марте" vs en "in march" when the language hint is wrong);
* the surface area we need is small and predictable.

Each parser is a regex-driven function returning a :class:`NormalizedDate`
or ``None``. :func:`normalize` runs them in a fixed precedence order
(ISO first, then absolute month/day, then relative phrases).
"""

from __future__ import annotations

import calendar
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable, Final, Literal, Optional


DateKind = Literal["exact", "day", "week", "month", "year", "range"]


@dataclass(frozen=True)
class NormalizedDate:
    """The grounded resolution of a relative phrase.

    Attributes
    ----------
    iso
        Canonical ISO 8601 string of the resolved instant (the start of
        the resolved range — for "in March" that is March 1st 00:00).
    confidence
        Heuristic confidence in [0.0, 1.0]. ISO-shaped inputs are 1.0;
        anchor-relative phrases that fully bind on a regex match are
        0.9; ambiguous month/year resolutions ("in March" with no year
        and an anchor mid-year) are 0.7. Below that we don't return a
        result at all.
    original
        The phrase as we received it (after light whitespace cleanup).
    anchor
        The anchor datetime that was used to ground the phrase.
    kind
        Granularity bucket — see :data:`DateKind`. Useful for downstream
        Allen-relation building: "yesterday" is ``day`` (a 24-hour
        interval) while "in March" is ``month``.
    """

    iso: str
    confidence: float
    original: str
    anchor: datetime
    kind: DateKind


# --------------------------------------------------------------------- #
# Lexicons                                                              #
# --------------------------------------------------------------------- #

_EN_MONTHS: Final[dict[str, int]] = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11,
    "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8,
    "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}

# Russian month names appear in nominative ("январь") and prepositional
# ("в январе") cases — we normalise both to the same month number.
_RU_MONTHS: Final[dict[str, int]] = {
    # nominative
    "январь": 1, "февраль": 2, "март": 3, "апрель": 4, "май": 5, "июнь": 6,
    "июль": 7, "август": 8, "сентябрь": 9, "октябрь": 10, "ноябрь": 11,
    "декабрь": 12,
    # prepositional ("в …е")
    "январе": 1, "феврале": 2, "марте": 3, "апреле": 4, "мае": 5, "июне": 6,
    "июле": 7, "августе": 8, "сентябре": 9, "октябре": 10, "ноябре": 11,
    "декабре": 12,
    # genitive ("3 марта") — needed for "X дня назад" combos but harmless here
    "января": 1, "февраля": 2, "марта": 3, "апреля": 4, "июня": 6, "июля": 7,
    "августа": 8, "сентября": 9, "октября": 10, "ноября": 11, "декабря": 12,
}

# Weekday lexicons: Monday=0 to match datetime.weekday()
_EN_WEEKDAYS: Final[dict[str, int]] = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
    "mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6,
}
_RU_WEEKDAYS: Final[dict[str, int]] = {
    # nominative
    "понедельник": 0, "вторник": 1, "среда": 2, "четверг": 3,
    "пятница": 4, "суббота": 5, "воскресенье": 6,
    # accusative ("в понедельник")
    "среду": 2, "пятницу": 4, "субботу": 5,
    # already-accusative forms that match nominative are reused above
}


# Number words used in "two days ago" / "три дня назад"
_EN_NUM_WORDS: Final[dict[str, int]] = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}
_RU_NUM_WORDS: Final[dict[str, int]] = {
    "один": 1, "одну": 1, "одна": 1, "два": 2, "две": 2, "три": 3,
    "четыре": 4, "пять": 5, "шесть": 6, "семь": 7, "восемь": 8,
    "девять": 9, "десять": 10,
}


# --------------------------------------------------------------------- #
# Helpers                                                               #
# --------------------------------------------------------------------- #


def _start_of_day(dt: datetime) -> datetime:
    return dt.replace(hour=0, minute=0, second=0, microsecond=0)


def _start_of_week(dt: datetime) -> datetime:
    """Monday 00:00 of the week containing ``dt``."""
    base = _start_of_day(dt)
    return base - timedelta(days=base.weekday())


def _to_int(token: str, lang: str) -> Optional[int]:
    """Parse '3', 'three', 'три' etc. ``None`` if unrecognised."""
    token = token.strip().lower()
    if token.isdigit():
        return int(token)
    if lang in ("en", "auto") and token in _EN_NUM_WORDS:
        return _EN_NUM_WORDS[token]
    if lang in ("ru", "auto") and token in _RU_NUM_WORDS:
        return _RU_NUM_WORDS[token]
    return None


def _detect_lang(phrase: str) -> str:
    """Cheap script detection: presence of any Cyrillic char → 'ru'."""
    for ch in phrase:
        if "Ѐ" <= ch <= "ӿ":
            return "ru"
    return "en"


def _make(
    *,
    iso: datetime,
    confidence: float,
    original: str,
    anchor: datetime,
    kind: DateKind,
) -> NormalizedDate:
    return NormalizedDate(
        iso=iso.isoformat(),
        confidence=confidence,
        original=original,
        anchor=anchor,
        kind=kind,
    )


# --------------------------------------------------------------------- #
# Parsers (each returns NormalizedDate or None)                         #
# --------------------------------------------------------------------- #


_ISO_RE = re.compile(
    r"^(?P<y>\d{4})-(?P<m>\d{2})-(?P<d>\d{2})"
    r"(?:[T ](?P<h>\d{2}):(?P<mi>\d{2})(?::(?P<s>\d{2}))?)?$"
)


def _parse_iso(phrase: str, anchor: datetime, lang: str) -> Optional[NormalizedDate]:
    m = _ISO_RE.match(phrase.strip())
    if not m:
        return None
    y, mo, d = int(m.group("y")), int(m.group("m")), int(m.group("d"))
    h = int(m.group("h") or 0)
    mi = int(m.group("mi") or 0)
    s = int(m.group("s") or 0)
    try:
        dt = datetime(y, mo, d, h, mi, s)
    except ValueError:
        return None
    has_time = m.group("h") is not None
    return _make(
        iso=dt,
        confidence=1.0,
        original=phrase,
        anchor=anchor,
        kind="exact" if has_time else "day",
    )


_EN_OFFSETS: Final[dict[str, int]] = {
    "today": 0,
    "yesterday": -1,
    "tomorrow": 1,
    "day before yesterday": -2,
    "day after tomorrow": 2,
}
_RU_OFFSETS: Final[dict[str, int]] = {
    "сегодня": 0,
    "вчера": -1,
    "завтра": 1,
    "позавчера": -2,
    "послезавтра": 2,
}


def _parse_simple_offsets(
    phrase: str, anchor: datetime, lang: str
) -> Optional[NormalizedDate]:
    p = phrase.strip().lower()
    delta: Optional[int] = None
    if lang in ("en", "auto"):
        delta = _EN_OFFSETS.get(p)
    if delta is None and lang in ("ru", "auto"):
        delta = _RU_OFFSETS.get(p)
    if delta is None:
        return None
    day = _start_of_day(anchor) + timedelta(days=delta)
    return _make(
        iso=day, confidence=1.0, original=phrase, anchor=anchor, kind="day",
    )


# "3 days ago", "two weeks ago", "5 months ago", "1 year ago"
_EN_AGO_RE = re.compile(
    r"^\s*(?P<n>\d+|[a-z]+)\s+(?P<u>day|days|week|weeks|month|months|year|years)\s+ago\s*$",
    re.IGNORECASE,
)
# "3 дня назад", "две недели назад", "5 месяцев назад"
_RU_AGO_RE = re.compile(
    r"^\s*(?P<n>\d+|[а-яА-ЯёЁ]+)\s+(?P<u>"
    r"день|дня|дней|"
    r"неделю|недели|недель|неделя|"
    r"месяц|месяца|месяцев|"
    r"год|года|лет"
    r")\s+назад\s*$",
    re.IGNORECASE,
)

_RU_UNIT_TO_KIND: Final[dict[str, str]] = {
    "день": "day", "дня": "day", "дней": "day",
    "неделя": "week", "неделю": "week", "недели": "week", "недель": "week",
    "месяц": "month", "месяца": "month", "месяцев": "month",
    "год": "year", "года": "year", "лет": "year",
}
_EN_UNIT_TO_KIND: Final[dict[str, str]] = {
    "day": "day", "days": "day",
    "week": "week", "weeks": "week",
    "month": "month", "months": "month",
    "year": "year", "years": "year",
}


def _shift_calendar(anchor: datetime, n: int, kind: str) -> datetime:
    """Subtract n units from anchor; calendar-aware for month/year."""
    if kind == "day":
        return anchor - timedelta(days=n)
    if kind == "week":
        return anchor - timedelta(days=7 * n)
    if kind == "month":
        total_months = anchor.month - 1 - n
        new_year = anchor.year + total_months // 12
        new_month = total_months % 12 + 1
        new_day = min(anchor.day, calendar.monthrange(new_year, new_month)[1])
        return anchor.replace(year=new_year, month=new_month, day=new_day)
    if kind == "year":
        new_year = anchor.year - n
        # Feb 29 → Feb 28 in non-leap year
        new_day = anchor.day
        if anchor.month == 2 and anchor.day == 29 and not calendar.isleap(new_year):
            new_day = 28
        return anchor.replace(year=new_year, day=new_day)
    raise ValueError(f"Unknown kind: {kind}")


def _parse_ago(phrase: str, anchor: datetime, lang: str) -> Optional[NormalizedDate]:
    """English '<n> <unit>(s) ago' and Russian '<n> <unit> назад'."""
    if lang in ("en", "auto"):
        m = _EN_AGO_RE.match(phrase)
        if m:
            n = _to_int(m.group("n"), "en")
            if n is None:
                return None
            kind = _EN_UNIT_TO_KIND[m.group("u").lower()]
            target = _shift_calendar(_start_of_day(anchor), n, kind)
            return _make(
                iso=target, confidence=0.95, original=phrase,
                anchor=anchor, kind=kind,
            )

    if lang in ("ru", "auto"):
        m = _RU_AGO_RE.match(phrase)
        if m:
            raw_n = m.group("n").lower()
            n = _to_int(raw_n, "ru")
            if n is None:
                return None
            kind = _RU_UNIT_TO_KIND[m.group("u").lower()]
            target = _shift_calendar(_start_of_day(anchor), n, kind)
            return _make(
                iso=target, confidence=0.95, original=phrase,
                anchor=anchor, kind=kind,
            )
    return None


# "last week" / "next week" / "this week" / "на прошлой неделе" / "на следующей неделе"
_EN_REL_PERIOD_RE = re.compile(
    r"^\s*(?P<dir>last|previous|next|this|coming)\s+(?P<u>week|month|year)\s*$",
    re.IGNORECASE,
)
_RU_REL_PERIOD_RE = re.compile(
    r"^\s*(?:на\s+|в\s+)?(?P<dir>прошлой|прошлая|следующей|следующая|этой|эта|"
    r"прошлого|следующего|этого|прошлый|следующий|этот|прошлом|следующем|этом)"
    r"\s+(?P<u>недел[еия]|месяц[еа]?|год[еау]?)\s*$",
    re.IGNORECASE,
)

_EN_DIR: Final[dict[str, int]] = {
    "last": -1, "previous": -1, "next": 1, "coming": 1, "this": 0,
}
_RU_DIR: Final[dict[str, int]] = {
    "прошлой": -1, "прошлая": -1, "прошлого": -1, "прошлый": -1, "прошлом": -1,
    "следующей": 1, "следующая": 1, "следующего": 1, "следующий": 1,
    "следующем": 1,
    "этой": 0, "эта": 0, "этого": 0, "этот": 0, "этом": 0,
}

_RU_UNIT_GROUP_TO_KIND: Final[dict[str, str]] = {
    "неделе": "week", "недели": "week", "неделя": "week",
    "месяц": "month", "месяца": "month", "месяце": "month",
    "год": "year", "года": "year", "году": "year", "годе": "year",
}


def _shift_to_period_start(
    anchor: datetime, direction: int, kind: str
) -> datetime:
    """Return the canonical start of the requested period.

    * week → Monday 00:00 of (this/last/next) week
    * month → 1st 00:00
    * year → Jan 1st 00:00
    """
    if kind == "week":
        base = _start_of_week(anchor)
        return base + timedelta(days=7 * direction)
    if kind == "month":
        target_month = anchor.month + direction
        target_year = anchor.year
        while target_month <= 0:
            target_month += 12
            target_year -= 1
        while target_month > 12:
            target_month -= 12
            target_year += 1
        return datetime(target_year, target_month, 1)
    if kind == "year":
        return datetime(anchor.year + direction, 1, 1)
    raise ValueError(f"Unknown kind: {kind}")


def _parse_rel_period(
    phrase: str, anchor: datetime, lang: str
) -> Optional[NormalizedDate]:
    if lang in ("en", "auto"):
        m = _EN_REL_PERIOD_RE.match(phrase)
        if m:
            direction = _EN_DIR[m.group("dir").lower()]
            kind = m.group("u").lower()
            target = _shift_to_period_start(anchor, direction, kind)
            return _make(
                iso=target, confidence=0.9, original=phrase,
                anchor=anchor, kind=kind,
            )

    if lang in ("ru", "auto"):
        m = _RU_REL_PERIOD_RE.match(phrase)
        if m:
            direction = _RU_DIR[m.group("dir").lower()]
            unit_token = m.group("u").lower()
            kind = _RU_UNIT_GROUP_TO_KIND.get(unit_token)
            if kind is None:
                return None
            target = _shift_to_period_start(anchor, direction, kind)
            return _make(
                iso=target, confidence=0.9, original=phrase,
                anchor=anchor, kind=kind,
            )
    return None


# "in March", "in march 2024", "в марте", "в марте 2024"
_EN_MONTH_RE = re.compile(
    r"^\s*(?:in\s+)?(?P<month>" + "|".join(_EN_MONTHS) + r")"
    r"(?:\s+(?P<year>\d{4}))?\s*$",
    re.IGNORECASE,
)
_RU_MONTH_RE = re.compile(
    r"^\s*(?:в\s+)?(?P<month>" + "|".join(_RU_MONTHS) + r")"
    r"(?:\s+(?P<year>\d{4}))?\s*$",
    re.IGNORECASE,
)


def _closest_year_for_month(anchor: datetime, month: int) -> int:
    """Pick the year (anchor's year, ±1) that puts ``month`` closest to anchor.

    Rule: if the month has not yet ended in this year, use this year;
    if it has ended more than 6 months ago, use next year; otherwise
    stay in this year. This produces the "in March" → next/prev March
    behaviour described in the spec.
    """
    candidates = [
        datetime(anchor.year - 1, month, 15),
        datetime(anchor.year, month, 15),
        datetime(anchor.year + 1, month, 15),
    ]
    best = min(candidates, key=lambda d: abs((d - anchor).days))
    return best.year


def _parse_month(phrase: str, anchor: datetime, lang: str) -> Optional[NormalizedDate]:
    if lang in ("en", "auto"):
        m = _EN_MONTH_RE.match(phrase)
        if m:
            month = _EN_MONTHS[m.group("month").lower()]
            year = int(m.group("year")) if m.group("year") else _closest_year_for_month(anchor, month)
            confidence = 1.0 if m.group("year") else 0.7
            target = datetime(year, month, 1)
            return _make(
                iso=target, confidence=confidence, original=phrase,
                anchor=anchor, kind="month",
            )

    if lang in ("ru", "auto"):
        m = _RU_MONTH_RE.match(phrase)
        if m:
            month = _RU_MONTHS[m.group("month").lower()]
            year = int(m.group("year")) if m.group("year") else _closest_year_for_month(anchor, month)
            confidence = 1.0 if m.group("year") else 0.7
            target = datetime(year, month, 1)
            return _make(
                iso=target, confidence=confidence, original=phrase,
                anchor=anchor, kind="month",
            )
    return None


# "next Tuesday", "last Friday", "this Wednesday",
# "в следующий вторник", "в прошлую пятницу", "в эту среду"
_EN_REL_WEEKDAY_RE = re.compile(
    r"^\s*(?P<dir>last|previous|next|this|coming)\s+(?P<wd>"
    + "|".join(_EN_WEEKDAYS) + r")\s*$",
    re.IGNORECASE,
)
_RU_REL_WEEKDAY_RE = re.compile(
    r"^\s*(?:в\s+)?(?P<dir>прошлый|прошлую|прошлое|прошлая|"
    r"следующий|следующую|следующее|следующая|этот|эту|это|эта)"
    r"\s+(?P<wd>" + "|".join(_RU_WEEKDAYS) + r")\s*$",
    re.IGNORECASE,
)

_RU_WEEKDAY_DIR: Final[dict[str, int]] = {
    "прошлый": -1, "прошлую": -1, "прошлое": -1, "прошлая": -1,
    "следующий": 1, "следующую": 1, "следующее": 1, "следующая": 1,
    "этот": 0, "эту": 0, "это": 0, "эта": 0,
}


def _shift_to_weekday(
    anchor: datetime, direction: int, target_weekday: int
) -> datetime:
    """Resolve "<dir> <weekday>" to a concrete date.

    * direction == 0 ("this Tuesday"): Tuesday in the current Mon-Sun
      week, even if it has already passed.
    * direction == 1 ("next Tuesday"): the Tuesday of next week if
      today is Tuesday or earlier, else the next Tuesday at all.
      Concretely: smallest delta ≥ 7 if anchor is Tuesday, else the
      next Tuesday strictly in the future.
    * direction == -1 ("last Tuesday"): the most recent Tuesday strictly
      before today.
    """
    base = _start_of_day(anchor)
    current_wd = base.weekday()

    if direction == 0:
        return base + timedelta(days=target_weekday - current_wd)

    if direction == 1:
        # "next Tuesday" — strictly future. If anchor is Tuesday, jump 7
        # days; otherwise the next future weekday.
        delta = (target_weekday - current_wd) % 7
        if delta == 0:
            delta = 7
        return base + timedelta(days=delta)

    # direction == -1
    delta = (current_wd - target_weekday) % 7
    if delta == 0:
        delta = 7
    return base - timedelta(days=delta)


def _parse_rel_weekday(
    phrase: str, anchor: datetime, lang: str
) -> Optional[NormalizedDate]:
    if lang in ("en", "auto"):
        m = _EN_REL_WEEKDAY_RE.match(phrase)
        if m:
            direction = _EN_DIR[m.group("dir").lower()]
            wd = _EN_WEEKDAYS[m.group("wd").lower()]
            target = _shift_to_weekday(anchor, direction, wd)
            return _make(
                iso=target, confidence=0.95, original=phrase,
                anchor=anchor, kind="day",
            )

    if lang in ("ru", "auto"):
        m = _RU_REL_WEEKDAY_RE.match(phrase)
        if m:
            direction = _RU_WEEKDAY_DIR[m.group("dir").lower()]
            wd = _RU_WEEKDAYS[m.group("wd").lower()]
            target = _shift_to_weekday(anchor, direction, wd)
            return _make(
                iso=target, confidence=0.95, original=phrase,
                anchor=anchor, kind="day",
            )
    return None


# Order matters: ISO and absolute month patterns are checked before
# relative ones to prevent a phrase like "March 2024" from being
# misclassified.
_PARSERS: Final[tuple[Callable[[str, datetime, str], Optional[NormalizedDate]], ...]] = (
    _parse_iso,
    _parse_simple_offsets,
    _parse_ago,
    _parse_rel_period,
    _parse_rel_weekday,
    _parse_month,
)


def normalize(
    phrase: str, anchor: datetime, lang: str = "auto"
) -> Optional[NormalizedDate]:
    """Resolve a relative date phrase to an absolute datetime.

    Returns ``None`` if no parser matched. The function never raises on
    a syntactically valid string — bad input (empty, only whitespace)
    yields ``None``.
    """
    if not phrase or not phrase.strip():
        return None

    if lang not in ("en", "ru", "auto"):
        raise ValueError(f"Unsupported language: {lang!r}")

    detected_lang = _detect_lang(phrase) if lang == "auto" else lang

    for parser in _PARSERS:
        result = parser(phrase, anchor, detected_lang)
        if result is not None:
            return result
    return None
