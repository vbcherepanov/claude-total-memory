"""Autofilter — sniff content and pick the right TOML filter automatically.

Called from save_knowledge when no explicit `filter=` was passed. Looks for
unambiguous CLI-output fingerprints. Falls back to `None` (no filter) for
normal knowledge so we don't accidentally mangle user text.
"""

from __future__ import annotations

import re

# Minimum content length before we bother sniffing — short text never wins.
MIN_LEN = 100

# Must exceed this density of matches (matches/lines) to classify.
_MIN_SIGNAL_RATIO = 0.15


# ────────────────────────────────────────────────────────────
# Fingerprints per filter. Tuple = (name, patterns[], min_hits).
# Order matters — first match wins. Specific filters before generic.
# ────────────────────────────────────────────────────────────
_SIGNATURES: list[tuple[str, list[re.Pattern[str]], int]] = [
    (
        "stack_trace",
        [
            re.compile(r"^Traceback \(most recent call last\)", re.MULTILINE),
            re.compile(r"^\s+File \"[^\"]+\", line \d+", re.MULTILINE),
            re.compile(r"^\w+(Error|Exception): ", re.MULTILINE),
            re.compile(r"^panic: ", re.MULTILINE),
            re.compile(r"^goroutine \d+ \[", re.MULTILINE),
            re.compile(r"^\s+at \S+\.\w+\(", re.MULTILINE),  # JS / Java
        ],
        2,
    ),
    (
        "pytest",
        [
            re.compile(r"test session starts", re.IGNORECASE),
            re.compile(r"^(FAILED|PASSED|ERROR)\b", re.MULTILINE),
            re.compile(r"::\w+ (PASSED|FAILED|ERROR)"),
            re.compile(r"\d+ failed,.*\d+ passed"),
            re.compile(r"AssertionError"),
            re.compile(r"^=+ ", re.MULTILINE),
        ],
        2,
    ),
    (
        "cargo",
        [
            re.compile(r"^\s+Compiling \S+ v\d", re.MULTILINE),
            re.compile(r"^\s+Finished (dev|release)", re.MULTILINE),
            re.compile(r"^error\[E\d+\]:", re.MULTILINE),
            re.compile(r"^warning: ", re.MULTILINE),
            re.compile(r"-->\s+\S+\.rs:\d+:\d+"),
        ],
        2,
    ),
    (
        "git_status",
        [
            re.compile(r"^On branch ", re.MULTILINE),
            re.compile(r"Your branch is (up to date|ahead|behind)"),
            re.compile(r"^(Changes (not staged|to be committed)|Untracked files):", re.MULTILINE),
            re.compile(r"^\s+(modified|new file|deleted):\s+", re.MULTILINE),
        ],
        2,
    ),
    (
        "docker_ps",
        [
            re.compile(r"^CONTAINER ID\s+IMAGE", re.MULTILINE),
            re.compile(r"\bUp \d+ (seconds|minutes|hours|days)"),
            re.compile(r"\bExited \(\d+\)"),
        ],
        2,
    ),
    (
        "npm_yarn",
        [
            re.compile(r"^npm (ERR|WARN|notice)", re.MULTILINE),
            re.compile(r"^yarn (install|add|run) ", re.MULTILINE),
            re.compile(r"^\[\d+/\d+\] ", re.MULTILINE),  # yarn progress
            re.compile(r"added \d+ packages? in"),
            re.compile(r"\d+ vulnerabilit"),
        ],
        2,
    ),
    (
        "http_log",
        [
            re.compile(r'"\w+ \S+ HTTP/[\d.]+" \d{3}'),         # combined log format
            re.compile(r"^\S+ - - \[\d+/\w+/\d{4}", re.MULTILINE),  # nginx/apache
            re.compile(r"\b(GET|POST|PUT|DELETE|PATCH) /\S+ HTTP/"),
        ],
        2,
    ),
    (
        "sql_explain",
        [
            re.compile(r"(Seq Scan|Index Scan|Index Only Scan|Bitmap Heap Scan|Hash Join|Nested Loop|Merge Join)"),
            re.compile(r"actual time=[\d.]+\.\.[\d.]+"),
            re.compile(r"cost=[\d.]+\.\.[\d.]+"),
            re.compile(r"(Planning Time|Execution Time): [\d.]+ ms"),
        ],
        2,
    ),
    (
        "json_blob",
        [
            # Big bracket-heavy structure
            re.compile(r'^\s*[{\[]', re.MULTILINE),
            re.compile(r'^\s*"[\w_-]+":\s*', re.MULTILINE),
            re.compile(r'^\s*[}\]],?\s*$', re.MULTILINE),
        ],
        3,
    ),
    (
        "markdown_doc",
        [
            re.compile(r"^#{1,3}\s+\S", re.MULTILINE),    # headings
            re.compile(r"^[-*]\s+\S", re.MULTILINE),      # bullets
            re.compile(r"```\w*\n"),                       # code fence
            re.compile(r"^\d+\.\s+\S", re.MULTILINE),      # numbered list
        ],
        3,
    ),
    (
        # Fallback: generic DEBUG/INFO/WARN/ERROR/FATAL noise.
        "generic_logs",
        [
            re.compile(r"^\s*(DEBUG|INFO|WARN|WARNING|ERROR|FATAL|TRACE)\b", re.MULTILINE),
            re.compile(r"^\s*\[(DEBUG|INFO|WARN|ERROR|FATAL)\]", re.MULTILINE),
            re.compile(r"^Traceback \(most recent call last\):", re.MULTILINE),
        ],
        3,
    ),
]


def _looks_like_markdown_doc(text: str) -> bool:
    """Avoid triggering on user text that merely quotes log fragments.

    Heuristic: substantial prose + fenced code blocks + inline backticks →
    this is documentation, leave it alone.
    """
    has_fence = "```" in text
    inline_code = text.count("`") - 2 * text.count("```")
    has_prose = bool(re.search(r"\b[A-Za-z]{4,}\s+[a-z]{3,}\s+[a-z]{3,}", text))
    return (has_fence or inline_code >= 4) and has_prose


def detect_filter(text: str) -> str | None:
    """Return the best-matching filter name, or None to skip filtering."""
    if not text or len(text) < MIN_LEN:
        return None
    # Don't apply log/CLI filters to documentation/conversational text.
    # markdown_doc filter handles that case explicitly.
    is_doc = _looks_like_markdown_doc(text)

    lines = text.count("\n") + 1

    for name, patterns, min_hits in _SIGNATURES:
        # markdown_doc only fires when content really looks like a doc AND is long.
        if name == "markdown_doc":
            if not is_doc or len(text) < 600:
                continue
        # Other CLI/log filters skipped on docs (they'd mangle examples).
        elif is_doc and name not in ("stack_trace",):
            continue

        # hits = count of DIFFERENT patterns that matched at least once
        # matches = total findall count across all patterns (for high-density signals)
        hits = 0
        matches = 0
        for p in patterns:
            found = p.findall(text)
            if found:
                hits += 1
                matches += len(found)
        # Density-based filters: generic_logs, http_log, json_blob, markdown_doc
        threshold_met = (
            matches >= min_hits if name in ("generic_logs", "http_log", "json_blob", "markdown_doc")
            else hits >= min_hits
        )
        if threshold_met:
            if lines < 5 and hits < min_hits + 1 and name not in ("generic_logs", "json_blob"):
                continue
            return name

    return None
