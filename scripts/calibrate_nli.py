"""W5 — NLI verifier calibration CLI.

Two modes:

* ``--build-fixture``
    Synthesise NLI triples from LoCoMo dialogues (deterministic, seed=42)
    and dump them to a JSON file. Each triple is
    ``(premise, hypothesis, gold_label, kind)``.

* ``--tune``
    Load the fixture, run candidate NLI models, sweep thresholds, pick the
    best (lowest false-contradict subject to recall constraints), write
    the chosen config to ``--output`` and an optional markdown report.

The script intentionally has no dependency on ``memory_core`` and only
imports ``ai_layer.verifier`` for the underlying model + label resolution.
That keeps the v11 layer wall (memory_core ↛ ai_layer) intact.

Determinism notes
-----------------
* ``random.Random(42)`` is the only RNG.
* Hypothesis corruption rules are bucketed by content (numeric / named
  entity / phrase) and one rule per QA — documented in ``rule`` field.
* Fixture md5 is stable as long as the LoCoMo source is untouched.

Threshold tuning
----------------
Search space:
    p_contradict ∈ [0.30, 0.95] step 0.05
    p_entail     ∈ [0.30, 0.90] step 0.05
    margin       ∈ [0.00, 0.40] step 0.05
That's ~14 × 13 × 9 = ~1640 candidates. Cheap given probs are precomputed.

Selection criterion (in order):
    1) hard constraint: contradict_recall >= 0.80 AND entail_recall >= 0.75
    2) minimise false_contradict_rate (false-positive on non-contradict)
    3) tie-break: maximise balanced accuracy

If no point meets the constraints, fall back to maximising balanced
accuracy and emit a clear warning in the report.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import hashlib
import json
import os
import random
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

# Make src/ importable when run directly.
_REPO = Path(__file__).resolve().parent.parent
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


_LOCOMO_PATH = _REPO / "benchmarks" / "data" / "locomo" / "data" / "locomo10.json"

# Default candidate model list for Path B. The first entry MUST be the
# current production model — Path A (threshold tuning on the existing
# model) is always tried.
_PRODUCTION_MODEL = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
_PATH_B_CANDIDATES: list[str] = [
    "cross-encoder/nli-deberta-v3-base",
    "microsoft/deberta-v3-base-tasksource-nli",
]


# ────────────────────────────────────────────────────────────────────
# Fixture types
# ────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class NLITriple:
    premise: str
    hypothesis: str
    gold_label: str  # entail | neutral | contradict
    kind: str        # entail | contradict_numeric | contradict_entity | contradict_negate | neutral_unrelated
    rule: str        # short, deterministic description for traceability
    source_qa_idx: int
    source_conv_idx: int
    source_dia_id: str | None

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


# ────────────────────────────────────────────────────────────────────
# LoCoMo loaders
# ────────────────────────────────────────────────────────────────────


def _load_locomo(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list) or not data:
        raise RuntimeError(f"unexpected LoCoMo payload at {path}")
    return data


def _index_dialog_by_id(conv: dict) -> dict[str, dict]:
    """Build {dia_id: {speaker, text, session}} for one conversation."""
    idx: dict[str, dict] = {}
    conv_root = conv.get("conversation", {})
    for k, v in conv_root.items():
        if not k.startswith("session_") or k.endswith("_date_time"):
            continue
        if not isinstance(v, list):
            continue
        for turn in v:
            dia_id = turn.get("dia_id")
            if not dia_id:
                continue
            idx[dia_id] = {
                "speaker": turn.get("speaker", ""),
                "text": turn.get("text", ""),
                "session": k,
            }
    return idx


def _all_dia_ids_for(conv: dict) -> list[str]:
    return list(_index_dialog_by_id(conv).keys())


_ENGLISH_STOP_CAPS = {
    "I", "The", "He", "She", "They", "We", "You", "It",
    "Yeah", "Yes", "No", "Oh", "Hey", "Hi", "Wow", "Well", "So",
    "Actually", "Right", "OK", "Okay", "Sure", "Maybe", "Now",
    "When", "Where", "What", "How", "Why", "Who", "Whose",
    "Today", "Yesterday", "Tomorrow", "Tonight", "Lately",
    "And", "But", "Or", "Also", "Still", "Just", "Then",
    "Honestly", "Definitely", "Really", "Truly",
    "Mom", "Dad", "Mum", "Mommy", "Daddy",
    "God", "Lord",
}


def _named_entities_in_conv(conv: dict) -> list[str]:
    """Speakers + mid-sentence capitalised tokens that look like proper nouns.

    To avoid grabbing sentence-initial tokens (which produce garbage swaps),
    we only collect tokens that are NOT the first word of their sentence
    and NOT in a stop-list of common English starts.
    """
    candidates: set[str] = set()
    for k in ("speaker_a", "speaker_b"):
        v = conv.get("conversation", {}).get(k)
        if isinstance(v, str) and v.strip():
            candidates.add(v.strip())

    cap_re = re.compile(r"(?<=[\w\s,])\b([A-Z][a-zA-Z]{2,})\b")
    sent_split = re.compile(r"[.!?]\s+|^")
    for turn in _index_dialog_by_id(conv).values():
        text = turn["text"]
        # Mark every position that begins a sentence so we skip those tokens.
        sent_starts: set[int] = {0}
        for m in sent_split.finditer(text):
            sent_starts.add(m.end())
        for m in cap_re.finditer(text):
            tok = m.group(1)
            if tok in _ENGLISH_STOP_CAPS:
                continue
            # Skip if this token sits at the start of a sentence (those are
            # capitalised by convention, not because they're proper nouns).
            if m.start(1) in sent_starts:
                continue
            candidates.add(tok)
    # Also seed the conversation's two speaker names if they aren't there yet
    # (already done above).
    return sorted(candidates)


# ────────────────────────────────────────────────────────────────────
# Hypothesis corruption rules (deterministic)
# ────────────────────────────────────────────────────────────────────


_NUM_RE = re.compile(r"\b(\d+)\b")
_YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")
_NEG_PREFIX = "It is not the case that "


def _flip_number(
    answer: str,
    premise: str,
    rng: random.Random,
) -> tuple[str, str] | None:
    """Replace a number/year in ``answer`` that is also present in ``premise``.

    We only generate a contradict pair when the *same* numeric token appears
    in both — that way the corrupted hypothesis genuinely contradicts the
    premise (not just makes a numeric claim about something the premise
    never mentioned, which would be neutral, not contradict).
    """
    # Years first (more anchored).
    for m in _YEAR_RE.finditer(answer):
        original = m.group(1)
        if original in premise:
            delta = rng.choice([-7, -5, -3, 3, 5, 7])
            new_val = str(int(original) + delta)
            if new_val == original:
                new_val = str(int(original) + 4)
            new_text = answer[: m.start()] + new_val + answer[m.end():]
            return new_text, f"year-flip {original}->{new_val}"

    for m in _NUM_RE.finditer(answer):
        original = m.group(1)
        if original in premise:
            n = int(original)
            new_n = n + rng.choice([2, 3, 4, 5, 7])
            if new_n == n:
                new_n = n + 1
            new_text = answer[: m.start()] + str(new_n) + answer[m.end():]
            return new_text, f"num-flip {original}->{new_n}"
    return None


def _swap_entity(
    answer: str,
    premise: str,
    candidates: list[str],
    rng: random.Random,
) -> tuple[str, str] | None:
    """Swap an entity that appears in BOTH ``answer`` and ``premise``.

    Both anchors required so that the corrupted hypothesis genuinely
    contradicts the premise. We refuse to swap a sentence-initial token
    of either string (those are capitalised by convention, not because
    they're proper nouns) — that prevents nonsense rewrites like
    ``"Had a tough time"`` → ``"Connecting a tough time"``.
    """
    if "," in answer or " and " in answer:
        return None

    # Indices of sentence-starts in the answer; skip swaps that would
    # replace a capitalised-by-convention leading word.
    sent_starts_ans: set[int] = {0}
    for m in re.finditer(r"[.!?]\s+", answer):
        sent_starts_ans.add(m.end())

    for ent in candidates:
        idx = answer.find(ent)
        if idx < 0:
            continue
        if idx in sent_starts_ans:
            continue
        left_ok = idx == 0 or not answer[idx - 1].isalnum()
        right_ok = (
            idx + len(ent) == len(answer) or not answer[idx + len(ent)].isalnum()
        )
        if not (left_ok and right_ok):
            continue
        if ent not in premise:
            continue
        # Replacement must be a *different* word, not also in the strings,
        # and ideally a proper-noun-like token (capitalised, length ≥ 3).
        alt_pool = [
            c
            for c in candidates
            if c != ent
            and c not in answer
            and c not in premise
            and len(c) >= 3
            and c[0].isupper()
        ]
        if not alt_pool:
            continue
        replacement = rng.choice(alt_pool)
        new_text = answer[:idx] + replacement + answer[idx + len(ent):]
        return new_text, f"entity-swap {ent}->{replacement}"
    return None


_NEG_AUX_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b(\w+) is\b"), r"\1 is not"),
    (re.compile(r"\b(\w+) was\b"), r"\1 was not"),
    (re.compile(r"\b(\w+) are\b"), r"\1 are not"),
    (re.compile(r"\b(\w+) were\b"), r"\1 were not"),
    (re.compile(r"\b(\w+) has\b"), r"\1 has not"),
    (re.compile(r"\b(\w+) have\b"), r"\1 have not"),
    (re.compile(r"\b(\w+) had\b"), r"\1 had not"),
    (re.compile(r"\b(\w+) will\b"), r"\1 will not"),
    (re.compile(r"\b(\w+) would\b"), r"\1 would not"),
    (re.compile(r"\b(\w+) does\b"), r"\1 does not"),
    (re.compile(r"\b(\w+) did\b"), r"\1 did not"),
    (re.compile(r"\b(\w+) can\b"), r"\1 cannot"),
    (re.compile(r"\b(\w+) likes\b"), r"\1 does not like"),
    (re.compile(r"\b(\w+) loves\b"), r"\1 does not love"),
    (re.compile(r"\b(\w+) wants\b"), r"\1 does not want"),
    (re.compile(r"\b(\w+) needs\b"), r"\1 does not need"),
    (re.compile(r"\b(\w+) thinks\b"), r"\1 does not think"),
    (re.compile(r"\b(\w+) went\b"), r"\1 did not go"),
    (re.compile(r"\b(\w+) got\b"), r"\1 did not get"),
    (re.compile(r"\b(\w+) started\b"), r"\1 did not start"),
    (re.compile(r"\b(\w+) moved\b"), r"\1 did not move"),
]


def _negate(claim: str) -> tuple[str, str] | None:
    """Negate a third-person declarative claim by inserting ``not`` after
    the first matching auxiliary/verb.

    The naive "Definitely not: <claim>" prefix is *not* recognised by the
    NLI model as a polarity flip (it reads as a header). Direct in-sentence
    insertion of ``not`` works robustly. We only apply the FIRST match so
    that already-negated claims stay valid.
    """
    s = claim.strip()
    if not s:
        return None
    lower = s.lower()
    if " not " in lower or " never " in lower or "n't " in lower:
        # Already negative; double-negation is unreliable.
        return None
    for pat, rep in _NEG_AUX_PATTERNS:
        m = pat.search(s)
        if not m:
            continue
        new_s = s[: m.start()] + pat.sub(rep, s[m.start():], count=1)
        if new_s != s:
            return new_s, f"insert-not-after `{m.group(0)}`"
    # Fallback: prepend "It is not true that " — works because XNLI
    # has plenty of "It is not true that …" examples.
    return f"It is not true that {s[0].lower() + s[1:]}", "it-is-not-true-prefix"


def _normalise_answer(value) -> str:
    """LoCoMo answers can be int / float / list / str. Coerce to clean str."""
    if isinstance(value, list):
        return ", ".join(str(v).strip() for v in value if str(v).strip())
    return str(value).strip()


# ────────────────────────────────────────────────────────────────────
# Fixture builder
# ────────────────────────────────────────────────────────────────────


@dataclass
class FixtureStats:
    entail: int = 0
    contradict_numeric: int = 0
    contradict_entity: int = 0
    contradict_negate: int = 0
    neutral_unrelated: int = 0
    neutral_idk: int = 0  # adversarial — IDK-style answers vs real dialogue

    def total(self) -> int:
        return (
            self.entail
            + self.contradict_numeric
            + self.contradict_entity
            + self.contradict_negate
            + self.neutral_unrelated
            + self.neutral_idk
        )

    def to_dict(self) -> dict:
        return {
            "entail": self.entail,
            "contradict_numeric": self.contradict_numeric,
            "contradict_entity": self.contradict_entity,
            "contradict_negate": self.contradict_negate,
            "neutral_unrelated": self.neutral_unrelated,
            "neutral_idk": self.neutral_idk,
            "total": self.total(),
        }


_FIRST_PERSON_RE = re.compile(r"\b(I|I'm|I've|I'll|I'd|me|my|mine)\b")
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _is_question(s: str) -> bool:
    s = s.strip()
    return s.endswith("?") or s.lower().startswith(
        ("what ", "when ", "where ", "who ", "why ", "how ", "do ", "did ", "are ", "is ", "can ", "could ")
    )


def _to_third_person(speaker: str, sentence: str) -> str:
    """Convert a first-person dialogue clause to third-person.

    Cheap rule-based rewrite that doesn't try to be perfect — it just needs
    to produce a declarative paraphrase the NLI model can recognise as
    entailed by the original.
    """
    s = sentence.strip()
    # Strip leading interjections ("Yeah, ", "Yes! ", "Hey Mel, ").
    s = re.sub(r"^(Yeah|Yes|No|Oh|Hey|Hi|Wow|Well|So|Actually|Right)[,!.\s]+", "", s, count=1)
    # Cheap pronoun/verb rewrites.
    rules = [
        (r"\bI'm\b", f"{speaker} is"),
        (r"\bI've\b", f"{speaker} has"),
        (r"\bI'll\b", f"{speaker} will"),
        (r"\bI'd\b", f"{speaker} would"),
        (r"\bI am\b", f"{speaker} is"),
        (r"\bI was\b", f"{speaker} was"),
        (r"\bI have\b", f"{speaker} has"),
        (r"\bI had\b", f"{speaker} had"),
        (r"\bI will\b", f"{speaker} will"),
        (r"\bI would\b", f"{speaker} would"),
        (r"\bI did\b", f"{speaker} did"),
        (r"\bI do\b", f"{speaker} does"),
        (r"\bI like\b", f"{speaker} likes"),
        (r"\bI love\b", f"{speaker} loves"),
        (r"\bI started\b", f"{speaker} started"),
        (r"\bI went\b", f"{speaker} went"),
        (r"\bI got\b", f"{speaker} got"),
        (r"\bI moved\b", f"{speaker} moved"),
        (r"\bI live\b", f"{speaker} lives"),
        (r"\bI work\b", f"{speaker} works"),
        (r"\bI think\b", f"{speaker} thinks"),
        (r"\bI need\b", f"{speaker} needs"),
        (r"\bI want\b", f"{speaker} wants"),
        (r"\bI \b", f"{speaker} "),
        (r"\bmy\b", f"{speaker}'s"),
        (r"\bme\b", speaker),
        (r"\bmine\b", f"{speaker}'s"),
    ]
    for pat, rep in rules:
        s = re.sub(pat, rep, s)
    # Capitalise leading char.
    if s and s[0].islower():
        s = s[0].upper() + s[1:]
    # Ensure final punctuation.
    if s and s[-1] not in ".!?":
        s += "."
    return s


def _select_premise_clauses(text: str) -> list[str]:
    """Pick declarative-looking clauses with a minimum length."""
    sents = [s.strip() for s in _SENT_SPLIT_RE.split(text) if s.strip()]
    out = []
    for s in sents:
        if _is_question(s):
            continue
        if len(s) < 28 or len(s) > 220:
            continue
        out.append(s)
    return out


def _has_anchor_for_corruption(clause: str) -> str | None:
    """Detect what corruption rule(s) this clause supports.

    Returns 'numeric' / 'entity' / 'negate' or None.
    """
    if _YEAR_RE.search(clause) or _NUM_RE.search(clause):
        return "numeric"
    cap_re = re.compile(r"\b([A-Z][a-zA-Z]{2,})\b")
    nontrivial = [
        m.group(1) for m in cap_re.finditer(clause)
        if m.group(1) not in {"I", "The", "He", "She", "They", "We", "You", "It", "Yeah", "Yes", "No", "Oh"}
    ]
    if nontrivial:
        return "entity"
    return "negate"  # negation always works


def build_fixture(
    *,
    seed: int,
    target_entail: int,
    target_contradict: int,
    target_neutral: int,
    locomo_path: Path = _LOCOMO_PATH,
) -> tuple[list[NLITriple], FixtureStats]:
    """Generate the labelled NLI calibration set deterministically.

    Strategy
    --------
    Instead of using LoCoMo QA pairs (whose gold answers often require
    multi-turn reasoning and are NOT directly entailed by a single
    evidence turn), we synthesise NLI triples directly from dialogue
    turns:

    * ENTAIL — premise is a dialogue turn containing a clean
      declarative clause; hypothesis is a third-person paraphrase
      of that exact clause (so the entailment is real).
    * CONTRADICT — same premise, but the hypothesis flips a number,
      year, or entity that appears in the original clause, OR
      explicitly negates the clause.
    * NEUTRAL — premise from one conversation, hypothesis paraphrased
      from a clause in a *different* conversation (so the topics are
      genuinely disjoint).
    """
    rng = random.Random(seed)
    convs = _load_locomo(locomo_path)
    indices = [_index_dialog_by_id(c) for c in convs]
    entities_per_conv = [_named_entities_in_conv(c) for c in convs]

    # Pool = (conv_idx, dia_id, speaker, clause, anchor_kind)
    clause_pool: list[tuple[int, str, str, str, str]] = []
    for ci, idx in enumerate(indices):
        for dia_id, turn in idx.items():
            speaker = (turn.get("speaker") or "").strip()
            if not speaker:
                continue
            for clause in _select_premise_clauses(turn["text"]):
                if not _FIRST_PERSON_RE.search(clause):
                    # We need a first-person clause we can rewrite to
                    # third-person — that gives us a clean entail pair.
                    continue
                anchor = _has_anchor_for_corruption(clause)
                if anchor is None:
                    continue
                clause_pool.append((ci, dia_id, speaker, clause, anchor))
    rng.shuffle(clause_pool)

    triples: list[NLITriple] = []
    stats = FixtureStats()

    # ── ENTAIL ───────────────────────────────────────────────────────
    used_dia: set[tuple[int, str]] = set()
    cursor = 0
    while stats.entail < target_entail and cursor < len(clause_pool):
        ci, dia_id, speaker, clause, _anchor = clause_pool[cursor]
        cursor += 1
        if (ci, dia_id) in used_dia:
            continue
        hypothesis = _to_third_person(speaker, clause)
        if hypothesis == clause or len(hypothesis) < 18:
            continue
        # Light dedupe: skip if hypothesis equals premise verbatim or is too short.
        premise = f"{speaker}: {clause}".strip()
        triples.append(
            NLITriple(
                premise=premise,
                hypothesis=hypothesis,
                gold_label="entail",
                kind="entail",
                rule="paraphrase-1st-to-3rd-person",
                source_qa_idx=-1,
                source_conv_idx=ci,
                source_dia_id=dia_id,
            )
        )
        used_dia.add((ci, dia_id))
        stats.entail += 1

    # ── CONTRADICT ───────────────────────────────────────────────────
    target_per_kind = max(1, target_contradict // 3)
    quotas = {
        "numeric": target_per_kind,
        "entity": target_per_kind,
        "negate": target_contradict - 2 * target_per_kind,
    }
    for kind in ("numeric", "entity", "negate"):
        produced = 0
        scan = list(clause_pool)
        rng.shuffle(scan)
        for ci, dia_id, speaker, clause, anchor in scan:
            if produced >= quotas[kind]:
                break
            if kind == "numeric" and anchor != "numeric":
                continue
            if kind == "entity" and anchor != "entity":
                continue

            # Build the truthful third-person paraphrase, then corrupt that.
            base_hyp = _to_third_person(speaker, clause)
            if not base_hyp or len(base_hyp) < 18:
                continue

            corrupted: tuple[str, str] | None
            if kind == "numeric":
                corrupted = _flip_number(base_hyp, clause, rng)
                bucket = "contradict_numeric"
            elif kind == "entity":
                corrupted = _swap_entity(base_hyp, clause, entities_per_conv[ci], rng)
                bucket = "contradict_entity"
            else:  # negate
                corrupted = _negate(base_hyp)
                bucket = "contradict_negate"

            if not corrupted:
                continue
            new_hyp, rule = corrupted
            if not new_hyp or new_hyp.strip() == base_hyp.strip():
                continue

            premise = f"{speaker}: {clause}".strip()
            triples.append(
                NLITriple(
                    premise=premise,
                    hypothesis=new_hyp,
                    gold_label="contradict",
                    kind=bucket,
                    rule=rule,
                    source_qa_idx=-1,
                    source_conv_idx=ci,
                    source_dia_id=dia_id,
                )
            )
            produced += 1
            if bucket == "contradict_numeric":
                stats.contradict_numeric += 1
            elif bucket == "contradict_entity":
                stats.contradict_entity += 1
            else:
                stats.contradict_negate += 1

    # ── NEUTRAL — unrelated paraphrase ───────────────────────────────
    target_unrelated = max(1, target_neutral - target_neutral // 3)
    target_idk = target_neutral - target_unrelated

    produced = 0
    scan = list(clause_pool)
    rng.shuffle(scan)
    for ci, dia_id, speaker, clause, _anchor in scan:
        if produced >= target_unrelated:
            break
        other_ci = rng.choice([j for j in range(len(convs)) if j != ci])
        other_clauses = [
            (s, t["speaker"], c)
            for s, t in indices[other_ci].items()
            for c in _select_premise_clauses(t["text"])
            if t.get("speaker") and _FIRST_PERSON_RE.search(c)
        ]
        if not other_clauses:
            continue
        _o_dia, o_speaker, o_clause = rng.choice(other_clauses)
        hyp = _to_third_person(o_speaker, o_clause)
        if not hyp or len(hyp) < 18:
            continue
        premise = f"{speaker}: {clause}".strip()
        if premise.lower() == hyp.lower():
            continue
        triples.append(
            NLITriple(
                premise=premise,
                hypothesis=hyp,
                gold_label="neutral",
                kind="neutral_unrelated",
                rule=f"unrelated-paraphrase-from-conv{other_ci}",
                source_qa_idx=-1,
                source_conv_idx=ci,
                source_dia_id=dia_id,
            )
        )
        produced += 1
        stats.neutral_unrelated += 1

    # ── NEUTRAL — IDK-style adversarial ──────────────────────────────
    # Mirrors the production failure mode: when the LLM gen returns
    # "Not specified in the conversation" / "I don't know", the verifier
    # was previously labelling it `contradict` against arbitrary
    # dialogue turns. Those should be NEUTRAL — the answer makes no
    # claim that can entail or contradict the premise.
    idk_answers = [
        "Not specified in the conversation.",
        "I don't know.",
        "This is not mentioned in the conversation.",
        "No information is provided.",
        "The conversation does not say.",
        "It is not stated.",
        "Not enough information.",
        "I am not sure.",
    ]
    produced = 0
    scan = list(clause_pool)
    rng.shuffle(scan)
    for ci, dia_id, speaker, clause, _anchor in scan:
        if produced >= target_idk:
            break
        premise = f"{speaker}: {clause}".strip()
        hyp = idk_answers[produced % len(idk_answers)]
        triples.append(
            NLITriple(
                premise=premise,
                hypothesis=hyp,
                gold_label="neutral",
                kind="neutral_idk",
                rule="idk-answer-vs-real-dialogue",
                source_qa_idx=-1,
                source_conv_idx=ci,
                source_dia_id=dia_id,
            )
        )
        produced += 1
        stats.neutral_idk += 1

    return triples, stats


def _fixture_payload(triples: list[NLITriple], stats: FixtureStats, seed: int) -> dict:
    return {
        "schema_version": 1,
        "seed": seed,
        "stats": stats.to_dict(),
        "triples": [t.to_dict() for t in triples],
    }


def _md5_of_payload(payload: dict) -> str:
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.md5(blob).hexdigest()


def write_fixture(triples: list[NLITriple], stats: FixtureStats, seed: int, out: Path) -> str:
    payload = _fixture_payload(triples, stats, seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return _md5_of_payload(payload)


def load_fixture(path: Path) -> tuple[list[NLITriple], FixtureStats, int, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    triples = [
        NLITriple(
            premise=row["premise"],
            hypothesis=row["hypothesis"],
            gold_label=row["gold_label"],
            kind=row["kind"],
            rule=row.get("rule", ""),
            source_qa_idx=row.get("source_qa_idx", -1),
            source_conv_idx=row.get("source_conv_idx", -1),
            source_dia_id=row.get("source_dia_id"),
        )
        for row in payload["triples"]
    ]
    stats_d = payload.get("stats", {})
    stats = FixtureStats(
        entail=stats_d.get("entail", 0),
        contradict_numeric=stats_d.get("contradict_numeric", 0),
        contradict_entity=stats_d.get("contradict_entity", 0),
        contradict_negate=stats_d.get("contradict_negate", 0),
        neutral_unrelated=stats_d.get("neutral_unrelated", 0),
        neutral_idk=stats_d.get("neutral_idk", 0),
    )
    return triples, stats, payload.get("seed", 42), _md5_of_payload(payload)


# ────────────────────────────────────────────────────────────────────
# Inference + tuning
# ────────────────────────────────────────────────────────────────────


def _split_train_test(
    triples: Sequence[NLITriple], seed: int, test_frac: float = 0.30,
) -> tuple[list[int], list[int]]:
    """Stratified split by gold_label so both halves see all classes."""
    by_label: dict[str, list[int]] = {"entail": [], "neutral": [], "contradict": []}
    for i, t in enumerate(triples):
        by_label.setdefault(t.gold_label, []).append(i)
    rng = random.Random(seed + 1)
    train: list[int] = []
    test: list[int] = []
    for label, idxs in by_label.items():
        rng.shuffle(idxs)
        cut = int(round(len(idxs) * (1.0 - test_frac)))
        train.extend(idxs[:cut])
        test.extend(idxs[cut:])
    train.sort()
    test.sort()
    return train, test


def _score_with_model(
    model_id: str,
    triples: Sequence[NLITriple],
    *,
    batch_size: int = 16,
) -> list[tuple[float, float, float]]:
    """Run the chosen NLI model on every triple. Returns per-row probs."""
    import torch  # noqa: WPS433
    from transformers import (  # noqa: WPS433
        AutoModelForSequenceClassification,
        AutoTokenizer,
    )

    from ai_layer.verifier import NLIDecision, _NLIModel  # type: ignore  # noqa: WPS433

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    model.eval()
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model.to(device)

    id2label = getattr(model.config, "id2label", None) or {}
    idx_map = _NLIModel._resolve_label_indices(id2label)
    e_idx = idx_map[NLIDecision.ENTAIL]
    n_idx = idx_map[NLIDecision.NEUTRAL]
    c_idx = idx_map[NLIDecision.CONTRADICT]

    out: list[tuple[float, float, float]] = []
    with torch.no_grad():
        for start in range(0, len(triples), batch_size):
            chunk = triples[start : start + batch_size]
            prems = [t.premise for t in chunk]
            hyps = [t.hypothesis for t in chunk]
            enc = tokenizer(
                prems,
                hyps,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1).detach().to("cpu").tolist()
            for row in probs:
                out.append(
                    (float(row[e_idx]), float(row[n_idx]), float(row[c_idx]))
                )
    return out


def _decide(
    p_entail: float,
    p_neutral: float,
    p_contradict: float,
    *,
    p_entail_threshold: float,
    p_contradict_threshold: float,
    p_contradict_margin: float,
) -> str:
    """Calibrated decision rule (mirrors verifier._decide)."""
    if (
        p_contradict >= p_contradict_threshold
        and (p_contradict - p_entail) >= p_contradict_margin
    ):
        return "contradict"
    if (
        p_entail >= p_entail_threshold
        and p_entail >= p_neutral
        and p_entail >= p_contradict
    ):
        return "entail"
    return "neutral"


@dataclass
class EvalReport:
    confusion: dict[str, dict[str, int]]  # gold -> pred -> count
    accuracy: float
    balanced_accuracy: float
    entail_recall: float
    contradict_recall: float
    neutral_recall: float
    false_contradict_rate: float          # P(pred=contradict | gold!=contradict)
    n: int

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def _evaluate(
    triples: Sequence[NLITriple],
    probs: Sequence[tuple[float, float, float]],
    indices: Iterable[int],
    *,
    p_entail_threshold: float,
    p_contradict_threshold: float,
    p_contradict_margin: float,
) -> EvalReport:
    labels = ["entail", "neutral", "contradict"]
    confusion = {g: {p: 0 for p in labels} for g in labels}
    n = 0
    for i in indices:
        gold = triples[i].gold_label
        pe, pn, pc = probs[i]
        pred = _decide(
            pe, pn, pc,
            p_entail_threshold=p_entail_threshold,
            p_contradict_threshold=p_contradict_threshold,
            p_contradict_margin=p_contradict_margin,
        )
        confusion[gold][pred] += 1
        n += 1

    if n == 0:
        return EvalReport(
            confusion=confusion,
            accuracy=0.0,
            balanced_accuracy=0.0,
            entail_recall=0.0,
            contradict_recall=0.0,
            neutral_recall=0.0,
            false_contradict_rate=0.0,
            n=0,
        )

    correct = sum(confusion[g][g] for g in labels)
    accuracy = correct / n

    def _recall(label: str) -> float:
        gold_n = sum(confusion[label].values())
        if gold_n == 0:
            return 0.0
        return confusion[label][label] / gold_n

    entail_recall = _recall("entail")
    neutral_recall = _recall("neutral")
    contradict_recall = _recall("contradict")
    balanced_accuracy = (entail_recall + neutral_recall + contradict_recall) / 3.0

    # False-contradict rate: among non-contradict gold, fraction predicted contradict.
    non_contra_total = sum(confusion["entail"].values()) + sum(
        confusion["neutral"].values()
    )
    false_contradict = (
        confusion["entail"]["contradict"] + confusion["neutral"]["contradict"]
    )
    false_contradict_rate = (
        false_contradict / non_contra_total if non_contra_total else 0.0
    )

    return EvalReport(
        confusion=confusion,
        accuracy=accuracy,
        balanced_accuracy=balanced_accuracy,
        entail_recall=entail_recall,
        contradict_recall=contradict_recall,
        neutral_recall=neutral_recall,
        false_contradict_rate=false_contradict_rate,
        n=n,
    )


@dataclass
class TuneCandidate:
    p_entail_threshold: float
    p_contradict_threshold: float
    p_contradict_margin: float
    train_report: EvalReport
    test_report: EvalReport
    model_id: str

    def meets_constraints(
        self,
        *,
        min_contradict_recall: float,
        min_entail_recall: float,
    ) -> bool:
        return (
            self.train_report.contradict_recall >= min_contradict_recall
            and self.train_report.entail_recall >= min_entail_recall
        )


def _sweep_thresholds(
    triples: Sequence[NLITriple],
    probs: Sequence[tuple[float, float, float]],
    train_idx: list[int],
    test_idx: list[int],
    *,
    model_id: str,
    min_contradict_recall: float,
    min_entail_recall: float,
) -> tuple[TuneCandidate, list[TuneCandidate]]:
    """Grid sweep on TRAIN, evaluate selected combo on TEST."""
    contra_grid = [round(0.30 + 0.05 * i, 2) for i in range(14)]  # 0.30 .. 0.95
    entail_grid = [round(0.30 + 0.05 * i, 2) for i in range(13)]  # 0.30 .. 0.90
    margin_grid = [round(0.05 * i, 2) for i in range(9)]          # 0.00 .. 0.40

    candidates: list[TuneCandidate] = []
    for tc in contra_grid:
        for te in entail_grid:
            for tm in margin_grid:
                train_rep = _evaluate(
                    triples,
                    probs,
                    train_idx,
                    p_entail_threshold=te,
                    p_contradict_threshold=tc,
                    p_contradict_margin=tm,
                )
                cand = TuneCandidate(
                    p_entail_threshold=te,
                    p_contradict_threshold=tc,
                    p_contradict_margin=tm,
                    train_report=train_rep,
                    test_report=EvalReport(
                        confusion={}, accuracy=0.0, balanced_accuracy=0.0,
                        entail_recall=0.0, contradict_recall=0.0,
                        neutral_recall=0.0, false_contradict_rate=0.0, n=0,
                    ),
                    model_id=model_id,
                )
                candidates.append(cand)

    feasible = [
        c
        for c in candidates
        if c.meets_constraints(
            min_contradict_recall=min_contradict_recall,
            min_entail_recall=min_entail_recall,
        )
    ]
    if feasible:
        # 1) min false_contradict, 2) max balanced_accuracy.
        feasible.sort(
            key=lambda c: (
                c.train_report.false_contradict_rate,
                -c.train_report.balanced_accuracy,
            )
        )
        winner = feasible[0]
    else:
        # No combo meets the recall constraints — fall back to balanced acc.
        candidates.sort(key=lambda c: -c.train_report.balanced_accuracy)
        winner = candidates[0]

    # Evaluate winner on test split.
    winner.test_report = _evaluate(
        triples,
        probs,
        test_idx,
        p_entail_threshold=winner.p_entail_threshold,
        p_contradict_threshold=winner.p_contradict_threshold,
        p_contradict_margin=winner.p_contradict_margin,
    )
    return winner, candidates


# ────────────────────────────────────────────────────────────────────
# Reporting
# ────────────────────────────────────────────────────────────────────


def _format_confusion_md(confusion: dict[str, dict[str, int]]) -> str:
    labels = ["entail", "neutral", "contradict"]
    lines = ["| gold \\ pred | entail | neutral | contradict |", "|---|---|---|---|"]
    for g in labels:
        row = confusion.get(g, {})
        lines.append(
            f"| **{g}** | {row.get('entail', 0)} | {row.get('neutral', 0)} | {row.get('contradict', 0)} |"
        )
    return "\n".join(lines)


def _baseline_evaluation(
    triples: Sequence[NLITriple],
    probs: Sequence[tuple[float, float, float]],
    indices: list[int],
) -> EvalReport:
    """Pre-calibration evaluation using the original v11 thresholds."""
    return _evaluate(
        triples,
        probs,
        indices,
        p_entail_threshold=0.50,
        p_contradict_threshold=0.60,
        p_contradict_margin=0.0,
    )


def _render_report(
    *,
    fixture_path: Path,
    fixture_md5: str,
    fixture_stats: FixtureStats,
    train_n: int,
    test_n: int,
    triples: Sequence[NLITriple],
    per_model: dict[str, dict],
    winner_model: str,
    winner: TuneCandidate,
    chosen_path: str,
    notes: list[str],
) -> str:
    lines: list[str] = []
    lines.append("# NLI Verifier Calibration — Report (W5)")
    lines.append("")
    lines.append(f"Generated: {datetime.datetime.now(datetime.UTC).isoformat(timespec='seconds').replace('+00:00', 'Z')}")
    lines.append("")
    lines.append("## Fixture")
    lines.append("")
    lines.append(f"* Path: `{fixture_path}`")
    lines.append(f"* MD5: `{fixture_md5}`")
    lines.append(f"* Total: {fixture_stats.total()}  ({fixture_stats.to_dict()})")
    lines.append(f"* Train / Test split (stratified, seed=43): {train_n} / {test_n}")
    lines.append("")
    lines.append("Construction:")
    lines.append(
        "* **Entail** — premise is a first-person clause taken from a LoCoMo "
        "dialogue turn; hypothesis is a third-person paraphrase of the same "
        "clause (so the entailment is real, not requiring inference)."
    )
    lines.append(
        "* **Contradict** — same premise, hypothesis = corrupted paraphrase: "
        "(a) numeric/year flip on a token that appears in both premise and "
        "hypothesis; (b) entity swap with a proper noun present in both; or "
        "(c) in-sentence negation (`X is` → `X is not`)."
    )
    lines.append(
        "* **Neutral (unrelated)** — premise from one conversation, hypothesis "
        "paraphrased from a clause in a *different* conversation."
    )
    lines.append(
        "* **Neutral (IDK adversarial)** — premise = real dialogue turn, "
        "hypothesis = an IDK-style answer (`Not specified in the conversation`, "
        "`I don't know`, …). Mirrors the production failure mode where the "
        "verifier was emitting `contradict` 58% of the time for "
        "(answer, evidence) pairs where the answer was an IDK token."
    )
    lines.append("")
    lines.append(
        "Leakage caveat: LoCoMo has no formal train/test split. The bench task is "
        "RAG QA, the calibration task is NLI triple classification — different "
        "task formulations, but the same underlying dialogue text. Documenting "
        "honestly per the W5 spec."
    )
    lines.append("")

    lines.append("## Per-model summary")
    lines.append("")
    lines.append(
        "| model | train fc-rate | test fc-rate | entail rec (test) | "
        "contradict rec (test) | bal acc (test) |"
    )
    lines.append("|---|---|---|---|---|---|")
    for model_id, payload in per_model.items():
        cand: TuneCandidate = payload["winner"]
        lines.append(
            f"| `{model_id}` | "
            f"{cand.train_report.false_contradict_rate:.3f} | "
            f"{cand.test_report.false_contradict_rate:.3f} | "
            f"{cand.test_report.entail_recall:.3f} | "
            f"{cand.test_report.contradict_recall:.3f} | "
            f"{cand.test_report.balanced_accuracy:.3f} |"
        )
    lines.append("")

    lines.append(f"## Chosen path: **{chosen_path}**  —  `{winner_model}`")
    lines.append("")
    lines.append("### Thresholds")
    lines.append("")
    lines.append(f"* `p_entail_threshold = {winner.p_entail_threshold}`")
    lines.append(f"* `p_contradict_threshold = {winner.p_contradict_threshold}`")
    lines.append(f"* `p_contradict_margin = {winner.p_contradict_margin}`")
    lines.append("")
    lines.append("### Before vs after — test split (same model)")
    lines.append("")
    base_eval = per_model[winner_model]["baseline_test"]
    lines.append(
        "| metric | baseline (p_c>0.6, p_e>=0.5, no margin) | calibrated |"
    )
    lines.append("|---|---|---|")
    lines.append(
        f"| false-contradict rate | {base_eval.false_contradict_rate:.3f} | "
        f"{winner.test_report.false_contradict_rate:.3f} |"
    )
    lines.append(
        f"| entail recall | {base_eval.entail_recall:.3f} | "
        f"{winner.test_report.entail_recall:.3f} |"
    )
    lines.append(
        f"| contradict recall | {base_eval.contradict_recall:.3f} | "
        f"{winner.test_report.contradict_recall:.3f} |"
    )
    lines.append(
        f"| neutral recall | {base_eval.neutral_recall:.3f} | "
        f"{winner.test_report.neutral_recall:.3f} |"
    )
    lines.append(
        f"| balanced accuracy | {base_eval.balanced_accuracy:.3f} | "
        f"{winner.test_report.balanced_accuracy:.3f} |"
    )
    lines.append("")

    lines.append("### Confusion — baseline (test split)")
    lines.append("")
    lines.append(_format_confusion_md(base_eval.confusion))
    lines.append("")
    lines.append("### Confusion — calibrated (test split)")
    lines.append("")
    lines.append(_format_confusion_md(winner.test_report.confusion))
    lines.append("")

    if notes:
        lines.append("## Notes")
        lines.append("")
        for n in notes:
            lines.append(f"* {n}")
        lines.append("")
    return "\n".join(lines) + "\n"


# ────────────────────────────────────────────────────────────────────
# v11 bench impact estimator (re-decide cached predictions)
# ────────────────────────────────────────────────────────────────────


def _estimate_bench_impact(
    bench_path: Path,
    *,
    p_entail_threshold: float,
    p_contradict_threshold: float,
    p_contradict_margin: float,
    triples: Sequence[NLITriple],
    probs: Sequence[tuple[float, float, float]],
) -> dict | None:
    """Approximate how the chosen thresholds affect the cached Haiku 200 run.

    The cached run does NOT store p_entail / p_neutral / p_contradict — only
    the categorical decision. We can only count a *lower bound* on the
    decision change: how many records had v11_nli_decision='contradict'
    under the old thresholds (and would have triggered a veto if the
    veto wire-up had been on).

    For an actual override-count estimate under the new thresholds we would
    need raw probs in the bench JSON; flagged in the report.
    """
    if not bench_path.exists():
        return None
    try:
        data = json.loads(bench_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None
    records = data.get("records") or []
    if not records:
        return None
    has_probs = any("v11_nli_p_contradict" in r for r in records)
    counts = {"entail": 0, "neutral": 0, "contradict": 0, "missing": 0}
    for r in records:
        d = r.get("v11_nli_decision")
        if d in counts:
            counts[d] += 1
        else:
            counts["missing"] += 1
    return {
        "bench_path": str(bench_path),
        "n_records": len(records),
        "decisions_under_baseline": counts,
        "raw_probs_available": has_probs,
        "note": (
            "Cached predictions store only categorical decisions; per-record "
            "probabilities are not available, so an exact count of NEW vetoes "
            "under the calibrated thresholds cannot be computed without re-running. "
            "Lower bound: at most "
            f"{counts['contradict']} records were ever flagged contradict by the "
            "old (uncalibrated) thresholds."
        ),
    }


# ────────────────────────────────────────────────────────────────────
# CLI plumbing
# ────────────────────────────────────────────────────────────────────


def _expand(p: str) -> Path:
    return Path(os.path.expanduser(os.path.expandvars(p)))


def cmd_build_fixture(args: argparse.Namespace) -> int:
    triples, stats = build_fixture(
        seed=args.seed,
        target_entail=args.target_entail,
        target_contradict=args.target_contradict,
        target_neutral=args.target_neutral,
    )
    out = _expand(args.output)
    md5 = write_fixture(triples, stats, args.seed, out)
    print(f"[fixture] wrote {len(triples)} triples to {out}")
    print(f"[fixture] stats: {stats.to_dict()}")
    print(f"[fixture] md5: {md5}")
    return 0


def cmd_tune(args: argparse.Namespace) -> int:
    fixture_path = _expand(args.fixture)
    triples, stats, _seed, fixture_md5 = load_fixture(fixture_path)
    train_idx, test_idx = _split_train_test(triples, seed=42, test_frac=0.30)
    print(
        f"[tune] fixture={fixture_path} md5={fixture_md5} "
        f"train={len(train_idx)} test={len(test_idx)}"
    )

    # Decide which models to actually probe.
    candidate_models: list[str] = [_PRODUCTION_MODEL]
    if not args.no_path_b:
        for m in _PATH_B_CANDIDATES:
            candidate_models.append(m)

    per_model: dict[str, dict] = {}
    notes: list[str] = []
    for model_id in candidate_models:
        try:
            print(f"[tune] scoring with {model_id}")
            probs = _score_with_model(model_id, triples, batch_size=args.batch_size)
        except Exception as exc:  # noqa: BLE001 — model swap may legitimately fail offline.
            notes.append(f"Skipped `{model_id}`: {type(exc).__name__}: {exc}")
            continue
        baseline_test = _baseline_evaluation(triples, probs, test_idx)
        winner, _all = _sweep_thresholds(
            triples,
            probs,
            train_idx,
            test_idx,
            model_id=model_id,
            min_contradict_recall=args.min_contradict_recall,
            min_entail_recall=args.min_entail_recall,
        )
        per_model[model_id] = {
            "winner": winner,
            "baseline_test": baseline_test,
        }
        print(
            f"[tune] {model_id} -> τc={winner.p_contradict_threshold} "
            f"τe={winner.p_entail_threshold} margin={winner.p_contradict_margin} "
            f"test_fc={winner.test_report.false_contradict_rate:.3f} "
            f"test_recall_c={winner.test_report.contradict_recall:.3f} "
            f"test_recall_e={winner.test_report.entail_recall:.3f}"
        )

    if not per_model:
        print("[tune] FATAL: no model produced predictions", file=sys.stderr)
        return 2

    # Pick winner: lowest test fc-rate subject to recall constraints; tie-break
    # prefer the production model (Path A) per spec.
    def _score(model_id: str) -> tuple:
        cand: TuneCandidate = per_model[model_id]["winner"]
        meets = (
            cand.test_report.contradict_recall >= args.min_contradict_recall
            and cand.test_report.entail_recall >= args.min_entail_recall
        )
        return (
            0 if meets else 1,
            cand.test_report.false_contradict_rate,
            -cand.test_report.balanced_accuracy,
            0 if model_id == _PRODUCTION_MODEL else 1,
        )

    ranked_models = sorted(per_model.keys(), key=_score)
    winner_model = ranked_models[0]
    winner = per_model[winner_model]["winner"]
    chosen_path = "A (threshold tuning, no model swap)" if winner_model == _PRODUCTION_MODEL else "B (model swap)"

    # Write the calibration JSON.
    out = _expand(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "model_name": winner_model,
        "p_entail_threshold": float(winner.p_entail_threshold),
        "p_contradict_threshold": float(winner.p_contradict_threshold),
        "p_contradict_margin": float(winner.p_contradict_margin),
        "calibrated_at": datetime.datetime.now(datetime.UTC)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "calibration_set_md5": fixture_md5,
        "calibration_set_path": str(fixture_path),
        "metrics_test_split": winner.test_report.to_dict(),
        "metrics_train_split": winner.train_report.to_dict(),
        "chosen_path": chosen_path,
    }
    out.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"[tune] wrote calibration → {out}")

    # Optional: bench impact estimate.
    if args.bench:
        impact = _estimate_bench_impact(
            _expand(args.bench),
            p_entail_threshold=winner.p_entail_threshold,
            p_contradict_threshold=winner.p_contradict_threshold,
            p_contradict_margin=winner.p_contradict_margin,
            triples=triples,
            probs=[],  # not used; we only have categorical predictions on bench
        )
        if impact:
            notes.append(f"Bench impact (cached run): {json.dumps(impact, ensure_ascii=False)}")

    if args.report:
        report_md = _render_report(
            fixture_path=fixture_path,
            fixture_md5=fixture_md5,
            fixture_stats=stats,
            train_n=len(train_idx),
            test_n=len(test_idx),
            triples=triples,
            per_model=per_model,
            winner_model=winner_model,
            winner=winner,
            chosen_path=chosen_path,
            notes=notes,
        )
        rep_path = _expand(args.report)
        rep_path.parent.mkdir(parents=True, exist_ok=True)
        rep_path.write_text(report_md, encoding="utf-8")
        print(f"[tune] wrote report → {rep_path}")

    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="W5 NLI calibration (build fixture + tune thresholds).")
    sub = p.add_subparsers(dest="cmd")

    # Top-level convenience flags so the suggested commands in the W5 spec work.
    p.add_argument("--build-fixture", action="store_true", help="Generate the calibration fixture.")
    p.add_argument("--tune", action="store_true", help="Run threshold/model tuning.")
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--fixture", type=str, default=str(_REPO / "tests/fixtures/nli_calibration_set.json"))
    p.add_argument("--report", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--target-entail", type=int, default=120)
    p.add_argument("--target-contradict", type=int, default=120)
    p.add_argument("--target-neutral", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--min-contradict-recall", type=float, default=0.80)
    p.add_argument("--min-entail-recall", type=float, default=0.75)
    p.add_argument("--no-path-b", action="store_true", help="Skip Path B model-swap candidates.")
    p.add_argument("--bench", type=str, default=None, help="Optional cached bench JSON for impact estimation.")

    args = p.parse_args(argv)

    if args.build_fixture and args.tune:
        print("Pass either --build-fixture or --tune, not both.", file=sys.stderr)
        return 2
    if args.build_fixture:
        if not args.output:
            args.output = str(_REPO / "tests/fixtures/nli_calibration_set.json")
        return cmd_build_fixture(args)
    if args.tune:
        if not args.output:
            args.output = "~/.claude-memory/nli_calibration.json"
        return cmd_tune(args)

    p.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
