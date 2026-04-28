"""Tests for memory_core.entity_resolver — Global Entity Resolver (W1-F).

Each test gets an isolated in-memory SQLite with migration 024 applied
on top of the base v5 schema (we only need the `migrations` table and
the two tables migration 024 creates). Embeddings are stubbed with a
deterministic bag-of-tokens hash so cosine similarity is reproducible
without loading FastEmbed/Ollama.
"""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from memory_core import entity_resolver as er
from memory_core.entity_resolver import (
    ResolveResult,
    add_alias,
    get_canonical,
    is_pronoun,
    list_aliases,
    merge_canonicals,
    normalize,
    resolve,
)


# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────


_MIGRATIONS_DIR = Path(__file__).parent.parent / "migrations"


def _apply_migration_024(conn: sqlite3.Connection) -> None:
    # Bare minimum to satisfy `INSERT OR IGNORE INTO migrations`
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS migrations (
            version TEXT PRIMARY KEY,
            description TEXT,
            applied_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
        );
        """
    )
    conn.executescript((_MIGRATIONS_DIR / "024_entity_aliases.sql").read_text())


@pytest.fixture
def conn(tmp_path):
    """Per-test SQLite file under tmp_path (isolated, on-disk so PRAGMA
    foreign_keys behaves identically to production)."""
    db_path = tmp_path / "entity_resolver.sqlite"
    c = sqlite3.connect(str(db_path))
    c.execute("PRAGMA foreign_keys = ON")
    _apply_migration_024(c)
    yield c
    c.close()


# ──────────────────────────────────────────────
# Deterministic mock embedder
# ──────────────────────────────────────────────


_DIM = 64


def _token_index(token: str) -> int:
    h = hashlib.md5(token.encode("utf-8")).digest()
    return int.from_bytes(h[:4], "big") % _DIM


def fake_embed(text: str) -> np.ndarray:
    """Bag-of-normalized-tokens indicator vector.

    Two strings sharing tokens (e.g. "Sarah" and "Dr. Sarah Williams")
    overlap in ≥1 dimension → cosine > 0. Disjoint token sets → cosine
    = 0. Stable across calls. dim=64 keeps the chance of accidental
    collisions on unrelated short strings low for the fixtures below.
    """
    vec = np.zeros(_DIM, dtype=np.float32)
    norm = normalize(text)
    if not norm:
        return vec
    for tok in norm.split():
        vec[_token_index(tok)] += 1.0
    return vec


def disjoint_embed(text: str) -> np.ndarray:
    """Same shape, but uses the *full normalized string* as the only
    token. Ensures unrelated mentions never accidentally share a token
    hash — used for the negative test that proves embedding match
    refuses to merge two genuinely different people."""
    vec = np.zeros(_DIM, dtype=np.float32)
    norm = normalize(text)
    if not norm:
        return vec
    vec[_token_index(norm)] = 1.0
    return vec


# ──────────────────────────────────────────────
# normalize() unit tests
# ──────────────────────────────────────────────


def test_normalize_lowercases():
    assert normalize("Sarah") == "sarah"


def test_normalize_strips_punctuation():
    assert normalize("Dr. Sarah Williams!") == "dr sarah williams"


def test_normalize_collapses_whitespace():
    assert normalize("  Sarah   \t\n Williams  ") == "sarah williams"


def test_normalize_strips_accents():
    # NFKD decomposes "é" → "e" + combining acute, which we drop
    assert normalize("Sára") == "sara"
    assert normalize("Café") == "cafe"


def test_normalize_handles_unicode_punctuation():
    # U+2019 RIGHT SINGLE QUOTATION MARK and em-dash are category P*
    assert normalize("O’Connor—Smith") == "o connor smith"


def test_normalize_empty_inputs():
    assert normalize("") == ""
    assert normalize("   ") == ""
    assert normalize("...") == ""
    assert normalize(None) == ""  # type: ignore[arg-type]


def test_normalize_cyrillic_preserved():
    # Cyrillic doesn't have combining marks here; just casefold
    assert normalize("Саша") == "саша"
    assert normalize("ПЁТР") == normalize("пётр")


def test_is_pronoun_english():
    assert is_pronoun("she")
    assert is_pronoun("She")
    assert is_pronoun("THEY")


def test_is_pronoun_russian():
    assert is_pronoun("она")
    assert is_pronoun("Они")


def test_is_pronoun_negative():
    assert not is_pronoun("Sarah")
    assert not is_pronoun("Python")


# ──────────────────────────────────────────────
# resolve() — happy paths
# ──────────────────────────────────────────────


def test_resolve_creates_new_canonical(conn):
    res = resolve(conn, "Sarah", "proj", "person", embed_fn=fake_embed)
    assert isinstance(res, ResolveResult)
    assert res.is_new is True
    assert res.matched_via == "created"
    assert res.canonical_id > 0
    assert res.canonical_name == "Sarah"
    assert res.confidence == 1.0


def test_resolve_exact_match_across_sessions(conn):
    """First call creates; second call finds it via exact normalize match."""
    a = resolve(conn, "Sarah", "proj", "person", embed_fn=fake_embed)
    b = resolve(conn, "Sarah", "proj", "person", embed_fn=fake_embed)
    assert a.canonical_id == b.canonical_id
    assert b.is_new is False
    # Note: matched_via on b is "alias", because resolve() seeds the
    # canonical name as an alias in step 5. Either "alias" or "exact"
    # is correct (alias hits first in the lookup chain), so we accept
    # both — the contract is "same id, not new".
    assert b.matched_via in ("alias", "exact")


def test_resolve_case_and_punctuation_insensitive(conn):
    a = resolve(conn, "Sarah Williams", "proj", "person", embed_fn=fake_embed)
    b = resolve(conn, "  sarah   williams  ", "proj", "person", embed_fn=fake_embed)
    c = resolve(conn, "Sarah Williams!", "proj", "person", embed_fn=fake_embed)
    assert a.canonical_id == b.canonical_id == c.canonical_id
    assert b.is_new is False
    assert c.is_new is False


def test_resolve_accent_insensitive(conn):
    a = resolve(conn, "Sara", "proj", "person", embed_fn=fake_embed)
    b = resolve(conn, "Sára", "proj", "person", embed_fn=fake_embed)
    assert a.canonical_id == b.canonical_id
    assert b.is_new is False


# ──────────────────────────────────────────────
# resolve() — alias paths
# ──────────────────────────────────────────────


def test_resolve_explicit_alias_match(conn):
    """Insert canonical 'Dr. Sarah Williams' explicitly + alias 'Sarah';
    later 'sarah' resolves via alias_norm lookup."""
    res = resolve(conn, "Dr. Sarah Williams", "proj", "person", embed_fn=fake_embed)
    add_alias(
        conn,
        canonical_id=res.canonical_id,
        alias="Sarah",
        source="explicit",
        confidence=1.0,
    )
    hit = resolve(conn, "Sarah", "proj", "person", embed_fn=fake_embed)
    assert hit.canonical_id == res.canonical_id
    assert hit.matched_via == "alias"
    assert hit.confidence == 1.0
    assert hit.is_new is False


def test_resolve_embedding_match_promotes_alias(conn):
    """'Dr. Sarah Williams' is created. 'Sarah Williams' has 2/3 token
    overlap → cosine well above threshold → matches, and the surface
    form is stored as an alias for next time."""
    first = resolve(
        conn, "Dr. Sarah Williams", "proj", "person",
        embed_fn=fake_embed, threshold=0.5,
    )
    second = resolve(
        conn, "Sarah Williams", "proj", "person",
        embed_fn=fake_embed, threshold=0.5,
    )
    assert second.canonical_id == first.canonical_id
    assert second.matched_via == "embedding"
    assert second.confidence >= 0.5
    # Subsequent identical lookup should now be an alias hit (cheaper)
    third = resolve(
        conn, "Sarah Williams", "proj", "person",
        embed_fn=fake_embed, threshold=0.5,
    )
    assert third.matched_via == "alias"
    assert third.canonical_id == first.canonical_id


def test_embedding_match_below_threshold_creates_new(conn):
    """Two unrelated mentions, disjoint embedder → cosine = 0 → new
    canonical even though both are person-typed."""
    a = resolve(
        conn, "Sarah Williams", "proj", "person",
        embed_fn=disjoint_embed, threshold=0.5,
    )
    b = resolve(
        conn, "Sarah Connor", "proj", "person",
        embed_fn=disjoint_embed, threshold=0.5,
    )
    assert a.canonical_id != b.canonical_id
    assert b.is_new is True
    assert b.matched_via == "created"


# ──────────────────────────────────────────────
# resolve() — pronoun guard
# ──────────────────────────────────────────────


def test_resolve_pronoun_returns_sentinel(conn):
    res = resolve(conn, "she", "proj", "person", embed_fn=fake_embed)
    assert res.canonical_id == -1
    assert res.matched_via == "pronoun"
    assert res.confidence == 0.0
    assert res.is_new is False
    # And nothing was inserted
    cnt = conn.execute("SELECT COUNT(*) FROM canonical_entities").fetchone()[0]
    assert cnt == 0


def test_resolve_pronoun_russian(conn):
    for p in ("она", "он", "они", "Она"):
        res = resolve(conn, p, "proj", "person", embed_fn=fake_embed)
        assert res.canonical_id == -1
        assert res.matched_via == "pronoun"


def test_resolve_pronoun_does_not_clobber_existing(conn):
    sarah = resolve(conn, "Sarah", "proj", "person", embed_fn=fake_embed)
    res = resolve(conn, "she", "proj", "person", embed_fn=fake_embed)
    assert res.canonical_id == -1
    # Sarah is still resolvable
    again = resolve(conn, "Sarah", "proj", "person", embed_fn=fake_embed)
    assert again.canonical_id == sarah.canonical_id


# ──────────────────────────────────────────────
# Isolation
# ──────────────────────────────────────────────


def test_cross_project_isolation(conn):
    a = resolve(conn, "Sarah", "proj_a", "person", embed_fn=fake_embed)
    b = resolve(conn, "Sarah", "proj_b", "person", embed_fn=fake_embed)
    assert a.canonical_id != b.canonical_id
    assert a.is_new and b.is_new


def test_type_isolation(conn):
    a = resolve(conn, "Python", "proj", "technology", embed_fn=fake_embed)
    b = resolve(conn, "Python", "proj", "project", embed_fn=fake_embed)
    assert a.canonical_id != b.canonical_id
    assert a.is_new and b.is_new


def test_alias_lookup_respects_project_and_type(conn):
    """An alias attached to project A must NOT match the same surface
    form looked up in project B."""
    a = resolve(conn, "Dr. Sarah Williams", "proj_a", "person", embed_fn=fake_embed)
    add_alias(conn, a.canonical_id, "Sarah", "explicit", 1.0)

    b = resolve(conn, "Sarah", "proj_b", "person", embed_fn=disjoint_embed)
    # Different project ⇒ alias lookup misses ⇒ new canonical
    assert b.canonical_id != a.canonical_id
    assert b.is_new is True


# ──────────────────────────────────────────────
# add_alias / list_aliases
# ──────────────────────────────────────────────


def test_add_alias_idempotent(conn):
    res = resolve(conn, "Sarah", "proj", "person", embed_fn=fake_embed)
    add_alias(conn, res.canonical_id, "S.", "explicit", 0.9)
    add_alias(conn, res.canonical_id, "S.", "explicit", 0.9)  # duplicate
    add_alias(conn, res.canonical_id, "s", "explicit", 0.9)   # same norm as "S."
    aliases = list_aliases(conn, res.canonical_id)
    # Canonical name "Sarah" is seeded on creation, plus single "S."
    # ("s" normalizes to same token and is deduped).
    assert "Sarah" in aliases
    assert "S." in aliases
    # Exactly two — proves dedup worked
    assert len(aliases) == 2


def test_add_alias_rejects_invalid(conn):
    res = resolve(conn, "Sarah", "proj", "person", embed_fn=fake_embed)
    with pytest.raises(ValueError):
        add_alias(conn, res.canonical_id, "", "explicit", 1.0)
    with pytest.raises(ValueError):
        add_alias(conn, res.canonical_id, "   ", "explicit", 1.0)
    with pytest.raises(ValueError):
        add_alias(conn, res.canonical_id, "...", "explicit", 1.0)
    with pytest.raises(ValueError):
        add_alias(conn, 0, "x", "explicit", 1.0)


def test_list_aliases_empty_for_unknown_id(conn):
    assert list_aliases(conn, 9999) == []
    assert list_aliases(conn, 0) == []


# ──────────────────────────────────────────────
# merge_canonicals
# ──────────────────────────────────────────────


def test_merge_canonicals_moves_aliases(conn):
    """Two canonicals accidentally created (e.g. mid-session ambiguity);
    merge folds drop into keep, all aliases follow."""
    keep = resolve(
        conn, "Dr. Sarah Williams", "proj", "person",
        embed_fn=disjoint_embed, threshold=0.99,
    )
    drop = resolve(
        conn, "Sarah W.", "proj", "person",
        embed_fn=disjoint_embed, threshold=0.99,
    )
    add_alias(conn, drop.canonical_id, "SW", "explicit", 0.8)

    moved = merge_canonicals(conn, keep_id=keep.canonical_id, drop_ids=[drop.canonical_id])
    # We moved at least the explicit alias "SW" + the drop's seed alias
    # ("Sarah W."). Drop's canonical-name-as-alias gets re-inserted via
    # add_alias with source="merge" — that inserts a new row, not a
    # move. So `moved` counts the seed alias + "SW".
    assert moved >= 1

    # drop canonical is gone
    assert get_canonical(conn, drop.canonical_id) is None

    # All aliases now under keep
    survivors = list_aliases(conn, keep.canonical_id)
    assert "SW" in survivors
    # The drop's name folded in as an alias (source="merge")
    assert any("Sarah" in a for a in survivors)

    # Look-ups for old surface forms still hit the survivor
    hit = resolve(conn, "SW", "proj", "person", embed_fn=disjoint_embed)
    assert hit.canonical_id == keep.canonical_id


def test_merge_canonicals_dedupes_overlapping_aliases(conn):
    """If both canonicals share an alias_norm, the duplicate is dropped
    rather than violating uniqueness."""
    keep = resolve(conn, "Foo", "proj", "thing", embed_fn=disjoint_embed, threshold=0.99)
    drop = resolve(conn, "Bar", "proj", "thing", embed_fn=disjoint_embed, threshold=0.99)
    add_alias(conn, keep.canonical_id, "shared", "explicit", 1.0)
    add_alias(conn, drop.canonical_id, "shared", "explicit", 1.0)

    merge_canonicals(conn, keep_id=keep.canonical_id, drop_ids=[drop.canonical_id])

    # Only one "shared" alias survives on keep
    aliases = list_aliases(conn, keep.canonical_id)
    assert aliases.count("shared") == 1


def test_merge_canonicals_validates_inputs(conn):
    keep = resolve(conn, "X", "proj", "thing", embed_fn=fake_embed)
    # No drops → 0 moved, no error
    assert merge_canonicals(conn, keep_id=keep.canonical_id, drop_ids=[]) == 0
    # Self in drop list is filtered out
    assert merge_canonicals(
        conn, keep_id=keep.canonical_id, drop_ids=[keep.canonical_id]
    ) == 0
    # Non-existent keep
    with pytest.raises(ValueError):
        merge_canonicals(conn, keep_id=99999, drop_ids=[keep.canonical_id])
    # Invalid keep id
    with pytest.raises(ValueError):
        merge_canonicals(conn, keep_id=0, drop_ids=[keep.canonical_id])


# ──────────────────────────────────────────────
# Validation / error paths
# ──────────────────────────────────────────────


def test_resolve_rejects_empty_mention(conn):
    with pytest.raises(ValueError):
        resolve(conn, "", "proj", "person", embed_fn=fake_embed)
    with pytest.raises(ValueError):
        resolve(conn, "   ", "proj", "person", embed_fn=fake_embed)


def test_resolve_rejects_punctuation_only(conn):
    with pytest.raises(ValueError):
        resolve(conn, "...", "proj", "person", embed_fn=fake_embed)
    with pytest.raises(ValueError):
        resolve(conn, "!!!", "proj", "person", embed_fn=fake_embed)


def test_resolve_rejects_non_string_mention(conn):
    with pytest.raises(TypeError):
        resolve(conn, 42, "proj", "person", embed_fn=fake_embed)  # type: ignore[arg-type]


def test_resolve_rejects_empty_project_or_type(conn):
    with pytest.raises(ValueError):
        resolve(conn, "Sarah", "", "person", embed_fn=fake_embed)
    with pytest.raises(ValueError):
        resolve(conn, "Sarah", "proj", "", embed_fn=fake_embed)


def test_resolve_works_without_embed_fn(conn):
    """Embedding is optional — exact and alias paths must still work."""
    a = resolve(conn, "Sarah", "proj", "person", embed_fn=None)
    b = resolve(conn, "Sarah", "proj", "person", embed_fn=None)
    assert a.canonical_id == b.canonical_id
    assert b.is_new is False


def test_resolve_create_if_missing_false_returns_miss(conn):
    """Read-only mode for cases where the caller wants to know whether
    an entity exists without creating it."""
    res = resolve(
        conn, "Ghost", "proj", "person",
        embed_fn=fake_embed, create_if_missing=False,
    )
    assert res.canonical_id == 0
    assert res.matched_via == "miss"
    assert res.is_new is False
    # And nothing was inserted
    cnt = conn.execute("SELECT COUNT(*) FROM canonical_entities").fetchone()[0]
    assert cnt == 0


def test_resolve_tolerates_failing_embed_fn(conn):
    """If embed_fn raises, resolver still creates via name-only path."""
    def bad_embed(_text: str) -> np.ndarray:
        raise RuntimeError("model crashed")

    res = resolve(conn, "Sarah", "proj", "person", embed_fn=bad_embed)
    assert res.is_new is True
    assert res.matched_via == "created"

    # Subsequent lookup also handles failing embed_fn
    again = resolve(conn, "Sarah", "proj", "person", embed_fn=bad_embed)
    assert again.canonical_id == res.canonical_id


# ──────────────────────────────────────────────
# Migration idempotency
# ──────────────────────────────────────────────


def test_migration_024_is_idempotent(conn):
    """Re-running the migration script must not raise."""
    conn.executescript((_MIGRATIONS_DIR / "024_entity_aliases.sql").read_text())
    conn.executescript((_MIGRATIONS_DIR / "024_entity_aliases.sql").read_text())
    # Migrations row recorded once
    rows = conn.execute(
        "SELECT version FROM migrations WHERE version = '024'"
    ).fetchall()
    assert len(rows) == 1


def test_unique_constraint_on_canonical(conn):
    """(project, type, name_norm) must be UNIQUE — second insert with
    the same triple violates the constraint."""
    conn.execute(
        """INSERT INTO canonical_entities (project, type, name, name_norm)
           VALUES ('p', 't', 'Foo', 'foo')"""
    )
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            """INSERT INTO canonical_entities (project, type, name, name_norm)
               VALUES ('p', 't', 'Foo', 'foo')"""
        )


def test_cascade_delete_aliases(conn):
    """Deleting a canonical cascades into entity_aliases."""
    res = resolve(conn, "Sarah", "proj", "person", embed_fn=fake_embed)
    add_alias(conn, res.canonical_id, "S", "explicit", 1.0)
    conn.execute("DELETE FROM canonical_entities WHERE id = ?", (res.canonical_id,))
    conn.commit()
    rows = conn.execute(
        "SELECT COUNT(*) FROM entity_aliases WHERE canonical_id = ?",
        (res.canonical_id,),
    ).fetchone()
    assert rows[0] == 0


# ──────────────────────────────────────────────
# get_canonical
# ──────────────────────────────────────────────


def test_get_canonical_round_trip(conn):
    res = resolve(conn, "Sarah", "proj", "person", embed_fn=fake_embed)
    fetched = get_canonical(conn, res.canonical_id)
    assert fetched is not None
    assert fetched["id"] == res.canonical_id
    assert fetched["project"] == "proj"
    assert fetched["type"] == "person"
    assert fetched["name"] == "Sarah"
    assert fetched["name_norm"] == "sarah"


def test_get_canonical_missing(conn):
    assert get_canonical(conn, 9999) is None
    assert get_canonical(conn, 0) is None
    assert get_canonical(conn, -1) is None
