#!/usr/bin/env python3
"""Re-embed all knowledge rows under the current V9_EMBED_BACKEND.

Idempotent: rows whose ``embed_model`` already matches the target model are
skipped (resume-friendly — interrupted runs continue where they left off).
Batches requests (default 64) and writes back float32 + binary quantized
vectors into the existing ``embeddings`` table.

Usage:
    python scripts/reembed.py --confirm
    python scripts/reembed.py --backend bge-m3 --confirm
    python scripts/reembed.py --backend e5-large --batch 32 --confirm --db /path/to/memory.db

OpenAI cloud path (LoCoMo / production swap):
    # Estimate cost only — no API calls.
    python scripts/reembed.py --provider openai \\
        --model text-embedding-3-large --dry-run

    # Actually run (after MEMORY_EMBED_API_KEY is exported).
    python scripts/reembed.py --provider openai \\
        --model text-embedding-3-large --confirm

Re-embedding the semantic ``fact_index`` (currently 384-d FastEmbed) under a
3072-d OpenAI model requires rebuilding the index after this run finishes:
``fact_index.build_semantic_index(model_name=...)`` reads the new vectors
back and rewrites its in-memory matrix; the binary-quantization sign-bit
search still works because OpenAI vectors are L2-normalised at write time.

Safety:
    - Requires explicit ``--confirm``. Without it, exits with code 2 after
      printing what *would* happen (and the cost estimate for cloud runs).
    - For ``--provider openai`` an interactive confirmation prompt is shown
      before the first API call unless ``--yes`` is also passed.
    - DB writes happen in a single transaction per batch; Ctrl-C between
      batches leaves the DB consistent.
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import struct
import sys
import time
from pathlib import Path

# Make src importable.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import choose_embed  # noqa: E402
import embed_provider as _ep  # noqa: E402


# ──────────────────────────────────────────────
# OpenAI cost / dim tables
# ──────────────────────────────────────────────
#
# Source: https://platform.openai.com/docs/pricing (2026-Q2 — text-embedding-3
# tier). Update here if OpenAI republishes the schedule. Numbers are USD per
# 1M input tokens; output tokens for embeddings are zero by definition.
_OPENAI_COST_USD_PER_1M_TOKENS: dict[str, float] = {
    "text-embedding-3-small": 0.02,
    "text-embedding-3-large": 0.13,
    "text-embedding-ada-002": 0.10,
}

_OPENAI_DIM: dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}

# OpenAI's own docs use the "≈ chars / 4" heuristic for English-heavy text.
# The estimator is intentionally conservative; the real tokeniser would
# require importing tiktoken (extra dep) for a single decimal of accuracy.
_CHARS_PER_TOKEN_HEURISTIC: float = 4.0


DEFAULT_DB = Path(
    os.environ.get("CLAUDE_MEMORY_DIR", os.path.expanduser("~/.claude-memory"))
) / "memory.db"


# ──────────────────────────────────────────────
# Quantization helpers (mirror src/server.py behavior)
# ──────────────────────────────────────────────


def _to_float32_blob(vec: list[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


def _to_binary_blob(vec: list[float]) -> bytes:
    """Sign-bit binary quantization: 1 bit per dim, packed big-endian."""
    bits = bytearray((len(vec) + 7) // 8)
    for i, v in enumerate(vec):
        if v > 0:
            bits[i >> 3] |= 1 << (7 - (i & 7))
    return bytes(bits)


# ──────────────────────────────────────────────
# Main re-embed loop
# ──────────────────────────────────────────────


def _count_rows(db: sqlite3.Connection) -> tuple[int, int]:
    total = db.execute("SELECT COUNT(*) FROM knowledge").fetchone()[0]
    with_embed = db.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
    return total, with_embed


def _iter_targets(db: sqlite3.Connection, target_model: str):
    """Yield (knowledge_id, content) rows whose embed_model != target_model.

    Includes knowledge rows with no embedding at all.
    """
    cur = db.execute(
        """
        SELECT k.id, k.content
        FROM knowledge k
        LEFT JOIN embeddings e ON e.knowledge_id = k.id
        WHERE k.content IS NOT NULL
          AND k.content != ''
          AND (e.embed_model IS NULL OR e.embed_model != ?)
        ORDER BY k.id
        """,
        (target_model,),
    )
    for row in cur:
        yield int(row[0]), row[1] or ""


def _persist_batch(
    db: sqlite3.Connection,
    rows: list[tuple[int, str]],
    vectors: list[list[float]],
    model: str,
    now_iso: str,
) -> None:
    dim = len(vectors[0]) if vectors else 0
    with db:
        for (kid, _content), vec in zip(rows, vectors):
            bin_blob = _to_binary_blob(vec)
            f32_blob = _to_float32_blob(vec)
            db.execute(
                """
                INSERT INTO embeddings (
                    knowledge_id, binary_vector, float32_vector,
                    embed_model, embed_dim, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(knowledge_id) DO UPDATE SET
                    binary_vector = excluded.binary_vector,
                    float32_vector = excluded.float32_vector,
                    embed_model = excluded.embed_model,
                    embed_dim = excluded.embed_dim,
                    created_at = excluded.created_at
                """,
                (kid, bin_blob, f32_blob, model, dim, now_iso),
            )


def estimate_tokens(texts: list[str]) -> int:
    """Heuristic token count: ceil(sum(len(text)) / chars_per_token).

    OpenAI's published guidance is ~4 chars/token for English. Using the
    heuristic instead of tiktoken keeps `--dry-run` zero-dependency.
    """
    if not texts:
        return 0
    total_chars = sum(len(t or "") for t in texts)
    return int((total_chars + _CHARS_PER_TOKEN_HEURISTIC - 1) // _CHARS_PER_TOKEN_HEURISTIC)


def estimate_cost_usd(model: str, tokens: int) -> float:
    """USD cost for `tokens` input tokens at the published per-model rate.

    Returns 0.0 for unknown models so the caller can branch on > 0 to gate
    confirmation prompts.
    """
    rate = _OPENAI_COST_USD_PER_1M_TOKENS.get(model, 0.0)
    return (tokens / 1_000_000.0) * rate


def _resolve_target(
    provider: str | None,
    model: str | None,
    backend: str | None,
) -> tuple[str, str, object]:
    """Decide the (effective_backend_label, target_model, provider_instance).

    `provider` overrides `backend` when given. `model` overrides the
    provider-default model. The returned label is what gets stored in
    `embeddings.embed_model` so resume / skip logic stays correct.
    """
    if provider:
        p = provider.strip().lower()
        if p == "openai":
            target_model = model or "text-embedding-3-large"
            inst = _ep.make_embed_provider(
                "openai",
                model=target_model,
                # Force normalisation: cosine == dot at search time.
                normalize=True,
            )
            return f"openai/{target_model}", target_model, inst
        if p == "fastembed":
            inst = _ep.make_embed_provider("fastembed", model=model)
            label = model or "fastembed"
            return label, label, inst
        if p == "cohere":
            target_model = model or "embed-multilingual-v3.0"
            inst = _ep.make_embed_provider("cohere", model=target_model)
            return f"cohere/{target_model}", target_model, inst
        raise ValueError(f"unknown --provider {provider!r}; expected openai|fastembed|cohere")

    # Legacy V9 backend selector path.
    effective_backend = choose_embed.resolve_backend(backend)
    target_model = model or choose_embed.resolve_model_name(effective_backend)
    inst = choose_embed.get_provider(effective_backend)
    return effective_backend, target_model, inst


def _confirm_interactive(prompt: str) -> bool:
    """Read a yes/no answer from stdin; returns False for non-tty/EOF."""
    try:
        ans = input(f"{prompt} [y/N]: ").strip().lower()
    except EOFError:
        return False
    return ans in ("y", "yes")


def _progress_line(done: int, total: int, elapsed: float) -> str:
    pct = (done / total * 100.0) if total else 0.0
    rate = done / elapsed if elapsed > 0 else 0.0
    eta = (total - done) / rate if rate > 0 else 0
    bar_w = 30
    filled = int(bar_w * (done / total)) if total else 0
    bar = "#" * filled + "-" * (bar_w - filled)
    return (
        f"\r[{bar}] {done}/{total} ({pct:5.1f}%) "
        f"rate={rate:6.1f}/s eta={eta:5.0f}s"
    )


def run(
    db_path: Path,
    backend: str | None,
    batch_size: int,
    confirm: bool,
    dry_run: bool,
    provider_name: str | None = None,
    model: str | None = None,
    yes: bool = False,
) -> int:
    effective_backend, target_model, provider = _resolve_target(
        provider_name, model, backend
    )

    print(f"[reembed] db:       {db_path}")
    print(f"[reembed] backend:  {effective_backend}")
    print(f"[reembed] model:    {target_model}")

    if not db_path.exists():
        print(f"[reembed] ERROR: db not found: {db_path}", file=sys.stderr)
        return 1

    db = sqlite3.connect(str(db_path))
    db.row_factory = sqlite3.Row
    try:
        total_rows, with_embed = _count_rows(db)
        targets = list(_iter_targets(db, target_model))
        to_process = len(targets)
        print(f"[reembed] knowledge rows:      {total_rows}")
        print(f"[reembed] rows with embedding: {with_embed}")
        print(f"[reembed] rows to re-embed:    {to_process}")
        print(f"[reembed] batch size:          {batch_size}")

        if to_process == 0:
            print("[reembed] nothing to do; all rows already use target model.")
            return 0

        # Cost estimate — printed for cloud providers regardless of dry-run,
        # so operators can sanity-check the bill before flipping --confirm.
        is_openai = (provider_name or "").strip().lower() == "openai"
        if is_openai:
            texts_for_estimate = [content for _, content in targets]
            tokens = estimate_tokens(texts_for_estimate)
            cost_usd = estimate_cost_usd(target_model, tokens)
            rate = _OPENAI_COST_USD_PER_1M_TOKENS.get(target_model, 0.0)
            print(
                f"[reembed] estimated tokens:    {tokens:,} "
                f"(~{_CHARS_PER_TOKEN_HEURISTIC:.0f} chars/token heuristic)"
            )
            print(
                f"[reembed] estimated cost:      ${cost_usd:,.2f} "
                f"@ ${rate:.2f}/1M tokens for {target_model}"
            )

        if dry_run or not confirm:
            print(
                "[reembed] --confirm not given; exiting without writes.\n"
                "          Re-run with --confirm to perform the re-embedding."
            )
            return 2

        if not provider.available():  # type: ignore[union-attr]
            print(
                f"[reembed] ERROR: provider for backend {effective_backend!r} is unavailable."
                " For OpenAI: ensure MEMORY_EMBED_API_KEY (or OPENAI_API_KEY) is set."
                " For FastEmbed/ST: install the library and let the model download.",
                file=sys.stderr,
            )
            return 1

        # Interactive guardrail for cloud spend. `--yes` (or non-tty stdin)
        # skips the prompt — useful for CI, but the cost line above is still
        # printed so the run is auditable in logs.
        if is_openai and not yes and sys.stdin.isatty():
            if not _confirm_interactive(
                f"[reembed] proceed with OpenAI run (~${cost_usd:,.2f})?"
            ):
                print("[reembed] aborted by user.")
                return 2

        now_iso = time.strftime("%Y-%m-%dT%H:%M:%S")
        done = 0
        start = time.time()

        for start_idx in range(0, to_process, batch_size):
            batch = targets[start_idx : start_idx + batch_size]
            texts = [content for _, content in batch]
            vectors = provider.embed(texts)  # type: ignore[union-attr]
            _persist_batch(db, batch, vectors, target_model, now_iso)
            done += len(batch)
            sys.stdout.write(_progress_line(done, to_process, time.time() - start))
            sys.stdout.flush()

        sys.stdout.write("\n")
        print(f"[reembed] done. {done} rows re-embedded in {time.time() - start:.1f}s.")
        return 0
    finally:
        db.close()


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Re-embed knowledge rows under the current V9_EMBED_BACKEND.",
    )
    p.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB,
        help=f"Path to memory.db (default: {DEFAULT_DB})",
    )
    p.add_argument(
        "--backend",
        type=str,
        default=None,
        help=(
            "Override V9_EMBED_BACKEND. Local: fastembed|minilm|e5-large|bge-m3| "
            "locomo-tuned-minilm. Cloud: openai-3-small|openai-3-large "
            "(requires MEMORY_EMBED_API_KEY)."
        ),
    )
    p.add_argument(
        "--provider",
        type=str,
        default=None,
        choices=("openai", "fastembed", "cohere"),
        help=(
            "Direct provider override (skips the V9_EMBED_BACKEND table). "
            "Use with --model to pin a specific model, e.g. "
            "`--provider openai --model text-embedding-3-large`. Reads "
            "MEMORY_EMBED_API_KEY / OPENAI_API_KEY from env when openai."
        ),
    )
    p.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            "Embedding model name. Required only if --provider is set and "
            "the provider default isn't what you want; e.g. "
            "`text-embedding-3-large` (3072d) or `text-embedding-3-small` "
            "(1536d) for openai."
        ),
    )
    p.add_argument(
        "--batch",
        dest="batch_size",
        type=int,
        default=64,
        help="Embedding batch size (default: 64).",
    )
    p.add_argument(
        "--confirm",
        action="store_true",
        help="Required to actually write to the DB. Without it the script just reports.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Report counts (and OpenAI token/cost estimate) only; never "
            "touch the DB and never call any external API. Equivalent to "
            "omitting --confirm but more explicit in CI."
        ),
    )
    p.add_argument(
        "--yes",
        action="store_true",
        help=(
            "Skip the interactive cost-confirmation prompt for cloud "
            "providers. Implied when stdin is not a TTY (CI runs)."
        ),
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    return run(
        db_path=args.db,
        backend=args.backend,
        batch_size=args.batch_size,
        confirm=args.confirm,
        dry_run=args.dry_run,
        provider_name=args.provider,
        model=args.model,
        yes=args.yes,
    )


if __name__ == "__main__":
    raise SystemExit(main())
