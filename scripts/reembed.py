#!/usr/bin/env python3
"""Re-embed all knowledge rows under the current V9_EMBED_BACKEND.

Idempotent: rows whose ``embed_model`` already matches the target model are
skipped. Batches requests (default 64) and writes back float32 + binary
quantized vectors into the existing ``embeddings`` table.

Usage:
    python scripts/reembed.py --confirm
    python scripts/reembed.py --backend bge-m3 --confirm
    python scripts/reembed.py --backend e5-large --batch 32 --confirm --db /path/to/memory.db

Safety:
    - Requires explicit ``--confirm``. Without it, exits with code 2 after
      printing what *would* happen.
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
) -> int:
    effective_backend = choose_embed.resolve_backend(backend)
    target_model = choose_embed.resolve_model_name(effective_backend)

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

        if dry_run or not confirm:
            print(
                "[reembed] --confirm not given; exiting without writes.\n"
                "          Re-run with --confirm to perform the re-embedding."
            )
            return 2

        provider = choose_embed.get_provider(effective_backend)
        if not provider.available():
            print(
                f"[reembed] ERROR: provider for backend {effective_backend!r} is unavailable."
                " Check that the required library (fastembed / sentence-transformers)"
                " is installed and the model can be downloaded.",
                file=sys.stderr,
            )
            return 1

        now_iso = time.strftime("%Y-%m-%dT%H:%M:%S")
        done = 0
        start = time.time()

        for start_idx in range(0, to_process, batch_size):
            batch = targets[start_idx : start_idx + batch_size]
            texts = [content for _, content in batch]
            vectors = provider.embed(texts)
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
        help="Report counts only; never touch the DB (same as omitting --confirm).",
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
    )


if __name__ == "__main__":
    raise SystemExit(main())
