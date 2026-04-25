#!/usr/bin/env python3
"""v9.0 D5 — fine-tune embedding model on LoCoMo conversational data.

Pipeline:
  1. Mine (query, positive, hard_negative) triplets from a held-in slice of
     LoCoMo conversations. Positive = concat of evidence turns. Hard negs =
     top-K non-evidence turns of the SAME conversation by base-model cosine
     similarity (so negs share topic / speakers / vocabulary with positive —
     classic "in-domain hard negatives" recipe).
  2. Train sentence-transformers with MultipleNegativesRankingLoss
     (anchor, positive [+ optional hard_neg]) — uses in-batch negatives plus
     explicit hard negatives. ~3-5x more efficient than triplet loss.
  3. Save to <output_dir> + write meta.json (train_conv_ids, base_model,
     mining_hash) so bench reports can prove no train/test leakage.

Usage:
    # mine triplets only
    python scripts/finetune_embedding.py mine \
        --output-jsonl ./models/locomo-tuned-minilm/triplets.jsonl \
        --train-conv-ids 0,1,2,3,4,5,6 \
        --hard-neg-per-pos 4

    # train (uses cached triplets if --output-jsonl already exists)
    python scripts/finetune_embedding.py train \
        --base-model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 \
        --triplets ./models/locomo-tuned-minilm/triplets.jsonl \
        --output-dir ./models/locomo-tuned-minilm \
        --epochs 3 --batch-size 32

    # one-shot (mine + train)
    python scripts/finetune_embedding.py all \
        --output-dir ./models/locomo-tuned-minilm \
        --train-conv-ids 0,1,2,3,4,5,6 \
        --epochs 3 --batch-size 32

Notes:
    * The model lives at <output_dir>/. Plug it in via:
          V9_EMBED_BACKEND=locomo-tuned-minilm
      (after the alias is registered in src/choose_embed.py / src/config.py
      — done in this commit).
    * Held-out conv ids (those NOT in --train-conv-ids) are the only legal
      eval set — run bench with `--samples 7,8,9` (or whatever's left).
    * No OpenAI / cloud dep. CPU works on macOS Apple Silicon (~30 min/epoch
      on M-class for 1.5k triplets); CUDA cuts it ~10x.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = ROOT / "benchmarks" / "data" / "locomo" / "data" / "locomo10.json"


# ──────────────────────────────────────────────
# Dataset utilities
# ──────────────────────────────────────────────


@dataclass
class Turn:
    dia_id: str
    speaker: str
    text: str
    session: int


@dataclass
class Sample:
    sample_id: str
    sample_idx: int
    turns: list[Turn] = field(default_factory=list)
    qa: list[dict] = field(default_factory=list)


def _iter_samples(path: Path) -> list[Sample]:
    raw = json.loads(path.read_text())
    out: list[Sample] = []
    for idx, sample in enumerate(raw):
        s = Sample(
            sample_id=str(sample.get("sample_id", idx)),
            sample_idx=idx,
            qa=list(sample.get("qa", [])),
        )
        conv = sample.get("conversation", {})
        session_keys = sorted(
            (k for k in conv if k.startswith("session_") and not k.endswith("_date_time")),
            key=lambda k: int(k.split("_")[1]),
        )
        for sk in session_keys:
            sess_idx = int(sk.split("_")[1])
            for turn in conv[sk]:
                s.turns.append(
                    Turn(
                        dia_id=str(turn.get("dia_id", "")),
                        speaker=str(turn.get("speaker", "")),
                        text=str(turn.get("text", "")),
                        session=sess_idx,
                    )
                )
        out.append(s)
    return out


def _parse_conv_ids(raw: str | None, default_count: int) -> list[int]:
    if not raw:
        # Default: first 70% of conversations.
        cutoff = max(1, int(default_count * 0.7))
        return list(range(cutoff))
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


# ──────────────────────────────────────────────
# Mining
# ──────────────────────────────────────────────


def _format_turn(t: Turn) -> str:
    return f"{t.speaker}: {t.text}"


def _gather_positive_text(sample: Sample, evidence: Sequence[str]) -> str:
    by_id = {t.dia_id: t for t in sample.turns}
    parts = []
    for ev in evidence:
        t = by_id.get(ev)
        if t is not None:
            parts.append(_format_turn(t))
    return "\n".join(parts).strip()


def _embed_corpus(model, texts: list[str], batch_size: int = 64):
    import numpy as np  # noqa: PLC0415

    out = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return np.asarray(out, dtype="float32")


def _cosine_topk(query_vec, doc_matrix, top_k: int):
    """Assume both inputs already L2-normalized; cosine = dot."""
    import numpy as np  # noqa: PLC0415

    sims = doc_matrix @ query_vec
    if top_k >= len(sims):
        idxs = np.argsort(-sims)
    else:
        # argpartition is faster but unsorted; finalize with argsort over the
        # top slice.
        part = np.argpartition(-sims, top_k)[:top_k]
        idxs = part[np.argsort(-sims[part])]
    return [(int(i), float(sims[i])) for i in idxs]


def mine_triplets(
    samples: list[Sample],
    train_conv_ids: list[int],
    base_model_name: str,
    hard_neg_per_pos: int,
    max_pos_chars: int = 500,
    seed: int = 1337,
) -> list[dict]:
    """Build (query, positive, hard_negative) triples. Returns dict-per-row."""
    rnd = random.Random(seed)
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]
    except ImportError as exc:
        raise SystemExit(
            "sentence-transformers is required for mining. "
            "Install: pip install sentence-transformers"
        ) from exc

    print(f"[finetune] loading base model: {base_model_name}")
    model = SentenceTransformer(base_model_name)

    rows: list[dict] = []
    for sample in samples:
        if sample.sample_idx not in train_conv_ids:
            continue
        if not sample.qa:
            continue

        # Embed all turns in this conversation once — reused across QAs.
        turn_texts = [_format_turn(t) for t in sample.turns]
        if not turn_texts:
            continue
        turn_matrix = _embed_corpus(model, turn_texts)

        for qa in sample.qa:
            evidence = list(qa.get("evidence") or [])
            if not evidence:
                # No evidence → no usable positive (open-domain / adversarial).
                continue
            question = str(qa.get("question", "")).strip()
            if not question:
                continue
            positive = _gather_positive_text(sample, evidence)
            if not positive:
                continue
            if len(positive) > max_pos_chars:
                positive = positive[:max_pos_chars]

            evidence_set = set(evidence)
            non_evidence_idx = [
                i for i, t in enumerate(sample.turns) if t.dia_id not in evidence_set
            ]
            if not non_evidence_idx:
                continue

            # Mine top-K nearest non-evidence turns to the question — these
            # are the *hardest* negatives because they look semantically
            # close to the query but don't contain the answer.
            qvec = _embed_corpus(model, [question])[0]
            sub_matrix = turn_matrix[non_evidence_idx]
            top = _cosine_topk(qvec, sub_matrix, top_k=hard_neg_per_pos)

            for sub_pos, _sim in top:
                neg_turn_idx = non_evidence_idx[sub_pos]
                negative = turn_texts[neg_turn_idx]
                rows.append(
                    {
                        "anchor": question,
                        "positive": positive,
                        "negative": negative,
                        "category": qa.get("category"),
                        "sample_idx": sample.sample_idx,
                    }
                )

    rnd.shuffle(rows)
    return rows


def write_triplets(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_triplets(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ──────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────


def train(
    triplets: list[dict],
    base_model_name: str,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float = 2e-5,
    warmup_ratio: float = 0.1,
) -> None:
    if not triplets:
        raise SystemExit("No triplets — run `mine` first or check --triplets path.")

    try:
        from sentence_transformers import (  # type: ignore[import-not-found]
            InputExample,
            SentenceTransformer,
            losses,
        )
        from torch.utils.data import DataLoader  # type: ignore[import-not-found]
    except ImportError as exc:
        raise SystemExit(
            "Install training deps: pip install 'sentence-transformers>=2.2' torch"
        ) from exc

    print(f"[finetune] base_model={base_model_name}")
    print(f"[finetune] triplets={len(triplets)}  epochs={epochs}  batch_size={batch_size}")

    model = SentenceTransformer(base_model_name)
    examples = [
        InputExample(texts=[r["anchor"], r["positive"], r["negative"]])
        for r in triplets
    ]
    loader = DataLoader(examples, shuffle=True, batch_size=batch_size)
    loss = losses.MultipleNegativesRankingLoss(model)

    warmup_steps = max(1, int(len(loader) * epochs * warmup_ratio))

    output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    model.fit(
        train_objectives=[(loader, loss)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": learning_rate},
        output_path=str(output_dir),
        show_progress_bar=True,
    )
    elapsed = time.time() - t0
    print(f"[finetune] training done in {elapsed:.0f}s → {output_dir}")


def write_meta(
    output_dir: Path,
    base_model_name: str,
    train_conv_ids: list[int],
    triplet_path: Path,
    n_triplets: int,
    hard_neg_per_pos: int,
) -> None:
    triplet_hash = ""
    if triplet_path.exists():
        h = hashlib.sha256()
        with triplet_path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(64 * 1024), b""):
                h.update(chunk)
        triplet_hash = h.hexdigest()[:16]

    meta = {
        "base_model": base_model_name,
        "train_conv_ids": sorted(set(train_conv_ids)),
        "n_triplets": n_triplets,
        "hard_neg_per_pos": hard_neg_per_pos,
        "triplet_sha256_short": triplet_hash,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "framework": "sentence-transformers",
        "loss": "MultipleNegativesRankingLoss",
    }
    (output_dir / "locomo_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False)
    )


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    common.add_argument(
        "--base-model",
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )

    mine = sub.add_parser("mine", parents=[common], help="Mine triplets only.")
    mine.add_argument("--output-jsonl", type=Path, required=True)
    mine.add_argument("--train-conv-ids", type=str, default=None,
                      help="Comma-separated conv ids (default: first 70%).")
    mine.add_argument("--hard-neg-per-pos", type=int, default=4)
    mine.add_argument("--max-pos-chars", type=int, default=500)
    mine.add_argument("--seed", type=int, default=1337)

    train_p = sub.add_parser("train", parents=[common], help="Train on existing triplets.")
    train_p.add_argument("--triplets", type=Path, required=True)
    train_p.add_argument("--output-dir", type=Path, required=True)
    train_p.add_argument("--epochs", type=int, default=3)
    train_p.add_argument("--batch-size", type=int, default=32)
    train_p.add_argument("--lr", type=float, default=2e-5)
    train_p.add_argument("--warmup-ratio", type=float, default=0.1)

    all_p = sub.add_parser("all", parents=[common], help="Mine + train in one shot.")
    all_p.add_argument("--output-dir", type=Path, required=True)
    all_p.add_argument("--train-conv-ids", type=str, default=None)
    all_p.add_argument("--hard-neg-per-pos", type=int, default=4)
    all_p.add_argument("--max-pos-chars", type=int, default=500)
    all_p.add_argument("--seed", type=int, default=1337)
    all_p.add_argument("--epochs", type=int, default=3)
    all_p.add_argument("--batch-size", type=int, default=32)
    all_p.add_argument("--lr", type=float, default=2e-5)
    all_p.add_argument("--warmup-ratio", type=float, default=0.1)

    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    samples = _iter_samples(args.dataset)
    print(f"[finetune] loaded {len(samples)} conversations from {args.dataset}")

    if args.command == "mine":
        train_ids = _parse_conv_ids(args.train_conv_ids, len(samples))
        rows = mine_triplets(
            samples,
            train_conv_ids=train_ids,
            base_model_name=args.base_model,
            hard_neg_per_pos=args.hard_neg_per_pos,
            max_pos_chars=args.max_pos_chars,
            seed=args.seed,
        )
        write_triplets(rows, args.output_jsonl)
        print(f"[finetune] wrote {len(rows)} triplets → {args.output_jsonl}")
        return 0

    if args.command == "train":
        rows = read_triplets(args.triplets)
        train(
            triplets=rows,
            base_model_name=args.base_model,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            warmup_ratio=args.warmup_ratio,
        )
        return 0

    if args.command == "all":
        train_ids = _parse_conv_ids(args.train_conv_ids, len(samples))
        triplets_path = args.output_dir / "triplets.jsonl"
        rows = mine_triplets(
            samples,
            train_conv_ids=train_ids,
            base_model_name=args.base_model,
            hard_neg_per_pos=args.hard_neg_per_pos,
            max_pos_chars=args.max_pos_chars,
            seed=args.seed,
        )
        write_triplets(rows, triplets_path)
        print(f"[finetune] mined {len(rows)} triplets")
        train(
            triplets=rows,
            base_model_name=args.base_model,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            warmup_ratio=args.warmup_ratio,
        )
        write_meta(
            output_dir=args.output_dir,
            base_model_name=args.base_model,
            train_conv_ids=train_ids,
            triplet_path=triplets_path,
            n_triplets=len(rows),
            hard_neg_per_pos=args.hard_neg_per_pos,
        )
        held_out = sorted(set(range(len(samples))) - set(train_ids))
        print(f"[finetune] held-out conv ids (legal eval): {held_out}")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
