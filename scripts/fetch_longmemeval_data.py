#!/usr/bin/env python3
"""Fetch LongMemEval dataset (cleaned variant) from HuggingFace into
`benchmarks/data/`. Idempotent: skips download if the expected file is already
present. Pass `--force` to redownload.

The dataset is ~264 MB and is intentionally excluded from the repository
(see `.gitignore`). CI jobs must call this script before running the
benchmark.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO_ID = "xiaowu0162/longmemeval-cleaned"
REPO_TYPE = "dataset"
TARGET_FILENAME = "longmemeval_s_cleaned.json"
DEFAULT_TARGET_DIR = (
    Path(__file__).resolve().parent.parent / "benchmarks" / "data"
)


def _snapshot_download(target_dir: Path) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print(
            "ERROR: huggingface_hub is not installed. "
            "Install with: pip install huggingface_hub",
            file=sys.stderr,
        )
        raise SystemExit(2)

    print(f"[fetch] Downloading {REPO_ID} -> {target_dir}")
    snapshot_path = snapshot_download(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        local_dir=str(target_dir),
        # Limit to the JSON variant we actually use to save bandwidth.
        allow_patterns=[TARGET_FILENAME, "*.json"],
    )
    return Path(snapshot_path)


def fetch(target_dir: Path, force: bool = False) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    target_file = target_dir / TARGET_FILENAME

    if target_file.exists() and not force:
        size_mb = target_file.stat().st_size / (1024 * 1024)
        print(
            f"[fetch] Already present: {target_file} ({size_mb:.1f} MB) — skip."
        )
        return target_file

    if force and target_file.exists():
        print(f"[fetch] --force: removing existing {target_file}")
        target_file.unlink()

    snapshot_path = _snapshot_download(target_dir)

    # snapshot_download may place the file under `target_dir` directly or at
    # the returned snapshot path. Normalize to the canonical target location.
    if not target_file.exists():
        candidate = Path(snapshot_path) / TARGET_FILENAME
        if candidate.exists() and candidate != target_file:
            shutil.copy2(candidate, target_file)

    if not target_file.exists():
        print(
            f"ERROR: expected {TARGET_FILENAME} was not produced under "
            f"{target_dir} after download.",
            file=sys.stderr,
        )
        raise SystemExit(2)

    size_mb = target_file.stat().st_size / (1024 * 1024)
    print(f"[fetch] OK: {target_file} ({size_mb:.1f} MB)")
    return target_file


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Download LongMemEval cleaned dataset from HuggingFace."
    )
    p.add_argument(
        "--target-dir",
        type=Path,
        default=DEFAULT_TARGET_DIR,
        help=f"Target directory (default: {DEFAULT_TARGET_DIR}).",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Redownload even if the file already exists.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    fetch(args.target_dir, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
