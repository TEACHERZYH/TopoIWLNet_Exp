#!/usr/bin/env python
"""Create train/val/test CSV splits from a unified dataset."""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from topoiwl.utils.io import list_files  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def write_csv(path: Path, stems: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["stem"])
        writer.writeheader()
        for stem in stems:
            writer.writerow({"stem": stem})


def main() -> None:
    args = parse_args()
    stems = sorted(p.stem for p in list_files(args.root / "images"))
    random.seed(args.seed)
    random.shuffle(stems)
    n = len(stems)
    n_train = int(n * args.train_ratio)
    n_val = int(n * args.val_ratio)
    train = stems[:n_train]
    val = stems[n_train : n_train + n_val]
    test = stems[n_train + n_val :]
    write_csv(args.out_dir / "train.csv", train)
    write_csv(args.out_dir / "val.csv", val)
    write_csv(args.out_dir / "test.csv", test)
    print(f"Created splits in {args.out_dir}: train={len(train)}, val={len(val)}, test={len(test)}")


if __name__ == "__main__":
    main()

