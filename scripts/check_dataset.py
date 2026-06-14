#!/usr/bin/env python
"""Check unified TopoIWL-Net dataset structure."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from topoiwl.utils.io import list_files  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    return parser.parse_args()


def stems(folder: Path) -> set[str]:
    if not folder.exists():
        return set()
    return {p.stem for p in list_files(folder)}


def main() -> None:
    args = parse_args()
    root = args.root
    required = {
        "images": stems(root / "images"),
        "masks": stems(root / "masks"),
        "boundary": stems(root / "boundary"),
        "distance_npy": stems(root / "distance_npy"),
    }
    for name, s in required.items():
        print(f"{name}: {len(s)} files")
    common = set.intersection(*required.values()) if all(required.values()) else set()
    print(f"matched samples: {len(common)}")
    for name, s in required.items():
        missing = sorted(common.symmetric_difference(s))[:10]
        if len(s) != len(common):
            print(f"warning: {name} has unmatched stems, examples: {missing}")


if __name__ == "__main__":
    main()

