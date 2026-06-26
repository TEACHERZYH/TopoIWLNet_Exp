#!/usr/bin/env python
"""Run a full TopoIWL-Net train/evaluate/predict workflow for one config."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--boundary-source", default="head", choices=["head", "mask"])
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-predict", action="store_true")
    return parser.parse_args()


def load_exp_dir(config_path: Path) -> Path:
    import yaml

    with config_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    return Path(cfg["output"]["exp_dir"])


def best_thresholds(csv_path: Path) -> tuple[float, float]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"No threshold rows found in {csv_path}")
    best = max(rows, key=lambda row: float(row.get("bf1_3", row.get("bf1", 0.0))))
    return float(best["mask_threshold"]), float(best["boundary_threshold"])


def main() -> None:
    args = parse_args()
    config_path = args.config
    exp_dir = load_exp_dir(config_path)
    checkpoint = exp_dir / "best.pt"
    sweep_csv = exp_dir / "threshold_sweep_val_fast.csv"
    metrics_csv = exp_dir / "metrics_test_val_best_bf1.csv"
    pred_dir = PROJECT_ROOT / "outputs" / f"{exp_dir.name}_test_predictions"

    if not args.skip_train:
        run([sys.executable, "scripts/train.py", "--config", str(config_path)])

    run(
        [
            sys.executable,
            "scripts/sweep_thresholds_fast.py",
            "--config",
            str(config_path),
            "--checkpoint",
            str(checkpoint),
            "--split",
            "val",
            "--boundary-source",
            args.boundary_source,
            "--out-csv",
            str(sweep_csv),
        ]
    )
    mask_threshold, boundary_threshold = best_thresholds(sweep_csv)
    print(f"best thresholds: mask={mask_threshold:.3f}, boundary={boundary_threshold:.3f}", flush=True)

    run(
        [
            sys.executable,
            "scripts/evaluate.py",
            "--config",
            str(config_path),
            "--checkpoint",
            str(checkpoint),
            "--split",
            "test",
            "--boundary-source",
            args.boundary_source,
            "--mask-threshold",
            str(mask_threshold),
            "--boundary-threshold",
            str(boundary_threshold),
            "--out-csv",
            str(metrics_csv),
        ]
    )

    if not args.skip_predict:
        run(
            [
                sys.executable,
                "scripts/predict.py",
                "--config",
                str(config_path),
                "--checkpoint",
                str(checkpoint),
                "--split",
                "test",
                "--boundary-source",
                args.boundary_source,
                "--mask-threshold",
                str(mask_threshold),
                "--out-dir",
                str(pred_dir),
            ]
        )


if __name__ == "__main__":
    main()
