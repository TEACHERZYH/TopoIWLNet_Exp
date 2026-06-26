#!/usr/bin/env python
"""Run baseline training and the standard post-training evaluation pipeline."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from topoiwl.config import load_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--skip-train-if-done", action="store_true")
    parser.add_argument("--predict-samples", type=int, default=12)
    parser.add_argument("--efficiency-iters", type=int, default=500)
    return parser.parse_args()


def run(cmd: list[str]) -> None:
    print("RUN", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def best_thresholds(path: Path) -> tuple[float, float]:
    with path.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"No rows found in {path}")
    best = max(rows, key=lambda row: float(row["bf1_3"]))
    return float(best["mask_threshold"]), float(best["boundary_threshold"])


def training_is_complete(exp_dir: Path, expected_epochs: int) -> bool:
    checkpoint = exp_dir / "last.pt"
    marker = exp_dir / "training_complete.txt"
    if not checkpoint.exists() or not marker.exists():
        return False
    try:
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        try:
            state = torch.load(checkpoint, map_location="cpu")
        except Exception:
            return False
    except Exception:
        return False
    return int(state.get("epoch", 0)) >= expected_epochs


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    exp_dir = Path(cfg["output"]["exp_dir"])
    checkpoint = exp_dir / "best.pt"
    expected_epochs = int(cfg["train"]["epochs"])
    if not (args.skip_train_if_done and checkpoint.exists() and training_is_complete(exp_dir, expected_epochs)):
        run([sys.executable, "scripts/train_baseline.py", "--config", str(args.config)])

    sweep_csv = exp_dir / "threshold_sweep_val_fast.csv"
    run(
        [
            sys.executable,
            "scripts/sweep_thresholds_fast.py",
            "--config",
            str(args.config),
            "--checkpoint",
            str(checkpoint),
            "--split",
            "val",
            "--boundary-source",
            "mask",
            "--out-csv",
            str(sweep_csv),
        ]
    )
    mask_threshold, _ = best_thresholds(sweep_csv)
    run(
        [
            sys.executable,
            "scripts/evaluate.py",
            "--config",
            str(args.config),
            "--checkpoint",
            str(checkpoint),
            "--split",
            "test",
            "--boundary-source",
            "mask",
            "--mask-threshold",
            str(mask_threshold),
            "--out-csv",
            str(exp_dir / "metrics_test_mask_boundary_val_best_bf1.csv"),
        ]
    )
    run(
        [
            sys.executable,
            "scripts/measure_efficiency.py",
            "--config",
            str(args.config),
            "--checkpoint",
            str(checkpoint),
            "--height",
            "512",
            "--width",
            "512",
            "--batch-size",
            "1",
            "--warmup",
            "100",
            "--iters",
            str(args.efficiency_iters),
            "--out-csv",
            str(exp_dir / "efficiency_v100_fp32_b1_512_fair.csv"),
        ]
    )
    out_dir = Path("outputs") / f"{exp_dir.name}_test_predictions"
    run(
        [
            sys.executable,
            "scripts/predict.py",
            "--config",
            str(args.config),
            "--checkpoint",
            str(checkpoint),
            "--split",
            "test",
            "--boundary-source",
            "mask",
            "--mask-threshold",
            str(mask_threshold),
            "--out-dir",
            str(out_dir),
            "--max-samples",
            str(args.predict_samples),
        ]
    )
    print(f"BASELINE_DONE {args.config} mask_threshold={mask_threshold}", flush=True)


if __name__ == "__main__":
    main()
