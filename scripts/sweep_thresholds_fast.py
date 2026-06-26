#!/usr/bin/env python
"""Fast validation-threshold sweep for TopoIWL-Net checkpoints."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from topoiwl.config import ensure_dir, load_config  # noqa: E402
from topoiwl.data.dataset import WaterlineDataset  # noqa: E402
from topoiwl.metrics import boundary_f1, confusion_metrics  # noqa: E402
from topoiwl.models.topoiwl_net import build_model  # noqa: E402
from topoiwl.utils.morphology import binary_dilation, binary_erosion  # noqa: E402


def parse_thresholds(text: str) -> list[float]:
    return [float(part.strip()) for part in text.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--boundary-source", default="head", choices=["head", "mask"])
    parser.add_argument("--mask-thresholds", default="0.30,0.40,0.50,0.60,0.70,0.80")
    parser.add_argument("--boundary-thresholds", default="0.20,0.30,0.40,0.50,0.60,0.70,0.80")
    parser.add_argument("--boundary-tolerance", type=int, default=3)
    parser.add_argument("--mask-boundary-width", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--out-csv", required=True, type=Path)
    return parser.parse_args()


def mask_to_boundary(mask: np.ndarray, width: int = 1) -> np.ndarray:
    mask = mask.astype(bool)
    dilated = binary_dilation(mask, iterations=width)
    eroded = binary_erosion(mask, iterations=width)
    return np.logical_and(dilated, ~eroded)


def mean_dict(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = rows[0].keys()
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    split_key = f"{args.split}_split"
    ds_cfg = cfg["dataset"]
    ds = WaterlineDataset(
        ds_cfg["root"],
        ds_cfg[split_key],
        augment=False,
        input_channels=cfg["model"]["in_channels"],
        image_mean=ds_cfg.get("image_mean"),
        image_std=ds_cfg.get("image_std"),
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() and cfg["train"].get("use_cuda", True) else "cpu")
    model = build_model(cfg["model"]).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    samples: list[dict[str, np.ndarray]] = []
    with torch.no_grad():
        for sample_index, batch in enumerate(loader, start=1):
            pred = model(batch["image"].to(device))
            samples.append(
                {
                    "mask_prob": torch.sigmoid(pred["mask"])[0, 0].cpu().numpy(),
                    "boundary_prob": torch.sigmoid(pred["boundary"])[0, 0].cpu().numpy(),
                    "mask_gt": batch["mask"][0, 0].numpy() > 0.5,
                    "boundary_gt": batch["boundary"][0, 0].numpy() > 0.5,
                }
            )
            if args.max_samples and sample_index >= args.max_samples:
                break
    if not samples:
        raise RuntimeError("No samples were evaluated")

    rows: list[dict[str, float]] = []
    mask_thresholds = parse_thresholds(args.mask_thresholds)
    boundary_thresholds = parse_thresholds(args.boundary_thresholds)
    boundary_mean: dict[float, float] = {}
    if args.boundary_source == "head":
        for boundary_th in boundary_thresholds:
            bf1_values = []
            for sample in samples:
                _, _, bf1 = boundary_f1(
                    sample["boundary_gt"],
                    sample["boundary_prob"] >= boundary_th,
                    args.boundary_tolerance,
                )
                bf1_values.append(bf1)
            boundary_mean[boundary_th] = float(np.mean(bf1_values))
    for mask_th in mask_thresholds:
        mask_metrics = [confusion_metrics(sample["mask_gt"], sample["mask_prob"] >= mask_th) for sample in samples]
        mask_mean = mean_dict(mask_metrics)
        if args.boundary_source == "mask":
            bf1_values = []
            for sample in samples:
                boundary_pred = mask_to_boundary(sample["mask_prob"] >= mask_th, width=args.mask_boundary_width)
                _, _, bf1 = boundary_f1(sample["boundary_gt"], boundary_pred, args.boundary_tolerance)
                bf1_values.append(bf1)
            rows.append(
                {
                    "mask_threshold": mask_th,
                    "boundary_threshold": mask_th,
                    "iou": mask_mean["iou"],
                    "f1": mask_mean["f1"],
                    "precision": mask_mean["precision"],
                    "recall": mask_mean["recall"],
                    f"bf1_{args.boundary_tolerance}": float(np.mean(bf1_values)),
                }
            )
            continue
        for boundary_th in boundary_thresholds:
            rows.append(
                {
                    "mask_threshold": mask_th,
                    "boundary_threshold": boundary_th,
                    "iou": mask_mean["iou"],
                    "f1": mask_mean["f1"],
                    "precision": mask_mean["precision"],
                    "recall": mask_mean["recall"],
                    f"bf1_{args.boundary_tolerance}": boundary_mean[boundary_th],
                }
            )

    ensure_dir(args.out_csv.parent)
    with args.out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    best_iou = max(rows, key=lambda row: row["iou"])
    best_bf1 = max(rows, key=lambda row: row[f"bf1_{args.boundary_tolerance}"])
    print(f"Wrote threshold sweep: {args.out_csv}")
    print("best_iou", best_iou)
    print("best_bf1", best_bf1)


if __name__ == "__main__":
    main()
