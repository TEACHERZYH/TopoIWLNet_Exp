#!/usr/bin/env python
"""Evaluate a trained TopoIWL-Net checkpoint."""

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
from topoiwl.metrics import boundary_f1, confusion_metrics, distance_metrics, topology_stats  # noqa: E402
from topoiwl.models.topoiwl_net import build_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--out-csv", type=Path, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


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

    rows = []
    with torch.no_grad():
        for sample_index, batch in enumerate(loader, start=1):
            image = batch["image"].to(device)
            pred = model(image)
            mask_prob = torch.sigmoid(pred["mask"])[0, 0].cpu().numpy()
            boundary_prob = torch.sigmoid(pred["boundary"])[0, 0].cpu().numpy()
            mask_pred = mask_prob >= cfg["eval"]["mask_threshold"]
            boundary_pred = boundary_prob >= cfg["eval"]["boundary_threshold"]
            mask_gt = batch["mask"][0, 0].numpy() > 0.5
            boundary_gt = batch["boundary"][0, 0].numpy() > 0.5
            row = {"stem": batch["stem"][0]}
            row.update(confusion_metrics(mask_gt, mask_pred))
            for tol in cfg["eval"]["boundary_tolerances"]:
                _, _, bf1 = boundary_f1(boundary_gt, boundary_pred, tol)
                row[f"bf1_{tol}"] = bf1
            row.update(distance_metrics(boundary_gt, boundary_pred))
            row.update(topology_stats(boundary_gt, boundary_pred, cfg["eval"]["main_tolerance"]))
            rows.append(row)
            if args.max_samples and sample_index >= args.max_samples:
                break

    out_csv = args.out_csv or Path(cfg["output"]["exp_dir"]) / f"metrics_{args.split}.csv"
    ensure_dir(out_csv.parent)
    if not rows:
        raise RuntimeError("No samples were evaluated")
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
        mean_row = {"stem": "MEAN"}
        for key in rows[0].keys():
            if key == "stem":
                continue
            values = np.array([r[key] for r in rows], dtype=np.float64)
            values = values[np.isfinite(values)]
            mean_row[key] = float(values.mean()) if values.size else np.nan
        writer.writerow(mean_row)
    print(f"Wrote metrics: {out_csv}")
    print(mean_row)


if __name__ == "__main__":
    main()
