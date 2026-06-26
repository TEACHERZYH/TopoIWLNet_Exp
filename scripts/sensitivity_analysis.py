#!/usr/bin/env python
"""Threshold and radius sensitivity analysis for trained TopoIWL-Net models."""

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from topoiwl.config import ensure_dir, load_config  # noqa: E402
from topoiwl.data.dataset import WaterlineDataset  # noqa: E402
from topoiwl.metrics import boundary_f1, confusion_metrics, distance_metrics  # noqa: E402
from topoiwl.models.topoiwl_net import build_model  # noqa: E402
from topoiwl.utils.morphology import binary_dilation  # noqa: E402


def parse_list(text: str, cast=float) -> List:
    return [cast(part.strip()) for part in text.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--mask-thresholds", default="0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80")
    parser.add_argument("--boundary-thresholds", default="0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90")
    parser.add_argument("--radii", default="1,2,3,5,7,10")
    parser.add_argument("--existing-threshold-grid", type=Path, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=2)
    return parser.parse_args()


def finite_mean(values: List[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else float("nan")


def mean_dict(rows: List[Dict[str, float]]) -> Dict[str, float]:
    return {key: finite_mean([row[key] for row in rows]) for key in rows[0].keys()}


def buffered_match(gt: np.ndarray, pred: np.ndarray, radius: int, eps: float = 1e-6) -> Tuple[float, float, float]:
    gt = gt.astype(bool)
    pred = pred.astype(bool)
    if gt.sum() == 0 and pred.sum() == 0:
        return 1.0, 1.0, 1.0
    if gt.sum() == 0 or pred.sum() == 0:
        return 0.0, 0.0, 0.0
    gt_buf = binary_dilation(gt, iterations=radius)
    pred_buf = binary_dilation(pred, iterations=radius)
    precision = float((pred & gt_buf).sum()) / (float(pred.sum()) + eps)
    recall = float((gt & pred_buf).sum()) / (float(gt.sum()) + eps)
    f1 = 2.0 * precision * recall / (precision + recall + eps)
    return precision, recall, f1


def write_csv(path: Path, rows: List[Dict]) -> None:
    ensure_dir(path.parent)
    if not rows:
        raise RuntimeError(f"No rows to write for {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def best_from_existing_threshold_grid(path: Path) -> Dict:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"No threshold rows found in {path}")
    best_mask = max(rows, key=lambda row: float(row["iou"]))
    best_boundary = max(rows, key=lambda row: float(row.get("bf1_3", row.get("bf1", 0.0))))
    result = {
        "best_mask_threshold_by_iou": float(best_mask["mask_threshold"]),
        "best_val_iou": float(best_mask["iou"]),
        "best_val_f1": float(best_mask["f1"]),
        "best_boundary_threshold_by_bf1_3": float(best_boundary["boundary_threshold"]),
        "best_val_bf1_3": float(best_boundary.get("bf1_3", best_boundary.get("bf1", 0.0))),
    }
    for key in ["chamfer", "hausdorff"]:
        if key in best_boundary:
            result[f"best_val_{key}_at_boundary_threshold"] = float(best_boundary[key])
    return result


def load_samples(cfg: Dict, checkpoint: Path, split: str, max_samples: Optional[int], num_workers: int) -> List[Dict]:
    split_key = f"{split}_split"
    ds_cfg = cfg["dataset"]
    if cfg.get("_dataset_root_override") is not None:
        root = Path(cfg["_dataset_root_override"])
        ds_cfg = dict(ds_cfg)
        ds_cfg["root"] = str(root)
        ds_cfg["train_split"] = str(root / "splits" / "train.csv")
        ds_cfg["val_split"] = str(root / "splits" / "val.csv")
        ds_cfg["test_split"] = str(root / "splits" / "test.csv")
    ds = WaterlineDataset(
        ds_cfg["root"],
        ds_cfg[split_key],
        augment=False,
        input_channels=cfg["model"]["in_channels"],
        image_mean=ds_cfg.get("image_mean"),
        image_std=ds_cfg.get("image_std"),
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=num_workers)
    device = torch.device("cuda" if torch.cuda.is_available() and cfg["train"].get("use_cuda", True) else "cpu")
    model_cfg = dict(cfg["model"])
    model_cfg["pretrained"] = False
    model_cfg["pretrained_weights"] = None
    model = build_model(model_cfg).to(device)
    ckpt = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    samples = []
    with torch.no_grad():
        for sample_index, batch in enumerate(loader, start=1):
            pred = model(batch["image"].to(device))
            samples.append(
                {
                    "stem": batch["stem"][0],
                    "mask_prob": torch.sigmoid(pred["mask"])[0, 0].cpu().numpy(),
                    "boundary_prob": torch.sigmoid(pred["boundary"])[0, 0].cpu().numpy(),
                    "mask_gt": batch["mask"][0, 0].numpy() > 0.5,
                    "boundary_gt": batch["boundary"][0, 0].numpy() > 0.5,
                }
            )
            if max_samples and sample_index >= max_samples:
                break
    if not samples:
        raise RuntimeError(f"No {split} samples loaded")
    return samples


def threshold_grid(samples: List[Dict], mask_thresholds: List[float], boundary_thresholds: List[float]) -> Tuple[List[Dict], Dict]:
    mask_rows = {}
    for mask_th in mask_thresholds:
        mask_rows[mask_th] = mean_dict(
            [confusion_metrics(sample["mask_gt"], sample["mask_prob"] >= mask_th) for sample in samples]
        )

    boundary_rows = {}
    for boundary_th in boundary_thresholds:
        metric_rows = []
        for sample in samples:
            pred = sample["boundary_prob"] >= boundary_th
            gt = sample["boundary_gt"]
            p3, r3, bf3 = boundary_f1(gt, pred, 3)
            dist = distance_metrics(gt, pred)
            metric_rows.append(
                {
                    "boundary_precision_3": p3,
                    "boundary_recall_3": r3,
                    "bf1_3": bf3,
                    "chamfer": dist["chamfer"],
                    "hausdorff": dist["hausdorff"],
                }
            )
        boundary_rows[boundary_th] = mean_dict(metric_rows)

    rows = []
    for mask_th in mask_thresholds:
        for boundary_th in boundary_thresholds:
            row = {
                "mask_threshold": mask_th,
                "boundary_threshold": boundary_th,
                **mask_rows[mask_th],
                **boundary_rows[boundary_th],
            }
            rows.append(row)

    best_mask = max(mask_rows.items(), key=lambda item: item[1]["iou"])
    best_boundary = max(boundary_rows.items(), key=lambda item: item[1]["bf1_3"])
    best = {
        "best_mask_threshold_by_iou": float(best_mask[0]),
        "best_val_iou": float(best_mask[1]["iou"]),
        "best_val_f1": float(best_mask[1]["f1"]),
        "best_boundary_threshold_by_bf1_3": float(best_boundary[0]),
        "best_val_bf1_3": float(best_boundary[1]["bf1_3"]),
        "best_val_chamfer_at_boundary_threshold": float(best_boundary[1]["chamfer"]),
        "best_val_hausdorff_at_boundary_threshold": float(best_boundary[1]["hausdorff"]),
    }
    return rows, best


def tolerance_radius_rows(samples: List[Dict], boundary_threshold: float, radii: List[int]) -> List[Dict]:
    rows = []
    for radius in radii:
        metric_rows = []
        for sample in samples:
            pred = sample["boundary_prob"] >= boundary_threshold
            p, r, f1 = boundary_f1(sample["boundary_gt"], pred, radius)
            metric_rows.append({"precision": p, "recall": r, "bf1": f1})
        mean = mean_dict(metric_rows)
        rows.append({"radius_px": radius, **mean})
    return rows


def buffer_radius_rows(samples: List[Dict], boundary_threshold: float, radii: List[int]) -> List[Dict]:
    rows = []
    for radius in radii:
        metric_rows = []
        for sample in samples:
            pred = sample["boundary_prob"] >= boundary_threshold
            p, r, f1 = buffered_match(sample["boundary_gt"], pred, radius)
            metric_rows.append({"topology_precision": p, "topology_recall": r, "topology_f1": f1})
        mean = mean_dict(metric_rows)
        rows.append({"buffer_radius_px": radius, **mean})
    return rows


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.dataset_root is not None:
        cfg["_dataset_root_override"] = str(args.dataset_root)
    mask_thresholds = parse_list(args.mask_thresholds, float)
    boundary_thresholds = parse_list(args.boundary_thresholds, float)
    radii = parse_list(args.radii, int)
    ensure_dir(args.out_dir)

    if args.existing_threshold_grid is not None:
        shutil.copy2(args.existing_threshold_grid, args.out_dir / "threshold_grid_val.csv")
        best = best_from_existing_threshold_grid(args.existing_threshold_grid)
    else:
        val_samples = load_samples(cfg, args.checkpoint, "val", args.max_samples, args.num_workers)
        grid_rows, best = threshold_grid(val_samples, mask_thresholds, boundary_thresholds)
        write_csv(args.out_dir / "threshold_grid_val.csv", grid_rows)
    with (args.out_dir / "best_thresholds.json").open("w", encoding="utf-8") as handle:
        json.dump({"dataset": args.dataset_name, **best}, handle, indent=2)

    test_samples = load_samples(cfg, args.checkpoint, "test", args.max_samples, args.num_workers)
    boundary_threshold = best["best_boundary_threshold_by_bf1_3"]
    tol_rows = tolerance_radius_rows(test_samples, boundary_threshold, radii)
    buf_rows = buffer_radius_rows(test_samples, boundary_threshold, radii)
    write_csv(args.out_dir / "tolerance_radius_test.csv", tol_rows)
    write_csv(args.out_dir / "buffer_radius_test.csv", buf_rows)

    summary_rows = []
    for row in tol_rows:
        summary_rows.append(
            {
                "dataset": args.dataset_name,
                "analysis": "bf1_tolerance_radius",
                "radius_px": row["radius_px"],
                "precision": row["precision"],
                "recall": row["recall"],
                "f1": row["bf1"],
                "selected_boundary_threshold": boundary_threshold,
            }
        )
    for row in buf_rows:
        summary_rows.append(
            {
                "dataset": args.dataset_name,
                "analysis": "buffered_topology_radius",
                "radius_px": row["buffer_radius_px"],
                "precision": row["topology_precision"],
                "recall": row["topology_recall"],
                "f1": row["topology_f1"],
                "selected_boundary_threshold": boundary_threshold,
            }
        )
    write_csv(args.out_dir / "sensitivity_summary.csv", summary_rows)
    print(f"Wrote sensitivity outputs to {args.out_dir}")
    print(json.dumps({"dataset": args.dataset_name, **best}, indent=2))


if __name__ == "__main__":
    main()
