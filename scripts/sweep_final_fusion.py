#!/usr/bin/env python
"""Sweep final waterline fusion parameters for trained TopoIWL-Net checkpoints."""

import argparse
import csv
import json
import math
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
from topoiwl.utils.morphology import binary_dilation, binary_erosion, label  # noqa: E402


def parse_list(text, cast):
    return [cast(part.strip()) for part in text.split(",") if part.strip()]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--mask-thresholds", default="0.30,0.40,0.50,0.60,0.70,0.80")
    parser.add_argument("--final-thresholds", default="0.20,0.30,0.40,0.50,0.60,0.70,0.80")
    parser.add_argument("--alphas", default="0.00,0.25,0.50,0.75,1.00")
    parser.add_argument("--mask-boundary-width", type=int, default=1)
    parser.add_argument("--mask-buffer-iters", default="0")
    parser.add_argument("--gap-bridge-iters", default="0")
    parser.add_argument("--min-component-sizes", default="0")
    parser.add_argument("--select-metric", default="bf1_3", choices=["bf1_1", "bf1_2", "bf1_3", "bf1_5"])
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--skip-test", action="store_true", help="Only write validation sweep results.")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=25)
    return parser.parse_args()


def mask_to_boundary(mask, width=1):
    mask = mask.astype(bool)
    dilated = binary_dilation(mask, iterations=width)
    eroded = binary_erosion(mask, iterations=width)
    return np.logical_and(dilated, ~eroded)


def remove_small_components(mask, min_size):
    if min_size <= 0:
        return mask.astype(bool)
    labels, n_labels = label(mask)
    if n_labels == 0:
        return mask.astype(bool)
    out = np.zeros(mask.shape, dtype=bool)
    for idx in range(1, n_labels + 1):
        comp = labels == idx
        if int(comp.sum()) >= min_size:
            out |= comp
    return out


def bridge_gaps(mask, iters):
    if iters <= 0:
        return mask.astype(bool)
    bridged = binary_dilation(mask.astype(bool), iterations=iters)
    bridged = binary_erosion(bridged, iterations=iters)
    return bridged.astype(bool)


def mean_row(rows):
    out = {}
    keys = [key for key in rows[0].keys() if key != "stem"]
    for key in keys:
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        values = values[np.isfinite(values)]
        out[key] = float(values.mean()) if values.size else math.nan
    return out


def build_dataset_config(cfg, dataset_root):
    ds_cfg = dict(cfg["dataset"])
    if dataset_root is not None:
        root = Path(dataset_root)
        ds_cfg["root"] = str(root)
        ds_cfg["train_split"] = str(root / "splits" / "train.csv")
        ds_cfg["val_split"] = str(root / "splits" / "val.csv")
        ds_cfg["test_split"] = str(root / "splits" / "test.csv")
    return ds_cfg


def load_samples(cfg, checkpoint, split, dataset_root, max_samples, num_workers):
    ds_cfg = build_dataset_config(cfg, dataset_root)
    split_key = f"{split}_split"
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
            if max_samples is not None and sample_index >= max_samples:
                break
    if not samples:
        raise RuntimeError(f"No samples loaded for split={split}")
    return samples


def predict_boundary(sample, mask_threshold, final_threshold, alpha, mask_boundary_width, mask_buffer_iters, gap_bridge_iters, min_component_size):
    mask_pred = sample["mask_prob"] >= mask_threshold
    mask_boundary = mask_to_boundary(mask_pred, width=mask_boundary_width)
    final_prob = alpha * sample["boundary_prob"] + (1.0 - alpha) * mask_boundary.astype(np.float32)
    pred = final_prob >= final_threshold
    if mask_buffer_iters > 0:
        gate = binary_dilation(mask_boundary, iterations=mask_buffer_iters)
        pred = np.logical_and(pred, gate)
    pred = bridge_gaps(pred, gap_bridge_iters)
    pred = remove_small_components(pred, min_component_size)
    return mask_pred, pred


def evaluate_setting(samples, cfg, params, full_metrics=False):
    rows = []
    for sample in samples:
        mask_pred, boundary_pred = predict_boundary(sample, **params)
        row = {}
        if full_metrics:
            row["stem"] = sample["stem"]
            row.update(confusion_metrics(sample["mask_gt"], mask_pred))
        tolerances = cfg["eval"].get("boundary_tolerances", [1, 2, 3, 5])
        for tol in tolerances:
            _, _, bf1 = boundary_f1(sample["boundary_gt"], boundary_pred, tol)
            row[f"bf1_{tol}"] = bf1
        if full_metrics:
            row.update(distance_metrics(sample["boundary_gt"], boundary_pred))
            row.update(topology_stats(sample["boundary_gt"], boundary_pred, cfg["eval"].get("main_tolerance", 3)))
        rows.append(row)
    return rows, mean_row(rows)


def write_csv(path, rows):
    ensure_dir(path.parent)
    if not rows:
        raise RuntimeError(f"No rows to write: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    ensure_dir(args.out_dir)

    mask_thresholds = parse_list(args.mask_thresholds, float)
    final_thresholds = parse_list(args.final_thresholds, float)
    alphas = parse_list(args.alphas, float)
    mask_buffer_iters_values = parse_list(args.mask_buffer_iters, int)
    gap_bridge_iters_values = parse_list(args.gap_bridge_iters, int)
    min_component_sizes = parse_list(args.min_component_sizes, int)

    max_val_samples = args.max_val_samples if args.max_val_samples is not None else args.max_samples
    max_test_samples = args.max_test_samples if args.max_test_samples is not None else args.max_samples
    val_samples = load_samples(cfg, args.checkpoint, "val", args.dataset_root, max_val_samples, args.num_workers)
    print(f"Loaded {len(val_samples)} validation samples", flush=True)
    sweep_rows = []
    best = None
    total_settings = (
        len(mask_thresholds)
        * len(final_thresholds)
        * len(alphas)
        * len(mask_buffer_iters_values)
        * len(gap_bridge_iters_values)
        * len(min_component_sizes)
    )
    setting_index = 0
    for mask_threshold in mask_thresholds:
        for final_threshold in final_thresholds:
            for alpha in alphas:
                for mask_buffer_iters in mask_buffer_iters_values:
                    for gap_bridge_iters in gap_bridge_iters_values:
                        for min_component_size in min_component_sizes:
                            setting_index += 1
                            params = {
                                "mask_threshold": mask_threshold,
                                "final_threshold": final_threshold,
                                "alpha": alpha,
                                "mask_boundary_width": args.mask_boundary_width,
                                "mask_buffer_iters": mask_buffer_iters,
                                "gap_bridge_iters": gap_bridge_iters,
                                "min_component_size": min_component_size,
                            }
                            _, metrics = evaluate_setting(val_samples, cfg, params, full_metrics=False)
                            row = {"dataset": args.dataset_name, **params, **metrics}
                            sweep_rows.append(row)
                            score = metrics.get(args.select_metric, -1.0)
                            if best is None or score > best["score"]:
                                best = {"score": score, "params": params, "metrics": metrics}
                            if args.progress_every > 0 and setting_index % args.progress_every == 0:
                                print(
                                    f"Sweep progress {setting_index}/{total_settings}; "
                                    f"current best {args.select_metric}={best['score']:.6f}",
                                    flush=True,
                                )

    write_csv(args.out_dir / "fusion_sweep_val.csv", sweep_rows)

    if args.skip_test:
        summary = {
            "dataset": args.dataset_name,
            "select_metric": args.select_metric,
            "best_val_score": best["score"],
            "best_params": best["params"],
            "best_val_metrics": best["metrics"],
            "test_metrics": None,
        }
        with (args.out_dir / "fusion_best_summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        print(json.dumps(summary, indent=2))
        return

    test_samples = load_samples(cfg, args.checkpoint, "test", args.dataset_root, max_test_samples, args.num_workers)
    print(f"Loaded {len(test_samples)} test samples", flush=True)

    test_rows, test_mean = evaluate_setting(test_samples, cfg, best["params"], full_metrics=True)
    test_rows_with_mean = list(test_rows)
    test_rows_with_mean.append({"stem": "MEAN", **test_mean})
    write_csv(args.out_dir / "metrics_test_fusion_val_best.csv", test_rows_with_mean)

    summary = {
        "dataset": args.dataset_name,
        "select_metric": args.select_metric,
        "best_val_score": best["score"],
        "best_params": best["params"],
        "best_val_metrics": best["metrics"],
        "test_metrics": test_mean,
    }
    with (args.out_dir / "fusion_best_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
