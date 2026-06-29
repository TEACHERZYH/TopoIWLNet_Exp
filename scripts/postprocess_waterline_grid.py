#!/usr/bin/env python
"""Validation-selected final-waterline post-processing grid.

The script is intentionally versioned by output directory: it refuses to write
into a non-empty directory. Create a new version name and output directory for
every formal experiment so previous versions remain available for comparison.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from topoiwl.config import ensure_dir, load_config  # noqa: E402
from topoiwl.data.dataset import WaterlineDataset  # noqa: E402
from topoiwl.metrics import confusion_metrics, topology_stats  # noqa: E402
from topoiwl.models.topoiwl_net import build_model  # noqa: E402
from topoiwl.utils.io import save_prob  # noqa: E402
from topoiwl.utils.morphology import binary_dilation, binary_erosion, distance_transform_edt, label  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--version-name", required=True)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--mask-thresholds", default="0.55,0.65,0.75")
    parser.add_argument("--final-thresholds", default="0.55,0.65,0.75")
    parser.add_argument("--alphas", default="0.30,0.40,0.50")
    parser.add_argument("--mask-boundary-widths", default="1,2")
    parser.add_argument("--mask-buffer-iters", default="0,1")
    parser.add_argument("--gap-bridge-iters", default="0,1")
    parser.add_argument("--min-component-sizes", default="64,128,256")
    parser.add_argument("--max-components", default="0,1,2")
    parser.add_argument("--select", default="composite", choices=["composite", "bf1_3", "chamfer", "hausdorff"])
    parser.add_argument("--save-predictions", action="store_true")
    return parser.parse_args()


def parse_float_list(text: str) -> list[float]:
    return [float(part.strip()) for part in text.split(",") if part.strip()]


def parse_int_list(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def build_dataset_config(cfg: dict, dataset_root: Path | None) -> dict:
    ds_cfg = dict(cfg["dataset"])
    if dataset_root is not None:
        root = Path(dataset_root)
        ds_cfg["root"] = str(root)
        ds_cfg["train_split"] = str(root / "splits" / "train.csv")
        ds_cfg["val_split"] = str(root / "splits" / "val.csv")
        ds_cfg["test_split"] = str(root / "splits" / "test.csv")
    return ds_cfg


def mask_to_boundary(mask: np.ndarray, width: int) -> np.ndarray:
    mask = mask.astype(bool)
    dilated = binary_dilation(mask, iterations=width)
    eroded = binary_erosion(mask, iterations=width)
    return np.logical_and(dilated, ~eroded)


def bridge_gaps(mask: np.ndarray, iters: int) -> np.ndarray:
    if iters <= 0:
        return mask.astype(bool)
    out = binary_dilation(mask.astype(bool), iterations=iters)
    out = binary_erosion(out, iterations=iters)
    return out.astype(bool)


def remove_small_components(mask: np.ndarray, min_size: int) -> np.ndarray:
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


def keep_largest_components(mask: np.ndarray, max_components: int) -> np.ndarray:
    if max_components <= 0:
        return mask.astype(bool)
    labels, n_labels = label(mask)
    if n_labels <= max_components:
        return mask.astype(bool)
    sizes: list[tuple[int, int]] = []
    for idx in range(1, n_labels + 1):
        sizes.append((int((labels == idx).sum()), idx))
    keep = {idx for _, idx in sorted(sizes, reverse=True)[:max_components]}
    return np.isin(labels, list(keep))


def fuse_boundary(sample: dict[str, np.ndarray], params: dict[str, float | int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask_pred = sample["mask_prob"] >= float(params["mask_threshold"])
    mask_boundary = mask_to_boundary(mask_pred, int(params["mask_boundary_width"]))
    final_prob = float(params["alpha"]) * sample["boundary_prob"] + (1.0 - float(params["alpha"])) * mask_boundary.astype(np.float32)
    final_pred = final_prob >= float(params["final_threshold"])
    if int(params["mask_buffer_iters"]) > 0:
        gate = binary_dilation(mask_boundary, iterations=int(params["mask_buffer_iters"]))
        final_pred = np.logical_and(final_pred, gate)
    final_pred = bridge_gaps(final_pred, int(params["gap_bridge_iters"]))
    final_pred = remove_small_components(final_pred, int(params["min_component_size"]))
    final_pred = keep_largest_components(final_pred, int(params["max_components"]))
    return mask_pred, final_prob, final_pred


def evaluate_prediction(sample: dict[str, np.ndarray], mask_pred: np.ndarray, boundary_pred: np.ndarray, tolerances: list[int], main_tolerance: int) -> dict[str, float]:
    row: dict[str, float] = {}
    row.update(confusion_metrics(sample["mask_gt"], mask_pred))
    boundary_gt = sample["boundary_gt"].astype(bool)
    boundary_pred = boundary_pred.astype(bool)
    dt_gt = sample["boundary_gt_dt"]
    dt_pred = None
    if boundary_gt.sum() == 0 and boundary_pred.sum() == 0:
        for tol in tolerances:
            row[f"bf1_{tol}"] = 1.0
        row.update({"chamfer": 0.0, "assd": 0.0, "hausdorff": 0.0})
        row.update(topology_stats(boundary_gt, boundary_pred, main_tolerance))
        return row
    if boundary_gt.sum() == 0 or boundary_pred.sum() == 0:
        for tol in tolerances:
            row[f"bf1_{tol}"] = 0.0
        row.update({"chamfer": float("nan"), "assd": float("nan"), "hausdorff": float("nan")})
        row.update(topology_stats(boundary_gt, boundary_pred, main_tolerance))
        return row
    dt_pred = distance_transform_edt(~boundary_pred)
    p2g = dt_gt[boundary_pred]
    g2p = dt_pred[boundary_gt]
    for tol in tolerances:
        precision = float((p2g <= tol).sum()) / (float(boundary_pred.sum()) + 1e-6)
        recall = float((g2p <= tol).sum()) / (float(boundary_gt.sum()) + 1e-6)
        bf1 = 2 * precision * recall / (precision + recall + 1e-6)
        row[f"bf1_{tol}"] = bf1
    chamfer = float(p2g.mean() + g2p.mean())
    row.update({"chamfer": chamfer, "assd": 0.5 * chamfer, "hausdorff": float(max(p2g.max(), g2p.max()))})
    row.update(topology_stats(boundary_gt, boundary_pred, main_tolerance))
    return row


def mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = rows[0].keys()
    out: dict[str, float] = {}
    for key in keys:
        raw_values = np.array([row[key] for row in rows], dtype=np.float64)
        finite_values = raw_values[np.isfinite(raw_values)]
        out[key] = float(finite_values.mean()) if finite_values.size else float("nan")
        if key in {"chamfer", "assd", "hausdorff"}:
            out[f"{key}_nan_rate"] = float((~np.isfinite(raw_values)).mean())
    return out


def score_metrics(metrics: dict[str, float], select: str) -> float:
    if select == "bf1_3":
        return float(metrics.get("bf1_3", float("-inf")))
    if select == "chamfer":
        value = float(metrics.get("chamfer", float("inf")))
        return -value if np.isfinite(value) else float("-inf")
    if select == "hausdorff":
        value = float(metrics.get("hausdorff", float("inf")))
        return -value if np.isfinite(value) else float("-inf")
    bf1 = float(metrics.get("bf1_3", 0.0))
    chamfer = float(metrics.get("chamfer", 1e6))
    hausdorff = float(metrics.get("hausdorff", 1e6))
    comp = abs(float(metrics.get("component_diff", 0.0)))
    if not np.isfinite(chamfer):
        chamfer = 1e6
    if not np.isfinite(hausdorff):
        hausdorff = 1e6
    nan_penalty = 2.0 * float(metrics.get("chamfer_nan_rate", 0.0)) + 2.0 * float(metrics.get("hausdorff_nan_rate", 0.0))
    return bf1 - 0.003 * chamfer - 0.001 * hausdorff - 0.01 * comp - nan_penalty


def make_dataset(cfg: dict, ds_cfg: dict, split: str) -> WaterlineDataset:
    return WaterlineDataset(
        ds_cfg["root"],
        ds_cfg[f"{split}_split"],
        augment=False,
        input_channels=cfg["model"]["in_channels"],
        image_mean=ds_cfg.get("image_mean"),
        image_std=ds_cfg.get("image_std"),
    )


def collect_samples(model: torch.nn.Module, cfg: dict, ds_cfg: dict, split: str, max_samples: int | None, num_workers: int) -> list[dict[str, np.ndarray]]:
    ds = make_dataset(cfg, ds_cfg, split)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=num_workers)
    device = next(model.parameters()).device
    samples: list[dict[str, np.ndarray]] = []
    model.eval()
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
            samples[-1]["boundary_gt_dt"] = distance_transform_edt(~samples[-1]["boundary_gt"])
            if max_samples is not None and sample_index >= max_samples:
                break
    if not samples:
        raise RuntimeError(f"No samples collected for split={split}")
    return samples


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    ensure_dir(path.parent)
    fieldnames = list(rows[0].keys())
    for row in rows[1:]:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {args.out_dir}. "
            "Use a new version name and output directory instead of overwriting."
        )
    out_dir = ensure_dir(args.out_dir)
    cfg = load_config(args.config)
    ds_cfg = build_dataset_config(cfg, args.dataset_root)
    device = torch.device("cuda" if torch.cuda.is_available() and cfg["train"].get("use_cuda", True) else "cpu")

    model_cfg = dict(cfg["model"])
    model_cfg["pretrained"] = False
    model_cfg["pretrained_weights"] = None
    model = build_model(model_cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])

    val_samples = collect_samples(model, cfg, ds_cfg, "val", args.max_val_samples, args.num_workers)
    grid_params = []
    for values in product(
        parse_float_list(args.mask_thresholds),
        parse_float_list(args.final_thresholds),
        parse_float_list(args.alphas),
        parse_int_list(args.mask_boundary_widths),
        parse_int_list(args.mask_buffer_iters),
        parse_int_list(args.gap_bridge_iters),
        parse_int_list(args.min_component_sizes),
        parse_int_list(args.max_components),
    ):
        grid_params.append(
            {
                "mask_threshold": values[0],
                "final_threshold": values[1],
                "alpha": values[2],
                "mask_boundary_width": values[3],
                "mask_buffer_iters": values[4],
                "gap_bridge_iters": values[5],
                "min_component_size": values[6],
                "max_components": values[7],
            }
        )

    tolerances = [int(v) for v in cfg["eval"]["boundary_tolerances"]]
    main_tolerance = int(cfg["eval"]["main_tolerance"])
    sweep_rows: list[dict[str, object]] = []
    best_row: dict[str, object] | None = None
    best_score = float("-inf")
    for params in grid_params:
        metric_rows = []
        for sample in val_samples:
            mask_pred, _, boundary_pred = fuse_boundary(sample, params)
            metric_rows.append(evaluate_prediction(sample, mask_pred, boundary_pred, tolerances, main_tolerance))
        metrics = mean_metrics(metric_rows)
        score = score_metrics(metrics, args.select)
        row = {"version": args.version_name, "split": "val", "score": score, **params, **metrics}
        sweep_rows.append(row)
        if score > best_score:
            best_score = score
            best_row = row
    if best_row is None:
        raise RuntimeError("No best row selected")
    write_csv(out_dir / "postprocess_sweep_val.csv", sweep_rows)

    best_params = {key: best_row[key] for key in [
        "mask_threshold",
        "final_threshold",
        "alpha",
        "mask_boundary_width",
        "mask_buffer_iters",
        "gap_bridge_iters",
        "min_component_size",
        "max_components",
    ]}

    test_samples = collect_samples(model, cfg, ds_cfg, "test", args.max_test_samples, args.num_workers)
    test_rows: list[dict[str, object]] = []
    for sample in test_samples:
        mask_pred, final_prob, boundary_pred = fuse_boundary(sample, best_params)
        metrics = evaluate_prediction(sample, mask_pred, boundary_pred, tolerances, main_tolerance)
        test_rows.append({"stem": sample["stem"], **metrics})
        if args.save_predictions:
            save_prob(out_dir / "predictions" / "mask_pred" / f"{sample['stem']}.png", mask_pred.astype(np.float32))
            save_prob(out_dir / "predictions" / "fusion_prob" / f"{sample['stem']}.png", final_prob)
            save_prob(out_dir / "predictions" / "boundary_pred" / f"{sample['stem']}.png", boundary_pred.astype(np.float32))

    mean_row = {"stem": "MEAN", **mean_metrics([{k: float(v) for k, v in row.items() if k != "stem"} for row in test_rows])}
    test_rows.append(mean_row)
    write_csv(out_dir / "metrics_test_postprocess.csv", test_rows)

    summary = {
        "version": args.version_name,
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "dataset_root": str(ds_cfg["root"]),
        "select": args.select,
        "best_val_score": best_score,
        "best_params": best_params,
        "best_val_metrics": {key: best_row[key] for key in best_row if key not in {"version", "split", "score", *best_params.keys()}},
        "test_metrics": {key: mean_row[key] for key in mean_row if key != "stem"},
    }
    (out_dir / "postprocess_best_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote postprocess version: {out_dir}")
    print(json.dumps({"version": args.version_name, "best_val_score": best_score, "test_metrics": summary["test_metrics"]}, indent=2))


if __name__ == "__main__":
    main()
