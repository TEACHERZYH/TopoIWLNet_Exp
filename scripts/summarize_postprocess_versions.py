#!/usr/bin/env python
"""Summarize versioned post-processing experiments without modifying them."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments-root", type=Path, default=Path("experiments"))
    parser.add_argument("--pattern", default="remote_glh_postprocess_*")
    parser.add_argument("--out-csv", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows: list[dict[str, object]] = []
    for exp_dir in sorted(args.experiments_root.glob(args.pattern)):
        summary_path = exp_dir / "postprocess_best_summary.json"
        if not summary_path.exists():
            rows.append({"version": exp_dir.name, "status": "missing_summary", "path": str(exp_dir)})
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        metrics = summary.get("test_metrics", {})
        params = summary.get("best_params", {})
        rows.append(
            {
                "version": summary.get("version", exp_dir.name),
                "status": "completed",
                "path": str(exp_dir),
                "select": summary.get("select", ""),
                "best_val_score": summary.get("best_val_score", ""),
                "iou": metrics.get("iou", ""),
                "f1": metrics.get("f1", ""),
                "bf1_3": metrics.get("bf1_3", ""),
                "chamfer": metrics.get("chamfer", ""),
                "hausdorff": metrics.get("hausdorff", ""),
                "broken_segments": metrics.get("broken_segments", ""),
                "pred_components": metrics.get("pred_components", ""),
                "component_diff": metrics.get("component_diff", ""),
                "mask_threshold": params.get("mask_threshold", ""),
                "final_threshold": params.get("final_threshold", ""),
                "alpha": params.get("alpha", ""),
                "mask_boundary_width": params.get("mask_boundary_width", ""),
                "mask_buffer_iters": params.get("mask_buffer_iters", ""),
                "gap_bridge_iters": params.get("gap_bridge_iters", ""),
                "min_component_size": params.get("min_component_size", ""),
                "max_components": params.get("max_components", ""),
            }
        )
    if not rows:
        raise RuntimeError(f"No experiment directories matched {args.pattern}")
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    for row in rows[1:]:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with args.out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote version summary: {args.out_csv}")


if __name__ == "__main__":
    main()
