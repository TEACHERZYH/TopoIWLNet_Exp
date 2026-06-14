"""Segmentation and waterline metrics."""

from __future__ import annotations

import math

import numpy as np

from topoiwl.utils.morphology import binary_dilation, distance_transform_edt, label


def confusion_metrics(gt: np.ndarray, pred: np.ndarray, eps: float = 1e-6) -> dict[str, float]:
    gt = gt.astype(bool)
    pred = pred.astype(bool)
    tp = float(np.logical_and(gt, pred).sum())
    fp = float(np.logical_and(~gt, pred).sum())
    fn = float(np.logical_and(gt, ~pred).sum())
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)
    return {"iou": iou, "f1": f1, "precision": precision, "recall": recall}


def boundary_f1(gt: np.ndarray, pred: np.ndarray, tolerance: int = 3, eps: float = 1e-6) -> tuple[float, float, float]:
    gt = gt.astype(bool)
    pred = pred.astype(bool)
    if gt.sum() == 0 and pred.sum() == 0:
        return 1.0, 1.0, 1.0
    if gt.sum() == 0 or pred.sum() == 0:
        return 0.0, 0.0, 0.0
    dt_gt = distance_transform_edt(~gt)
    dt_pred = distance_transform_edt(~pred)
    precision = float((dt_gt[pred] <= tolerance).sum()) / (float(pred.sum()) + eps)
    recall = float((dt_pred[gt] <= tolerance).sum()) / (float(gt.sum()) + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    return precision, recall, f1


def distance_metrics(gt: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    gt = gt.astype(bool)
    pred = pred.astype(bool)
    if gt.sum() == 0 and pred.sum() == 0:
        return {"chamfer": 0.0, "assd": 0.0, "hausdorff": 0.0}
    if gt.sum() == 0 or pred.sum() == 0:
        return {"chamfer": math.nan, "assd": math.nan, "hausdorff": math.nan}
    dt_gt = distance_transform_edt(~gt)
    dt_pred = distance_transform_edt(~pred)
    p2g = dt_gt[pred]
    g2p = dt_pred[gt]
    chamfer = float(p2g.mean() + g2p.mean())
    return {"chamfer": chamfer, "assd": 0.5 * chamfer, "hausdorff": float(max(p2g.max(), g2p.max()))}


def topology_stats(gt: np.ndarray, pred: np.ndarray, tolerance: int = 3) -> dict[str, float]:
    gt = gt.astype(bool)
    pred = pred.astype(bool)
    pred_buf = binary_dilation(pred, iterations=tolerance)
    missing = np.logical_and(gt, ~pred_buf)
    _, gap_count = label(missing)
    _, gt_components = label(gt)
    _, pred_components = label(pred)
    return {
        "broken_segments": float(gap_count),
        "gt_components": float(gt_components),
        "pred_components": float(pred_components),
        "component_diff": float(pred_components - gt_components),
    }
