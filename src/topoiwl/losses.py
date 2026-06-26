"""Loss functions for TopoIWL-Net."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def dice_loss_with_logits(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    dims = tuple(range(2, prob.ndim))
    intersection = (prob * target).sum(dim=dims)
    union = prob.sum(dim=dims) + target.sum(dim=dims)
    dice = (2 * intersection + eps) / (union + eps)
    return 1.0 - dice.mean()


def weighted_bce_with_logits(logits: torch.Tensor, target: torch.Tensor, pos_weight: float = 1.0) -> torch.Tensor:
    weight = torch.ones_like(target)
    weight = torch.where(target > 0.5, weight * pos_weight, weight)
    return F.binary_cross_entropy_with_logits(logits, target, weight=weight)


def pyramid_mask_loss(logits: torch.Tensor, target: torch.Tensor, scales: tuple[int, ...] = (2, 4, 8)) -> torch.Tensor:
    """Multi-scale mask supervision for large-scene consistency."""

    losses = []
    height, width = target.shape[-2:]
    for scale in scales:
        if height // scale < 2 or width // scale < 2:
            continue
        size = (height // scale, width // scale)
        scaled_logits = F.interpolate(logits, size=size, mode="bilinear", align_corners=False)
        scaled_target = F.interpolate(target, size=size, mode="area")
        losses.append(F.binary_cross_entropy_with_logits(scaled_logits, scaled_target) + dice_loss_with_logits(scaled_logits, scaled_target))
    if not losses:
        return logits.new_tensor(0.0)
    return torch.stack(losses).mean()


def soft_erode(x: torch.Tensor) -> torch.Tensor:
    p1 = -F.max_pool2d(-x, kernel_size=(3, 1), stride=1, padding=(1, 0))
    p2 = -F.max_pool2d(-x, kernel_size=(1, 3), stride=1, padding=(0, 1))
    return torch.min(p1, p2)


def soft_dilate(x: torch.Tensor) -> torch.Tensor:
    return F.max_pool2d(x, kernel_size=3, stride=1, padding=1)


def soft_open(x: torch.Tensor) -> torch.Tensor:
    return soft_dilate(soft_erode(x))


def soft_skeletonize(x: torch.Tensor, iterations: int = 8) -> torch.Tensor:
    x = x.clamp(0, 1)
    skel = F.relu(x - soft_open(x))
    for _ in range(iterations):
        x = soft_erode(x)
        delta = F.relu(x - soft_open(x))
        skel = skel + F.relu(delta - skel * delta)
    return skel.clamp(0, 1)


def buffered_cldice_loss(logits: torch.Tensor, target_boundary: torch.Tensor, buffer_iters: int = 3, eps: float = 1e-6) -> torch.Tensor:
    pred = torch.sigmoid(logits)
    pred_skel = soft_skeletonize(pred)
    gt_skel = soft_skeletonize(target_boundary)
    gt_buffer = target_boundary
    pred_buffer = pred
    for _ in range(max(buffer_iters, 1)):
        gt_buffer = soft_dilate(gt_buffer)
        pred_buffer = soft_dilate(pred_buffer)
    tprec = (pred_skel * gt_buffer).sum(dim=(2, 3)) / (pred_skel.sum(dim=(2, 3)) + eps)
    trec = (gt_skel * pred_buffer).sum(dim=(2, 3)) / (gt_skel.sum(dim=(2, 3)) + eps)
    cldice = (2 * tprec * trec + eps) / (tprec + trec + eps)
    return 1.0 - cldice.mean()


class TopoIWLLoss(nn.Module):
    def __init__(
        self,
        lambda_boundary: float = 1.0,
        lambda_distance: float = 0.5,
        lambda_topology: float = 0.2,
        boundary_pos_weight: float = 8.0,
        topology_buffer_iters: int = 3,
        lambda_pyramid: float = 0.0,
        pyramid_scales: tuple[int, ...] | list[int] = (2, 4, 8),
    ) -> None:
        super().__init__()
        self.lambda_boundary = lambda_boundary
        self.lambda_distance = lambda_distance
        self.lambda_topology = lambda_topology
        self.boundary_pos_weight = boundary_pos_weight
        self.topology_buffer_iters = topology_buffer_iters
        self.lambda_pyramid = lambda_pyramid
        self.pyramid_scales = tuple(int(scale) for scale in pyramid_scales)

    def forward(self, pred: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        mask_target = batch["mask"]
        boundary_target = batch["boundary"]
        distance_target = batch["distance"].clamp(-1, 1)

        mask_loss = F.binary_cross_entropy_with_logits(pred["mask"], mask_target) + dice_loss_with_logits(pred["mask"], mask_target)
        boundary_loss = weighted_bce_with_logits(pred["boundary"], boundary_target, self.boundary_pos_weight) + dice_loss_with_logits(
            pred["boundary"], boundary_target
        )
        distance_weight = 1.0 + 4.0 * boundary_target
        distance_loss = (F.smooth_l1_loss(torch.tanh(pred["distance"]), distance_target, reduction="none") * distance_weight).mean()
        topology_loss = buffered_cldice_loss(pred["boundary"], boundary_target, self.topology_buffer_iters)
        pyramid_loss = pyramid_mask_loss(pred["mask"], mask_target, self.pyramid_scales)

        total = (
            mask_loss
            + self.lambda_boundary * boundary_loss
            + self.lambda_distance * distance_loss
            + self.lambda_topology * topology_loss
            + self.lambda_pyramid * pyramid_loss
        )
        logs = {
            "loss": float(total.detach().cpu()),
            "mask_loss": float(mask_loss.detach().cpu()),
            "boundary_loss": float(boundary_loss.detach().cpu()),
            "distance_loss": float(distance_loss.detach().cpu()),
            "topology_loss": float(topology_loss.detach().cpu()),
            "pyramid_loss": float(pyramid_loss.detach().cpu()),
        }
        return total, logs
