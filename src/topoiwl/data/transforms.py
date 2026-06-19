"""Lightweight paired transforms for image/mask/boundary/distance tensors."""

from __future__ import annotations

import random

import numpy as np


def random_flip_rotate(sample: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    if random.random() < 0.5:
        sample = {k: np.flip(v, axis=1).copy() for k, v in sample.items()}
    if random.random() < 0.5:
        sample = {k: np.flip(v, axis=0).copy() for k, v in sample.items()}
    k_rot = random.randint(0, 3)
    if k_rot:
        sample = {k: np.rot90(v, k_rot, axes=(0, 1)).copy() for k, v in sample.items()}
    return sample


def random_brightness_contrast(image: np.ndarray, brightness: float = 0.12, contrast: float = 0.12) -> np.ndarray:
    alpha = 1.0 + random.uniform(-contrast, contrast)
    beta = random.uniform(-brightness, brightness)
    return np.clip(image * alpha + beta, 0.0, 1.0)

