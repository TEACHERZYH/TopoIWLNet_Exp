"""Morphology and distance-transform utilities with optional SciPy/skimage acceleration."""

from __future__ import annotations

from collections import deque

import numpy as np

try:
    from scipy.ndimage import binary_dilation as _sp_binary_dilation
    from scipy.ndimage import binary_erosion as _sp_binary_erosion
    from scipy.ndimage import distance_transform_edt as _sp_distance_transform_edt
    from scipy.ndimage import label as _sp_label
except Exception:  # pragma: no cover
    _sp_binary_dilation = None
    _sp_binary_erosion = None
    _sp_distance_transform_edt = None
    _sp_label = None

try:
    from skimage.morphology import skeletonize as _sk_skeletonize
except Exception:  # pragma: no cover
    _sk_skeletonize = None


def binary_dilation(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    if _sp_binary_dilation is not None:
        return _sp_binary_dilation(mask, iterations=iterations)
    out = mask.astype(bool)
    for _ in range(max(iterations, 1)):
        padded = np.pad(out, 1, mode="constant", constant_values=False)
        nxt = np.zeros_like(out, dtype=bool)
        h, w = out.shape
        for dy in range(3):
            for dx in range(3):
                nxt |= padded[dy : dy + h, dx : dx + w]
        out = nxt
    return out


def binary_erosion(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    if _sp_binary_erosion is not None:
        return _sp_binary_erosion(mask, iterations=iterations)
    out = mask.astype(bool)
    for _ in range(max(iterations, 1)):
        padded = np.pad(out, 1, mode="constant", constant_values=False)
        nxt = np.ones_like(out, dtype=bool)
        h, w = out.shape
        for dy in range(3):
            for dx in range(3):
                nxt &= padded[dy : dy + h, dx : dx + w]
        out = nxt
    return out


def _edt_1d(f: np.ndarray) -> np.ndarray:
    n = f.shape[0]
    v = np.zeros(n, dtype=np.int64)
    z = np.zeros(n + 1, dtype=np.float64)
    d = np.zeros(n, dtype=np.float64)
    k = 0
    v[0] = 0
    z[0] = -np.inf
    z[1] = np.inf
    for q in range(1, n):
        while True:
            denom = 2 * q - 2 * v[k]
            s = ((f[q] + q * q) - (f[v[k]] + v[k] * v[k])) / denom
            if s > z[k]:
                break
            k -= 1
            if k < 0:
                k = 0
                break
        k += 1
        v[k] = q
        z[k] = s
        z[k + 1] = np.inf
    k = 0
    for q in range(n):
        while z[k + 1] < q:
            k += 1
        d[q] = (q - v[k]) ** 2 + f[v[k]]
    return d


def distance_transform_edt(mask: np.ndarray) -> np.ndarray:
    if _sp_distance_transform_edt is not None:
        return _sp_distance_transform_edt(mask)
    mask = mask.astype(bool)
    if not mask.any():
        return np.zeros(mask.shape, dtype=np.float64)
    inf = float(mask.shape[0] * mask.shape[0] + mask.shape[1] * mask.shape[1] + 1)
    f = np.where(mask, inf, 0.0).astype(np.float64)
    tmp = np.apply_along_axis(_edt_1d, 0, f)
    dist_sq = np.apply_along_axis(_edt_1d, 1, tmp)
    return np.sqrt(dist_sq)


def label(mask: np.ndarray) -> tuple[np.ndarray, int]:
    if _sp_label is not None:
        return _sp_label(mask)
    mask = mask.astype(bool)
    labels = np.zeros(mask.shape, dtype=np.int32)
    current = 0
    h, w = mask.shape
    for y in range(h):
        for x in range(w):
            if not mask[y, x] or labels[y, x] != 0:
                continue
            current += 1
            labels[y, x] = current
            queue: deque[tuple[int, int]] = deque([(y, x)])
            while queue:
                cy, cx = queue.popleft()
                for ny in range(max(0, cy - 1), min(h, cy + 2)):
                    for nx in range(max(0, cx - 1), min(w, cx + 2)):
                        if mask[ny, nx] and labels[ny, nx] == 0:
                            labels[ny, nx] = current
                            queue.append((ny, nx))
    return labels, current


def skeletonize(mask: np.ndarray) -> np.ndarray:
    if _sk_skeletonize is not None:
        return _sk_skeletonize(mask)
    # Fallback: boundary itself is a conservative skeleton proxy.
    return mask.astype(bool)


def has_fast_morphology() -> bool:
    return _sp_binary_dilation is not None and _sk_skeletonize is not None

