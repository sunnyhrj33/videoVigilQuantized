"""
core/geometry.py
────────────────────────────────────────────────────────────────────────────────
Shared bounding-box geometry and image sharpness helpers used across detection,
open-set Re-ID, spatial security, video search, and gallery building.
────────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

Box = Tuple[int, int, int, int]


def l2n(x: np.ndarray) -> np.ndarray:
    """L2-normalise rows (or a single vector)."""
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, 1e-6)


def centre(b: Box) -> Tuple[float, float]:
    """Return (cx, cy) centre of an (x1, y1, x2, y2) bounding box."""
    return (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0


def area(b: Box) -> float:
    """Return pixel area of an (x1, y1, x2, y2) bounding box (at least 1.0)."""
    return max(1.0, float((b[2] - b[0]) * (b[3] - b[1])))


def iou(a: Box, b: Box) -> float:
    """Intersection-over-Union of two (x1, y1, x2, y2) bounding boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / max(1e-9, union))


def overlap_ratio(person: Box, obj: Box) -> float:
    """Area(person ∩ object) / Area(object)."""
    ix1, iy1 = max(person[0], obj[0]), max(person[1], obj[1])
    ix2, iy2 = min(person[2], obj[2]), min(person[3], obj[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    obj_area = max(1, (obj[2] - obj[0]) * (obj[3] - obj[1]))
    return float(inter / obj_area)


def sharpness(img: np.ndarray) -> float:
    """Variance of Laplacian on a 64×128 grayscale resize."""
    gray = cv2.cvtColor(cv2.resize(img, (64, 128)), cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())
