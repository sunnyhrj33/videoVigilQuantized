"""
console/mosaic.py
────────────────────────────────────────────────────────────────────────────────
Multi-camera grid for the Live Monitor: every camera in one large view, laid out
to make the tiles as big as the canvas allows. Click a tile to enlarge it.
────────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

Rect = Tuple[int, int, int, int]          # x, y, w, h
GAP = 4
BG = (25, 17, 13)                         # BGR of the video well (#0d1119)
TILE_BG = (37, 26, 19)                    # #131a25
LABEL_BG = (20, 14, 11)                   # #0b0e14
ALERT = (82, 82, 240)                     # #f05252 signal red
MUTED = (135, 117, 104)                   # #687587


def grid_shape(n: int, width: int, height: int, aspect: float = 1.5) -> Tuple[int, int]:
    """(cols, rows) giving the largest tiles of the given aspect ratio (w/h) for n tiles."""
    best, best_area = (1, max(n, 1)), -1.0
    for cols in range(1, max(n, 1) + 1):
        rows = math.ceil(n / cols)
        tw = (width - GAP * (cols + 1)) / cols
        th = (height - GAP * (rows + 1)) / rows
        if tw <= 0 or th <= 0:
            continue
        w = min(tw, th * aspect)
        if w * (w / aspect) > best_area:
            best, best_area = (cols, rows), w * (w / aspect)
    return best


def layout(n: int, width: int, height: int, aspect: float = 1.5) -> List[Rect]:
    """Tile rectangles, row by row, the last row centred."""
    if n <= 0:
        return []
    cols, rows = grid_shape(n, width, height, aspect)
    tw = (width - GAP * (cols + 1)) / cols
    th = (height - GAP * (rows + 1)) / rows
    w = int(min(tw, th * aspect))
    h = int(w / aspect)
    total_h = rows * h + (rows - 1) * GAP
    y0 = (height - total_h) // 2
    rects = []
    for r in range(rows):
        k = min(cols, n - r * cols)
        row_w = k * w + (k - 1) * GAP
        x0 = (width - row_w) // 2
        for c in range(k):
            rects.append((x0 + c * (w + GAP), y0 + r * (h + GAP), w, h))
    return rects


def tile_at(rects: Sequence[Rect], x: int, y: int) -> Optional[int]:
    for i, (rx, ry, rw, rh) in enumerate(rects):
        if rx <= x < rx + rw and ry <= y < ry + rh:
            return i
    return None


def _label(img: np.ndarray, text: str, colour=(245, 237, 232), bg=LABEL_BG):
    h = img.shape[0]
    scale = max(0.45, min(0.6, h / 420))
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    cv2.rectangle(img, (0, 0), (tw + 16, th + 14), bg, -1)
    cv2.putText(img, text, (8, th + 7), cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 1, cv2.LINE_AA)


def compose(frames: Sequence[Optional[np.ndarray]], labels: Sequence[str], width: int, height: int,
            ended: Optional[Sequence[bool]] = None,
            alerts: Optional[Sequence[int]] = None) -> Tuple[np.ndarray, List[Rect]]:
    """All camera frames in one width×height BGR image + the tile rectangles (for clicks).
    Tiles whose index is in `alerts` (open, unacknowledged alerts) get a red frame + badge."""
    alerts = set(alerts or ())
    canvas = np.full((max(1, height), max(1, width), 3), BG, np.uint8)
    first = next((f for f in frames if f is not None), None)
    aspect = first.shape[1] / first.shape[0] if first is not None else 1.5
    rects = layout(len(frames), width, height, aspect)
    for i, ((x, y, w, h), frame) in enumerate(zip(rects, frames)):
        if w < 2 or h < 2:
            continue
        if frame is None:
            tile = np.full((h, w, 3), TILE_BG, np.uint8)
            cv2.putText(tile, "STARTING", (12, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, MUTED, 1, cv2.LINE_AA)
        else:
            tile = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
        done = bool(ended and ended[i])
        if done:
            tile = (tile * 0.45).astype(np.uint8)
        alarm = i in alerts
        _label(tile, f"{labels[i]}{'  ALERT' if alarm else '  ENDED' if done else ''}".upper(),
               bg=ALERT if alarm else LABEL_BG)
        if alarm:
            cv2.rectangle(tile, (1, 1), (w - 2, h - 2), ALERT, 3)
        canvas[y:y + h, x:x + w] = tile
    return canvas, rects
