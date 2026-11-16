"""
core/overlay.py
────────────────────────────────────────────────────────────────────────────────
OpenCV drawing helpers for video overlays: bounding boxes, labels, motion trails,
dashed restricted zones, monitored asset anchors/boxes, alert banners, and FPS.
────────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from typing import Callable, Dict, Iterable, Tuple

import cv2
import numpy as np

from core.geometry import Box
from core.security import ABANDONED, IN_TRANSIT, MISSING, zone_name_pretty

# BGR colours for the security overlay
COL_KNOWN = (60, 200, 60)
COL_UNKNOWN = (0, 140, 255)
COL_ALERT = (40, 40, 230)
COL_PENDING = (160, 160, 160)
COL_ASSET = (230, 220, 0)
COL_ZONE = (0, 215, 255)

_PALETTE: Dict[int, Tuple[int, int, int]] = {}


def track_colour(track_id: int) -> Tuple[int, int, int]:
    """Return a stable BGR colour for a given track-id."""
    if track_id not in _PALETTE:
        rng = np.random.default_rng(track_id + 42)
        _PALETTE[track_id] = tuple(int(c) for c in rng.integers(80, 230, 3))
    return _PALETTE[track_id]


def draw_label(img: np.ndarray, text: str, org, bg: Tuple[int, int, int],
               fg: Tuple[int, int, int], font_scale: float = 0.55):
    """Draw a filled label pill above `org`."""
    x, y = int(org[0]), int(org[1])
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
    y = max(y, th + 6)
    cv2.rectangle(img, (x, y - th - 6), (x + tw + 4, y), bg, -1)
    cv2.putText(img, text, (x + 2, y - 4), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, fg, 1, cv2.LINE_AA)


def draw_dashed_poly(img: np.ndarray, poly: np.ndarray, colour: Tuple[int, int, int],
                     dash: int = 14, thick: int = 2):
    """Draw a closed dashed polygon on `img`."""
    pts = [tuple(int(v) for v in p) for p in poly]
    for a, b in zip(pts, pts[1:] + pts[:1]):
        length = float(np.hypot(b[0] - a[0], b[1] - a[1]))
        if length <= 0:
            continue
        for s in range(0, int(length), dash * 2):
            e = min(s + dash, length)
            p1 = (int(a[0] + (b[0] - a[0]) * s / length), int(a[1] + (b[1] - a[1]) * s / length))
            p2 = (int(a[0] + (b[0] - a[0]) * e / length), int(a[1] + (b[1] - a[1]) * e / length))
            cv2.line(img, p1, p2, colour, thick, cv2.LINE_AA)


def draw_zones(img: np.ndarray, polys: Dict[str, np.ndarray], font_scale: float = 0.55):
    """Draw all active restricted security zone polygons."""
    for name, poly in polys.items():
        draw_dashed_poly(img, poly, COL_ZONE)
        x, y = poly[:, 0].min(), poly[:, 1].min()
        draw_label(img, zone_name_pretty(name).replace("_", " "), (x, y), COL_ZONE, (0, 0, 0), font_scale)


def draw_trail(img: np.ndarray, trails: dict, tid: int, box: Box,
               colour: Tuple[int, int, int], box_thick: int = 2,
               show_trail: bool = True, bottom_anchor: bool = True):
    """Append the current position for `tid` and draw its fading motion trail."""
    if not show_trail or tid < 0:
        return
    pt = ((box[0] + box[2]) // 2, box[3] if bottom_anchor else (box[1] + box[3]) // 2)
    trails[tid].append(pt)
    pts = list(trails[tid])
    for j in range(1, len(pts)):
        alpha = j / len(pts)
        cv2.line(img, pts[j - 1], pts[j], tuple(int(c * alpha) for c in colour),
                 max(1, box_thick - 1))


def draw_assets(img: np.ndarray, assets: Iterable, frame_idx: int,
                suspect_of: Dict[int, str], suspects_dict: dict,
                asset_state_fn: Callable, box_thick: int = 2, font_scale: float = 0.55):
    """Draw monitored asset boxes, dashed home anchors, and status labels."""
    for a in assets:
        label = a.class_name.upper()
        if a.status in (IN_TRANSIT, ABANDONED, MISSING) or a.last_seen < 0:
            hx1, hy1, hx2, hy2 = a.home
            pts = np.array([[hx1, hy1], [hx2, hy1], [hx2, hy2], [hx1, hy2]], np.int32)
            draw_dashed_poly(img, pts, COL_ALERT if a.status == MISSING else COL_PENDING, dash=8, thick=1)
        if a.status == MISSING:
            draw_label(img, f"{label}: {asset_state_fn(a)}", a.home[:2], COL_ALERT, (255, 255, 255), font_scale)
            continue
        if a.last_seen < 0:
            draw_label(img, f"{a.name} anchor", a.home[:2], COL_PENDING, (0, 0, 0), font_scale)
            continue
        if a.last_seen != frame_idx:
            continue
        if a.status == IN_TRANSIT:
            sid = suspect_of.get(a.handler)
            who = suspects_dict[sid].label if sid and sid in suspects_dict else f"TRACK_ID {a.handler}"
            col, text = COL_ASSET, f"{label}: WITH {who}"
        elif a.status == ABANDONED:
            col, text = COL_UNKNOWN, f"{label}: {asset_state_fn(a)}"
        else:
            col, text = COL_KNOWN, f"{label}: SECURED"
        cv2.rectangle(img, a.box[:2], a.box[2:], col, box_thick)
        draw_label(img, text, (a.box[0], a.box[3] + 20), col, (0, 0, 0), font_scale)


def draw_banners(img: np.ndarray, banners: Iterable[Tuple[str, float]], t_sec: float):
    """Draw up to two active top-of-screen alert banners."""
    active = [b for b in banners if b[1] >= t_sec]
    y = 40
    for text, _ in active[-2:]:
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
        x = max(10, (img.shape[1] - tw) // 2)
        cv2.rectangle(img, (x - 12, y - th - 12), (x + tw + 12, y + 10), COL_ALERT, -1)
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
        y += th + 30


def draw_fps(img: np.ndarray, fps: float):
    """Overlay live FPS counter in the top-left corner."""
    cv2.putText(img, f"FPS {fps:.1f}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 255, 100), 2, cv2.LINE_AA)
