"""
security.py
────────────────────────────────────────────────────────────────────────────────
Spatial security logic on top of detection + identity:

  • ZoneMonitor   – restricted polygons (per camera); unknown person whose
                    bottom-centre point is inside → intrusion alert
  • AssetMonitor  – monitored objects (laptop / backpack / handbag); person
                    overlap + displacement → possession or theft alert,
                    unknown lingering on an asset → interaction alert
  • AuditLogger   – forensic CSV log (output/audit_log.csv)
────────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from open_set import display_name

logger = logging.getLogger(__name__)

Box = Tuple[int, int, int, int]

AUDIT_HEADER = ["Timestamp", "Camera_ID", "Track_ID", "Identity", "Status",
                "Confidence", "Zone_Alert", "Asset_Interaction"]


def overlap_ratio(person: Box, obj: Box) -> float:
    """Area(person ∩ object) / Area(object)."""
    ix1, iy1 = max(person[0], obj[0]), max(person[1], obj[1])
    ix2, iy2 = min(person[2], obj[2]), min(person[3], obj[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area = max(1, (obj[2] - obj[0]) * (obj[3] - obj[1]))
    return inter / area


def iou(a: Box, b: Box) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / max(1, union)


def zone_name_pretty(name: str) -> str:
    """'restricted_desk_area' → 'Restricted_Desk_Area'."""
    return "_".join(p.capitalize() for p in name.split("_"))


# ─────────────────────────────────────────────────────────────────────────────
#  Restricted zones
# ─────────────────────────────────────────────────────────────────────────────

class ZoneMonitor:
    """
    Zones are normalised polygons [[x, y], ...] in config:
        security_zones:
          restricted_desk_area: [[...]]          # default for every camera
          per_camera:
            camera_3: {restricted_desk_area: [[...]]}
    """

    def __init__(self, cfg: dict):
        self._cfg = cfg.get("security_zones", {}) or {}
        self.zones: Dict[str, np.ndarray] = {}

    def configure(self, camera_key: Optional[str]):
        """Select the zones for this camera (per-camera override, else defaults)."""
        per_cam = (self._cfg.get("per_camera") or {}).get(camera_key or "", None)
        source = per_cam if per_cam is not None else {
            k: v for k, v in self._cfg.items()
            if k not in ("per_camera", "monitored_assets", "enabled") and isinstance(v, list)
        }
        self.zones = {name: np.array(poly, dtype=np.float32) for name, poly in source.items() if poly}

    def pixel_polygons(self, w: int, h: int) -> Dict[str, np.ndarray]:
        return {n: (p * [w, h]).astype(np.int32) for n, p in self.zones.items()}

    def zones_containing(self, box: Box, w: int, h: int) -> List[str]:
        """Zones containing the person's bottom-centre (feet) point."""
        pt = ((box[0] + box[2]) / 2.0, float(box[3]))
        return [n for n, poly in self.pixel_polygons(w, h).items()
                if cv2.pointPolygonTest(poly.reshape(-1, 1, 2), pt, False) >= 0]


# ─────────────────────────────────────────────────────────────────────────────
#  Monitored assets
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Asset:
    asset_id: int
    class_name: str
    box: Box
    anchor: Box
    last_seen: int
    seen_count: int = 1
    state: str = "Static"                 # Static | Moved by Emp XX | ALERT
    holder: Optional[str] = None
    unknown_contact_since: Optional[float] = None


@dataclass
class AssetEvent:
    asset: Asset
    track_id: int
    kind: str          # "possession" | "displaced_unknown" | "unknown_interaction"
    text: str


class AssetMonitor:
    """
    Keeps a registry of monitored objects matched frame-to-frame by IoU, so an
    asset keeps its identity even if the tracker re-numbers it.
    """

    def __init__(self, cfg: dict):
        zcfg = cfg.get("security_zones", {}) or {}
        self.rules = {a["class_name"]: a for a in (zcfg.get("monitored_assets") or [])}
        acfg = cfg.get("assets", {}) or {}
        self.overlap_thresh = float(acfg.get("overlap_thresh", 0.30))
        self.move_thresh = float(acfg.get("move_thresh", 0.35))   # × anchor box diagonal
        self.anchor_frames = int(acfg.get("anchor_frames", 15))
        self.max_missing = int(acfg.get("max_missing_frames", 90))
        self.assets: List[Asset] = []
        self._next_id = 1

    def reset(self):
        self.assets.clear()
        self._next_id = 1

    def update(self, frame_idx: int, t_sec: float, objects: List[Tuple[str, Box]],
               persons: List[Tuple[int, Box, str, Optional[str]]]) -> List[AssetEvent]:
        """
        objects: [(class_name, box)] for monitored classes.
        persons: [(track_id, box, status, label)] with status KNOWN / UNKNOWN / PENDING.
        """
        # Match detections to the registry
        for cls_name, box in objects:
            if cls_name not in self.rules:
                continue
            best, best_iou = None, 0.2
            for a in self.assets:
                if a.class_name == cls_name and frame_idx - a.last_seen <= self.max_missing:
                    v = iou(a.box, box)
                    if v > best_iou:
                        best, best_iou = a, v
            if best is None:
                self.assets.append(Asset(self._next_id, cls_name, box, box, frame_idx))
                self._next_id += 1
            else:
                best.box = box
                best.last_seen = frame_idx
                best.seen_count += 1
                if best.seen_count <= self.anchor_frames and best.state == "Static":
                    best.anchor = box    # settle anchor while the object is still new

        self.assets = [a for a in self.assets if frame_idx - a.last_seen <= self.max_missing]

        events: List[AssetEvent] = []
        for a in self.assets:
            if a.last_seen != frame_idx or a.seen_count < self.anchor_frames:
                continue
            rule = self.rules[a.class_name]
            toucher = max(persons, key=lambda p: overlap_ratio(p[1], a.box), default=None)
            touching = toucher is not None and overlap_ratio(toucher[1], a.box) > self.overlap_thresh
            moved = self._displacement(a) > self.move_thresh
            name = a.class_name.capitalize()

            if not touching:
                a.unknown_contact_since = None
                continue
            tid, _, status, label = toucher

            if moved and status == "KNOWN":
                holder = display_name(label)
                if a.holder != holder:
                    a.holder = holder
                    a.state = f"Moved by {holder}"
                    events.append(AssetEvent(a, tid, "possession",
                                             f"{name} in possession of {holder}"))
            elif moved and status == "UNKNOWN":
                if a.state != "ALERT":
                    a.state = "ALERT"
                    events.append(AssetEvent(a, tid, "displaced_unknown",
                                             f"Security Alert: {name} Displaced by Unknown Person!"))

            if status == "UNKNOWN" and not rule.get("allow_unknown_interaction", False):
                if a.unknown_contact_since is None:
                    a.unknown_contact_since = t_sec
                limit = float(rule.get("zone_stay_limit_seconds", 5))
                if t_sec - a.unknown_contact_since >= limit and a.state != "ALERT":
                    a.state = "ALERT"
                    events.append(AssetEvent(a, tid, "unknown_interaction",
                                             f"Security Alert: Unknown Person at {name} > {limit:.0f}s"))
            else:
                a.unknown_contact_since = None
        return events

    @staticmethod
    def _displacement(a: Asset) -> float:
        acx, acy = (a.anchor[0] + a.anchor[2]) / 2, (a.anchor[1] + a.anchor[3]) / 2
        cx, cy = (a.box[0] + a.box[2]) / 2, (a.box[1] + a.box[3]) / 2
        diag = max(1.0, float(np.hypot(a.anchor[2] - a.anchor[0], a.anchor[3] - a.anchor[1])))
        return float(np.hypot(cx - acx, cy - acy)) / diag


# ─────────────────────────────────────────────────────────────────────────────
#  Audit log
# ─────────────────────────────────────────────────────────────────────────────

class AuditLogger:
    """Append-only forensic CSV log. Rows are flushed immediately."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        new = not self.path.exists() or self.path.stat().st_size == 0
        self._f = open(self.path, "a", newline="", encoding="utf-8")
        self._w = csv.writer(self._f)
        if new:
            self._w.writerow(AUDIT_HEADER)
            self._f.flush()

    def log(self, timestamp: str, camera: str, track_id: int, identity: str,
            status: str, confidence: float, zone_alert: str = "None",
            asset: str = "None"):
        self._w.writerow([timestamp, camera, track_id, identity, status,
                          f"{confidence:.2f}", zone_alert, asset])
        self._f.flush()

    def close(self):
        try:
            self._f.close()
        except Exception:
            pass


@dataclass
class Clock:
    """Maps frame index → wall-clock string (CHIRLA start time + frame/fps, else now)."""
    start: Optional[datetime] = None
    fps: float = 30.0
    seconds: float = field(default=0.0)

    def set_frame(self, frame_idx: int):
        self.seconds = frame_idx / max(self.fps, 1e-3)

    def stamp(self) -> str:
        if self.start is not None:
            return (self.start + timedelta(seconds=self.seconds)).strftime("%H:%M:%S")
        return datetime.now().strftime("%H:%M:%S")
