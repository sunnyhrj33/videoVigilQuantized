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
#  Monitored assets (static anchor + spatial displacement)
# ─────────────────────────────────────────────────────────────────────────────

COCO_ASSETS = {24: "backpack", 26: "handbag", 63: "laptop"}

SECURED = "SECURED"          # on its home anchor
IN_TRANSIT = "IN_TRANSIT"    # displaced, a person (handler) is with it
ABANDONED = "ABANDONED"      # displaced, nobody with it any more
MISSING = "MISSING"          # gone from view (taken away / hidden)


def centre(b: Box) -> Tuple[float, float]:
    return (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0


def area(b: Box) -> float:
    return max(1.0, float((b[2] - b[0]) * (b[3] - b[1])))


@dataclass
class Asset:
    asset_id: int
    name: str                      # e.g. "laptop_1" or a configured / drawn name
    class_name: str
    home: Box                      # registered anchor (home) box
    box: Box                       # last observed box
    last_seen: int
    origin: str = "auto"           # auto | config | drawn
    status: str = SECURED
    handler: Optional[int] = None  # track id of the person who displaced / carries it
    moved_frames: int = 0          # consecutive frames beyond the displacement threshold
    away_since: Optional[int] = None
    last_toucher: Optional[Tuple[int, int]] = None      # (track id, frame)
    unknown_contact_since: Optional[float] = None
    last_unknown_contact: Optional[float] = None
    interaction_alerted: bool = False

    @property
    def displacement(self) -> float:
        (hx, hy), (cx, cy) = centre(self.home), centre(self.box)
        return float(np.hypot(cx - hx, cy - hy))


@dataclass
class AssetEvent:
    asset: Asset
    track_id: Optional[int]
    kind: str          # displacement | returned | abandoned | missing | unknown_interaction
    dist: float = 0.0


class AssetMonitor:
    """
    Static asset anchoring + displacement detection (FYP-2 theft workflow).

    A monitored asset has a HOME anchor box (from config, drawn in the console,
    or auto-registered once a detected laptop has been still for
    `anchor_frames`). Each frame the asset's current box is compared to home:

        delta_d = || centre(box) - centre(home) ||   (pixels)

    If delta_d > displacement_threshold_px (for `confirm_frames` frames, with a
    similar box size so occlusion is not mistaken for movement) and a person
    overlaps the asset with Area(person ∩ asset) / Area(asset) > interaction
    threshold → DISPLACEMENT by that person (IN_TRANSIT). The handler leaving
    it → ABANDONED at (x, y); gone from view → MISSING; back home → SECURED.
    """

    def __init__(self, cfg: dict):
        a = cfg.get("asset_monitoring", {}) or {}
        self.enabled = bool(a.get("enabled", True))
        classes = a.get("target_asset_classes", [63])
        self.classes = {COCO_ASSETS.get(c, str(c)) if isinstance(c, int) else str(c) for c in classes}
        self.move_px = float(a.get("displacement_threshold_px", 35))
        self.overlap_thresh = float(a.get("interaction_iou_thresh", 0.25))
        self.confirm_frames = int(a.get("confirm_frames", 5))
        self.anchor_frames = int(a.get("anchor_frames", 15))
        self.abandon_frames = int(a.get("abandon_frames", 45))
        self.missing_frames = int(a.get("missing_frames", 90))
        self.stay_limit = float(a.get("unknown_stay_limit_seconds", 5))
        self.allow_unknown = bool(a.get("allow_unknown_interaction", False))
        self.contact_grace = float(a.get("contact_grace_seconds", 1.5))
        self.configured = a.get("monitored_zones", {}) or {}
        self.assets: List[Asset] = []
        self._pending: List[dict] = []            # auto-registration candidates
        self._next_id = 1

    # ── setup ────────────────────────────────────────────────────────────────

    def reset(self):
        self.assets.clear()
        self._pending.clear()
        self._next_id = 1

    def configure(self, camera_key: Optional[str], drawn: Optional[Dict[str, List[int]]] = None):
        """New source: register configured + console-drawn anchors for this camera."""
        self.reset()
        cam_cfg = self.configured.get(camera_key or "", {}) or {}
        for origin, anchors in (("config", cam_cfg), ("drawn", drawn or {})):
            for name, box in anchors.items():
                self._add(tuple(int(v) for v in box), "laptop", origin, name=name, seen=-1)

    def _add(self, box: Box, cls_name: str, origin: str, name: Optional[str] = None, seen: int = 0) -> Asset:
        a = Asset(self._next_id, name or f"{cls_name}_{self._next_id}", cls_name, box, box, seen, origin)
        self._next_id += 1
        self.assets.append(a)
        return a

    # ── per frame ────────────────────────────────────────────────────────────

    def update(self, frame_idx: int, t_sec: float, objects: List[Tuple[str, Box]],
               persons: List[Tuple[int, Box, str, Optional[str]]]) -> List[AssetEvent]:
        """
        objects: [(class_name, box)] from the detector.
        persons: [(track_id, box, status, label)], status KNOWN / UNKNOWN / PENDING.
        """
        if not self.enabled:
            return []
        dets = [(c, b) for c, b in objects if c in self.classes]
        used = self._associate(dets, frame_idx)
        self._auto_register([d for i, d in enumerate(dets) if i not in used], frame_idx)

        events: List[AssetEvent] = []
        for a in self.assets:
            visible = a.last_seen == frame_idx
            toucher = max(persons, key=lambda p: overlap_ratio(p[1], a.box), default=None)
            touching = toucher is not None and overlap_ratio(toucher[1], a.box) > self.overlap_thresh
            if touching:
                a.last_toucher = (toucher[0], frame_idx)
            events += self._step(a, frame_idx, visible, toucher if touching else None, persons)
            ev = self._interaction(a, t_sec, toucher if touching else None)
            if ev:
                events.append(ev)
        return events

    def _associate(self, dets, frame_idx) -> set:
        """Greedy match of detections to assets: IoU with last box, else nearest centre."""
        used = set()
        for a in sorted(self.assets, key=lambda x: x.status != IN_TRANSIT):
            best, best_score = None, 0.0
            diag = float(np.hypot(a.box[2] - a.box[0], a.box[3] - a.box[1]))
            for i, (c, b) in enumerate(dets):
                if i in used or c != a.class_name:
                    continue
                v = iou(a.box, b)
                if v > 0.2:
                    score = 1.0 + v
                else:
                    (ax, ay), (bx, by) = centre(a.box), centre(b)
                    d = float(np.hypot(ax - bx, ay - by))
                    reach = 1.5 * diag if a.status == IN_TRANSIT else 0.75 * diag
                    score = 1.0 - d / reach if d < reach else 0.0
                if score > best_score:
                    best, best_score = i, score
            if best is not None:
                used.add(best)
                a.box = dets[best][1]
                a.last_seen = frame_idx
        return used

    def _auto_register(self, dets, frame_idx):
        """A detection that stays still for anchor_frames becomes a monitored asset."""
        keep = []
        for c, b in dets:
            if any(iou(a.home, b) > 0.3 or iou(a.box, b) > 0.3 for a in self.assets):
                continue
            cand = next((p for p in self._pending if p["cls"] == c and iou(p["box"], b) > 0.5), None)
            if cand is None:
                cand = {"cls": c, "box": b, "first": b, "count": 0}
            (fx, fy), (bx, by) = centre(cand["first"]), centre(b)
            cand["count"] = cand["count"] + 1 if np.hypot(bx - fx, by - fy) < self.move_px else 1
            cand["box"], cand["last"] = b, frame_idx
            if cand["count"] == 1:
                cand["first"] = b
            if cand["count"] >= self.anchor_frames:
                self._add(b, c, "auto", seen=frame_idx)
            else:
                keep.append(cand)
        self._pending = [p for p in keep if frame_idx - p.get("last", frame_idx) < 30]

    def _step(self, a: Asset, f: int, visible: bool, toucher, persons) -> List[AssetEvent]:
        ev: List[AssetEvent] = []
        if not visible:
            if a.status in (SECURED, IN_TRANSIT) and a.last_seen >= 0 and f - a.last_seen > self.missing_frames:
                tid = a.handler if a.handler is not None else (a.last_toucher[0] if a.last_toucher else None)
                a.status, a.handler = MISSING, tid
                ev.append(AssetEvent(a, tid, "missing", a.displacement))
            return ev

        moved = a.displacement > self.move_px and 0.5 < area(a.box) / area(a.home) < 2.0
        a.moved_frames = a.moved_frames + 1 if moved else 0
        confirmed = a.moved_frames >= self.confirm_frames

        if a.status == MISSING:
            a.status = SECURED if not moved else (IN_TRANSIT if toucher else ABANDONED)
            ev.append(AssetEvent(a, a.handler, "returned" if a.status == SECURED else "abandoned", a.displacement))
            if a.status == SECURED:
                a.handler = None
        elif a.status == SECURED:
            if confirmed and toucher is not None:
                a.status, a.handler, a.away_since = IN_TRANSIT, toucher[0], None
                ev.append(AssetEvent(a, toucher[0], "displacement", a.displacement))
            elif confirmed and a.moved_frames > self.abandon_frames:
                a.home, a.moved_frames = a.box, 0     # nudged with nobody around → re-anchor quietly
        elif a.status in (IN_TRANSIT, ABANDONED):
            if not moved:
                a.status, a.handler, a.away_since = SECURED, None, None
                ev.append(AssetEvent(a, None, "returned", 0.0))
            elif a.status == IN_TRANSIT:
                handler_near = any(p[0] == a.handler and overlap_ratio(p[1], a.box) > self.overlap_thresh
                                   for p in persons)
                if handler_near:
                    a.away_since = None
                else:
                    a.away_since = a.away_since if a.away_since is not None else f
                    if f - a.away_since >= self.abandon_frames:
                        a.status = ABANDONED
                        ev.append(AssetEvent(a, a.handler, "abandoned", a.displacement))
            # ABANDONED + a different person picking it up = hand-over (future work)
        return ev

    def _interaction(self, a: Asset, t_sec: float, toucher) -> Optional[AssetEvent]:
        """Unknown person lingering at an asset for > stay limit. Short detection gaps
        (≤ contact_grace s, e.g. a seated person briefly missed) do not restart the timer."""
        if toucher is not None and toucher[2] == "UNKNOWN" and not self.allow_unknown:
            gap = t_sec - a.last_unknown_contact if a.last_unknown_contact is not None else None
            if a.unknown_contact_since is None or gap is None or gap > self.contact_grace:
                a.unknown_contact_since = t_sec
            a.last_unknown_contact = t_sec
            if t_sec - a.unknown_contact_since >= self.stay_limit and not a.interaction_alerted:
                a.interaction_alerted = True
                return AssetEvent(a, toucher[0], "unknown_interaction")
        elif a.last_unknown_contact is not None and t_sec - a.last_unknown_contact > self.contact_grace:
            a.unknown_contact_since = a.last_unknown_contact = None
            a.interaction_alerted = False
        return None


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
