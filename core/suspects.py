"""
core/suspects.py
────────────────────────────────────────────────────────────────────────────────
Automated suspect gallery + tracking history (FYP-2 theft workflow).

  • SuspectGallery – persistent, auto-enrolled suspect profiles
                     (SUSPECT_01 …): crops, L2-normalised centroid prototype
                     P_suspect = 1/K Σ e_i, creation reason and sightings.
                     Kept across video sources so a suspect can be
                     re-identified in other cameras.
  • TrackHistory   – per-second track positions → output/track_history.csv,
                     plus first-seen info per track for forensic backtracking
                     (where / when the person entered the view).
────────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import csv
import json
import logging
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from core.geometry import l2n

logger = logging.getLogger(__name__)


@dataclass
class Suspect:
    suspect_id: str                    # "SUSPECT_01"
    created: str                       # wall/video time stamp
    camera: str
    source: str                        # video file / camera index
    frame: int
    reason: str                        # displacement | unknown_interaction | manual
    n_crops: int
    entry: Dict = field(default_factory=dict)      # forensic backtrack of the origin track
    sightings: List[Dict] = field(default_factory=list)

    @property
    def number(self) -> str:
        return self.suspect_id.split("_")[-1]

    @property
    def label(self) -> str:
        return f"SUSPECT {self.number}"


class SuspectGallery:
    """Persistent suspect profiles in <dir>/: suspects.json, prototypes.npz, SUSPECT_xx/cropK.jpg."""

    def __init__(self, directory: str | Path, match_thresh: float = 0.75):
        self.dir = Path(directory)
        self.match_thresh = float(match_thresh)
        self.suspects: Dict[str, Suspect] = {}
        self.prototypes: Dict[str, np.ndarray] = {}
        self._load()

    # ── persistence ──────────────────────────────────────────────────────────

    def _load(self):
        meta, protos = self.dir / "suspects.json", self.dir / "prototypes.npz"
        if not meta.exists():
            return
        try:
            for d in json.loads(meta.read_text()):
                s = Suspect(**d)
                self.suspects[s.suspect_id] = s
            if protos.exists():
                data = np.load(protos)
                self.prototypes = {k: data[k] for k in data.files if k in self.suspects}
            logger.info("✓ Loaded %d suspect(s) from %s", len(self.suspects), self.dir)
        except Exception as exc:
            logger.warning("Could not load suspect gallery (%s) – starting empty", exc)
            self.suspects, self.prototypes = {}, {}

    def _save(self):
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "suspects.json").write_text(json.dumps([asdict(s) for s in self.suspects.values()], indent=2))
        if self.prototypes:
            np.savez(self.dir / "prototypes.npz", **self.prototypes)
        elif (self.dir / "prototypes.npz").exists():
            (self.dir / "prototypes.npz").unlink()

    # ── gallery operations ───────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.suspects)

    def _next_id(self) -> str:
        n = 1 + max((int(s.split("_")[-1]) for s in self.suspects), default=0)
        return f"SUSPECT_{n:02d}"

    def enrol(self, embs: np.ndarray, thumbs: List[np.ndarray], *, created: str, camera: str,
              source: str, frame: int, reason: str, entry: Optional[Dict] = None) -> Optional[Suspect]:
        """Auto-enrol a suspect from tracklet crops: prototype = L2-normalised mean embedding."""
        if embs is None or len(embs) == 0:
            return None
        sid = self._next_id()
        proto = l2n(l2n(np.asarray(embs, np.float32)).mean(axis=0))
        s = Suspect(sid, created, camera, str(source), int(frame), reason, len(embs), entry or {})
        self.suspects[sid] = s
        self.prototypes[sid] = proto
        crop_dir = self.dir / sid
        crop_dir.mkdir(parents=True, exist_ok=True)
        for k, img in enumerate(thumbs, 1):
            cv2.imwrite(str(crop_dir / f"crop{k}.jpg"), img)
        self._save()
        logger.warning("SUSPECT ENROLLED: %s (%s, %d crops, %s %s)", sid, reason, len(embs), camera, created)
        return s

    def match(self, embs: np.ndarray, exclude: Optional[set] = None) -> Tuple[Optional[str], float]:
        """Best suspect for a tracklet (centroid cosine); (None, best_sim) below the threshold."""
        if not self.prototypes or embs is None or len(embs) == 0:
            return None, 0.0
        c = l2n(l2n(np.asarray(embs, np.float32)).mean(axis=0))
        best, best_sim = None, 0.0
        for sid, p in self.prototypes.items():
            if exclude and sid in exclude:
                continue
            s = float(c @ p)
            if s > best_sim:
                best, best_sim = sid, s
        return (best, best_sim) if best_sim >= self.match_thresh else (None, best_sim)

    def add_sighting(self, sid: str, **info):
        s = self.suspects.get(sid)
        if s is None:
            return
        s.sightings.append(info)
        s.sightings = s.sightings[-200:]
        self._save()

    def delete(self, sid: str):
        self.suspects.pop(sid, None)
        self.prototypes.pop(sid, None)
        shutil.rmtree(self.dir / sid, ignore_errors=True)
        self._save()
        logger.warning("Suspect %s deleted from %s", sid, self.dir)

    def clear(self):
        for sid in list(self.suspects):
            self.delete(sid)

    def crops(self, sid: str) -> List[Path]:
        return sorted((self.dir / sid).glob("crop*.jpg"))


# ─────────────────────────────────────────────────────────────────────────────
#  Track history + forensic backtrack
# ─────────────────────────────────────────────────────────────────────────────

HISTORY_HEADER = ["Timestamp", "Camera_ID", "Source", "Frame", "Track_ID", "Identity",
                  "Status", "x1", "y1", "x2", "y2"]


def entry_side(box, w: int, h: int, margin: float = 0.06) -> str:
    """Where a track first appeared: the image edge its box touched, else 'inside view'."""
    x1, y1, x2, y2 = box
    sides = []
    if x1 <= w * margin:
        sides.append("left edge")
    if x2 >= w * (1 - margin):
        sides.append("right edge")
    if y1 <= h * margin:
        sides.append("top edge")
    if y2 >= h * (1 - margin):
        sides.append("bottom edge")
    return " / ".join(sides) if sides else "inside view (appeared mid-frame, e.g. from behind an occluder)"


class TrackHistory:
    """Appends one row per track per `interval` frames; remembers first sighting of every track."""

    def __init__(self, path: str | Path, interval: int = 30):
        self.path = Path(path)
        self.interval = max(1, int(interval))
        self._f = None
        self._w = None
        self._last: Dict[int, int] = {}
        self.first: Dict[int, Dict] = {}

    def reset(self):
        """New source: per-track memory restarts (the CSV keeps growing)."""
        self._last.clear()
        self.first.clear()

    def _writer(self):
        if self._w is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            new = not self.path.exists() or self.path.stat().st_size == 0
            self._f = open(self.path, "a", newline="", encoding="utf-8")
            self._w = csv.writer(self._f)
            if new:
                self._w.writerow(HISTORY_HEADER)
        return self._w

    def update(self, frame_idx: int, stamp: str, camera: str, source: str, size: Tuple[int, int],
               tid: int, box, identity: str, status: str):
        if tid not in self.first:
            self.first[tid] = {"time": stamp, "frame": frame_idx, "camera": camera, "source": str(source),
                               "box": [int(v) for v in box], "side": entry_side(box, *size)}
        if frame_idx - self._last.get(tid, -10**9) >= self.interval:
            self._last[tid] = frame_idx
            self._writer().writerow([stamp, camera, source, frame_idx, tid, identity, status, *map(int, box)])
            self._f.flush()

    def backtrack(self, tid: int) -> Dict:
        """Forensic origin of a track: when / where it first entered this camera view."""
        return dict(self.first.get(tid, {}))

    def close(self):
        if self._f is not None:
            self._f.close()
            self._f = self._w = None
