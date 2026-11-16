"""
core/recordings.py
────────────────────────────────────────────────────────────────────────────────
Recorded days for multi-camera playback:

    <root>/<YYYY-MM-DD>/<sequence>/camera_<N>_<date>-<hh:mm:ss>.avi

  • scan_days        – {date: [Sequence]}; a sequence holds one video per camera
  • Sequence.missing – expected cameras with no video (played without them)
  • sync_offsets     – start frame per camera so every view shows the same moment
────────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from core.chirla import camera_key, is_iso_date, parse_video_name

logger = logging.getLogger(__name__)

VIDEO_EXTS = {".avi", ".mp4", ".mov", ".mkv", ".m4v", ".wmv", ".mpg", ".mpeg", ".webm"}


@dataclass(frozen=True)
class CameraVideo:
    camera: str                    # "camera_3" (key used by per-camera zones / anchors)
    cam_id: str                    # "Cam_03"   (shown in the UI and logs)
    path: Path
    start: Optional[datetime]      # wall-clock start from the file name

    @property
    def number(self) -> int:
        return int(self.camera.split("_")[-1])


@dataclass
class Sequence:
    name: str                      # "seq_000"
    date: str                      # "2023-06-02"
    path: Path
    cameras: List[CameraVideo] = field(default_factory=list)
    missing: List[str] = field(default_factory=list)     # e.g. ["camera_5"]
    duplicates: List[Path] = field(default_factory=list)  # extra videos of a camera (ignored)

    @property
    def start(self) -> Optional[datetime]:
        starts = [c.start for c in self.cameras if c.start is not None]
        return min(starts) if starts else None


def recordings_cfg(cfg: dict) -> dict:
    return {"root": "recordings", "cameras": 7, **(cfg.get("recordings") or {})}


def scan_sequence(path: Path, date: str, expected: int = 7) -> Sequence:
    """One sequence folder → its camera videos (one per camera, sorted by camera number)."""
    seq = Sequence(path.name, date, path)
    by_cam: Dict[str, List[Path]] = {}
    for f in sorted(path.iterdir()):
        if not f.is_file() or f.suffix.lower() not in VIDEO_EXTS:
            continue
        key = camera_key(f.name)
        if key is None:
            logger.debug("Skipping %s: no camera_<N> in the name", f)
            continue
        by_cam.setdefault(key, []).append(f)
    for key, files in by_cam.items():
        cam_id, start = parse_video_name(files[0].name)
        seq.cameras.append(CameraVideo(key, cam_id or key, files[0], start))
        seq.duplicates += files[1:]
    seq.cameras.sort(key=lambda c: c.number)
    present = {c.camera for c in seq.cameras}
    seq.missing = [f"camera_{n}" for n in range(1, expected + 1) if f"camera_{n}" not in present]
    return seq


def scan_days(root: str | Path, expected: int = 7) -> Dict[str, List[Sequence]]:
    """{'2023-06-02': [Sequence(seq_000), ...]} for every date folder with at least one camera video."""
    root = Path(root)
    days: Dict[str, List[Sequence]] = {}
    if not root.is_dir():
        return days
    for day in sorted(p for p in root.iterdir() if p.is_dir() and is_iso_date(p.name)):
        seqs = [scan_sequence(s, day.name, expected) for s in sorted(p for p in day.iterdir() if p.is_dir())]
        seqs = [s for s in seqs if s.cameras]
        if seqs:
            days[day.name] = seqs
    return days


def sync_offsets(cameras: List[CameraVideo], fps: Dict[str, float]) -> Dict[str, int]:
    """
    Start frame per camera so all views begin at the same wall-clock moment: the latest
    camera start. Cameras without a parseable start time begin at frame 0.
    """
    starts = [c.start for c in cameras if c.start is not None]
    if not starts:
        return {c.camera: 0 for c in cameras}
    t0 = max(starts)
    return {c.camera: (max(0, round((t0 - c.start).total_seconds() * fps.get(c.camera, 30.0)))
                       if c.start is not None else 0)
            for c in cameras}
