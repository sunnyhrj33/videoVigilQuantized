"""
core/chirla.py
────────────────────────────────────────────────────────────────────────────────
Helpers for the CHIRLA multi-camera office dataset:
  • Employee / visitor partition and train / test sequence split (from config)
  • Index of the benchmark person crops  (seq, camera, person_id, frame) → path
  • Camera-ID and start-time parsing from CHIRLA video filenames
────────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# Windows stores the ':' of CHIRLA filenames as the private-use char U+F03A;
# exported demo clips use '_' (e.g. camera_5_2023-06-02-12_03_33).
_COLON_CHARS = ":\uf03a_"
_NAME_RE = re.compile(
    r"camera_(\d+)_(\d{4}-\d{2}-\d{2})-(\d{2})[" + _COLON_CHARS + r"](\d{2})[" + _COLON_CHARS + r"](\d{2})"
)


@dataclass(frozen=True)
class CropRecord:
    seq: str
    camera: str        # e.g. "camera_3"
    pid: int           # CHIRLA person id (positive)
    frame: int
    path: Path


def chirla_cfg(cfg: dict) -> dict:
    return cfg.get("chirla", {})


def employee_ids(cfg: dict) -> List[int]:
    return [int(i) for i in chirla_cfg(cfg).get("employees", [])]


def visitor_ids(cfg: dict) -> List[int]:
    return [int(i) for i in chirla_cfg(cfg).get("visitors", [])]


def emp_label(pid: int) -> str:
    """Folder / gallery key for an enrolled employee, e.g. 4 → 'emp_04'."""
    return f"emp_{pid:02d}"


def parse_video_name(name: str) -> Tuple[Optional[str], Optional[datetime]]:
    """
    Return (camera_id, start_time) for a CHIRLA video filename such as
    'camera_2_2023-06-02-11:14:26.avi'. Either value is None if not parseable.
    """
    m = _NAME_RE.search(str(name))
    if not m:
        return None, None
    cam = f"Cam_{int(m.group(1)):02d}"
    try:
        start = datetime.strptime(
            f"{m.group(2)} {m.group(3)}:{m.group(4)}:{m.group(5)}", "%Y-%m-%d %H:%M:%S"
        )
    except ValueError:
        start = None
    return cam, start


def camera_key(name: str) -> Optional[str]:
    """'.../camera_3_2023-...avi' → 'camera_3' (key used in per-camera zone config)."""
    m = _NAME_RE.search(str(name))
    return f"camera_{int(m.group(1))}" if m else None


def sequence_dates(root: str | Path) -> Dict[str, str]:
    """{'seq_000': '2023-06-02', ...} from the video filenames."""
    out = {}
    for d in sorted((Path(root) / "videos").glob("seq_*")):
        for v in sorted(d.glob("camera_*.avi")):
            _, start = parse_video_name(v.name)
            if start is not None:
                out[d.name] = start.strftime("%Y-%m-%d")
                break
    return out


def source_date(name: str) -> Optional[str]:
    """Recording date ('2023-06-02') of a CHIRLA video / exported clip, if parseable."""
    _, start = parse_video_name(name)
    return start.strftime("%Y-%m-%d") if start else None


def day_gallery_path(gallery_file: str | Path, date: Optional[str]) -> Path:
    """enrolled_gallery/gallery_prototypes.pkl → enrolled_gallery/<date>/gallery_prototypes.pkl."""
    p = Path(gallery_file)
    return p.parent / date / p.name if date else p


def index_benchmark_crops(root: str | Path) -> List[CropRecord]:
    """
    Index every unique positive-ID crop under <root>/benchmark/**/imgs/<camera>/<id>/frame_N.png.
    The same crop appears in several benchmark splits; it is kept once.
    Negative IDs (CHIRLA's flagged / heavily-occluded samples) are skipped.
    """
    root = Path(root)
    seen: Dict[Tuple[str, str, int, int], CropRecord] = {}
    for p in (root / "benchmark").glob("**/imgs/*/*/frame_*.png"):
        try:
            pid = int(p.parent.name)
        except ValueError:
            continue
        if pid <= 0:
            continue
        cam = p.parent.parent.name.split("_20")[0]
        seq = p.parent.parent.parent.parent.name
        frame = int(p.stem.split("_")[-1])
        key = (seq, cam, pid, frame)
        if key not in seen:
            seen[key] = CropRecord(seq, cam, pid, frame, p)
    return sorted(seen.values(), key=lambda r: (r.seq, r.camera, r.pid, r.frame))


def imread(path: str | Path) -> Optional[np.ndarray]:
    """cv2.imread that also works with non-ASCII Windows paths (CHIRLA's U+F03A)."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


def video_path(root: str | Path, seq: str, camera: str) -> Optional[Path]:
    """Find the .avi for a given sequence / camera ('camera_3')."""
    for p in sorted((Path(root) / "videos" / seq).glob(f"{camera}_*.avi")):
        return p
    return None
