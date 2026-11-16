"""
console/checkin.py
────────────────────────────────────────────────────────────────────────────────
Morning check-in analysis for the desktop console (Gallery tab → Build from
check-in video): run the engine over one video or every camera of a recorded
sequence and turn the tracks into person cards (photos + Re-ID views), ready for
core.enrollment.build_staff / write_day_gallery.

The same track logging / grouping as the web dashboard (webapp.pipeline) is used;
tracks are only merged within one camera.
────────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from console.worker import frame_step
from webapp.pipeline import (STATUS_SUSPECT, STATUS_UNCLEAR, TrackLog, best_spread,
                             group_tracks, group_verdict, web_settings)
from core.open_set import STATUS_UNKNOWN

logger = logging.getLogger("console.checkin")


class CheckinCancelled(Exception):
    pass


def collect_people(engine, videos: Sequence[Tuple[str, str]], date: str, out_dir: Path, cfg: dict,
                   progress: Optional[Callable[[float, str], None]] = None,
                   stop: Optional[threading.Event] = None) -> Tuple[List[dict], Dict[str, np.ndarray]]:
    """
    videos: [(path, camera label)]. Returns (people, embeddings):
      people = [{"id", "cam", "status", "name", "confidence", "photos": [Path], "visible_s"}]
      embeddings = {id: (n, d) Re-ID views}  (only people seen clearly enough)
    Asset monitoring and zones are switched off for the run (a check-in must not raise
    theft alerts or enrol suspects) and restored afterwards.
    """
    s = web_settings(cfg)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    keep = int(s["thumbs_per_person"])
    min_s = float(s["min_person_seconds"])
    people: List[dict] = []
    embs_out: Dict[str, np.ndarray] = {}
    saved = (engine._assets.enabled, engine.zones_enabled, engine._identity_mode, engine._reid_enabled)
    engine._assets.enabled, engine.zones_enabled, engine._reid_enabled = False, False, True
    engine.set_identity_mode("tracklet")         # the Re-ID views come from the tracklet buffers
    try:
        for v, (path, cam) in enumerate(videos):
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                logger.warning("Check-in: cannot open %s", path)
                if progress:
                    progress(v / len(videos), f"{cam}: could not open {Path(path).name} – skipped")
                continue
            fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
            fps = fps if 1.0 <= fps <= 240.0 else 30.0
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            step = frame_step(fps, float(s["analysis_fps"]))
            out_fps = fps / step
            engine.set_source(str(path), fps=out_fps, gallery_date=date)
            logs: Dict[int, TrackLog] = {}
            i = n = 0
            t0 = time.time()
            try:
                while True:
                    if stop is not None and stop.is_set():
                        raise CheckinCancelled()
                    ok, frame = cap.read()
                    if not ok:
                        break
                    _, stats = engine.process_frame(frame)
                    t = i / fps
                    for p in stats.get("people", []):
                        log = logs.setdefault(p["tid"], TrackLog(p["tid"]))
                        log.observe(n, t, p)
                        log.offer_photo(frame, p["box"], t, keep)
                        if n % 3 == 0:
                            log.add_embeddings(engine.track_embeddings(p["tid"])[0])
                    n += 1
                    for _ in range(step - 1):
                        if not cap.grab():
                            break
                    i += step
                    if progress and n % 10 == 0:
                        frac = min(i / total, 1.0) if total else 0.0
                        el = time.time() - t0
                        eta = f", about {el / frac - el:.0f} s left" if frac > 0.05 else ""
                        progress((v + frac) / len(videos), f"{cam}: analysing {frac:.0%}{eta}")
            finally:
                cap.release()
            people += _people_of_camera(logs, cam, out_fps, min_s, keep, out_dir, embs_out,
                                        float(s["same_person_threshold"]), start=len(people))
    finally:
        engine._assets.enabled, engine.zones_enabled, mode, engine._reid_enabled = saved
        engine.set_identity_mode(mode)
    if progress:
        progress(1.0, f"Found {len(people)} people")
    return people, embs_out


def _people_of_camera(logs, cam, fps, min_s, keep, out_dir, embs_out, same_thresh, start) -> List[dict]:
    items = []
    for log in logs.values():
        status, name, conf, weight = log.verdict()
        items.append({"log": log, "status": status, "name": name, "conf": conf, "weight": weight,
                      "centroid": log.centroid(), "frames": log.frames})
    out = []
    for g in group_tracks(items, same_thresh):
        members = [items[k] for k in g]
        frames = {f for it in members for f in it["log"].frames}
        if len(frames) / fps < min_s:
            continue
        status, name, conf = group_verdict(members)
        pid = f"P{start + len(out) + 1:02d}"
        if status in (STATUS_UNKNOWN, STATUS_UNCLEAR):
            name = f"Visitor {start + len(out) + 1}"
        elif status == STATUS_SUSPECT:
            name = (name or "Suspect").title()
        photos = best_spread([p for it in members for p in it["log"].photos], keep)
        photos.sort(key=lambda p: -p[0])
        files = []
        for k, (_, _, img) in enumerate(photos, 1):
            f = out_dir / f"{pid}_{k}.jpg"
            cv2.imwrite(str(f), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
            files.append(f)
        embs = [e for it in members for e in it["log"].embs.values()]
        if embs:
            embs_out[pid] = np.stack(embs).astype(np.float32)
        out.append({"id": pid, "cam": cam, "status": status, "name": name or "?", "confidence": conf,
                    "photos": files, "visible_s": round(len(frames) / fps, 1)})
    return out
