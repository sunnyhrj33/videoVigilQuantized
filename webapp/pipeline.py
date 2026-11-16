"""
webapp/pipeline.py
────────────────────────────────────────────────────────────────────────────────
Back end of the web dashboard (webapp/server.py): upload a video, get a people gallery.

  • KnownPeople   – people the dashboard user has named (+ read-only enrolled staff)
  • TrackLog      – everything one tracker ID did over the whole video
  • group_tracks  – merges the tracks that belong to one person (ID switches,
                    leaving and coming back) using Re-ID centroids
  • JobManager    – analysis queue on one worker thread (the engine keeps
                    per-source state, so videos are analysed one at a time)
────────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import copy
import json
import logging
import pickle
import queue
import re
import shutil
import subprocess
import threading
import time
import uuid
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from core.geometry import l2n, sharpness
from core.open_set import (STATUS_KNOWN, STATUS_PENDING, STATUS_UNKNOWN,
                           PrototypeGallery, display_name)

logger = logging.getLogger(__name__)

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg", ".wmv"}
STATUS_SUSPECT = "SUSPECT"
STATUS_UNCLEAR = "UNCLEAR"     # never seen clearly enough for a decision
OVERLAP_FRAMES = 3             # tracks visible together longer than this are different people
SEGMENT_GAP = 2.0              # seconds off screen that split one appearance into two
MAX_TRACK_EMBS = 80

DEFAULTS = {
    "host": "127.0.0.1", "port": 8000, "data_dir": "output/web", "detector": "yolo11n_int8_480",
    "analysis_fps": 15, "output_width": 1280, "max_upload_mb": 2048, "min_person_seconds": 1.0,
    "same_person_threshold": 0.75, "thumbs_per_person": 6,
}


class AnalysisError(Exception):
    """A problem with the uploaded video, worded for the dashboard user."""


def web_settings(cfg: dict) -> dict:
    return {**DEFAULTS, **(cfg.get("web") or {})}


def engine_config(cfg: dict, data_dir: Path) -> dict:
    """Engine config for the dashboard: every file the engine writes goes under data_dir."""
    c = copy.deepcopy(cfg)
    c.setdefault("output", {}).update(output_dir=str(data_dir), audit_log=str(data_dir / "audit_log.csv"),
                                      save_video=False)
    c.setdefault("suspect_tracking", {}).update(gallery_dir=str(data_dir / "suspects"),
                                                audit_log_path=str(data_dir / "theft_audit_log.csv"),
                                                history_log=str(data_dir / "track_history.csv"))
    c.setdefault("tracking", {})["tracker"] = "bytetrack"
    c.setdefault("reid", {})["mode"] = "tracklet"
    return c


class VideoSink:
    """
    Browser-playable annotated video. Preferred: H.264 MP4 through the ffmpeg bundled
    with imageio-ffmpeg. Fallback: OpenCV's VP8/WebM encoder.
    """

    def __init__(self, base: Path, frame: np.ndarray, fps: float, max_width: int):
        h, w = frame.shape[:2]
        if w > max_width:
            w, h = max_width, int(round(h * max_width / w))
        self.size = (w - w % 2, h - h % 2)
        self.proc = self.cv = None
        try:
            import imageio_ffmpeg
            exe = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception as exc:
            logger.warning("imageio-ffmpeg unavailable (%s) – using OpenCV VP8 (slower)", exc)
            exe = None
        if exe:
            self.path = base.with_suffix(".mp4")
            self._log = open(base.with_name("ffmpeg.log"), "wb")
            self.proc = subprocess.Popen(
                [exe, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                 "-s", f"{self.size[0]}x{self.size[1]}", "-r", f"{fps:.3f}", "-i", "-",
                 "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23", "-pix_fmt", "yuv420p",
                 "-movflags", "+faststart", str(self.path)],
                stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self._log,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        else:
            self.path = base.with_suffix(".webm")
            self.cv = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"VP80"), fps, self.size)
            if not self.cv.isOpened():
                raise AnalysisError("The annotated video could not be written: install imageio-ffmpeg "
                                    "(pip install imageio-ffmpeg).")

    def write(self, frame: np.ndarray):
        if (frame.shape[1], frame.shape[0]) != self.size:
            frame = cv2.resize(frame, self.size, interpolation=cv2.INTER_AREA)
        if self.cv is not None:
            self.cv.write(frame)
            return
        try:
            self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        except (BrokenPipeError, OSError):
            raise AnalysisError("The video encoder stopped unexpectedly (see ffmpeg.log in the job folder).")

    def close(self, ok: bool = True):
        if self.cv is not None:
            self.cv.release()
            return
        try:
            self.proc.stdin.close()
        except OSError:
            pass
        if not ok:
            self.proc.kill()
        code = self.proc.wait(timeout=600)
        self._log.close()
        if ok and code != 0:
            raise AnalysisError("The annotated video could not be finished (see ffmpeg.log in the job folder).")


def best_spread(photos: List[tuple], k: int, min_gap: float = 1.0) -> List[tuple]:
    """Best-scoring (score, t, img) photos, preferring ones at least min_gap seconds apart."""
    ranked = sorted(photos, key=lambda p: -p[0])
    chosen = []
    for p in ranked:
        if all(abs(p[1] - c[1]) >= min_gap for c in chosen):
            chosen.append(p)
        if len(chosen) == k:
            return chosen
    rest = [p for p in ranked if not any(p is c for c in chosen)]
    return chosen + rest[:k - len(chosen)]


# ─────────────────────────────────────────────────────────────────────────────
#  Known people (named from the dashboard) + enrolled staff (read-only)
# ─────────────────────────────────────────────────────────────────────────────

class KnownPeople:
    """
    People the dashboard user named, stored like an enrolled gallery
    ({"prototypes": {name: emb}, "meta": ...}) so the engine can merge it into
    every source's gallery.
    """

    NAME_RE = re.compile(r"^[\w .'\-]{2,40}$")

    def __init__(self, directory: Path, model: str, staff_gallery: Path):
        self.dir = Path(directory)
        self.file = self.dir / "gallery_prototypes.pkl"
        self.model = model
        self.staff_gallery = Path(staff_gallery)
        self._lock = threading.Lock()
        self.prototypes: Dict[str, np.ndarray] = {}
        self.info: Dict[str, dict] = {}
        if self.file.exists():
            with open(self.file, "rb") as f:
                data = pickle.load(f)
            self.prototypes = data.get("prototypes", {})
            self.info = data.get("meta", {}).get("people", {})

    def _save(self):
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = self.file.with_suffix(".tmp")
        with open(tmp, "wb") as f:
            pickle.dump({"prototypes": self.prototypes,
                         "meta": {"model": self.model, "date": "dashboard", "people": self.info}}, f)
        tmp.replace(self.file)

    def gallery(self) -> Optional[PrototypeGallery]:
        with self._lock:
            if not self.prototypes:
                return None
            labels = list(self.prototypes)
            return PrototypeGallery(labels, np.stack([self.prototypes[k] for k in labels]),
                                    {"model": self.model, "date": "dashboard"})

    def _staff(self) -> List[dict]:
        if not self.staff_gallery.exists():
            return []
        with open(self.staff_gallery, "rb") as f:
            data = pickle.load(f)
        date = data.get("meta", {}).get("date")
        out = []
        for label in data.get("prototypes", {}):
            folder = self.staff_dir(label)
            photos = sorted(p.name for p in folder.glob("*.jpg")) if folder else []
            out.append({"label": label, "name": display_name(label), "photos": photos, "date": date})
        return out

    def staff_dir(self, label: str) -> Optional[Path]:
        if not self.staff_gallery.exists():
            return None
        with open(self.staff_gallery, "rb") as f:
            date = pickle.load(f).get("meta", {}).get("date")
        for d in (self.staff_gallery.parent / str(date) / label, self.staff_gallery.parent / label):
            if d.is_dir():
                return d
        return None

    def person_dir(self, name: str) -> Optional[Path]:
        info = self.info.get(name)
        return self.dir / info["folder"] if info else None

    def validate_name(self, name: str) -> str:
        name = " ".join(str(name or "").split())
        if not self.NAME_RE.match(name):
            raise ValueError("Use 2–40 letters, numbers, spaces, dots, dashes or apostrophes.")
        low = name.lower()
        if low.startswith(("emp ", "emp_", "suspect")):
            raise ValueError("That name is reserved for staff and suspects – please pick another.")
        taken = {n.lower() for n in self.info} | {s["name"].lower() for s in self._staff()}
        if low in taken:
            raise ValueError(f"Someone called “{name}” is already known.")
        return name

    def add(self, name: str, embs: np.ndarray, photos: List[Path], source: str) -> str:
        name = self.validate_name(name)
        if embs is None or len(embs) == 0:
            raise ValueError("We didn't get a clear enough look at this person to remember them.")
        with self._lock:
            folder = re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_") + "_" + uuid.uuid4().hex[:4]
            dest = self.dir / folder
            dest.mkdir(parents=True, exist_ok=True)
            kept = []
            for k, p in enumerate(photos, 1):
                if Path(p).exists():
                    shutil.copyfile(p, dest / f"photo{k}.jpg")
                    kept.append(f"photo{k}.jpg")
            self.prototypes[name] = l2n(l2n(np.asarray(embs, np.float32)).mean(axis=0))
            self.info[name] = {"folder": folder, "photos": kept, "source": source, "samples": int(len(embs)),
                               "added": datetime.now().isoformat(timespec="seconds")}
            self._save()
        logger.info("✓ Dashboard: %s added to known people (%d samples)", name, len(embs))
        return name

    def remove(self, name: str):
        with self._lock:
            info = self.info.pop(name, None)
            if info is None:
                raise KeyError(name)
            self.prototypes.pop(name, None)
            shutil.rmtree(self.dir / info["folder"], ignore_errors=True)
            self._save()

    def listing(self) -> dict:
        with self._lock:
            yours = [{"name": n, **{k: v for k, v in i.items() if k != "folder"}} for n, i in self.info.items()]
        return {"yours": sorted(yours, key=lambda p: p["added"], reverse=True), "staff": self._staff()}


# ─────────────────────────────────────────────────────────────────────────────
#  Per-track log and grouping tracks into people
# ─────────────────────────────────────────────────────────────────────────────

class TrackLog:
    """What one tracker ID did over the whole video (the engine only keeps a short window)."""

    def __init__(self, tid: int):
        self.tid = tid
        self.frames: List[int] = []
        self.times: List[float] = []
        self.votes: Counter = Counter()
        self.conf: Dict[tuple, List[float]] = defaultdict(list)
        self.carrying = False
        self.intrusion = False
        self.embs: Dict[bytes, np.ndarray] = {}
        self.photos: List[tuple] = []
        self._last_photo_t = -1e9

    def observe(self, n: int, t: float, person: dict):
        self.frames.append(n)
        self.times.append(t)
        if person["status"] != STATUS_PENDING:
            key = (person["status"], person["name"])
            self.votes[key] += 1
            self.conf[key].append(float(person.get("confidence") or 0.0))
        self.carrying |= bool(person.get("carrying"))
        self.intrusion |= bool(person.get("intrusion"))

    def add_embeddings(self, embs: np.ndarray):
        for e in embs:
            if len(self.embs) >= MAX_TRACK_EMBS:
                return
            self.embs.setdefault(e.tobytes(), e)

    def offer_photo(self, frame: np.ndarray, box, t: float, keep: int):
        x1, y1, x2, y2 = (int(v) for v in box)
        h, w = y2 - y1, x2 - x1
        if h < 48 or w < 16 or t - self._last_photo_t < 0.5:
            return
        fh, fw = frame.shape[:2]
        px, py = int(w * 0.08), int(h * 0.05)
        crop = frame[max(0, y1 - py):min(fh, y2 + py), max(0, x1 - px):min(fw, x2 + px)]
        if crop.size == 0:
            return
        edge = x1 <= 2 or y1 <= 2 or x2 >= fw - 2 or y2 >= fh - 2
        squat = h < 1.6 * w
        score = h * min(sharpness(crop), 300.0) * (0.25 if edge else 1.0) * (0.5 if squat else 1.0)
        if crop.shape[0] > 256:
            crop = cv2.resize(crop, (max(1, int(crop.shape[1] * 256 / crop.shape[0])), 256),
                              interpolation=cv2.INTER_AREA)
        self.photos.append((score, t, crop.copy()))
        self._last_photo_t = t
        if len(self.photos) > keep * 3:
            self.photos = best_spread(self.photos, keep * 2)

    def verdict(self) -> tuple:
        if not self.votes:
            return STATUS_UNCLEAR, None, 0.0, 0
        suspect = [k for k in self.votes if k[0] == STATUS_SUSPECT]
        if suspect:
            key = max(suspect, key=self.votes.get)
        else:
            known = [k for k in self.votes if k[0] == STATUS_KNOWN]
            n_known = sum(self.votes[k] for k in known)
            n_unknown = sum(v for k, v in self.votes.items() if k[0] == STATUS_UNKNOWN)
            key = max(known, key=self.votes.get) if known and n_known >= n_unknown else \
                max((k for k in self.votes if k[0] == STATUS_UNKNOWN), key=self.votes.get)
        return key[0], key[1], float(np.mean(self.conf[key])), sum(self.votes.values())

    def centroid(self) -> Optional[np.ndarray]:
        if not self.embs:
            return None
        return l2n(l2n(np.stack(list(self.embs.values()))).mean(axis=0))


def group_tracks(tracks: List[dict], same_thresh: float) -> List[List[int]]:
    n = len(tracks)
    parent = list(range(n))
    seen = [set(t["frames"]) for t in tracks]
    names = [{t["name"]} if t["status"] in (STATUS_KNOWN, STATUS_SUSPECT) else set() for t in tracks]

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            a, b = tracks[i], tracks[j]
            if a["status"] in (STATUS_KNOWN, STATUS_SUSPECT) and a["status"] == b["status"] \
                    and a["name"] == b["name"]:
                pairs.append((2.0, i, j))
            elif a["centroid"] is not None and b["centroid"] is not None:
                sim = float(a["centroid"] @ b["centroid"])
                if sim >= same_thresh:
                    pairs.append((sim, i, j))

    for _, i, j in sorted(pairs, reverse=True):
        ri, rj = find(i), find(j)
        if ri == rj or len(seen[ri] & seen[rj]) > OVERLAP_FRAMES or len(names[ri] | names[rj]) > 1:
            continue
        parent[rj] = ri
        seen[ri] |= seen[rj]
        names[ri] |= names[rj]

    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    return sorted(groups.values(), key=lambda g: min(min(tracks[i]["frames"]) for i in g))


def group_verdict(items: List[dict]) -> tuple:
    suspects = [it for it in items if it["status"] == STATUS_SUSPECT]
    if suspects:
        return STATUS_SUSPECT, max(suspects, key=lambda it: it["weight"])["name"], None
    weight = Counter()
    for it in items:
        weight[it["status"]] += it["weight"]
    if not weight[STATUS_KNOWN] and not weight[STATUS_UNKNOWN]:
        return STATUS_UNCLEAR, None, None
    status = STATUS_KNOWN if weight[STATUS_KNOWN] >= weight[STATUS_UNKNOWN] else STATUS_UNKNOWN
    hits = [it for it in items if it["status"] == status]
    name = max(hits, key=lambda it: it["weight"])["name"]
    w = sum(it["weight"] for it in hits)
    p = sum(it["conf"] * it["weight"] for it in hits) / max(w, 1)
    return status, name, round(p if status == STATUS_KNOWN else 1.0 - p, 3)


def segments(times: List[float], step: float) -> List[List[float]]:
    out = []
    for t in sorted(set(times)):
        if out and t - out[-1][1] <= SEGMENT_GAP:
            out[-1][1] = t
        else:
            out.append([t, t])
    return [[round(a, 2), round(b + step, 2)] for a, b in out]


# ─────────────────────────────────────────────────────────────────────────────
#  Jobs
# ─────────────────────────────────────────────────────────────────────────────

class JobManager:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.s = web_settings(cfg)
        self.data_dir = Path(self.s["data_dir"])
        self.jobs_dir = self.data_dir / "jobs"
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        reid = cfg.get("reid", {})
        self.known = KnownPeople(self.data_dir / "known_people", reid.get("model", ""),
                                 Path(reid.get("gallery_file", "enrolled_gallery/gallery_prototypes.pkl")))
        self.engine = None
        self.engine_state = "loading"
        self.engine_error: Optional[str] = None
        self._jobs: Dict[str, dict] = {}
        self._lock = threading.Lock()
        self._queue: "queue.Queue[str]" = queue.Queue()
        self._load_jobs()
        threading.Thread(target=self._worker, name="web-analysis", daemon=True).start()

    # ── persistence ──────────────────────────────────────────────────────────

    def _load_jobs(self):
        for f in sorted(self.jobs_dir.glob("*/job.json")):
            try:
                job = json.loads(f.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning("Skipping unreadable job %s (%s)", f, exc)
                continue
            self._jobs[job["id"]] = job
            if job["status"] in ("queued", "running"):
                job.update(status="queued", stage="Waiting in line", progress=0.0, eta=None)
                self._queue.put(job["id"])

    def _save(self, job: dict):
        d = self.jobs_dir / job["id"]
        if not d.exists():
            return
        tmp = d / "job.json.tmp"
        tmp.write_text(json.dumps(job, indent=1), encoding="utf-8")
        tmp.replace(d / "job.json")

    def _update(self, job_id: str, save: bool = False, **fields):
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.update(fields)
            if save:
                self._save(job)

    # ── public API ───────────────────────────────────────────────────────────

    def job_dir(self, job_id: str) -> Optional[Path]:
        with self._lock:
            return self.jobs_dir / job_id if job_id in self._jobs else None

    def get(self, job_id: str) -> Optional[dict]:
        with self._lock:
            job = self._jobs.get(job_id)
            return copy.deepcopy(job) if job else None

    def list(self) -> List[dict]:
        with self._lock:
            jobs = [{k: v for k, v in j.items() if k not in ("people", "events")} for j in self._jobs.values()]
        return sorted(jobs, key=lambda j: j["created"], reverse=True)

    def status(self) -> dict:
        with self._lock:
            running = [j["id"] for j in self._jobs.values() if j["status"] == "running"]
            waiting = sum(1 for j in self._jobs.values() if j["status"] == "queued")
        return {"engine": self.engine_state, "error": self.engine_error, "running": running, "waiting": waiting}

    def create(self, filename: str, save_upload) -> dict:
        ext = Path(filename).suffix.lower()
        if ext not in VIDEO_EXTS:
            raise ValueError(f"“{filename}” isn't a supported video. Use one of: "
                             + ", ".join(sorted(e.lstrip('.') for e in VIDEO_EXTS)) + ".")
        job_id = datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
        d = self.jobs_dir / job_id
        d.mkdir(parents=True)
        path = d / filename
        try:
            save_upload(path)
        except Exception:
            shutil.rmtree(d, ignore_errors=True)
            raise
        job = {"id": job_id, "name": filename, "video": filename, "created": datetime.now().isoformat(timespec="seconds"),
               "size_mb": round(path.stat().st_size / 2**20, 1), "status": "queued", "stage": "Waiting in line",
               "progress": 0.0, "eta": None, "error": None, "summary": None, "people": [], "events": []}
        with self._lock:
            self._jobs[job_id] = job
            self._save(job)
        self._queue.put(job_id)
        return copy.deepcopy(job)

    def delete(self, job_id: str):
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            if job["status"] == "running":
                raise RuntimeError("This video is being analysed right now – wait for it to finish.")
            del self._jobs[job_id]
        shutil.rmtree(self.jobs_dir / job_id, ignore_errors=True)

    def remember_person(self, job_id: str, pid: str, name: str) -> dict:
        with self._lock:
            job = self._jobs.get(job_id)
            person = next((p for p in (job or {}).get("people", []) if p["id"] == pid), None)
        if person is None:
            raise KeyError(pid)
        if person["status"] == STATUS_KNOWN:
            raise ValueError("This person is already recognised.")
        embs = None
        npz = self.jobs_dir / job_id / "people.npz"
        if npz.exists():
            with np.load(npz) as data:
                embs = data[pid] if pid in data.files else None
        photos = [self.jobs_dir / job_id / p for p in person["photos"]]
        name = self.known.add(name, embs, photos, job["name"])
        with self._lock:
            old = person["name"]
            person.update(status=STATUS_KNOWN, name=name, named_by_you=True, confidence=None)
            for ev in job["events"]:
                if ev.get("person") == pid:
                    ev["text"] = ev["text"].replace(old, name)
            job["summary"] = self._summary(job["people"], job["events"], job["summary"]["duration"],
                                           job["summary"]["processing_s"])
            self._save(job)
            return copy.deepcopy(person)

    def suspects(self):
        if self.engine is not None:
            return self.engine.suspects
        from core.suspects import SuspectGallery
        return SuspectGallery(self.data_dir / "suspects")

    def delete_suspect(self, sid: str):
        if self.status()["running"]:
            raise RuntimeError("Wait for the current analysis to finish, then try again.")
        gallery = self.suspects()
        if sid not in gallery.suspects:
            raise KeyError(sid)
        gallery.delete(sid)

    # ── worker ───────────────────────────────────────────────────────────────

    def _start_engine(self):
        try:
            from core.engine import DetectionEngine
            engine = DetectionEngine(engine_config(self.cfg, self.data_dir))
            engine.load_model(self.s["detector"])
            engine.show_fps = False
            engine.show_tid = False
            self.engine = engine
            self.engine_state = "ready"
            logger.info("✓ Dashboard engine ready (%s)", self.s["detector"])
        except Exception as exc:
            logger.exception("Dashboard engine failed to start")
            self.engine_state, self.engine_error = "error", str(exc)

    def _worker(self):
        self._start_engine()
        while True:
            job_id = self._queue.get()
            job = self.get(job_id)
            if job is None or job["status"] != "queued":
                continue
            if self.engine is None:
                self._update(job_id, save=True, status="failed", stage="Failed",
                             error=f"The AI models could not be loaded: {self.engine_error}")
                continue
            self._update(job_id, save=True, status="running", stage="Analysing the video", progress=0.0)
            t0 = time.time()
            try:
                result = self._analyse(job_id, job)
                self._update(job_id, save=True, status="done", stage="Finished", progress=1.0, eta=None,
                             **result, processing_s=round(time.time() - t0, 1))
                logger.info("✓ Dashboard job %s done in %.0fs: %s", job_id, time.time() - t0, result["summary"])
            except AnalysisError as exc:
                self._update(job_id, save=True, status="failed", stage="Failed", error=str(exc))
            except Exception as exc:
                logger.exception("Dashboard job %s failed", job_id)
                self._update(job_id, save=True, status="failed", stage="Failed",
                             error=f"Something went wrong while analysing this video ({exc}).")

    def _analyse(self, job_id: str, job: dict) -> dict:
        d = self.jobs_dir / job_id
        cap = cv2.VideoCapture(str(d / job["video"]))
        if not cap.isOpened():
            raise AnalysisError("This file could not be opened as a video.")
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        fps = fps if 1.0 <= fps <= 240.0 else 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        step = max(1, round(fps / float(self.s["analysis_fps"])))
        out_fps = fps / step
        keep = int(self.s["thumbs_per_person"])
        engine = self.engine

        engine.extra_gallery = self.known.gallery()
        engine.set_source(str(d / job["video"]), fps=out_fps)
        last_ev = max((e[0] for e in engine.events_since(0)), default=0)

        logs: Dict[int, TrackLog] = {}
        raw_events: List[dict] = []
        sink: Optional[VideoSink] = None
        poster_at, poster_done = int(total * 0.1), False
        i = n = 0
        t0 = time.time()
        ok_run = False
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                t = i / fps
                annotated, stats = engine.process_frame(frame)
                for p in stats.get("people", []):
                    log = logs.setdefault(p["tid"], TrackLog(p["tid"]))
                    log.observe(n, t, p)
                    log.offer_photo(frame, p["box"], t, keep)
                    if n % 3 == 0:
                        log.add_embeddings(engine.track_embeddings(p["tid"])[0])
                for no, level, _stamp, _cam, text in engine.events_since(last_ev):
                    last_ev = no
                    if level not in ("info", "identity"):
                        raw_events.append({"t": round(t, 2), "level": level, "text": text})

                if sink is None:
                    sink = VideoSink(d / "annotated", annotated, out_fps, int(self.s["output_width"]))
                sink.write(annotated)
                if n == 0 or (not poster_done and i >= poster_at):
                    self._write_poster(d / "poster.jpg", frame)
                    poster_done = n > 0

                n += 1
                for _ in range(step - 1):
                    if not cap.grab():
                        break
                i += step
                if n % 10 == 0 and total:
                    frac = min(i / total, 1.0)
                    el = time.time() - t0
                    self._update(job_id, progress=round(frac * 0.97, 3),
                                 eta=round(el / frac - el) if frac > 0.02 else None)
            ok_run = True
        finally:
            cap.release()
            if sink is not None:
                sink.close(ok=ok_run)
        if n == 0:
            raise AnalysisError("No frames could be read from this video.")

        self._update(job_id, stage="Building the people gallery", progress=0.98, eta=None)
        people, events = self._build_people(d, logs, raw_events, out_fps)
        duration = round(i / fps, 2)
        return {"people": people, "events": events, "fps": round(out_fps, 2),
                "summary": self._summary(people, events, duration, round(time.time() - t0, 1)),
                "video_url": sink.path.name, "poster": "poster.jpg"}

    @staticmethod
    def _write_poster(path: Path, frame: np.ndarray):
        h, w = frame.shape[:2]
        if w > 640:
            frame = cv2.resize(frame, (640, int(h * 640 / w)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])

    def _build_people(self, d: Path, logs: Dict[int, TrackLog], raw_events: List[dict], fps: float):
        items = []
        for log in logs.values():
            status, name, conf, weight = log.verdict()
            items.append({"log": log, "status": status, "name": name, "conf": conf, "weight": weight,
                          "centroid": log.centroid(), "frames": log.frames})
        groups = group_tracks(items, float(self.s["same_person_threshold"]))

        thumbs = d / "thumbs"
        shutil.rmtree(thumbs, ignore_errors=True)
        thumbs.mkdir()
        people, embs_out, tid_person = [], {}, {}
        n_visitor = 0
        for g in groups:
            members = [items[k] for k in g]
            frames = {f for it in members for f in it["log"].frames}
            if len(frames) / fps < float(self.s["min_person_seconds"]):
                continue
            status, name, conf = group_verdict(members)
            pid = f"P{len(people) + 1:02d}"
            if status in (STATUS_UNKNOWN, STATUS_UNCLEAR):
                n_visitor += 1
                name = f"Visitor {n_visitor}"
            elif status == STATUS_SUSPECT:
                name = name.title()
            photos = best_spread([p for it in members for p in it["log"].photos], int(self.s["thumbs_per_person"]))
            photos.sort(key=lambda p: -p[0])
            files = []
            for k, (_, _, img) in enumerate(photos, 1):
                cv2.imwrite(str(thumbs / f"{pid}_{k}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
                files.append(f"thumbs/{pid}_{k}.jpg")
            embs = [e for it in members for e in it["log"].embs.values()]
            if embs:
                embs_out[pid] = np.stack(embs).astype(np.float32)
            times = [t for it in members for t in it["log"].times]
            for it in members:
                tid_person[it["log"].tid] = (pid, name)
            people.append({
                "id": pid, "status": status, "name": name, "confidence": conf,
                "first_seen": round(min(times), 2), "last_seen": round(max(times) + 1 / fps, 2),
                "visible_s": round(len(frames) / fps, 1), "segments": segments(times, 1 / fps),
                "photos": files, "carrying": any(it["log"].carrying for it in members),
                "intrusion": any(it["log"].intrusion for it in members),
                "tracks": len(members), "can_remember": pid in embs_out,
            })
        np.savez(d / "people.npz", **embs_out)

        events = [{"t": p["first_seen"], "level": "person", "person": p["id"],
                   "text": f"{p['name']} appeared"} for p in people]
        for ev in raw_events:
            pids = []

            def name_of(m):
                hit = tid_person.get(int(m.group(1)))
                if hit:
                    pids.append(hit[0])
                return hit[1] if hit else "someone"
            text = re.sub(r"\bT(\d+)\b", name_of, ev["text"])
            events.append({**ev, "text": text, "person": pids[0] if pids else None})
        events.sort(key=lambda e: e["t"])
        return people, events

    @staticmethod
    def _summary(people: List[dict], events: List[dict], duration: float, processing_s: float) -> dict:
        count = Counter(p["status"] for p in people)
        return {"duration": duration, "processing_s": processing_s, "people": len(people),
                "known": count[STATUS_KNOWN], "unknown": count[STATUS_UNKNOWN] + count[STATUS_UNCLEAR],
                "suspects": count[STATUS_SUSPECT],
                "alerts": sum(1 for e in events if e["level"] in ("alert", "suspect", "asset"))}
