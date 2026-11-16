"""
console/worker.py
────────────────────────────────────────────────────────────────────────────────
Background workers for the operator console:
  • QueueLogHandler  – forwards Python logging records to the UI queue
  • VideoWorker      – reads frames from OpenCV VideoCapture and runs the engine
  • SubprocessRunner – runs offline CLI scripts with streamed stdout & progress
────────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import logging
import os
import queue
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable, List, Optional

import cv2

from console.theme import IS_WIN
from core.engine import DetectionEngine

logger = logging.getLogger("console.worker")

PIPE_NOISE = re.compile(
    r"Cython|warnings\.warn|params:|flops:|^Model:|discarded|Successfully loaded|^\s*$"
)
PROGRESS_RE = re.compile(r"embedded (\d+) / (\d+)")
INDEX_RE = re.compile(r"\[(\d+)/(\d+)\].*indexed (\d+) / (\d+) frames")


class QueueLogHandler(logging.Handler):
    """Forward log records to the UI log panel."""

    def __init__(self, q: queue.Queue):
        super().__init__()
        self.q = q

    def emit(self, record):
        try:
            self.q.put_nowait(("log", record.levelno, record.name, record.getMessage(), record.created))
        except Exception:
            pass


class VideoWorker(threading.Thread):
    """Reads frames, runs the engine, pushes results to a queue for the GUI."""

    def __init__(self, source, engine: DetectionEngine, out_q: queue.Queue, target_fps: int = 30,
                 realtime: bool = False, start_frame: int = 0, gallery_date: Optional[str] = None,
                 analysis_fps: Optional[float] = None):
        super().__init__(daemon=True)
        self.source = source
        self.start_frame = int(start_frame)
        self.gallery_date = gallery_date      # recording date → that day's staff gallery
        self.engine = engine
        self.out_q = out_q
        self.target_fps = target_fps
        self.realtime = realtime
        self.analysis_fps = analysis_fps      # frames analysed per video second (None = all); live-tunable
        self._stop_evt = threading.Event()

    def stop(self):
        self._stop_evt.set()

    def run(self):
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            self.out_q.put(("error", f"Cannot open source: {self.source}"))
            return

        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        if self.start_frame and isinstance(self.source, str):
            cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
        self.engine.set_source(self.source, src_fps, self.start_frame, gallery_date=self.gallery_date)
        is_file = isinstance(self.source, str)
        t_start, frame_no = time.perf_counter(), 0

        while not self._stop_evt.is_set():
            t0 = time.perf_counter()
            step = frame_step(src_fps, self.analysis_fps) if is_file else 1
            frame_delay = step / src_fps if step > 1 else (1.0 / self.target_fps if self.target_fps > 0 else 0.0)
            if self.realtime and is_file:
                target = int((t0 - t_start) * src_fps)
                while frame_no < target - 1:
                    if not cap.grab():
                        break
                    frame_no += 1
                    self.engine.skip_frame()
            ok, frame = cap.read()
            frame_no += 1
            if not ok:
                self.out_q.put(("done", None))
                break
            try:
                annotated, stats = self.engine.process_frame(frame)
                if self.out_q.qsize() < 3:
                    self.out_q.put(("frame", (annotated, stats)))
            except Exception as exc:
                logger.exception("Frame processing error")
                self.out_q.put(("error", str(exc)))
                break
            for _ in range(step - 1):           # analysis rate below the video's frame rate
                if not cap.grab():
                    break
                frame_no += 1
                self.engine.skip_frame()
            elapsed = time.perf_counter() - t0
            if frame_delay > elapsed:
                time.sleep(frame_delay - elapsed)
        cap.release()


def frame_step(src_fps: float, analysis_fps: Optional[float]) -> int:
    """Analyse every `step`-th frame so about analysis_fps frames per video second are analysed."""
    if not analysis_fps or analysis_fps <= 0 or not src_fps:
        return 1
    return max(1, round(src_fps / analysis_fps))


def analysis_rate(choice, n_cameras: int, video_cfg: dict) -> Optional[float]:
    """
    Frames analysed per video second, per camera, from a setting:
    'auto' → all frames for one video, video.multi_camera_fps for several cameras;
    'all' → None (every frame); a number → that rate.
    """
    text = str(choice if choice is not None else "auto").strip().lower().replace("fps", "").strip()
    if text in ("", "auto"):
        return float(video_cfg.get("multi_camera_fps", 5)) if n_cameras > 1 else None
    if text in ("all", "all frames"):
        return None
    try:
        value = float(text)
    except ValueError:
        logger.warning("Analysis rate %r not understood – analysing every frame", choice)
        return None
    return value if value > 0 else None


class MultiCameraWorker(threading.Thread):
    """
    Plays every camera of one recorded sequence together on one thread, one engine per
    camera, time-aligned from the file-name start times. Each round advances the shared
    video clock by 1/analysis_fps seconds and analyses one frame per camera; with
    `realtime`, rounds that fall behind the wall clock skip ahead (frames are grabbed,
    not analysed). Running the cameras in turn keeps the shared Re-ID model, suspect
    gallery and logs single-threaded.

    Queue messages: ("status", text) · ("warning", text) · ("multi", payload) ·
    ("done", None) · ("error", text). payload = {"frames": [...], "stats": [...],
    "ended": [...]} aligned with self.cameras (the cameras that opened).
    """

    def __init__(self, cameras, engine_for: Callable[[int], DetectionEngine], out_q: queue.Queue,
                 gallery_date: Optional[str], analysis_fps: Optional[float], realtime: bool = True):
        super().__init__(daemon=True)
        self.requested = list(cameras)        # core.recordings.CameraVideo
        self.cameras: list = []               # the ones that opened
        self.engines: List[DetectionEngine] = []
        self.engine_for = engine_for          # i → engine for the i-th opened camera (built on demand)
        self.out_q = out_q
        self.gallery_date = gallery_date
        self.analysis_fps = analysis_fps      # live-tunable
        self.realtime = realtime              # live-tunable
        self._stop_evt = threading.Event()

    def stop(self):
        self._stop_evt.set()

    def _open(self):
        from core.recordings import sync_offsets

        caps, fps = [], {}
        for cam in self.requested:
            cap = cv2.VideoCapture(str(cam.path))
            if not cap.isOpened():
                self.out_q.put(("warning", f"{cam.cam_id}: could not open {cam.path.name} – playing without it"))
                logger.warning("Multi-camera: cannot open %s", cam.path)
                continue
            self.cameras.append(cam)
            caps.append(cap)
            fps[cam.camera] = cap.get(cv2.CAP_PROP_FPS) or 30.0
        offsets = sync_offsets(self.cameras, fps)
        for i, (cam, cap) in enumerate(zip(self.cameras, caps)):
            if self._stop_evt.is_set():
                break
            self.out_q.put(("status", f"Preparing {cam.cam_id} ({i + 1}/{len(self.cameras)}) …"))
            engine = self.engine_for(i)
            if offsets[cam.camera]:
                cap.set(cv2.CAP_PROP_POS_FRAMES, offsets[cam.camera])
            engine.set_source(str(cam.path), fps[cam.camera], offsets[cam.camera], gallery_date=self.gallery_date)
            self.engines.append(engine)
        return caps, fps, offsets

    def run(self):
        caps: list = []
        try:
            caps, fps, offsets = self._open()
            if not caps:
                self.out_q.put(("error", "None of the camera videos could be opened."))
                return
            n = len(caps)
            pos = [offsets[c.camera] for c in self.cameras]       # next frame index to read
            ended = [False] * n
            last = [None] * n
            stats = [{} for _ in range(n)]
            vt, t_start = 0.0, time.perf_counter()
            self.out_q.put(("status", f"Warming up {n} detector{'s' if n > 1 else ''} – the first frames "
                                      f"take a few seconds …"))
            while not self._stop_evt.is_set() and not all(ended):
                fastest = max(fps.values())
                rate = self.analysis_fps or fastest
                vt_next = vt + 1.0 / rate
                if self.realtime:
                    vt_next = max(vt_next, time.perf_counter() - t_start)
                for i, (cam, cap, engine) in enumerate(zip(self.cameras, caps, self.engines)):
                    if ended[i] or self._stop_evt.is_set():
                        continue
                    target = offsets[cam.camera] + int(vt_next * fps[cam.camera])
                    if pos[i] >= target:                           # rate above this video's fps
                        continue
                    while pos[i] < target - 1:                     # skipped, not analysed
                        if not cap.grab():
                            ended[i] = True
                            break
                        pos[i] += 1
                        engine.skip_frame()
                    if ended[i]:
                        continue
                    ok, frame = cap.read()
                    if not ok:
                        ended[i] = True
                        continue
                    pos[i] += 1
                    last[i], stats[i] = engine.process_frame(frame)
                vt = vt_next
                if self.out_q.qsize() < 3:
                    self.out_q.put(("multi", {"frames": list(last), "stats": list(stats), "ended": list(ended),
                                              "labels": [c.cam_id for c in self.cameras]}))
                ahead = vt - (time.perf_counter() - t_start)       # never faster than real time
                if ahead > 0:
                    self._stop_evt.wait(ahead)
            if not self._stop_evt.is_set():
                self.out_q.put(("done", None))
        except Exception as exc:
            logger.exception("Multi-camera playback error")
            self.out_q.put(("error", str(exc)))
        finally:
            for cap in caps:
                cap.release()


class SubprocessRunner:
    """
    Reusable background subprocess runner for offline pipeline steps and search
    index builds. Streams stdout lines onto a Tkinter widget via `.after()` polling.
    """

    def __init__(self, tk_root, cwd: Path):
        self.root = tk_root
        self.cwd = cwd
        self.proc: Optional[subprocess.Popen] = None
        self._q: "queue.Queue[tuple]" = queue.Queue()
        self._on_line: Optional[Callable[[str], None]] = None
        self._on_done: Optional[Callable[[int, float], None]] = None
        self._t0: float = 0.0

    @property
    def is_running(self) -> bool:
        return self.proc is not None

    def start(self, cmd_args: List[str], on_line: Callable[[str], None],
              on_done: Callable[[int, float], None]):
        if self.proc is not None:
            raise RuntimeError("A subprocess is already running.")
        self._on_line = on_line
        self._on_done = on_done
        self._t0 = time.time()
        env = dict(
            os.environ,
            PYTHONIOENCODING="utf-8",
            PYTHONUNBUFFERED="1",
            PYTHONPATH=str(self.cwd),
        )
        flags = subprocess.CREATE_NO_WINDOW if IS_WIN else 0
        self.proc = subprocess.Popen(
            [sys.executable, "-u", *cmd_args],
            cwd=self.cwd, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace",
            creationflags=flags,
        )
        threading.Thread(target=self._reader, args=(self.proc,), daemon=True).start()
        self.root.after(100, self._poll)

    def _reader(self, proc: subprocess.Popen):
        for line in proc.stdout:
            self._q.put(("line", line.rstrip("\n")))
        rc = proc.wait()
        self._q.put(("done", rc, time.time() - self._t0))

    def _poll(self):
        try:
            while True:
                item = self._q.get_nowait()
                if item[0] == "line":
                    if self._on_line:
                        self._on_line(item[1])
                elif item[0] == "done":
                    _, rc, elapsed = item
                    self.proc = None
                    if self._on_done:
                        self._on_done(rc, elapsed)
                    return
        except queue.Empty:
            pass
        if self.proc is not None:
            self.root.after(100, self._poll)

    def terminate(self):
        if self.proc is not None:
            try:
                self.proc.terminate()
            except Exception:
                pass
            self.proc = None
