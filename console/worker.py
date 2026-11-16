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
                 realtime: bool = False, start_frame: int = 0):
        super().__init__(daemon=True)
        self.source = source
        self.start_frame = int(start_frame)
        self.engine = engine
        self.out_q = out_q
        self.target_fps = target_fps
        self.realtime = realtime
        self._stop_evt = threading.Event()

    def stop(self):
        self._stop_evt.set()

    def run(self):
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            self.out_q.put(("error", f"Cannot open source: {self.source}"))
            return

        frame_delay = 1.0 / self.target_fps if self.target_fps > 0 else 0.0
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        if self.start_frame and isinstance(self.source, str):
            cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
        self.engine.set_source(self.source, src_fps, self.start_frame)
        is_file = isinstance(self.source, str)
        t_start, frame_no = time.perf_counter(), 0

        while not self._stop_evt.is_set():
            t0 = time.perf_counter()
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
            elapsed = time.perf_counter() - t0
            if frame_delay > elapsed:
                time.sleep(frame_delay - elapsed)
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
