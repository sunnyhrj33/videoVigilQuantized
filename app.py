"""
app.py
────────────────────────────────────────────────────────────────────────────────
VIDEO VIGIL – security console (Tkinter)

  Live Monitor  video + people-in-view table + alert feed + live stat tiles
  Pipeline      run Re-ID export → enrollment → open-set training → demo clips
                with progress and streamed output
  Gallery       enrolled employee crops, per recording day
  Evaluation    naive single-frame vs tracklet open-set results (charts)
  Audit Log     live view of output/audit_log.csv
  Live log      colour-coded application log under every tab

All tunable parameters come from config.yaml.
────────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import csv
import json
import logging
import os
import queue
import re
import subprocess
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageTk

from console_tabs import INDEX_RE, SearchTab, SuspectsTab
from detection_engine import DetectionEngine
from system_utils import get_available_trackers, recommend_models

CONFIG_PATH = "config.yaml"
PROJECT_DIR = Path(__file__).resolve().parent
DEMO_MANIFEST = Path("output/demo/demo_clips.json")

from ui_theme import (ACCENT, ACCENT2, BG, BORDER, CARD, CARD2, CYAN, DANGER, FG, FG_DIM,  # noqa: E402
                      FONT, IS_WIN, MONO, PANEL, PURPLE, WARN, ScrollFrame)

REID_MODELS = ["osnet_ain_x1_0", "osnet_x1_0", "osnet_x0_25", "resnet50", "resnet101",
               "fastraid", "osnet_ibn", "transreid", "vitreid"]
MODES = [("tracklet", "Tracklet  (open-set)"), ("naive", "Naive  (single frame)"),
         ("legacy", "Legacy  (runtime IDs)")]

PIPELINE_STEPS = [
    {"key": "export", "title": "Re-ID → OpenVINO", "script": "export_reid_openvino.py",
     "desc": "Convert the MSMT17-trained OSNet-AIN to OpenVINO IR (≈7× faster on CPU)."},
    {"key": "gallery", "title": "Enroll employees", "script": "build_gallery.py",
     "desc": "Each day: clearest crops from every employee's earliest footage → "
             "L2-normalised prototype."},
    {"key": "train", "title": "Train open-set model", "script": "train_open_set.py",
     "desc": "5 tracklet features → logistic regression P(Known). Evaluated on held-out "
             "days vs the naive baseline."},
    {"key": "clips", "title": "Build demo clips", "script": "make_demo_clips.py",
     "desc": "Find & export the 3 evaluation clips (persistence, rejection, asset)."},
    {"key": "index", "title": "Index CCTV for search", "script": "video_search.py",
     "args": ["index", "--day", "2023-06-08", "--day", "2023-07-26"],
     "desc": "Video-wide search index of the demo days: person tracklets + OSNet embeddings "
             "(≈15 min per day)."},
]
PIPE_NOISE = re.compile(r"Cython|warnings\.warn|params:|flops:|^Model:|discarded|"
                        r"Successfully loaded|^\s*$")
PROGRESS_RE = re.compile(r"embedded (\d+) / (\d+)")


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


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s – %(message)s")
logger = logging.getLogger("app")


# ─────────────────────────────────────────────────────────────────────────────
#  Video Worker Thread
# ─────────────────────────────────────────────────────────────────────────────

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
        # realtime: a video file plays at its true speed – frames the CPU cannot
        # keep up with are skipped instead of the video running in slow motion.
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
                # jump to the frame that "should" be showing now (grab() skips decoding)
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
                if self.out_q.qsize() < 3:       # drop frames if the UI is behind
                    self.out_q.put(("frame", (annotated, stats)))
            except Exception as exc:
                logger.exception("Frame processing error")
                self.out_q.put(("error", str(exc)))
                break
            elapsed = time.perf_counter() - t0
            if frame_delay > elapsed:
                time.sleep(frame_delay - elapsed)
        cap.release()


# ─────────────────────────────────────────────────────────────────────────────
#  Small widgets
# ─────────────────────────────────────────────────────────────────────────────

class StatTile(tk.Frame):
    def __init__(self, parent, caption, colour):
        super().__init__(parent, bg=CARD, highlightthickness=1, highlightbackground=BORDER)
        self.value = tk.Label(self, text="—", bg=CARD, fg=colour, font=(FONT, 20, "bold"))
        self.value.pack(anchor="w", padx=12, pady=(6, 0))
        tk.Label(self, text=caption, bg=CARD, fg=FG_DIM, font=(FONT, 8, "bold")).pack(anchor="w", padx=12, pady=(0, 8))

    def set(self, v):
        self.value.config(text=str(v))


class Chip(tk.Label):
    def __init__(self, parent, text="", colour=ACCENT2):
        super().__init__(parent, text=text, bg=CARD2, fg=colour, font=(FONT, 9, "bold"), padx=8, pady=3)


# ─────────────────────────────────────────────────────────────────────────────
#  Main Application
# ─────────────────────────────────────────────────────────────────────────────

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        os.chdir(PROJECT_DIR)
        self.title("VIDEO VIGIL — Open-Set Security Console")
        self.configure(bg=BG)
        self.minsize(1200, 720)
        self.after_idle(self._maximize_window)

        self._log_q: queue.Queue = queue.Queue()
        logging.getLogger().addHandler(QueueLogHandler(self._log_q))

        self.system_spec, _ = recommend_models(prioritize="balanced")
        self.cfg = DetectionEngine.load_config(CONFIG_PATH)
        self.engine = DetectionEngine(self.cfg)
        self._tracker_names = get_available_trackers()

        self._worker: Optional[VideoWorker] = None
        self._frame_q: queue.Queue = queue.Queue(maxsize=5)
        self._running = False
        self._after_id = None
        self._last_event = 0
        self._last_people_update = 0.0
        self._pipe_proc: Optional[subprocess.Popen] = None
        self._pipe_queue: list = []
        self._audit_mtime = 0.0
        self._thumbs: list = []

        self._setup_style()
        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self._poll_logs()
        self._tick()
        self.after(400, self._auto_load_model)

    # ── Style ─────────────────────────────────────────────────────────────────

    def _setup_style(self):
        s = ttk.Style(self)
        s.theme_use("clam")
        s.configure(".", background=PANEL, foreground=FG, font=(FONT, 10))
        s.configure("TNotebook", background=BG, borderwidth=0, tabmargins=(8, 6, 8, 0))
        s.configure("TNotebook.Tab", background=PANEL, foreground=FG_DIM, padding=(16, 7),
                    font=(FONT, 10, "bold"), borderwidth=0)
        s.map("TNotebook.Tab", background=[("selected", CARD2)], foreground=[("selected", ACCENT)])
        s.configure("TCombobox", fieldbackground=CARD, background=CARD2, foreground=FG,
                    arrowcolor=FG, bordercolor=BORDER, lightcolor=CARD, darkcolor=CARD, padding=5)
        s.map("TCombobox", fieldbackground=[("readonly", CARD)], foreground=[("readonly", FG)],
              selectbackground=[("readonly", CARD)], selectforeground=[("readonly", FG)])
        self.option_add("*TCombobox*Listbox.background", CARD)
        self.option_add("*TCombobox*Listbox.foreground", FG)
        self.option_add("*TCombobox*Listbox.selectBackground", ACCENT2)
        self.option_add("*TCombobox*Listbox.font", (FONT, 10))
        s.configure("Treeview", background=CARD, fieldbackground=CARD, foreground=FG,
                    rowheight=24, borderwidth=0, font=(FONT, 9))
        s.configure("Treeview.Heading", background=CARD2, foreground=FG_DIM, relief="flat",
                    font=(FONT, 8, "bold"))
        s.map("Treeview", background=[("selected", BORDER)])
        s.map("Treeview.Heading", background=[("active", BORDER)])
        for name, col in (("Green", ACCENT), ("Purple", PURPLE), ("Blue", ACCENT2)):
            s.configure(f"{name}.Horizontal.TProgressbar", troughcolor=CARD, background=col,
                        bordercolor=CARD, lightcolor=col, darkcolor=col, thickness=6)
        s.configure("Dark.Vertical.TScrollbar", background=CARD2, troughcolor=PANEL,
                    bordercolor=PANEL, arrowcolor=FG_DIM, lightcolor=CARD2, darkcolor=CARD2)
        s.configure("TPanedwindow", background=BG)
        s.configure("Dark.Horizontal.TScale", background=PANEL, troughcolor=CARD)

    # ── UI Construction ────────────────────────────────────────────────────────

    def _maximize_window(self):
        try:
            self.state("zoomed")
        except tk.TclError:
            self.geometry(f"{self.winfo_screenwidth()}x{self.winfo_screenheight()}+0+0")

    def _build_ui(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
        self._build_header()

        paned = ttk.PanedWindow(self, orient="vertical")
        paned.grid(row=1, column=0, sticky="nsew")
        body = tk.Frame(paned, bg=BG)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)
        paned.add(body, weight=5)
        self._build_sidebar(body)

        self.nb = ttk.Notebook(body)
        self.nb.grid(row=0, column=1, sticky="nsew", padx=(0, 6), pady=(4, 0))
        self._tabs = {}
        self._build_live_tab()
        self.suspects_tab = SuspectsTab(self.nb, self)
        self.nb.add(self.suspects_tab, text="  🕵  Suspects  ")
        self._tabs["suspects"] = self.suspects_tab
        self.search_tab = SearchTab(self.nb, self)
        self.nb.add(self.search_tab, text="  🔎  Video Search  ")
        self._tabs["search"] = self.search_tab
        self._build_pipeline_tab()
        self._build_gallery_tab()
        self._build_eval_tab()
        self._build_audit_tab()
        self.nb.bind("<<NotebookTabChanged>>", lambda e: self._on_tab())

        log_frame = tk.Frame(paned, bg=PANEL)
        paned.add(log_frame, weight=1)
        self._build_log_panel(log_frame)

        bar = tk.Frame(self, bg=PANEL)
        bar.grid(row=2, column=0, sticky="ew")
        self._status_var = tk.StringVar(value="Starting …")
        tk.Label(bar, textvariable=self._status_var, bg=PANEL, fg=FG_DIM, font=(FONT, 9),
                 anchor="w", padx=14, pady=4).pack(side="left", fill="x", expand=True)
        spec = self.system_spec
        tk.Label(bar, text=f"CPU {spec.cpu_cores} cores · RAM {spec.total_ram_gb:.0f} GB · "
                           f"GPU {spec.gpu_name if spec.has_gpu else 'none'}",
                 bg=PANEL, fg=FG_DIM, font=(FONT, 9), padx=14).pack(side="right")

    def _build_header(self):
        hdr = tk.Frame(self, bg=PANEL, height=56)
        hdr.grid(row=0, column=0, sticky="ew")
        hdr.grid_propagate(False)
        hdr.columnconfigure(1, weight=1)
        logo = tk.Frame(hdr, bg=PANEL)
        logo.grid(row=0, column=0, sticky="w", padx=16, pady=8)
        tk.Label(logo, text="◆ VIDEO VIGIL", bg=PANEL, fg=ACCENT, font=(FONT, 17, "bold")).pack(side="left")
        tk.Label(logo, text="  open-set security console",
                 bg=PANEL, fg=FG_DIM, font=(FONT, 10)).pack(side="left", pady=(5, 0))

        chips = tk.Frame(hdr, bg=PANEL)
        chips.grid(row=0, column=2, sticky="e", padx=12)
        self._chip_mode = Chip(chips, colour=ACCENT)
        self._chip_model = Chip(chips, colour=ACCENT2)
        self._chip_reid = Chip(chips, colour=PURPLE)
        self._chip_gallery = Chip(chips, colour=CYAN)
        for c in (self._chip_mode, self._chip_model, self._chip_reid, self._chip_gallery):
            c.pack(side="left", padx=4)
        self._clock = tk.Label(chips, bg=PANEL, fg=FG, font=(FONT, 11, "bold"), padx=10)
        self._clock.pack(side="left")
        self._refresh_chips()

    def _section(self, parent, text):
        tk.Label(parent, text=text, bg=PANEL, fg=FG_DIM, font=(FONT, 8, "bold"),
                 anchor="w").pack(fill="x", padx=16, pady=(12, 4))

    def _button(self, parent, text, cmd, colour, fg=BG, **kw):
        return tk.Button(parent, text=text, command=cmd, bg=colour, fg=fg, activebackground=FG,
                         activeforeground=BG, font=(FONT, 9, "bold"), bd=0, relief="flat",
                         padx=10, pady=6, cursor="hand2", disabledforeground="#3d4658", **kw)

    def _combo(self, parent, var, values, width=26):
        cb = ttk.Combobox(parent, textvariable=var, values=values, state="readonly",
                          font=(FONT, 10), width=width)
        return cb

    def _build_sidebar(self, body):
        sf = ScrollFrame(body, bg=PANEL, width=290)
        sf.grid(row=0, column=0, sticky="nsw")
        sf.grid_propagate(False)
        sb = sf.inner

        # Source
        self._section(sb, "VIDEO SOURCE")
        self._source_var = tk.StringVar(value="Webcam (device 0)")
        tk.Entry(sb, textvariable=self._source_var, bg=CARD, fg=FG, insertbackground=ACCENT,
                 font=(FONT, 9), bd=0, relief="flat", highlightthickness=1,
                 highlightbackground=BORDER, highlightcolor=ACCENT).pack(fill="x", padx=16, ipady=5)
        row = tk.Frame(sb, bg=PANEL)
        row.pack(fill="x", padx=16, pady=(6, 0))
        self._button(row, "📷  Webcam", lambda: self._source_var.set("Webcam (device 0)"),
                     CARD2, fg=FG).pack(side="left", fill="x", expand=True, padx=(0, 3))
        self._button(row, "📂  Browse…", self._browse_file, CARD2, fg=FG).pack(
            side="left", fill="x", expand=True, padx=(3, 0))

        self._section(sb, "DEMO CLIPS")
        self._demo_var = tk.StringVar()
        self._demo_combo = self._combo(sb, self._demo_var, [])
        self._demo_combo.pack(fill="x", padx=16)
        self._demo_combo.bind("<<ComboboxSelected>>", lambda e: self._select_demo())
        self._demo_note = tk.Label(sb, text="", bg=PANEL, fg=FG_DIM, font=(FONT, 8), justify="left",
                                   anchor="w", wraplength=250)
        self._demo_note.pack(fill="x", padx=16, pady=(4, 0))
        self._button(sb, "▶  Play smooth render", self._play_render, CARD2, fg=FG).pack(
            fill="x", padx=16, pady=(6, 0))
        self._reload_demo_clips()

        # Models
        self._section(sb, "MODELS")
        model_keys = list(self.cfg.get("models", {}).keys())
        default_model = next((k for k in ("yolo11n_int8_480", "yolo11n_int8") if k in model_keys),
                             model_keys[0] if model_keys else "")
        self._model_var = tk.StringVar(value=default_model)
        self._labelled(sb, "Detector", self._combo(sb, self._model_var, model_keys))
        self._reid_model_var = tk.StringVar(value=self.cfg.get("reid", {}).get("model", REID_MODELS[0]))
        self._labelled(sb, "Re-ID", self._combo(sb, self._reid_model_var, REID_MODELS))
        self._tracker_var = tk.StringVar(value=self.cfg.get("tracking", {}).get("tracker", "bytetrack"))
        self._labelled(sb, "Tracker", self._combo(sb, self._tracker_var, self._tracker_names))
        self._button(sb, "⬡  Load models", self._load_model, ACCENT2).pack(fill="x", padx=16, pady=(8, 0))

        # Identity mode (segmented)
        self._section(sb, "IDENTITY MODE")
        self._mode_var = tk.StringVar(value=self.cfg.get("reid", {}).get("mode", "tracklet"))
        self._mode_buttons = {}
        for key, label in MODES:
            b = tk.Button(sb, text=label, bd=0, relief="flat", font=(FONT, 9, "bold"), anchor="w",
                          padx=12, pady=5, cursor="hand2", command=lambda k=key: self._set_mode(k))
            b.pack(fill="x", padx=16, pady=1)
            self._mode_buttons[key] = b
        self._paint_mode_buttons()

        # Detection tweaks
        self._section(sb, "DETECTION & IDENTITY THRESHOLDS")
        det = self.cfg["detection"]
        self._conf_var = tk.DoubleVar(value=det.get("conf_threshold", 0.35))
        self._person_conf_var = tk.DoubleVar(value=det.get("person_conf_threshold", 0.5))
        self._pknown_var = tk.DoubleVar(value=self.cfg["reid"].get("open_set_p_threshold", 0.5))
        self._iou_var = tk.DoubleVar(value=det.get("iou_threshold", 0.45))
        self._slider(sb, "Detector confidence (all objects)", self._conf_var, 0.05, 1.0,
                     lambda v: setattr(self.engine, "conf", v))
        self._slider(sb, "Person confidence (hides chairs etc.)", self._person_conf_var, 0.05, 1.0,
                     lambda v: setattr(self.engine, "person_conf", v))
        self._slider(sb, "Unknown threshold  P(Known) <", self._pknown_var, 0.05, 0.95,
                     self.engine.set_open_set_threshold)
        self._slider(sb, "IoU threshold", self._iou_var, 0.1, 1.0, lambda v: setattr(self.engine, "iou", v))
        tk.Label(sb, text="Person confidence = how sure YOLO is the box is a person.\n"
                          "Unknown threshold = below this P(Known) a person is UNKNOWN.",
                 bg=PANEL, fg=FG_DIM, font=(FONT, 8), justify="left", anchor="w").pack(fill="x", padx=16)
        toggles = tk.Frame(sb, bg=PANEL)
        toggles.pack(fill="x", padx=12, pady=(4, 0))
        self._trail_var = tk.BooleanVar(value=True)
        self._reid_var = tk.BooleanVar(value=True)
        self._zones_var = tk.BooleanVar(value=self.engine.zones_enabled)
        self._realtime_var = tk.BooleanVar(value=True)
        for i, (text, var, cmd) in enumerate((
                ("Show trails", self._trail_var, lambda: setattr(self.engine, "show_trail", self._trail_var.get())),
                ("Identity / Re-ID", self._reid_var, None),
                ("Restricted zones", self._zones_var, self._toggle_zones),
                ("Real-time playback", self._realtime_var, self._toggle_realtime))):
            tk.Checkbutton(toggles, text=text, variable=var, bg=PANEL, fg=FG, selectcolor=CARD,
                           activebackground=PANEL, activeforeground=ACCENT, font=(FONT, 9),
                           cursor="hand2", command=cmd).grid(row=i // 2, column=i % 2, sticky="w", padx=2)

        # Assets (static anchors)
        self._section(sb, "ASSET MONITORING")
        self._assets_var = tk.BooleanVar(value=self.engine._assets.enabled)
        tk.Checkbutton(sb, text="Monitor laptops / bags (displacement → theft alert)", variable=self._assets_var,
                       bg=PANEL, fg=FG, selectcolor=CARD, activebackground=PANEL, activeforeground=ACCENT,
                       font=(FONT, 9), cursor="hand2",
                       command=lambda: setattr(self.engine._assets, "enabled", self._assets_var.get())
                       ).pack(anchor="w", padx=12)
        arow = tk.Frame(sb, bg=PANEL)
        arow.pack(fill="x", padx=16, pady=(4, 0))
        self._draw_btn = self._button(arow, "✏  Draw laptop anchor", self._begin_anchor, CARD2, fg=FG)
        self._draw_btn.pack(side="left", fill="x", expand=True, padx=(0, 3))
        self._button(arow, "Clear", self._clear_anchors, CARD2, fg=FG).pack(side="left")
        tk.Label(sb, text="Laptops that stay still are anchored automatically; draw a box to fix an "
                          "anchor yourself.", bg=PANEL, fg=FG_DIM, font=(FONT, 8), justify="left",
                 wraplength=250, anchor="w").pack(fill="x", padx=16, pady=(2, 0))

        # Controls
        ctrl = tk.Frame(sb, bg=PANEL)
        ctrl.pack(fill="x", padx=16, pady=(16, 16))
        self._start_btn = self._button(ctrl, "▶  START", self._start, ACCENT)
        self._start_btn.config(font=(FONT, 11, "bold"), pady=9)
        self._start_btn.pack(side="left", fill="x", expand=True, padx=(0, 3))
        self._stop_btn = self._button(ctrl, "■  STOP", self._stop, DANGER, state="disabled")
        self._stop_btn.config(font=(FONT, 11, "bold"), pady=9)
        self._stop_btn.pack(side="left", fill="x", expand=True, padx=(3, 0))

    def _labelled(self, parent, label, widget):
        f = tk.Frame(parent, bg=PANEL)
        f.pack(fill="x", padx=16, pady=2)
        tk.Label(f, text=label, bg=PANEL, fg=FG_DIM, font=(FONT, 9), width=8, anchor="w").pack(side="left")
        widget.pack(in_=f, side="left", fill="x", expand=True)
        widget.lift(f)     # created before its row frame → raise it above the frame

    def _slider(self, parent, label, var, lo, hi, on_change):
        f = tk.Frame(parent, bg=PANEL)
        f.pack(fill="x", padx=16, pady=(2, 0))
        tk.Label(f, text=label, bg=PANEL, fg=FG_DIM, font=(FONT, 9)).pack(side="left")
        val = tk.Label(f, text=f"{var.get():.2f}", bg=PANEL, fg=ACCENT, font=(FONT, 9, "bold"))
        val.pack(side="right")

        def changed(_):
            val.config(text=f"{var.get():.2f}")
            on_change(var.get())

        tk.Scale(parent, from_=lo, to=hi, resolution=0.01, orient="horizontal", variable=var,
                 bg=PANEL, fg=FG, troughcolor=CARD, activebackground=ACCENT, highlightthickness=0,
                 bd=0, showvalue=False, sliderlength=18, width=8, command=changed).pack(fill="x", padx=14)

    # ── Live tab ──────────────────────────────────────────────────────────────

    def _build_live_tab(self):
        tab = tk.Frame(self.nb, bg=BG)
        self.nb.add(tab, text="  ◉  Live Monitor  ")
        self._tabs["live"] = tab
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(tab, bg="#05070b", bd=0, highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew", padx=(6, 6), pady=6)
        self.canvas.bind("<Configure>", lambda e: (not self._running) and self._draw_placeholder())
        self.canvas.bind("<ButtonPress-1>", self._canvas_press)
        self.canvas.bind("<B1-Motion>", self._canvas_motion)
        self.canvas.bind("<ButtonRelease-1>", self._canvas_release)
        self._drawing, self._drag, self._disp, self._preview = False, None, None, None

        side = tk.Frame(tab, bg=BG, width=310)
        side.grid(row=0, column=1, sticky="ns", pady=6, padx=(0, 6))
        side.grid_propagate(False)
        side.columnconfigure((0, 1), weight=1)
        side.rowconfigure(5, weight=1)

        self._tiles = {}
        tiles = tk.Frame(side, bg=BG)
        tiles.grid(row=0, column=0, columnspan=2, rowspan=2, sticky="ew")
        tiles.columnconfigure((0, 1, 2), weight=1)
        for i, (key, cap, col) in enumerate([("fps", "FPS", ACCENT2), ("known", "KNOWN", ACCENT),
                                             ("unknown", "UNKNOWN", WARN), ("alerts", "ALERTS", DANGER),
                                             ("suspects", "SUSPECTS", DANGER), ("tracks", "TRACKS", FG_DIM)]):
            t = StatTile(tiles, cap, col)
            t.grid(row=i // 3, column=i % 3, sticky="ew", padx=2, pady=(0, 4))
            self._tiles[key] = t
        self._tiles["suspects"].set(len(self.engine.suspects))

        self._cam_label = tk.Label(side, text="No source", bg=CARD, fg=FG, font=(FONT, 9),
                                   anchor="w", padx=10, pady=6)
        self._cam_label.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(0, 6))

        tk.Label(side, text="PEOPLE IN VIEW", bg=BG, fg=FG_DIM, font=(FONT, 8, "bold"),
                 anchor="w").grid(row=3, column=0, columnspan=2, sticky="ew")
        cols = ("id", "who", "p", "status")
        self._people = ttk.Treeview(side, columns=cols, show="headings", height=6)
        for c, w, t in zip(cols, (40, 100, 70, 90), ("ID", "IDENTITY", "P(KNOWN)", "STATUS")):
            self._people.heading(c, text=t)
            self._people.column(c, width=w, anchor="w", stretch=c == "who")
        for tag, col in (("KNOWN", ACCENT), ("UNKNOWN", WARN), ("PENDING", FG_DIM), ("INTRUSION", DANGER),
                         ("SUSPECT", DANGER)):
            self._people.tag_configure(tag, foreground=col)
        self._people.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(2, 2))
        acts = tk.Frame(side, bg=BG)
        acts.grid(row=5, column=0, columnspan=2, sticky="ew", pady=(0, 6))
        acts.columnconfigure((0, 1), weight=1)
        self._button(acts, "🚩 Flag suspect", self._flag_selected, CARD2, fg=FG).grid(
            row=0, column=0, sticky="ew", padx=(0, 2))
        self._button(acts, "🔎 Search person", self._search_selected, CARD2, fg=FG).grid(
            row=0, column=1, sticky="ew", padx=(2, 0))

        side.rowconfigure(5, weight=0)
        side.rowconfigure(6, weight=1)
        feed_hdr = tk.Frame(side, bg=BG)
        feed_hdr.grid(row=6, column=0, columnspan=2, sticky="new")
        tk.Label(feed_hdr, text="ALERTS & EVENTS", bg=BG, fg=FG_DIM, font=(FONT, 8, "bold")).pack(side="left")
        self._feed = tk.Text(side, bg=CARD, fg=FG, font=(FONT, 9), bd=0, wrap="word",
                             highlightthickness=0, padx=8, pady=6, state="disabled")
        self._feed.grid(row=6, column=0, columnspan=2, sticky="nsew", pady=(20, 0))
        for tag, col in (("alert", DANGER), ("identity", ACCENT), ("asset", CYAN), ("info", ACCENT2),
                         ("time", FG_DIM), ("unknown", WARN), ("suspect", DANGER)):
            self._feed.tag_configure(tag, foreground=col)
        self._feed.tag_configure("alert", font=(FONT, 9, "bold"))

    # ── Pipeline tab ──────────────────────────────────────────────────────────

    def _build_pipeline_tab(self):
        tab = tk.Frame(self.nb, bg=BG)
        self.nb.add(tab, text="  ⚙  Pipeline  ")
        self._tabs["pipeline"] = tab
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(2, weight=1)

        top = tk.Frame(tab, bg=BG)
        top.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 4))
        tk.Label(top, text="Offline pipeline", bg=BG, fg=FG, font=(FONT, 13, "bold")).pack(side="left")
        tk.Label(top, text="   Re-ID export → daily enrollment → open-set training → demo clips",
                 bg=BG, fg=FG_DIM, font=(FONT, 9)).pack(side="left", pady=(3, 0))
        self._run_all_btn = self._button(top, "▶▶  Run all steps", self._run_all, PURPLE)
        self._run_all_btn.pack(side="right")

        cards = tk.Frame(tab, bg=BG)
        cards.grid(row=1, column=0, sticky="ew", padx=4)
        self._steps = {}
        for i, st in enumerate(PIPELINE_STEPS):
            cards.columnconfigure(i, weight=1, uniform="steps")
            card = tk.Frame(cards, bg=CARD, highlightthickness=1, highlightbackground=BORDER)
            card.grid(row=0, column=i, sticky="nsew", padx=4, pady=4)
            head = tk.Frame(card, bg=CARD)
            head.pack(fill="x", padx=12, pady=(10, 2))
            tk.Label(head, text=f" {i + 1} ", bg=PURPLE, fg=BG, font=(FONT, 10, "bold")).pack(side="left")
            tk.Label(head, text="  " + st["title"], bg=CARD, fg=FG, font=(FONT, 11, "bold")).pack(side="left")
            tk.Label(card, text=st["desc"], bg=CARD, fg=FG_DIM, font=(FONT, 8), justify="left",
                     wraplength=230, anchor="w").pack(fill="x", padx=12, pady=(2, 6))
            status = tk.Label(card, text="", bg=CARD, fg=FG_DIM, font=(FONT, 9, "bold"), anchor="w")
            status.pack(fill="x", padx=12)
            prog = ttk.Progressbar(card, style="Purple.Horizontal.TProgressbar", mode="determinate", maximum=100)
            prog.pack(fill="x", padx=12, pady=(4, 6))
            btn = self._button(card, "Run", lambda k=st["key"]: self._run_steps([k]), CARD2, fg=FG)
            btn.pack(fill="x", padx=12, pady=(0, 12))
            self._steps[st["key"]] = {"status": status, "prog": prog, "btn": btn, **st}
        self._refresh_step_status()

        con = tk.Frame(tab, bg=CARD)
        con.grid(row=2, column=0, sticky="nsew", padx=8, pady=(4, 8))
        con.rowconfigure(1, weight=1)
        con.columnconfigure(0, weight=1)
        tk.Label(con, text="PIPELINE OUTPUT", bg=CARD, fg=FG_DIM, font=(FONT, 8, "bold"),
                 anchor="w").grid(row=0, column=0, sticky="ew", padx=10, pady=(6, 0))
        self._console = tk.Text(con, bg="#0d1119", fg="#c8d0dc", font=(MONO, 9), bd=0,
                                highlightthickness=0, padx=10, pady=6, state="disabled", wrap="none")
        self._console.grid(row=1, column=0, sticky="nsew", padx=6, pady=6)
        vsb = ttk.Scrollbar(con, orient="vertical", command=self._console.yview, style="Dark.Vertical.TScrollbar")
        vsb.grid(row=1, column=1, sticky="ns", pady=6)
        self._console.configure(yscrollcommand=vsb.set)
        for tag, col in (("ok", ACCENT), ("warn", WARN), ("err", DANGER), ("head", PURPLE), ("table", CYAN)):
            self._console.tag_configure(tag, foreground=col)
        self._console.tag_configure("head", font=(MONO, 9, "bold"))

    # ── Gallery tab ───────────────────────────────────────────────────────────

    def _build_gallery_tab(self):
        tab = tk.Frame(self.nb, bg=BG)
        self.nb.add(tab, text="  👥  Gallery  ")
        self._tabs["gallery"] = tab
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(1, weight=1)
        top = tk.Frame(tab, bg=BG)
        top.grid(row=0, column=0, sticky="ew", padx=8, pady=8)
        tk.Label(top, text="Enrolled employees", bg=BG, fg=FG, font=(FONT, 13, "bold")).pack(side="left")
        tk.Label(top, text="   recording day", bg=BG, fg=FG_DIM, font=(FONT, 9)).pack(side="left", padx=(12, 4))
        self._gday_var = tk.StringVar()
        self._gday_combo = self._combo(top, self._gday_var, [], width=14)
        self._gday_combo.pack(side="left")
        self._gday_combo.bind("<<ComboboxSelected>>", lambda e: self._render_gallery())
        self._ginfo = tk.Label(top, text="", bg=BG, fg=FG_DIM, font=(FONT, 9))
        self._ginfo.pack(side="left", padx=12)
        self._gscroll = ScrollFrame(tab, bg=BG)
        self._gscroll.grid(row=1, column=0, sticky="nsew", padx=4)

    # ── Evaluation tab ────────────────────────────────────────────────────────

    def _build_eval_tab(self):
        tab = tk.Frame(self.nb, bg=BG)
        self.nb.add(tab, text="  📊  Evaluation  ")
        self._tabs["eval"] = tab
        self._eval_canvas = tk.Canvas(tab, bg=BG, highlightthickness=0)
        self._eval_canvas.pack(fill="both", expand=True, padx=6, pady=6)
        self._eval_canvas.bind("<Configure>", lambda e: self._render_eval())

    # ── Audit tab ─────────────────────────────────────────────────────────────

    def _build_audit_tab(self):
        tab = tk.Frame(self.nb, bg=BG)
        self.nb.add(tab, text="  🧾  Audit Log  ")
        self._tabs["audit"] = tab
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(1, weight=1)
        top = tk.Frame(tab, bg=BG)
        top.grid(row=0, column=0, sticky="ew", padx=8, pady=8)
        tk.Label(top, text="Forensic audit log", bg=BG, fg=FG, font=(FONT, 13, "bold")).pack(side="left")
        self._audit_count = tk.Label(top, text="", bg=BG, fg=FG_DIM, font=(FONT, 9))
        self._audit_count.pack(side="left", padx=12)
        self._button(top, "Open folder", self._open_output, CARD2, fg=FG).pack(side="right", padx=(4, 0))
        self._button(top, "Open CSV", self._open_audit, CARD2, fg=FG).pack(side="right", padx=4)
        self._audit_filter = tk.StringVar(value="All events")
        cb = self._combo(top, self._audit_filter, ["All events", "Alerts only", "Unknown only", "Known only"], 14)
        cb.pack(side="right", padx=4)
        cb.bind("<<ComboboxSelected>>", lambda e: self._refresh_audit(force=True))

        cols = ("Timestamp", "Camera_ID", "Track_ID", "Identity", "Status", "Confidence",
                "Zone_Alert", "Asset_Interaction")
        frame = tk.Frame(tab, bg=BG)
        frame.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 8))
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
        self._audit = ttk.Treeview(frame, columns=cols, show="headings")
        for c, w in zip(cols, (90, 90, 70, 110, 90, 90, 180, 240)):
            self._audit.heading(c, text=c.replace("_", " ").upper())
            self._audit.column(c, width=w, anchor="w")
        self._audit.grid(row=0, column=0, sticky="nsew")
        vsb = ttk.Scrollbar(frame, orient="vertical", command=self._audit.yview, style="Dark.Vertical.TScrollbar")
        vsb.grid(row=0, column=1, sticky="ns")
        self._audit.configure(yscrollcommand=vsb.set)
        self._audit.tag_configure("alert", foreground=DANGER)
        self._audit.tag_configure("UNKNOWN", foreground=WARN)
        self._audit.tag_configure("KNOWN", foreground=ACCENT)

    # ── Log panel ─────────────────────────────────────────────────────────────

    def _build_log_panel(self, frame):
        frame.rowconfigure(1, weight=1)
        frame.columnconfigure(0, weight=1)
        hdr = tk.Frame(frame, bg=PANEL)
        hdr.grid(row=0, column=0, columnspan=2, sticky="ew")
        tk.Label(hdr, text="●", bg=PANEL, fg=ACCENT, font=(FONT, 9)).pack(side="left", padx=(12, 4))
        tk.Label(hdr, text="LIVE LOG", bg=PANEL, fg=FG_DIM, font=(FONT, 8, "bold")).pack(side="left")
        self._autoscroll = tk.BooleanVar(value=True)
        tk.Checkbutton(hdr, text="auto-scroll", variable=self._autoscroll, bg=PANEL, fg=FG_DIM,
                       selectcolor=CARD, activebackground=PANEL, font=(FONT, 8)).pack(side="right", padx=8)
        tk.Button(hdr, text="clear", command=lambda: self._clear_text(self._log), bg=PANEL, fg=FG_DIM,
                  bd=0, font=(FONT, 8), cursor="hand2", activebackground=PANEL).pack(side="right")
        self._log = tk.Text(frame, bg="#0d1119", fg="#b9c2cf", font=(MONO, 9), bd=0, height=8,
                            highlightthickness=0, padx=10, pady=4, state="disabled", wrap="none")
        self._log.grid(row=1, column=0, sticky="nsew", padx=(6, 0), pady=(0, 4))
        vsb = ttk.Scrollbar(frame, orient="vertical", command=self._log.yview, style="Dark.Vertical.TScrollbar")
        vsb.grid(row=1, column=1, sticky="ns", pady=(0, 4))
        self._log.configure(yscrollcommand=vsb.set)
        for tag, col in (("time", FG_DIM), ("INFO", "#b9c2cf"), ("WARNING", WARN), ("ERROR", DANGER),
                         ("ALERT", DANGER), ("src", ACCENT2), ("PIPE", PURPLE)):
            self._log.tag_configure(tag, foreground=col)
        self._log.tag_configure("ALERT", font=(MONO, 9, "bold"))

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _append(widget: tk.Text, chunks, max_lines: int = 2000, autoscroll: bool = True, top=False):
        """chunks: [(text, tag)]."""
        widget.configure(state="normal")
        if top:
            for text, tag in reversed(chunks):
                widget.insert("1.0", text, tag)
        else:
            for text, tag in chunks:
                widget.insert("end", text, tag)
        lines = int(widget.index("end-1c").split(".")[0])
        if lines > max_lines:
            if top:
                widget.delete(f"{max_lines}.0", "end")
            else:
                widget.delete("1.0", f"{lines - max_lines}.0")
        widget.configure(state="disabled")
        if autoscroll and not top:
            widget.see("end")

    @staticmethod
    def _clear_text(widget: tk.Text):
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.configure(state="disabled")

    def _set_status(self, msg: str):
        self._status_var.set(msg)

    def _refresh_chips(self):
        mode = self._mode_var.get() if hasattr(self, "_mode_var") else self.cfg["reid"].get("mode", "tracklet")
        self._chip_mode.config(text=f"● {mode.upper()}")
        key = getattr(self.engine, "_model_key", None) if self.engine._model is not None else None
        self._chip_model.config(text=key or "no detector")
        ext = self.engine._reid_extractor
        backend = "OpenVINO" if ext is not None and ext._ov_model is not None else "PyTorch" if ext else "—"
        self._chip_reid.config(text=f"{self.engine._reid_model_name or '—'} · {backend}")
        self._chip_gallery.config(text=f"👥 {self.engine.gallery_info}")

    TAB_ORDER = ["live", "suspects", "search", "pipeline", "gallery", "eval", "audit"]

    def select_tab(self, name: str):
        self.nb.select(self._tabs[name])

    def current_tab(self) -> str:
        cur = self.nb.select()
        return next((k for k, t in self._tabs.items() if str(t) == cur), "")

    def _tick(self):
        self._clock.config(text=time.strftime("%H:%M:%S"))
        tab = self.current_tab()
        if tab == "audit":
            self._refresh_audit()
        sig = self.suspects_tab.signature()
        if sig != getattr(self, "_suspect_sig", None):
            self._suspect_sig = sig
            self.search_tab.refresh_queries()
            self._tiles["suspects"].set(len(self.engine.suspects))
            if tab == "suspects":
                self.suspects_tab.render()
        self.after(1000, self._tick)

    def _on_tab(self):
        tab = self.current_tab()
        if tab == "gallery":
            self._load_gallery_days()
        elif tab == "eval":
            self._render_eval()
        elif tab == "audit":
            self._refresh_audit(force=True)
        elif tab == "suspects":
            self.suspects_tab.render()
        elif tab == "search":
            self.search_tab.refresh_days()

    def _draw_placeholder(self):
        c = self.canvas
        c.delete("all")
        w, h = c.winfo_width() or 800, c.winfo_height() or 500
        c.create_text(w // 2, h // 2 - 40, text="◆", fill=BORDER, font=(FONT, 48))
        c.create_text(w // 2, h // 2 + 20, text="No video", fill=FG_DIM, font=(FONT, 14, "bold"))
        c.create_text(w // 2, h // 2 + 50, fill=FG_DIM, font=(FONT, 10),
                      text="Pick a demo clip or source on the left and press  ▶ START")

    # ── Actions ───────────────────────────────────────────────────────────────

    def _auto_load_model(self):
        self._draw_placeholder()
        if self._model_var.get():
            self._load_model(quiet=True)

    def _load_model(self, quiet: bool = False):
        if self._running:
            self._stop()
        key, reid_model = self._model_var.get(), self._reid_model_var.get()
        self._set_status(f"Loading {key} + Re-ID {reid_model} …")
        self.update_idletasks()
        try:
            self.engine.set_reid_model(reid_model)
            self.engine.load_model(key)
            self.engine.set_tracker(self._tracker_var.get())
            ident = ("enrolled gallery (open-set)" if self.engine._tracklets is not None
                     else "runtime IDs – no enrolled gallery for this Re-ID model")
            self._set_status(f"✓ {key} + {reid_model} loaded · identity: {ident} – press ▶ START")
        except Exception as exc:
            if not quiet:
                messagebox.showerror("Model Error", f"Failed to load model:\n\n{exc}")
            self._set_status(f"Model load failed: {exc}")
        self._refresh_chips()

    def _set_mode(self, mode: str):
        self._mode_var.set(mode)
        self.engine.set_identity_mode(mode)
        self._paint_mode_buttons()
        self._refresh_chips()
        self._set_status(f"Identity mode → {mode}")

    def _paint_mode_buttons(self):
        for key, b in self._mode_buttons.items():
            on = key == self._mode_var.get()
            b.config(bg=ACCENT if on else CARD, fg=BG if on else FG, activebackground=ACCENT2)

    def _toggle_zones(self):
        self.engine.zones_enabled = self._zones_var.get()
        self._set_status(f"Restricted zones {'ON' if self.engine.zones_enabled else 'OFF'}")
        logger.info("Restricted zones %s", "enabled" if self.engine.zones_enabled else "disabled")

    def _toggle_realtime(self):
        if self._worker is not None:
            self._worker.realtime = self._realtime_var.get()
        self._set_status("Real-time playback ON – frames the CPU can't keep up with are skipped"
                         if self._realtime_var.get() else
                         "Real-time playback OFF – every frame is analysed (video runs slower)")

    def _browse_file(self):
        path = filedialog.askopenfilename(title="Select video", filetypes=[
            ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv"), ("All files", "*.*")])
        if path:
            self._source_var.set(path)

    def _reload_demo_clips(self):
        self._demo_items = {}
        if DEMO_MANIFEST.exists():
            try:
                manifest = json.loads(DEMO_MANIFEST.read_text())
            except Exception:
                manifest = {}
            titles = {"demo1": "Demo 1 · Known employee", "demo2": "Demo 2 · Unknown visitor",
                      "demo3": "Demo 3 · Asset interaction"}
            for k in sorted(manifest):
                m = manifest[k]
                label = f"{titles.get(k, k)} ({m.get('focus', '')})"
                self._demo_items[label] = (k, m)
        self._demo_combo.config(values=list(self._demo_items) or ["(run Pipeline step 4)"])
        if self._demo_var.get() not in self._demo_items:
            self._demo_var.set("— choose a demo clip —" if self._demo_items else "(run Pipeline step 4)")

    def _select_demo(self):
        item = self._demo_items.get(self._demo_var.get())
        if not item:
            return
        key, m = item
        self._source_var.set(str(DEMO_MANIFEST.parent / m["file"]))
        tips = {"demo1": "Watch the employee keep a stable green identity through the occlusion.",
                "demo2": "Run once in NAIVE mode (visitor gets an employee name), then TRACKLET "
                         "(visitor becomes UNKNOWN).",
                "demo3": "Unknown person at a laptop → asset / zone alert + audit log entry."}
        people = ", ".join(f"emp {p}" for p in m.get("employees", [])) or "—"
        visitors = ", ".join(str(v) for v in m.get("visitors", [])) or "—"
        self._demo_note.config(text=f"{m.get('source', '')} · {m.get('date', '')}\n"
                                    f"In view: {people} · visitors {visitors}\n{tips.get(key, '')}")

    def _play_render(self):
        """Open the offline-rendered (smooth) MP4 of the selected demo in the system player."""
        item = self._demo_items.get(self._demo_var.get())
        if not item:
            messagebox.showinfo("Pick a demo", "Choose a demo clip first.")
            return
        stem = item[1]["file"].split("__")[0]
        renders = sorted(Path("output/rendered").glob(f"{stem}__*.mp4"))
        mode = self._mode_var.get()
        pick = next((r for r in renders if r.stem.endswith(mode)), renders[0] if renders else None)
        if pick is None:
            messagebox.showinfo("Not rendered", f"Run:  python render_demo.py {item[0]} --mode {mode}")
            return
        logger.info("Playing rendered clip %s", pick)
        if IS_WIN:
            os.startfile(pick.resolve())

    def _resolve_source(self):
        raw = self._source_var.get().strip()
        if raw.lower().startswith("webcam") or raw == "0":
            return 0
        try:
            return int(raw)
        except ValueError:
            return raw

    def play_from(self, path: str, frame: int = 0):
        """Open a recording at a given frame in the Live Monitor (search results, suspect origin)."""
        if self._running:
            self._stop()
        self._source_var.set(path)
        self._start(start_frame=frame)

    def find_video(self, name: str) -> Optional[Path]:
        """Locate a recording by file name (demo clips, renders or the CHIRLA videos)."""
        p = Path(name)
        if p.exists():
            return p
        norm = name.replace("\uf03a", ":")
        for root in (Path("output/demo"), Path("output/rendered"),
                     Path(self.cfg.get("chirla", {}).get("root", "")) / "videos"):
            if root.exists():
                for f in root.rglob("*"):
                    if f.name.replace("\uf03a", ":") == norm:
                        return f
        return None

    def _start(self, start_frame: int = 0):
        if self.engine._model is None:
            messagebox.showwarning("No model", "Load the models first.")
            return
        if self._running:
            return
        self.engine._reid_enabled = self._reid_var.get()
        self.engine.show_trail = self._trail_var.get()
        if self.engine._reid_model_name != self._reid_model_var.get():
            self.engine.set_reid_model(self._reid_model_var.get())
        self.engine.set_tracker(self._tracker_var.get())
        self.engine.set_identity_mode(self._mode_var.get())

        source = self._resolve_source()
        self._frame_q = queue.Queue(maxsize=5)
        self._worker = VideoWorker(source, self.engine, self._frame_q,
                                   self.cfg.get("video", {}).get("target_fps", 30),
                                   realtime=self._realtime_var.get(), start_frame=start_frame)
        self._worker.start()
        self._running = True
        self._clear_text(self._feed)
        self._start_btn.config(state="disabled")
        self._stop_btn.config(state="normal")
        self.select_tab("live")
        self._set_status(f"Running · {source}")
        self._poll_frames()

    def _stop(self):
        self._running = False
        worker = self._worker
        if worker:
            worker.stop()
            worker.join(timeout=2.0)
        self._worker = None
        if self._after_id:
            self.after_cancel(self._after_id)
            self._after_id = None
        self._start_btn.config(state="normal")
        self._stop_btn.config(state="disabled")
        self._set_status("Stopped.")
        self._draw_placeholder()

    def _poll_frames(self):
        if not self._running:
            return
        try:
            while True:
                kind, payload = self._frame_q.get_nowait()
                if kind == "frame":
                    frame, stats = payload
                    self._display_frame(frame)
                    self._update_stats(stats)
                elif kind == "done":
                    self._drain_events()
                    self._stop()
                    self._set_status("Video ended.")
                    return
                elif kind == "error":
                    messagebox.showerror("Error", payload)
                    self._stop()
                    return
        except queue.Empty:
            pass
        self._drain_events()
        self._after_id = self.after(15, self._poll_frames)

    def _display_frame(self, bgr: np.ndarray):
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        if cw < 2 or ch < 2:
            return
        h, w = bgr.shape[:2]
        scale = min(cw / w, ch / h)
        nw, nh = int(w * scale), int(h * scale)
        rgb = cv2.cvtColor(cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
        img = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.canvas.delete("all")
        self.canvas.create_image((cw - nw) // 2, (ch - nh) // 2, anchor="nw", image=img)
        self.canvas._img_ref = img
        self._disp = ((cw - nw) // 2, (ch - nh) // 2, scale, w, h)
        if self._drag is not None:
            x0, y0, x1, y1 = self._drag
            self.canvas.create_rectangle(x0, y0, x1, y1, outline=CYAN, width=2, dash=(6, 4))
        elif self._drawing:
            self.canvas.create_text(cw // 2, 24, text="Drag a box around the laptop on its desk",
                                    fill=CYAN, font=(FONT, 12, "bold"))

    # ── asset anchor drawing ──────────────────────────────────────────────────

    def _begin_anchor(self):
        """Next mouse drag on the video defines a laptop home anchor for this camera."""
        if not self._running:
            source = self._resolve_source()
            cap = cv2.VideoCapture(source)
            ok, frame = cap.read()
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            cap.release()
            if not ok:
                messagebox.showinfo("Draw anchor", "Pick a video source (or start playback) first.")
                return
            self.engine.set_source(source, fps)
            self._preview = frame
        self._drawing, self._drag = True, None
        self._draw_btn.config(bg=CYAN, fg=BG)
        self._set_status("Draw mode: drag a box around the laptop in the video")
        if not self._running:
            self._display_frame(self._preview)

    def _canvas_press(self, e):
        if self._drawing:
            self._drag = [e.x, e.y, e.x, e.y]

    def _canvas_motion(self, e):
        if self._drawing and self._drag is not None:
            self._drag[2:] = [e.x, e.y]
            if not self._running and self._preview is not None:
                self._display_frame(self._preview)

    def _canvas_release(self, e):
        if not self._drawing or self._drag is None or self._disp is None:
            return
        ox, oy, scale, w, h = self._disp
        x0, y0, x1, y1 = self._drag
        box = [int((min(x0, x1) - ox) / scale), int((min(y0, y1) - oy) / scale),
               int((max(x0, x1) - ox) / scale), int((max(y0, y1) - oy) / scale)]
        box = [max(0, min(w, box[0])), max(0, min(h, box[1])), max(0, min(w, box[2])), max(0, min(h, box[3]))]
        self._drawing, self._drag = False, None
        self._draw_btn.config(bg=CARD2, fg=FG)
        if box[2] - box[0] < 8 or box[3] - box[1] < 8:
            self._set_status("Anchor too small – try again")
            return
        name = self.engine.add_asset_anchor(box)
        self._set_status(f"✓ Anchor '{name}' saved for {self.engine._cam_key} at {box}")
        if not self._running and self._preview is not None:
            self._display_frame(self._preview)

    def _clear_anchors(self):
        self.engine.clear_asset_anchors()
        self._set_status(f"Drawn anchors cleared for {self.engine._cam_key or 'this camera'}")

    # ── people actions ────────────────────────────────────────────────────────

    def _selected_tid(self) -> Optional[int]:
        sel = self._people.selection()
        if not sel:
            messagebox.showinfo("Select a person", "Select a person in the PEOPLE IN VIEW table first.")
            return None
        return int(str(self._people.item(sel[0], "values")[0]).lstrip("T"))

    def _flag_selected(self):
        tid = self._selected_tid()
        if tid is None:
            return
        s = self.engine.flag_suspect(tid)
        if s is None:
            messagebox.showinfo("Flag suspect", f"T{tid} has no sharp crops yet – try again in a moment.")
            return
        self._set_status(f"🚩 T{tid} flagged as {s.label}")
        self.suspects_tab.render(force=True)

    def _search_selected(self):
        tid = self._selected_tid()
        if tid is None:
            return
        embs, thumbs = self.engine.track_embeddings(tid)
        if len(embs) == 0:
            messagebox.showinfo("Search person", f"T{tid} has no sharp crops yet – try again in a moment.")
            return
        self.search_tab.query_live(tid, embs, thumbs, self.engine._camera_id)

    def _update_stats(self, stats: dict):
        self._tiles["fps"].set(f"{stats.get('fps', 0):.1f}")
        for k in ("known", "unknown", "alerts"):
            self._tiles[k].set(stats.get(k, "—"))
        self._tiles["tracks"].set(stats.get("tracks", "—"))
        if stats.get("camera"):
            assets = ", ".join(f"{n}: {s}" for n, s in stats.get("assets", [])) or "no assets in view"
            self._cam_label.config(text=f"📹 {stats['camera']}   🕒 {stats.get('time', '')}   ·   {assets}")

        now = time.time()
        if now - self._last_people_update < 0.25:
            return
        self._last_people_update = now
        sel = self._people.selection()
        self._keep_sel = int(str(self._people.item(sel[0], "values")[0]).lstrip("T")) if sel else None
        self._people.delete(*self._people.get_children())
        for p in sorted(stats.get("people", []), key=lambda p: p["tid"]):
            status = p["status"]
            tag = "INTRUSION" if p["intrusion"] else status
            shown = "PENDING" if status == "PENDING" else status
            if p.get("carrying"):
                shown += " 💼"
            if p["zone"]:
                shown += " ⚠" if p["intrusion"] else " ·zone"
            pk = f"{p['confidence']:.2f}" if status != "PENDING" else f"{p['crops']} crops"
            iid = self._people.insert("", "end", values=(f"T{p['tid']}", p["name"], pk, shown), tags=(tag,))
            if p["tid"] == self._keep_sel:
                self._people.selection_set(iid)

    def _drain_events(self):
        for no, level, stamp, cam, text in self.engine.events_since(self._last_event):
            self._last_event = no
            if level == "info":
                tag = "info"
            elif level == "identity" and "UNKNOWN" in text:
                tag = "unknown"
            else:
                tag = level
            icon = {"alert": "🚨 ", "identity": "👤 ", "asset": "💼 ", "info": "ℹ ", "suspect": "🕵 "}.get(level, "")
            self._append(self._feed, [(f"{stamp} {cam}\n", "time"), (f"{icon}{text}\n\n", tag)],
                         max_lines=400, top=True)

    # ── Logs ──────────────────────────────────────────────────────────────────

    def _poll_logs(self):
        chunks = []
        try:
            for _ in range(200):
                item = self._log_q.get_nowait()
                if item[0] == "log":
                    _, levelno, name, msg, created = item
                    level = "ALERT" if "ALERT" in msg else logging.getLevelName(levelno)
                    chunks += [(time.strftime("%H:%M:%S ", time.localtime(created)), "time"),
                               (f"{name[:16]:<16} ", "src"),
                               (msg + "\n", level if level in ("WARNING", "ERROR", "ALERT") else "INFO")]
                elif item[0] == "pipe":
                    self._handle_pipe_line(item[1], item[2])
                elif item[0] == "pipe_done":
                    self._on_step_done(item[1], item[2], item[3])
        except queue.Empty:
            pass
        if chunks:
            self._append(self._log, chunks, autoscroll=self._autoscroll.get())
        self.after(100, self._poll_logs)

    # ── Pipeline ──────────────────────────────────────────────────────────────

    def _refresh_step_status(self):
        reid = self.cfg["reid"]
        model = reid["model"]
        checks = {
            "export": Path("models/reid") / f"{model}_openvino" / f"{model}.xml",
            "gallery": Path(reid["gallery_file"]),
            "train": Path(reid["open_set_model"]),
            "clips": DEMO_MANIFEST,
            "index": Path(self.cfg.get("video_search", {}).get("index_dir", "output/index")),
        }
        for key, st in self._steps.items():
            p = checks[key]
            if p.exists():
                when = time.strftime("%d %b %H:%M", time.localtime(p.stat().st_mtime))
                st["status"].config(text=f"✓ ready · {when}", fg=ACCENT)
                st["prog"]["value"] = 100
            else:
                st["status"].config(text="○ not run yet", fg=FG_DIM)
                st["prog"]["value"] = 0

    def _run_all(self):
        self._run_steps([s["key"] for s in PIPELINE_STEPS])

    def _run_steps(self, keys):
        if self._pipe_proc is not None:
            messagebox.showinfo("Pipeline busy", "A pipeline step is already running.")
            return
        if self._running:
            self._stop()
        self._pipe_queue = list(keys)
        self.select_tab("pipeline")
        self._clear_text(self._console)
        self._set_pipeline_buttons("disabled")
        self._next_step()

    def _set_pipeline_buttons(self, state):
        self._run_all_btn.config(state=state)
        for st in self._steps.values():
            st["btn"].config(state=state)

    def _next_step(self):
        if not self._pipe_queue:
            self._set_pipeline_buttons("normal")
            self._set_status("✓ Pipeline finished")
            return
        key = self._pipe_queue.pop(0)
        st = self._steps[key]
        st["status"].config(text="⟳ running …", fg=PURPLE)
        st["prog"].config(mode="indeterminate")
        st["prog"].start(12)
        st["t0"] = time.time()
        args = st.get("args", [])
        self._append(self._console, [(f"\n━━━ {st['title']}  ·  python {st['script']} {' '.join(args)} ━━━\n",
                                      "head")])
        logger.info("Pipeline step started: %s", st["title"])
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
        flags = subprocess.CREATE_NO_WINDOW if IS_WIN else 0
        self._pipe_proc = subprocess.Popen(
            [sys.executable, "-u", st["script"], *args], cwd=PROJECT_DIR, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            encoding="utf-8", errors="replace", creationflags=flags)
        threading.Thread(target=self._read_pipe, args=(key, self._pipe_proc), daemon=True).start()

    def _read_pipe(self, key, proc):
        for line in proc.stdout:
            self._log_q.put(("pipe", key, line.rstrip("\n")))
        rc = proc.wait()
        self._log_q.put(("pipe_done", key, rc, time.time()))

    def _handle_pipe_line(self, key, line):
        if PIPE_NOISE.search(line):
            return
        st = self._steps[key]
        mi = INDEX_RE.search(line)
        if mi:
            k, n, f, total = (int(x) for x in mi.groups())
            if str(st["prog"].cget("mode")) != "determinate":
                st["prog"].stop()
                st["prog"].config(mode="determinate")
            st["prog"]["value"] = 100 * ((k - 1) + f / max(total, 1)) / n
            st["status"].config(text=f"⟳ video {k}/{n} · frame {f:,}/{total:,}", fg=PURPLE)
            return
        m = PROGRESS_RE.search(line)
        if m:
            done, total = int(m.group(1)), int(m.group(2))
            if str(st["prog"].cget("mode")) != "determinate":
                st["prog"].stop()
                st["prog"].config(mode="determinate")
            st["prog"]["value"] = 100 * done / max(total, 1)
            st["status"].config(text=f"⟳ embedding {done:,} / {total:,}", fg=PURPLE)
        msg = re.sub(r"^\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d+ ", "", line)
        tag = ("err" if re.search(r"Error|Traceback|ERROR", line) else "warn" if "WARN" in line
               else "table" if msg.startswith(("|", "#", "-", "TAR")) else "ok" if "✓" in line else None)
        self._append(self._console, [(msg + "\n", tag)])

    def _on_step_done(self, key, rc, t_end):
        st = self._steps[key]
        st["prog"].stop()
        st["prog"].config(mode="determinate")
        self._pipe_proc = None
        secs = t_end - st.get("t0", t_end)
        if rc == 0:
            st["prog"]["value"] = 100
            st["status"].config(text=f"✓ done in {secs:.0f}s", fg=ACCENT)
            logger.info("Pipeline step finished: %s (%.0fs)", st["title"], secs)
            self._after_step(key)
            self._next_step()
        else:
            st["prog"]["value"] = 0
            st["status"].config(text=f"✗ failed (exit {rc})", fg=DANGER)
            logger.error("Pipeline step failed: %s (exit %d)", st["title"], rc)
            self._pipe_queue = []
            self._set_pipeline_buttons("normal")

    def _after_step(self, key):
        """Pick up new artefacts in the running app."""
        if key in ("export", "gallery", "train"):
            self.engine._reid_extractor_cache.clear()
            self.engine._reid_model_name = None
            self.engine.set_reid_model(self._reid_model_var.get())
            self._refresh_chips()
        if key == "clips":
            self._reload_demo_clips()
        if key == "index":
            self.search_tab.refresh_days()

    # ── Gallery rendering ─────────────────────────────────────────────────────

    def _load_gallery_days(self):
        root = Path(self.cfg["reid"]["gallery_file"]).parent
        days = sorted(p.name for p in root.iterdir() if p.is_dir() and (p / "gallery_prototypes.pkl").exists()) \
            if root.exists() else []
        self._gday_combo.config(values=days)
        if days and self._gday_var.get() not in days:
            demo = self.cfg.get("chirla", {}).get("demo_day")
            self._gday_var.set(demo if demo in days else days[0])
        self._render_gallery()

    def _render_gallery(self):
        for w in self._gscroll.inner.winfo_children():
            w.destroy()
        self._thumbs = []
        day = self._gday_var.get()
        if not day:
            tk.Label(self._gscroll.inner, text="No gallery yet – run Pipeline step 2.", bg=BG, fg=FG_DIM,
                     font=(FONT, 11)).pack(pady=40)
            return
        import pickle
        gdir = Path(self.cfg["reid"]["gallery_file"]).parent / day
        try:
            meta = pickle.loads((gdir / "gallery_prototypes.pkl").read_bytes()).get("meta", {})
        except Exception:
            meta = {}
        emps = sorted(p for p in gdir.iterdir() if p.is_dir())
        self._ginfo.config(text=f"{len(emps)} employees enrolled · model {meta.get('model', '?')} · "
                                f"crops from each person's earliest footage that day")
        grid = self._gscroll.inner
        ncol = 3
        for c in range(ncol):
            grid.columnconfigure(c, weight=1, uniform="g")
        for i, d in enumerate(emps):
            card = tk.Frame(grid, bg=CARD, highlightthickness=1, highlightbackground=BORDER)
            card.grid(row=i // ncol, column=i % ncol, sticky="nsew", padx=5, pady=5)
            head = tk.Frame(card, bg=CARD)
            head.pack(fill="x", padx=10, pady=(8, 4))
            tk.Label(head, text=d.name.replace("emp_", "Emp "), bg=CARD, fg=ACCENT,
                     font=(FONT, 11, "bold")).pack(side="left")
            cut = meta.get("enroll_cutoff", {}).get(d.name)
            if cut:
                tk.Label(head, text=f"enrolled until {cut[0]} f{cut[1]}", bg=CARD, fg=FG_DIM,
                         font=(FONT, 8)).pack(side="right")
            row = tk.Frame(card, bg=CARD)
            row.pack(padx=10, pady=(0, 10), anchor="w")
            for f in sorted(d.glob("*.jpg")):
                try:
                    img = Image.open(f).convert("RGB").resize((54, 108))
                except Exception:
                    continue
                ph = ImageTk.PhotoImage(img)
                self._thumbs.append(ph)
                tk.Label(row, image=ph, bg=CARD, bd=0).pack(side="left", padx=2)

    # ── Evaluation rendering ──────────────────────────────────────────────────

    def _render_eval(self):
        c = self._eval_canvas
        c.delete("all")
        W, H = max(c.winfo_width(), 600), max(c.winfo_height(), 400)
        path = Path(self.cfg["reid"]["open_set_model"])
        try:
            data = json.loads(path.read_text())
            m = data["metrics"]
        except Exception:
            c.create_text(W // 2, H // 2, text="No evaluation yet – run Pipeline step 3.",
                          fill=FG_DIM, font=(FONT, 12))
            return

        c.create_text(16, 18, anchor="w", text="Open-set evaluation on held-out days",
                      fill=FG, font=(FONT, 14, "bold"))
        days = ", ".join(m.get("test_days", []))
        c.create_text(16, 42, anchor="w", fill=FG_DIM, font=(FONT, 9),
                      text=f"Test days {days} · visitors never seen in training · same-day enrollment")
        try:
            se = json.loads(Path("output/search_eval.json").read_text())
            ex = se.get("exclude_employees") or se.get("plain")
            c.create_text(W - 16, 18, anchor="e", fill=CYAN, font=(FONT, 10, "bold"),
                          text=f"Video-wide search ({ex['queries']} visitor queries, CHIRLA GT): "
                               f"top-1 {ex['rank1']:.0%} · top-5 {ex['top5']:.0%} · mAP {ex['mAP']:.0%}")
            c.create_text(W - 16, 42, anchor="e", fill=FG_DIM, font=(FONT, 9),
                          text=f"days {', '.join(se.get('days', []))} · employees excluded · "
                               f"python video_search.py eval")
        except Exception:
            pass

        methods = [("Naive\nclosed-set", m.get("naive_closed_set"), "#5b6577"),
                   (f"Naive\ncos ≥ {m.get('naive_thresh_value', 0.45)}", m.get("naive_threshold"), "#7b8599"),
                   ("Tracklet\nrule-based", m.get("tracklet_rule"), ACCENT2),
                   ("Tracklet\n+ log. regression", m.get("test_tracklet_lr"), ACCENT)]
        methods = [x for x in methods if x[1]]
        panels = [("TAR", "Employee → correct ID", True), ("MisID", "Employee → wrong ID", False),
                  ("FAR", "Visitor → passed as employee", False), ("TRR", "Visitor → rejected UNKNOWN", True)]

        top, bottom = 70, H - 150
        pw = (W - 32) / len(panels)
        for i, (key, title, higher) in enumerate(panels):
            x0 = 16 + i * pw
            c.create_rectangle(x0 + 4, top, x0 + pw - 4, bottom + 46, fill=CARD, outline=BORDER)
            c.create_text(x0 + 16, top + 16, anchor="w", text=title, fill=FG, font=(FONT, 10, "bold"))
            c.create_text(x0 + pw - 16, top + 16, anchor="e", fill=ACCENT if higher else WARN,
                          font=(FONT, 9, "bold"), text="higher is better ↑" if higher else "lower is better ↓")
            chart_top, chart_bot = top + 44, bottom
            bw = (pw - 40) / max(len(methods), 1)
            for j, (name, mm, col) in enumerate(methods):
                v = float(mm.get(key, 0) or 0)
                bx0 = x0 + 20 + j * bw + bw * 0.15
                bx1 = x0 + 20 + (j + 1) * bw - bw * 0.15
                by = chart_bot - (chart_bot - chart_top - 20) * v
                c.create_rectangle(bx0, chart_top + 20, bx1, chart_bot, fill=CARD2, outline="")
                c.create_rectangle(bx0, by, bx1, chart_bot, fill=col, outline="")
                c.create_text((bx0 + bx1) / 2, by - 9, text=f"{v:.0%}", fill=FG, font=(FONT, 9, "bold"))
                c.create_text((bx0 + bx1) / 2, chart_bot + 20, text=name, fill=FG_DIM,
                              font=(FONT, 7), justify="center")

        # Summary row
        y = H - 92
        lr = m.get("test_tracklet_lr", {})
        cards = [("ROC-AUC (tracklet LR)", f"{m.get('test_auc', 0):.3f}", ACCENT),
                 ("Test tracklets", f"{lr.get('n_employee', 0)} emp · {lr.get('n_visitor', 0)} visitor", ACCENT2),
                 ("Visitor rejection", f"{(m.get('naive_threshold') or {}).get('TRR', 0):.0%} → "
                                       f"{lr.get('TRR', 0):.0%}", WARN),
                 ("Wrong-ID rate", f"{(m.get('naive_closed_set') or {}).get('MisID', 0):.0%} → "
                                   f"{lr.get('MisID', 0):.0%}", DANGER)]
        cw = (W - 32) / len(cards)
        for i, (cap, val, col) in enumerate(cards):
            x0 = 16 + i * cw
            c.create_rectangle(x0 + 4, y, x0 + cw - 4, y + 72, fill=CARD, outline=BORDER)
            c.create_text(x0 + 18, y + 22, anchor="w", text=cap.upper(), fill=FG_DIM, font=(FONT, 8, "bold"))
            c.create_text(x0 + 18, y + 48, anchor="w", text=val, fill=col, font=(FONT, 15, "bold"))

    # ── Audit log ─────────────────────────────────────────────────────────────

    def _refresh_audit(self, force: bool = False):
        path = Path(self.engine._audit_path)
        if not path.exists():
            self._audit_count.config(text="no events yet")
            self._audit.delete(*self._audit.get_children())
            return
        mtime = path.stat().st_mtime
        if not force and mtime == self._audit_mtime:
            return
        self._audit_mtime = mtime
        try:
            with open(path, newline="", encoding="utf-8") as f:
                rows = list(csv.reader(f))[1:]
        except Exception:
            return
        flt = self._audit_filter.get()
        shown = []
        for r in reversed(rows[-2000:]):
            if len(r) < 8:
                continue
            alert = r[6] != "None" or r[7] != "None"
            if (flt == "Alerts only" and not alert) or (flt == "Unknown only" and r[4] != "UNKNOWN") \
                    or (flt == "Known only" and r[4] != "KNOWN"):
                continue
            shown.append((r, "alert" if alert else r[4]))
        self._audit.delete(*self._audit.get_children())
        for r, tag in shown[:1000]:
            self._audit.insert("", "end", values=r[:8], tags=(tag,))
        n_alerts = sum(1 for r in rows if len(r) >= 8 and (r[6] != "None" or r[7] != "None"))
        self._audit_count.config(text=f"{len(rows):,} events · {n_alerts:,} alerts · {path}")

    def _open_audit(self):
        p = Path(self.engine._audit_path)
        if p.exists() and IS_WIN:
            os.startfile(p)

    def _open_output(self):
        p = Path(self.engine._audit_path).parent
        p.mkdir(parents=True, exist_ok=True)
        if IS_WIN:
            os.startfile(p)

    def _on_close(self):
        self._stop()
        if self._pipe_proc is not None:
            self._pipe_proc.terminate()
        self.engine.close()
        self.destroy()


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args():
    import argparse
    ap = argparse.ArgumentParser(description="VIDEO VIGIL security console")
    ap.add_argument("--demo", choices=["demo1", "demo2", "demo3"], help="preselect a demo clip")
    ap.add_argument("--source", help="video file / camera index to preselect")
    ap.add_argument("--mode", choices=[m for m, _ in MODES], help="identity mode")
    ap.add_argument("--start", action="store_true", help="start playback once models are loaded")
    ap.add_argument("--search", metavar="SUSPECT_ID", help="open Video Search for a suspect, e.g. SUSPECT_01")
    ap.add_argument("--tab", type=int, default=0,
                    help="tab to open: 0 live, 1 suspects, 2 search, 3 pipeline, 4 gallery, 5 eval, 6 audit")
    return ap.parse_args()


def _apply_args(app: "App", args):
    if args.mode:
        app._set_mode(args.mode)
    if args.demo:
        for label, (key, _) in app._demo_items.items():
            if key == args.demo:
                app._demo_var.set(label)
                app._select_demo()
    if args.source:
        app._source_var.set(args.source)
    if args.tab:
        app.select_tab(App.TAB_ORDER[args.tab] if args.tab < len(App.TAB_ORDER) else "live")
    if args.start:
        app._start()
    if args.search:
        app.search_tab.query_suspect(args.search)


if __name__ == "__main__":
    cli = _parse_args()
    app = App()
    app.after(1500, lambda: _apply_args(app, cli))
    app.mainloop()
