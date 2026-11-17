"""
console/app.py
────────────────────────────────────────────────────────────────────────────────
VIDEO VIGIL – Open-Set Security Console (Tkinter shell)

Main window shell, header, left control sidebar, bottom live log panel, and
tab notebook orchestration. Individual tab views live in `console/tabs/`.
────────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import queue
import time
import tkinter as tk
from pathlib import Path
from datetime import date as _date
from tkinter import filedialog, simpledialog, ttk
from typing import Optional

from console.icons import icon
from console.tabs import (AuditTab, EvalTab, GalleryTab, LiveTab, PipelineTab,
                          SearchTab, SuspectsTab)
from console.theme import (ACCENT, BG, BG2, BORDER, CARD, DANGER, FG, FG_DIM, FG_MUTED, FONT,
                           IS_WIN, MONO, OK, PANEL, WARN, Chip, Collapsible, ScrollFrame,
                           append_text, apply_style, button, caption, check, clear_text, combo,
                           dialogs as messagebox, divider, entry, labelled, note, section, slider)
from console.worker import MultiCameraWorker, QueueLogHandler, VideoWorker, analysis_rate
from core.chirla import date_in_name, is_iso_date
from core.detector import get_available_trackers, get_system_spec
from core.engine import DetectionEngine
from core.open_set import list_day_galleries
from core.recordings import recordings_cfg, scan_days
from core.reid import REID_MODELS

CONFIG_PATH = "config.yaml"
PROJECT_DIR = Path(__file__).resolve().parents[1]
DEMO_MANIFEST = Path("output/demo/demo_clips.json")
DEMO_PLACEHOLDER = "— choose a demo clip —"
DATE_PLACEHOLDER = "— choose recording date —"
DATE_OTHER = "Other date…"
SOURCE_MODES = [("demo", "Demo clip"), ("video", "Video / webcam"), ("day", "Recorded day")]
# Analysis rate (frames analysed per video second, per camera): label → config value
RATES = [("Auto", "auto"), ("All frames  (most accurate)", "all"), ("15 fps", "15"), ("10 fps", "10"),
         ("5 fps", "5"), ("3 fps  (smoothest)", "3")]

MODES = [
    ("tracklet", "Reliable  ·  several views per person"),
    ("naive", "Fast  ·  single frame"),
    ("legacy", "Basic  ·  session IDs, no staff list"),
]
# Notebook: operator views first, setup / research views after them
TABS = [("live", "Live Monitor", "target"), ("suspects", "Suspects", "flag"), ("search", "Video Search", "search"),
        ("audit", "Audit Log", "list"), ("pipeline", "Pipeline", "terminal"), ("gallery", "Staff Gallery", "users"),
        ("eval", "Evaluation", "chart")]
RUN_STATES = {"stopped": ("■ STOPPED", FG_DIM, CARD), "loading": ("◌ LOADING", WARN, CARD),
              "live": ("● LIVE", OK, CARD)}

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s – %(message)s")
logger = logging.getLogger("app")


class App(tk.Tk):
    TAB_ORDER = ["live", "suspects", "search", "pipeline", "gallery", "eval", "audit"]

    def __init__(self):
        super().__init__()
        self.project_dir = PROJECT_DIR
        os.chdir(self.project_dir)
        self.title("Video Vigil — Security Console")
        self.configure(bg=BG)
        self.minsize(1200, 720)
        self.after_idle(self._maximize_window)

        self._log_q: queue.Queue = queue.Queue()
        logging.getLogger().addHandler(QueueLogHandler(self._log_q))

        self.system_spec = get_system_spec()
        self.cfg = DetectionEngine.load_config(CONFIG_PATH)
        self.engine = DetectionEngine(self.cfg)
        self._tracker_names = get_available_trackers()

        self._worker = None                     # VideoWorker | MultiCameraWorker
        self._frame_q: queue.Queue = queue.Queue(maxsize=5)
        self._running = False
        self._after_id = None
        self._last_event: dict = {}             # id(engine) → last event number shown
        # Extra engines for cameras 2…N of a recorded day (camera 1 uses self.engine); they
        # share its Re-ID model, suspect gallery and logs, and are kept for the next run.
        self._engine_pool: list = []
        self._last_multi: Optional[dict] = None
        self._stopping = None                   # multi-camera worker still winding down
        self.checkin_busy = False               # Gallery → check-in analysis is using self.engine

        apply_style(self)
        self._alert_open = 0
        self._build_ui()
        self._bind_keys()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self._poll_logs()
        self._tick()
        self.after(400, self._auto_load_model)

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

        self._paned = paned = ttk.PanedWindow(self, orient="vertical")
        paned.grid(row=1, column=0, sticky="nsew")
        body = tk.Frame(paned, bg=BG)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)
        paned.add(body, weight=5)
        self._build_sidebar(body)

        self.nb = ttk.Notebook(body)
        self.nb.grid(row=0, column=1, sticky="nsew", padx=(0, 8), pady=(4, 0))
        self.live_tab = LiveTab(self.nb, self)
        self.suspects_tab = SuspectsTab(self.nb, self)
        self.search_tab = SearchTab(self.nb, self)
        self.pipeline_tab = PipelineTab(self.nb, self)
        self.gallery_tab = GalleryTab(self.nb, self)
        self.eval_tab = EvalTab(self.nb, self)
        self.audit_tab = AuditTab(self.nb, self)
        self._tabs = {
            "live": self.live_tab, "suspects": self.suspects_tab, "search": self.search_tab,
            "pipeline": self.pipeline_tab, "gallery": self.gallery_tab, "eval": self.eval_tab,
            "audit": self.audit_tab,
        }
        for key, text, icon_name in TABS:
            self.nb.add(self._tabs[key], text=f" {text} ", image=icon(icon_name, FG_DIM, 16), compound="left")
        self.nb.bind("<<NotebookTabChanged>>", lambda e: self._on_tab())

        self._log_frame = tk.Frame(paned, bg=BG2)
        paned.add(self._log_frame, weight=1)
        self._build_log_panel(self._log_frame)

        bar = tk.Frame(self, bg=PANEL)
        bar.grid(row=2, column=0, sticky="ew")
        divider(bar, edge="top")
        self._status_var = tk.StringVar(value="Starting …")
        tk.Label(bar, textvariable=self._status_var, bg=PANEL, fg=FG_DIM, font=(FONT, 9),
                 anchor="w", padx=14, pady=5).pack(side="left", fill="x", expand=True)
        spec = self.system_spec
        caption(bar, f"CPU {spec.cpu_cores} cores · RAM {spec.total_ram_gb:.0f} GB · "
                     f"GPU {spec.gpu_name if spec.has_gpu else 'none'}", bg=PANEL, padx=14).pack(side="right")
        caption(bar, "Space start/stop · A acknowledge · Esc back · Ctrl+L log · F11 full screen",
                bg=PANEL, padx=14).pack(side="right")

    def _build_header(self):
        hdr = tk.Frame(self, bg=PANEL, height=52)
        hdr.grid(row=0, column=0, sticky="ew")
        hdr.grid_propagate(False)
        hdr.columnconfigure(1, weight=1)
        hdr.rowconfigure(0, weight=1)
        logo = tk.Frame(hdr, bg=PANEL)
        logo.grid(row=0, column=0, sticky="w", padx=16)
        tk.Label(logo, image=icon("target", ACCENT, 20), bg=PANEL).pack(side="left", padx=(0, 8))
        tk.Label(logo, text="VIDEO VIGIL", bg=PANEL, fg=FG, font=(FONT, 13, "bold")).pack(side="left")
        caption(logo, "Security console", bg=PANEL).pack(side="left", padx=(12, 0), pady=(3, 0))
        self._run_pill = Chip(logo)
        self._run_pill.pack(side="left", padx=(20, 0))
        self._alert_pill = Chip(logo)
        self._alert_pill.bind("<Button-1>", lambda e: self.select_tab("live"))
        self._alert_flash = False

        chips = tk.Frame(hdr, bg=PANEL)
        chips.grid(row=0, column=2, sticky="e", padx=12)
        self._chip_mode = Chip(chips, colour=ACCENT)
        self._chip_model = Chip(chips)
        self._chip_reid = Chip(chips)
        self._chip_gallery = Chip(chips)
        for c in (self._chip_mode, self._chip_model, self._chip_reid, self._chip_gallery):
            c.pack(side="left", padx=(0, 6))
        self._clock = tk.Label(chips, bg=PANEL, fg=FG, font=(MONO, 11), padx=8)
        self._clock.pack(side="left")
        divider(hdr, edge="bottom")
        self._set_run_state("stopped")
        self._refresh_chips()

    def _build_sidebar(self, body):
        side = tk.Frame(body, bg=PANEL, width=300, highlightthickness=1, highlightbackground=BORDER)
        side.grid(row=0, column=0, sticky="nsw", padx=(8, 8), pady=(8, 0))
        side.pack_propagate(False)
        # START / STOP stay pinned under the scrolling settings
        ctrl = tk.Frame(side, bg=PANEL)
        ctrl.pack(side="bottom", fill="x")
        divider(ctrl)
        row = tk.Frame(ctrl, bg=PANEL)
        row.pack(fill="x", padx=16, pady=12)
        self._start_btn = button(row, "START", self._start, ACCENT, icon="play", size=10)
        self._start_btn.config(pady=8)
        self._start_btn.pack(side="left", fill="x", expand=True, padx=(0, 4))
        self._stop_btn = button(row, "STOP", self._stop, DANGER, icon="stop", size=10, state="disabled")
        self._stop_btn.config(pady=8)
        self._stop_btn.pack(side="left", fill="x", expand=True, padx=(4, 0))
        sf = ScrollFrame(side, bg=PANEL)
        sf.pack(side="top", fill="both", expand=True)
        sb = sf.inner

        # Video Source: demo clip · own video / webcam · recorded day (all cameras together)
        caption(sb, "Video source", bg=PANEL).pack(fill="x", padx=16, pady=(14, 6))
        self._src_mode = tk.StringVar(value="video")
        mode_row = tk.Frame(sb, bg=PANEL)
        mode_row.pack(fill="x", padx=16)
        self._src_mode_buttons = {}
        for i, (key, text) in enumerate(SOURCE_MODES):
            b = button(mode_row, text, lambda k=key: self._set_source_mode(k), size=8)
            b.config(padx=4)
            b.grid(row=0, column=i, sticky="ew", padx=(0 if i == 0 else 3, 0))
            mode_row.columnconfigure(i, weight=1, uniform="src")
            self._src_mode_buttons[key] = b
        src_box = tk.Frame(sb, bg=PANEL)
        src_box.pack(fill="x", pady=(8, 0))
        self._src_panels = {k: tk.Frame(src_box, bg=PANEL) for k, _ in SOURCE_MODES}

        # ··· Demo clip
        p = self._src_panels["demo"]
        self._demo_var = tk.StringVar()
        self._demo_combo = combo(p, self._demo_var, [])
        self._demo_combo.pack(fill="x", padx=16)
        self._demo_combo.bind("<<ComboboxSelected>>", lambda e: self._select_demo())
        self._demo_note = note(p)
        self._demo_note.pack(fill="x", padx=16, pady=(4, 0))
        button(p, "Play smooth render", self._play_render, icon="play").pack(fill="x", padx=16, pady=(6, 0))

        # ··· Own video or webcam (+ its recording date → that day's staff gallery; required for
        #     video files – a date in the file name is only offered as a hint)
        p = self._src_panels["video"]
        self._source_var = tk.StringVar(value="Webcam (device 0)")
        entry(p, self._source_var).pack(fill="x", padx=16, ipady=5)
        row = tk.Frame(p, bg=PANEL)
        row.pack(fill="x", padx=16, pady=(6, 0))
        button(row, "Webcam", lambda: self._source_var.set("Webcam (device 0)"), icon="webcam").pack(
            side="left", fill="x", expand=True, padx=(0, 3))
        button(row, "Browse…", self._browse_file, icon="folder").pack(side="left", fill="x", expand=True, padx=(3, 0))
        self._video_date: Optional[str] = None
        self._date_items: dict = {}
        self._date_var = tk.StringVar(value=DATE_PLACEHOLDER)
        self._date_combo = combo(p, self._date_var, [])
        labelled(p, "Recorded", self._date_combo)
        self._date_combo.bind("<<ComboboxSelected>>", lambda e: self._on_date_pick())
        self._date_hint = tk.Frame(p, bg=PANEL)
        self._date_hint.pack(fill="x", padx=16)
        self._date_hint_btn = button(self._date_hint, "", self._use_name_date)
        self._date_note = note(p)
        self._date_note.pack(fill="x", padx=16, pady=(2, 0))

        # ··· Recorded day: every camera of one sequence
        p = self._src_panels["day"]
        self._days: dict = {}
        self._day_items: dict = {}
        self._seq_items: dict = {}
        self._day_var, self._seq_var = tk.StringVar(), tk.StringVar()
        self._day_combo = combo(p, self._day_var, [])
        labelled(p, "Day", self._day_combo)
        self._day_combo.bind("<<ComboboxSelected>>", lambda e: self._select_day())
        self._seq_combo = combo(p, self._seq_var, [])
        labelled(p, "Sequence", self._seq_combo)
        self._seq_combo.bind("<<ComboboxSelected>>", lambda e: self._update_day_note())
        self._day_note = note(p)
        self._day_note.pack(fill="x", padx=16, pady=(2, 0))
        button(p, "Rescan recordings", self._reload_days, icon="refresh").pack(fill="x", padx=16, pady=(6, 0))

        # Analysis rate: accuracy ↔ smoothness (applies to all modes, live)
        self._rate_var = tk.StringVar(value=self._rate_label(self.cfg.get("video", {}).get("analysis_fps", "auto")))
        rate = combo(sb, self._rate_var, [label for label, _ in RATES])
        labelled(sb, "Analysis", rate)
        rate.bind("<<ComboboxSelected>>", lambda e: self._on_rate_changed())
        self._rate_note = note(sb)
        self._rate_note.pack(fill="x", padx=16, pady=(2, 0))

        self._reload_gallery_days()
        self._reload_demo_clips()
        self._reload_days()
        self._source_var.trace_add("write", lambda *_: self._on_source_changed())
        self._set_source_mode("video")
        self._update_rate_note()

        # Asset monitoring
        section(sb, "Asset monitoring")
        self._assets_var = tk.BooleanVar(value=self.engine._assets.enabled)
        check(sb, "Watch laptops and bags", self._assets_var,
              lambda: self._for_engines(lambda e: setattr(e._assets, "enabled", self._assets_var.get()))).pack(
            fill="x", padx=12)
        arow = tk.Frame(sb, bg=PANEL)
        arow.pack(fill="x", padx=16, pady=(6, 0))
        self._draw_btn = button(arow, "Draw laptop anchor", lambda: self.live_tab.begin_anchor(self._draw_btn),
                                icon="pencil")
        self._draw_btn.pack(side="left", fill="x", expand=True, padx=(0, 4))
        button(arow, "Clear", self._clear_anchors, icon="trash").pack(side="left")
        note(sb, "Moving one raises a theft alert unless staff move it. Laptops that stay still are "
                 "anchored automatically; draw a box to set one yourself.").pack(
            fill="x", padx=16, pady=(4, 0))

        # Display & detection switches
        section(sb, "Monitoring")
        toggles = tk.Frame(sb, bg=PANEL)
        toggles.pack(fill="x", padx=12)
        toggles.columnconfigure((0, 1), weight=1)
        self._trail_var = tk.BooleanVar(value=True)
        self._reid_var = tk.BooleanVar(value=True)
        self._zones_var = tk.BooleanVar(value=self.engine.zones_enabled)
        self._realtime_var = tk.BooleanVar(value=True)
        for i, (text, var, cmd) in enumerate((
            ("Show trails", self._trail_var,
             lambda: self._for_engines(lambda e: setattr(e, "show_trail", self._trail_var.get()))),
            ("Recognise people", self._reid_var,
             lambda: self._for_engines(lambda e: setattr(e, "_reid_enabled", self._reid_var.get()))),
            ("Restricted zones", self._zones_var, self._toggle_zones),
            ("Real-time playback", self._realtime_var, self._toggle_realtime),
        )):
            check(toggles, text, var, cmd).grid(row=i // 2, column=i % 2, sticky="w")

        # Advanced: models, identity method, thresholds (closed by default – operators rarely need it)
        adv = Collapsible(sb, "Advanced", hint="models · thresholds")
        adv.pack(fill="x", pady=(0, 12))
        ab = adv.body
        caption(ab, "Models", bg=PANEL).pack(fill="x", padx=16, pady=(8, 4))
        model_keys = list(self.cfg.get("models", {}).keys())
        default_model = next((k for k in ("yolo11n_int8_480", "yolo11n_int8") if k in model_keys),
                             model_keys[0] if model_keys else "")
        self._model_var = tk.StringVar(value=default_model)
        labelled(ab, "Detector", combo(ab, self._model_var, model_keys))
        self._reid_model_var = tk.StringVar(value=self.cfg.get("reid", {}).get("model", REID_MODELS[0]))
        labelled(ab, "Re-ID", combo(ab, self._reid_model_var, REID_MODELS))
        self._tracker_var = tk.StringVar(value=self.cfg.get("tracking", {}).get("tracker", "bytetrack"))
        labelled(ab, "Tracker", combo(ab, self._tracker_var, self._tracker_names))
        button(ab, "Load models", self._load_model, icon="layers").pack(fill="x", padx=16, pady=(8, 0))

        caption(ab, "Recognition method", bg=PANEL).pack(fill="x", padx=16, pady=(14, 4))
        self._mode_var = tk.StringVar(value=self.cfg.get("reid", {}).get("mode", "tracklet"))
        self._mode_buttons = {}
        for key, label in MODES:
            b = button(ab, label, lambda k=key: self._set_mode(k))
            b.config(anchor="w", padx=12)
            b.pack(fill="x", padx=16, pady=(0, 3))
            self._mode_buttons[key] = b
        self._paint_mode_buttons()

        caption(ab, "Thresholds", bg=PANEL).pack(fill="x", padx=16, pady=(14, 0))
        det = self.cfg["detection"]
        self._conf_var = tk.DoubleVar(value=det.get("conf_threshold", 0.35))
        self._person_conf_var = tk.DoubleVar(value=det.get("person_conf_threshold", 0.5))
        self._pknown_var = tk.DoubleVar(value=self.cfg["reid"].get("open_set_p_threshold", 0.5))
        self._iou_var = tk.DoubleVar(value=det.get("iou_threshold", 0.45))
        slider(ab, "Detector confidence (all objects)", self._conf_var, 0.05, 1.0,
               lambda v: self._for_engines(lambda e: setattr(e, "conf", v)))
        slider(ab, "Person confidence (hides chairs etc.)", self._person_conf_var, 0.05, 1.0,
               lambda v: self._for_engines(lambda e: setattr(e, "person_conf", v)))
        slider(ab, "Staff match threshold", self._pknown_var, 0.05, 0.95,
               lambda v: self._for_engines(lambda e: e.set_open_set_threshold(v)))
        slider(ab, "Box overlap (IoU)", self._iou_var, 0.1, 1.0,
               lambda v: self._for_engines(lambda e: setattr(e, "iou", v)))
        note(ab, "Staff match threshold: below this certainty a person is shown as UNKNOWN. "
                 "Higher = stricter.").pack(fill="x", padx=16, pady=(6, 0))
        button(ab, "Reset to defaults", self._reset_thresholds, icon="refresh").pack(fill="x", padx=16, pady=(8, 0))

    def _reset_thresholds(self):
        det, reid = self.cfg["detection"], self.cfg["reid"]
        for var, value, apply in (
            (self._conf_var, det.get("conf_threshold", 0.35), lambda e, v: setattr(e, "conf", v)),
            (self._person_conf_var, det.get("person_conf_threshold", 0.5), lambda e, v: setattr(e, "person_conf", v)),
            (self._pknown_var, reid.get("open_set_p_threshold", 0.5), lambda e, v: e.set_open_set_threshold(v)),
            (self._iou_var, det.get("iou_threshold", 0.45), lambda e, v: setattr(e, "iou", v)),
        ):
            var.set(value)
            self._for_engines(lambda e: apply(e, value))
        self._set_status("Thresholds reset to the config.yaml defaults")

    def _build_log_panel(self, frame):
        frame.rowconfigure(1, weight=1)
        frame.columnconfigure(0, weight=1)
        hdr = tk.Frame(frame, bg=BG2)
        hdr.grid(row=0, column=0, columnspan=2, sticky="ew")
        divider(hdr, edge="top")
        caption(hdr, "System log", bg=BG2).pack(side="left", padx=(14, 0), pady=(6, 2))
        self._autoscroll = tk.BooleanVar(value=True)
        check(hdr, "Auto-scroll", self._autoscroll, bg=BG2).pack(side="right", padx=8)
        button(hdr, "Clear", lambda: clear_text(self._log), size=8).pack(side="right", pady=(4, 2))
        button(hdr, "Hide  ·  Ctrl+L", self._toggle_log, size=8).pack(side="right", padx=(0, 6), pady=(4, 2))
        self._log = tk.Text(frame, bg=BG2, fg=FG_DIM, font=(MONO, 8), bd=0, height=7,
                            highlightthickness=0, padx=14, pady=4, state="disabled", wrap="none")
        self._log.grid(row=1, column=0, sticky="nsew", pady=(0, 4))
        vsb = ttk.Scrollbar(frame, orient="vertical", command=self._log.yview, style="Dark.Vertical.TScrollbar")
        vsb.grid(row=1, column=1, sticky="ns", pady=(0, 4))
        self._log.configure(yscrollcommand=vsb.set)
        for tag, col in (("time", FG_MUTED), ("INFO", FG_DIM), ("WARNING", WARN), ("ERROR", DANGER),
                         ("ALERT", DANGER), ("src", FG_MUTED), ("PIPE", ACCENT)):
            self._log.tag_configure(tag, foreground=col)
    # ── Navigation & Status ───────────────────────────────────────────────────

    def _set_status(self, msg: str):
        self._status_var.set(msg)

    def _refresh_chips(self):
        mode = self._mode_var.get() if hasattr(self, "_mode_var") else self.cfg["reid"].get("mode", "tracklet")
        self._chip_mode.set({"tracklet": "Reliable", "naive": "Fast", "legacy": "Basic"}.get(mode, mode))
        key = self.engine._model_key if self.engine._model is not None else None
        self._chip_model.set(key or "no detector", FG_DIM if key else WARN)
        ext = self.engine._reid_extractor
        backend = "OpenVINO" if ext is not None and ext._ov_model is not None else "PyTorch" if ext else "—"
        self._chip_reid.set(f"{self.engine._reid_model_name or '—'} · {backend}", FG_DIM)
        self._chip_gallery.set(f"Staff · {self.engine.gallery_info}", FG_DIM)

    def _set_run_state(self, state: str):
        text, colour, bg = RUN_STATES[state]
        self._run_pill.set(text, colour, bg, edge=colour if state != "stopped" else None)

    def on_alerts_changed(self, n_open: int):
        """Header alert pill: visible and flashing while alerts are unacknowledged."""
        self._alert_open = n_open
        if n_open:
            self._alert_pill.set(f"{n_open} open alert{'s' if n_open > 1 else ''}", FG, DANGER, edge=DANGER)
            self._alert_pill.pack(side="left", padx=(6, 0))
        else:
            self._alert_pill.pack_forget()

    def on_alerts_acknowledged(self, items: list):
        texts = "; ".join(t for _, _, _, t in items)[:400]
        self.engine.log_operator_action("ALERT_ACKNOWLEDGED", f"{len(items)} alert(s): {texts}")
        self._set_status(f"✓ {len(items)} alert{'s' if len(items) > 1 else ''} acknowledged · logged to the audit log")

    def _bind_keys(self):
        """Global shortcuts; ignored while typing in an entry / combobox."""
        def typing() -> bool:
            w = self.focus_get()
            return isinstance(w, (tk.Entry, ttk.Entry, ttk.Combobox, tk.Text, tk.Button, tk.Checkbutton))

        def key(seq, fn):
            self.bind_all(seq, lambda e: None if typing() else (fn(), "break")[1])

        key("<space>", lambda: self._stop() if self._running else self._start())
        key("<KeyPress-a>", self.live_tab.alerts.acknowledge)
        key("<Escape>", lambda: self.live_tab.cancel_anchor() or self.live_tab.set_focus(None))
        self.bind_all("<Control-l>", lambda e: self._toggle_log())
        self.bind_all("<F11>", lambda e: self.attributes("-fullscreen", not self.attributes("-fullscreen")))

    def _toggle_log(self):
        if str(self._log_frame) in self._paned.panes():
            self._paned.forget(self._log_frame)
        else:
            self._paned.add(self._log_frame, weight=1)

    def select_tab(self, name: str):
        self.nb.select(self._tabs[name])

    def current_tab(self) -> str:
        cur = self.nb.select()
        return next((k for k, t in self._tabs.items() if str(t) == cur), "")

    def _tick(self):
        self._clock.config(text=time.strftime("%H:%M:%S"))
        tab = self.current_tab()
        if tab == "audit":
            self.audit_tab.refresh()
        sig = self.suspects_tab.signature()
        if sig != getattr(self, "_suspect_sig", None):
            self._suspect_sig = sig
            self.search_tab.refresh_queries()
            n = len(self.engine.suspects)
            self.live_tab.tiles["suspects"].set(n)
            self.nb.tab(self.suspects_tab, text=f" Suspects · {n} " if n else " Suspects ")
            if tab == "suspects":
                self.suspects_tab.render()
        self.after(1000, self._tick)

    def _on_tab(self):
        tab = self.current_tab()
        if tab == "gallery":
            self.gallery_tab.load_days()
        elif tab == "eval":
            self.eval_tab.render()
        elif tab == "audit":
            self.audit_tab.refresh(force=True)
        elif tab == "suspects":
            self.suspects_tab.render()
        elif tab == "search":
            self.search_tab.refresh_days()

    # ── Actions ───────────────────────────────────────────────────────────────

    def _auto_load_model(self):
        self.live_tab.draw_placeholder()
        if self._model_var.get():
            self._load_model(quiet=True)

    def _load_model(self, quiet: bool = False):
        if self.checkin_busy:
            if not quiet:
                messagebox.showinfo("Check-in running", "Wait for the check-in analysis to finish before "
                                                        "loading other models.")
            return
        if self._running:
            self._stop()
        key, reid_model = self._model_var.get(), self._reid_model_var.get()
        self._set_status(f"Loading {key} + Re-ID {reid_model} …")
        self._set_run_state("loading")
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
        self._set_run_state("stopped")
        self._refresh_chips()
        self._reload_gallery_days()      # which days are usable depends on the Re-ID model

    def _set_mode(self, mode: str):
        self._mode_var.set(mode)
        self._for_engines(lambda e: e.set_identity_mode(mode))
        self._paint_mode_buttons()
        self._refresh_chips()
        self._set_status(f"Identity mode → {mode}")

    def _paint_mode_buttons(self):
        for key, b in self._mode_buttons.items():
            b.set_active(key == self._mode_var.get())

    def _toggle_zones(self):
        self._for_engines(lambda e: setattr(e, "zones_enabled", self._zones_var.get()))
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
            titles = {
                "demo1": "Demo 1 · Known employee",
                "demo2": "Demo 2 · Unknown visitor",
                "demo3": "Demo 3 · Asset interaction",
            }
            for k in sorted(manifest):
                m = manifest[k]
                label = f"{titles.get(k, k)} ({m.get('focus', '')})"
                self._demo_items[label] = (k, m)
        self._demo_combo.config(values=list(self._demo_items) or ["(run Pipeline step 4)"])
        if self._demo_var.get() not in self._demo_items:
            self._demo_var.set(DEMO_PLACEHOLDER if self._demo_items else "(run Pipeline step 4)")

    def _demo_path(self, m: dict) -> str:
        return str(DEMO_MANIFEST.parent / m["file"])

    # ── Recording date (staff gallery of that day) ────────────────────────────

    def _reload_gallery_days(self):
        """Fill the date picker with every enrolled day; keep the current choice."""
        model = self.engine._reid_model_name or self.cfg.get("reid", {}).get("model")
        gallery_file = self.cfg.get("reid", {}).get("gallery_file", "enrolled_gallery/gallery_prototypes.pkl")
        try:
            days = list_day_galleries(gallery_file, model)
        except Exception as exc:
            logger.warning("Could not list enrolled days (%s)", exc)
            days = []
        self._gallery_days = {d["date"]: d for d in days}
        self._date_items = {}
        for d in days:
            label = f"{d['date']}  ·  {d['staff']} staff" if d["usable"] else f"{d['date']}  ·  can't be used"
            self._date_items[label] = d["date"]
        if self._video_date and self._video_date not in self._gallery_days:
            self._date_items[f"{self._video_date}  ·  no staff enrolled"] = self._video_date
        self._date_combo.config(values=list(self._date_items) + [DATE_OTHER])
        self._show_video_date()

    def _set_video_date(self, iso: Optional[str]):
        self._video_date = iso if is_iso_date(iso) else None
        if self._video_date and self._video_date not in self._date_items.values():
            self._date_items[f"{self._video_date}  ·  no staff enrolled"] = self._video_date
            self._date_combo.config(values=list(self._date_items) + [DATE_OTHER])
        self._show_video_date()

    def _show_video_date(self):
        label = next((k for k, v in self._date_items.items() if v == self._video_date), None)
        self._date_var.set(label if self._video_date and label else DATE_PLACEHOLDER)
        self._update_date_note()

    def _on_date_pick(self):
        picked = self._date_var.get()
        if picked != DATE_OTHER:
            self._set_video_date(self._date_items.get(picked))
            return
        hint = date_in_name(self._source_var.get()) or self._video_date or _date.today().isoformat()
        while True:
            iso = simpledialog.askstring("Recording date", "Date the video was recorded (YYYY-MM-DD):",
                                         initialvalue=hint, parent=self)
            if iso is None:                     # cancelled: keep the previous choice
                self._show_video_date()
                return
            iso = iso.strip()
            if not is_iso_date(iso):
                messagebox.showerror("Recording date", f"“{iso}” isn't a valid date – use YYYY-MM-DD.")
            elif iso > _date.today().isoformat():
                messagebox.showerror("Recording date", "The recording date can't be in the future.")
            else:
                self._set_video_date(iso)
                return
            hint = iso

    def _use_name_date(self):
        self._set_video_date(date_in_name(self._source_var.get()))

    def _update_date_note(self):
        source = self._resolve_source()
        iso, is_file = self._video_date, isinstance(source, str)
        suggested = date_in_name(source) if is_file else None
        if suggested and suggested != iso:
            self._date_hint_btn.config(text=f"Use {suggested} (date in the file name)")
            self._date_hint_btn.pack(fill="x", pady=(4, 0))
        else:
            self._date_hint_btn.pack_forget()

        colour = FG_DIM
        if iso is None:
            if is_file:
                text, colour = "Choose the date this video was recorded – the staff enrolled that day are " \
                               "recognised.", WARN
            else:
                text = "Webcam: uses the default staff gallery. Choose a date to use another day's staff."
        else:
            day = self._gallery_days.get(iso)
            if day and day["usable"]:
                text, colour = f"✓ The {day['staff']} staff enrolled on {iso} will be recognised.", OK
            else:
                why = day["problem"] if day and day.get("problem") else f"No staff are enrolled for {iso}"
                text, colour = f"⚠ {why}. Everyone will be UNKNOWN.", WARN
            if suggested and suggested != iso:
                text += f"\nThe file name mentions {suggested} – check the date."
                colour = WARN
        self._date_note.config(text=text, fg=colour)

    def _on_source_changed(self):
        """A new source needs its own recording date; a demo selection no longer applies."""
        src = self._source_var.get().strip()
        item = self._demo_items.get(self._demo_var.get())
        if item and self._demo_path(item[1]) != src:
            self._demo_var.set(DEMO_PLACEHOLDER)
            self._demo_note.config(text="")
        self._video_date = None
        self._show_video_date()

    def _select_demo(self):
        item = self._demo_items.get(self._demo_var.get())
        if not item:
            return
        key, m = item
        self._source_var.set(self._demo_path(m))
        self._set_video_date(m.get("date") or date_in_name(m["file"]))
        tips = {
            "demo1": "Watch the employee keep a stable green identity through the occlusion.",
            "demo2": "Run once in NAIVE mode (visitor gets an employee name), then TRACKLET "
                     "(visitor becomes UNKNOWN).",
            "demo3": "Unknown person at a laptop → asset / zone alert + audit log entry.",
        }
        people = ", ".join(f"emp {p}" for p in m.get("employees", [])) or "—"
        visitors = ", ".join(str(v) for v in m.get("visitors", [])) or "—"
        self._demo_note.config(text=f"{m.get('source', '')} · {m.get('date', '')}\n"
                                    f"In view: {people} · visitors {visitors}\n{tips.get(key, '')}")

    # ── Source mode ───────────────────────────────────────────────────────────

    def source_mode(self) -> str:
        return self._src_mode.get()

    def _set_source_mode(self, mode: str):
        self._src_mode.set(mode)
        for key, panel in self._src_panels.items():
            panel.pack_forget()
            self._src_mode_buttons[key].set_active(key == mode)
        self._src_panels[mode].pack(fill="x")
        self._update_rate_note()

    # ── Recorded days (every camera of one sequence) ──────────────────────────

    def _reload_days(self):
        rc = recordings_cfg(self.cfg)
        try:
            self._days = scan_days(rc["root"], int(rc["cameras"]))
        except Exception as exc:
            logger.warning("Could not scan recordings in %s (%s)", rc["root"], exc)
            self._days = {}
        self._day_items = {f"{d}  ·  {len(s)} sequence{'s' if len(s) > 1 else ''}": d
                           for d, s in self._days.items()}
        self._day_combo.config(values=list(self._day_items))
        if self._day_var.get() not in self._day_items:
            self._day_var.set("— choose a day —" if self._day_items else "(no recordings found)")
        self._select_day()

    def _select_day(self):
        seqs = self._days.get(self._day_items.get(self._day_var.get()), [])
        self._seq_items = {}
        for s in seqs:
            start = s.start.strftime("%H:%M:%S") if s.start else "time ?"
            self._seq_items[f"{s.name}  ·  {start}  ·  {len(s.cameras)} cam"] = s
        values = list(self._seq_items)
        self._seq_combo.config(values=values)
        if self._seq_var.get() not in self._seq_items:
            self._seq_var.set(values[0] if values else "")
        self._update_day_note()

    def selected_sequence(self):
        return self._seq_items.get(self._seq_var.get())

    def _update_day_note(self):
        root = recordings_cfg(self.cfg)["root"]
        seq = self.selected_sequence()
        colour = FG_DIM
        if not self._days:
            text, colour = (f"No recorded days found in “{root}”. Expected "
                            f"{root}/YYYY-MM-DD/<sequence>/camera_N_….avi"), WARN
        elif seq is None:
            text = "Choose a day; then pick one of its sequences."
        else:
            cams = ", ".join(c.cam_id.replace("Cam_", "") for c in seq.cameras)
            lines = [f"{len(seq.cameras)} cameras play together: {cams}"]
            day = self._gallery_days.get(seq.date)
            if day and day["usable"]:
                lines.append(f"✓ {day['staff']} staff enrolled on {seq.date} will be recognised.")
            else:
                why = day["problem"] if day and day.get("problem") else f"No staff are enrolled for {seq.date}"
                lines.append(f"⚠ {why} – everyone will be UNKNOWN.")
                colour = WARN
            if seq.missing:
                lines.append(f"⚠ Missing: {', '.join(seq.missing)} – playing the {len(seq.cameras)} available.")
                colour = WARN
            if seq.duplicates:
                lines.append(f"Extra videos ignored: {', '.join(p.name for p in seq.duplicates)}")
            text = "\n".join(lines)
        self._day_note.config(text=text, fg=colour)
        self._update_rate_note()

    # ── Analysis rate (accuracy ↔ smoothness) ─────────────────────────────────

    @staticmethod
    def _rate_label(value) -> str:
        text = str(value).strip().lower()
        for label, v in RATES:
            if v == text or (v.isdigit() and text.replace(".0", "") == v):
                return label
        return RATES[0][0]

    def _rate_cameras(self) -> int:
        w = self._worker
        if isinstance(w, MultiCameraWorker):
            return len(w.requested)
        seq = self.selected_sequence() if self.source_mode() == "day" else None
        return len(seq.cameras) if seq else 1

    def _current_rate(self, n_cameras: int) -> Optional[float]:
        return analysis_rate(dict(RATES).get(self._rate_var.get(), "auto"), n_cameras, self.cfg.get("video", {}))

    def _on_rate_changed(self):
        w = self._worker
        if w is not None:
            w.analysis_fps = self._current_rate(self._rate_cameras())
        self._update_rate_note()

    def _update_rate_note(self):
        if not hasattr(self, "_rate_note"):
            return
        n = self._rate_cameras()
        rate = self._current_rate(n)
        per = " per camera" if n > 1 else ""
        if rate is None:
            text = (f"Every frame is analysed{per} – most accurate tracking and identities; "
                    f"playback lags behind real time when the CPU can't keep up.")
        else:
            text = (f"Up to {rate:g} frames per second{per} are analysed (the FPS tile shows what this "
                    f"computer reaches). Lower = smoother playback, but fewer looks at each person: "
                    f"identities take longer and are less certain.")
        self._rate_note.config(text=text)

    # ── Engines (one per camera for a recorded day) ───────────────────────────

    def _engines(self) -> list:
        w = self._worker
        if isinstance(w, MultiCameraWorker) and w.engines:
            return list(w.engines)
        return [self.engine]

    def _for_engines(self, fn):
        for e in self._engines():
            fn(e)

    def engine_at(self, index: int) -> DetectionEngine:
        engines = self._engines()
        return engines[index] if 0 <= index < len(engines) else self.engine

    def active_engine(self) -> DetectionEngine:
        """The enlarged camera's engine in the grid, else the main engine."""
        tab = self.live_tab
        if tab.multi is not None and tab.focus is not None:
            return self.engine_at(tab.focus)
        return self.engine

    def _engine_for_camera(self, i: int) -> DetectionEngine:
        """Engine for the i-th camera (runs on the playback thread; may load a detector)."""
        if i == 0:
            return self.engine
        while len(self._engine_pool) < i:
            self._engine_pool.append(DetectionEngine(self.cfg, shared=self.engine))
        engine = self._engine_pool[i - 1]
        engine.adopt_settings(self.engine)
        return engine

    def on_focus_changed(self):
        if self._last_multi is not None:
            self._show_multi_stats(self._last_multi)

    def _play_render(self):
        item = self._demo_items.get(self._demo_var.get())
        if not item:
            messagebox.showinfo("Pick a demo", "Choose a demo clip first.")
            return
        stem = item[1]["file"].split("__")[0]
        renders = sorted(Path("output/rendered").glob(f"{stem}__*.mp4"))
        mode = self._mode_var.get()
        pick = next((r for r in renders if r.stem.endswith(mode)), renders[0] if renders else None)
        if pick is None:
            messagebox.showinfo("Not rendered", f"Run:  python scripts/render_demo.py {item[0]} --mode {mode}")
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
        if self._running:
            if not messagebox.askyesno("Stop live monitoring?",
                                       f"Playing {Path(path).name} stops the current monitoring session. "
                                       "Alerts from the live source are not raised while you review.",
                                       ok="Stop and play", cancel="Keep monitoring"):
                return
            self._stop()
        self._set_source_mode("video")
        self._source_var.set(path)
        # Search / suspect results point at CHIRLA videos and demo clips, whose names carry the
        # recording date. Anything else asks the user (see _start).
        self._set_video_date(date_in_name(Path(path).name))
        self._start(start_frame=frame)

    def find_video(self, name: str) -> Optional[Path]:
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

    def _clear_anchors(self):
        if self.live_tab.multi is not None and self.live_tab.focus is None:
            messagebox.showinfo("Clear anchors", "Click a camera to enlarge it first, then clear its anchors.")
            return
        engine = self.active_engine()
        if not messagebox.askyesno("Clear anchors", f"Remove every drawn laptop anchor for {engine._camera_id}? "
                                   "Auto-detected anchors come back once the laptops are still again.",
                                   ok="Clear anchors", cancel="Cancel", danger=True):
            return
        engine.clear_asset_anchors()
        self._set_status(f"Drawn anchors cleared for {engine._cam_key or 'this camera'}")

    def _start(self, start_frame: int = 0):
        if self.engine._model is None:
            messagebox.showwarning("No model", "Load the models first.")
            return
        if self._running:
            return
        if self.checkin_busy:
            messagebox.showinfo("Check-in running", "A morning check-in is being analysed (Gallery tab). "
                                                    "Wait for it to finish or cancel it, then press START.")
            return
        if self.engine._model_key != self._model_var.get():
            self._load_model()
            if self.engine._model_key != self._model_var.get():
                return
        self.engine._reid_enabled = self._reid_var.get()
        self.engine.show_trail = self._trail_var.get()
        if self.engine._reid_model_name != self._reid_model_var.get():
            self.engine.set_reid_model(self._reid_model_var.get())
        self.engine.set_tracker(self._tracker_var.get())
        self.engine.set_identity_mode(self._mode_var.get())

        mode = self.source_mode()
        if mode == "day":
            self._start_day()
            return
        if mode == "demo" and not self._demo_items.get(self._demo_var.get()):
            messagebox.showinfo("Demo clip", "Choose a demo clip first – or switch to “Video / webcam” "
                                             "to play your own video.")
            return
        source = self._resolve_source()
        if isinstance(source, str) and self._video_date is None:
            messagebox.showinfo(
                "Recording date",
                "Choose the date this video was recorded (VIDEO SOURCE → Recorded).\n\n"
                "The staff enrolled on that day are used to recognise employees. If nobody was "
                "enrolled that day, choose “Other date…” and everyone will be shown as UNKNOWN.")
            self._date_combo.focus_set()
            return
        self._frame_q = queue.Queue(maxsize=5)
        self._worker = VideoWorker(source, self.engine, self._frame_q,
                                   self.cfg.get("video", {}).get("target_fps", 30),
                                   realtime=self._realtime_var.get(), start_frame=start_frame,
                                   gallery_date=self._video_date, analysis_fps=self._current_rate(1))
        self._begin_run(f"Running · {source}")

    def _start_day(self):
        """All cameras of the chosen recorded sequence, together (missing cameras: warning)."""
        seq = self.selected_sequence()
        if seq is None:
            messagebox.showinfo("Recorded day", "Choose a recorded day and one of its sequences first.")
            return
        if self._stopping is not None and self._stopping.is_alive():
            messagebox.showinfo("Recorded day", "The previous playback is still shutting down – "
                                                "try again in a moment.")
            return
        self._frame_q = queue.Queue(maxsize=5)
        self._worker = MultiCameraWorker(seq.cameras, self._engine_for_camera, self._frame_q, seq.date,
                                         self._current_rate(len(seq.cameras)), realtime=self._realtime_var.get())
        self.live_tab.start_multi([c.cam_id for c in seq.cameras])
        self._begin_run(f"Starting {seq.date} {seq.name} · {len(seq.cameras)} cameras …")
        if seq.missing:
            text = (f"{', '.join(seq.missing)} missing from {seq.date}/{seq.name} – playing the "
                    f"{len(seq.cameras)} available cameras")
            logger.warning(text)
            self.live_tab.append_event("alert", time.strftime("%H:%M:%S"), "—", text)

    def _begin_run(self, status: str):
        self._worker.start()
        self._gallery_shown = False
        self._last_multi = None
        self._running = True
        self.live_tab.clear_feed()
        self._start_btn.config(state="disabled")
        self._stop_btn.config(state="normal")
        self._set_run_state("live")
        self.select_tab("live")
        self._set_status(status)
        self._poll_frames()

    def _stop(self):
        self._running = False
        worker = self._worker
        if worker:
            worker.stop()
            worker.join(timeout=2.0)
            self._drain_events_of(getattr(worker, "engines", None) or [self.engine])
            # a multi-camera worker may still be loading a detector; don't reuse its engines yet
            self._stopping = worker if worker.is_alive() else None
        self._worker = None
        if self._after_id:
            self.after_cancel(self._after_id)
            self._after_id = None
        self._start_btn.config(state="normal")
        self._stop_btn.config(state="disabled")
        self._set_run_state("stopped")
        self._set_status("Stopped.")
        self.live_tab.end_multi()
        self.live_tab.draw_placeholder()

    def _poll_frames(self):
        if not self._running:
            return
        try:
            while True:
                kind, payload = self._frame_q.get_nowait()
                if kind == "frame":
                    frame, stats = payload
                    self.live_tab.display_frame(frame)
                    self.live_tab.update_stats(stats)
                    if not self._gallery_shown:      # the worker has bound the source's gallery
                        self._gallery_shown = True
                        self._refresh_chips()
                        rep = self.engine.gallery_report
                        if rep.get("warning") and rep.get("how") != "legacy":
                            self._set_status(f"Running · ⚠ {rep['warning']} – everyone is shown as UNKNOWN")
                elif kind == "multi":
                    self._last_multi = payload
                    self.live_tab.show_multi(payload)
                    self._show_multi_stats(payload)
                    if not self._gallery_shown:
                        self._gallery_shown = True
                        self._refresh_chips()
                        n = len(payload["frames"])
                        rep = self.engine.gallery_report
                        warn = f" · ⚠ {rep['warning']} – everyone is UNKNOWN" if rep.get("warning") else ""
                        self._set_status(f"Running · {n} camera{'s' if n > 1 else ''} · click a camera to "
                                         f"enlarge it{warn}")
                elif kind == "status":
                    self._set_status(payload)
                elif kind == "warning":
                    logger.warning(payload)
                    self.live_tab.append_event("alert", time.strftime("%H:%M:%S"), "—", payload)
                elif kind == "done":
                    self._drain_events()
                    self._stop()
                    self._set_status("Video ended." if self._last_multi is None else "All cameras ended.")
                    return
                elif kind == "error":
                    messagebox.showerror("Error", payload)
                    self._stop()
                    return
        except queue.Empty:
            pass
        self._drain_events()
        self._after_id = self.after(15, self._poll_frames)

    def _show_multi_stats(self, payload: dict):
        """Totals over every camera for the stat tiles + all cameras' people in one table."""
        stats = payload["stats"]
        labels = payload.get("labels") or []
        people = [{**p, "cam": i, "cam_id": s.get("camera") or (labels[i] if i < len(labels) else "")}
                  for i, s in enumerate(stats) for p in s.get("people", [])]
        live = [s for s in stats if s]
        agg = {k: sum(s.get(k, 0) or 0 for s in live) for k in ("known", "unknown", "alerts", "tracks")}
        agg["fps"] = sum(s.get("fps", 0) for s in live) / len(live) if live else 0.0
        agg["people"] = people
        focus = self.live_tab.focus
        if focus is not None and focus < len(stats) and stats[focus]:
            s = stats[focus]
            agg.update(camera=f"{s.get('camera', '')} (enlarged)", time=s.get("time", ""), assets=s.get("assets", []))
        elif live:
            ended = sum(payload["ended"])
            agg.update(camera=f"{len(stats)} cameras" + (f", {ended} ended" if ended else ""),
                       time=live[0].get("time", ""),
                       assets=[(f"{s.get('camera', '')} {n}", st) for s in live for n, st in s.get("assets", [])])
        self.live_tab.update_stats(agg)

    def _drain_events(self):
        self._drain_events_of(self._engines())

    def _drain_events_of(self, engines):
        multi = len(engines) > 1 or self._last_multi is not None
        for index, engine in enumerate(engines):
            last = self._last_event.get(id(engine), 0)
            for no, level, stamp, cam, text in engine.events_since(last):
                last = no
                self.live_tab.append_event(level, stamp, cam, text, index if multi else None)
            self._last_event[id(engine)] = last

    def on_gallery_changed(self, day: str):
        """A day's staff gallery was (re)built from a check-in: refresh every place that lists galleries."""
        self._reload_gallery_days()
        self._update_day_note()
        self.gallery_tab.load_days(select=day)
        self._refresh_chips()
        self._set_status(f"✓ Staff gallery for {day} updated – used by videos recorded that day from now on")

    def on_pipeline_step_finished(self, key: str):
        if key in ("export", "gallery", "train"):
            self.engine._reid_extractor_cache.clear()
            self.engine._reid_model_name = None
            self.engine.set_reid_model(self._reid_model_var.get())
            self._drop_engine_pool()        # rebuilt with the new models / gallery / classifier
            self._refresh_chips()
            self._reload_gallery_days()
        if key == "clips":
            self._reload_demo_clips()
        if key == "index":
            self.search_tab.refresh_days()

    def _poll_logs(self):
        chunks = []
        try:
            for _ in range(200):
                item = self._log_q.get_nowait()
                if item[0] == "log":
                    _, levelno, name, msg, created = item
                    level = "ALERT" if "ALERT" in msg else logging.getLevelName(levelno)
                    chunks += [
                        (time.strftime("%H:%M:%S ", time.localtime(created)), "time"),
                        (f"{name[:16]:<16} ", "src"),
                        (msg + "\n", level if level in ("WARNING", "ERROR", "ALERT") else "INFO"),
                    ]
        except queue.Empty:
            pass
        if chunks:
            append_text(self._log, chunks, autoscroll=self._autoscroll.get())
        self.after(100, self._poll_logs)

    def _drop_engine_pool(self):
        if isinstance(self._worker, MultiCameraWorker):
            self._stop()
        for engine in self._engine_pool:
            engine.close()
        self._engine_pool.clear()

    def _on_close(self):
        busy = [what for what, on in (
            ("live monitoring", self._running), ("a pipeline step", self.pipeline_tab._runner.is_running),
            ("a search index build", self.search_tab._runner.is_running), ("a check-in analysis", self.checkin_busy),
        ) if on]
        if busy and not messagebox.askyesno("Quit Video Vigil?", f"This stops {', '.join(busy)}.",
                                            ok="Quit", cancel="Keep running", danger=True):
            return
        self._stop()
        for engine in self._engine_pool:
            engine.close()
        self.pipeline_tab.terminate()
        self.search_tab._runner.terminate()
        self.engine.close()
        self.destroy()


def _parse_args():
    ap = argparse.ArgumentParser(description="VIDEO VIGIL security console")
    ap.add_argument("--demo", choices=["demo1", "demo2", "demo3"], help="preselect a demo clip")
    ap.add_argument("--source", help="video file / camera index to preselect")
    ap.add_argument("--date", help="recording date of --source (YYYY-MM-DD) → that day's staff gallery")
    ap.add_argument("--day", help="recorded day to preselect (YYYY-MM-DD): all its cameras play together")
    ap.add_argument("--seq", help="sequence of --day, e.g. seq_000 (default: the day's first)")
    ap.add_argument("--rate", help="analysis rate per camera: auto | all | 15 | 10 | 5 | 3")
    ap.add_argument("--mode", choices=[m for m, _ in MODES], help="identity mode")
    ap.add_argument("--start", action="store_true", help="start playback once models are loaded")
    ap.add_argument("--search", metavar="SUSPECT_ID", help="open Video Search for a suspect, e.g. SUSPECT_01")
    ap.add_argument("--tab", type=int, default=0,
                    help="tab to open: 0 live, 1 suspects, 2 search, 3 pipeline, 4 gallery, 5 eval, 6 audit")
    return ap.parse_args()


def _apply_args(app: App, args):
    if args.mode:
        app._set_mode(args.mode)
    if args.rate:
        app._rate_var.set(app._rate_label(args.rate))
        app._on_rate_changed()
    if args.demo:
        app._set_source_mode("demo")
        for label, (key, _) in app._demo_items.items():
            if key == args.demo:
                app._demo_var.set(label)
                app._select_demo()
    if args.source:
        app._set_source_mode("video")
        app._source_var.set(args.source)
    if args.day:
        app._set_source_mode("day")
        label = next((k for k, d in app._day_items.items() if d == args.day), None)
        if label is None:
            logger.warning("--day %s: no recordings for that day", args.day)
        else:
            app._day_var.set(label)
            app._select_day()
            seq = next((k for k, s in app._seq_items.items() if s.name == args.seq), None) if args.seq else None
            if seq:
                app._seq_var.set(seq)
                app._update_day_note()
            elif args.seq:
                logger.warning("--seq %s not found on %s – using %s", args.seq, args.day, app._seq_var.get())
    if args.date:
        if is_iso_date(args.date):
            app._set_video_date(args.date)
        else:
            logger.warning("--date %s ignored: use YYYY-MM-DD", args.date)
    if args.tab:
        app.select_tab(App.TAB_ORDER[args.tab] if args.tab < len(App.TAB_ORDER) else "live")
    if args.start:
        app._start()
    if args.search:
        app.search_tab.query_suspect(args.search)


def main():
    cli = _parse_args()
    app = App()
    app.after(1500, lambda: _apply_args(app, cli))
    app.mainloop()


if __name__ == "__main__":
    main()
