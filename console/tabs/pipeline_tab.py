"""
console/tabs/pipeline_tab.py
────────────────────────────────────────────────────────────────────────────────
Offline Pipeline tab: run Re-ID OpenVINO export, daily employee enrollment,
open-set classifier training, demo clip extraction, and CCTV search indexing.
────────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import logging
import re
import time
import tkinter as tk
from pathlib import Path
from tkinter import ttk
from console.theme import dialogs as messagebox  # themed drop-in

from console.theme import (FONT_SEMI, LABEL, MONO_SEMI, OK, ACCENT, BG, BORDER, CARD, CARD2, CYAN, DANGER, FG,
                           FG_DIM, FONT, MONO, PURPLE, WARN, append_text,
                           button, clear_text)
from console.worker import INDEX_RE, PIPE_NOISE, PROGRESS_RE, SubprocessRunner

logger = logging.getLogger("console.pipeline")

DEMO_MANIFEST = Path("output/demo/demo_clips.json")

PIPELINE_STEPS = [
    {"key": "export", "title": "Re-ID → OpenVINO", "cmd": ["scripts/export_reid_openvino.py"],
     "desc": "Convert the MSMT17-trained OSNet-AIN to OpenVINO IR (≈7× faster on CPU)."},
    {"key": "gallery", "title": "Enroll employees", "cmd": ["scripts/build_gallery.py"],
     "desc": "Each day: clearest crops from every employee's earliest footage → "
             "L2-normalised prototype."},
    {"key": "train", "title": "Train open-set model", "cmd": ["scripts/train_open_set.py"],
     "desc": "5 tracklet features → logistic regression P(Known). Evaluated on held-out "
             "days vs the naive baseline."},
    {"key": "clips", "title": "Build demo clips", "cmd": ["scripts/make_demo_clips.py"],
     "desc": "Find & export the 3 evaluation clips (persistence, rejection, asset)."},
    {"key": "index", "title": "Index CCTV for search",
     "cmd": ["-m", "core.search", "index", "--day", "2023-06-08", "--day", "2023-07-26"],
     "desc": "Video-wide search index of the demo days: person tracklets + OSNet embeddings "
             "(≈15 min per day)."},
]


class PipelineTab(tk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, bg=BG)
        self.app = app
        self._runner = SubprocessRunner(self, app.project_dir)
        self._queue: list = []
        self._current_key: str = ""
        self.columnconfigure(0, weight=1)
        self.rowconfigure(2, weight=1)

        top = tk.Frame(self, bg=BG)
        top.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 4))
        tk.Label(top, text="Offline pipeline", bg=BG, fg=FG, font=(FONT, 13, "bold")).pack(side="left")
        tk.Label(top, text="   Re-ID export → daily enrollment → open-set training → demo clips",
                 bg=BG, fg=FG_DIM, font=(FONT, 9)).pack(side="left", pady=(3, 0))
        self._run_all_btn = button(top, "Run all steps", self.run_all, ACCENT, icon="play")
        self._run_all_btn.pack(side="right")

        cards = tk.Frame(self, bg=BG)
        cards.grid(row=1, column=0, sticky="ew", padx=4)
        self._steps = {}
        for i, st in enumerate(PIPELINE_STEPS):
            cards.columnconfigure(i, weight=1, uniform="steps")
            card = tk.Frame(cards, bg=CARD, highlightthickness=1, highlightbackground=BORDER)
            card.grid(row=0, column=i, sticky="nsew", padx=4, pady=4)
            head = tk.Frame(card, bg=CARD)
            head.pack(fill="x", padx=12, pady=(10, 2))
            tk.Label(head, text=f" {i + 1} ", bg=PURPLE, fg=BG, font=(FONT_SEMI, 10)).pack(side="left")
            tk.Label(head, text="  " + st["title"], bg=CARD, fg=FG, font=(FONT_SEMI, 11)).pack(side="left")
            tk.Label(card, text=st["desc"], bg=CARD, fg=FG_DIM, font=(FONT, 8), justify="left",
                     wraplength=230, anchor="w").pack(fill="x", padx=12, pady=(2, 6))
            status = tk.Label(card, text="", bg=CARD, fg=FG_DIM, font=(FONT_SEMI, 9), anchor="w")
            status.pack(fill="x", padx=12)
            prog = ttk.Progressbar(card, style="Purple.Horizontal.TProgressbar", mode="determinate", maximum=100)
            prog.pack(fill="x", padx=12, pady=(4, 6))
            btn = button(card, "Run", lambda k=st["key"]: self.run_steps([k]), CARD2, fg=FG)
            btn.pack(fill="x", padx=12, pady=(0, 12))
            self._steps[st["key"]] = {"status": status, "prog": prog, "btn": btn, **st}
        self.refresh_status()

        con = tk.Frame(self, bg=CARD)
        con.grid(row=2, column=0, sticky="nsew", padx=8, pady=(4, 8))
        con.rowconfigure(1, weight=1)
        con.columnconfigure(0, weight=1)
        tk.Label(con, text="PIPELINE OUTPUT", bg=CARD, fg=FG_DIM, font=LABEL,
                 anchor="w").grid(row=0, column=0, sticky="ew", padx=10, pady=(6, 0))
        self._console = tk.Text(con, bg="#0d1119", fg="#c8d0dc", font=(MONO, 9), bd=0,
                                highlightthickness=0, padx=10, pady=6, state="disabled", wrap="none")
        self._console.grid(row=1, column=0, sticky="nsew", padx=6, pady=6)
        vsb = ttk.Scrollbar(con, orient="vertical", command=self._console.yview, style="Dark.Vertical.TScrollbar")
        vsb.grid(row=1, column=1, sticky="ns", pady=6)
        self._console.configure(yscrollcommand=vsb.set)
        for tag, col in (("ok", OK), ("warn", WARN), ("err", DANGER), ("head", PURPLE), ("table", CYAN)):
            self._console.tag_configure(tag, foreground=col)
        self._console.tag_configure("head", font=(MONO_SEMI, 9))

    def terminate(self):
        self._runner.terminate()

    def refresh_status(self):
        reid = self.app.cfg["reid"]
        model = reid["model"]
        checks = {
            "export": Path("models/reid") / f"{model}_openvino" / f"{model}.xml",
            "gallery": Path(reid["gallery_file"]),
            "train": Path(reid["open_set_model"]),
            "clips": DEMO_MANIFEST,
            "index": Path(self.app.cfg.get("video_search", {}).get("index_dir", "output/index")),
        }
        for key, st in self._steps.items():
            p = checks[key]
            if p.exists():
                when = time.strftime("%d %b %H:%M", time.localtime(p.stat().st_mtime))
                st["status"].config(text=f"✓ ready · {when}", fg=OK)
                st["prog"]["value"] = 100
            else:
                st["status"].config(text="○ not run yet", fg=FG_DIM)
                st["prog"]["value"] = 0

    def run_all(self):
        self.run_steps([s["key"] for s in PIPELINE_STEPS])

    def run_steps(self, keys):
        if self._runner.is_running:
            messagebox.showinfo("Pipeline busy", "A pipeline step is already running.")
            return
        if self.app._running:
            if not messagebox.askyesno("Stop live monitoring?", "Pipeline steps need the CPU and the models, so "
                                       "live monitoring stops while they run.", ok="Stop and run",
                                       cancel="Keep monitoring"):
                return
            self.app._stop()
        self._queue = list(keys)
        self.app.select_tab("pipeline")
        clear_text(self._console)
        self._set_buttons("disabled")
        self._next_step()

    def _set_buttons(self, state: str):
        self._run_all_btn.config(state=state)
        for st in self._steps.values():
            st["btn"].config(state=state)

    def _next_step(self):
        if not self._queue:
            self._set_buttons("normal")
            self.app._set_status("✓ Pipeline finished")
            return
        key = self._queue.pop(0)
        self._current_key = key
        st = self._steps[key]
        st["status"].config(text="⟳ running …", fg=PURPLE)
        st["prog"].config(mode="indeterminate")
        st["prog"].start(12)
        cmd_str = " ".join(st["cmd"])
        append_text(self._console, [(f"\n━━━ {st['title']}  ·  python {cmd_str} ━━━\n", "head")])
        logger.info("Pipeline step started: %s", st["title"])
        self._runner.start(st["cmd"], on_line=self._on_line, on_done=self._on_done)

    def _on_line(self, line: str):
        if PIPE_NOISE.search(line):
            return
        st = self._steps[self._current_key]
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
        append_text(self._console, [(msg + "\n", tag)])

    def _on_done(self, rc: int, secs: float):
        key = self._current_key
        st = self._steps[key]
        st["prog"].stop()
        st["prog"].config(mode="determinate")
        if rc == 0:
            st["prog"]["value"] = 100
            st["status"].config(text=f"✓ done in {secs:.0f}s", fg=OK)
            logger.info("Pipeline step finished: %s (%.0fs)", st["title"], secs)
            self.app.on_pipeline_step_finished(key)
            self._next_step()
        else:
            st["prog"]["value"] = 0
            st["status"].config(text=f"✗ failed (exit {rc})", fg=DANGER)
            logger.error("Pipeline step failed: %s (exit %d)", st["title"], rc)
            self._queue = []
            self._set_buttons("normal")
