"""
console/tabs/gallery_tab.py
────────────────────────────────────────────────────────────────────────────────
Gallery tab: view enrolled employee crops per recording day.
────────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import pickle
import tkinter as tk
from pathlib import Path
from console.theme import dialogs as messagebox  # themed drop-in

from PIL import Image, ImageTk

from console.theme import (FONT_SEMI, OK, ACCENT, BG, BORDER, CARD, FG, FG_DIM, FONT,
                           ScrollFrame, button, combo)
from core.chirla import is_iso_date


class GalleryTab(tk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, bg=BG)
        self.app = app
        self._thumbs = []
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        top = tk.Frame(self, bg=BG)
        top.grid(row=0, column=0, sticky="ew", padx=8, pady=8)
        tk.Label(top, text="Enrolled employees", bg=BG, fg=FG, font=(FONT, 13, "bold")).pack(side="left")
        tk.Label(top, text="   recording day", bg=BG, fg=FG_DIM, font=(FONT, 9)).pack(side="left", padx=(12, 4))
        self.gday_var = tk.StringVar()
        self.gday_combo = combo(top, self.gday_var, [], width=14)
        self.gday_combo.pack(side="left")
        self.gday_combo.bind("<<ComboboxSelected>>", lambda e: self.render())
        button(top, "Build from check-in video…", self._open_checkin, ACCENT, icon="plus").pack(side="right")
        self.ginfo = tk.Label(top, text="", bg=BG, fg=FG_DIM, font=(FONT, 9), anchor="w", justify="left")
        self.ginfo.pack(side="left", padx=12, fill="x", expand=True)

        self.gscroll = ScrollFrame(self, bg=BG)
        self.gscroll.grid(row=1, column=0, sticky="nsew", padx=4)

    def _open_checkin(self):
        from console.tabs.checkin_dialog import CheckinDialog
        if self.app.checkin_busy:
            messagebox.showinfo("Check-in", "A check-in is already being analysed.")
            return
        CheckinDialog(self.app, initial_day=self.gday_var.get() or None)

    def load_days(self, select: str = None):
        root = Path(self.app.cfg["reid"]["gallery_file"]).parent
        days = sorted(
            p.name for p in root.iterdir()
            if p.is_dir() and is_iso_date(p.name) and (p / "gallery_prototypes.pkl").exists()
        ) if root.exists() else []
        self.gday_combo.config(values=days)
        if select in days:
            self.gday_var.set(select)
        elif days and self.gday_var.get() not in days:
            demo = self.app.cfg.get("chirla", {}).get("demo_day")
            self.gday_var.set(demo if demo in days else days[0])
        self.render()

    def render(self):
        self.gscroll.clear()
        self._thumbs = []
        day = self.gday_var.get()
        if not day:
            tk.Label(self.gscroll.inner, text="No gallery yet – run Pipeline step 2, or build one from a "
                                              "morning check-in video (button above).", bg=BG, fg=FG_DIM,
                     font=(FONT, 11)).pack(pady=40)
            return
        gdir = Path(self.app.cfg["reid"]["gallery_file"]).parent / day
        try:
            data = pickle.loads((gdir / "gallery_prototypes.pkl").read_bytes())
        except Exception:
            data = {}
        meta = data.get("meta", {})
        names = {folder: label for label, folder in meta.get("folders", {}).items()}   # check-in built days
        labels = set(data.get("prototypes", {}))
        emps = sorted(p for p in gdir.iterdir() if p.is_dir() and (names.get(p.name, p.name) in labels or not labels))
        how = (f"built from a check-in video ({meta.get('updated', '')[:16].replace('T', ' ')})"
               if meta.get("built_by") == "dashboard check-in" else "crops from each person's earliest footage that day")
        self.ginfo.config(text=f"{len(emps)} employees enrolled · model {meta.get('model', '?')} · {how}")
        grid = self.gscroll.inner
        ncol = 3
        for c in range(ncol):
            grid.columnconfigure(c, weight=1, uniform="g")
        for i, d in enumerate(emps):
            card = tk.Frame(grid, bg=CARD, highlightthickness=1, highlightbackground=BORDER)
            card.grid(row=i // ncol, column=i % ncol, sticky="nsew", padx=5, pady=5)
            head = tk.Frame(card, bg=CARD)
            head.pack(fill="x", padx=10, pady=(8, 4))
            label = names.get(d.name, d.name)
            tk.Label(head, text=label.replace("emp_", "Emp "), bg=CARD, fg=OK,
                     font=(FONT_SEMI, 11)).pack(side="left")
            cut = meta.get("enroll_cutoff", {}).get(label)
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
