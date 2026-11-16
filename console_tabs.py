"""
console_tabs.py
────────────────────────────────────────────────────────────────────────────────
Console tabs for the theft workflow:

  SuspectsTab – auto-enrolled suspect gallery: crops, why / when / where they
                were enrolled, forensic backtrack (where they entered the
                view), re-ID sightings; search them across all CCTV.
  SearchTab   – video-wide search: query = suspect / enrolled employee /
                person picked in the live view / photo; ranked results over
                the indexed recordings with "play from here"; index builder.
────────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import sys
import threading
import time
import tkinter as tk
from collections import Counter
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageTk

import video_search as vs
from chirla import day_gallery_path, sequence_dates
from open_set import PrototypeGallery
from ui_theme import (ACCENT, ACCENT2, BG, BORDER, CARD, CARD2, DANGER, FG, FG_DIM, FONT, IS_WIN, MONO,
                      PANEL, PURPLE, WARN, ScrollFrame, button, combo)

logger = logging.getLogger("console")


def _photo(img_bgr: np.ndarray, size=(64, 128)) -> ImageTk.PhotoImage:
    rgb = cv2.cvtColor(cv2.resize(img_bgr, size), cv2.COLOR_BGR2RGB)
    return ImageTk.PhotoImage(Image.fromarray(rgb))


# ─────────────────────────────────────────────────────────────────────────────
#  Suspects
# ─────────────────────────────────────────────────────────────────────────────

class SuspectsTab(tk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, bg=BG)
        self.app = app
        self._imgs = []
        self._shown = None
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
        top = tk.Frame(self, bg=BG)
        top.grid(row=0, column=0, sticky="ew", padx=8, pady=8)
        tk.Label(top, text="Suspect gallery", bg=BG, fg=FG, font=(FONT, 13, "bold")).pack(side="left")
        self.count = tk.Label(top, text="", bg=BG, fg=FG_DIM, font=(FONT, 9))
        self.count.pack(side="left", padx=12)
        button(top, "Clear all", self.clear_all, CARD2, fg=FG).pack(side="right", padx=(4, 0))
        button(top, "Refresh", lambda: self.render(force=True), CARD2, fg=FG).pack(side="right", padx=4)
        self.scroll = ScrollFrame(self, bg=BG)
        self.scroll.grid(row=1, column=0, sticky="nsew", padx=4)

    @property
    def gallery(self):
        return self.app.engine.suspects

    def signature(self):
        g = self.gallery
        return tuple((s.suspect_id, len(s.sightings)) for s in g.suspects.values())

    def render(self, force=False):
        sig = self.signature()
        if not force and sig == self._shown:
            return
        self._shown = sig
        self.scroll.clear()
        self._imgs = []
        g = self.gallery
        self.count.config(text=f"{len(g)} suspect(s) · auto-enrolled on unauthorised asset displacement or "
                               f"lingering, or flagged by the operator · {g.dir}")
        if not len(g):
            tk.Label(self.scroll.inner, bg=BG, fg=FG_DIM, font=(FONT, 11), justify="center",
                     text="No suspects yet.\n\nThey are enrolled automatically when an UNKNOWN person displaces "
                          "a monitored laptop\n(or lingers at it), or when you select a person in the Live "
                          "Monitor and press  🚩 Flag suspect.").pack(pady=60)
            return
        for s in sorted(g.suspects.values(), key=lambda s: s.suspect_id):
            self._card(s)

    def _card(self, s):
        card = tk.Frame(self.scroll.inner, bg=CARD, highlightthickness=1, highlightbackground=DANGER)
        card.pack(fill="x", padx=6, pady=5)
        head = tk.Frame(card, bg=CARD)
        head.pack(fill="x", padx=12, pady=(8, 2))
        tk.Label(head, text=f"🕵 {s.label}", bg=CARD, fg=DANGER, font=(FONT, 12, "bold")).pack(side="left")
        reason = {"displacement": "unauthorised asset displacement", "unknown_interaction":
                  "unknown person lingering at an asset", "manual": "flagged by operator"}.get(s.reason, s.reason)
        tk.Label(head, text=f"   enrolled {s.created} · {s.camera} · {reason} · {s.n_crops} crops",
                 bg=CARD, fg=FG_DIM, font=(FONT, 9)).pack(side="left")
        for txt, cmd, col in (("Delete", lambda: self.delete(s.suspect_id), CARD2),
                              ("▶ Play origin", lambda: self.play_origin(s), CARD2),
                              ("🔎 Search all CCTV", lambda: self.app.search_tab.query_suspect(s.suspect_id),
                               ACCENT2)):
            button(head, txt, cmd, col, fg=FG if col == CARD2 else BG).pack(side="right", padx=3)

        body = tk.Frame(card, bg=CARD)
        body.pack(fill="x", padx=12, pady=(4, 10))
        crops = tk.Frame(body, bg=CARD)
        crops.pack(side="left", anchor="n")
        for f in self.gallery.crops(s.suspect_id)[:8]:
            img = cv2.imread(str(f))
            if img is None:
                continue
            ph = _photo(img, (48, 96))
            self._imgs.append(ph)
            tk.Label(crops, image=ph, bg=CARD).pack(side="left", padx=1)

        info = tk.Frame(body, bg=CARD)
        info.pack(side="left", fill="x", expand=True, padx=14, anchor="n")
        e = s.entry or {}
        origin = (f"Entered {e.get('camera', '?')} at {e.get('time', '?')} · {e.get('side', '?')}"
                  if e else "Origin unknown (track started before history logging)")
        tk.Label(info, text="FORENSIC BACKTRACK", bg=CARD, fg=FG_DIM, font=(FONT, 8, "bold"),
                 anchor="w").pack(fill="x")
        tk.Label(info, text=origin, bg=CARD, fg=FG, font=(FONT, 9), anchor="w", justify="left",
                 wraplength=520).pack(fill="x")
        tk.Label(info, text=f"RE-ID SIGHTINGS ({len(s.sightings)})", bg=CARD, fg=FG_DIM, font=(FONT, 8, "bold"),
                 anchor="w").pack(fill="x", pady=(6, 0))
        if s.sightings:
            for x in s.sightings[-5:][::-1]:
                tk.Label(info, text=f"• {x.get('time')} · {x.get('camera')} · T{x.get('track')} · "
                                    f"similarity {x.get('sim', 0):.2f} · {x.get('source', '')}",
                         bg=CARD, fg=WARN, font=(FONT, 9), anchor="w").pack(fill="x")
        else:
            tk.Label(info, text="none yet – play other cameras / clips, or use 🔎 Search all CCTV",
                     bg=CARD, fg=FG_DIM, font=(FONT, 9), anchor="w").pack(fill="x")

    def play_origin(self, s):
        src = s.entry.get("source") or s.source
        path = self.app.find_video(src)
        if path is None:
            messagebox.showinfo("Not found", f"Cannot locate the source video:\n{src}")
            return
        self.app.play_from(str(path), max(0, int(s.entry.get("frame", s.frame)) - 30))

    def delete(self, sid):
        self.gallery.delete(sid)
        self.render(force=True)
        self.app.search_tab.refresh_queries()

    def clear_all(self):
        if len(self.gallery) and messagebox.askyesno("Clear suspects", "Delete ALL suspect profiles?"):
            self.gallery.clear()
            self.render(force=True)
            self.app.search_tab.refresh_queries()


# ─────────────────────────────────────────────────────────────────────────────
#  Video-wide search
# ─────────────────────────────────────────────────────────────────────────────

QUERY_TYPES = ["Suspect", "Enrolled employee", "Selected live person", "Photo file"]
INDEX_RE = re.compile(r"\[(\d+)/(\d+)\].*indexed (\d+) / (\d+) frames")


class SearchTab(tk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, bg=BG)
        self.app = app
        self.cfg = app.cfg
        self._vs = self.cfg.get("video_search", {}) or {}
        self._imgs = []
        self._query = None            # (label, embs, thumbs)
        self._live_query = None
        self._index_cache = {}
        self._proc: Optional[subprocess.Popen] = None
        self._q = []
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)
        self._build_left()
        self._build_right()
        self.refresh_queries()
        self.refresh_days()

    # ── layout ───────────────────────────────────────────────────────────────

    def _build_left(self):
        sf = ScrollFrame(self, bg=PANEL, width=300)
        sf.grid(row=0, column=0, sticky="nsw", padx=(6, 0), pady=6)
        sf.grid_propagate(False)
        p = sf.inner

        def sec(t):
            tk.Label(p, text=t, bg=PANEL, fg=FG_DIM, font=(FONT, 8, "bold"), anchor="w").pack(
                fill="x", padx=14, pady=(12, 4))

        sec("WHO ARE YOU LOOKING FOR?")
        self.qtype = tk.StringVar(value=QUERY_TYPES[0])
        cb = combo(p, self.qtype, QUERY_TYPES)
        cb.pack(fill="x", padx=14)
        cb.bind("<<ComboboxSelected>>", lambda e: self._show_query_widgets())
        self.qbox = tk.Frame(p, bg=PANEL)
        self.qbox.pack(fill="x", padx=14, pady=(6, 0))
        self.suspect_var = tk.StringVar()
        self.suspect_cb = combo(self.qbox, self.suspect_var, [])
        self.emp_day = tk.StringVar()
        self.emp_var = tk.StringVar()
        self.emp_day_cb = combo(self.qbox, self.emp_day, [], 12)
        self.emp_cb = combo(self.qbox, self.emp_var, [], 12)
        self.emp_day_cb.bind("<<ComboboxSelected>>", lambda e: self._load_employees())
        self.live_lbl = tk.Label(self.qbox, bg=PANEL, fg=FG, font=(FONT, 9), anchor="w", justify="left",
                                 text="Select a person in Live Monitor → 🔎 Search person")
        self.photo_btn = button(self.qbox, "📂  Choose photo…", self._choose_photo, CARD2, fg=FG)
        self.photo_lbl = tk.Label(self.qbox, bg=PANEL, fg=FG_DIM, font=(FONT, 8), anchor="w")
        self.qthumbs = tk.Frame(p, bg=PANEL)
        self.qthumbs.pack(fill="x", padx=14, pady=(6, 0))

        sec("SEARCH IN (INDEXED RECORDINGS)")
        self.days_lb = tk.Listbox(p, selectmode="multiple", height=6, bg=CARD, fg=FG, bd=0,
                                  highlightthickness=1, highlightbackground=BORDER, selectbackground=ACCENT2,
                                  font=(FONT, 9), activestyle="none", exportselection=False)
        self.days_lb.pack(fill="x", padx=14)
        row = tk.Frame(p, bg=PANEL)
        row.pack(fill="x", padx=14, pady=(6, 0))
        self.build_day = tk.StringVar()
        self.build_cb = combo(row, self.build_day, [], 12)
        self.build_cb.pack(side="left", fill="x", expand=True)
        self.build_btn = button(row, "Index day", self._build_index, PURPLE)
        self.build_btn.pack(side="left", padx=(4, 0))
        self.build_prog = ttk.Progressbar(p, style="Purple.Horizontal.TProgressbar", maximum=100)
        self.build_prog.pack(fill="x", padx=14, pady=(6, 0))
        self.build_lbl = tk.Label(p, text="", bg=PANEL, fg=FG_DIM, font=(FONT, 8), anchor="w", justify="left",
                                  wraplength=260)
        self.build_lbl.pack(fill="x", padx=14)

        sec("OPTIONS")
        self.excl = tk.BooleanVar(value=bool(self._vs.get("exclude_employees", True)))
        tk.Checkbutton(p, text="Exclude recognised employees", variable=self.excl, bg=PANEL, fg=FG,
                       selectcolor=CARD, activebackground=PANEL, font=(FONT, 9)).pack(anchor="w", padx=10)
        f = tk.Frame(p, bg=PANEL)
        f.pack(fill="x", padx=14, pady=(4, 0))
        tk.Label(f, text="Min similarity", bg=PANEL, fg=FG_DIM, font=(FONT, 9)).pack(side="left")
        self.min_sim = tk.DoubleVar(value=float(self._vs.get("min_similarity", 0.6)))
        val = tk.Label(f, text=f"{self.min_sim.get():.2f}", bg=PANEL, fg=ACCENT, font=(FONT, 9, "bold"))
        val.pack(side="right")
        tk.Scale(p, from_=0.3, to=0.95, resolution=0.01, orient="horizontal", variable=self.min_sim, bg=PANEL,
                 troughcolor=CARD, highlightthickness=0, bd=0, showvalue=False, sliderlength=18, width=8,
                 command=lambda _: val.config(text=f"{self.min_sim.get():.2f}")).pack(fill="x", padx=12)

        self.search_btn = button(p, "🔎  SEARCH", self.run_search, ACCENT)
        self.search_btn.config(font=(FONT, 11, "bold"), pady=9)
        self.search_btn.pack(fill="x", padx=14, pady=(14, 6))
        tk.Label(p, text="Results are ranked by appearance similarity of whole tracklets – verify them by eye. "
                         "Same-day only: people change clothes between days.",
                 bg=PANEL, fg=FG_DIM, font=(FONT, 8), justify="left", wraplength=260, anchor="w").pack(
            fill="x", padx=14, pady=(0, 12))
        self._show_query_widgets()

    def _build_right(self):
        right = tk.Frame(self, bg=BG)
        right.grid(row=0, column=1, sticky="nsew", padx=6, pady=6)
        right.rowconfigure(1, weight=1)
        right.columnconfigure(0, weight=1)
        self.res_hdr = tk.Label(right, text="Pick who to look for and press SEARCH.", bg=BG, fg=FG,
                                font=(FONT, 12, "bold"), anchor="w")
        self.res_hdr.grid(row=0, column=0, sticky="ew", pady=(0, 2))
        self.results = ScrollFrame(right, bg=BG)
        self.results.grid(row=1, column=0, sticky="nsew")

    # ── query selection ──────────────────────────────────────────────────────

    def _show_query_widgets(self):
        for w in self.qbox.winfo_children():
            w.pack_forget()
        t = self.qtype.get()
        if t == "Suspect":
            self.suspect_cb.pack(fill="x")
        elif t == "Enrolled employee":
            self.emp_day_cb.pack(side="left", fill="x", expand=True)
            self.emp_cb.pack(side="left", fill="x", expand=True, padx=(4, 0))
        elif t == "Selected live person":
            self.live_lbl.pack(fill="x")
        else:
            self.photo_btn.pack(fill="x")
            self.photo_lbl.pack(fill="x")

    def refresh_queries(self):
        ids = sorted(self.app.engine.suspects.suspects)
        self.suspect_cb.config(values=ids)
        if ids and self.suspect_var.get() not in ids:
            self.suspect_var.set(ids[-1])
        elif not ids:
            self.suspect_var.set("(no suspects yet)")
        root = Path(self.cfg["reid"]["gallery_file"]).parent
        days = sorted(p.name for p in root.iterdir() if p.is_dir()) if root.exists() else []
        self.emp_day_cb.config(values=days)
        if days and self.emp_day.get() not in days:
            self.emp_day.set(days[0])
            self._load_employees()

    def _load_employees(self):
        try:
            g = PrototypeGallery.load(day_gallery_path(self.cfg["reid"]["gallery_file"], self.emp_day.get()))
            self.emp_cb.config(values=g.labels)
            self.emp_var.set(g.labels[0] if g.labels else "")
        except Exception:
            self.emp_cb.config(values=[])

    def query_suspect(self, sid: str):
        self.refresh_queries()
        self.qtype.set("Suspect")
        self.suspect_var.set(sid)
        self._show_query_widgets()
        self.app.select_tab("search")
        self.run_search()

    def query_live(self, tid: int, embs, thumbs, camera: str):
        self._live_query = (f"T{tid} on {camera}", embs, thumbs)
        self.live_lbl.config(text=f"T{tid} on {camera} · {len(embs)} crops")
        self.qtype.set("Selected live person")
        self._show_query_widgets()
        self.app.select_tab("search")
        self.run_search()

    def _choose_photo(self):
        path = filedialog.askopenfilename(title="Photo of the person",
                                          filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp"), ("All", "*.*")])
        if not path:
            return
        img = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            messagebox.showerror("Photo", "Could not read that image.")
            return
        embs = self.app.engine.embed_image(img)
        self._photo_query = (Path(path).name, embs, [cv2.resize(img, (64, 128))])
        self.photo_lbl.config(text=Path(path).name)

    def _resolve_query(self):
        t = self.qtype.get()
        if t == "Suspect":
            sid = self.suspect_var.get()
            g = self.app.engine.suspects
            if sid not in g.prototypes:
                return None
            thumbs = [cv2.imread(str(f)) for f in g.crops(sid)[:6]]
            return (g.suspects[sid].label, g.prototypes[sid][None, :], [t for t in thumbs if t is not None])
        if t == "Enrolled employee":
            day, emp = self.emp_day.get(), self.emp_var.get()
            g = PrototypeGallery.load(day_gallery_path(self.cfg["reid"]["gallery_file"], day))
            if emp not in g.labels:
                return None
            crops_dir = day_gallery_path(self.cfg["reid"]["gallery_file"], day).parent / emp
            thumbs = [cv2.imread(str(f)) for f in sorted(crops_dir.glob("*.jpg"))]
            return (f"{emp.replace('emp_', 'Emp ')} ({day})", g.prototypes[g.labels.index(emp)][None, :],
                    [t for t in thumbs if t is not None])
        if t == "Selected live person":
            return self._live_query
        return getattr(self, "_photo_query", None)

    # ── index ────────────────────────────────────────────────────────────────

    def refresh_days(self):
        days = vs.indexed_days(self.cfg)
        sel = {self.days_lb.get(i).split()[0] for i in self.days_lb.curselection()}
        self.days_lb.delete(0, "end")
        for d, n in days.items():
            self.days_lb.insert("end", f"{d}   ({n} videos)")
            if not sel or d in sel:
                self.days_lb.selection_set("end")
        try:
            all_days = sorted(set(sequence_dates(self.cfg["chirla"]["root"]).values()))
        except Exception:
            all_days = []
        self.build_cb.config(values=all_days)
        if all_days and not self.build_day.get():
            self.build_day.set(next((d for d in all_days if d not in days), all_days[0]))

    def _build_index(self):
        day = self.build_day.get()
        if not day or self._proc is not None:
            return
        self.build_btn.config(state="disabled")
        self.build_prog["value"] = 0
        self.build_lbl.config(text=f"Indexing {day} … (≈10–25 min on this CPU; you can keep working)")
        logger.info("Search index build started for %s", day)
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
        flags = subprocess.CREATE_NO_WINDOW if IS_WIN else 0
        self._proc = subprocess.Popen([sys.executable, "-u", "video_search.py", "index", "--day", day],
                                      stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                      encoding="utf-8", errors="replace", creationflags=flags, env=env)
        threading.Thread(target=self._read_proc, args=(self._proc,), daemon=True).start()
        self.after(300, self._poll_proc)

    def _read_proc(self, proc):
        for line in proc.stdout:
            self._q.append(line.rstrip())
        self._q.append(("__done__", proc.wait()))

    def _poll_proc(self):
        while self._q:
            item = self._q.pop(0)
            if isinstance(item, tuple):
                rc = item[1]
                self._proc = None
                self.build_btn.config(state="normal")
                self.build_lbl.config(text="✓ Index built" if rc == 0 else f"✗ Index build failed (exit {rc})")
                self.build_prog["value"] = 100 if rc == 0 else 0
                logger.info("Search index build finished (exit %d)", rc)
                self._index_cache.clear()
                self.refresh_days()
                return
            m = INDEX_RE.search(item)
            if m:
                k, n, f, total = (int(x) for x in m.groups())
                self.build_prog["value"] = 100 * ((k - 1) + f / max(total, 1)) / n
                self.build_lbl.config(text=f"Indexing video {k}/{n} · frame {f:,}/{total:,}")
            elif "✓" in item or "Error" in item or "Traceback" in item:
                logger.info("index: %s", re.sub(r"^\S+ \S+ \[\w+\] ", "", item))
        self.after(300, self._poll_proc)

    # ── search ───────────────────────────────────────────────────────────────

    def _entries(self, days):
        key = tuple(days)
        if key not in self._index_cache:
            self._index_cache = {key: vs.load_index(self.cfg, list(days))}
        return self._index_cache[key]

    def run_search(self):
        q = self._resolve_query()
        if q is None:
            messagebox.showinfo("Search", "Choose who to look for first.")
            return
        days = [self.days_lb.get(i).split()[0] for i in self.days_lb.curselection()]
        if not days:
            messagebox.showinfo("Search", "No indexed recordings selected.\nIndex a day first (left panel).")
            return
        label, embs, thumbs = q
        self._query = q
        self.res_hdr.config(text=f"Searching for {label} in {', '.join(days)} …")
        self.search_btn.config(state="disabled")
        excl, min_sim, top_k = self.excl.get(), self.min_sim.get(), int(self._vs.get("top_k", 30))
        p_thresh = float(self.cfg["reid"].get("open_set_p_threshold", 0.5))

        def work():
            t0 = time.time()
            entries = self._entries(days)
            res = vs.search(np.asarray(embs, np.float32), entries, exclude_known=excl, p_thresh=p_thresh,
                            min_sim=min_sim, top_k=top_k)
            self.after(0, lambda: self._show(label, thumbs, res, len(entries), time.time() - t0))

        threading.Thread(target=work, daemon=True).start()

    def _show(self, label, thumbs, res, n_entries, secs):
        self.search_btn.config(state="normal")
        self.results.clear()
        self._imgs = []
        cams = len({r.entry["video"] for r in res})
        self.res_hdr.config(text=f"{label}: {len(res)} match(es) in {cams} video(s) · searched {n_entries:,} "
                                 f"tracklets in {secs:.1f}s")
        logger.info("Search for %s → %d results", label, len(res))
        qrow = tk.Frame(self.results.inner, bg=BG)
        qrow.pack(fill="x", padx=4, pady=(0, 6))
        tk.Label(qrow, text="QUERY", bg=BG, fg=FG_DIM, font=(FONT, 8, "bold")).pack(side="left", padx=(0, 6))
        for t in thumbs[:6]:
            ph = _photo(t, (40, 80))
            self._imgs.append(ph)
            tk.Label(qrow, image=ph, bg=BG).pack(side="left", padx=1)
        gts = Counter(r.entry["gt"] for r in res[:10] if r.entry.get("gt", -1) >= 0)
        if gts:
            tk.Label(qrow, text="   CHIRLA ground truth in top 10: " +
                     ", ".join(f"id {g} ×{n}" for g, n in gts.most_common()),
                     bg=BG, fg=FG_DIM, font=(FONT, 9)).pack(side="left")
        if not res:
            tk.Label(self.results.inner, text="No matches above the similarity threshold.", bg=BG, fg=FG_DIM,
                     font=(FONT, 11)).pack(pady=40)
            return
        grid = tk.Frame(self.results.inner, bg=BG)
        grid.pack(fill="x")
        ncol = 4
        for c in range(ncol):
            grid.columnconfigure(c, weight=1, uniform="r")
        for i, r in enumerate(res):
            self._result_card(grid, i, r).grid(row=i // ncol, column=i % ncol, sticky="nsew", padx=4, pady=4)

    def _result_card(self, parent, rank, r):
        e = r.entry
        col = ACCENT if r.sim >= 0.75 else WARN if r.sim >= 0.65 else FG_DIM
        card = tk.Frame(parent, bg=CARD, highlightthickness=1, highlightbackground=BORDER)
        top = tk.Frame(card, bg=CARD)
        top.pack(fill="x", padx=8, pady=(6, 2))
        tk.Label(top, text=f"#{rank + 1}", bg=CARD, fg=FG_DIM, font=(FONT, 9, "bold")).pack(side="left")
        tk.Label(top, text=f"{r.sim:.2f}", bg=CARD, fg=col, font=(FONT, 11, "bold")).pack(side="right")
        body = tk.Frame(card, bg=CARD)
        body.pack(fill="x", padx=8)
        img = cv2.imdecode(np.frombuffer(e["thumb"], np.uint8), cv2.IMREAD_COLOR)
        ph = _photo(img, (56, 112))
        self._imgs.append(ph)
        tk.Label(body, image=ph, bg=CARD).pack(side="left")
        info = tk.Frame(body, bg=CARD)
        info.pack(side="left", fill="both", expand=True, padx=(8, 0))
        rows = [(f"📹 {e['camera']}", FG), (f"{e['date']}", FG_DIM), (f"🕒 {e['t0']} – {e['t1']}", FG),
                (f"P(Known) {e['p_known']:.2f}", FG_DIM)]
        if e.get("gt", -1) >= 0:
            rows.append((f"GT: CHIRLA id {e['gt']}", ACCENT2))
        for txt, c in rows:
            tk.Label(info, text=txt, bg=CARD, fg=c, font=(FONT, 8), anchor="w").pack(fill="x")
        bar = tk.Canvas(card, height=4, bg=CARD2, highlightthickness=0)
        bar.pack(fill="x", padx=8, pady=(6, 4))
        bar.bind("<Configure>", lambda ev, s=r.sim: (bar.delete("all"),
                                                     bar.create_rectangle(0, 0, ev.width * s, 4, fill=col,
                                                                          outline="")))
        button(card, "▶ Play from here", lambda: self.app.play_from(e["video"], max(0, e["f0"] - 30)),
               CARD2, fg=FG).pack(fill="x", padx=8, pady=(0, 8))
        return card
