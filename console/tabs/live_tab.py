"""
console/tabs/live_tab.py
────────────────────────────────────────────────────────────────────────────────
Live Monitor tab: video canvas with mouse-drag asset anchor drawing, 6 live stat
tiles, people-in-view table (flag suspect / search person), and live event feed.
────────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import time
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageTk

from console.mosaic import compose, tile_at
from console.theme import (ACCENT, ACCENT2, BG, BORDER, CARD, CARD2, CYAN,
                           DANGER, FG, FG_DIM, FONT, WARN, StatTile,
                           append_text, button, clear_text)


class LiveTab(tk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, bg=BG)
        self.app = app
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(self, bg="#05070b", bd=0, highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew", padx=(6, 6), pady=6)
        self.canvas.bind("<Configure>", lambda e: self.render_multi() if self.multi is not None
                         else (not self.app._running) and self.draw_placeholder())
        self.canvas.bind("<ButtonPress-1>", self._canvas_press)
        self.canvas.bind("<B1-Motion>", self._canvas_motion)
        self.canvas.bind("<ButtonRelease-1>", self._canvas_release)
        self._drawing = False
        self._drag = None
        self._disp = None
        self._preview = None
        self._last_people_update = 0.0
        self._keep_sel = None
        self._rows: dict = {}            # people-table row → (camera index, track id)
        # Several cameras: {"labels", "frames", "ended", "rects"}; focus = enlarged camera index
        self.multi: Optional[dict] = None
        self.focus: Optional[int] = None
        self.canvas.bind("<Escape>", lambda e: self.set_focus(None))

        side = tk.Frame(self, bg=BG, width=310)
        side.grid(row=0, column=1, sticky="ns", pady=6, padx=(0, 6))
        side.grid_propagate(False)
        side.columnconfigure((0, 1), weight=1)
        side.rowconfigure(6, weight=1)

        self.tiles = {}
        tiles = tk.Frame(side, bg=BG)
        tiles.grid(row=0, column=0, columnspan=2, rowspan=2, sticky="ew")
        tiles.columnconfigure((0, 1, 2), weight=1)
        for i, (key, cap, col) in enumerate([
            ("fps", "FPS", ACCENT2), ("known", "KNOWN", ACCENT),
            ("unknown", "UNKNOWN", WARN), ("alerts", "ALERTS", DANGER),
            ("suspects", "SUSPECTS", DANGER), ("tracks", "TRACKS", FG_DIM),
        ]):
            t = StatTile(tiles, cap, col)
            t.grid(row=i // 3, column=i % 3, sticky="ew", padx=2, pady=(0, 4))
            self.tiles[key] = t
        self.tiles["suspects"].set(len(self.app.engine.suspects))

        self.cam_label = tk.Label(side, text="No source", bg=CARD, fg=FG, font=(FONT, 9),
                                  anchor="w", padx=10, pady=6)
        self.cam_label.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(0, 6))

        tk.Label(side, text="PEOPLE IN VIEW", bg=BG, fg=FG_DIM, font=(FONT, 8, "bold"),
                 anchor="w").grid(row=3, column=0, columnspan=2, sticky="ew")
        cols = ("cam", "id", "who", "p", "status")
        self.people = ttk.Treeview(side, columns=cols, show="headings", height=6)
        for c, w, t in zip(cols, (50, 40, 90, 60, 80), ("CAM", "ID", "IDENTITY", "P(KNOWN)", "STATUS")):
            self.people.heading(c, text=t)
            self.people.column(c, width=w, anchor="w", stretch=c == "who")
        for tag, col in (("KNOWN", ACCENT), ("UNKNOWN", WARN), ("PENDING", FG_DIM),
                         ("INTRUSION", DANGER), ("SUSPECT", DANGER)):
            self.people.tag_configure(tag, foreground=col)
        self.people.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(2, 2))

        acts = tk.Frame(side, bg=BG)
        acts.grid(row=5, column=0, columnspan=2, sticky="ew", pady=(0, 6))
        acts.columnconfigure((0, 1), weight=1)
        button(acts, "🚩 Flag suspect", self._flag_selected, CARD2, fg=FG).grid(
            row=0, column=0, sticky="ew", padx=(0, 2))
        button(acts, "🔎 Search person", self._search_selected, CARD2, fg=FG).grid(
            row=0, column=1, sticky="ew", padx=(2, 0))

        feed_hdr = tk.Frame(side, bg=BG)
        feed_hdr.grid(row=6, column=0, columnspan=2, sticky="new")
        tk.Label(feed_hdr, text="ALERTS & EVENTS", bg=BG, fg=FG_DIM, font=(FONT, 8, "bold")).pack(side="left")
        self.feed = tk.Text(side, bg=CARD, fg=FG, font=(FONT, 9), bd=0, wrap="word",
                            highlightthickness=0, padx=8, pady=6, state="disabled")
        self.feed.grid(row=6, column=0, columnspan=2, sticky="nsew", pady=(20, 0))
        for tag, col in (("alert", DANGER), ("identity", ACCENT), ("asset", CYAN), ("info", ACCENT2),
                         ("time", FG_DIM), ("unknown", WARN), ("suspect", DANGER)):
            self.feed.tag_configure(tag, foreground=col)
        self.feed.tag_configure("alert", font=(FONT, 9, "bold"))

    def clear_feed(self):
        clear_text(self.feed)

    def append_event(self, level: str, stamp: str, cam: str, text: str):
        if level == "info":
            tag = "info"
        elif level == "identity" and "UNKNOWN" in text:
            tag = "unknown"
        else:
            tag = level
        icon = {"alert": "🚨 ", "identity": "👤 ", "asset": "💼 ", "info": "ℹ ", "suspect": "🕵 "}.get(level, "")
        append_text(self.feed, [(f"{stamp} {cam}\n", "time"), (f"{icon}{text}\n\n", tag)],
                    max_lines=400, top=True)

    def draw_placeholder(self):
        c = self.canvas
        c.delete("all")
        w, h = c.winfo_width() or 800, c.winfo_height() or 500
        c.create_text(w // 2, h // 2 - 40, text="◆", fill=BORDER, font=(FONT, 48))
        c.create_text(w // 2, h // 2 + 20, text="No video", fill=FG_DIM, font=(FONT, 14, "bold"))
        c.create_text(w // 2, h // 2 + 56, fill=FG_DIM, font=(FONT, 10), justify="center",
                      text="Pick a demo clip, your own video (with its recording date) or a recorded day"
                           "\n(all its cameras together) on the left, then press  ▶ START")

    def display_frame(self, bgr: np.ndarray):
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

    def update_stats(self, stats: dict):
        self.tiles["fps"].set(f"{stats.get('fps', 0):.1f}")
        for k in ("known", "unknown", "alerts"):
            self.tiles[k].set(stats.get(k, "—"))
        self.tiles["tracks"].set(stats.get("tracks", "—"))
        if stats.get("camera"):
            assets = ", ".join(f"{n}: {s}" for n, s in stats.get("assets", [])) or "no assets in view"
            self.cam_label.config(text=f"📹 {stats['camera']}   🕒 {stats.get('time', '')}   ·   {assets}")

        now = time.time()
        if now - self._last_people_update < 0.25:
            return
        self._last_people_update = now
        sel = self.people.selection()
        self._keep_sel = self._rows.get(sel[0]) if sel else None
        self.people.delete(*self.people.get_children())
        self._rows = {}
        for p in sorted(stats.get("people", []), key=lambda p: (p.get("cam", 0), p["tid"])):
            status = p["status"]
            tag = "INTRUSION" if p["intrusion"] else status
            shown = "PENDING" if status == "PENDING" else status
            if p.get("carrying"):
                shown += " 💼"
            if p["zone"]:
                shown += " ⚠" if p["intrusion"] else " ·zone"
            pk = f"{p['confidence']:.2f}" if status != "PENDING" else f"{p['crops']} crops"
            key = (p.get("cam", 0), p["tid"])
            cam_id = p.get("cam_id") or stats.get("camera", "")
            iid = self.people.insert("", "end", values=(cam_id.replace("Cam_", "C"), f"T{p['tid']}", p["name"],
                                                        pk, shown), tags=(tag,))
            self._rows[iid] = key
            if key == self._keep_sel:
                self.people.selection_set(iid)

    # ── Several cameras: grid + click-to-enlarge ──────────────────────────────

    def start_multi(self, labels):
        self.multi = {"labels": list(labels), "frames": [None] * len(labels),
                      "ended": [False] * len(labels), "rects": []}
        self.focus = None
        self.render_multi()

    def end_multi(self):
        self.multi = None
        self.focus = None

    def show_multi(self, payload: dict):
        m = self.multi
        if m is None:
            return
        m["frames"], m["ended"] = payload["frames"], payload["ended"]
        m["labels"] = payload.get("labels") or m["labels"]      # only the cameras that opened
        if self.focus is not None and self.focus >= len(m["frames"]):
            self.focus = None
        self.render_multi()

    def set_focus(self, index: Optional[int]):
        if self.multi is None or self._drawing:
            return
        self.focus = index
        self.render_multi()
        self.app.on_focus_changed()

    def render_multi(self):
        m = self.multi
        if m is None:
            return
        f = self.focus
        if f is not None and m["frames"][f] is not None:
            self.display_frame(m["frames"][f])
            if not self._drawing:
                w = self.canvas.winfo_width()
                self.canvas.create_rectangle(0, 0, w, 26, fill="#05070b", outline="")
                self.canvas.create_text(w // 2, 13, fill=FG, font=(FONT, 9, "bold"),
                                        text=f"{m['labels'][f]}{' · ENDED' if m['ended'][f] else ''}   ·   "
                                             f"click (or Esc) to return to all {len(m['labels'])} cameras")
            return
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        if cw < 2 or ch < 2:
            return
        mosaic, m["rects"] = compose(m["frames"], m["labels"], cw, ch, m["ended"])
        self.display_frame(mosaic)

    # ── Asset anchor drawing ──────────────────────────────────────────────────

    def begin_anchor(self, draw_btn: tk.Button):
        """Next mouse drag on the video defines a laptop home anchor for this camera."""
        if self.multi is not None:
            if self.focus is None:
                messagebox.showinfo("Draw anchor", "Click a camera to enlarge it first, then draw its anchor.")
                return
        elif self.app.source_mode() == "day":
            messagebox.showinfo("Draw anchor", "Start the recorded day, click a camera to enlarge it, "
                                               "then draw its anchor.")
            return
        elif not self.app._running:
            source = self.app._resolve_source()
            cap = cv2.VideoCapture(source)
            ok, frame = cap.read()
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            cap.release()
            if not ok:
                messagebox.showinfo("Draw anchor", "Pick a video source (or start playback) first.")
                return
            self.app.engine.set_source(source, fps, gallery_date=self.app._video_date)
            self._preview = frame
        self._drawing, self._drag = True, None
        draw_btn.config(bg=CYAN, fg=BG)
        self.app._set_status("Draw mode: drag a box around the laptop in the video")
        if not self.app._running and self._preview is not None:
            self.display_frame(self._preview)

    def _canvas_press(self, e):
        if self._drawing:
            self._drag = [e.x, e.y, e.x, e.y]
        elif self.multi is not None:
            self.canvas.focus_set()               # for Esc
            if self.focus is not None:
                self.set_focus(None)
            else:
                hit = tile_at(self.multi["rects"], e.x, e.y)
                if hit is not None:
                    self.set_focus(hit)

    def _canvas_motion(self, e):
        if self._drawing and self._drag is not None:
            self._drag[2:] = [e.x, e.y]
            if not self.app._running and self._preview is not None:
                self.display_frame(self._preview)

    def _canvas_release(self, e):
        if not self._drawing or self._drag is None or self._disp is None:
            return
        ox, oy, scale, w, h = self._disp
        x0, y0, x1, y1 = self._drag
        box = [int((min(x0, x1) - ox) / scale), int((min(y0, y1) - oy) / scale),
               int((max(x0, x1) - ox) / scale), int((max(y0, y1) - oy) / scale)]
        box = [max(0, min(w, box[0])), max(0, min(h, box[1])),
               max(0, min(w, box[2])), max(0, min(h, box[3]))]
        self._drawing, self._drag = False, None
        self.app._draw_btn.config(bg=CARD2, fg=FG)
        if box[2] - box[0] < 8 or box[3] - box[1] < 8:
            self.app._set_status("Anchor too small – try again")
            return
        engine = self.app.active_engine()
        name = engine.add_asset_anchor(box)
        self.app._set_status(f"✓ Anchor '{name}' saved for {engine._cam_key} at {box}")
        if not self.app._running and self._preview is not None:
            self.display_frame(self._preview)

    # ── People actions ────────────────────────────────────────────────────────

    def _selected(self):
        """(engine, track id) of the person selected in PEOPLE IN VIEW, or (None, None)."""
        sel = self.people.selection()
        if not sel or sel[0] not in self._rows:
            messagebox.showinfo("Select a person", "Select a person in the PEOPLE IN VIEW table first.")
            return None, None
        cam, tid = self._rows[sel[0]]
        return self.app.engine_at(cam), tid

    def _flag_selected(self):
        engine, tid = self._selected()
        if engine is None:
            return
        s = engine.flag_suspect(tid)
        if s is None:
            messagebox.showinfo("Flag suspect", f"T{tid} has no sharp crops yet – try again in a moment.")
            return
        self.app._set_status(f"🚩 {engine._camera_id} T{tid} flagged as {s.label}")
        self.app.suspects_tab.render(force=True)

    def _search_selected(self):
        engine, tid = self._selected()
        if engine is None:
            return
        embs, thumbs = engine.track_embeddings(tid)
        if len(embs) == 0:
            messagebox.showinfo("Search person", f"T{tid} has no sharp crops yet – try again in a moment.")
            return
        self.app.search_tab.query_live(tid, embs, thumbs, engine._camera_id)
