"""
console/tabs/live_tab.py
────────────────────────────────────────────────────────────────────────────────
Live Monitor tab: video canvas (single camera or click-to-enlarge grid) with
mouse-drag asset anchor drawing, live stat tiles, the open-alert strip, the
people-in-view table (flag suspect / search person) and a filterable event feed.
────────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import time
import tkinter as tk
from collections import deque
from tkinter import ttk
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageTk

from console.alerts import AlertCenter, classify_event
from console.icons import icon
from console.mosaic import compose, tile_at
from console.theme import (ACCENT, ACCENT_HOVER, BG, BG2, BORDER, DANGER, FG, FG_DIM, FG_MUTED, FONT,
                           FONT_SEMI, LABEL, MONO, OK, PANEL, WARN, StatTile, append_text, button, caption,
                           clear_text, dialogs)

FILTERS = [("priority", "Priority"), ("alert", "Alerts"), ("people", "People"), ("all", "All")]
SHOWN = {"priority": {"alert", "asset", "info"}, "alert": {"alert"}, "people": {"people"},
         "all": {"alert", "asset", "people", "info"}}
FEED_COLOURS = {"alert": DANGER, "warn": WARN, "asset": ACCENT_HOVER, "identity": OK, "unknown": WARN,
                "info": FG_DIM, "time": FG_MUTED}


class LiveTab(tk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, bg=BG)
        self.app = app
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(self, bg=BG2, bd=0, highlightthickness=1, highlightbackground=BORDER)
        self.canvas.grid(row=0, column=0, sticky="nsew", padx=(8, 8), pady=8)
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
        self._rows: dict = {}            # people-table row id → (camera index, track id)
        self._events: deque = deque(maxlen=400)
        # Several cameras: {"labels", "frames", "ended", "rects"}; focus = enlarged camera index
        self.multi: Optional[dict] = None
        self.focus: Optional[int] = None
        self.alert_cams: set = set()

        side = tk.Frame(self, bg=BG, width=330)
        side.grid(row=0, column=1, sticky="ns", pady=8, padx=(0, 8))
        side.grid_propagate(False)
        side.columnconfigure(0, weight=1)
        side.rowconfigure(7, weight=1)

        self.tiles = {}
        tiles = tk.Frame(side, bg=BG)
        tiles.grid(row=0, column=0, sticky="ew")
        tiles.columnconfigure((0, 1, 2), weight=1, uniform="tile")
        for i, (key, cap, col) in enumerate([
            ("fps", "FPS", FG), ("known", "Staff", OK), ("unknown", "Unknown", WARN),
            ("alerts", "Alerts", DANGER), ("suspects", "Suspects", DANGER), ("tracks", "Tracks", FG_DIM),
        ]):
            t = StatTile(tiles, cap, col)
            t.grid(row=i // 3, column=i % 3, sticky="ew", padx=(0 if i % 3 == 0 else 4, 0), pady=(0, 4))
            self.tiles[key] = t
        self.tiles["suspects"].set(len(self.app.engine.suspects))

        self.cam_label = tk.Label(side, text="NO SOURCE", bg=PANEL, fg=FG_DIM, font=LABEL, anchor="w",
                                  justify="left", padx=10, pady=7, wraplength=300,
                                  highlightthickness=1, highlightbackground=BORDER)
        self.cam_label.grid(row=1, column=0, sticky="ew", pady=(0, 8))

        self.alerts = AlertCenter(side, on_ack=self.app.on_alerts_acknowledged, on_change=self._alerts_changed)
        self.alerts.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        self.alerts.grid_remove()

        caption(side, "People in view", bg=BG).grid(row=3, column=0, sticky="ew", pady=(0, 4))
        cols = ("cam", "id", "who", "p", "status")
        self.people = ttk.Treeview(side, columns=cols, show="headings", height=6, selectmode="browse")
        for c, w, t in zip(cols, (40, 40, 90, 80, 70), ("CAM", "ID", "IDENTITY", "CERTAINTY", "STATUS")):
            self.people.heading(c, text=t, anchor="w")
            self.people.column(c, width=w, anchor="w", stretch=c == "who")
        for tag, col in (("KNOWN", OK), ("UNKNOWN", WARN), ("PENDING", FG_DIM),
                         ("INTRUSION", DANGER), ("SUSPECT", DANGER)):
            self.people.tag_configure(tag, foreground=col)
        self.people.grid(row=4, column=0, sticky="ew")

        acts = tk.Frame(side, bg=BG)
        acts.grid(row=5, column=0, sticky="ew", pady=(6, 10))
        acts.columnconfigure((0, 1), weight=1, uniform="act")
        button(acts, "Flag suspect", self._flag_selected, icon="flag").grid(row=0, column=0, sticky="ew", padx=(0, 3))
        button(acts, "Search person", self._search_selected, icon="search").grid(row=0, column=1, sticky="ew",
                                                                                padx=(3, 0))

        feed_hdr = tk.Frame(side, bg=BG)
        feed_hdr.grid(row=6, column=0, sticky="ew", pady=(0, 4))
        caption(feed_hdr, "Events", bg=BG).pack(side="left")
        self._filter = "priority"
        self._filter_btns = {}
        for key, text in reversed(FILTERS):
            b = button(feed_hdr, text, lambda k=key: self.set_filter(k), size=8)
            b.config(padx=7, pady=2)
            b.pack(side="right", padx=(3, 0))
            self._filter_btns[key] = b
        self._paint_filters()

        self.feed = tk.Text(side, bg=PANEL, fg=FG, font=(FONT, 9), bd=0, wrap="word", highlightthickness=1,
                            highlightbackground=BORDER, padx=10, pady=8, state="disabled", spacing1=1)
        self.feed.grid(row=7, column=0, sticky="nsew")
        for tag, col in FEED_COLOURS.items():
            self.feed.tag_configure(tag, foreground=col)
        self.feed.tag_configure("time", font=(MONO, 8), spacing1=6)
        self.feed.tag_configure("alert", font=(FONT_SEMI, 9))

    # ── Event feed & alerts ───────────────────────────────────────────────────

    def clear_feed(self):
        self._events.clear()
        clear_text(self.feed)

    def append_event(self, level: str, stamp: str, cam: str, text: str, cam_index: Optional[int] = None):
        category, tag, attention = classify_event(level, text)
        item = (category, tag, stamp, cam, text)
        self._events.append(item)
        if attention:
            self.alerts.push(stamp, cam, cam_index, text)
        if self._shown(item):
            append_text(self.feed, self._chunks(item), max_lines=400, top=True)

    def set_filter(self, key: str):
        self._filter = key
        self._paint_filters()
        clear_text(self.feed)
        chunks = [c for item in reversed(self._events) if self._shown(item) for c in self._chunks(item)]
        append_text(self.feed, chunks, max_lines=400, top=True)      # newest first
        self.feed.see("1.0")

    def _shown(self, item) -> bool:
        category, tag = item[0], item[1]
        return category in SHOWN[self._filter] or (self._filter == "alert" and tag == "warn")

    @staticmethod
    def _chunks(item):
        _, tag, stamp, cam, text = item
        return [(f"{stamp}  {cam}\n".upper(), "time"), (f"{text}\n", tag)]

    def _paint_filters(self):
        for key, b in self._filter_btns.items():
            b.set_active(key == self._filter)

    def _alerts_changed(self, n_open: int, cams: set):
        self.alert_cams = cams
        self.tiles["alerts"].alarm(n_open > 0)
        self.app.on_alerts_changed(n_open)
        if self.multi is not None and not self.app._running:
            self.render_multi()

    # ── Video canvas ──────────────────────────────────────────────────────────

    def draw_placeholder(self):
        c = self.canvas
        c.delete("all")
        w, h = c.winfo_width() or 800, c.winfo_height() or 500
        c.create_image(w // 2, h // 2 - 46, image=icon("monitor", FG_MUTED, 40))
        c.create_text(w // 2, h // 2 + 4, text="NO SOURCE", fill=FG_DIM, font=LABEL)
        c.create_text(w // 2, h // 2 + 36, fill=FG_MUTED, font=(FONT, 9), justify="center",
                      text="Choose a demo clip, a video or webcam, or a recorded day in the sidebar,\n"
                           "then press START  (Space)")

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
            self.canvas.create_rectangle(x0, y0, x1, y1, outline=ACCENT_HOVER, width=2, dash=(6, 4))
        if self._drawing:
            self._overlay_bar("DRAW ANCHOR  ·  drag a box around the laptop  ·  Esc to cancel", ACCENT)

    def _overlay_bar(self, text: str, edge: str = BORDER):
        w = self.canvas.winfo_width()
        self.canvas.create_rectangle(0, 0, w, 28, fill=BG, outline="")
        self.canvas.create_line(0, 28, w, 28, fill=edge)
        self.canvas.create_text(12, 14, anchor="w", fill=FG, font=LABEL, text=text.upper())

    def update_stats(self, stats: dict):
        self.tiles["fps"].set(f"{stats.get('fps', 0):.1f}")
        for k in ("known", "unknown", "alerts"):
            self.tiles[k].set(stats.get(k, "—"))
        self.tiles["tracks"].set(stats.get("tracks", "—"))
        if stats.get("camera"):
            assets = "   ".join(f"{n} {s}" for n, s in stats.get("assets", [])) or "no assets in view"
            self.cam_label.config(text=f"{stats['camera']}   {stats.get('time', '')}\n{assets}".upper(), fg=FG)

        now = time.time()
        if now - self._last_people_update < 0.25:
            return
        self._last_people_update = now
        seen = set()
        for p in sorted(stats.get("people", []), key=lambda p: (p.get("cam", 0), p["tid"])):
            status = p["status"]
            shown = status
            if p.get("carrying"):
                shown += " · ASSET"
            if p["zone"]:
                shown += " · ZONE" if not p["intrusion"] else " · INTRUSION"
            certainty = f"{p['confidence']:.2f}" if status != "PENDING" else f"{p['crops']} views"
            key = (p.get("cam", 0), p["tid"])
            iid = f"{key[0]}:{key[1]}"
            cam_id = (p.get("cam_id") or stats.get("camera", "")).replace("Cam_", "C")
            values = (cam_id, f"T{p['tid']}", p["name"], certainty, shown)
            tags = ("INTRUSION" if p["intrusion"] else status,)
            if self.people.exists(iid):
                self.people.item(iid, values=values, tags=tags)
            else:
                self.people.insert("", "end", iid=iid, values=values, tags=tags)
            self._rows[iid] = key
            seen.add(iid)
        gone = [i for i in self.people.get_children() if i not in seen]
        if gone:
            self.people.delete(*gone)
            for i in gone:
                self._rows.pop(i, None)

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
                state = " · ALERT" if f in self.alert_cams else " · ENDED" if m["ended"][f] else ""
                self._overlay_bar(f"{m['labels'][f]}{state}   ·   click or Esc: back to all "
                                  f"{len(m['labels'])} cameras", DANGER if state == " · ALERT" else BORDER)
            return
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        if cw < 2 or ch < 2:
            return
        mosaic, m["rects"] = compose(m["frames"], m["labels"], cw, ch, m["ended"], self.alert_cams)
        self.display_frame(mosaic)

    # ── Asset anchor drawing ──────────────────────────────────────────────────

    def begin_anchor(self, draw_btn: tk.Button):
        """Toggle: the next mouse drag on the video defines a laptop home anchor for this camera."""
        if self._drawing:
            self.cancel_anchor()
            return
        if self.multi is not None:
            if self.focus is None:
                dialogs.showinfo("Draw anchor", "Click a camera to enlarge it first, then draw its anchor.")
                return
        elif self.app.source_mode() == "day":
            dialogs.showinfo("Draw anchor", "Start the recorded day, click a camera to enlarge it, "
                                            "then draw its anchor.")
            return
        elif not self.app._running:
            source = self.app._resolve_source()
            cap = cv2.VideoCapture(source)
            ok, frame = cap.read()
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            cap.release()
            if not ok:
                dialogs.showinfo("Draw anchor", "Pick a video source (or start playback) first.")
                return
            self.app.engine.set_source(source, fps, gallery_date=self.app._video_date)
            self._preview = frame
        self._drawing, self._drag = True, None
        draw_btn.set_active(True)
        self.app._set_status("Draw mode: drag a box around the laptop in the video · Esc cancels")
        self._redraw_still()

    def cancel_anchor(self) -> bool:
        if not self._drawing:
            return False
        self._drawing, self._drag = False, None
        self.app._draw_btn.set_active(False)
        self.app._set_status("Anchor drawing cancelled")
        self._redraw_still()
        return True

    def _redraw_still(self):
        if not self.app._running and self._preview is not None:
            self.display_frame(self._preview)

    def _canvas_press(self, e):
        self.canvas.focus_set()
        if self._drawing:
            self._drag = [e.x, e.y, e.x, e.y]
        elif self.multi is not None:
            if self.focus is not None:
                self.set_focus(None)
            else:
                hit = tile_at(self.multi["rects"], e.x, e.y)
                if hit is not None:
                    self.set_focus(hit)

    def _canvas_motion(self, e):
        if self._drawing and self._drag is not None:
            self._drag[2:] = [e.x, e.y]
            self._redraw_still()

    def _canvas_release(self, e):
        if not self._drawing or self._drag is None or self._disp is None:
            return
        ox, oy, scale, w, h = self._disp
        x0, y0, x1, y1 = self._drag
        box = [int((min(x0, x1) - ox) / scale), int((min(y0, y1) - oy) / scale),
               int((max(x0, x1) - ox) / scale), int((max(y0, y1) - oy) / scale)]
        box = [max(0, min(w, box[0])), max(0, min(h, box[1])),
               max(0, min(w, box[2])), max(0, min(h, box[3]))]
        if box[2] - box[0] < 8 or box[3] - box[1] < 8:
            self._drag = None
            self.app._set_status("Anchor too small – drag a larger box, or press Esc to cancel")
            self._redraw_still()
            return
        self._drawing, self._drag = False, None
        self.app._draw_btn.set_active(False)
        engine = self.app.active_engine()
        name = engine.add_asset_anchor(box)
        self.app._set_status(f"✓ Anchor “{name}” saved for {engine._camera_id}")
        self._redraw_still()

    # ── People actions ────────────────────────────────────────────────────────

    def _selected(self):
        """(engine, track id) of the person selected in PEOPLE IN VIEW, or (None, None)."""
        sel = self.people.selection()
        if not sel or sel[0] not in self._rows:
            dialogs.showinfo("Select a person", "Select a person in the People in view table first.")
            return None, None
        cam, tid = self._rows[sel[0]]
        return self.app.engine_at(cam), tid

    def _flag_selected(self):
        engine, tid = self._selected()
        if engine is None:
            return
        s = engine.flag_suspect(tid)
        if s is None:
            dialogs.showinfo("Flag suspect", f"T{tid} has no clear views yet – try again in a moment.")
            return
        self.app._set_status(f"{engine._camera_id} T{tid} flagged as {s.label}")
        self.app.suspects_tab.render(force=True)

    def _search_selected(self):
        engine, tid = self._selected()
        if engine is None:
            return
        embs, thumbs = engine.track_embeddings(tid)
        if len(embs) == 0:
            dialogs.showinfo("Search person", f"T{tid} has no clear views yet – try again in a moment.")
            return
        self.app.search_tab.query_live(tid, embs, thumbs, engine._camera_id)
