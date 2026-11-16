"""
console/tabs/suspects_tab.py
────────────────────────────────────────────────────────────────────────────────
Suspects tab: auto-enrolled suspect gallery, crops, forensic backtrack origin,
cross-camera sightings, play origin, and one-click CCTV search.
────────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox

import cv2

from console.theme import (ACCENT2, BG, CARD, CARD2, DANGER, FG, FG_DIM, FONT,
                           WARN, ScrollFrame, button, photo)


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

    def render(self, force: bool = False):
        sig = self.signature()
        if not force and sig == self._shown:
            return
        self._shown = sig
        self.scroll.clear()
        self._imgs = []
        g = self.gallery
        self.count.config(
            text=f"{len(g)} suspect(s) · auto-enrolled on unauthorised asset displacement or "
                 f"lingering, or flagged by the operator · {g.dir}"
        )
        if not len(g):
            tk.Label(
                self.scroll.inner, bg=BG, fg=FG_DIM, font=(FONT, 11), justify="center",
                text="No suspects yet.\n\nThey are enrolled automatically when an UNKNOWN person displaces "
                     "a monitored laptop\n(or lingers at it), or when you select a person in the Live "
                     "Monitor and press  🚩 Flag suspect."
            ).pack(pady=60)
            return
        for s in sorted(g.suspects.values(), key=lambda s: s.suspect_id):
            self._card(s)

    def _card(self, s):
        card = tk.Frame(self.scroll.inner, bg=CARD, highlightthickness=1, highlightbackground=DANGER)
        card.pack(fill="x", padx=6, pady=5)
        head = tk.Frame(card, bg=CARD)
        head.pack(fill="x", padx=12, pady=(8, 2))
        tk.Label(head, text=f"🕵 {s.label}", bg=CARD, fg=DANGER, font=(FONT, 12, "bold")).pack(side="left")
        reason = {
            "displacement": "unauthorised asset displacement",
            "unknown_interaction": "unknown person lingering at an asset",
            "manual": "flagged by operator",
        }.get(s.reason, s.reason)
        tk.Label(
            head, text=f"   enrolled {s.created} · {s.camera} · {reason} · {s.n_crops} crops",
            bg=CARD, fg=FG_DIM, font=(FONT, 9)
        ).pack(side="left")
        for txt, cmd, col in (
            ("Delete", lambda sid=s.suspect_id: self.delete(sid), CARD2),
            ("▶ Play origin", lambda sus=s: self.play_origin(sus), CARD2),
            ("🔎 Search all CCTV", lambda sid=s.suspect_id: self.app.search_tab.query_suspect(sid), ACCENT2),
        ):
            button(head, txt, cmd, col, fg=FG if col == CARD2 else BG).pack(side="right", padx=3)

        body = tk.Frame(card, bg=CARD)
        body.pack(fill="x", padx=12, pady=(4, 10))
        crops = tk.Frame(body, bg=CARD)
        crops.pack(side="left", anchor="n")
        for f in self.gallery.crops(s.suspect_id)[:8]:
            img = cv2.imread(str(f))
            if img is None:
                continue
            ph = photo(img, (48, 96))
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
        tk.Label(info, text=f"RE-ID SIGHTINGS ({len(s.sightings)})", bg=CARD, fg=FG_DIM,
                 font=(FONT, 8, "bold"), anchor="w").pack(fill="x", pady=(6, 0))
        if s.sightings:
            for x in s.sightings[-5:][::-1]:
                tk.Label(
                    info, text=f"• {x.get('time')} · {x.get('camera')} · T{x.get('track')} · "
                               f"similarity {x.get('sim', 0):.2f} · {x.get('source', '')}",
                    bg=CARD, fg=WARN, font=(FONT, 9), anchor="w"
                ).pack(fill="x")
        else:
            tk.Label(
                info, text="none yet – play other cameras / clips, or use 🔎 Search all CCTV",
                bg=CARD, fg=FG_DIM, font=(FONT, 9), anchor="w"
            ).pack(fill="x")

    def play_origin(self, s):
        src = (s.entry or {}).get("source") or s.source
        path = self.app.find_video(src)
        if path is None:
            messagebox.showinfo("Not found", f"Cannot locate the source video:\n{src}")
            return
        self.app.play_from(str(path), max(0, int((s.entry or {}).get("frame", s.frame)) - 30))

    def delete(self, sid: str):
        self.gallery.delete(sid)
        self.render(force=True)
        self.app.search_tab.refresh_queries()

    def clear_all(self):
        if len(self.gallery) and messagebox.askyesno("Clear suspects", "Delete ALL suspect profiles?"):
            self.gallery.clear()
            self.render(force=True)
            self.app.search_tab.refresh_queries()
