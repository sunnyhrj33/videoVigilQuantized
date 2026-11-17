"""
console/alerts.py
────────────────────────────────────────────────────────────────────────────────
Operator alerting for the Live Monitor:
  • classify_event – engine event → (feed category, colour tag, needs attention?)
  • AlertCenter    – open (unacknowledged) alerts, sound with rate limit and
                     reminders, a flashing strip, acknowledge → audit CSV
────────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import logging
import time
import tkinter as tk
from collections import deque
from typing import Callable, Optional, Set

from console.theme import DANGER, FG, FONT, FONT_SEMI, IS_WIN, PANEL, button, caption, mix
from console.icons import icon

logger = logging.getLogger("console.alerts")

# Asset texts that mean "something is wrong" even though the engine logs them as asset events
_ASSET_ATTENTION = ("ABANDONED", "MISSING", "moved", "displaced by")


def classify_event(level: str, text: str):
    """→ (category, tag, attention). Categories: alert · asset · people · info."""
    if level in ("alert", "suspect"):
        return "alert", "alert", True
    if level == "asset":
        attention = any(k in text for k in _ASSET_ATTENTION)
        return "asset", "warn" if attention else "asset", False
    if level == "identity":
        return "people", "unknown" if "UNKNOWN" in text else "identity", False
    return "info", "info", False


class AlertCenter(tk.Frame):
    """
    Strip shown above the event feed while alerts are open: count, newest alert,
    Acknowledge (A) and Mute. Sound plays at most every `min_gap` s and repeats every
    `remind` s until acknowledged. `on_change(n_open, cams)` lets the shell flash the
    header and the camera tiles.
    """

    def __init__(self, parent, on_ack: Callable[[list], None], on_change: Callable[[int, Set[int]], None],
                 min_gap: float = 3.0, remind: float = 20.0):
        super().__init__(parent, bg=PANEL, highlightthickness=1, highlightbackground=DANGER)
        self.open: deque = deque(maxlen=100)          # (stamp, cam label, cam index, text)
        self.muted = False
        self._on_ack, self._on_change = on_ack, on_change
        self._min_gap, self._remind = min_gap, remind
        self._last_sound = 0.0
        self._phase = False
        self._visible = False

        head = tk.Frame(self, bg=PANEL)
        head.pack(fill="x", padx=10, pady=(8, 2))
        tk.Label(head, image=icon("alert", DANGER, 16), bg=PANEL).pack(side="left")
        self.count = tk.Label(head, text="", bg=PANEL, fg=DANGER, font=(FONT_SEMI, 10))
        self.count.pack(side="left", padx=(6, 0))
        self.mute_btn = button(head, "", self.toggle_mute, icon="bell")
        self.mute_btn.pack(side="right")
        self.text = tk.Label(self, text="", bg=PANEL, fg=FG, font=(FONT, 9), anchor="w", justify="left",
                             wraplength=280)
        self.text.pack(fill="x", padx=10)
        self.meta = caption(self, "", bg=PANEL)
        self.meta.pack(fill="x", padx=10, pady=(2, 6))
        button(self, "Acknowledge  ·  A", self.acknowledge, DANGER, icon="check").pack(fill="x", padx=10,
                                                                                      pady=(0, 10))
        self._tick()

    # ── public ───────────────────────────────────────────────────────────────

    def push(self, stamp: str, cam_label: str, cam_index: Optional[int], text: str):
        self.open.append((stamp, cam_label, cam_index, text))
        self._render()
        self._beep(force=True)

    def acknowledge(self):
        if not self.open:
            return
        items = list(self.open)
        self.open.clear()
        self._on_ack(items)
        logger.info("Operator acknowledged %d alert(s)", len(items))
        self._render()

    def toggle_mute(self):
        self.muted = not self.muted
        self.mute_btn._icon = "bell-off" if self.muted else "bell"
        self.mute_btn._paint()

    def cameras(self) -> Set[int]:
        return {c for _, _, c, _ in self.open if c is not None}

    # ── internals ────────────────────────────────────────────────────────────

    def _render(self):
        n = len(self.open)
        if n:
            stamp, cam, _, text = self.open[-1]
            self.count.config(text=f"{n} open alert{'s' if n > 1 else ''}")
            self.text.config(text=text)
            self.meta.config(text=f"{stamp}  ·  {cam}".upper())
            if not self._visible:
                self.grid()
                self._visible = True
        elif self._visible:
            self.grid_remove()
            self._visible = False
        self._on_change(n, self.cameras())

    def _beep(self, force: bool = False):
        now = time.time()
        if self.muted or now - self._last_sound < (self._min_gap if force else self._remind):
            return
        self._last_sound = now
        try:
            if IS_WIN:
                import winsound
                winsound.PlaySound("SystemHand", winsound.SND_ALIAS | winsound.SND_ASYNC)
            else:
                self.bell()
        except Exception:
            self.bell()

    def _tick(self):
        """500 ms flash of the strip border while alerts are open + reminder sound."""
        if self.open:
            self._phase = not self._phase
            self.config(highlightbackground=DANGER if self._phase else mix(DANGER, PANEL, 0.6))
            self._beep()
        self.after(500, self._tick)
