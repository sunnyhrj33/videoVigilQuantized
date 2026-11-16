"""
ui_theme.py – shared fonts, colours and small widget helpers for the console.
"""

from __future__ import annotations

import platform
import tkinter as tk
from tkinter import ttk

IS_WIN = platform.system() == "Windows"
FONT = "Segoe UI" if IS_WIN else "DejaVu Sans"
MONO = "Consolas" if IS_WIN else "DejaVu Sans Mono"

# ── Colour palette (dark theme) ───────────────────────────────────────────────
BG = "#0b0e14"
PANEL = "#121722"
CARD = "#19202e"
CARD2 = "#202839"
BORDER = "#283247"
FG = "#e6eaf0"
FG_DIM = "#7d889c"
ACCENT = "#00e5a0"      # known / OK
ACCENT2 = "#3b9eff"     # info / actions
WARN = "#ff9f1c"        # unknown
DANGER = "#ff4d6d"      # alerts / suspects
CYAN = "#22d3ee"        # assets
PURPLE = "#a78bfa"      # pipeline


def button(parent, text, cmd, colour, fg=BG, **kw) -> tk.Button:
    return tk.Button(parent, text=text, command=cmd, bg=colour, fg=fg, activebackground=FG,
                     activeforeground=BG, font=(FONT, 9, "bold"), bd=0, relief="flat",
                     padx=10, pady=6, cursor="hand2", disabledforeground="#3d4658", **kw)


def combo(parent, var, values, width=26) -> ttk.Combobox:
    return ttk.Combobox(parent, textvariable=var, values=values, state="readonly",
                        font=(FONT, 10), width=width)


def section(parent, text, bg=PANEL):
    tk.Label(parent, text=text, bg=bg, fg=FG_DIM, font=(FONT, 8, "bold"),
             anchor="w").pack(fill="x", padx=16, pady=(12, 4))


class ScrollFrame(tk.Frame):
    """Vertically scrollable frame (mouse wheel while hovered)."""

    def __init__(self, parent, bg=PANEL, **kw):
        super().__init__(parent, bg=bg, **kw)
        self.canvas = tk.Canvas(self, bg=bg, highlightthickness=0, bd=0)
        self.vsb = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview,
                                 style="Dark.Vertical.TScrollbar")
        self.inner = tk.Frame(self.canvas, bg=bg)
        self._win = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.canvas.configure(yscrollcommand=self.vsb.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.vsb.pack(side="right", fill="y")
        self.inner.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfigure(self._win, width=e.width))
        self.bind("<Enter>", lambda e: self.bind_all("<MouseWheel>", self._wheel))
        self.bind("<Leave>", lambda e: self.unbind_all("<MouseWheel>"))

    def _wheel(self, e):
        self.canvas.yview_scroll(int(-e.delta / 120), "units")

    def clear(self):
        for w in self.inner.winfo_children():
            w.destroy()
        self.canvas.yview_moveto(0)
