"""
console/theme.py
────────────────────────────────────────────────────────────────────────────────
Shared dark-theme palette, typography, and reusable Tkinter widget helpers for
the operator console and its tabs.
────────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import platform
import tkinter as tk
from tkinter import ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

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


def button(parent, text: str, cmd, colour: str, fg: str = BG, **kw) -> tk.Button:
    return tk.Button(
        parent, text=text, command=cmd, bg=colour, fg=fg,
        activebackground=FG, activeforeground=BG, font=(FONT, 9, "bold"),
        bd=0, relief="flat", padx=10, pady=6, cursor="hand2",
        disabledforeground="#3d4658", **kw
    )


def combo(parent, var, values, width: int = 26) -> ttk.Combobox:
    return ttk.Combobox(
        parent, textvariable=var, values=values, state="readonly",
        font=(FONT, 10), width=width
    )


def section(parent, text: str, bg: str = PANEL, padx: int = 16):
    tk.Label(
        parent, text=text, bg=bg, fg=FG_DIM, font=(FONT, 8, "bold"), anchor="w"
    ).pack(fill="x", padx=padx, pady=(12, 4))


def labelled(parent, label: str, widget, bg: str = PANEL):
    f = tk.Frame(parent, bg=bg)
    f.pack(fill="x", padx=16, pady=2)
    tk.Label(f, text=label, bg=bg, fg=FG_DIM, font=(FONT, 9), width=8, anchor="w").pack(side="left")
    widget.pack(in_=f, side="left", fill="x", expand=True)
    widget.lift(f)


def slider(parent, label: str, var: tk.DoubleVar, lo: float, hi: float, on_change, bg: str = PANEL):
    f = tk.Frame(parent, bg=bg)
    f.pack(fill="x", padx=16, pady=(2, 0))
    tk.Label(f, text=label, bg=bg, fg=FG_DIM, font=(FONT, 9)).pack(side="left")
    val = tk.Label(f, text=f"{var.get():.2f}", bg=bg, fg=ACCENT, font=(FONT, 9, "bold"))
    val.pack(side="right")

    def changed(_):
        val.config(text=f"{var.get():.2f}")
        if on_change:
            on_change(var.get())

    tk.Scale(
        parent, from_=lo, to=hi, resolution=0.01, orient="horizontal", variable=var,
        bg=bg, fg=FG, troughcolor=CARD, activebackground=ACCENT, highlightthickness=0,
        bd=0, showvalue=False, sliderlength=18, width=8, command=changed
    ).pack(fill="x", padx=14)


def photo(img_bgr: np.ndarray, size=(64, 128)) -> ImageTk.PhotoImage:
    rgb = cv2.cvtColor(cv2.resize(img_bgr, size), cv2.COLOR_BGR2RGB)
    return ImageTk.PhotoImage(Image.fromarray(rgb))


def append_text(widget: tk.Text, chunks, max_lines: int = 2000, autoscroll: bool = True, top: bool = False):
    """Append (text, tag) chunks to a disabled Text widget with line capping."""
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


def clear_text(widget: tk.Text):
    widget.configure(state="normal")
    widget.delete("1.0", "end")
    widget.configure(state="disabled")


class StatTile(tk.Frame):
    def __init__(self, parent, caption: str, colour: str):
        super().__init__(parent, bg=CARD, highlightthickness=1, highlightbackground=BORDER)
        self.value = tk.Label(self, text="—", bg=CARD, fg=colour, font=(FONT, 20, "bold"))
        self.value.pack(anchor="w", padx=12, pady=(6, 0))
        tk.Label(self, text=caption, bg=CARD, fg=FG_DIM, font=(FONT, 8, "bold")).pack(
            anchor="w", padx=12, pady=(0, 8)
        )

    def set(self, v):
        self.value.config(text=str(v))


class Chip(tk.Label):
    def __init__(self, parent, text: str = "", colour: str = ACCENT2):
        super().__init__(parent, text=text, bg=CARD2, fg=colour, font=(FONT, 9, "bold"), padx=8, pady=3)


class ScrollFrame(tk.Frame):
    """Vertically scrollable frame (mouse wheel while hovered)."""

    def __init__(self, parent, bg: str = PANEL, **kw):
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
