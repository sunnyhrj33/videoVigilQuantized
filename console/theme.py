"""
console/theme.py
────────────────────────────────────────────────────────────────────────────────
"Industrial Intelligence" design system for the operator console: colour and
type tokens, bundled Inter / JetBrains Mono, ttk styling, and the reusable
widgets every tab builds from (buttons, chips, stat tiles, sections,
collapsible panels, themed dialogs, scroll frames).

Colour roles: blue = interaction · green / amber / red = operational state only.
────────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import ctypes
import platform
import tkinter as tk
from pathlib import Path
from tkinter import ttk
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageTk

from console.icons import icon

IS_WIN = platform.system() == "Windows"

# ── Colour tokens ─────────────────────────────────────────────────────────────
BG = "#0b0e14"            # midnight graphite (window)
BG2 = "#0d1119"           # secondary background (video well, logs, tables)
PANEL = "#131a25"         # primary surface (header, sidebar, dialogs)
CARD = "#192231"          # elevated surface (cards, inputs, buttons)
CARD2 = "#1f2a3c"         # elevated + hover
BORDER = "#262e3a"        # rgba(180,198,225,.13) over the surfaces
FG = "#e8edf5"            # primary text
FG_DIM = "#9aa7b8"        # secondary text
FG_MUTED = "#687587"      # muted text / technical labels
ACCENT = "#4d8dff"        # intelligence blue – the interaction colour
ACCENT_HOVER = "#76aaff"
ACTIVE = "#16253d"        # restrained blue tint for active / selected
OK = "#35c98b"            # operational green (known staff, success)
WARN = "#f2b34f"          # amber (unknown, attention)
DANGER = "#f05252"        # signal red (alerts, suspects)
# legacy names used by the tabs – mapped onto the roles above
ACCENT2 = ACCENT
CYAN = ACCENT_HOVER
PURPLE = ACCENT

# ── Typography: bundled Inter + JetBrains Mono (falls back to system fonts) ───
FONT_DIR = Path(__file__).resolve().parent / "assets" / "fonts"


def _load_bundled_fonts() -> bool:
    """Register the bundled .ttf files for this process only (no install needed)."""
    if not IS_WIN or not FONT_DIR.exists():
        return False
    files = list(FONT_DIR.glob("*.ttf"))
    return bool(files) and all(ctypes.windll.gdi32.AddFontResourceExW(str(f), 0x10, 0) for f in files)


if _load_bundled_fonts():
    FONT, FONT_SEMI, MONO, MONO_SEMI = "Inter", "Inter SemiBold", "JetBrains Mono Medium", "JetBrains Mono SemiBold"
elif IS_WIN:
    FONT, FONT_SEMI, MONO, MONO_SEMI = "Segoe UI", "Segoe UI Semibold", "Cascadia Mono", "Cascadia Mono SemiBold"
else:
    FONT = FONT_SEMI = "DejaVu Sans"
    MONO = MONO_SEMI = "DejaVu Sans Mono"

LABEL = (MONO_SEMI, 8)     # uppercase technical labels (≈11 px)
NUM = (MONO_SEMI, 17)      # counters – monospaced digits are tabular


def mix(a: str, b: str, t: float) -> str:
    """Blend hex colour a toward b by t (0–1)."""
    ca, cb = (tuple(int(c.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4)) for c in (a, b))
    return "#" + "".join(f"{round(x + (y - x) * t):02x}" for x, y in zip(ca, cb))


# ── Window chrome & ttk styling ───────────────────────────────────────────────

def dark_titlebar(win: tk.Misc):
    """Dark native title bar on Windows 10 (1809+) / 11; no-op elsewhere."""
    if not IS_WIN:
        return
    try:
        win.update_idletasks()
        hwnd = ctypes.windll.user32.GetParent(win.winfo_id())
        for attr in (20, 19):                    # DWMWA_USE_IMMERSIVE_DARK_MODE (new, old id)
            if ctypes.windll.dwmapi.DwmSetWindowAttribute(hwnd, attr, ctypes.byref(ctypes.c_int(1)), 4) == 0:
                break
    except Exception:
        pass


def apply_style(root: tk.Tk):
    """Global ttk styles + option defaults. Call once on the root window."""
    s = ttk.Style(root)
    s.theme_use("clam")
    s.configure(".", background=PANEL, foreground=FG, font=(FONT, 9), bordercolor=BORDER,
                focuscolor=ACCENT, troughcolor=BG2)
    s.configure("TNotebook", background=BG, borderwidth=1, tabmargins=(0, 6, 0, 0), bordercolor=BORDER,
                lightcolor=BG, darkcolor=BG)
    s.configure("TNotebook.Tab", background=PANEL, foreground=FG_DIM, padding=(14, 7), font=(FONT_SEMI, 9),
                borderwidth=1, bordercolor=BORDER, lightcolor=PANEL, focuscolor=PANEL)
    s.map("TNotebook.Tab", background=[("selected", CARD), ("active", CARD)],
          foreground=[("selected", FG), ("active", FG)], lightcolor=[("selected", ACCENT)],
          bordercolor=[("selected", BORDER)])
    s.configure("TCombobox", fieldbackground=CARD, background=CARD, foreground=FG, arrowcolor=FG_DIM,
                bordercolor=BORDER, lightcolor=CARD, darkcolor=CARD, padding=5, arrowsize=12)
    s.map("TCombobox", fieldbackground=[("readonly", CARD)], foreground=[("readonly", FG)],
          selectbackground=[("readonly", CARD)], selectforeground=[("readonly", FG)],
          bordercolor=[("focus", ACCENT), ("hover", FG_MUTED)], arrowcolor=[("hover", FG)])
    root.option_add("*TCombobox*Listbox.background", CARD)
    root.option_add("*TCombobox*Listbox.foreground", FG)
    root.option_add("*TCombobox*Listbox.selectBackground", ACTIVE)
    root.option_add("*TCombobox*Listbox.selectForeground", FG)
    root.option_add("*TCombobox*Listbox.font", (FONT, 9))
    s.configure("Treeview", background=BG2, fieldbackground=BG2, foreground=FG, rowheight=26,
                borderwidth=0, font=(FONT, 9))
    s.configure("Treeview.Heading", background=PANEL, foreground=FG_MUTED, relief="flat", font=LABEL,
                bordercolor=BORDER, lightcolor=PANEL, darkcolor=PANEL, padding=(6, 5))
    s.map("Treeview", background=[("selected", ACTIVE)], foreground=[("selected", FG)])
    s.map("Treeview.Heading", background=[("active", CARD)])
    for name, col in (("Green", OK), ("Purple", ACCENT), ("Blue", ACCENT)):
        s.configure(f"{name}.Horizontal.TProgressbar", troughcolor=BG2, background=col,
                    bordercolor=BG2, lightcolor=col, darkcolor=col, thickness=4)
    s.configure("Dark.Vertical.TScrollbar", background=CARD, troughcolor=BG2, bordercolor=BG2,
                arrowcolor=FG_MUTED, lightcolor=CARD, darkcolor=CARD, arrowsize=11)
    s.map("Dark.Vertical.TScrollbar", background=[("active", CARD2)])
    s.configure("TPanedwindow", background=BG)
    s.configure("Sash", sashthickness=4, background=BORDER)
    root.option_add("*Entry.selectBackground", ACTIVE)
    root.option_add("*Text.selectBackground", ACTIVE)
    dark_titlebar(root)


# ── Buttons ───────────────────────────────────────────────────────────────────

class Button(tk.Button):
    """
    Flat button. `colour` picks the variant: a surface colour (CARD/CARD2) → bordered
    secondary, anything else → filled (blue primary, red danger, green OK). Handles
    hover, the blue keyboard-focus ring, a visibly disabled state and `set_active`.
    """

    def __init__(self, parent, text: str, cmd, colour: str = CARD2, fg: Optional[str] = None,
                 icon_name: Optional[str] = None, size: int = 9, **kw):
        self._filled = colour not in (CARD, CARD2, PANEL, BG, BG2)
        self._bg = colour if self._filled else CARD
        self._fg = fg if fg and self._filled else (BG if self._filled else FG)
        self._icon, self._active = icon_name, False
        super().__init__(parent, text=text, command=cmd, bd=0, relief="flat", cursor="hand2",
                         font=(FONT_SEMI, size), padx=10, pady=5, highlightthickness=1,
                         highlightcolor=ACCENT_HOVER, takefocus=1, compound="left",
                         disabledforeground=FG_MUTED, **kw)
        self.bind("<Enter>", lambda e: self._paint(hover=True))
        self.bind("<Leave>", lambda e: self._paint())
        self._paint()

    def configure(self, cnf=None, **kw):
        out = super().configure(cnf, **kw)
        if "state" in kw:
            self._paint()
        return out

    config = configure

    def set_active(self, on: bool):
        """Selected / toggled-on look: blue tint + accent border (segmented controls, modes)."""
        self._active = on
        self._paint()

    def _paint(self, hover: bool = False):
        enabled = str(self["state"]) != "disabled"
        hover = hover and enabled
        if not enabled:
            bg, fg, edge = PANEL, FG_MUTED, BORDER
        elif self._active:
            bg, fg, edge = mix(ACTIVE, ACCENT, 0.12) if hover else ACTIVE, ACCENT_HOVER, ACCENT
        elif self._filled:
            bg = mix(self._bg, "#ffffff", 0.14) if hover else self._bg
            fg, edge = self._fg, bg
        else:
            bg, fg, edge = (CARD2, FG, FG_MUTED) if hover else (CARD, FG, BORDER)
        show_icon = bool(self._icon) and enabled          # Tk dithers images on disabled buttons
        img = icon(self._icon, fg, 16) if show_icon else ""
        label = self["text"].lstrip()
        super().configure(bg=bg, fg=fg, activebackground=bg, activeforeground=fg, highlightbackground=edge,
                          image=img, text=f" {label}" if show_icon and label else label)


def button(parent, text: str, cmd, colour: str = CARD2, fg: Optional[str] = None, icon: Optional[str] = None,
           **kw) -> Button:
    return Button(parent, text, cmd, colour, fg, icon_name=icon, **kw)


def check(parent, text: str, var, cmd=None, bg: str = PANEL) -> tk.Checkbutton:
    return tk.Checkbutton(parent, text=text, variable=var, command=cmd, bg=bg, fg=FG, selectcolor=CARD,
                          activebackground=bg, activeforeground=FG, font=(FONT, 9), cursor="hand2",
                          highlightthickness=1, highlightbackground=bg, highlightcolor=ACCENT, bd=0,
                          anchor="w")


def combo(parent, var, values, width: int = 26) -> ttk.Combobox:
    return ttk.Combobox(parent, textvariable=var, values=values, state="readonly", font=(FONT, 9), width=width)


def entry(parent, var, bg: str = CARD) -> tk.Entry:
    return tk.Entry(parent, textvariable=var, bg=bg, fg=FG, insertbackground=ACCENT, font=(FONT, 9), bd=0,
                    relief="flat", highlightthickness=1, highlightbackground=BORDER, highlightcolor=ACCENT,
                    disabledbackground=PANEL, disabledforeground=FG_MUTED)


# ── Labels, sections, layout helpers ──────────────────────────────────────────

def caption(parent, text: str, bg: str = PANEL, fg: str = FG_MUTED, **kw) -> tk.Label:
    """Uppercase JetBrains Mono technical label."""
    return tk.Label(parent, text=text.upper(), bg=bg, fg=fg, font=LABEL, anchor="w", **kw)


def divider(parent, bg: str = BORDER, padx: int = 0, edge: Optional[str] = None) -> tk.Frame:
    """1 px rule: packed in the flow, or pinned to the parent's 'top' / 'bottom' edge."""
    f = tk.Frame(parent, bg=bg, height=1)
    if edge:
        f.place(relx=0, rely=0 if edge == "top" else 1, relwidth=1, anchor="nw" if edge == "top" else "sw")
    else:
        f.pack(fill="x", padx=padx)
    return f


def section(parent, text: str, bg: str = PANEL, padx: int = 16):
    """Divider + technical caption that starts a sidebar / form group."""
    divider(parent, padx=padx).pack_configure(pady=(14, 0))
    caption(parent, text, bg=bg).pack(fill="x", padx=padx, pady=(10, 6))


def note(parent, text: str = "", bg: str = PANEL, fg: str = FG_DIM, wrap: int = 250) -> tk.Label:
    return tk.Label(parent, text=text, bg=bg, fg=fg, font=(FONT, 8), justify="left", anchor="w", wraplength=wrap)


def labelled(parent, label: str, widget, bg: str = PANEL):
    f = tk.Frame(parent, bg=bg)
    f.pack(fill="x", padx=16, pady=2)
    tk.Label(f, text=label, bg=bg, fg=FG_DIM, font=(FONT, 9), width=8, anchor="w").pack(side="left")
    widget.pack(in_=f, side="left", fill="x", expand=True)
    widget.lift(f)


def slider(parent, label: str, var: tk.DoubleVar, lo: float, hi: float, on_change, bg: str = PANEL):
    f = tk.Frame(parent, bg=bg)
    f.pack(fill="x", padx=16, pady=(4, 0))
    tk.Label(f, text=label, bg=bg, fg=FG_DIM, font=(FONT, 9)).pack(side="left")
    val = tk.Label(f, text=f"{var.get():.2f}", bg=bg, fg=ACCENT_HOVER, font=(MONO_SEMI, 9))
    val.pack(side="right")

    def changed(_):
        val.config(text=f"{var.get():.2f}")
        if on_change:
            on_change(var.get())

    tk.Scale(parent, from_=lo, to=hi, resolution=0.01, orient="horizontal", variable=var, bg=FG_MUTED,
             fg=FG, troughcolor=CARD, activebackground=ACCENT, highlightthickness=0, bd=0, showvalue=False,
             sliderlength=14, width=6, sliderrelief="flat", command=changed).pack(fill="x", padx=16, pady=(2, 0))


class Collapsible(tk.Frame):
    """Section whose body is hidden until its header is clicked (progressive disclosure)."""

    def __init__(self, parent, title: str, bg: str = PANEL, opened: bool = False, hint: str = ""):
        super().__init__(parent, bg=bg)
        divider(self).pack_configure(padx=16, pady=(14, 0))
        self._head = tk.Label(self, bg=bg, fg=FG_MUTED, font=LABEL, anchor="w", cursor="hand2",
                              compound="left", padx=2, takefocus=1, highlightthickness=1,
                              highlightbackground=bg, highlightcolor=ACCENT)
        self._head.pack(fill="x", padx=14, pady=(8, 2))
        self._title, self._hint, self._bg = title.upper(), hint, bg
        self.body = tk.Frame(self, bg=bg)
        for seq in ("<Button-1>", "<Return>", "<space>"):
            self._head.bind(seq, lambda e: self.toggle())
        self._head.bind("<Enter>", lambda e: self._head.config(fg=FG))
        self._head.bind("<Leave>", lambda e: self._head.config(fg=FG_MUTED))
        self.opened = not opened
        self.toggle()

    def toggle(self):
        self.opened = not self.opened
        hint = "" if self.opened or not self._hint else f"   {self._hint}"
        self._head.config(text=f" {self._title}{hint}",
                          image=icon("chev-down" if self.opened else "chev-right", FG_MUTED, 14))
        if self.opened:
            self.body.pack(fill="x")
        else:
            self.body.pack_forget()


class StatTile(tk.Frame):
    def __init__(self, parent, caption_text: str, colour: str):
        super().__init__(parent, bg=PANEL, highlightthickness=1, highlightbackground=BORDER)
        self.value = tk.Label(self, text="—", bg=PANEL, fg=colour, font=NUM, anchor="w")
        self.value.pack(fill="x", padx=10, pady=(6, 0))
        caption(self, caption_text, bg=PANEL).pack(fill="x", padx=10, pady=(0, 7))

    def set(self, v):
        self.value.config(text=str(v))

    def alarm(self, on: bool):
        self.config(highlightbackground=DANGER if on else BORDER)


class Chip(tk.Label):
    """Compact bordered status tag (rectangular – no pills)."""

    def __init__(self, parent, text: str = "", colour: str = FG_DIM, bg: str = CARD):
        super().__init__(parent, text=text, bg=bg, fg=colour, font=LABEL, padx=8, pady=3,
                         highlightthickness=1, highlightbackground=BORDER)

    def set(self, text: str, colour: Optional[str] = None, bg: Optional[str] = None, edge: Optional[str] = None):
        self.config(text=text.upper(), **({"fg": colour} if colour else {}), **({"bg": bg} if bg else {}),
                    highlightbackground=edge or BORDER)


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


# ── Themed dialogs (drop-in for tkinter.messagebox) ───────────────────────────

class dialogs:
    """Same call shape as tkinter.messagebox, drawn in the console's style."""

    KINDS = {"info": ("check", ACCENT), "warning": ("alert", WARN), "error": ("x", DANGER), "ask": ("alert", WARN)}

    @staticmethod
    def _show(kind: str, title: str, message: str, parent=None, ok: str = "OK", cancel: Optional[str] = None,
              danger: bool = False) -> bool:
        parent = parent or tk._default_root
        top = tk.Toplevel(parent, bg=PANEL, highlightthickness=1, highlightbackground=BORDER)
        top.withdraw()
        top.title(title)
        top.resizable(False, False)
        top.transient(parent.winfo_toplevel())
        result = {"ok": False}
        icon_name, colour = dialogs.KINDS[kind]
        body = tk.Frame(top, bg=PANEL)
        body.pack(fill="both", padx=20, pady=(18, 12))
        tk.Label(body, image=icon(icon_name, colour, 20), bg=PANEL).grid(row=0, column=0, sticky="n", padx=(0, 12))
        tk.Label(body, text=title, bg=PANEL, fg=FG, font=(FONT_SEMI, 11), anchor="w").grid(row=0, column=1, sticky="w")
        tk.Label(body, text=message, bg=PANEL, fg=FG_DIM, font=(FONT, 9), justify="left", anchor="w",
                 wraplength=400).grid(row=1, column=1, sticky="w", pady=(6, 0))
        divider(top)
        bar = tk.Frame(top, bg=BG2)
        bar.pack(fill="x")

        def close(value: bool):
            result["ok"] = value
            top.grab_release()
            top.destroy()

        primary = button(bar, ok, lambda: close(True), DANGER if danger else ACCENT)
        primary.pack(side="right", padx=(6, 16), pady=10)
        if cancel:
            button(bar, cancel, lambda: close(False)).pack(side="right", pady=10)
        if not danger:                       # a destructive choice needs a deliberate click
            top.bind("<Return>", lambda e: close(True))
        top.bind("<Escape>", lambda e: close(False))
        top.protocol("WM_DELETE_WINDOW", lambda: close(False))
        top.update_idletasks()
        px, py = parent.winfo_rootx(), parent.winfo_rooty()
        pw, ph = parent.winfo_width(), parent.winfo_height()
        w, h = top.winfo_reqwidth(), top.winfo_reqheight()
        top.geometry(f"+{px + max(0, (pw - w) // 2)}+{py + max(0, (ph - h) // 3)}")
        top.deiconify()
        dark_titlebar(top)
        top.grab_set()
        (primary if not cancel or not danger else top).focus_set()
        parent.wait_window(top)
        return result["ok"]

    @staticmethod
    def showinfo(title, message, parent=None):
        dialogs._show("info", title, message, parent)

    @staticmethod
    def showwarning(title, message, parent=None):
        dialogs._show("warning", title, message, parent)

    @staticmethod
    def showerror(title, message, parent=None):
        dialogs._show("error", title, message, parent)

    @staticmethod
    def askyesno(title, message, parent=None, ok: str = "Yes", cancel: str = "No", danger: bool = False) -> bool:
        return dialogs._show("ask", title, message, parent, ok=ok, cancel=cancel, danger=danger)


# ── Misc helpers ──────────────────────────────────────────────────────────────

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
