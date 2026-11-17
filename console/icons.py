"""
console/icons.py
────────────────────────────────────────────────────────────────────────────────
Lucide-style outline icons for Tkinter, drawn with PIL (24-unit grid, ~1.7 px
stroke, 4× supersampled). `icon(name, colour, size)` returns a cached PhotoImage.
────────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from typing import Dict, Tuple

from PIL import Image, ImageDraw, ImageTk

# Ops on a 24×24 grid: ("l", pts) open line · ("p", pts) closed outline · ("c", x, y, r) circle ·
# ("a", x, y, r, start°, end°) arc (clockwise from 3 o'clock) · ("r", x0, y0, x1, y1, radius) · ("d", x, y) dot
SHAPES: Dict[str, list] = {
    "play": [("p", [(7, 4), (19, 12), (7, 20)])],
    "stop": [("r", 5, 5, 19, 19, 2)],
    "search": [("c", 11, 11, 7), ("l", [(16.5, 16.5), (21, 21)])],
    "flag": [("l", [(5, 22), (5, 3)]), ("p", [(5, 4), (9, 3), (14, 5), (19, 4), (19, 14), (14, 15), (9, 13), (5, 14)])],
    "video": [("r", 2, 6, 16, 18, 2), ("p", [(16, 10.5), (22, 7), (22, 17), (16, 13.5)])],
    "webcam": [("c", 12, 10, 7), ("c", 12, 10, 2.5), ("l", [(12, 17), (12, 21)]), ("l", [(7, 21), (17, 21)])],
    "folder": [("p", [(3, 5), (9, 5), (11, 7.5), (21, 7.5), (21, 19), (3, 19)])],
    "refresh": [("a", 12, 12, 9, 0, 310), ("l", [(21, 3), (21, 8), (16, 8)])],
    "pencil": [("p", [(16, 4), (20, 8), (8, 20), (4, 20), (4, 16)]), ("l", [(14, 6), (18, 10)])],
    "trash": [("l", [(3, 6), (21, 6)]), ("l", [(5, 6), (6, 21), (18, 21), (19, 6)]),
              ("l", [(9, 6), (9, 3), (15, 3), (15, 6)])],
    "x": [("l", [(6, 6), (18, 18)]), ("l", [(18, 6), (6, 18)])],
    "alert": [("p", [(12, 3), (22, 20), (2, 20)]), ("l", [(12, 9), (12, 13.5)]), ("d", 12, 16.8)],
    "check": [("l", [(4, 12.5), (9, 17.5), (20, 6.5)])],
    "bell": [("a", 12, 9, 6, 180, 360), ("l", [(6, 9), (6, 14), (4, 17), (20, 17), (18, 14), (18, 9)]),
             ("l", [(10, 21), (14, 21)])],
    "bell-off": [("a", 12, 9, 6, 180, 360), ("l", [(6, 9), (6, 14), (4, 17), (20, 17), (18, 14), (18, 9)]),
                 ("l", [(10, 21), (14, 21)]), ("l", [(3, 3), (21, 21)])],
    "chev-right": [("l", [(9, 6), (15, 12), (9, 18)])],
    "chev-down": [("l", [(6, 9), (12, 15), (18, 9)])],
    "monitor": [("r", 2, 3, 22, 16, 2), ("l", [(12, 16), (12, 21)]), ("l", [(8, 21), (16, 21)])],
    "target": [("c", 12, 12, 9), ("c", 12, 12, 3), ("l", [(12, 1.5), (12, 5)]), ("l", [(12, 19), (12, 22.5)]),
               ("l", [(1.5, 12), (5, 12)]), ("l", [(19, 12), (22.5, 12)])],
    "users": [("c", 9, 7.5, 3.5), ("l", [(2.5, 20), (2.5, 18), (5, 15), (13, 15), (15.5, 18), (15.5, 20)]),
              ("a", 15.5, 7.5, 3.5, 270, 90), ("l", [(18.5, 15.2), (21.5, 18), (21.5, 20)])],
    "list": [("l", [(9, 6), (21, 6)]), ("l", [(9, 12), (21, 12)]), ("l", [(9, 18), (21, 18)]),
             ("d", 4, 6), ("d", 4, 12), ("d", 4, 18)],
    "activity": [("l", [(2, 12), (6, 12), (9, 4), (15, 20), (18, 12), (22, 12)])],
    "chart": [("l", [(4, 20), (20, 20)]), ("l", [(7, 20), (7, 14)]), ("l", [(12, 20), (12, 6)]),
              ("l", [(17, 20), (17, 10)])],
    "layers": [("p", [(12, 3), (22, 8), (12, 13), (2, 8)]), ("l", [(2, 12.5), (12, 17.5), (22, 12.5)]),
               ("l", [(2, 17), (12, 22), (22, 17)])],
    "plus": [("l", [(12, 5), (12, 19)]), ("l", [(5, 12), (19, 12)])],
    "terminal": [("l", [(4, 17), (10, 11), (4, 5)]), ("l", [(12, 19), (20, 19)])],
}

_cache: Dict[Tuple[str, str, int], ImageTk.PhotoImage] = {}
SS = 4                                  # supersampling factor


def _rgb(colour: str) -> Tuple[int, int, int]:
    c = colour.lstrip("#")
    return int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)


def render(name: str, colour: str, size: int = 18, stroke: float = 1.7) -> Image.Image:
    """The icon as an RGBA PIL image (size × size)."""
    px, fill = size * SS, _rgb(colour) + (255,)
    k, w = px / 24.0, max(1, round(stroke * SS))
    img = Image.new("RGBA", (px, px), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    def dot(x, y, r):
        d.ellipse((x - r, y - r, x + r, y + r), fill=fill)

    for op in SHAPES[name]:
        kind, args = op[0], op[1:]
        if kind in ("l", "p"):
            pts = [(x * k, y * k) for x, y in args[0]]
            d.line(pts + ([pts[0]] if kind == "p" else []), fill=fill, width=w, joint="curve")
            for x, y in pts:                     # round caps and corners
                dot(x, y, w / 2)
        elif kind == "c":
            x, y, r = (v * k for v in args)
            d.ellipse((x - r, y - r, x + r, y + r), outline=fill, width=w)
        elif kind == "a":
            x, y, r = (v * k for v in args[:3])
            d.arc((x - r, y - r, x + r, y + r), args[3], args[4], fill=fill, width=w)
        elif kind == "r":
            x0, y0, x1, y1, rad = (v * k for v in args)
            d.rounded_rectangle((x0, y0, x1, y1), radius=rad, outline=fill, width=w)
        elif kind == "d":
            dot(args[0] * k, args[1] * k, w * 0.75)
    return img.resize((size, size), Image.LANCZOS)


def icon(name: str, colour: str, size: int = 18) -> ImageTk.PhotoImage:
    """Cached Tk image of a named icon (needs a Tk root)."""
    key = (name, colour.lower(), size)
    if key not in _cache:
        _cache[key] = ImageTk.PhotoImage(render(name, colour, size))
    return _cache[key]
