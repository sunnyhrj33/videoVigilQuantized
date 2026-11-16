"""
console/tabs/eval_tab.py
────────────────────────────────────────────────────────────────────────────────
Evaluation tab: custom Canvas bar charts comparing Naive closed-set, Naive
threshold, Tracklet rule-based, and Tracklet logistic regression on held-out
CHIRLA test days.
────────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import json
import tkinter as tk
from pathlib import Path

from console.theme import (ACCENT, ACCENT2, BG, BORDER, CARD, CARD2, CYAN,
                           DANGER, FG, FG_DIM, FONT, WARN)


class EvalTab(tk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, bg=BG)
        self.app = app
        self.canvas = tk.Canvas(self, bg=BG, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, padx=6, pady=6)
        self.canvas.bind("<Configure>", lambda e: self.render())

    def render(self):
        c = self.canvas
        c.delete("all")
        W, H = max(c.winfo_width(), 600), max(c.winfo_height(), 400)
        path = Path(self.app.cfg["reid"]["open_set_model"])
        try:
            data = json.loads(path.read_text())
            m = data["metrics"]
        except Exception:
            c.create_text(W // 2, H // 2, text="No evaluation yet – run Pipeline step 3.",
                          fill=FG_DIM, font=(FONT, 12))
            return

        c.create_text(16, 18, anchor="w", text="Open-set evaluation on held-out days",
                      fill=FG, font=(FONT, 14, "bold"))
        days = ", ".join(m.get("test_days", []))
        c.create_text(16, 42, anchor="w", fill=FG_DIM, font=(FONT, 9),
                      text=f"Test days {days} · visitors never seen in training · same-day enrollment")
        try:
            se = json.loads(Path("output/search_eval.json").read_text())
            ex = se.get("exclude_employees") or se.get("plain")
            c.create_text(W - 16, 18, anchor="e", fill=CYAN, font=(FONT, 10, "bold"),
                          text=f"Video-wide search ({ex['queries']} visitor queries, CHIRLA GT): "
                               f"top-1 {ex['rank1']:.0%} · top-5 {ex['top5']:.0%} · mAP {ex['mAP']:.0%}")
            c.create_text(W - 16, 42, anchor="e", fill=FG_DIM, font=(FONT, 9),
                          text=f"days {', '.join(se.get('days', []))} · employees excluded · "
                               f"python -m core.search eval")
        except Exception:
            pass

        methods = [
            ("Naive\nclosed-set", m.get("naive_closed_set"), "#5b6577"),
            (f"Naive\ncos ≥ {m.get('naive_thresh_value', 0.45)}", m.get("naive_threshold"), "#7b8599"),
            ("Tracklet\nrule-based", m.get("tracklet_rule"), ACCENT2),
            ("Tracklet\n+ log. regression", m.get("test_tracklet_lr"), ACCENT),
        ]
        methods = [x for x in methods if x[1]]
        panels = [
            ("TAR", "Employee → correct ID", True),
            ("MisID", "Employee → wrong ID", False),
            ("FAR", "Visitor → passed as employee", False),
            ("TRR", "Visitor → rejected UNKNOWN", True),
        ]

        top, bottom = 70, H - 150
        pw = (W - 32) / len(panels)
        for i, (key, title, higher) in enumerate(panels):
            x0 = 16 + i * pw
            c.create_rectangle(x0 + 4, top, x0 + pw - 4, bottom + 46, fill=CARD, outline=BORDER)
            c.create_text(x0 + 16, top + 16, anchor="w", text=title, fill=FG, font=(FONT, 10, "bold"))
            c.create_text(x0 + pw - 16, top + 16, anchor="e", fill=ACCENT if higher else WARN,
                          font=(FONT, 9, "bold"), text="higher is better ↑" if higher else "lower is better ↓")
            chart_top, chart_bot = top + 44, bottom
            bw = (pw - 40) / max(len(methods), 1)
            for j, (name, mm, col) in enumerate(methods):
                v = float(mm.get(key, 0) or 0)
                bx0 = x0 + 20 + j * bw + bw * 0.15
                bx1 = x0 + 20 + (j + 1) * bw - bw * 0.15
                by = chart_bot - (chart_bot - chart_top - 20) * v
                c.create_rectangle(bx0, chart_top + 20, bx1, chart_bot, fill=CARD2, outline="")
                c.create_rectangle(bx0, by, bx1, chart_bot, fill=col, outline="")
                c.create_text((bx0 + bx1) / 2, by - 9, text=f"{v:.0%}", fill=FG, font=(FONT, 9, "bold"))
                c.create_text((bx0 + bx1) / 2, chart_bot + 20, text=name, fill=FG_DIM,
                              font=(FONT, 7), justify="center")

        y = H - 92
        lr = m.get("test_tracklet_lr", {})
        cards = [
            ("ROC-AUC (tracklet LR)", f"{m.get('test_auc', 0):.3f}", ACCENT),
            ("Test tracklets", f"{lr.get('n_employee', 0)} emp · {lr.get('n_visitor', 0)} visitor", ACCENT2),
            ("Visitor rejection", f"{(m.get('naive_threshold') or {}).get('TRR', 0):.0%} → "
                                  f"{lr.get('TRR', 0):.0%}", WARN),
            ("Wrong-ID rate", f"{(m.get('naive_closed_set') or {}).get('MisID', 0):.0%} → "
                              f"{lr.get('MisID', 0):.0%}", DANGER),
        ]
        cw = (W - 32) / len(cards)
        for i, (cap, val, col) in enumerate(cards):
            x0 = 16 + i * cw
            c.create_rectangle(x0 + 4, y, x0 + cw - 4, y + 72, fill=CARD, outline=BORDER)
            c.create_text(x0 + 18, y + 22, anchor="w", text=cap.upper(), fill=FG_DIM, font=(FONT, 8, "bold"))
            c.create_text(x0 + 18, y + 48, anchor="w", text=val, fill=col, font=(FONT, 15, "bold"))
