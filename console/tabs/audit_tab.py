"""
console/tabs/audit_tab.py
────────────────────────────────────────────────────────────────────────────────
Audit Log tab: live Treeview of output/audit_log.csv with filtering and quick
buttons to open the CSV or output folder.
────────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import csv
import os
import tkinter as tk
from pathlib import Path
from tkinter import ttk

from console.theme import (OK, BG, CARD2, DANGER, FG, FG_DIM, FONT, IS_WIN,
                           WARN, button, combo)


class AuditTab(tk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, bg=BG)
        self.app = app
        self._audit_mtime = 0.0
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        top = tk.Frame(self, bg=BG)
        top.grid(row=0, column=0, sticky="ew", padx=8, pady=8)
        tk.Label(top, text="Forensic audit log", bg=BG, fg=FG, font=(FONT, 13, "bold")).pack(side="left")
        self.audit_count = tk.Label(top, text="", bg=BG, fg=FG_DIM, font=(FONT, 9))
        self.audit_count.pack(side="left", padx=12)
        button(top, "Open folder", self._open_output, CARD2, fg=FG).pack(side="right", padx=(4, 0))
        button(top, "Open CSV", self._open_audit, CARD2, fg=FG).pack(side="right", padx=4)
        self.audit_filter = tk.StringVar(value="All events")
        cb = combo(top, self.audit_filter, ["All events", "Alerts only", "Unknown only", "Known only"], 14)
        cb.pack(side="right", padx=4)
        cb.bind("<<ComboboxSelected>>", lambda e: self.refresh(force=True))

        cols = ("Timestamp", "Camera_ID", "Track_ID", "Identity", "Status", "Confidence",
                "Zone_Alert", "Asset_Interaction")
        frame = tk.Frame(self, bg=BG)
        frame.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 8))
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
        self.tree = ttk.Treeview(frame, columns=cols, show="headings")
        for c, w in zip(cols, (90, 90, 70, 110, 90, 90, 180, 240)):
            self.tree.heading(c, text=c.replace("_", " ").upper())
            self.tree.column(c, width=w, anchor="w")
        self.tree.grid(row=0, column=0, sticky="nsew")
        vsb = ttk.Scrollbar(frame, orient="vertical", command=self.tree.yview, style="Dark.Vertical.TScrollbar")
        vsb.grid(row=0, column=1, sticky="ns")
        self.tree.configure(yscrollcommand=vsb.set)
        self.tree.tag_configure("alert", foreground=DANGER)
        self.tree.tag_configure("UNKNOWN", foreground=WARN)
        self.tree.tag_configure("KNOWN", foreground=OK)

    def refresh(self, force: bool = False):
        path = Path(self.app.engine._audit_path)
        if not path.exists():
            self.audit_count.config(text="no events yet")
            self.tree.delete(*self.tree.get_children())
            return
        mtime = path.stat().st_mtime
        if not force and mtime == self._audit_mtime:
            return
        self._audit_mtime = mtime
        try:
            with open(path, newline="", encoding="utf-8") as f:
                rows = list(csv.reader(f))[1:]
        except Exception:
            return
        flt = self.audit_filter.get()
        shown = []
        for r in reversed(rows[-2000:]):
            if len(r) < 8:
                continue
            alert = r[6] != "None" or r[7] != "None"
            if ((flt == "Alerts only" and not alert)
                    or (flt == "Unknown only" and r[4] != "UNKNOWN")
                    or (flt == "Known only" and r[4] != "KNOWN")):
                continue
            shown.append((r, "alert" if alert else r[4]))
        self.tree.delete(*self.tree.get_children())
        for r, tag in shown[:1000]:
            self.tree.insert("", "end", values=r[:8], tags=(tag,))
        n_alerts = sum(1 for r in rows if len(r) >= 8 and (r[6] != "None" or r[7] != "None"))
        self.audit_count.config(text=f"{len(rows):,} events · {n_alerts:,} alerts · {path}")

    def _open_audit(self):
        p = Path(self.app.engine._audit_path)
        if p.exists() and IS_WIN:
            os.startfile(p)

    def _open_output(self):
        p = Path(self.app.engine._audit_path).parent
        p.mkdir(parents=True, exist_ok=True)
        if IS_WIN:
            os.startfile(p)
