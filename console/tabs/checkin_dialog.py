"""
console/tabs/checkin_dialog.py
────────────────────────────────────────────────────────────────────────────────
Gallery tab → "Build from check-in video…": enroll a day's staff from a morning
check-in recording.

  1. Source – a video file (its recording date is required) or the cameras of a
     recorded day; analysed in the background (cancellable).
  2. People – give each employee a staff label (emp_04, a name, …), choose merge
     or replace if the day already has a gallery, build. The same rules as the web
     dashboard apply (core.enrollment): every problem is listed and nothing is
     written until all are fixed; the old gallery is backed up first.
────────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import logging
import queue
import shutil
import threading
import tkinter as tk
from datetime import date as _date, datetime
from pathlib import Path
from tkinter import filedialog, ttk
from console.theme import dialogs as messagebox  # themed drop-in
from typing import Optional

from PIL import Image, ImageTk

from console.theme import (FONT_SEMI, LABEL, OK, ACCENT, BG, BORDER, CARD, CARD2, DANGER, FG, FG_DIM, FONT,
                           PANEL, WARN, ScrollFrame, button, combo)
from core.chirla import date_in_name, is_iso_date
from core.enrollment import (EnrollmentError, all_staff_labels, build_staff, describe_existing,
                             eligibility, write_day_gallery)
from webapp.pipeline import web_settings

logger = logging.getLogger("console.checkin")

STATUS_COL = {"KNOWN": OK, "UNKNOWN": WARN, "SUSPECT": DANGER, "UNCLEAR": FG_DIM}


class CheckinDialog(tk.Toplevel):
    def __init__(self, app, initial_day: Optional[str] = None):
        super().__init__(app)
        self.app = app
        self.title("Build a day's staff gallery from a morning check-in")
        self.configure(bg=BG)
        self.geometry("900x760")
        self.minsize(760, 600)
        self.transient(app)
        self.protocol("WM_DELETE_WINDOW", self._close)
        s = web_settings(app.cfg)
        self.k, self.min_samples = int(s["enroll_samples"]), int(s["enroll_min_samples"])
        self.gallery_file = Path(app.cfg["reid"]["gallery_file"])
        self._q: "queue.Queue[tuple]" = queue.Queue()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._work_dir: Optional[Path] = None
        self._people, self._embs, self._date = [], {}, None
        self._label_vars: dict = {}
        self._thumbs = []
        self.columnconfigure(0, weight=1)
        self.rowconfigure(2, weight=1)
        self._build_source(initial_day)
        self._build_people()
        self.after(100, self._poll)

    # ── 1 · source ────────────────────────────────────────────────────────────

    def _build_source(self, initial_day):
        f = tk.Frame(self, bg=PANEL)
        f.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 0))
        f.columnconfigure(2, weight=1)
        tk.Label(f, text="1 · CHECK-IN RECORDING", bg=PANEL, fg=FG_DIM, font=LABEL).grid(
            row=0, column=0, columnspan=4, sticky="w", padx=12, pady=(10, 4))
        tk.Label(f, text="Use footage from the start of the day (people change clothes between days). "
                         "Don't later test recognition on this same footage.",
                 bg=PANEL, fg=FG_DIM, font=(FONT, 8), anchor="w").grid(row=1, column=0, columnspan=4,
                                                                       sticky="w", padx=12)
        self.kind = tk.StringVar(value="day" if self.app._days else "file")
        radio = dict(bg=PANEL, fg=FG, selectcolor=CARD, activebackground=PANEL, activeforeground=ACCENT,
                     font=(FONT, 9), command=self._kind_changed, variable=self.kind)

        tk.Radiobutton(f, text="Video file", value="file", **radio).grid(row=2, column=0, sticky="w", padx=8)
        self.path_var = tk.StringVar()
        self.path_var.trace_add("write", lambda *_: self._name_hint())
        self.path_entry = tk.Entry(f, textvariable=self.path_var, bg=CARD, fg=FG, insertbackground=ACCENT,
                                   bd=0, font=(FONT, 9), highlightthickness=1, highlightbackground=BORDER)
        self.path_entry.grid(row=2, column=1, columnspan=2, sticky="ew", padx=4, ipady=4)
        self.browse_btn = button(f, "Browse…", self._browse, icon="folder")
        self.browse_btn.grid(row=2, column=3, sticky="ew", padx=(4, 12), pady=2)
        tk.Label(f, text="Recorded on", bg=PANEL, fg=FG_DIM, font=(FONT, 9)).grid(row=3, column=1, sticky="e")
        df = tk.Frame(f, bg=PANEL)
        df.grid(row=3, column=2, columnspan=2, sticky="w", padx=4, pady=2)
        self.date_var = tk.StringVar()
        self.date_entry = tk.Entry(df, textvariable=self.date_var, width=12, bg=CARD, fg=FG,
                                   insertbackground=ACCENT, bd=0, font=(FONT, 10), highlightthickness=1,
                                   highlightbackground=BORDER)
        self.date_entry.pack(side="left", ipady=3)
        tk.Label(df, text=" YYYY-MM-DD (required)", bg=PANEL, fg=FG_DIM, font=(FONT, 8)).pack(side="left")
        self.hint_btn = button(df, "", self._use_name_date, CARD2, fg=FG)

        tk.Radiobutton(f, text="Recorded day", value="day", **radio).grid(row=4, column=0, sticky="w", padx=8,
                                                                         pady=(6, 0))
        rf = tk.Frame(f, bg=PANEL)
        rf.grid(row=4, column=1, columnspan=3, sticky="ew", padx=4, pady=(6, 0))
        self.day_var, self.seq_var = tk.StringVar(), tk.StringVar()
        self.day_combo = combo(rf, self.day_var, list(self.app._day_items), width=24)
        self.day_combo.pack(side="left")
        self.day_combo.bind("<<ComboboxSelected>>", lambda e: self._day_changed())
        self.seq_combo = combo(rf, self.seq_var, [], width=26)
        self.seq_combo.pack(side="left", padx=6)
        self.seq_combo.bind("<<ComboboxSelected>>", lambda e: self._seq_changed())
        self.cam_frame = tk.Frame(f, bg=PANEL)
        self.cam_frame.grid(row=5, column=1, columnspan=3, sticky="w", padx=4)
        self._cam_vars: dict = {}

        bf = tk.Frame(f, bg=PANEL)
        bf.grid(row=6, column=0, columnspan=4, sticky="ew", padx=12, pady=10)
        self.analyse_btn = button(bf, "Analyse check-in", self._analyse, ACCENT, icon="play")
        self.analyse_btn.pack(side="left")
        self.cancel_btn = button(bf, "Cancel", self._cancel, CARD2, fg=FG, state="disabled")
        self.cancel_btn.pack(side="left", padx=6)
        self.progress = ttk.Progressbar(bf, style="Green.Horizontal.TProgressbar", maximum=1.0)
        self.progress.pack(side="left", fill="x", expand=True, padx=8)
        self.status = tk.Label(f, text="", bg=PANEL, fg=FG_DIM, font=(FONT, 9), anchor="w", justify="left")
        self.status.grid(row=7, column=0, columnspan=4, sticky="ew", padx=12, pady=(0, 10))

        label = next((k for k, d in self.app._day_items.items() if d == initial_day), None)
        if label:
            self.day_var.set(label)
        elif self.app._day_items:
            self.day_var.set(next(iter(self.app._day_items)))
        self._day_changed()
        self._kind_changed()

    def _kind_changed(self):
        file_mode = self.kind.get() == "file"
        for w in (self.path_entry, self.browse_btn, self.date_entry):
            w.config(state="normal" if file_mode else "disabled")
        for w in (self.day_combo, self.seq_combo):
            w.config(state="readonly" if not file_mode else "disabled")
        for cb in self.cam_frame.winfo_children():
            cb.config(state="normal" if not file_mode else "disabled")
        self._name_hint()

    def _browse(self):
        path = filedialog.askopenfilename(parent=self, title="Morning check-in video", filetypes=[
            ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv"), ("All files", "*.*")])
        if path:
            self.path_var.set(path)

    def _name_hint(self):
        suggested = date_in_name(Path(self.path_var.get()).name) if self.kind.get() == "file" else None
        if suggested and suggested != self.date_var.get():
            self.hint_btn.config(text=f"Use {suggested} (in file name)")
            self.hint_btn.pack(side="left", padx=8)
        else:
            self.hint_btn.pack_forget()

    def _use_name_date(self):
        self.date_var.set(date_in_name(Path(self.path_var.get()).name) or "")
        self._name_hint()

    def _day_changed(self):
        seqs = self.app._days.get(self.app._day_items.get(self.day_var.get()), [])
        self._seqs = {f"{s.name}  ·  {s.start.strftime('%H:%M:%S') if s.start else '?'}  ·  "
                      f"{len(s.cameras)} cam": s for s in seqs}
        self.seq_combo.config(values=list(self._seqs))
        self.seq_var.set(next(iter(self._seqs), ""))
        self._seq_changed()

    def _seq_changed(self):
        for w in self.cam_frame.winfo_children():
            w.destroy()
        self._cam_vars = {}
        seq = self._seqs.get(self.seq_var.get())
        if seq is None:
            return
        tk.Label(self.cam_frame, text="Cameras:", bg=PANEL, fg=FG_DIM, font=(FONT, 8)).pack(side="left")
        for c in seq.cameras:
            v = tk.BooleanVar(value=True)
            self._cam_vars[c.camera] = (v, c)
            tk.Checkbutton(self.cam_frame, text=c.cam_id.replace("Cam_", "C"), variable=v, bg=PANEL, fg=FG,
                           selectcolor=CARD, activebackground=PANEL, font=(FONT, 8)).pack(side="left")
        if seq.missing:
            tk.Label(self.cam_frame, text=f"  ({', '.join(seq.missing)} missing)", bg=PANEL, fg=WARN,
                     font=(FONT, 8)).pack(side="left")

    def _sources(self):
        """[(path, camera label)], date – or raises ValueError with a message for the user."""
        if self.kind.get() == "file":
            path = self.path_var.get().strip()
            if not path or not Path(path).is_file():
                raise ValueError("Choose the check-in video file.")
            d = self.date_var.get().strip()
            if not d:
                raise ValueError("Enter the date the video was recorded (YYYY-MM-DD).")
            if not is_iso_date(d):
                raise ValueError(f"“{d}” isn't a valid date – use YYYY-MM-DD.")
            if d > _date.today().isoformat():
                raise ValueError("The recording date can't be in the future.")
            return [(path, Path(path).name)], d
        seq = self._seqs.get(self.seq_var.get())
        if seq is None:
            raise ValueError("Choose a recorded day and sequence.")
        cams = [(str(c.path), c.cam_id) for v, c in self._cam_vars.values() if v.get()]
        if not cams:
            raise ValueError("Tick at least one camera.")
        return cams, seq.date

    def _analyse(self):
        try:
            videos, day = self._sources()
        except ValueError as exc:
            messagebox.showinfo("Check-in", str(exc), parent=self)
            return
        app = self.app
        if app.engine._model is None:
            messagebox.showwarning("Check-in", "Load the models first (sidebar → Load models).", parent=self)
            return
        if app._running:
            if not messagebox.askyesno("Check-in", "Playback has to stop while the check-in is analysed. Stop it?",
                                       parent=self):
                return
            app._stop()
        self._clear_people()
        self._discard_work()
        self._work_dir = Path("output/checkin") / datetime.now().strftime("%Y%m%d-%H%M%S")
        self._date = day
        self._stop.clear()
        app.checkin_busy = True
        self.analyse_btn.config(state="disabled")
        self.cancel_btn.config(state="normal")
        self.progress["value"] = 0
        self.status.config(text=f"Analysing {len(videos)} video{'s' if len(videos) > 1 else ''} for {day} …",
                           fg=FG_DIM)

        def work():
            from console.checkin import CheckinCancelled, collect_people
            try:
                people, embs = collect_people(app.engine, videos, day, self._work_dir, app.cfg,
                                              progress=lambda f, t: self._q.put(("progress", f, t)),
                                              stop=self._stop)
                self._q.put(("done", people, embs))
            except CheckinCancelled:
                self._q.put(("cancelled",))
            except Exception as exc:
                logger.exception("Check-in analysis failed")
                self._q.put(("error", str(exc)))
            finally:
                app.checkin_busy = False          # also if the dialog was closed meanwhile

        self._thread = threading.Thread(target=work, name="checkin", daemon=True)
        self._thread.start()

    def _cancel(self):
        self._stop.set()
        self.cancel_btn.config(state="disabled")
        self.status.config(text="Cancelling …")

    def _poll(self):
        try:
            while True:
                msg = self._q.get_nowait()
                if msg[0] == "progress":
                    self.progress["value"] = msg[1]
                    self.status.config(text=msg[2], fg=FG_DIM)
                    continue
                self.analyse_btn.config(state="normal")
                self.cancel_btn.config(state="disabled")
                if msg[0] == "done":
                    self._people, self._embs = msg[1], msg[2]
                    self.progress["value"] = 1.0
                    self._show_people()
                elif msg[0] == "cancelled":
                    self.status.config(text="Cancelled – nothing was saved.", fg=WARN)
                    self._discard_work()
                else:
                    self.status.config(text=f"Analysis failed: {msg[1]}", fg=DANGER)
                    self._discard_work()
        except queue.Empty:
            pass
        if self.winfo_exists():
            self.after(100, self._poll)

    # ── 2 · people → staff labels → build ─────────────────────────────────────

    def _build_people(self):
        f = tk.Frame(self, bg=PANEL)
        f.grid(row=1, column=0, sticky="ew", padx=10, pady=(10, 0))
        tk.Label(f, text="2 · LABEL THE STAFF", bg=PANEL, fg=FG_DIM, font=LABEL).pack(
            anchor="w", padx=12, pady=(10, 2))
        self.people_info = tk.Label(f, text="Analyse a check-in recording first.", bg=PANEL, fg=FG_DIM,
                                    font=(FONT, 9), anchor="w", justify="left", wraplength=840)
        self.people_info.pack(fill="x", padx=12)
        self.mode = tk.StringVar(value="merge")
        self.mode_frame = tk.Frame(f, bg=PANEL)
        self.mode_frame.pack(fill="x", padx=12, pady=(2, 8))

        self.list = ScrollFrame(self, bg=BG)
        self.list.grid(row=2, column=0, sticky="nsew", padx=10, pady=6)
        bar = tk.Frame(self, bg=BG)
        bar.grid(row=3, column=0, sticky="ew", padx=10, pady=(0, 10))
        self.fill_btn = button(bar, "Fill in recognised staff", self._fill, CARD2, fg=FG, state="disabled")
        self.fill_btn.pack(side="left")
        button(bar, "Close", self._close, CARD2, fg=FG).pack(side="right")
        self.build_btn = button(bar, "Build gallery", self._build, ACCENT, icon="check", state="disabled")
        self.build_btn.pack(side="right", padx=6)

    def _clear_people(self):
        self.list.clear()
        self._label_vars, self._thumbs, self._people, self._embs = {}, [], [], {}
        for w in self.mode_frame.winfo_children():
            w.destroy()
        self.build_btn.config(state="disabled")
        self.fill_btn.config(state="disabled")

    def _show_people(self):
        day, people = self._date, self._people
        self.existing = describe_existing(self.gallery_file, day, self.app.engine._reid_model_name)
        ex = self.existing
        n_ok = sum(1 for p in people if not self._problem(p))
        self.status.config(text=f"Found {len(people)} people · {n_ok} can be enrolled.",
                           fg=OK if n_ok else WARN)
        if ex is None:
            info = f"No staff are enrolled for {day} yet – this creates the day's gallery."
        elif ex["unreadable"]:
            info = f"The current staff gallery for {day} can't be read, so it can only be replaced."
        else:
            info = (f"{day} already has {len(ex['staff'])} staff ({', '.join(ex['staff']) or 'none'}), "
                    f"from {ex['built_by']}.")
            if not ex["can_merge"]:
                info += f" It was built with another Re-ID model ({ex['model']}), so it can only be replaced."
        self.people_info.config(text=info + f"  Type each employee's label (e.g. emp_04 or a name); blank = "
                                            f"not enrolled. The {self.k} clearest, most varied views of each "
                                            f"person become their entry for this day.")
        if ex is not None:
            self.mode.set("merge" if ex["can_merge"] else "replace")
            radio = dict(variable=self.mode, bg=PANEL, fg=FG, selectcolor=CARD, activebackground=PANEL,
                         font=(FONT, 9))
            tk.Radiobutton(self.mode_frame, text="Add or update these people, keep the others", value="merge",
                           state="normal" if ex["can_merge"] else "disabled", **radio).pack(side="left")
            tk.Radiobutton(self.mode_frame, text="Replace the whole day", value="replace", **radio).pack(
                side="left", padx=12)
            tk.Label(self.mode_frame, text="(the current gallery is backed up first)", bg=PANEL, fg=FG_DIM,
                     font=(FONT, 8)).pack(side="left")

        labels = all_staff_labels(self.gallery_file)
        grid = self.list.inner
        if not people:
            tk.Label(grid, text="No people were found in this recording.", bg=BG, fg=FG_DIM,
                     font=(FONT, 11)).pack(pady=30)
            return
        for p in people:
            problem = self._problem(p)
            row = tk.Frame(grid, bg=CARD, highlightthickness=1, highlightbackground=BORDER)
            row.pack(fill="x", padx=4, pady=3)
            if p["photos"]:
                try:
                    ph = ImageTk.PhotoImage(Image.open(p["photos"][0]).convert("RGB").resize((40, 80)))
                    self._thumbs.append(ph)
                    tk.Label(row, image=ph, bg=CARD).pack(side="left", padx=6, pady=4)
                except Exception:
                    pass
            text = tk.Frame(row, bg=CARD)
            text.pack(side="left", fill="x", expand=True, padx=6)
            tk.Label(text, text=f"{p['id']} · {p['name']}   [{p['cam']}]", bg=CARD,
                     fg=STATUS_COL.get(p["status"], FG), font=(FONT_SEMI, 10), anchor="w").pack(fill="x")
            n = len(self._embs.get(p["id"], ()))
            tk.Label(text, text=f"Can't be enrolled: {problem}" if problem
                     else f"{p['status'].lower()} · {n} clear views · on screen {p['visible_s']} s",
                     bg=CARD, fg=WARN if problem else FG_DIM, font=(FONT, 8), anchor="w").pack(fill="x")
            var = tk.StringVar()
            cb = ttk.Combobox(row, textvariable=var, values=labels, width=22, font=(FONT, 10),
                              state="disabled" if problem else "normal")
            cb.pack(side="right", padx=10)
            if not problem:
                self._label_vars[p["id"]] = (var, p)
        self.build_btn.config(state="normal" if self._label_vars else "disabled")
        self.fill_btn.config(state="normal" if any(self._suggest(p) for _, p in self._label_vars.values())
                             else "disabled")

    def _problem(self, p) -> Optional[str]:
        return eligibility(p["status"], len(self._embs.get(p["id"], ())), self.min_samples)

    @staticmethod
    def _suggest(p) -> str:
        name = p.get("name") or ""
        if p["status"] != "KNOWN":
            return ""
        return "emp_" + name[4:] if name.startswith("Emp ") else name

    def _fill(self):
        for var, p in self._label_vars.values():
            if not var.get().strip():
                var.set(self._suggest(p))

    def _build(self):
        assignments = [{"person": pid, "label": var.get().strip()}
                       for pid, (var, _) in self._label_vars.items() if var.get().strip()]
        if not assignments:
            messagebox.showinfo("Build gallery", "Give at least one person a staff label.", parent=self)
            return
        mode = self.mode.get() if self.existing is not None else "merge"
        if self.existing is not None and mode == "replace" and not messagebox.askyesno(
                "Replace staff gallery", f"Replace the staff gallery for {self._date}? Staff not labelled here "
                                         f"will no longer be recognised in videos from that day (a backup is kept).",
                parent=self):
            return
        people = {p["id"]: p for p in self._people}
        try:
            staff = build_staff(people, self._embs, assignments, k=self.k, min_samples=self.min_samples)
            source = (Path(self.path_var.get()).name if self.kind.get() == "file"
                      else f"{self._date} {self.seq_var.get().split()[0]}")
            s = write_day_gallery(self.gallery_file, self._date, staff, model=self.app.engine._reid_model_name,
                                  mode=mode, source=f"console check-in “{source}”")
        except EnrollmentError as exc:
            messagebox.showerror("Build gallery", str(exc), parent=self)
            return
        except OSError as exc:
            logger.exception("Writing the staff gallery failed")
            messagebox.showerror("Build gallery", f"The staff gallery could not be written ({exc}).\n"
                                                  f"The previous gallery was left in place.", parent=self)
            return
        lines = [f"Added: {', '.join(s['added']) or 'none'}", f"Updated: {', '.join(s['replaced']) or 'none'}",
                 f"Kept: {', '.join(s['kept']) or 'none'}"]
        if s["removed"]:
            lines.append(f"Removed: {', '.join(s['removed'])}")
        if s["backup"]:
            lines.append(f"\nPrevious gallery backed up to {s['backup']}")
        lines.append(f"\n{s['staff']} staff are now enrolled for {s['date']}. Don't test recognition on this "
                     f"same check-in footage.")
        self.app.on_gallery_changed(s["date"])
        messagebox.showinfo("Staff gallery saved", "\n".join(lines), parent=self)
        self._close()

    # ── housekeeping ──────────────────────────────────────────────────────────

    def _discard_work(self):
        if self._work_dir is not None:
            shutil.rmtree(self._work_dir, ignore_errors=True)
            self._work_dir = None

    def _close(self):
        if self._thread is not None and self._thread.is_alive():
            if not messagebox.askyesno("Check-in", "Stop the analysis and close?", parent=self):
                return
            self._stop.set()
            self._thread.join(timeout=5)
        self._discard_work()
        self.destroy()
