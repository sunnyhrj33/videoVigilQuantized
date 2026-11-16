"""
core/enrollment.py
────────────────────────────────────────────────────────────────────────────────
Daily staff enrollment from a "morning check-in" recording (web dashboard):

  • daily_prototype  – K sharp, typical, spread-out samples → P_daily = l2n(mean)
  • check_label      – a staff label that is safe as a gallery key and folder name
  • write_day_gallery – enrolled_gallery/<date>/ (same format as build_gallery.py),
                        merge or replace, old folder backed up, atomic swap
────────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import logging
import pickle
import re
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from core.chirla import is_iso_date
from core.geometry import l2n

logger = logging.getLogger(__name__)

LABEL_RE = re.compile(r"^[A-Za-z0-9][\w .'\-]{0,38}[A-Za-z0-9_]$")
_FOLDER_RE = re.compile(r"^[A-Za-z0-9_\-]{1,40}$")
_WINDOWS_RESERVED = {"con", "prn", "aux", "nul", *(f"com{i}" for i in range(1, 10)),
                     *(f"lpt{i}" for i in range(1, 10))}
BACKUP_DIR = "_backups"


class EnrollmentError(ValueError):
    """A problem with an enrollment request, worded for the dashboard user."""


def check_label(label) -> str:
    """'emp_04', 'Sara Khan' … – 2–40 letters / digits / space . ' - _, starting and ending with one."""
    label = " ".join(str(label or "").split())
    if not LABEL_RE.match(label) or label.lower() in _WINDOWS_RESERVED:
        raise EnrollmentError(f"“{label[:40]}” can't be used as a staff label – use 2–40 letters, numbers, "
                              f"spaces, dots, dashes or apostrophes (e.g. emp_04 or Sara Khan).")
    return label


def label_folder(label: str, taken: set) -> str:
    """Gallery sub-folder for a label: the label itself when it is a plain name, else a safe slug."""
    if _FOLDER_RE.match(label) and label.lower() not in taken:
        return label
    base = re.sub(r"[^A-Za-z0-9_-]+", "_", label).strip("_")[:30] or "staff"
    folder = base
    while folder.lower() in taken or folder.lower() in _WINDOWS_RESERVED:
        folder = f"{base}_{uuid.uuid4().hex[:4]}"
    return folder


def daily_prototype(embs: np.ndarray, k: int = 5, min_samples: int = 3,
                    outlier_margin: float = 0.15) -> tuple:
    """
    P_daily from one person's check-in embeddings:
      1. drop atypical samples (cosine to the mean of the others < median − margin),
      2. pick k spread-out samples (different viewpoints), starting from the most typical,
      3. P_daily = l2n(mean of the k samples).
    Returns (prototype, chosen_indices, n_outliers). Raises EnrollmentError if too few samples.
    """
    e = l2n(np.asarray(embs, np.float32).reshape(len(embs), -1)) if len(embs) else np.zeros((0, 1))
    if len(e) < min_samples:
        raise EnrollmentError(f"only {len(e)} clear view{'s' if len(e) != 1 else ''} (need {min_samples})")
    total = e.sum(axis=0)
    typicality = np.array([float(e[i] @ l2n(total - e[i])) for i in range(len(e))])
    keep = [i for i in np.argsort(-typicality) if typicality[i] >= np.median(typicality) - outlier_margin]
    if len(keep) < min_samples:
        raise EnrollmentError(f"the views disagree too much ({len(keep)} consistent, need {min_samples})")
    chosen = [int(keep[0])]
    while len(chosen) < min(k, len(keep)):
        rest = [i for i in keep if i not in chosen]
        sims = e[rest] @ e[chosen].T                     # (rest, chosen)
        chosen.append(int(rest[int(np.argmin(sims.max(axis=1)))]))
    return l2n(e[chosen].mean(axis=0)), chosen, len(e) - len(keep)


SAME_STAFF_MIN_SIM = 0.6       # two cards given one label must look at least this alike


def eligibility(status: str, n_samples: int, min_samples: int) -> Optional[str]:
    """Why a detected person can't be enrolled (None = they can)."""
    if status == "SUSPECT":
        return "on the watch list – remove them from it first"
    if n_samples < min_samples:
        return f"only {n_samples} clear view{'s' if n_samples != 1 else ''} (need {min_samples})"
    return None


def build_staff(people: Dict[str, dict], embs: Dict[str, np.ndarray], assignments, *, k: int,
                min_samples: int, reserved: Optional[set] = None) -> List[dict]:
    """
    Turn labelled check-in people into gallery entries for write_day_gallery.
    people: {pid: {"status", "name", "photos": [Path]}}; embs: {pid: (n, d)};
    assignments: [{"person": pid, "label": str}] (blank label = not enrolled);
    reserved: lower-case names that can't be staff labels (people named in the dashboard).
    Several cards with one label are pooled if they look like one person. Every problem
    is collected; any problem raises EnrollmentError("Nothing was saved: …").
    """
    if not isinstance(assignments, list):
        raise EnrollmentError("Send the people to enroll as a list.")
    reserved = reserved or set()
    errors, by_label, seen = [], {}, set()
    for a in assignments:
        a = a if isinstance(a, dict) else {}
        pid = str(a.get("person", ""))
        raw = " ".join(str(a.get("label") or "").split())
        if not raw:
            continue
        if pid not in people:
            errors.append(f"{pid or '?'}: no such person in this video")
            continue
        if pid in seen:
            errors.append(f"{pid}: listed twice")
            continue
        seen.add(pid)
        try:
            label = check_label(raw)
        except EnrollmentError as exc:
            errors.append(f"{pid}: {exc}")
            continue
        if label.lower() in reserved:
            errors.append(f"{pid}: “{label}” is already one of the people you named – pick another label")
            continue
        problem = eligibility(people[pid]["status"], len(embs.get(pid, ())), min_samples)
        if problem:
            errors.append(f"{pid} ({people[pid].get('name', '')}): {problem}")
            continue
        by_label.setdefault(label, []).append(pid)
    if not by_label and not errors:
        raise EnrollmentError("Give at least one person a staff label.")

    staff = []
    for label, pids in by_label.items():
        if len(pids) > 1:                            # one person split over several cards?
            cents = [l2n(l2n(embs[p]).mean(axis=0)) for p in pids]
            worst = min(float(a @ b) for i, a in enumerate(cents) for b in cents[i + 1:])
            if worst < SAME_STAFF_MIN_SIM:
                errors.append(f"{' and '.join(pids)} were both labelled “{label}” but don't look like "
                              f"the same person (similarity {worst:.2f})")
                continue
        try:
            proto, chosen, outliers = daily_prototype(np.concatenate([embs[p] for p in pids]), k, min_samples)
        except EnrollmentError as exc:
            errors.append(f"{label}: {exc}")
            continue
        photos = [Path(ph) for p in pids for ph in people[p].get("photos", []) if Path(ph).exists()][:k]
        if not photos:
            errors.append(f"{label}: no photos were saved for {' / '.join(pids)}")
            continue
        staff.append({"label": label, "prototype": proto, "photos": photos, "samples": len(chosen),
                      "outliers": outliers, "people": pids})
    if errors:
        raise EnrollmentError("Nothing was saved:\n" + "\n".join(errors))
    return staff


def read_day_gallery(gallery_file: str | Path, date: str) -> Optional[dict]:
    """The pickle of enrolled_gallery/<date>/ ({"prototypes", "meta"}), None if absent / unreadable."""
    path = Path(gallery_file).parent / date / Path(gallery_file).name
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as exc:
        logger.warning("Could not read %s (%s)", path, exc)
        return {"prototypes": {}, "meta": {}, "unreadable": True}


def all_staff_labels(gallery_file: str | Path) -> List[str]:
    """Every staff label enrolled on any day (suggestions, e.g. emp_04)."""
    root, name = Path(gallery_file).parent, Path(gallery_file).name
    labels = set()
    for f in root.glob(f"*/{name}"):
        if is_iso_date(f.parent.name):
            labels |= set((read_day_gallery(gallery_file, f.parent.name) or {}).get("prototypes", {}))
    return sorted(labels)


def describe_existing(gallery_file: str | Path, date: str, model: Optional[str]) -> Optional[dict]:
    """The day's current gallery: {staff, unreadable, model, can_merge, built_by}, None if there is none."""
    old = read_day_gallery(gallery_file, date)
    if old is None:
        return None
    meta = old.get("meta", {})
    model_ok = not meta.get("model") or not model or meta["model"] == model
    return {"staff": sorted(old.get("prototypes", {})), "unreadable": bool(old.get("unreadable")),
            "model": meta.get("model"), "can_merge": model_ok and not old.get("unreadable"),
            "built_by": meta.get("built_by", "offline enrollment (build_gallery.py)")}


def write_day_gallery(gallery_file: str | Path, date: str, staff: List[dict], *, model: str,
                      mode: str = "merge", source: str = "dashboard") -> dict:
    """
    Write enrolled_gallery/<date>/gallery_prototypes.pkl + <folder>/cropK.jpg.
    staff: [{"label", "prototype", "photos": [Path], "samples": int, "outliers": int}].
    mode "merge" keeps the day's other staff (same label → replaced); "replace" drops them.
    The previous folder is moved to enrolled_gallery/_backups/<date>_<time>/ first; the new
    one is assembled next to it and swapped in, so a failure leaves the old gallery in place.
    """
    if not is_iso_date(date):
        raise EnrollmentError(f"“{date}” is not a valid date")
    if mode not in ("merge", "replace"):
        raise EnrollmentError("mode must be “merge” or “replace”")
    if not staff:
        raise EnrollmentError("Assign a staff label to at least one person.")
    root = Path(gallery_file).parent
    name = Path(gallery_file).name
    day_dir = root / date
    old = read_day_gallery(gallery_file, date)
    old_meta = (old or {}).get("meta", {})
    if old and mode == "merge":
        if old.get("unreadable"):
            raise EnrollmentError(f"The existing staff gallery for {date} can't be read – choose “replace”.")
        if old_meta.get("model") and old_meta["model"] != model:
            raise EnrollmentError(f"The existing staff gallery for {date} was built with another Re-ID model "
                                  f"({old_meta['model']}) – its people can't be mixed with new ones; "
                                  f"choose “replace”.")

    prototypes: Dict[str, np.ndarray] = {}
    folders: Dict[str, str] = {}
    sources: Dict[str, list] = {}
    kept = []
    new_labels = {s["label"] for s in staff}
    tmp = root / f".{date}.building-{uuid.uuid4().hex[:6]}"
    tmp.mkdir(parents=True)
    try:
        if old and mode == "merge":
            old_folders = old_meta.get("folders", {})
            for label, proto in old.get("prototypes", {}).items():
                if label in new_labels:
                    continue
                prototypes[label] = proto
                src = day_dir / old_folders.get(label, label)
                folder = old_folders.get(label, label)
                if src.is_dir():
                    shutil.copytree(src, tmp / folder)
                folders[label] = folder
                sources[label] = old_meta.get("sources", {}).get(label, [])
                kept.append(label)
        taken = {f.lower() for f in folders.values()}
        for s in staff:
            folder = label_folder(s["label"], taken)
            taken.add(folder.lower())
            (tmp / folder).mkdir()
            for k, photo in enumerate(s["photos"], 1):
                shutil.copyfile(photo, tmp / folder / f"crop{k}.jpg")
            prototypes[s["label"]] = np.asarray(s["prototype"], np.float32)
            folders[s["label"]] = folder
            sources[s["label"]] = [f"{source} · {s.get('samples', 0)} samples, "
                                   f"{s.get('outliers', 0)} outlier(s) rejected"]
        meta = dict(old_meta) if mode == "merge" else {}
        if "enroll_cutoff" in meta:          # CHIRLA cut-offs only stay valid for the kept staff
            meta["enroll_cutoff"] = {k: v for k, v in meta["enroll_cutoff"].items() if k in kept}
        meta.update(model=model, date=date, sources=sources, folders=folders,
                    built_by="dashboard check-in", updated=datetime.now().isoformat(timespec="seconds"))
        with open(tmp / name, "wb") as f:
            pickle.dump({"prototypes": prototypes, "meta": meta}, f)

        backup = None
        if day_dir.exists():
            backup = root / BACKUP_DIR / f"{date}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(day_dir), str(backup))
        try:
            tmp.rename(day_dir)
        except Exception:
            if backup is not None and not day_dir.exists():
                shutil.move(str(backup), str(day_dir))      # put the old gallery back
            raise
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    replaced = sorted(new_labels & set((old or {}).get("prototypes", {}))) if old else []
    logger.info("✓ Staff gallery %s written (%s): %d new / %d kept, backup %s",
                date, mode, len(staff), len(kept), backup)
    return {"date": date, "mode": mode, "added": sorted(new_labels - set(replaced)), "replaced": replaced,
            "kept": sorted(kept), "removed": sorted(set((old or {}).get("prototypes", {})) - new_labels - set(kept)),
            "staff": len(prototypes), "backup": str(backup) if backup else None}
