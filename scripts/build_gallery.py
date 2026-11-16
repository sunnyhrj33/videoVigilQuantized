"""
scripts/build_gallery.py
────────────────────────────────────────────────────────────────────────────────
Offline daily enrollment of the CHIRLA "employees" into persistent galleries.

Usage:  python scripts/build_gallery.py [--config config.yaml]
────────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import argparse
import logging
import pickle
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.chirla import (day_gallery_path, emp_label, employee_ids, imread,  # noqa: E402
                         index_benchmark_crops, sequence_dates, visitor_ids)
from core.engine import DetectionEngine  # noqa: E402
from core.geometry import l2n, sharpness  # noqa: E402
from core.reid import ReIDExtractor  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("build_gallery")

EARLY_FRACTIONS = (0.3, 0.5, 1.0)


def select_crops(records, n: int):
    """Score crops by height × sharpness, then greedily pick diverse ones."""
    scored = []
    for r in records:
        img = imread(r.path)
        if img is None:
            continue
        h, w = img.shape[:2]
        if h < 80 or h < 1.2 * w:
            continue
        scored.append((h * min(sharpness(img), 300.0), r, img))
    scored.sort(key=lambda s: -s[0])

    chosen, used = [], defaultdict(list)
    for _, r, img in scored:
        if any(abs(r.frame - f) < 90 for f in used[(r.seq, r.camera)]):
            continue
        chosen.append((r, img))
        used[(r.seq, r.camera)].append(r.frame)
        if len(chosen) == n:
            break
    return chosen


def enroll_person(person_records, extractor, n_crops):
    """Return (chosen [(record, img)], embeddings, cutoff_record, n_outliers) or None."""
    ordered = sorted(person_records, key=lambda r: (r.seq, r.frame))
    for frac in EARLY_FRACTIONS:
        pool = ordered[:max(8, int(len(ordered) * frac))]
        chosen = select_crops(pool, n_crops + 3)
        if len(chosen) >= 3:
            break
    else:
        return None

    embs = l2n(extractor.extract_batch([img for _, img in chosen]))
    typicality = np.array([
        float(embs[i] @ l2n(np.delete(embs, i, axis=0).mean(axis=0))) for i in range(len(embs))
    ])
    keep = [int(i) for i in np.argsort(-typicality) if typicality[i] >= np.median(typicality) - 0.15]
    dropped = len(chosen) - len(keep)
    keep = keep[:n_crops]
    return [chosen[i] for i in keep], embs[keep], pool[-1], dropped


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    cfg = DetectionEngine.load_config(args.config)
    ch, reid = cfg["chirla"], cfg["reid"]
    n_crops = int(ch.get("crops_per_employee", 5))
    gallery_file = Path(reid["gallery_file"])
    employees = employee_ids(cfg)

    dates = sequence_dates(ch["root"])
    records = index_benchmark_crops(ch["root"])
    logger.info("Indexed %d CHIRLA crops over %d days", len(records), len(set(dates.values())))
    by_day_pid = defaultdict(list)
    for r in records:
        by_day_pid[(dates[r.seq], r.pid)].append(r)

    extractor = ReIDExtractor(reid["model"], device=reid.get("device", "cpu"),
                              weights=reid.get("weights", "auto"))

    for day in sorted(set(dates.values())):
        out_file = day_gallery_path(gallery_file, day)
        out_dir = out_file.parent
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True)

        prototypes, sources, cutoff = {}, {}, {}
        for pid in employees:
            recs = by_day_pid.get((day, pid))
            if not recs:
                continue
            label = emp_label(pid)
            res = enroll_person(recs, extractor, n_crops)
            if res is None:
                logger.warning("%s %s: too few clear crops – not enrolled", day, label)
                continue
            chosen, embs, last, dropped = res
            (out_dir / label).mkdir()
            for k, (r, img) in enumerate(chosen, 1):
                cv2.imwrite(str(out_dir / label / f"crop{k}.jpg"), img)
            prototypes[label] = l2n(embs.mean(axis=0))
            sources[label] = [f"{r.seq}/{r.camera}/frame_{r.frame}" for r, _ in chosen]
            cutoff[label] = (last.seq, last.frame)
            logger.info("%s %s ← %d crops (%s), cut-off %s@%d, %d outlier(s) rejected", day, label,
                        len(chosen), ", ".join(sorted({r.camera for r, _ in chosen})),
                        last.seq, last.frame, dropped)

        meta = {"model": reid["model"], "date": day, "employees": employees,
                "visitors": visitor_ids(cfg), "sources": sources, "enroll_cutoff": cutoff}
        with open(out_file, "wb") as f:
            pickle.dump({"prototypes": prototypes, "meta": meta}, f)
        logger.info("✓ %s: %d employees enrolled → %s", day, len(prototypes), out_file)

    demo_day = ch.get("demo_day")
    if demo_day and day_gallery_path(gallery_file, demo_day).exists():
        shutil.copyfile(day_gallery_path(gallery_file, demo_day), gallery_file)
        logger.info("✓ Default gallery = %s → %s", demo_day, gallery_file)


if __name__ == "__main__":
    main()
