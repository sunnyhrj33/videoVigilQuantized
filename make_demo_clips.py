"""
make_demo_clips.py
────────────────────────────────────────────────────────────────────────────────
Select and export the three evaluation demo clips from CHIRLA using the
ground-truth annotations (never the enrollment sequence of the person shown):

  Demo 1 – Known employee persistence: an enrolled employee is visible, briefly
           disappears (occlusion / leaves view) and reappears.
  Demo 2 – Open-set unknown rejection: a never-enrolled visitor is clearly
           visible for several seconds.
  Demo 3 – Asset interaction: a visitor's box overlaps a laptop / backpack /
           handbag that YOLO detects. A restricted zone around that object is
           suggested for the camera's per_camera config.

Clips are written to output/demo/ with names that keep the CHIRLA camera and
(shifted) start time, so the app's Camera_ID and audit timestamps stay correct.
A manifest (output/demo/demo_clips.json) records what each clip contains.

Usage:  python make_demo_clips.py [--config config.yaml] [--seconds 25]
────────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from datetime import timedelta
from pathlib import Path

import cv2

from chirla import (day_gallery_path, emp_label, employee_ids, parse_video_name,
                    sequence_dates, video_path, visitor_ids)
from detection_engine import DetectionEngine
from open_set import PrototypeGallery

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("make_demo_clips")

FPS = 30
ASSET_CLASSES = {24: "backpack", 26: "handbag", 63: "laptop"}


def load_annotations(root: Path):
    """{(seq, camera): {frame: [(pid, box)]}} with 0-based frame indices."""
    out = {}
    for f in sorted((root / "annotations").glob("seq_*/camera_*.json")):
        cam = f.stem.split("_20")[0]
        data = json.loads(f.read_text())
        out[(f.parent.name, cam)] = {
            int(k) - 1: [(int(p["id"]), tuple(int(v) for v in p["BboxP"])) for p in v]
            for k, v in data.items()
        }
    return out


def presence(frames: dict, pid: int, min_h: int = 120):
    """Sorted frames where pid is visible with a box at least min_h tall."""
    return sorted(f for f, people in frames.items()
                  for p, b in people if p == pid and b[3] - b[1] >= min_h)


def best_persistence_window(pres, length):
    """Window with high coverage that contains a 1–5 s gap followed by reappearance."""
    if not pres:
        return None
    pres_set = set(pres)
    best = None
    for start in pres[::15]:
        end = start + length
        cov = sum(1 for f in range(start, end) if f in pres_set) / length
        inside = [f for f in pres if start <= f < end]
        gaps = [b - a for a, b in zip(inside, inside[1:]) if 30 <= b - a <= 150]
        if gaps and cov > 0.5:
            score = cov + 0.2 * min(len(gaps), 2)
            if best is None or score > best[0]:
                best = (score, start, cov, max(gaps))
    return best


def best_visible_window(pres, length):
    if not pres:
        return None
    pres_set = set(pres)
    best = None
    for start in pres[::15]:
        s = max(0, start - 60)                     # start shortly before they enter
        cov = sum(1 for f in range(s, s + length) if f in pres_set) / length
        if best is None or cov > best[0]:
            best = (cov, s)
    return best


def export_clip(src: Path, start: int, length: int, out: Path):
    cap = cv2.VideoCapture(str(src))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))
    n = 0
    while n < length:
        ok, frame = cap.read()
        if not ok:
            break
        vw.write(frame)
        n += 1
    vw.release()
    cap.release()
    return n


def clip_name(tag: str, src: Path, start: int) -> str:
    cam, t0 = parse_video_name(src.name)
    t = t0 + timedelta(seconds=start / FPS)
    return f"{tag}__camera_{int(cam.split('_')[1])}_{t:%Y-%m-%d-%H_%M_%S}.mp4"


def find_asset_scene(root, ann, visitors, length, model):
    """Sample frames where a visitor is visible; look for visitor ∩ monitored object."""
    best = None
    for (seq, cam), frames in sorted(ann.items()):
        src = video_path(root, seq, cam)
        if src is None:
            continue
        vis_frames = sorted(f for f, people in frames.items()
                            if any(p in visitors and b[3] - b[1] >= 100 for p, b in people))
        if not vis_frames:
            continue
        cap = cv2.VideoCapture(str(src))
        for f in vis_frames[::45]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, frame = cap.read()
            if not ok:
                continue
            res = model.predict(frame, classes=list(ASSET_CLASSES), conf=0.30, verbose=False)[0]
            for box, cls, conf in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.cls.cpu().numpy(),
                                      res.boxes.conf.cpu().numpy()):
                ob = tuple(int(v) for v in box)
                area = max(1, (ob[2] - ob[0]) * (ob[3] - ob[1]))
                for pid, pb in frames.get(f, []):
                    if pid not in visitors:
                        continue
                    inter = max(0, min(pb[2], ob[2]) - max(pb[0], ob[0])) * \
                        max(0, min(pb[3], ob[3]) - max(pb[1], ob[1]))
                    score = inter / area * float(conf)
                    if inter / area > 0.2 and (best is None or score > best["score"]):
                        best = {"score": score, "seq": seq, "camera": cam, "pid": pid,
                                "frame": f, "object": ASSET_CLASSES[int(cls)], "object_box": ob,
                                "conf": float(conf), "size": frame.shape[1::-1]}
        cap.release()
        if best:
            logger.info("  asset candidate so far: %s", {k: best[k] for k in ("seq", "camera", "pid", "object", "conf")})
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--seconds", type=int, default=25)
    ap.add_argument("--model", default="yolo11s_int8", help="config models key used to find assets")
    args = ap.parse_args()

    cfg = DetectionEngine.load_config(args.config)
    ch = cfg["chirla"]
    root = Path(ch["root"])
    length = args.seconds * FPS
    gap = int(ch.get("min_gap_after_enroll", 600))
    test_days = set(ch.get("test_days", []))
    employees, visitors = set(employee_ids(cfg)), set(visitor_ids(cfg))
    dates = sequence_dates(root)
    galleries = {}
    for day in set(dates.values()):
        p = day_gallery_path(cfg["reid"]["gallery_file"], day)
        if p.exists():
            galleries[day] = PrototypeGallery.load(p)

    out_dir = Path("output/demo")
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Loading annotations …")
    ann = load_annotations(root)
    manifest = {}

    def people_in(frames, start):
        ids = defaultdict(int)
        for f in range(start, start + length):
            for p, _ in frames.get(f, []):
                ids[p] += 1
        return {"employees": sorted(p for p, n in ids.items() if p in employees and n > 30),
                "visitors": sorted(p for p, n in ids.items() if p in visitors and n > 30)}

    # ── Demo 1: known employee persistence ──────────────────────────────────
    # Only an employee enrolled that day, only footage after their enrollment cut-off.
    best = None
    for (seq, cam), frames in ann.items():
        g = galleries.get(dates[seq])
        if g is None:
            continue
        for pid in employees:
            label = emp_label(pid)
            if label not in g.labels:
                continue
            c_seq, c_frame = g.meta["enroll_cutoff"][label]
            if seq < c_seq:
                continue
            pres = [f for f in presence(frames, pid) if seq > c_seq or f > c_frame + gap]
            r = best_persistence_window(pres, length)
            if r and (seq > c_seq or r[1] > c_frame + gap):
                crowd = people_in(frames, r[1])
                score = r[0] - 0.1 * len(crowd["visitors"]) + (0.3 if dates[seq] in test_days else 0)
                if best is None or score > best[0]:
                    best = (score, seq, cam, pid, r[1], r[2], r[3])
    if best:
        _, seq, cam, pid, start, cov, gap = best
        src = video_path(root, seq, cam)
        name = clip_name("demo1_known_persistence", src, start)
        n = export_clip(src, start, length, out_dir / name)
        manifest["demo1"] = {"file": name, "source": f"{seq}/{cam}", "date": dates[seq],
                             "start_frame": start, "frames": n,
                             "focus": f"emp_{pid:02d}", "coverage": round(cov, 2),
                             "longest_gap_s": round(gap / FPS, 1), **people_in(ann[(seq, cam)], start)}
        logger.info("Demo 1 → %s (emp_%02d, gap %.1fs)", name, pid, gap / FPS)

    # ── Demo 2: unknown visitor rejection ───────────────────────────────────
    best = None
    for (seq, cam), frames in ann.items():
        for pid in visitors:
            r = best_visible_window(presence(frames, pid, min_h=160), length)
            if r and dates[seq] in galleries:
                score = r[0] + (0.3 if dates[seq] in test_days else 0)
                if best is None or score > best[0]:
                    best = (score, seq, cam, pid, r[1], r[0])
    if best:
        _, seq, cam, pid, start, cov = best
        src = video_path(root, seq, cam)
        name = clip_name("demo2_unknown_rejection", src, start)
        n = export_clip(src, start, length, out_dir / name)
        manifest["demo2"] = {"file": name, "source": f"{seq}/{cam}", "date": dates[seq],
                             "start_frame": start, "frames": n,
                             "focus": f"visitor {pid}", "coverage": round(cov, 2),
                             **people_in(ann[(seq, cam)], start)}
        logger.info("Demo 2 → %s (visitor %d, coverage %.0f%%)", name, pid, cov * 100)

    # ── Demo 3: asset interaction ───────────────────────────────────────────
    from ultralytics import YOLO
    model = YOLO(cfg["models"][args.model], task="detect")
    logger.info("Searching for a visitor touching a monitored asset (YOLO %s) …", args.model)
    hit = find_asset_scene(root, ann, visitors, length, model)
    if hit:
        start = max(0, hit["frame"] - length // 2)
        src = video_path(root, hit["seq"], hit["camera"])
        name = clip_name("demo3_asset_interaction", src, start)
        n = export_clip(src, start, length, out_dir / name)
        w, h = hit["size"]
        x1, y1, x2, y2 = hit["object_box"]
        pad_x, pad_y = (x2 - x1) * 1.5, (y2 - y1) * 1.5
        zone = [[round(max(0, x1 - pad_x) / w, 3), round(max(0, y1 - pad_y) / h, 3)],
                [round(min(w, x2 + pad_x) / w, 3), round(max(0, y1 - pad_y) / h, 3)],
                [round(min(w, x2 + pad_x) / w, 3), round(min(h, y2 + pad_y) / h, 3)],
                [round(max(0, x1 - pad_x) / w, 3), round(min(h, y2 + pad_y) / h, 3)]]
        manifest["demo3"] = {"file": name, "source": f"{hit['seq']}/{hit['camera']}",
                             "date": dates[hit["seq"]], "start_frame": start,
                             "frames": n, "focus": f"visitor {hit['pid']} at {hit['object']}",
                             "object_conf": round(hit["conf"], 2), "suggested_zone": {hit["camera"]: zone},
                             **people_in(ann[(hit["seq"], hit["camera"])], start)}
        logger.info("Demo 3 → %s (visitor %d + %s)  suggested zone for %s: %s",
                    name, hit["pid"], hit["object"], hit["camera"], zone)
    else:
        logger.warning("Demo 3: no frame found where YOLO sees a monitored asset overlapping a visitor.")

    (out_dir / "demo_clips.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
