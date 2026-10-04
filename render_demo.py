"""
render_demo.py
────────────────────────────────────────────────────────────────────────────────
Run the full pipeline (detection → tracking → open-set identity → zones /
assets → audit log) over a clip offline, frame by frame, and write a smooth
annotated MP4 for presentations – real-time speed on a CPU is not needed.

Prints a summary: identities resolved, UNKNOWN tracks, alerts raised.

Usage:
  python render_demo.py demo2 --mode naive          # one demo, one mode
  python render_demo.py all                          # all demos (Demo 2 in both modes)
  python render_demo.py path/to/clip.mp4 --mode tracklet

Outputs: output/rendered/<clip>__<mode>.mp4 and the shared output/audit_log.csv
────────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections import Counter
from pathlib import Path

import cv2

from detection_engine import DetectionEngine

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("render_demo")

DEMO_DIR = Path("output/demo")
OUT_DIR = Path("output/rendered")


def render(engine: DetectionEngine, clip: Path, mode: str) -> dict:
    engine.set_identity_mode(mode)
    cap = cv2.VideoCapture(str(clip))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{clip.stem.split('__')[0]}__{mode}.mp4"
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    engine.set_source(str(clip), fps)
    engine.show_fps = False

    final_status = {}
    last_event = 0
    events = []
    t0 = time.time()
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        annotated, stats = engine.process_frame(frame)
        writer.write(annotated)
        n += 1
        for p in stats.get("people", []):
            if p["status"] != "PENDING":
                final_status[p["tid"]] = (p["status"], p["name"])
        new = engine.events_since(last_event)
        if new:
            last_event = new[-1][0]
            events += new
        if n % 150 == 0:
            print(f"    {n}/{total} frames  ({n / (time.time() - t0):.1f} fps)", flush=True)
    writer.release()
    cap.release()

    statuses = Counter(s for s, _ in final_status.values())
    names = Counter(name for s, name in final_status.values() if s == "KNOWN")
    alerts = [e for e in events if e[1] == "alert"]
    return {
        "clip": clip.name, "mode": mode, "frames": n, "seconds": round(time.time() - t0),
        "output": str(out_path),
        "tracks_resolved": len(final_status),
        "known": statuses.get("KNOWN", 0), "unknown": statuses.get("UNKNOWN", 0),
        "known_names": dict(names),
        "alerts": Counter(e[4] for e in alerts),
        "asset_events": [e[4] for e in events if e[1] == "asset"],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("target", help="demo1 | demo2 | demo3 | all | path to a clip")
    ap.add_argument("--mode", default="tracklet", choices=["tracklet", "naive", "legacy"])
    ap.add_argument("--model", default="yolo11n_int8_480")
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    cfg = DetectionEngine.load_config(args.config)
    engine = DetectionEngine(cfg)
    engine.load_model(args.model)

    manifest = json.loads((DEMO_DIR / "demo_clips.json").read_text()) if (DEMO_DIR / "demo_clips.json").exists() else {}
    if args.target == "all":
        jobs = [("demo1", "tracklet"), ("demo2", "naive"), ("demo2", "tracklet"), ("demo3", "tracklet")]
    else:
        jobs = [(args.target, args.mode)]

    results = []
    for target, mode in jobs:
        clip = DEMO_DIR / manifest[target]["file"] if target in manifest else Path(target)
        print(f"▶ {clip.name}  [{mode}]", flush=True)
        r = render(engine, clip, mode)
        results.append(r)
        print(f"  ✓ {r['output']}  ·  {r['frames']} frames in {r['seconds']}s  ·  "
              f"tracks: {r['known']} KNOWN / {r['unknown']} UNKNOWN  ·  names {r['known_names']}  ·  "
              f"alerts {dict(r['alerts'])}  ·  asset events {r['asset_events']}", flush=True)
    engine.close()
    (OUT_DIR / "render_summary.json").write_text(json.dumps(results, indent=2, default=dict))


if __name__ == "__main__":
    main()
