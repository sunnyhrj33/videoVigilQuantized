"""
video_search.py
────────────────────────────────────────────────────────────────────────────────
Video-wide person search: find a person (suspect, employee, a person picked in
the live view, or a photo) anywhere in the recorded CCTV.

1. INDEX (offline, once per recording day)
   Every video is sampled every `stride_frames` frames; YOLO + ByteTrack link
   people into tracks; sharp crops are embedded with OSNet (OpenVINO) and each
   track is stored as tracklets of ≤ `tracklet_window` crops with a centroid
   embedding, a thumbnail, time span, and P(Known) / best employee from that
   day's enrolled gallery. CHIRLA ground-truth IDs are stored for evaluation
   only (never used for searching).
       output/index/<date>/<seq>__<camera>.pkl

2. SEARCH (instant)
   Query embeddings → centroid → cosine similarity to every indexed tracklet,
   ranked best-first, one result per (video, track). Optionally drops
   tracklets confidently recognised as enrolled employees (the right default
   when looking for an unknown person).

Single crops are a poor match signal on CHIRLA (same/different-person cosine
distributions overlap strongly), which is why both sides are tracklets.

Usage
  python video_search.py index --day 2023-06-08          # one recording day
  python video_search.py index --all                     # every CHIRLA day
  python video_search.py index --videos a.mp4 b.mp4      # any recordings
  python video_search.py eval --day 2023-06-08           # GT retrieval check
────────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import pickle
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

import cv2
import numpy as np

from chirla import camera_key, day_gallery_path, parse_video_name, sequence_dates, source_date
from open_set import OpenSetClassifier, PrototypeGallery, crop_is_usable, l2n, tracklet_features

logger = logging.getLogger("video_search")
INDEX_VERSION = 1


def _cfg(cfg: dict) -> dict:
    return cfg.get("video_search", {}) or {}


def index_dir(cfg: dict) -> Path:
    return Path(_cfg(cfg).get("index_dir", "output/index"))


# ─────────────────────────────────────────────────────────────────────────────
#  Indexing
# ─────────────────────────────────────────────────────────────────────────────

def _iou(a, b) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    return i / max(1e-9, (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i)


def _chirla_annotations(video: Path) -> Optional[dict]:
    """CHIRLA GT boxes for a video (…/videos/seq_x/camera_y_….avi), if available."""
    try:
        root = video.parent.parent.parent
        files = glob.glob(str(root / "annotations" / video.parent.name / f"{camera_key(video.name)}_*.json"))
        return json.loads(Path(files[0]).read_text()) if files else None
    except Exception:
        return None


class VideoIndexer:
    def __init__(self, cfg: dict, model_key: Optional[str] = None):
        from detection_engine import DetectionEngine, ReIDExtractor  # heavy imports
        from ultralytics import YOLO

        self.cfg = cfg
        vs, reid, trk = _cfg(cfg), cfg["reid"], cfg["tracking"]
        self.stride = int(vs.get("stride_frames", 15))
        self.min_conf = float(vs.get("min_person_conf", 0.5))
        self.window = int(trk.get("tracklet_window", 10))
        q = reid.get("crop_quality", {})
        self.min_height = int(vs.get("min_height", q.get("min_height", 64)))
        self.min_sharp = float(q.get("min_sharpness", 20.0))
        key = model_key or vs.get("detector", "yolo11n_int8_480")
        path = Path(cfg["models"][key])
        self.model = YOLO(str(path), task="detect")
        meta = path / "metadata.yaml"
        import yaml
        sz = (yaml.safe_load(meta.read_text()) or {}).get("imgsz") if meta.exists() else None
        self.imgsz = int(sz[0] if isinstance(sz, (list, tuple)) else sz) if sz else 640
        self.extractor = ReIDExtractor(reid["model"], reid.get("device", "cpu"), reid.get("weights", "auto"))
        self.classifier = OpenSetClassifier(reid.get("open_set_model"))
        self.p_thresh = float(reid.get("open_set_p_threshold", 0.5))
        self.gallery_file = reid.get("gallery_file", "enrolled_gallery/gallery_prototypes.pkl")
        self._galleries: Dict[str, Optional[PrototypeGallery]] = {}
        self._DetectionEngine = DetectionEngine

    def _gallery(self, date: Optional[str]) -> Optional[PrototypeGallery]:
        if date not in self._galleries:
            p = day_gallery_path(self.gallery_file, date)
            p = p if p.exists() else Path(self.gallery_file)
            self._galleries[date] = PrototypeGallery.load(p) if p.exists() else None
        return self._galleries[date]

    def _reset_tracker(self):
        predictor = getattr(self.model, "predictor", None)
        for t in getattr(predictor, "trackers", None) or []:
            t.reset()

    def index_video(self, video: Path, progress: Optional[Callable[[int, int], None]] = None) -> dict:
        cap = cv2.VideoCapture(str(video))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cam, start = parse_video_name(video.name)
        date = source_date(video.name)
        ann = _chirla_annotations(video)
        gallery = self._gallery(date)
        self._reset_tracker()

        tracks: Dict[int, list] = defaultdict(list)     # tid → [(frame, box, emb, thumb, gt)]
        f = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            res = self.model.track(frame, persist=True, tracker="bytetrack.yaml", classes=[0],
                                   conf=self.min_conf, imgsz=self.imgsz, verbose=False)[0]
            crops, owners = [], []
            if res.boxes is not None and res.boxes.id is not None:
                for b, c, tid in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy(),
                                     res.boxes.id.cpu().numpy()):
                    x1, y1, x2, y2 = (int(v) for v in b)
                    crop = frame[max(0, y1):y2, max(0, x1):x2]
                    if crop_is_usable(crop, float(c), self.min_height, self.min_conf, self.min_sharp):
                        crops.append(crop)
                        owners.append((int(tid), (x1, y1, x2, y2)))
            if crops:
                embs = self.extractor.extract_batch(crops)
                for (tid, box), e, crop in zip(owners, embs, crops):
                    gt = -1
                    if ann is not None:
                        best = max(((_iou(box, p["BboxP"]), p["id"]) for p in ann.get(str(f + 1), [])),
                                   default=(0, -1))
                        gt = int(best[1]) if best[0] >= 0.5 else -1
                    tracks[tid].append((f, box, e.astype(np.float32), cv2.resize(crop, (64, 128)), gt))
            if progress and (f // self.stride) % 20 == 0:
                progress(f, total)
            # skip to the next sampled frame (grab() avoids full decoding where possible)
            for _ in range(self.stride - 1):
                if not cap.grab():
                    break
            f += self.stride
        cap.release()

        entries = []
        for tid, samples in tracks.items():
            for s in range(0, len(samples), self.window):
                chunk = samples[s:s + self.window]
                if len(chunk) < 2:
                    continue
                embs = np.stack([c[2] for c in chunk])
                best = max(chunk, key=lambda c: c[1][3] - c[1][1])       # tallest crop → thumbnail
                p_known, label = 0.0, None
                if gallery is not None and len(chunk) >= 3:
                    feats, win = tracklet_features(embs, gallery)
                    p_known, label = self.classifier.p_known(feats), gallery.labels[win]
                gts = [c[4] for c in chunk if c[4] >= 0]
                entries.append({
                    "video": str(video), "camera": cam, "date": date, "track": tid,
                    "f0": chunk[0][0], "f1": chunk[-1][0], "fps": fps,
                    "t0": _stamp(start, chunk[0][0], fps), "t1": _stamp(start, chunk[-1][0], fps),
                    "thumb": cv2.imencode(".jpg", best[3])[1].tobytes(), "thumb_frame": best[0],
                    "box": list(best[1]), "embs": embs.astype(np.float16),
                    "centroid": l2n(l2n(embs).mean(axis=0)).astype(np.float32),
                    "p_known": float(p_known), "label": label,
                    "gt": Counter(gts).most_common(1)[0][0] if gts else -1,
                })
        return {"version": INDEX_VERSION, "video": str(video), "camera": cam, "date": date,
                "fps": fps, "stride": self.stride, "frames": total, "entries": entries}


def _stamp(start, frame: int, fps: float) -> str:
    if start is None:
        return f"{frame / fps:7.1f}s"
    from datetime import timedelta
    return (start + timedelta(seconds=frame / fps)).strftime("%H:%M:%S")


def index_file(cfg: dict, video: Path) -> Path:
    date = source_date(video.name) or "misc"
    return index_dir(cfg) / date / f"{video.parent.name}__{video.stem.split('_20')[0]}.pkl"


def build_index(cfg: dict, videos: List[Path], force: bool = False):
    todo = [v for v in videos if force or not index_file(cfg, v).exists()]
    logger.info("Indexing %d video(s) (%d already indexed)", len(todo), len(videos) - len(todo))
    if not todo:
        return
    idx = VideoIndexer(cfg)
    t0 = time.time()
    for k, v in enumerate(todo, 1):
        def progress(f, total, k=k):
            print(f"  [{k}/{len(todo)}] {v.parent.name}/{camera_key(v.name)}: indexed {f} / {total} frames",
                  flush=True)
        data = idx.index_video(v, progress)
        out = index_file(cfg, v)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "wb") as fh:
            pickle.dump(data, fh)
        logger.info("✓ %s/%s → %d tracklets (%.0fs elapsed)", v.parent.name, camera_key(v.name),
                    len(data["entries"]), time.time() - t0)


# ─────────────────────────────────────────────────────────────────────────────
#  Search
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SearchResult:
    entry: dict
    sim: float

    @property
    def video(self) -> str:
        return self.entry["video"]


def indexed_days(cfg: dict) -> Dict[str, int]:
    root = index_dir(cfg)
    if not root.exists():
        return {}
    return {d.name: len(list(d.glob("*.pkl"))) for d in sorted(root.iterdir()) if d.is_dir()}


def load_index(cfg: dict, days: Optional[List[str]] = None) -> List[dict]:
    entries = []
    root = index_dir(cfg)
    for day_dir in sorted(root.glob("*")) if root.exists() else []:
        if days and day_dir.name not in days:
            continue
        for f in sorted(day_dir.glob("*.pkl")):
            with open(f, "rb") as fh:
                data = pickle.load(fh)
            if data.get("version") == INDEX_VERSION:
                entries += data["entries"]
    return entries


def search(query_embs: np.ndarray, entries: List[dict], *, exclude_known: bool = True,
           p_thresh: float = 0.5, min_sim: float = 0.6, top_k: int = 30,
           skip: Optional[Callable[[dict], bool]] = None) -> List[SearchResult]:
    """Rank indexed tracklets by cosine similarity to the query centroid (one result per track)."""
    if not entries or query_embs is None or len(query_embs) == 0:
        return []
    q = l2n(l2n(np.asarray(query_embs, np.float32)).mean(axis=0))
    C = np.stack([e["centroid"] for e in entries])
    sims = C @ q
    best: Dict[tuple, SearchResult] = {}
    for e, s in zip(entries, sims):
        if s < min_sim or (exclude_known and e["p_known"] >= p_thresh) or (skip and skip(e)):
            continue
        key = (e["video"], e["track"])
        if key not in best or s > best[key].sim:
            best[key] = SearchResult(e, float(s))
    return sorted(best.values(), key=lambda r: -r.sim)[:top_k]


# ─────────────────────────────────────────────────────────────────────────────
#  Ground-truth evaluation (CHIRLA)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(cfg: dict, days: List[str], visitors: List[int]) -> dict:
    """Query = a visitor's first indexed tracklet; relevant = their other tracklets (other cam / > 60 s)."""
    out = {}
    for exclude_known in (False, True):
        stats = []
        for day in days:
            entries = load_index(cfg, [day])
            for pid in visitors:
                mine = sorted([e for e in entries if e["gt"] == pid and len(e["embs"]) >= 5],
                              key=lambda e: (e["video"], e["f0"]))
                if not mine:
                    continue
                q = mine[0]

                def skip(e, q=q):
                    return e["video"] == q["video"] and abs(e["f0"] - q["f0"]) < 60 * q["fps"]
                res = search(q["embs"].astype(np.float32), entries, exclude_known=exclude_known,
                             p_thresh=float(cfg["reid"].get("open_set_p_threshold", 0.5)),
                             min_sim=0.0, top_k=10 ** 6, skip=skip)
                rel = np.array([r.entry["gt"] == pid for r in res])
                n_rel = sum(1 for e in entries if e["gt"] == pid and not skip(e))
                if n_rel == 0:
                    continue
                hits = np.cumsum(rel)
                ap = float((hits / np.arange(1, len(rel) + 1) * rel).sum() / n_rel) if len(rel) else 0.0
                stats.append((bool(rel[:1].any()), bool(rel[:5].any()), float(rel[:10].mean()) if len(rel) else 0,
                              ap, n_rel))
        if stats:
            s = np.array(stats, dtype=float)
            out["exclude_employees" if exclude_known else "plain"] = {
                "queries": len(s), "rank1": s[:, 0].mean(), "top5": s[:, 1].mean(),
                "precision@10": s[:, 2].mean(), "mAP": s[:, 3].mean()}
    return out


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    pi = sub.add_parser("index")
    pi.add_argument("--day", action="append", help="CHIRLA recording day (repeatable)")
    pi.add_argument("--all", action="store_true")
    pi.add_argument("--videos", nargs="*")
    pi.add_argument("--force", action="store_true")
    pe = sub.add_parser("eval")
    pe.add_argument("--day", action="append")
    for p in (pi, pe):
        p.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    from detection_engine import DetectionEngine
    cfg = DetectionEngine.load_config(args.config)
    root = Path(cfg["chirla"]["root"])

    if args.cmd == "index":
        if args.videos:
            videos = [Path(v) for v in args.videos]
        else:
            dates = sequence_dates(root)
            want = set(dates.values()) if args.all else set(args.day or [cfg["chirla"].get("demo_day")])
            videos = [v for seq, d in sorted(dates.items()) if d in want
                      for v in sorted((root / "videos" / seq).glob("camera_*.avi"))]
        build_index(cfg, videos, force=args.force)
    else:
        days = args.day or sorted(indexed_days(cfg))
        res = evaluate(cfg, days, [int(v) for v in cfg["chirla"].get("visitors", [])])
        Path("output").mkdir(exist_ok=True)
        Path("output/search_eval.json").write_text(json.dumps({"days": days, **res}, indent=2))
        for mode, m in res.items():
            print(f"{mode:18} queries {m['queries']}: rank-1 {m['rank1']:.1%}  top-5 {m['top5']:.1%}  "
                  f"precision@10 {m['precision@10']:.1%}  mAP {m['mAP']:.1%}")


if __name__ == "__main__":
    main()
