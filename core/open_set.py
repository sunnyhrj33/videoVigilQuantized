"""
core/open_set.py
────────────────────────────────────────────────────────────────────────────────
Open-set person identification against a persistent, enrolled employee gallery.

  • PrototypeGallery   – loads enrolled_gallery/gallery_prototypes.pkl
  • tracklet_features  – the 5 temporal features of a tracklet
  • OpenSetClassifier  – logistic regression P(Known) (rule-based fallback)
  • TrackletManager    – per-track buffer of sharp crop embeddings + decisions
────────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import json
import logging
import pickle
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Deque, Dict, List, Optional

import cv2
import numpy as np

from core.chirla import is_iso_date  # noqa: F401  (re-exported for callers of open_set)
from core.geometry import l2n, sharpness

logger = logging.getLogger(__name__)

FEATURE_NAMES = [
    "s_best",          # F1 mean cosine similarity to the winning identity
    "margin",          # F2 top-1 minus top-2 mean similarity
    "agreement",       # F3 fraction of crops voting for the winner
    "consistency",     # F4 mean pairwise similarity inside the tracklet
    "prototype_fit",   # F5 cosine(tracklet centroid, winner prototype)
]

STATUS_PENDING = "PENDING"
STATUS_KNOWN = "KNOWN"
STATUS_UNKNOWN = "UNKNOWN"

EMB_DIM = 512                  # OSNet / ResNet Re-ID embedding size (empty-gallery placeholder)
NO_STAFF = "none"              # gallery_date value: analyse without an enrolled staff gallery


# ─────────────────────────────────────────────────────────────────────────────
#  Enrolled gallery
# ─────────────────────────────────────────────────────────────────────────────

class PrototypeGallery:
    """Enrolled employee prototypes: one L2-normalised mean embedding per person."""

    def __init__(self, labels: List[str], prototypes: np.ndarray, meta: Optional[dict] = None):
        self.labels = list(labels)
        self.prototypes = l2n(np.asarray(prototypes, np.float32).reshape(len(self.labels), -1)
                              if len(self.labels) else np.zeros((0, EMB_DIM), np.float32))
        self.meta = meta or {}

    def __len__(self) -> int:
        return len(self.labels)

    @classmethod
    def empty(cls, meta: Optional[dict] = None) -> "PrototypeGallery":
        """No enrolled identities: every tracklet is UNKNOWN (unless merged with named people)."""
        return cls([], np.zeros((0, EMB_DIM), np.float32), meta)

    @classmethod
    def load(cls, path: str | Path) -> "PrototypeGallery":
        with open(path, "rb") as f:
            data = pickle.load(f)
        labels = list(data["prototypes"].keys())
        protos = np.stack([data["prototypes"][k] for k in labels]) if labels else None
        logger.info("✓ Loaded enrolled gallery: %d identities from %s", len(labels), path)
        return cls(labels, protos, data.get("meta", {}))

    def similarities(self, embs: np.ndarray) -> np.ndarray:
        """(n, d) embeddings → (n, k) cosine similarities."""
        return l2n(embs) @ self.prototypes.T

    def merged(self, other: "PrototypeGallery") -> "PrototypeGallery":
        """This gallery plus other's identities (other wins on a duplicate label)."""
        if not len(other):
            return self
        if not len(self):
            return PrototypeGallery(other.labels, other.prototypes, self.meta)
        keep = [i for i, label in enumerate(self.labels) if label not in set(other.labels)]
        return PrototypeGallery([self.labels[i] for i in keep] + other.labels,
                                np.vstack([self.prototypes[keep], other.prototypes]), self.meta)


def load_day_gallery(gallery_file: str | Path, date: str,
                     model: Optional[str]) -> tuple[Optional[PrototypeGallery], Optional[str]]:
    """
    The staff gallery enrolled on `date` (enrolled_gallery/<date>/gallery_prototypes.pkl).
    Returns (gallery, None), or (None, reason) when that day can't be used: not a
    valid date, nobody enrolled, unreadable, or built with a different Re-ID model.
    """
    if not is_iso_date(date):
        return None, f"“{date}” is not a valid date"
    path = Path(gallery_file).parent / date / Path(gallery_file).name
    if not path.exists():
        return None, f"No staff are enrolled for {date}"
    try:
        gallery = PrototypeGallery.load(path)
    except Exception as exc:
        logger.warning("Could not read staff gallery %s (%s)", path, exc)
        return None, f"The staff gallery for {date} could not be read"
    built_with = gallery.meta.get("model")
    if model and built_with and built_with != model:
        logger.warning("Gallery %s was built with %s but Re-ID model is %s", path, built_with, model)
        return None, (f"The staff gallery for {date} was built with a different Re-ID model "
                      f"({built_with}) and needs rebuilding")
    if not len(gallery):
        return None, f"The staff gallery for {date} has no enrolled employees"
    gallery.meta.setdefault("date", date)
    return gallery, None


def list_day_galleries(gallery_file: str | Path, model: Optional[str]) -> List[dict]:
    """Every enrolled day: [{'date', 'staff', 'usable', 'problem'}], oldest first."""
    root = Path(gallery_file).parent
    out = []
    if not root.is_dir():
        return out
    for d in sorted(p for p in root.iterdir() if p.is_dir() and is_iso_date(p.name)):
        if not (d / Path(gallery_file).name).exists():
            continue
        gallery, problem = load_day_gallery(gallery_file, d.name, model)
        out.append({"date": d.name, "staff": len(gallery) if gallery else 0,
                    "usable": gallery is not None, "problem": problem})
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  Tracklet features
# ─────────────────────────────────────────────────────────────────────────────

def tracklet_features(embs: np.ndarray, gallery: PrototypeGallery) -> tuple[np.ndarray, int]:
    """
    Compute the 5 temporal features for a tracklet of crop embeddings.
    Returns (features[5], winner_index_into_gallery).
    """
    e = l2n(embs)
    sims = e @ gallery.prototypes.T                       # (n, k)
    mean_sims = sims.mean(axis=0)                         # (k,)
    order = np.argsort(-mean_sims)
    win = int(order[0])

    s_best = float(mean_sims[win])
    margin = float(mean_sims[win] - mean_sims[order[1]]) if len(order) > 1 else s_best
    agreement = float(np.mean(np.argmax(sims, axis=1) == win))

    n = len(e)
    if n > 1:
        pair = e @ e.T
        consistency = float((pair.sum() - n) / (n * (n - 1)))
    else:
        consistency = 1.0

    centroid = l2n(e.mean(axis=0))
    prototype_fit = float(centroid @ gallery.prototypes[win])

    return np.array([s_best, margin, agreement, consistency, prototype_fit], dtype=np.float32), win


# ─────────────────────────────────────────────────────────────────────────────
#  Open-set classifier
# ─────────────────────────────────────────────────────────────────────────────

class OpenSetClassifier:
    """
    P(Known | tracklet features) = sigmoid(w · standardised(x) + b).
    Parameters come from train_open_set.py; without them a hand-set rule-based
    equivalent is used so the pipeline still runs.
    """

    _RULE_W = np.array([10.0, 25.0, 2.0, 1.0, 8.0], dtype=np.float32)
    _RULE_B = -12.5

    def __init__(self, path: Optional[str | Path] = None):
        self.trained = False
        self.mean = np.zeros(5, np.float32)
        self.scale = np.ones(5, np.float32)
        self.coef = self._RULE_W.copy()
        self.intercept = self._RULE_B
        if path and Path(path).exists():
            with open(path, "r") as f:
                p = json.load(f)
            self.mean = np.array(p["mean"], np.float32)
            self.scale = np.array(p["scale"], np.float32)
            self.coef = np.array(p["coef"], np.float32)
            self.intercept = float(p["intercept"])
            self.trained = True
            logger.info("✓ Loaded open-set classifier from %s", path)
        else:
            logger.warning("Open-set classifier file not found (%s) – using rule-based fallback", path)

    def p_known(self, feats: np.ndarray) -> float:
        z = (np.asarray(feats, np.float32) - self.mean) / self.scale
        logit = float(z @ self.coef + self.intercept)
        return float(1.0 / (1.0 + np.exp(-logit)))


# ─────────────────────────────────────────────────────────────────────────────
#  Per-track tracklet buffer
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Identity:
    status: str = STATUS_PENDING
    label: Optional[str] = None          # e.g. "emp_04" (also kept for UNKNOWN = best guess)
    confidence: float = 0.0              # P(Known) for tracklet mode, cosine sim for naive
    n_crops: int = 0
    features: Optional[np.ndarray] = None


@dataclass
class _Track:
    embs: Deque[np.ndarray]
    thumbs: Deque[np.ndarray] = field(default_factory=lambda: deque(maxlen=10))   # 64×128 crops
    last_seen: int = 0
    last_sample: int = -10**9
    identity: Identity = field(default_factory=Identity)


def crop_is_usable(crop: np.ndarray, det_conf: float, min_height: int,
                   min_conf: float, min_sharpness: float) -> bool:
    """Reject tiny, low-confidence or blurry person crops."""
    if crop is None or crop.size == 0:
        return False
    h, w = crop.shape[:2]
    if h < min_height or w < min_height // 4 or det_conf < min_conf:
        return False
    return sharpness(crop) >= min_sharpness


def display_name(label: Optional[str]) -> str:
    """'emp_04' → 'Emp 04'."""
    if not label:
        return "?"
    return label.replace("emp_", "Emp ")


class TrackletManager:
    """
    Keeps a circular buffer of the last `window` sharp crop embeddings per
    person track and turns them into a KNOWN / UNKNOWN decision.

    mode = "tracklet" : 5 tracklet features → OpenSetClassifier
    mode = "naive"    : single newest crop, closed-set nearest prototype
                        (accepted if cosine ≥ naive_thresh) – the baseline
    """

    def __init__(self, gallery: PrototypeGallery, classifier: OpenSetClassifier,
                 embed_fn: Callable[[List[np.ndarray]], np.ndarray], cfg: dict):
        track_cfg = cfg.get("tracking", {})
        reid_cfg = cfg.get("reid", {})
        q = reid_cfg.get("crop_quality", {})

        self.gallery = gallery
        self.classifier = classifier
        self.embed_fn = embed_fn
        self.window = int(track_cfg.get("tracklet_window", 10))
        self.min_crops = int(track_cfg.get("min_tracklet_crops", 4))
        self.sample_interval = max(1, int(reid_cfg.get("frame_interval", 3)))
        self.settled_interval = max(1, int(reid_cfg.get("settled_interval", 30)))
        self.settled_margin = float(reid_cfg.get("settled_margin", 0.2))
        self.max_embeds = max(1, int(reid_cfg.get("max_embeds_per_frame", 2)))
        self.max_age = int(track_cfg.get("track_buffer", 30)) * 2
        self.p_thresh = float(reid_cfg.get("open_set_p_threshold", 0.5))
        self.smoothing = float(reid_cfg.get("decision_smoothing", 0.5))
        self.hysteresis = float(reid_cfg.get("decision_hysteresis", 0.1))
        self.naive_thresh = float(reid_cfg.get("reid_thresh", 0.45))
        self.min_height = int(q.get("min_height", 64))
        self.min_conf = float(q.get("min_conf", 0.45))
        self.min_sharpness = float(q.get("min_sharpness", 20.0))
        self.mode = reid_cfg.get("mode", "tracklet")
        self._tracks: Dict[int, _Track] = {}
        self.watch: set = set()      # track ids always sampled at full rate (e.g. suspects)

    def reset(self):
        self._tracks.clear()

    def _is_settled(self, t: "_Track") -> bool:
        """A tracklet decision that is full enough and clearly away from the threshold."""
        ident = t.identity
        return (self.mode == "tracklet" and ident.status != STATUS_PENDING
                and len(t.embs) >= self.window // 2
                and abs(ident.confidence - self.p_thresh) >= self.settled_margin)

    def track_data(self, tid: int):
        """(embeddings (n, d), thumbnails [64×128 BGR]) of a track's tracklet buffer."""
        t = self._tracks.get(tid)
        if t is None or not t.embs:
            return np.zeros((0, 512), np.float32), []
        return np.stack(t.embs), list(t.thumbs)

    def set_gallery(self, gallery: PrototypeGallery):
        """Swap the enrolled gallery (e.g. per recording day); clears all tracklets."""
        self.gallery = gallery
        self.reset()

    def set_mode(self, mode: str):
        if mode != self.mode:
            self.mode = mode
            for t in self._tracks.values():
                t.identity = Identity(n_crops=len(t.embs))

    def update(self, frame: np.ndarray, frame_idx: int,
               persons: List[tuple]) -> Dict[int, Identity]:
        """
        persons: list of (track_id, (x1, y1, x2, y2), det_conf).
        Returns {track_id: Identity} for every person passed in.
        """
        due = []
        for tid, box, conf in persons:
            t = self._tracks.get(tid)
            if t is None:
                t = self._tracks[tid] = _Track(embs=deque(maxlen=self.window))
            t.last_seen = frame_idx
            settled = self._is_settled(t) and tid not in self.watch
            interval = self.settled_interval if settled else self.sample_interval
            overdue = frame_idx - t.last_sample - interval
            if overdue >= 0:
                due.append(((settled, len(t.embs), -overdue), tid, box, conf))
        due.sort(key=lambda d: d[0])

        to_embed, owners = [], []
        for _, tid, (x1, y1, x2, y2), conf in due:
            if len(to_embed) >= self.max_embeds:
                break
            crop = frame[max(0, y1):y2, max(0, x1):x2]
            if crop_is_usable(crop, conf, self.min_height, self.min_conf, self.min_sharpness):
                to_embed.append(crop)
                owners.append(tid)
                self._tracks[tid].last_sample = frame_idx

        if to_embed:
            embs = self.embed_fn(to_embed)
            for tid, e, crop in zip(owners, embs, to_embed):
                t = self._tracks[tid]
                t.embs.append(np.asarray(e, np.float32))
                t.thumbs.append(cv2.resize(crop, (64, 128)))
                self._decide(t)

        for tid in [k for k, t in self._tracks.items() if frame_idx - t.last_seen > self.max_age]:
            del self._tracks[tid]

        return {tid: self._tracks[tid].identity for tid, _, _ in persons}

    def _decide(self, t: _Track):
        n = len(t.embs)
        ident = t.identity
        ident.n_crops = n

        if not len(self.gallery):
            # Nobody enrolled for this source: once seen clearly enough, everyone is UNKNOWN.
            if n >= (1 if self.mode == "naive" else self.min_crops):
                ident.status, ident.label, ident.confidence, ident.features = STATUS_UNKNOWN, None, 0.0, None
            return

        if self.mode == "naive":
            sims = self.gallery.similarities(np.stack([t.embs[-1]]))[0]
            win = int(np.argmax(sims))
            ident.label = self.gallery.labels[win]
            ident.confidence = float(sims[win])
            ident.status = STATUS_KNOWN if sims[win] >= self.naive_thresh else STATUS_UNKNOWN
            return

        if n < self.min_crops:
            ident.status = STATUS_PENDING
            return
        feats, win = tracklet_features(np.stack(t.embs), self.gallery)
        p_now = self.classifier.p_known(feats)
        p = p_now if ident.status == STATUS_PENDING else \
            self.smoothing * ident.confidence + (1 - self.smoothing) * p_now
        if ident.status == STATUS_KNOWN:
            known = p >= self.p_thresh - self.hysteresis
        elif ident.status == STATUS_UNKNOWN:
            known = p >= self.p_thresh + self.hysteresis
        else:
            known = p >= self.p_thresh
        ident.features = feats
        ident.label = self.gallery.labels[win]
        ident.confidence = p
        ident.status = STATUS_KNOWN if known else STATUS_UNKNOWN
