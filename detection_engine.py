"""
detection_engine.py
────────────────────────────────────────────────────────────────────────────────
Core engine for YOLO object detection + ByteTrack / BoTSORT tracking + Re-ID.
Supports:
  • PyTorch  (.pt)  – fp32 / fp16 via Ultralytics
  • OpenVINO (.xml) – fp32 or INT8 quantised models via Ultralytics / OV runtime
  • Re-ID feature extraction with EMA gallery (torchreid / built-in OSNet)
────────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import logging
import sys
import time
import threading
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml

from chirla import camera_key, day_gallery_path, parse_video_name, source_date
from official_trackers import OCSORTOfficial, StrongSORTOfficial, TrackFormerOfficial
from open_set import (STATUS_KNOWN, STATUS_PENDING, STATUS_UNKNOWN, Identity,
                      OpenSetClassifier, PrototypeGallery, TrackletManager, display_name)
from security import AssetMonitor, AuditLogger, Clock, ZoneMonitor, zone_name_pretty

# BGR colours for the security overlay
COL_KNOWN = (60, 200, 60)
COL_UNKNOWN = (0, 140, 255)
COL_ALERT = (40, 40, 230)
COL_PENDING = (160, 160, 160)
COL_ASSET = (230, 220, 0)
COL_ZONE = (0, 215, 255)
PERSON_CLASS = 0

# Auto-discover FastReID if available
_fast_reid_paths = [
    "/home/sunny/FYP/videoVigilQuantized/fast-reid",
    "../fast-reid",
    "./fast-reid",
]
for _path in _fast_reid_paths:
    _p = Path(_path).resolve()
    if _p.exists() and _p not in sys.path:
        sys.path.insert(0, str(_p))

logger = logging.getLogger(__name__)


def get_available_trackers() -> List[str]:
    """
    Get list of ALL available trackers.
    Includes both Ultralytics native (ByteTrack, BoTSort) and official implementations (OC-SORT, StrongSORT, TrackFormer).
    """
    # All trackers: 2 native + 3 official implementations
    ALL_TRACKERS = ["bytetrack", "botsort", "ocsort", "strongsort", "trackformer"]
    
    try:
        # Try importing TRACKER_DIR from ultralytics
        try:
            from ultralytics.cfg import TRACKER_DIR
            tracker_dir = Path(TRACKER_DIR)
        except (ImportError, AttributeError):
            # Fall back to finding the trackers directory manually
            import ultralytics
            uv_path = Path(ultralytics.__file__).parent
            tracker_dir = uv_path / "cfg" / "trackers"
        
        if tracker_dir.exists():
            yaml_files = sorted(list(tracker_dir.glob("*.yaml")))
            available = [f.stem for f in yaml_files if f.stem in ALL_TRACKERS]
            # Add custom trackers
            custom = ["ocsort", "strongsort", "trackformer"]
            for ct in custom:
                if ct not in available:
                    available.append(ct)
            logger.info("✓ Found %d trackers (2 native + 3 custom): %s", len(available), sorted(available))
            return sorted(available) if available else ["bytetrack"]
    except Exception as e:
        logger.warning("Error detecting trackers: %s", str(e))
    
    logger.info("Using all trackers: %s", ALL_TRACKERS)
    return ALL_TRACKERS


def get_available_reid_models() -> Dict[str, bool]:
    """
    Check which Re-ID models are available and can be loaded.
    Returns a dict: {model_name: bool (can_load)}
    """
    reid_models = [
        # OSNet variants
        "osnet_ain_x1_0", "osnet_x1_0", "osnet_x0_25",
        # ResNet variants
        "resnet50", "resnet101",
        # FastReID
        "fastraid", "osnet_ibn",
        # TransReID
        "transreid",
        # VitREID
        "vitreid"
    ]
    
    available = {}
    torchreid_available = False
    timm_available = False
    
    try:
        import torchreid
        torchreid_available = True
    except ImportError:
        pass
    
    try:
        import timm
        timm_available = True
    except ImportError:
        pass
    
    for model in reid_models:
        # Check if the model can be loaded
        framework = ReIDExtractor._detect_framework(model)
        if framework == "torchreid":
            available[model] = torchreid_available
        elif framework == "vitreid":
            available[model] = timm_available
        else:
            # fastraid and others: always try to load (with fallback)
            available[model] = True
    
    return available


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

PALETTE: Dict[int, Tuple[int, int, int]] = {}


def _colour(track_id: int) -> Tuple[int, int, int]:
    """Return a stable BGR colour for a given track-id."""
    if track_id not in PALETTE:
        rng = np.random.default_rng(track_id + 42)
        PALETTE[track_id] = tuple(int(c) for c in rng.integers(80, 230, 3))
    return PALETTE[track_id]


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine distance in [0, 2] between two L2-normalised vectors."""
    a = a / (np.linalg.norm(a) + 1e-6)
    b = b / (np.linalg.norm(b) + 1e-6)
    return float(1.0 - np.dot(a, b))


# ─────────────────────────────────────────────────────────────────────────────
#  Re-ID Gallery
# ─────────────────────────────────────────────────────────────────────────────

class ReIDGallery:
    """
    Minimal EMA-based Re-ID gallery.
    Maps tracker IDs → persistent global IDs using cosine feature distance.
    """

    def __init__(self, thresh: float = 0.4, ema_rate: float = 0.9,
                 max_dist: float = 0.6):
        self.thresh = thresh
        self.ema_rate = ema_rate
        self.max_dist = max_dist

        self._gallery: Dict[int, np.ndarray] = {}   # reid_id → feature
        self._tracker_to_reid: Dict[int, int] = {}   # tracker_id → reid_id
        self._next_id = 1

    def update(self, tracker_id: int, feature: np.ndarray) -> int:
        """Return stable reid_id for this tracker_id, updating the gallery."""
        feat = feature / (np.linalg.norm(feature) + 1e-6)

        # Already assigned
        if tracker_id in self._tracker_to_reid:
            reid_id = self._tracker_to_reid[tracker_id]
            old = self._gallery[reid_id]
            # EMA update
            self._gallery[reid_id] = (
                self.ema_rate * old + (1 - self.ema_rate) * feat
            )
            return reid_id

        # Match against gallery
        best_id, best_dist = None, float("inf")
        for gid, gfeat in self._gallery.items():
            d = _cosine_distance(feat, gfeat)
            if d < best_dist:
                best_dist = d
                best_id = gid

        if best_id is not None and best_dist < self.max_dist:
            reid_id = best_id
            logger.debug("ReID match: tracker=%d matched existing R%d (dist=%.4f < %.4f)",
                         tracker_id, reid_id, best_dist, self.max_dist)
        else:
            reid_id = self._next_id
            self._next_id += 1
            self._gallery[reid_id] = feat
            logger.debug("ReID new: tracker=%d assigned new R%d (best_dist=%.4f)",
                         tracker_id, reid_id, best_dist)

        self._tracker_to_reid[tracker_id] = reid_id
        return reid_id

    def get_tracker_reid(self, tracker_id: int) -> Optional[int]:
        """Return the current Re-ID for a tracker, if one already exists."""
        return self._tracker_to_reid.get(tracker_id)

    def has_tracker(self, tracker_id: int) -> bool:
        return tracker_id in self._tracker_to_reid


# ─────────────────────────────────────────────────────────────────────────────
#  Re-ID Feature Extractor
# ─────────────────────────────────────────────────────────────────────────────

class ReIDExtractor:
    """
    Lightweight feature extractor supporting:
      • OSNet (osnet_ain_x1_0, osnet_x1_0, osnet_x0_25) via torchreid
      • ResNet (resnet50, resnet101) via torchreid
      • FastReID models via fast-reid framework (requires fast-reid installed)
      • TransReID models via torchreid
      • VitREID (Vision Transformer) via timm
    """

    _MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    _STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(self, model_name: str = "osnet_x0_25", device: str = "cpu",
                 weights: Optional[str] = None, backend: str = "auto"):
        """backend: 'auto' (OpenVINO IR if exported, else PyTorch) | 'openvino' | 'torch'."""
        self._extractor = None
        self._ov_model = None
        self._device = device
        self._model_name = model_name
        self._framework = self._detect_framework(model_name)
        self._fallback_used = False

        try:
            if self._framework == "torchreid":
                import torchreid
                model_path = self._resolve_weights(model_name, weights)
                ov_xml = (Path(model_path).parent / f"{model_name}_openvino" / f"{model_name}.xml"
                          if model_path else None)
                if backend in ("auto", "openvino") and ov_xml is not None and ov_xml.exists():
                    import openvino as ov
                    self._ov_model = ov.Core().compile_model(str(ov_xml), "CPU",
                                                             {"PERFORMANCE_HINT": "LATENCY"})
                    logger.info("✓ Re-ID extractor loaded (OpenVINO): %s [%s]", model_name, ov_xml.name)
                    return
                if backend == "openvino":
                    raise FileNotFoundError(f"OpenVINO Re-ID model not found: {ov_xml}")
                self._extractor = torchreid.utils.FeatureExtractor(
                    model_name=model_name,
                    model_path=model_path,
                    device=device,
                )
                logger.info("✓ Re-ID extractor loaded (torchreid): %s on %s [%s]", model_name, device,
                            f"Re-ID weights {Path(model_path).name}" if model_path else "ImageNet weights only")
            elif self._framework == "fastraid":
                try:
                    from fast_reid.fastreid.modeling import build_model
                    from fast_reid.fastreid.config import get_cfg
                    logger.info("✓ FastReID framework detected")
                    logger.warning("FastReID requires model config; using fallback feature extraction")
                    self._fallback_used = True
                except ImportError:
                    logger.warning("fast-reid not installed – using fallback features")
                    self._fallback_used = True
            elif self._framework == "vitreid":
                try:
                    import timm
                    logger.info("✓ Vision Transformer Re-ID framework detected (timm)")
                    # Note: VitReID support is partial; using fallback
                    self._fallback_used = True
                except ImportError:
                    logger.warning("timm not installed for VitREID – using fallback features")
                    self._fallback_used = True
        except (ImportError, Exception) as e:
            logger.warning(
                "Could not load %s Re-ID model (%s) – using fallback colour histogram. "
                "Install with: pip install torchreid  OR  pip install fast-reid  OR  pip install timm",
                self._framework, str(e)
            )
            self._fallback_used = True

    @staticmethod
    def _resolve_weights(model_name: str, weights: Optional[str]) -> str:
        """
        Return a Re-ID-trained checkpoint for this model, or "" for torchreid's
        ImageNet initialisation. `weights` may be a file or a folder of
        checkpoints named '<model_name>_*.pth' (e.g. models/reid/).
        """
        if not weights or weights == "auto":
            weights = "models/reid"
        p = Path(weights)
        if p.is_dir():
            matches = sorted(p.glob(f"{model_name}_*.pth"))
            p = matches[0] if matches else None
        if p is not None and p.is_file() and p.name.startswith(model_name):
            return str(p)
        logger.warning("No Re-ID-trained weights for %s (looked in %s) – using ImageNet weights; "
                       "identity matching will be weak.", model_name, weights)
        return ""

    @staticmethod
    def _detect_framework(model_name: str) -> str:
        """Detect which framework a model name belongs to."""
        if model_name.startswith("osnet") or model_name.startswith("resnet"):
            return "torchreid"
        elif model_name.startswith("fastraid") or model_name.startswith("osnet_ibn"):
            return "fastraid"
        elif model_name.startswith("transreid"):
            return "torchreid"  # TransReID is in torchreid
        elif model_name.startswith("vitreid") or model_name.startswith("vit"):
            return "vitreid"
        else:
            return "torchreid"  # default

    def extract(self, bgr_crop: np.ndarray) -> np.ndarray:
        """Return a 1-D feature vector for the given BGR crop."""
        if bgr_crop is None or bgr_crop.size == 0:
            return np.zeros(512, dtype=np.float32)

        if self._ov_model is not None:
            return self.extract_batch([bgr_crop])[0]

        if self._extractor is not None:
            rgb = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
            import torch
            with torch.no_grad():
                feat = self._extractor([rgb])
            return feat[0].cpu().numpy().flatten()

        return self._histogram_feature(bgr_crop)

    def extract_batch(self, bgr_crops: List[np.ndarray]) -> np.ndarray:
        """Return an (n, d) feature matrix for a list of BGR crops (one forward pass)."""
        if not bgr_crops:
            return np.zeros((0, 512), dtype=np.float32)
        if self._ov_model is not None:
            from PIL import Image
            # Same preprocessing as torchreid.FeatureExtractor: PIL bilinear 256×128, ImageNet norm
            batch = np.stack([
                np.asarray(Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)).resize((128, 256), Image.BILINEAR),
                           dtype=np.float32)
                for c in bgr_crops
            ])
            batch = ((batch / 255.0 - self._MEAN) / self._STD).transpose(0, 3, 1, 2)
            out = self._ov_model(np.ascontiguousarray(batch, dtype=np.float32))
            return np.asarray(out[self._ov_model.output(0)]).reshape(len(bgr_crops), -1)
        if self._extractor is not None:
            rgbs = [cv2.cvtColor(c, cv2.COLOR_BGR2RGB) for c in bgr_crops]
            import torch
            with torch.no_grad():
                feats = self._extractor(rgbs)
            return feats.cpu().numpy().reshape(len(bgr_crops), -1)
        return np.stack([self._histogram_feature(c) for c in bgr_crops])

    @staticmethod
    def _histogram_feature(bgr_crop: np.ndarray) -> np.ndarray:
        # Fallback: 64-dim colour histogram feature
        resized = cv2.resize(bgr_crop, (64, 128))
        hists = []
        for ch in range(3):
            h, _ = np.histogram(resized[:, :, ch], bins=32, range=(0, 256))
            hists.append(h.astype(np.float32))
        feat = np.concatenate(hists)
        return feat / (feat.sum() + 1e-6)


# ─────────────────────────────────────────────────────────────────────────────
#  Detection Engine
# ─────────────────────────────────────────────────────────────────────────────

class DetectionEngine:
    """
    Wraps Ultralytics YOLO for detection + tracking.
    Handles PyTorch (.pt) and OpenVINO (.xml) models transparently.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self._reid_cfg = cfg.get("reid", {}).copy()
        det = cfg.get("detection", {})
        vis = cfg.get("visualisation", {})
        reid_cfg = self._reid_cfg

        self.conf = det.get("conf_threshold", 0.35)
        self.iou = det.get("iou_threshold", 0.45)
        self.classes = det.get("target_classes", [0]) or None
        self.imgsz = det.get("imgsz", 640)
        self.max_det = det.get("max_detections", 300)
        self.person_conf = float(det.get("person_conf_threshold", 0.5))
        self.zones_enabled = bool(cfg.get("security_zones", {}).get("enabled", True))

        self.show_labels = vis.get("show_labels", True)
        self.show_conf = vis.get("show_confidence", True)
        self.show_tid = vis.get("show_track_id", True)
        self.show_rid = vis.get("show_reid_id", True)
        self.show_trail = vis.get("show_trail", True)
        self.trail_len = vis.get("trail_length", 40)
        self.box_thick = vis.get("box_thickness", 2)
        self.font_scale = vis.get("font_scale", 0.55)
        self.show_fps = vis.get("show_fps", True)

        # Guard model/tracker state during concurrent UI + worker operations.
        self._engine_lock = threading.RLock()

        # Tracking
        track_cfg = cfg.get("tracking", {})
        self._tracker_name = track_cfg.get("tracker", "bytetrack")
        self._custom_tracker = None  # For custom tracker implementations
        self._initialize_tracker()

        # Security layer: enrolled-gallery identity, zones, assets, audit log
        self._identity_mode = reid_cfg.get("mode", "tracklet")
        self._tracklets: Optional[TrackletManager] = None
        self.gallery_info = "—"
        self._zones = ZoneMonitor(cfg)
        self._zones.configure(None)
        self._assets = AssetMonitor(cfg)
        self._clock = Clock()
        self._camera_id = "Cam_00"
        self._audit: Optional[AuditLogger] = None
        self._audit_path = cfg.get("output", {}).get("audit_log", "output/audit_log.csv")
        self._logged_identity: Dict[int, Tuple[str, Optional[str]]] = {}
        self._zone_alert_t: Dict[Tuple[int, str], float] = {}
        self._banners: deque = deque(maxlen=3)        # (text, expires_at_sec)
        self._alert_count = 0
        # UI event feed: (seq_no, level, stamp, camera, text); read via events_since()
        self._events: deque = deque(maxlen=500)
        self._event_no = 0

        # Re-ID
        self._reid_enabled = reid_cfg.get("enabled", True)
        self._reid_model_name: Optional[str] = None
        self._reid_extractor_cache: Dict[str, ReIDExtractor] = {}
        self._reid_extractor: Optional[ReIDExtractor] = None
        self._gallery: Optional[ReIDGallery] = None
        if self._reid_enabled:
            self.set_reid_model(reid_cfg.get("model", "osnet_x0_25"))
        self._reid_frame_interval = max(1, int(reid_cfg.get("frame_interval", 3)))

        # Trail history  track_id → deque of (cx, cy)
        self._trails: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=self.trail_len)
        )

        self._model = None
        self._model_path: Optional[str] = None
        self._model_imgsz: Optional[int] = None

        # FPS
        self._fps_buffer: deque = deque(maxlen=30)
        self._t_prev = time.perf_counter()
        self._frame_idx = 0

    def set_reid_model(self, model_name: str):
        """Switch Re-ID model with error handling."""
        with self._engine_lock:
            if self._reid_enabled and model_name == self._reid_model_name and self._reid_extractor is not None:
                return

            self._reid_cfg["model"] = model_name
            self._reid_model_name = model_name
            if not self._reid_enabled:
                return

            try:
                if model_name not in self._reid_extractor_cache:
                    logger.info("Initializing Re-ID model: %s", model_name)
                    self._reid_extractor_cache[model_name] = ReIDExtractor(
                        model_name=model_name,
                        device=self._reid_cfg.get("device", "cpu"),
                        weights=self._reid_cfg.get("weights", "auto"),
                    )
                self._reid_extractor = self._reid_extractor_cache[model_name]
                logger.info("✓ Re-ID model set to: %s", model_name)
                self._gallery = ReIDGallery(
                    thresh=self._reid_cfg.get("reid_thresh", 0.4),
                    ema_rate=self._reid_cfg.get("feature_update_rate", 0.9),
                    max_dist=self._reid_cfg.get("max_reid_distance", 0.6),
                )
                self._reid_frame_interval = max(1, int(self._reid_cfg.get("frame_interval", 3)))
                self._init_open_set()
            except Exception as e:
                logger.error("Failed to load Re-ID model '%s': %s. Using fallback.", model_name, str(e))
                # Use a fallback extractor
                if "fallback" not in self._reid_extractor_cache:
                    self._reid_extractor_cache["fallback"] = ReIDExtractor(
                        model_name="osnet_x0_25",
                        device=self._reid_cfg.get("device", "cpu"),
                    )
                self._reid_extractor = self._reid_extractor_cache["fallback"]

    def _init_open_set(self):
        """Load the enrolled gallery + open-set classifier for the current Re-ID model."""
        self._tracklets = None
        gallery_file = Path(self._reid_cfg.get("gallery_file", "enrolled_gallery/gallery_prototypes.pkl"))
        if not gallery_file.exists():
            logger.warning("Enrolled gallery %s not found – run build_gallery.py. "
                           "Falling back to legacy runtime Re-ID.", gallery_file)
            return
        gallery = PrototypeGallery.load(gallery_file)
        self.gallery_info = f"{gallery.meta.get('date', 'default')} · {len(gallery)} enrolled"
        built_with = gallery.meta.get("model")
        if built_with and built_with != self._reid_model_name:
            logger.warning("Gallery was built with %s but Re-ID model is %s – embeddings are not "
                           "comparable. Rebuild the gallery; using legacy runtime Re-ID.",
                           built_with, self._reid_model_name)
            return
        classifier = OpenSetClassifier(self._reid_cfg.get("open_set_model"))
        cfg = dict(self.cfg)
        cfg["reid"] = {**self._reid_cfg, "mode": self._identity_mode}
        self._tracklets = TrackletManager(gallery, classifier, self._reid_extractor.extract_batch, cfg)

    def set_identity_mode(self, mode: str):
        """'tracklet' (open-set), 'naive' (single-frame baseline) or 'legacy' (runtime EMA IDs)."""
        with self._engine_lock:
            self._identity_mode = mode
            if self._tracklets is not None and mode != "legacy":
                self._tracklets.set_mode(mode)
            self._logged_identity.clear()
            logger.info("✓ Identity mode: %s", mode)

    def skip_frame(self):
        """A source frame was dropped (real-time playback): keep clock / timers in step."""
        with self._engine_lock:
            self._frame_idx += 1

    def set_open_set_threshold(self, p: float):
        """P(Known) cut-off between Employee and UNKNOWN (live)."""
        self._reid_cfg["open_set_p_threshold"] = float(p)
        if self._tracklets is not None:
            self._tracklets.p_thresh = float(p)

    def set_source(self, source, fps: float = 30.0):
        """Bind a new video source: camera ID, clock, per-camera zones; reset all state."""
        with self._engine_lock:
            cam, start = parse_video_name(str(source)) if isinstance(source, str) else (None, None)
            self._camera_id = cam or ("Webcam" if not isinstance(source, str) else "Cam_00")
            self._clock = Clock(start=start, fps=fps if fps and fps > 1 else 30.0)
            self._zones.configure(camera_key(str(source)))
            self._frame_idx = 0
            self._assets.reset()
            if self._tracklets is not None:
                self._tracklets.set_gallery(self._gallery_for(source))
            self._logged_identity.clear()
            self._zone_alert_t.clear()
            self._banners.clear()
            self._alert_count = 0
            self._emit("info", f"Source started · {self._camera_id} · gallery {self.gallery_info}")
            self.reset_trails()
            self._reset_native_tracker()
            logger.info("Source bound: camera=%s zones=%s gallery=%s", self._camera_id,
                        list(self._zones.zones), self.gallery_info)

    def _gallery_for(self, source) -> PrototypeGallery:
        """Same-day enrollment: a CHIRLA video uses the gallery enrolled on its recording date."""
        default = Path(self._reid_cfg.get("gallery_file", "enrolled_gallery/gallery_prototypes.pkl"))
        date = source_date(str(source)) if isinstance(source, str) else None
        path = day_gallery_path(default, date)
        if date and not path.exists():
            logger.warning("No gallery enrolled for %s – using default %s", date, default)
            path = default
        gallery = PrototypeGallery.load(path)
        self.gallery_info = f"{gallery.meta.get('date', 'default')} · {len(gallery)} enrolled"
        return gallery

    def _reset_native_tracker(self):
        predictor = getattr(self._model, "predictor", None) if self._model is not None else None
        for trk in getattr(predictor, "trackers", None) or []:
            try:
                trk.reset()
            except Exception:
                pass
        if self._custom_tracker is not None:
            self._initialize_tracker()

    def close(self):
        if self._audit is not None:
            self._audit.close()
            self._audit = None

    def set_tracker(self, tracker_name: str):
        """
        Switch tracking backend dynamically.
        Supported: bytetrack, botsort (native) + ocsort, strongsort, trackformer (custom)
        """
        with self._engine_lock:
            available = get_available_trackers()

            if tracker_name not in available:
                logger.warning(
                    "Tracker '%s' not available (available: %s). Falling back to bytetrack.",
                    tracker_name, available
                )
                tracker_name = "bytetrack"

            self._tracker_name = tracker_name
            self.cfg["tracking"]["tracker"] = tracker_name
            self._initialize_tracker()
            logger.info("✓ Tracker switched to: %s", tracker_name)
    
    def _initialize_tracker(self):
        """Initialize the appropriate tracker based on tracker_name."""
        if self._tracker_name in ["bytetrack", "botsort"]:
            # Native Ultralytics trackers - will be handled by YOLO.track()
            self._custom_tracker = None
            logger.info("✓ Using native Ultralytics tracker: %s", self._tracker_name)
        elif self._tracker_name == "ocsort":
            self._custom_tracker = OCSORTOfficial()
            logger.info("✓ Using official OC-SORT tracker with Kalman filtering")
        elif self._tracker_name == "strongsort":
            self._custom_tracker = StrongSORTOfficial()
            logger.info("✓ Using official StrongSORT tracker with Kalman filtering")
        elif self._tracker_name == "trackformer":
            self._custom_tracker = TrackFormerOfficial()
            logger.info("✓ Using official TrackFormer tracker with attention-based matching")
        else:
            logger.warning("Unknown tracker '%s', using bytetrack", self._tracker_name)
            self._tracker_name = "bytetrack"
            self._custom_tracker = None

    # ── Model loading ─────────────────────────────────────────────────────────

    def load_model(self, model_key: str):
        """
        Load (or reload) a model by its key from config.

        Ultralytics YOLO expects:
          • .pt  files  → pass path directly
          • OpenVINO    → pass the *folder* that contains
                          <name>.xml + <name>.bin + metadata.yaml
                          (NOT the bare .xml path)
        """
        with self._engine_lock:
            paths = self.cfg.get("models", {})
            model_path = paths.get(model_key, "").strip()
            if not model_path:
                raise ValueError(f"Model key '{model_key}' not found in config.")

            p = Path(model_path)

            # If config still points to a bare .xml, auto-redirect to its parent folder
            if p.suffix.lower() == ".xml":
                logger.warning(
                    "Path points to a .xml file ('%s'). "
                    "Ultralytics needs the *folder* for OpenVINO models. "
                    "Redirecting to parent folder: '%s'", p, p.parent
                )
                p = p.parent

            if not p.exists():
                raise FileNotFoundError(
                    f"Model path not found: {p}\n"
                    "For OpenVINO models the path must be a folder containing:\n"
                    "  <name>.xml  +  <name>.bin  +  metadata.yaml\n"
                    "Example: models/openvino/yolo11s_int8_openvino_model/"
                )

            from ultralytics import YOLO  # lazy import

            logger.info("Loading model: %s  (%s)", model_key, str(p))
            self._model = YOLO(str(p), task="detect")
            self._model_path = str(p)
            self._model_key = model_key
            # Static OpenVINO exports must be fed the size they were exported at.
            self._model_imgsz = None
            meta = p / "metadata.yaml" if p.is_dir() else None
            if meta is not None and meta.exists():
                sz = (yaml.safe_load(meta.read_text()) or {}).get("imgsz")
                if sz:
                    self._model_imgsz = int(sz[0] if isinstance(sz, (list, tuple)) else sz)
                    logger.info("Model input size: %d", self._model_imgsz)

            # Re-create tracker instance to clear any stale tracker state.
            self._initialize_tracker()
            self.reset_trails()

            logger.info("Model ready.")

    # ── Per-frame inference ───────────────────────────────────────────────────

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Run detection + tracking + Re-ID on one BGR frame.
        Returns (annotated_frame, stats_dict).
        """
        with self._engine_lock:
            if self._model is None:
                return frame, {}

            self._frame_idx += 1

            ov_cfg = self.cfg.get("openvino", {})
            device_arg = ov_cfg.get("device", "CPU") if self._is_openvino() else None

            # ── Inference ─────────────────────────────────────────────────────
            # For custom trackers, run detection only
            if self._custom_tracker is not None:
                kwargs = dict(
                    conf=self.conf,
                    iou=self.iou,
                    classes=self.classes,
                    imgsz=self._model_imgsz or self.imgsz,
                    max_det=self.max_det,
                    verbose=False,
                )
                if device_arg:
                    kwargs["device"] = device_arg

                try:
                    results = self._model.predict(frame, **kwargs)
                except Exception as e:
                    logger.error("Detection failed: %s", str(e))
                    return frame, {}

                # Apply custom tracking
                if results and results[0].boxes is not None:
                    boxes = results[0].boxes
                    detections = []
                    for i in range(len(boxes)):
                        xyxy = boxes.xyxy[i].cpu().numpy().astype(np.float32)
                        conf = float(boxes.conf[i].cpu())
                        cls_id = int(boxes.cls[i].cpu())
                        detections.append([*xyxy, conf, cls_id])

                    detections = np.array(detections) if detections else np.empty((0, 6))
                    tracked = self._custom_tracker.update(detections)

                    # Convert tracked results to Results format
                    class NumpyWrapper:
                        """Wrapper to make numpy arrays compatible with torch tensor operations."""
                        def __init__(self, arr):
                            if isinstance(arr, np.ndarray):
                                self.arr = arr.item() if arr.size == 1 else arr
                            else:
                                self.arr = arr

                        def cpu(self):
                            """Return self (already on CPU as numpy)."""
                            return self

                        def numpy(self):
                            """Return numpy array."""
                            if isinstance(self.arr, np.ndarray):
                                return self.arr
                            return np.array([self.arr])

                        def __float__(self):
                            """Support float conversion."""
                            if isinstance(self.arr, np.ndarray):
                                return float(self.arr.item())
                            return float(self.arr)

                        def __int__(self):
                            """Support int conversion."""
                            if isinstance(self.arr, np.ndarray):
                                return int(self.arr.item())
                            return int(self.arr)

                        def numel(self):
                            """Return number of elements (PyTorch compatibility)."""
                            if isinstance(self.arr, np.ndarray):
                                return self.arr.size
                            return 1

                    class TensorLikeList(list):
                        """List subclass that acts like a PyTorch tensor for .numel()."""
                        def numel(self):
                            """Return number of elements."""
                            return len(self)

                    class TrackedBoxesWrapper:
                        """Wrapper for tracked boxes from custom trackers."""
                        def __init__(self, tracked_arr):
                            self.xyxy = [NumpyWrapper(tracked_arr[i, :4]) for i in range(len(tracked_arr))] if len(tracked_arr) > 0 else []
                            self.conf = [NumpyWrapper(tracked_arr[i, 4]) for i in range(len(tracked_arr))] if len(tracked_arr) > 0 else []
                            self.cls = [NumpyWrapper(tracked_arr[i, 5]) for i in range(len(tracked_arr))] if len(tracked_arr) > 0 else []
                            # Use special list that has .numel() method
                            id_list = TensorLikeList([NumpyWrapper(tracked_arr[i, 6]) for i in range(len(tracked_arr))])
                            self.id = id_list if len(id_list) > 0 else None
                            self._len = len(tracked_arr)

                        def __len__(self):
                            return self._len

                    tracked_boxes = TrackedBoxesWrapper(tracked)

                    result = type('obj', (object,), {
                        'boxes': tracked_boxes,
                        'names': results[0].names
                    })()
                    results = [result]
            else:
                # Use native Ultralytics tracker (ByteTrack or BoTSort)
                kwargs = dict(
                    conf=self.conf,
                    iou=self.iou,
                    classes=self.classes,
                    imgsz=self._model_imgsz or self.imgsz,
                    max_det=self.max_det,
                    tracker=f"{self._tracker_name}.yaml",
                    persist=True,
                    verbose=False,
                )
                if device_arg:
                    kwargs["device"] = device_arg

                try:
                    results = self._model.track(frame, **kwargs)
                except AssertionError as e:
                    logger.error("Tracker not supported: %s", str(e))
                    # Fall back to bytetrack
                    logger.info("Falling back to bytetrack")
                    self._tracker_name = "bytetrack"
                    self._custom_tracker = None
                    kwargs["tracker"] = "bytetrack.yaml"
                    results = self._model.track(frame, **kwargs)
                except Exception as e:
                    logger.error("Tracking failed: %s", str(e))
                    return frame, {}

            # ── Build annotated frame ─────────────────────────────────────────
            annotated = frame.copy()
            stats = {"detections": 0, "tracks": 0}

            dets: List[Tuple[Tuple[int, int, int, int], float, int, int]] = []
            names: Dict[int, str] = {}
            if results and results[0].boxes is not None:
                boxes = results[0].boxes
                names = results[0].names
                for i in range(len(boxes)):
                    x1, y1, x2, y2 = (int(v) for v in boxes.xyxy[i].cpu().numpy())
                    tid = int(boxes.id[i].cpu()) if boxes.id is not None else -1
                    dets.append(((x1, y1, x2, y2), float(boxes.conf[i].cpu()),
                                 int(boxes.cls[i].cpu()), tid))
                stats["detections"] = len(dets)
                stats["tracks"] = sum(1 for d in dets if d[3] >= 0)

            if self._reid_enabled and self._identity_mode != "legacy" and self._tracklets is not None:
                self._annotate_security(frame, annotated, dets, names, stats)
            else:
                for (x1, y1, x2, y2), conf_val, cls_id, tid in dets:
                    cls_name = names.get(cls_id, str(cls_id))
                    colour = _colour(tid)

                    # Re-ID
                    reid_id = None
                    if self._reid_enabled and tid >= 0 and self._gallery is not None and self._reid_extractor is not None:
                        crop = frame[max(0, y1):y2, max(0, x1):x2]
                        if self._gallery.has_tracker(tid):
                            reid_id = self._gallery.get_tracker_reid(tid)
                        if crop.size > 0 and (
                            not self._gallery.has_tracker(tid)
                            or self._frame_idx % self._reid_frame_interval == 0
                        ):
                            feat = self._reid_extractor.extract(crop)
                            reid_id = self._gallery.update(tid, feat)
                            logger.debug("ReID assign: T%d -> R%d (cls=%s conf=%.2f)",
                                         tid, reid_id, cls_name, conf_val)

                    # Trail
                    if self.show_trail and tid >= 0:
                        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                        self._trails[tid].append((cx, cy))
                        pts = list(self._trails[tid])
                        for j in range(1, len(pts)):
                            alpha = j / len(pts)
                            t_col = tuple(int(c * alpha) for c in colour)
                            cv2.line(annotated, pts[j - 1], pts[j], t_col,
                                     max(1, self.box_thick - 1))

                    # Bounding box
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), colour,
                                  self.box_thick)

                    # Label
                    label_parts = []
                    if self.show_labels:
                        label_parts.append(cls_name)
                    if self.show_conf:
                        label_parts.append(f"{conf_val:.2f}")
                    if self.show_tid and tid >= 0:
                        label_parts.append(f"T{tid}")
                    if self.show_rid and reid_id is not None:
                        label_parts.append(f"R{reid_id}")

                    if label_parts:
                        label = " ".join(label_parts)
                        (tw, th), _ = cv2.getTextSize(
                            label, cv2.FONT_HERSHEY_SIMPLEX,
                            self.font_scale, 1
                        )
                        cv2.rectangle(annotated,
                                      (x1, y1 - th - 6), (x1 + tw + 4, y1),
                                      colour, -1)
                        cv2.putText(annotated, label, (x1 + 2, y1 - 4),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    self.font_scale, (255, 255, 255), 1,
                                    cv2.LINE_AA)

            # ── FPS overlay ───────────────────────────────────────────────────
            now = time.perf_counter()
            dt = now - self._t_prev
            self._t_prev = now
            if dt > 0:
                self._fps_buffer.append(1.0 / dt)
            fps = sum(self._fps_buffer) / max(len(self._fps_buffer), 1)
            stats["fps"] = round(fps, 1)

            if self.show_fps:
                cv2.putText(annotated, f"FPS {fps:.1f}",
                            (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 255, 100), 2, cv2.LINE_AA)

            return annotated, stats

    # ── Security layer (identity · zones · assets · audit) ─────────────────────

    def _annotate_security(self, frame: np.ndarray, annotated: np.ndarray,
                           dets: list, names: Dict[int, str], stats: dict):
        h, w = frame.shape[:2]
        self._clock.set_frame(self._frame_idx)
        t_sec = self._clock.seconds

        persons = [d for d in dets if d[2] == PERSON_CLASS and d[3] >= 0 and d[1] >= self.person_conf]
        objects = [(names.get(d[2], str(d[2])), d[0]) for d in dets if d[2] != PERSON_CLASS]
        idents = self._tracklets.update(frame, self._frame_idx,
                                        [(tid, box, conf) for box, conf, _, tid in persons])

        polys = self._zones.pixel_polygons(w, h) if self.zones_enabled else {}
        for name, poly in polys.items():
            self._draw_dashed_poly(annotated, poly, COL_ZONE)
            x, y = poly[:, 0].min(), poly[:, 1].min()
            self._draw_label(annotated, zone_name_pretty(name).replace("_", " "), (x, y), COL_ZONE, (0, 0, 0))

        person_info, person_zone, people = [], {}, []
        counts = {STATUS_KNOWN: 0, STATUS_UNKNOWN: 0, STATUS_PENDING: 0}
        for box, _, _, tid in persons:
            ident = idents.get(tid, Identity())
            counts[ident.status] += 1
            zones_in = self._zones.zones_containing(box, w, h) if polys else []
            person_zone[tid] = zones_in[0] if zones_in else None
            intrusion = ident.status == STATUS_UNKNOWN and bool(zones_in)
            people.append({"tid": tid, "status": ident.status, "confidence": ident.confidence,
                           "name": (display_name(ident.label) if ident.status == STATUS_KNOWN
                                    else "UNKNOWN" if ident.status == STATUS_UNKNOWN else "Identifying…"),
                           "crops": ident.n_crops, "zone": zone_name_pretty(zones_in[0]) if zones_in else "",
                           "intrusion": intrusion})

            self._log_identity(tid, ident)
            if intrusion:
                for z in zones_in:
                    self._zone_alert(tid, ident, z, t_sec)

            colour = (COL_ALERT if intrusion else COL_KNOWN if ident.status == STATUS_KNOWN
                      else COL_UNKNOWN if ident.status == STATUS_UNKNOWN else COL_PENDING)
            self._draw_trail(annotated, tid, box, colour)
            thick = self.box_thick + (1 if intrusion else 0)
            cv2.rectangle(annotated, box[:2], box[2:], colour, thick)
            self._draw_label(annotated, self._identity_text(tid, ident), box[:2], colour, (255, 255, 255))
            person_info.append((tid, box, ident.status, ident.label))

        for ev in self._assets.update(self._frame_idx, t_sec, objects, person_info):
            ident = idents.get(ev.track_id, Identity())
            zone = person_zone.get(ev.track_id)
            name = ev.asset.class_name.capitalize()
            tag = {"possession": f"{name}_Possession_{self._log_name(ident)}",
                   "displaced_unknown": f"{name}_Displaced",
                   "unknown_interaction": f"{name}_Unknown_Interaction"}[ev.kind]
            self._audit_log(ev.track_id, ident, zone_name_pretty(zone) if zone else "None", tag)
            if ev.kind != "possession":
                self._raise_banner(ev.text, t_sec)
            else:
                self._emit("asset", ev.text)

        for a in self._assets.assets:
            if a.last_seen != self._frame_idx:
                continue
            col = COL_ALERT if a.state == "ALERT" else COL_ASSET
            cv2.rectangle(annotated, a.box[:2], a.box[2:], col, self.box_thick)
            state = "ALERT" if a.state == "ALERT" else a.state
            self._draw_label(annotated, f"{a.class_name.capitalize()}: {state}", a.box[:2], col, (0, 0, 0))

        self._draw_banners(annotated, t_sec)
        stats.update(known=counts[STATUS_KNOWN], unknown=counts[STATUS_UNKNOWN],
                     pending=counts[STATUS_PENDING], alerts=self._alert_count,
                     camera=self._camera_id, time=self._clock.stamp(), people=people,
                     assets=[(a.class_name, a.state) for a in self._assets.assets
                             if a.last_seen == self._frame_idx])

    def _identity_text(self, tid: int, ident: Identity) -> str:
        suffix = f" | T{tid}" if self.show_tid else ""
        if ident.status == STATUS_PENDING:
            need = self._tracklets.min_crops if self._tracklets else 0
            return f"Identifying {ident.n_crops}/{need}{suffix}"
        name = display_name(ident.label) if ident.status == STATUS_KNOWN else "UNKNOWN"
        return f"{name} | Conf: {ident.confidence:.2f}{suffix}"

    @staticmethod
    def _log_name(ident: Identity) -> str:
        if ident.status == STATUS_KNOWN and ident.label:
            return ident.label.replace("emp_", "Emp_")
        return "UNKNOWN" if ident.status == STATUS_UNKNOWN else "PENDING"

    def _audit_log(self, tid: int, ident: Identity, zone: str = "None", asset: str = "None"):
        if self._audit is None:
            self._audit = AuditLogger(self._audit_path)
        self._audit.log(self._clock.stamp(), self._camera_id, tid, self._log_name(ident),
                        ident.status, ident.confidence, zone, asset)

    def _log_identity(self, tid: int, ident: Identity):
        """Log a track when its identity is first resolved or changes."""
        if ident.status == STATUS_PENDING:
            return
        key = (ident.status, ident.label if ident.status == STATUS_KNOWN else None)
        if self._logged_identity.get(tid) != key:
            self._logged_identity[tid] = key
            self._audit_log(tid, ident)
            who = display_name(ident.label) if ident.status == STATUS_KNOWN else "UNKNOWN visitor"
            self._emit("identity", f"T{tid} identified as {who} (conf {ident.confidence:.2f})")

    def _zone_alert(self, tid: int, ident: Identity, zone: str, t_sec: float, cooldown: float = 10.0):
        last = self._zone_alert_t.get((tid, zone))
        if last is not None and t_sec - last < cooldown:
            return
        self._zone_alert_t[(tid, zone)] = t_sec
        self._audit_log(tid, ident, zone_name_pretty(zone), "None")
        self._raise_banner("Intrusion Alert: Unregistered Person in Restricted Area", t_sec)

    def _emit(self, level: str, text: str):
        """Push an event to the UI feed (level: alert | identity | asset | info)."""
        self._event_no += 1
        self._events.append((self._event_no, level, self._clock.stamp(), self._camera_id, text))

    def events_since(self, last_no: int) -> list:
        """Events newer than last_no (thread-safe snapshot for the UI)."""
        return [e for e in list(self._events) if e[0] > last_no]

    def _raise_banner(self, text: str, t_sec: float, duration: float = 4.0):
        self._emit("alert", text)
        self._alert_count += 1
        # One banner per alert text; a repeat just extends it.
        for i, (old, _) in enumerate(self._banners):
            if old == text:
                del self._banners[i]
                break
        self._banners.append((text, t_sec + duration))
        logger.warning("ALERT [%s %s] %s", self._camera_id, self._clock.stamp(), text)

    def _draw_banners(self, img: np.ndarray, t_sec: float):
        active = [b for b in self._banners if b[1] >= t_sec]
        y = 40
        for text, _ in active[-2:]:
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
            x = max(10, (img.shape[1] - tw) // 2)
            cv2.rectangle(img, (x - 12, y - th - 12), (x + tw + 12, y + 10), COL_ALERT, -1)
            cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
            y += th + 30

    def _draw_trail(self, img: np.ndarray, tid: int, box, colour):
        if not self.show_trail:
            return
        self._trails[tid].append(((box[0] + box[2]) // 2, box[3]))
        pts = list(self._trails[tid])
        for j in range(1, len(pts)):
            alpha = j / len(pts)
            cv2.line(img, pts[j - 1], pts[j], tuple(int(c * alpha) for c in colour),
                     max(1, self.box_thick - 1))

    def _draw_label(self, img: np.ndarray, text: str, org, bg, fg):
        x, y = int(org[0]), int(org[1])
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, 1)
        y = max(y, th + 6)
        cv2.rectangle(img, (x, y - th - 6), (x + tw + 4, y), bg, -1)
        cv2.putText(img, text, (x + 2, y - 4), cv2.FONT_HERSHEY_SIMPLEX,
                    self.font_scale, fg, 1, cv2.LINE_AA)

    @staticmethod
    def _draw_dashed_poly(img: np.ndarray, poly: np.ndarray, colour, dash: int = 14, thick: int = 2):
        pts = [tuple(int(v) for v in p) for p in poly]
        for a, b in zip(pts, pts[1:] + pts[:1]):
            length = float(np.hypot(b[0] - a[0], b[1] - a[1]))
            for s in range(0, int(length), dash * 2):
                e = min(s + dash, length)
                p1 = (int(a[0] + (b[0] - a[0]) * s / length), int(a[1] + (b[1] - a[1]) * s / length))
                p2 = (int(a[0] + (b[0] - a[0]) * e / length), int(a[1] + (b[1] - a[1]) * e / length))
                cv2.line(img, p1, p2, colour, thick, cv2.LINE_AA)

    # ── Utility ───────────────────────────────────────────────────────────────

    def _is_openvino(self) -> bool:
        p = Path(self._model_path or "")
        # OpenVINO model is either a folder or a .xml file
        return p.is_dir() or p.suffix.lower() == ".xml"

    def reset_trails(self):
        with self._engine_lock:
            self._trails.clear()

    @staticmethod
    def load_config(path: str = "config.yaml") -> dict:
        with open(path, "r") as f:
            return yaml.safe_load(f)
