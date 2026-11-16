"""
core/reid.py
────────────────────────────────────────────────────────────────────────────────
Person Re-Identification (Re-ID) feature extraction and legacy runtime gallery.

Supports OSNet and ResNet models via torchreid + OpenVINO IR acceleration:
  • osnet_ain_x1_0 (default, MSMT17 trained, OpenVINO IR supported)
  • osnet_x1_0, osnet_x0_75, osnet_x0_5, osnet_x0_25
  • resnet50, resnet101
────────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

REID_MODELS = [
    "osnet_ain_x1_0",
    "osnet_x1_0",
    "osnet_x0_75",
    "osnet_x0_5",
    "osnet_x0_25",
    "resnet50",
    "resnet101",
]


def get_available_reid_models() -> Dict[str, bool]:
    """Check which Re-ID models are available via torchreid."""
    try:
        import torchreid  # noqa: F401
        ok = True
    except ImportError:
        ok = False
    return {m: ok for m in REID_MODELS}


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine distance in [0, 2] between two L2-normalised vectors."""
    a = a / (np.linalg.norm(a) + 1e-6)
    b = b / (np.linalg.norm(b) + 1e-6)
    return float(1.0 - np.dot(a, b))


# ─────────────────────────────────────────────────────────────────────────────
#  Legacy Runtime Re-ID Gallery
# ─────────────────────────────────────────────────────────────────────────────

class ReIDGallery:
    """
    Minimal EMA-based Re-ID gallery (used in legacy runtime ID mode).
    Maps tracker IDs → persistent global IDs using cosine feature distance.
    """

    def __init__(self, thresh: float = 0.4, ema_rate: float = 0.9, max_dist: float = 0.6):
        self.thresh = thresh
        self.ema_rate = ema_rate
        self.max_dist = max_dist

        self._gallery: Dict[int, np.ndarray] = {}      # reid_id → feature
        self._tracker_to_reid: Dict[int, int] = {}     # tracker_id → reid_id
        self._next_id = 1

    def update(self, tracker_id: int, feature: np.ndarray) -> int:
        """Return stable reid_id for this tracker_id, updating the gallery."""
        feat = feature / (np.linalg.norm(feature) + 1e-6)

        if tracker_id in self._tracker_to_reid:
            reid_id = self._tracker_to_reid[tracker_id]
            old = self._gallery[reid_id]
            self._gallery[reid_id] = self.ema_rate * old + (1 - self.ema_rate) * feat
            return reid_id

        best_id, best_dist = None, float("inf")
        for gid, gfeat in self._gallery.items():
            d = _cosine_distance(feat, gfeat)
            if d < best_dist:
                best_dist = d
                best_id = gid

        if best_id is not None and best_dist < self.max_dist:
            reid_id = best_id
        else:
            reid_id = self._next_id
            self._next_id += 1
            self._gallery[reid_id] = feat

        self._tracker_to_reid[tracker_id] = reid_id
        return reid_id

    def get_tracker_reid(self, tracker_id: int) -> Optional[int]:
        return self._tracker_to_reid.get(tracker_id)

    def has_tracker(self, tracker_id: int) -> bool:
        return tracker_id in self._tracker_to_reid


# ─────────────────────────────────────────────────────────────────────────────
#  Re-ID Feature Extractor
# ─────────────────────────────────────────────────────────────────────────────

class ReIDExtractor:
    """
    Feature extractor for OSNet and ResNet models via torchreid or OpenVINO IR.
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
        self._fallback_used = False

        try:
            import torchreid
            model_path = self._resolve_weights(model_name, weights)
            ov_xml = (Path(model_path).parent / f"{model_name}_openvino" / f"{model_name}.xml"
                      if model_path else Path("models/reid") / f"{model_name}_openvino" / f"{model_name}.xml")
            if backend in ("auto", "openvino") and ov_xml is not None and ov_xml.exists():
                import openvino as ov
                self._ov_model = ov.Core().compile_model(str(ov_xml), "CPU", {"PERFORMANCE_HINT": "LATENCY"})
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
        except Exception as e:
            logger.warning(
                "Could not load Re-ID model '%s' (%s) – using fallback colour histogram.",
                model_name, str(e)
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
        resized = cv2.resize(bgr_crop, (64, 128))
        hists = []
        for ch in range(3):
            h, _ = np.histogram(resized[:, :, ch], bins=32, range=(0, 256))
            hists.append(h.astype(np.float32))
        feat = np.concatenate(hists)
        return feat / (feat.sum() + 1e-6)
