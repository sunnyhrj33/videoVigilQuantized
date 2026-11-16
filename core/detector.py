"""
core/detector.py
────────────────────────────────────────────────────────────────────────────────
YOLO11 model loading (PyTorch .pt & OpenVINO IR folders), native Ultralytics
multi-object tracking (ByteTrack & BoTSORT), and lightweight hardware specs.
────────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import logging
import os
import platform
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml

from core.geometry import Box

logger = logging.getLogger(__name__)

NATIVE_TRACKERS = ["bytetrack", "botsort"]


def get_available_trackers() -> List[str]:
    """Return available native Ultralytics trackers (ByteTrack and BoTSORT)."""
    try:
        try:
            from ultralytics.cfg import TRACKER_DIR
            tracker_dir = Path(TRACKER_DIR)
        except (ImportError, AttributeError):
            import ultralytics
            tracker_dir = Path(ultralytics.__file__).parent / "cfg" / "trackers"
        if tracker_dir.exists():
            found = sorted(f.stem for f in tracker_dir.glob("*.yaml") if f.stem in NATIVE_TRACKERS)
            if found:
                return found
    except Exception as e:
        logger.warning("Could not query Ultralytics trackers (%s), using defaults", e)
    return list(NATIVE_TRACKERS)


@dataclass
class SystemSpec:
    """Lightweight hardware spec summary for the console status bar."""
    os_name: str
    cpu_cores: int
    total_ram_gb: float
    has_gpu: bool
    gpu_name: str


_CACHED_SPEC: Optional[SystemSpec] = None


def get_system_spec() -> SystemSpec:
    """Detect CPU cores, total RAM, and GPU name (cached after first call)."""
    global _CACHED_SPEC
    if _CACHED_SPEC is not None:
        return _CACHED_SPEC

    cores = os.cpu_count() or 4
    ram_gb = 8.0
    try:
        import psutil
        ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    except Exception:
        pass

    gpu_name = "none"
    has_gpu = False
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=3,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if res.returncode == 0 and res.stdout.strip():
            gpu_name = res.stdout.strip().splitlines()[0]
            has_gpu = True
    except Exception:
        pass

    _CACHED_SPEC = SystemSpec(platform.system(), cores, ram_gb, has_gpu, gpu_name)
    return _CACHED_SPEC


def resolve_model_path(cfg: dict, model_key: str) -> Tuple[Path, Optional[int]]:
    """
    Resolve a model key from config['models'] to a verified Path and optional static
    input size (`imgsz` from OpenVINO `metadata.yaml`).
    """
    paths = cfg.get("models", {})
    raw = paths.get(model_key, "").strip()
    if not raw:
        raise ValueError(f"Model key '{model_key}' not found in config.")

    p = Path(raw)
    if p.suffix.lower() == ".xml":
        logger.warning("Redirecting bare .xml path '%s' to parent folder '%s'", p, p.parent)
        p = p.parent

    if not p.exists():
        raise FileNotFoundError(
            f"Model path not found: {p}\n"
            "For OpenVINO models the path must be a folder containing:\n"
            "  <name>.xml  +  <name>.bin  +  metadata.yaml"
        )

    model_imgsz = None
    meta = p / "metadata.yaml" if p.is_dir() else None
    if meta is not None and meta.exists():
        sz = (yaml.safe_load(meta.read_text()) or {}).get("imgsz")
        if sz:
            model_imgsz = int(sz[0] if isinstance(sz, (list, tuple)) else sz)

    return p, model_imgsz


class YoloDetector:
    """Thin wrapper around Ultralytics YOLO for detection and native tracking."""

    def __init__(self):
        self.model = None
        self.model_key: Optional[str] = None
        self.model_path: Optional[str] = None
        self.model_imgsz: Optional[int] = None

    @property
    def is_ready(self) -> bool:
        return self.model is not None

    @property
    def is_openvino(self) -> bool:
        if not self.model_path:
            return False
        p = Path(self.model_path)
        return p.is_dir() or p.suffix.lower() == ".xml"

    def load(self, cfg: dict, model_key: str):
        p, imgsz = resolve_model_path(cfg, model_key)
        from ultralytics import YOLO

        logger.info("Loading model: %s (%s)", model_key, p)
        self.model = YOLO(str(p), task="detect")
        self.model_path = str(p)
        self.model_key = model_key
        self.model_imgsz = imgsz
        if imgsz:
            logger.info("Model static input size: %d", imgsz)
        logger.info("Model ready.")

    def reset_tracker(self):
        predictor = getattr(self.model, "predictor", None) if self.model is not None else None
        for trk in getattr(predictor, "trackers", None) or []:
            try:
                trk.reset()
            except Exception:
                pass

    def track(self, frame: np.ndarray, *, conf: float, iou: float, classes: Optional[List[int]],
              imgsz: int, max_det: int, tracker_name: str, device: Optional[str] = None
              ) -> Tuple[List[Tuple[Box, float, int, int]], Dict[int, str]]:
        """
        Run YOLO detection + native tracker on a frame.
        Returns (detections, class_names) where each detection is ((x1, y1, x2, y2), conf, cls_id, track_id).
        """
        if self.model is None:
            return [], {}

        kwargs = dict(
            conf=conf,
            iou=iou,
            classes=classes,
            imgsz=self.model_imgsz or imgsz,
            max_det=max_det,
            tracker=f"{tracker_name}.yaml",
            persist=True,
            verbose=False,
        )
        if device:
            kwargs["device"] = device

        results = self.model.track(frame, **kwargs)
        dets: List[Tuple[Box, float, int, int]] = []
        names: Dict[int, str] = {}
        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            names = results[0].names
            for i in range(len(boxes)):
                x1, y1, x2, y2 = (int(v) for v in boxes.xyxy[i].cpu().numpy())
                tid = int(boxes.id[i].cpu()) if boxes.id is not None else -1
                dets.append(((x1, y1, x2, y2), float(boxes.conf[i].cpu()), int(boxes.cls[i].cpu()), tid))
        return dets, names

    def predict(self, frame: np.ndarray, *, conf: float = 0.4, classes: Optional[List[int]] = None,
                imgsz: int = 640):
        """Run single-frame detection without tracking."""
        if self.model is None:
            return None
        return self.model.predict(
            frame, classes=classes, conf=conf, imgsz=self.model_imgsz or imgsz, verbose=False
        )[0]
