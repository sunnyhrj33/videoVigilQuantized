"""
core/engine.py
────────────────────────────────────────────────────────────────────────────────
Slim DetectionEngine orchestrating:
  • YOLO11 detection + native ByteTrack / BoTSORT tracking (core.detector)
  • Person Re-ID feature extraction & open-set classification (core.reid, core.open_set)
  • Spatial security zones, asset displacement & suspect tracking (core.security, core.suspects)
  • OpenCV frame annotation overlays (core.overlay)
────────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import defaultdict, deque
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml

from core.chirla import camera_key, day_gallery_path, parse_video_name, source_date
from core.detector import YoloDetector, get_available_trackers
from core.geometry import centre, l2n
from core.open_set import (NO_STAFF, STATUS_KNOWN, STATUS_PENDING, STATUS_UNKNOWN, Identity,
                           OpenSetClassifier, PrototypeGallery, TrackletManager,
                           display_name, load_day_gallery, tracklet_features)
from core.overlay import (COL_ALERT, COL_KNOWN, COL_PENDING, COL_UNKNOWN,
                          draw_assets, draw_banners, draw_fps, draw_label,
                          draw_trail, draw_zones, track_colour)
from core.reid import ReIDExtractor, ReIDGallery
from core.security import (ABANDONED, IN_TRANSIT, MISSING, AssetMonitor,
                           AuditLogger, Clock, ZoneMonitor, zone_name_pretty)
from core.suspects import SuspectGallery, TrackHistory

logger = logging.getLogger(__name__)

PERSON_CLASS = 0


class DetectionEngine:
    """
    Orchestrates YOLO detection, native multi-object tracking, open-set Re-ID,
    spatial security zones, asset monitoring, suspect auto-enrollment, and audit logging.
    """

    def __init__(self, cfg: dict, shared: Optional["DetectionEngine"] = None):
        """
        shared: another engine whose Re-ID models, suspect gallery and audit / theft logs this
        one reuses – one engine per camera when several cameras play together. Each engine
        keeps its own detector (the tracker state lives in it) and per-camera state.
        """
        self.cfg = cfg
        self._shared = shared
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

        self._engine_lock = threading.RLock()

        # Detector & Native Tracker
        self._detector = YoloDetector()
        track_cfg = cfg.get("tracking", {})
        self._tracker_name = track_cfg.get("tracker", "bytetrack")
        if self._tracker_name not in get_available_trackers():
            self._tracker_name = "bytetrack"

        # Security layer: enrolled-gallery identity, zones, assets, audit log
        self._identity_mode = reid_cfg.get("mode", "tracklet")
        self._tracklets: Optional[TrackletManager] = None
        self.gallery_info = "—"
        self.gallery_report: dict = {}
        self.extra_gallery: Optional[PrototypeGallery] = None
        self._zones = ZoneMonitor(cfg)
        self._zones.configure(None)
        self._assets = AssetMonitor(cfg)
        self._clock = Clock()
        self._camera_id = "Cam_00"
        self._audit: Optional[AuditLogger] = None
        self._audit_path = cfg.get("output", {}).get("audit_log", "output/audit_log.csv")
        self._logged_identity: Dict[int, Tuple[str, Optional[str]]] = {}
        self._zone_alert_t: Dict[Tuple[int, str], float] = {}
        self._banners: deque = deque(maxlen=3)
        self._alert_count = 0
        self._events: deque = deque(maxlen=500)
        self._event_no = 0

        # Suspect tracking
        self._st = cfg.get("suspect_tracking", {}) or {}
        self.suspects = shared.suspects if shared is not None else SuspectGallery(
            self._st.get("gallery_dir", "output/suspects"), self._st.get("match_threshold", 0.75))
        self._history = TrackHistory(self._st.get("history_log", "output/track_history.csv"),
                                     self._st.get("history_interval_frames", 30))
        self._theft: Optional[AuditLogger] = None
        self._theft_path = self._st.get("audit_log_path", "output/theft_audit_log.csv")
        self._suspect_of: Dict[int, str] = {}
        self._suspect_reid: Dict[int, bool] = {}
        self._suspect_checked: Dict[int, int] = {}
        self._suspect_weak: Dict[int, int] = {}
        self._suspect_cleared: set = set()
        self._pending_disp: Dict[int, tuple] = {}
        self._idents: Dict[int, Identity] = {}
        self._anchors_path = Path(cfg.get("output", {}).get("output_dir", "output/")) / "asset_anchors.json"
        self._cam_key: Optional[str] = None
        self._source_name = ""
        self._frame_size = (0, 0)

        # Re-ID
        self._reid_enabled = reid_cfg.get("enabled", True)
        self._reid_model_name: Optional[str] = None
        self._reid_extractor_cache: Dict[str, ReIDExtractor] = (
            shared._reid_extractor_cache if shared is not None else {})
        self._reid_extractor: Optional[ReIDExtractor] = None
        self._gallery: Optional[ReIDGallery] = None
        if self._reid_enabled:
            self.set_reid_model(reid_cfg.get("model", "osnet_x0_25"))
        self._reid_frame_interval = max(1, int(reid_cfg.get("frame_interval", 3)))

        # Trail & FPS state
        self._trails: Dict[int, deque] = defaultdict(lambda: deque(maxlen=self.trail_len))
        self._fps_buffer: deque = deque(maxlen=30)
        self._t_prev = time.perf_counter()
        self._frame_idx = 0

    @property
    def _model(self):
        return self._detector.model

    @property
    def _model_key(self) -> Optional[str]:
        return self._detector.model_key

    # ── Model, Re-ID & Tracker configuration ─────────────────────────────────

    def load_model(self, model_key: str):
        """Load (or reload) a YOLO model by its key from config."""
        with self._engine_lock:
            self._detector.load(self.cfg, model_key)
            self.reset_trails()

    def set_reid_model(self, model_name: str):
        """Switch Re-ID model and re-initialise the open-set tracklet manager."""
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
                self._gallery = ReIDGallery(
                    thresh=self._reid_cfg.get("reid_thresh", 0.4),
                    ema_rate=self._reid_cfg.get("feature_update_rate", 0.9),
                    max_dist=self._reid_cfg.get("max_reid_distance", 0.6),
                )
                self._reid_frame_interval = max(1, int(self._reid_cfg.get("frame_interval", 3)))
                self._init_open_set()
            except Exception as e:
                logger.error("Failed to load Re-ID model '%s': %s. Using fallback.", model_name, e)
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
            logger.warning("Enrolled gallery %s not found – run scripts/build_gallery.py. "
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

    def set_tracker(self, tracker_name: str):
        """Switch native Ultralytics tracking backend ('bytetrack' or 'botsort')."""
        with self._engine_lock:
            available = get_available_trackers()
            if tracker_name not in available:
                logger.warning("Tracker '%s' not available (%s). Falling back to bytetrack.", tracker_name, available)
                tracker_name = "bytetrack"
            self._tracker_name = tracker_name
            self.cfg.setdefault("tracking", {})["tracker"] = tracker_name
            self._detector.reset_tracker()
            logger.info("✓ Tracker switched to: %s", tracker_name)

    def set_identity_mode(self, mode: str):
        """'tracklet' (open-set), 'naive' (single-frame baseline) or 'legacy' (runtime EMA IDs)."""
        with self._engine_lock:
            self._identity_mode = mode
            if self._tracklets is not None and mode != "legacy":
                self._tracklets.set_mode(mode)
            self._logged_identity.clear()
            logger.info("✓ Identity mode: %s", mode)

    def set_open_set_threshold(self, p: float):
        """P(Known) cut-off between Employee and UNKNOWN (live)."""
        self._reid_cfg["open_set_p_threshold"] = float(p)
        if self._tracklets is not None:
            self._tracklets.p_thresh = float(p)

    def skip_frame(self):
        """A source frame was dropped (real-time playback): keep clock / timers in step."""
        with self._engine_lock:
            self._frame_idx += 1

    def set_source(self, source, fps: float = 30.0, start_frame: int = 0,
                   gallery_date: Optional[str] = None):
        """
        Bind a new video source: camera ID, clock, per-camera zones / anchors; reset state.
        gallery_date picks the enrolled staff gallery (see _gallery_for); None keeps the
        console behaviour of reading the date from a CHIRLA filename.
        """
        with self._engine_lock:
            cam, start = parse_video_name(str(source)) if isinstance(source, str) else (None, None)
            fps = fps if fps and fps > 1 else 30.0
            if start is not None and start_frame:
                start = start + timedelta(seconds=start_frame / fps)
            self._camera_id = cam or ("Webcam" if not isinstance(source, str) else "Cam_00")
            self._clock = Clock(start=start, fps=fps)
            self._cam_key = camera_key(str(source)) if isinstance(source, str) else "webcam"
            self._source_name = Path(str(source)).name if isinstance(source, str) else str(source)
            self._zones.configure(self._cam_key)
            self._frame_idx = 0
            self._assets.configure(self._cam_key, self._load_anchors().get(self._cam_key or "webcam", {}))
            self._history.reset()
            self._suspect_of.clear()
            self._suspect_reid.clear()
            self._suspect_checked.clear()
            self._suspect_weak.clear()
            self._suspect_cleared.clear()
            self._pending_disp.clear()
            if self._tracklets is not None:
                self._tracklets.set_gallery(self._gallery_for(source, gallery_date))
            else:
                self.gallery_report = {"date": None, "how": "legacy", "staff": 0, "named": 0,
                                       "warning": "Enrolled identification is off (legacy runtime Re-ID)"}
            self._logged_identity.clear()
            self._zone_alert_t.clear()
            self._banners.clear()
            self._alert_count = 0
            self._emit("info", f"Source started · {self._camera_id} · gallery {self.gallery_info}")
            self.reset_trails()
            self._detector.reset_tracker()
            logger.info("Source bound: camera=%s zones=%s gallery=%s", self._camera_id,
                        list(self._zones.zones), self.gallery_info)

    def _gallery_for(self, source, gallery_date: Optional[str] = None) -> PrototypeGallery:
        """
        Same-day enrollment: pick the staff gallery for this source.
          gallery_date=None          – the CHIRLA video's recording date, else the default gallery
          gallery_date="YYYY-MM-DD"  – the gallery enrolled that day; if that day can't be used,
                                       no staff gallery (another day's clothes would mislabel people)
          gallery_date="none"        – no staff gallery
        People named in the dashboard (extra_gallery) are merged in either way.
        Sets self.gallery_report = {date, how, staff, named, warning}.
        """
        default = Path(self._reid_cfg.get("gallery_file", "enrolled_gallery/gallery_prototypes.pkl"))
        warning = None
        if gallery_date is None:
            date = source_date(str(source)) if isinstance(source, str) else None
            path, how = day_gallery_path(default, date), "filename"
            if not date or not path.exists():
                if date:
                    logger.warning("No gallery enrolled for %s – using default %s", date, default)
                path, how = default, "default"
            staff = PrototypeGallery.load(path)
        elif gallery_date == NO_STAFF:
            staff, how = None, "none"
        else:
            staff, warning = load_day_gallery(default, gallery_date, self._reid_model_name)
            how = "selected"
            if warning:
                logger.warning("%s – analysing without enrolled staff", warning)
        date = staff.meta.get("date") if staff is not None else (
            gallery_date if gallery_date != NO_STAFF else None)
        gallery = staff if staff is not None else PrototypeGallery.empty({"date": date or "no staff"})
        n_staff = len(gallery)
        if self.extra_gallery is not None:
            gallery = gallery.merged(self.extra_gallery)
        self.gallery_report = {"date": date, "how": how, "staff": n_staff,
                               "named": len(gallery) - n_staff, "warning": warning}
        self.gallery_info = (f"{date or 'default'} · {n_staff} enrolled" if staff is not None
                             else f"{date or '—'} · no staff gallery")
        return gallery

    def close(self):
        if self._shared is None:            # shared logs belong to (and are closed by) the owner
            for log in (self._audit, self._theft):
                if log is not None:
                    log.close()
        self._audit = self._theft = None
        self._history.close()

    def adopt_settings(self, src: "DetectionEngine"):
        """Copy every operator-tunable setting (thresholds, overlays, models, mode) from src."""
        with self._engine_lock:
            for k in ("conf", "iou", "classes", "imgsz", "max_det", "person_conf", "zones_enabled",
                      "show_labels", "show_conf", "show_tid", "show_rid", "show_trail", "box_thick",
                      "font_scale", "show_fps", "_reid_enabled", "extra_gallery"):
                setattr(self, k, getattr(src, k))
            self._assets.enabled = src._assets.enabled
            if src._reid_model_name and src._reid_model_name != self._reid_model_name:
                self.set_reid_model(src._reid_model_name)
            if src._model_key and src._model_key != self._model_key:
                self.load_model(src._model_key)
            if src._tracker_name != self._tracker_name:
                self.set_tracker(src._tracker_name)
            if src._identity_mode != self._identity_mode:
                self.set_identity_mode(src._identity_mode)
            self.set_open_set_threshold(src._reid_cfg.get("open_set_p_threshold", 0.5))

    # ── Per-frame inference ───────────────────────────────────────────────────

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, dict]:
        """Run detection + tracking + Re-ID + security rules on one BGR frame."""
        with self._engine_lock:
            if not self._detector.is_ready:
                return frame, {}

            self._frame_idx += 1
            ov_cfg = self.cfg.get("openvino", {})
            device_arg = ov_cfg.get("device", "CPU") if self._detector.is_openvino else None

            try:
                dets, names = self._detector.track(
                    frame, conf=self.conf, iou=self.iou, classes=self.classes,
                    imgsz=self.imgsz, max_det=self.max_det,
                    tracker_name=self._tracker_name, device=device_arg,
                )
            except Exception as e:
                logger.error("Tracking failed: %s", e)
                return frame, {}

            annotated = frame.copy()
            stats = {
                "detections": len(dets),
                "tracks": sum(1 for d in dets if d[3] >= 0),
            }

            if self._reid_enabled and self._identity_mode != "legacy" and self._tracklets is not None:
                self._annotate_security(frame, annotated, dets, names, stats)
            else:
                self._annotate_legacy(frame, annotated, dets, names)

            now = time.perf_counter()
            dt = now - self._t_prev
            self._t_prev = now
            if dt > 0:
                self._fps_buffer.append(1.0 / dt)
            fps = sum(self._fps_buffer) / max(len(self._fps_buffer), 1)
            stats["fps"] = round(fps, 1)

            if self.show_fps:
                draw_fps(annotated, fps)

            return annotated, stats

    def _annotate_legacy(self, frame: np.ndarray, annotated: np.ndarray,
                         dets: list, names: Dict[int, str]):
        for (x1, y1, x2, y2), conf_val, cls_id, tid in dets:
            cls_name = names.get(cls_id, str(cls_id))
            colour = track_colour(tid)

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

            draw_trail(annotated, self._trails, tid, (x1, y1, x2, y2), colour,
                       self.box_thick, self.show_trail, bottom_anchor=False)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), colour, self.box_thick)

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
                draw_label(annotated, " ".join(label_parts), (x1, y1), colour, (255, 255, 255), self.font_scale)

    # ── Security layer (identity · zones · assets · suspects · audit) ─────────

    def _annotate_security(self, frame: np.ndarray, annotated: np.ndarray,
                           dets: list, names: Dict[int, str], stats: dict):
        h, w = frame.shape[:2]
        self._frame_size = (w, h)
        self._clock.set_frame(self._frame_idx)
        t_sec = self._clock.seconds

        persons = [d for d in dets if d[2] == PERSON_CLASS and d[3] >= 0 and d[1] >= self.person_conf]
        objects = [(names.get(d[2], str(d[2])), d[0]) for d in dets if d[2] != PERSON_CLASS]
        self._tracklets.watch = set(self._suspect_of)
        idents = self._tracklets.update(frame, self._frame_idx,
                                        [(tid, box, conf) for box, conf, _, tid in persons])
        self._idents = idents

        polys = self._zones.pixel_polygons(w, h) if self.zones_enabled else {}
        draw_zones(annotated, polys, self.font_scale)

        if len(self.suspects):
            for box, _, _, tid in persons:
                self._check_suspect(tid, idents.get(tid, Identity()), t_sec)

        person_info = [(tid, box, idents.get(tid, Identity()).status, idents.get(tid, Identity()).label)
                       for box, _, _, tid in persons]
        for ev in self._assets.update(self._frame_idx, t_sec, objects, person_info):
            self._on_asset_event(ev, idents, t_sec)
        self._resolve_pending_displacements(idents, t_sec)
        carrying = {a.handler: a for a in self._assets.assets if a.status == IN_TRANSIT and a.handler is not None}

        people = []
        counts = {STATUS_KNOWN: 0, STATUS_UNKNOWN: 0, STATUS_PENDING: 0}
        for box, _, _, tid in persons:
            ident = idents.get(tid, Identity())
            counts[ident.status] += 1
            zones_in = self._zones.zones_containing(box, w, h) if polys else []
            intrusion = ident.status == STATUS_UNKNOWN and bool(zones_in)
            sid = self._suspect_of.get(tid)
            suspect = self.suspects.suspects.get(sid) if sid else None

            self._log_identity(tid, ident)
            if intrusion:
                for z in zones_in:
                    self._zone_alert(tid, ident, z, t_sec)
            self._history.update(self._frame_idx, self._clock.stamp(), self._camera_id, self._source_name,
                                 (w, h), tid, box, sid or self._log_name(ident), ident.status)

            if suspect is not None:
                colour = COL_ALERT
                asset = carrying.get(tid)
                state = f"CARRYING {asset.class_name.upper()}" if asset else "STATUS: UNARMED"
                text = f"{suspect.label}{' (RE-ID MATCH)' if self._suspect_reid.get(tid) else ''} | {state}"
                if self.show_tid:
                    text += f" | T{tid}"
            else:
                colour = (COL_ALERT if intrusion else COL_KNOWN if ident.status == STATUS_KNOWN
                          else COL_UNKNOWN if ident.status == STATUS_UNKNOWN else COL_PENDING)
                text = self._identity_text(tid, ident)
                if tid in carrying:
                    text += f" | CARRYING {carrying[tid].class_name.upper()}"

            draw_trail(annotated, self._trails, tid, box, colour, self.box_thick, self.show_trail)
            cv2.rectangle(annotated, box[:2], box[2:], colour,
                          self.box_thick + (1 if intrusion or suspect is not None else 0))
            draw_label(annotated, text, box[:2], colour, (255, 255, 255), self.font_scale)

            people.append({
                "tid": tid, "box": box, "status": "SUSPECT" if suspect else ident.status,
                "confidence": ident.confidence,
                "name": (suspect.label if suspect else display_name(ident.label)
                         if ident.status == STATUS_KNOWN
                         else "UNKNOWN" if ident.status == STATUS_UNKNOWN else "Identifying…"),
                "crops": ident.n_crops, "zone": zone_name_pretty(zones_in[0]) if zones_in else "",
                "intrusion": intrusion, "carrying": tid in carrying,
            })

        draw_assets(annotated, self._assets.assets, self._frame_idx, self._suspect_of,
                    self.suspects.suspects, self._asset_state_text, self.box_thick, self.font_scale)
        draw_banners(annotated, self._banners, t_sec)
        stats.update(
            known=counts[STATUS_KNOWN], unknown=counts[STATUS_UNKNOWN],
            pending=counts[STATUS_PENDING], alerts=self._alert_count,
            camera=self._camera_id, time=self._clock.stamp(), people=people,
            suspects_in_view=sorted({self._suspect_of[p["tid"]] for p in people if p["tid"] in self._suspect_of}),
            assets=[(a.name, self._asset_state_text(a)) for a in self._assets.assets
                    if a.last_seen == self._frame_idx or a.status == MISSING],
        )

    # ── Assets ───────────────────────────────────────────────────────────────

    def _who(self, tid: Optional[int], ident: Identity) -> str:
        if tid is None:
            return "unknown person"
        sid = self._suspect_of.get(tid)
        if sid:
            return self.suspects.suspects[sid].label
        if ident.status == STATUS_KNOWN:
            return display_name(ident.label)
        return f"T{tid} (UNKNOWN)" if ident.status == STATUS_UNKNOWN else f"T{tid}"

    def _log_others(self, ev, idents: Dict[int, Identity]) -> str:
        """Theft-log every other person touching the asset; returns ' · also at <asset>: …'."""
        if not ev.others:
            return ""
        names = []
        for oid in ev.others:
            other = idents.get(oid, Identity())
            names.append(self._who(oid, other))
            self._theft_log(oid, other, "ASSET_CONTACT", f"{ev.asset.name} {ev.kind} – also touching it")
        return f" · also at {ev.asset.name}: {', '.join(names)}"

    def _on_asset_event(self, ev, idents: Dict[int, Identity], t_sec: float):
        a, tid = ev.asset, ev.track_id
        ident = idents.get(tid, Identity()) if tid is not None else Identity()
        detail = f"{a.name} Δd={ev.dist:.0f}px"
        others = self._log_others(ev, idents) if ev.kind in ("displacement", "unknown_interaction") else ""
        if ev.kind == "displacement":
            if tid in self._suspect_of or ident.status == STATUS_UNKNOWN:
                self._unauthorized_displacement(a, tid, ident, ev.dist, t_sec, others)
            elif ident.status == STATUS_KNOWN:
                self._authorized_relocation(a, tid, ident, ev.dist)
            else:
                self._pending_disp[tid] = (a, self._frame_idx, ev.dist, others)
                self._emit("asset", f"{a.name} displaced by T{tid} ({ev.dist:.0f}px) – identifying the mover…"
                                    f"{others}")
        elif ev.kind == "abandoned":
            x, y = (int(v) for v in centre(a.box))
            self._emit("asset", f"{a.name} ABANDONED at ({x}, {y}) by {self._who(tid, ident)}")
            self._theft_log(tid, ident, "ASSET_ABANDONED", f"{a.name} at ({x},{y})")
        elif ev.kind == "missing":
            who = self._who(tid, ident)
            if (tid in self._suspect_of or ident.status != STATUS_KNOWN) and tid is not None:
                self._raise_banner(f"CRITICAL: {a.name} MISSING – last with {who}", t_sec)
            else:
                self._emit("asset", f"{a.name} out of view – last with {who}")
            self._theft_log(tid, ident, "ASSET_MISSING", detail)
        elif ev.kind == "returned":
            self._emit("asset", f"{a.name} back on its anchor (SECURED)")
            self._theft_log(None, Identity(), "ASSET_SECURED", a.name)
        elif ev.kind == "unknown_interaction":
            limit = self._assets.stay_limit
            self._raise_banner(f"Security Alert: Unknown Person at {a.class_name.capitalize()} > {limit:.0f}s", t_sec)
            if others:
                self._emit("asset", f"{self._who(tid, ident)} at {a.name}{others}")
            self._audit_log(tid, ident, "None", f"{a.class_name.capitalize()}_Unknown_Interaction")
            self._theft_log(tid, ident, "UNAUTHORIZED_INTERACTION", f"{a.name}{others}")
            if self._st.get("auto_enrol_on_interaction", True) and tid not in self._suspect_of:
                self._enrol_suspect(tid, "unknown_interaction")

    def _authorized_relocation(self, a, tid, ident, dist):
        who = display_name(ident.label)
        self._emit("asset", f"Authorized Relocation of {a.name} by {who} ({dist:.0f}px)")
        self._audit_log(tid, ident, "None", f"{a.class_name.capitalize()}_Authorized_Relocation")
        self._theft_log(tid, ident, "AUTHORIZED_RELOCATION", f"{a.name} Δd={dist:.0f}px")

    def _unauthorized_displacement(self, a, tid, ident, dist, t_sec, note=""):
        self._raise_banner("CRITICAL: Unauthorized Asset Displacement", t_sec)
        self._emit("asset", f"{a.name} moved {dist:.0f}px by {self._who(tid, ident)}{note}")
        self._audit_log(tid, ident, "None", f"{a.class_name.capitalize()}_Displaced")
        self._theft_log(tid, ident, "THEFT_ALERT", f"{a.name} Δd={dist:.0f}px{note}")
        if self._st.get("auto_enrol_unknown_movers", True) and tid not in self._suspect_of:
            self._enrol_suspect(tid, "displacement")

    def _resolve_pending_displacements(self, idents, t_sec):
        timeout = float(self._st.get("pending_timeout_seconds", 6)) * self._clock.fps
        for tid, (a, f0, dist, others) in list(self._pending_disp.items()):
            ident = idents.get(tid, Identity())
            if ident.status == STATUS_KNOWN:
                self._authorized_relocation(a, tid, ident, dist)
            elif ident.status == STATUS_UNKNOWN:
                self._unauthorized_displacement(a, tid, ident, dist, t_sec, others)
            elif self._frame_idx - f0 > timeout:
                self._unauthorized_displacement(a, tid, ident, dist, t_sec, f" (mover not identified in time){others}")
            else:
                continue
            del self._pending_disp[tid]

    @staticmethod
    def _asset_state_text(a) -> str:
        if a.status == IN_TRANSIT:
            return f"WITH T{a.handler}"
        if a.status == ABANDONED:
            x, y = (int(v) for v in centre(a.box))
            return f"ABANDONED AT ({x}, {y})"
        if a.status == MISSING:
            return f"MISSING (last with T{a.handler})" if a.handler is not None else "MISSING"
        return "SECURED"

    def add_asset_anchor(self, box, name: Optional[str] = None) -> str:
        """Register a home anchor drawn in the console (pixel box) for the current camera."""
        with self._engine_lock:
            anchors = self._load_anchors()
            cam = self._cam_key or "webcam"
            mine = anchors.setdefault(cam, {})
            name = name or f"desk_laptop_{len(mine) + 1}"
            mine[name] = [int(v) for v in box]
            self._save_anchors(anchors)
            self._assets._add(tuple(mine[name]), "laptop", "drawn", name=name, seen=-1)
            self._emit("asset", f"Anchor '{name}' registered on {cam} at {mine[name]}")
            return name

    def clear_asset_anchors(self):
        with self._engine_lock:
            anchors = self._load_anchors()
            anchors.pop(self._cam_key or "webcam", None)
            self._save_anchors(anchors)
            self._assets.configure(self._cam_key)
            self._emit("asset", f"Drawn anchors cleared for {self._cam_key or 'webcam'}")

    def _load_anchors(self) -> dict:
        try:
            return json.loads(self._anchors_path.read_text())
        except Exception:
            return {}

    def _save_anchors(self, anchors: dict):
        self._anchors_path.parent.mkdir(parents=True, exist_ok=True)
        self._anchors_path.write_text(json.dumps(anchors, indent=2))

    # ── Suspects ─────────────────────────────────────────────────────────────

    def _enrol_suspect(self, tid: Optional[int], reason: str):
        if tid is None or self._tracklets is None:
            return None
        embs, thumbs = self._tracklets.track_data(tid)
        k = int(self._st.get("crops_to_sample", 8))
        if len(embs) == 0:
            self._emit("info", f"Cannot enrol T{tid} as suspect yet – no sharp crops collected")
            return None
        s = self.suspects.enrol(embs[-k:], thumbs[-k:], created=self._clock.stamp(), camera=self._camera_id,
                                source=self._source_name, frame=self._frame_idx, reason=reason,
                                entry=self._history.backtrack(tid))
        if s is None:
            return None
        self._suspect_of[tid] = s.suspect_id
        self._suspect_reid[tid] = False
        e = s.entry
        origin = f" · entered {e.get('side', '?')} at {e.get('time', '?')}" if e else ""
        self._emit("suspect", f"{s.label} auto-enrolled ({reason}, {len(embs[-k:])} crops){origin}")
        self._theft_log(tid, Identity(status=STATUS_UNKNOWN), "SUSPECT_ENROLLED", f"{s.suspect_id} ({reason})")
        return s

    def flag_suspect(self, tid: int):
        """Operator marks a live track as a suspect."""
        with self._engine_lock:
            if tid in self._suspect_of:
                return self.suspects.suspects.get(self._suspect_of[tid])
            return self._enrol_suspect(tid, "manual")

    def _verify_suspect(self, tid: int, ident: Identity):
        sid = self._suspect_of[tid]
        embs, _ = self._tracklets.track_data(tid)
        if self._suspect_checked.get(tid) == len(embs) and ident.status != STATUS_KNOWN:
            return
        self._suspect_checked[tid] = len(embs)
        recent = embs[-3:]
        sim = float(l2n(l2n(recent).mean(axis=0)) @ self.suspects.prototypes[sid]) if len(recent) else 1.0
        employee = ident.status == STATUS_KNOWN and ident.confidence >= self._tracklets.p_thresh + 0.2
        if not employee and len(recent) >= 3 and len(self._tracklets.gallery):
            feats, win = tracklet_features(recent, self._tracklets.gallery)
            if self._tracklets.classifier.p_known(feats) >= self._tracklets.p_thresh + 0.2:
                employee = True
                ident = Identity(status=STATUS_KNOWN, label=self._tracklets.gallery.labels[win])
        weak = sim < float(self._st.get("verify_threshold", 0.6))
        self._suspect_weak[tid] = self._suspect_weak.get(tid, 0) + 1 if weak else 0
        if employee or self._suspect_weak[tid] >= 2:
            label = self.suspects.suspects[sid].label
            del self._suspect_of[tid]
            self._suspect_reid.pop(tid, None)
            self._suspect_weak.pop(tid, None)
            self._suspect_cleared.add(tid)
            why = f"now recognised as {display_name(ident.label)}" if employee else f"appearance changed (sim {sim:.2f})"
            self._emit("suspect", f"{label} label withdrawn from T{tid} – {why} (tracker ID switch)")
            self._theft_log(tid, ident, "SUSPECT_LABEL_WITHDRAWN", f"{sid}: {why}")

    def _check_suspect(self, tid: int, ident: Identity, t_sec: float):
        if self._tracklets is None:
            return
        if tid in self._suspect_of:
            self._verify_suspect(tid, ident)
            return
        if tid in self._suspect_cleared:
            return
        if ident.status == STATUS_KNOWN and ident.confidence >= self._tracklets.p_thresh:
            return
        embs, _ = self._tracklets.track_data(tid)
        if len(embs) < self._tracklets.min_crops or self._suspect_checked.get(tid) == len(embs):
            return
        self._suspect_checked[tid] = len(embs)
        sid, sim = self.suspects.match(embs)
        if sid is None:
            return
        s = self.suspects.suspects[sid]
        self._suspect_of[tid] = sid
        self._suspect_reid[tid] = True
        self._raise_banner(f"{s.label} RE-IDENTIFIED on {self._camera_id} (sim {sim:.2f})", t_sec)
        self._theft_log(tid, ident, "SUSPECT_REID", f"{sid} sim={sim:.2f}")
        self.suspects.add_sighting(sid, time=self._clock.stamp(), camera=self._camera_id,
                                   source=self._source_name, frame=self._frame_idx, track=tid,
                                   sim=round(sim, 3))

    def _theft_log(self, tid, ident: Identity, status: str, detail: str):
        if self._theft is None:
            owner = self._shared or self
            if owner._theft is None:
                owner._theft = AuditLogger(owner._theft_path)
            self._theft = owner._theft
        who = self._suspect_of.get(tid) if tid is not None else None
        self._theft.log(self._clock.stamp(), self._camera_id, tid if tid is not None else -1,
                        who or self._log_name(ident), status, ident.confidence, "None", detail)

    def track_embeddings(self, tid: int):
        """Tracklet embeddings + thumbnails of a live track (for video-wide search)."""
        if self._tracklets is None:
            return np.zeros((0, 512), np.float32), []
        return self._tracklets.track_data(tid)

    def embed_image(self, bgr: np.ndarray) -> np.ndarray:
        """Embed a person photo; if it is a full frame, the largest detected person is used."""
        crop = bgr
        if self._detector.is_ready and bgr.shape[0] > 300:
            with self._engine_lock:
                r = self._detector.predict(bgr, classes=[PERSON_CLASS], conf=0.4, imgsz=self.imgsz)
            if r is not None and r.boxes is not None and len(r.boxes):
                b = max(r.boxes.xyxy.cpu().numpy(), key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
                x1, y1, x2, y2 = (int(v) for v in b)
                crop = bgr[max(0, y1):y2, max(0, x1):x2]
        return self._reid_extractor.extract_batch([crop])

    # ── Identity, Zone & Event helpers ───────────────────────────────────────

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

    def _audit_logger(self) -> AuditLogger:
        if self._audit is None:
            owner = self._shared or self
            if owner._audit is None:
                owner._audit = AuditLogger(owner._audit_path)
            self._audit = owner._audit
        return self._audit

    def _audit_log(self, tid: int, ident: Identity, zone: str = "None", asset: str = "None"):
        self._audit_logger().log(self._clock.stamp(), self._camera_id, tid, self._log_name(ident),
                                 ident.status, ident.confidence, zone, asset)

    def log_operator_action(self, action: str, detail: str):
        """Operator actions (e.g. ALERT_ACKNOWLEDGED) go into the same forensic audit CSV."""
        with self._engine_lock:
            self._audit_logger().log(self._clock.stamp(), self._camera_id or "—", -1, "OPERATOR",
                                     action, 0.0, "None", detail)

    def _log_identity(self, tid: int, ident: Identity):
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
        self._event_no += 1
        self._events.append((self._event_no, level, self._clock.stamp(), self._camera_id, text))

    def events_since(self, last_no: int) -> list:
        return [e for e in list(self._events) if e[0] > last_no]

    def _raise_banner(self, text: str, t_sec: float, duration: float = 4.0):
        self._emit("alert", text)
        self._alert_count += 1
        for i, (old, _) in enumerate(self._banners):
            if old == text:
                del self._banners[i]
                break
        self._banners.append((text, t_sec + duration))
        logger.warning("ALERT [%s %s] %s", self._camera_id, self._clock.stamp(), text)

    def reset_trails(self):
        with self._engine_lock:
            self._trails.clear()

    @staticmethod
    def load_config(path: str = "config.yaml") -> dict:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
