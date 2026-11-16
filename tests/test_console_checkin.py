"""
Desktop console check-in tests: turning a check-in recording into person cards
(collect_people) with a stub engine, cancellation, settings restored, and the cards
feeding build_staff / write_day_gallery. No models are loaded.
Run:  python -m unittest discover -s tests -v
"""
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from console.checkin import CheckinCancelled, collect_people  # noqa: E402
from core.enrollment import build_staff, write_day_gallery  # noqa: E402
from core.geometry import l2n  # noqa: E402
from core.open_set import load_day_gallery  # noqa: E402

MODEL = "osnet_ain_x1_0"


def write_video(path: Path, frames: int, fps: float = 10.0):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (320, 240))
    rng = np.random.default_rng(0)
    for _ in range(frames):
        w.write(rng.integers(0, 255, (240, 320, 3), dtype=np.uint8))   # texture → sharp crops
    w.release()


class StubEngine:
    """Two people walking through every frame; each has a stable appearance embedding."""

    def __init__(self):
        self._assets = SimpleNamespace(enabled=True)
        self.zones_enabled = True
        self._identity_mode = "legacy"
        self._reid_enabled = False
        self.modes, self.sources = [], []
        self.n = 0
        self.base = {1: l2n(np.random.default_rng(1).normal(size=512)),
                     2: l2n(np.random.default_rng(2).normal(size=512))}

    def set_identity_mode(self, mode):
        self.modes.append(mode)
        self._identity_mode = mode

    def set_source(self, source, fps=30.0, start_frame=0, gallery_date=None):
        self.sources.append((Path(source).name, round(fps, 1), gallery_date))
        assert not self._assets.enabled and not self.zones_enabled and self._identity_mode == "tracklet"

    def process_frame(self, frame):
        self.n += 1
        x = 10 + self.n
        people = [{"tid": 1, "box": (x, 20, x + 50, 200), "status": "UNKNOWN", "name": "UNKNOWN", "confidence": 0.9},
                  {"tid": 2, "box": (200, 20, 250, 200), "status": "KNOWN", "name": "Emp 04", "confidence": 0.95}]
        return frame, {"people": people}

    def track_embeddings(self, tid):
        rng = np.random.default_rng(self.n * 10 + tid)
        e = l2n(self.base[tid] + 0.02 * rng.normal(size=512))     # per-view cosine to base ≈ 0.9
        return np.stack([e]).astype(np.float32), []


class CollectPeopleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)
        self.video = self.d / "checkin.avi"
        write_video(self.video, 60)
        self.cfg = {"web": {"analysis_fps": 10, "min_person_seconds": 1.0, "thumbs_per_person": 4}}

    def tearDown(self):
        self.tmp.cleanup()

    def test_people_cards_and_settings_restored(self):
        eng = StubEngine()
        updates = []
        people, embs = collect_people(eng, [(str(self.video), "Cam_01"), (str(self.d / "nope.avi"), "Cam_02")],
                                      "2024-02-01", self.d / "work", self.cfg,
                                      progress=lambda f, t: updates.append((f, t)))
        self.assertEqual(len(people), 2)
        by_status = {p["status"]: p for p in people}
        self.assertEqual(by_status["KNOWN"]["name"], "Emp 04")
        self.assertTrue(by_status["UNKNOWN"]["name"].startswith("Visitor"))
        for p in people:
            self.assertEqual(p["cam"], "Cam_01")
            self.assertTrue(p["photos"] and all(f.exists() for f in p["photos"]))
            self.assertGreaterEqual(len(embs[p["id"]]), 5)
        self.assertEqual(eng.sources[0][2], "2024-02-01")
        self.assertTrue(any("could not open" in t for _, t in updates))
        self.assertEqual(updates[-1][0], 1.0)
        self.assertEqual((eng._assets.enabled, eng.zones_enabled, eng._identity_mode, eng._reid_enabled),
                         (True, True, "legacy", False))

        # cards → gallery
        gfile = self.d / "enrolled" / "gallery_prototypes.pkl"
        gfile.parent.mkdir()
        cards = {p["id"]: p for p in people}
        staff = build_staff(cards, embs, [{"person": by_status["KNOWN"]["id"], "label": "emp_04"}], k=5, min_samples=3)
        s = write_day_gallery(gfile, "2024-02-01", staff, model=MODEL)
        self.assertEqual(s["added"], ["emp_04"])
        g, problem = load_day_gallery(gfile, "2024-02-01", MODEL)
        self.assertIsNone(problem)
        self.assertGreater(float(g.prototypes[0] @ eng.base[2]), 0.9)

    def test_cancel_restores_settings(self):
        eng, stop = StubEngine(), threading.Event()
        stop.set()
        with self.assertRaises(CheckinCancelled):
            collect_people(eng, [(str(self.video), "Cam_01")], "2024-02-01", self.d / "w2", self.cfg, stop=stop)
        self.assertEqual((eng._assets.enabled, eng._identity_mode), (True, "legacy"))


if __name__ == "__main__":
    unittest.main()
