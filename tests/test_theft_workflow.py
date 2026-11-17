"""
Scripted-scenario tests for the theft workflow and shared core geometry / detector helpers.

Run:  python -m unittest discover -s tests -v
"""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.detector import get_available_trackers  # noqa: E402
from core.geometry import area, centre, iou, overlap_ratio, sharpness  # noqa: E402
from core.reid import REID_MODELS  # noqa: E402
from core.security import ABANDONED, IN_TRANSIT, MISSING, SECURED, AssetMonitor  # noqa: E402
from core.suspects import SuspectGallery, TrackHistory, entry_side  # noqa: E402

LAPTOP = (540, 280, 680, 410)          # home anchor (the document's example box)


def shift(box, dx, dy=0):
    return (box[0] + dx, box[1] + dy, box[2] + dx, box[3] + dy)


def monitor(**over):
    cfg = {"asset_monitoring": {"enabled": True, "target_asset_classes": [63], "displacement_threshold_px": 35,
                                "interaction_iou_thresh": 0.25, "confirm_frames": 5, "anchor_frames": 15,
                                "abandon_frames": 45, "missing_frames": 90, "unknown_stay_limit_seconds": 5,
                                "monitored_zones": {"camera_4": {"desk_laptop_1": list(LAPTOP)}}, **over}}
    m = AssetMonitor(cfg)
    m.configure("camera_4")
    return m


def run(m, frames, fps=30.0):
    """frames: list of (laptop_box or None, [(tid, person_box, status, label)]). Returns all events."""
    events = []
    for f, (lap, persons) in enumerate(frames):
        objs = [("laptop", lap)] if lap is not None else []
        events += m.update(f, f / fps, objs, persons)
    return events


class CoreHelpersTests(unittest.TestCase):
    def test_geometry_helpers(self):
        a = (0, 0, 100, 100)
        b = (50, 50, 150, 150)
        self.assertEqual(centre(a), (50.0, 50.0))
        self.assertEqual(area(a), 10000.0)
        self.assertAlmostEqual(iou(a, b), 2500.0 / 17500.0, places=5)
        self.assertAlmostEqual(overlap_ratio(a, b), 0.25, places=5)
        blank = np.zeros((128, 64, 3), dtype=np.uint8)
        self.assertEqual(sharpness(blank), 0.0)

    def test_trackers_and_reid_models(self):
        trackers = get_available_trackers()
        self.assertIn("bytetrack", trackers)
        self.assertIn("botsort", trackers)
        self.assertNotIn("strongsort", trackers)
        self.assertIn("osnet_ain_x1_0", REID_MODELS)
        self.assertNotIn("fastraid", REID_MODELS)


class AssetDisplacementTests(unittest.TestCase):
    def test_configured_anchor_registered(self):
        m = monitor()
        self.assertEqual(len(m.assets), 1)
        self.assertEqual(m.assets[0].name, "desk_laptop_1")
        self.assertEqual(m.assets[0].status, SECURED)

    def test_static_laptop_with_person_is_not_displacement(self):
        m = monitor()
        person = (500, 200, 700, 600)
        ev = run(m, [(LAPTOP, [(1, person, "KNOWN", "emp_04")])] * 60)
        self.assertEqual([e.kind for e in ev], [])
        self.assertEqual(m.assets[0].status, SECURED)

    def test_occlusion_shrinking_box_is_not_displacement(self):
        """A seated person covering half the laptop shifts its box centre, but the box area
        changes a lot → must not be taken as movement."""
        m = monitor()
        half = (LAPTOP[0], LAPTOP[1], LAPTOP[0] + 40, LAPTOP[1] + 40)
        person = (500, 200, 700, 600)
        ev = run(m, [(LAPTOP, [])] * 10 + [(half, [(1, person, "UNKNOWN", None)])] * 60)
        self.assertNotIn("displacement", [e.kind for e in ev])

    def test_unknown_moves_laptop_then_abandons(self):
        m = monitor()
        frames = [(LAPTOP, [])] * 10
        for k in range(30):
            lap = shift(LAPTOP, 4 * k)
            frames.append((lap, [(7, (lap[0] - 40, 150, lap[2] + 40, 650), "UNKNOWN", None)]))
        dropped = shift(LAPTOP, 120)
        frames += [(dropped, [])] * 60
        ev = run(m, frames)
        kinds = [e.kind for e in ev]
        self.assertIn("displacement", kinds)
        disp = next(e for e in ev if e.kind == "displacement")
        self.assertEqual(disp.track_id, 7)
        self.assertGreater(disp.dist, 35)
        self.assertIn("abandoned", kinds)
        self.assertEqual(m.assets[0].status, ABANDONED)

    def test_known_employee_relocation_reports_handler(self):
        m = monitor()
        frames = [(LAPTOP, [])] * 10
        for k in range(20):
            lap = shift(LAPTOP, 0, 5 * k)
            frames.append((lap, [(3, (lap[0] - 30, lap[1] - 200, lap[2] + 30, lap[3] + 150), "KNOWN", "emp_04")]))
        ev = run(m, frames)
        disp = [e for e in ev if e.kind == "displacement"]
        self.assertEqual(len(disp), 1)
        self.assertEqual(disp[0].track_id, 3)
        self.assertEqual(m.assets[0].status, IN_TRANSIT)
        self.assertEqual(m.assets[0].handler, 3)

    def test_laptop_taken_out_of_view_becomes_missing(self):
        m = monitor()
        frames = [(LAPTOP, [])] * 10
        for k in range(15):
            lap = shift(LAPTOP, 6 * k)
            frames.append((lap, [(9, (lap[0] - 40, 150, lap[2] + 40, 650), "UNKNOWN", None)]))
        frames += [(None, [])] * 100
        ev = run(m, frames)
        missing = [e for e in ev if e.kind == "missing"]
        self.assertEqual(len(missing), 1)
        self.assertEqual(missing[0].track_id, 9)
        self.assertEqual(m.assets[0].status, MISSING)

    def test_returned_to_anchor_is_secured_again(self):
        m = monitor()
        frames = [(LAPTOP, [])] * 10
        for k in range(15):
            lap = shift(LAPTOP, 5 * k)
            frames.append((lap, [(2, (lap[0] - 40, 150, lap[2] + 40, 650), "UNKNOWN", None)]))
        frames += [(LAPTOP, [])] * 10
        ev = run(m, frames)
        self.assertIn("returned", [e.kind for e in ev])
        self.assertEqual(m.assets[0].status, SECURED)

    def test_unknown_lingering_alert_survives_short_detection_gaps(self):
        m = monitor()
        person = (500, 200, 700, 600)
        frames = []
        for f in range(240):
            frames.append((LAPTOP, [] if f % 10 == 0 else [(4, person, "UNKNOWN", None)]))
        ev = run(m, frames)
        self.assertEqual([e.kind for e in ev].count("unknown_interaction"), 1)

    def test_unknown_mover_is_blamed_even_when_employee_covers_more(self):
        """An employee seated behind the desk covers the whole laptop; a visitor covers less of it
        but is not staff → the visitor is the mover and the employee is reported alongside."""
        m = monitor()
        frames = [(LAPTOP, [])] * 10
        for k in range(20):
            lap = shift(LAPTOP, 5 * k)
            emp = (lap[0] - 60, lap[1] - 200, lap[2] + 60, lap[3] + 150)          # covers all of it
            visitor = (lap[0] + 60, lap[1] - 150, lap[2] + 120, lap[3] + 150)     # covers ~57 %
            frames.append((lap, [(3, emp, "KNOWN", "emp_04"), (8, visitor, "UNKNOWN", None)]))
        disp = [e for e in run(m, frames) if e.kind == "displacement"]
        self.assertEqual(len(disp), 1)
        self.assertEqual(disp[0].track_id, 8)
        self.assertEqual(disp[0].others, [3])
        self.assertEqual(m.assets[0].handler, 8)

    def test_unknown_lingering_alert_fires_next_to_an_employee(self):
        m = monitor()
        emp, visitor = (480, 150, 740, 650), (600, 250, 760, 600)
        ev = run(m, [(LAPTOP, [(3, emp, "KNOWN", "emp_04"), (8, visitor, "UNKNOWN", None)])] * 200)
        alerts = [e for e in ev if e.kind == "unknown_interaction"]
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0].track_id, 8)
        self.assertEqual(alerts[0].others, [3])

    def test_auto_registration_of_still_laptop(self):
        cfg_m = monitor(monitored_zones={})
        cfg_m.configure("camera_1")
        box = (100, 100, 200, 180)
        run(cfg_m, [(box, [])] * 20)
        self.assertEqual(len(cfg_m.assets), 1)
        self.assertEqual(cfg_m.assets[0].origin, "auto")


class SuspectGalleryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        rng = np.random.default_rng(0)
        self.person_a = rng.normal(size=512)
        self.person_b = rng.normal(size=512)
        self.noise = lambda base: base + 0.3 * rng.normal(size=512)

    def tearDown(self):
        self.tmp.cleanup()

    def test_enrol_persist_and_match(self):
        g = SuspectGallery(self.tmp.name, match_thresh=0.75)
        thumbs = [np.zeros((128, 64, 3), np.uint8)] * 8
        s = g.enrol(np.stack([self.noise(self.person_a) for _ in range(8)]), thumbs, created="12:00:00",
                    camera="Cam_04", source="clip.mp4", frame=10, reason="displacement")
        self.assertEqual(s.suspect_id, "SUSPECT_01")
        self.assertAlmostEqual(float(np.linalg.norm(g.prototypes["SUSPECT_01"])), 1.0, places=5)
        g2 = SuspectGallery(self.tmp.name, match_thresh=0.75)
        sid, sim = g2.match(np.stack([self.noise(self.person_a) for _ in range(5)]))
        self.assertEqual(sid, "SUSPECT_01")
        self.assertGreater(sim, 0.75)
        sid_b, _ = g2.match(np.stack([self.noise(self.person_b) for _ in range(5)]))
        self.assertIsNone(sid_b)
        self.assertEqual(len(g2.crops("SUSPECT_01")), 8)

    def test_entry_side_backtrack(self):
        self.assertEqual(entry_side((0, 100, 80, 400), 1080, 720), "left edge")
        self.assertTrue(entry_side((500, 200, 600, 500), 1080, 720).startswith("inside view"))
        h = TrackHistory(Path(self.tmp.name) / "hist.csv", interval=30)
        h.update(5, "12:00:05", "Cam_02", "clip.mp4", (1080, 720), 3, (1000, 100, 1080, 600), "UNKNOWN", "UNKNOWN")
        h.update(6, "12:00:06", "Cam_02", "clip.mp4", (1080, 720), 3, (950, 100, 1040, 600), "UNKNOWN", "UNKNOWN")
        b = h.backtrack(3)
        self.assertEqual(b["time"], "12:00:05")
        self.assertEqual(b["side"], "right edge")
        h.close()


if __name__ == "__main__":
    unittest.main()
