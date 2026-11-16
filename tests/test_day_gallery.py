"""
Per-day staff gallery tests: loading the gallery enrolled on a chosen recording date,
graceful fallbacks (missing / empty / unreadable / wrong model → no staff), the
engine's gallery choice, and the dashboard's required recording date.
Run:  python -m unittest discover -s tests -v
"""
import pickle
import sys
import tempfile
import unittest
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.engine import DetectionEngine  # noqa: E402
from core.geometry import l2n  # noqa: E402
from core.open_set import (NO_STAFF, STATUS_PENDING, STATUS_UNKNOWN, OpenSetClassifier,  # noqa: E402
                           PrototypeGallery, TrackletManager, _Track, is_iso_date,
                           list_day_galleries, load_day_gallery)
from webapp.pipeline import JobManager, KnownPeople, check_video_date, date_in_name  # noqa: E402

MODEL = "osnet_ain_x1_0"


def vec(seed: int) -> np.ndarray:
    return l2n(np.random.default_rng(seed).normal(size=512))


def write_gallery(path: Path, labels, date, model=MODEL):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"prototypes": {k: vec(i) for i, k in enumerate(labels)},
                     "meta": {"model": model, "date": date}}, f)
    for k in labels:
        (path.parent / k).mkdir(exist_ok=True)
        (path.parent / k / "crop1.jpg").write_bytes(b"jpg")


class GalleryFolder(unittest.TestCase):
    """enrolled/gallery_prototypes.pkl (default = 07-26) + enrolled/<date>/... per day."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name) / "enrolled"
        self.default = root / "gallery_prototypes.pkl"
        write_gallery(root / "2023-06-02" / "gallery_prototypes.pkl", ["emp_01", "emp_02"], "2023-06-02")
        write_gallery(root / "2023-07-26" / "gallery_prototypes.pkl", ["emp_04"], "2023-07-26")
        write_gallery(root / "2023-06-08" / "gallery_prototypes.pkl", ["emp_05"], "2023-06-08", model="resnet50")
        write_gallery(root / "2023-06-13" / "gallery_prototypes.pkl", [], "2023-06-13")
        (root / "2023-12-01").mkdir()
        (root / "2023-12-01" / "gallery_prototypes.pkl").write_bytes(b"not a pickle")
        with open(root / "2023-07-26" / "gallery_prototypes.pkl", "rb") as f:
            self.default.write_bytes(f.read())

    def tearDown(self):
        self.tmp.cleanup()


class LoadDayGalleryTests(GalleryFolder):
    def test_loads_the_chosen_day(self):
        g, problem = load_day_gallery(self.default, "2023-06-02", MODEL)
        self.assertIsNone(problem)
        self.assertEqual((g.labels, g.meta["date"]), (["emp_01", "emp_02"], "2023-06-02"))

    def test_unusable_days_explain_why(self):
        cases = {"2023-05-01": "No staff are enrolled", "2023-06-08": "different Re-ID model",
                 "2023-06-13": "no enrolled employees", "2023-12-01": "could not be read",
                 "../2023-06-02": "not a valid date", "2023-02-30": "not a valid date"}
        for date, why in cases.items():
            g, problem = load_day_gallery(self.default, date, MODEL)
            self.assertIsNone(g, date)
            self.assertIn(why, problem, date)

    def test_listing(self):
        days = {d["date"]: d for d in list_day_galleries(self.default, MODEL)}
        self.assertEqual(sorted(days), ["2023-06-02", "2023-06-08", "2023-06-13", "2023-07-26", "2023-12-01"])
        self.assertEqual((days["2023-06-02"]["staff"], days["2023-06-02"]["usable"]), (2, True))
        self.assertFalse(days["2023-06-08"]["usable"])
        self.assertFalse(days["2023-12-01"]["usable"])

    def test_iso_date(self):
        self.assertTrue(is_iso_date("2023-06-02"))
        for bad in ("2023-6-2", "2023-13-01", "x", None, "2023-06-02/..", " 2023-06-02"):
            self.assertFalse(is_iso_date(bad), bad)


class EmptyGalleryTests(unittest.TestCase):
    def test_empty_merges_with_named_people(self):
        named = PrototypeGallery(["Ali"], np.stack([vec(9)]))
        merged = PrototypeGallery.empty({"date": "2023-05-01"}).merged(named)
        self.assertEqual((merged.labels, merged.meta["date"]), (["Ali"], "2023-05-01"))
        self.assertEqual(len(PrototypeGallery.empty().merged(PrototypeGallery.empty())), 0)

    def test_everyone_unknown_without_staff(self):
        tm = TrackletManager(PrototypeGallery.empty(), OpenSetClassifier(None), lambda crops: None,
                             {"tracking": {"min_tracklet_crops": 3}})
        t = _Track(embs=deque([vec(1), vec(1)], maxlen=10))
        tm._decide(t)
        self.assertEqual(t.identity.status, STATUS_PENDING)
        t.embs.append(vec(1))
        tm._decide(t)
        self.assertEqual((t.identity.status, t.identity.label), (STATUS_UNKNOWN, None))


class EngineGalleryChoiceTests(GalleryFolder):
    """DetectionEngine._gallery_for on a stub (no models loaded)."""

    def choose(self, source, gallery_date, extra=None):
        stub = SimpleNamespace(_reid_cfg={"gallery_file": str(self.default)}, _reid_model_name=MODEL,
                               extra_gallery=extra)
        g = DetectionEngine._gallery_for(stub, source, gallery_date)
        return g, stub.gallery_report

    def test_selected_date_wins_over_filename(self):
        g, rep = self.choose("camera_2_2023-07-26-11_14_26.mp4", "2023-06-02")
        self.assertEqual(g.labels, ["emp_01", "emp_02"])
        self.assertEqual((rep["date"], rep["how"], rep["staff"], rep["warning"]), ("2023-06-02", "selected", 2, None))

    def test_unusable_date_means_no_staff_not_another_day(self):
        named = PrototypeGallery(["Ali"], np.stack([vec(9)]))
        for date in ("2023-05-01", "2023-06-08", "2023-06-13", "2023-12-01"):
            g, rep = self.choose("clip.mp4", date, extra=named)
            self.assertEqual(g.labels, ["Ali"], date)
            self.assertEqual((rep["staff"], rep["named"], rep["date"]), (0, 1, date))
            self.assertTrue(rep["warning"], date)

    def test_no_staff(self):
        g, rep = self.choose("clip.mp4", NO_STAFF)
        self.assertEqual((len(g), rep["how"], rep["warning"]), (0, "none", None))

    def test_console_behaviour_unchanged(self):
        g, rep = self.choose("camera_2_2023-06-02-11_14_26.mp4", None)
        self.assertEqual((g.labels, rep["how"]), (["emp_01", "emp_02"], "filename"))
        g, rep = self.choose("webcam.mp4", None)
        self.assertEqual((g.labels, rep["how"]), (["emp_04"], "default"))


class RecordingDateTests(unittest.TestCase):
    def test_date_in_name(self):
        self.assertEqual(date_in_name("camera_2_2023-06-02-11:14:26.avi"), "2023-06-02")
        self.assertEqual(date_in_name("camera_2_2023-06-02-11\uf03a14\uf03a26.avi"), "2023-06-02")
        self.assertEqual(date_in_name("camera_2_2023-06-02-111426.avi"), "2023-06-02")
        self.assertEqual(date_in_name("office 2023-02-30 then 2023-03-01.mp4"), "2023-03-01")
        self.assertIsNone(date_in_name("clip.mp4"))
        self.assertIsNone(date_in_name("12023-06-021.mp4"))

    def test_check_video_date(self):
        self.assertEqual(check_video_date(" 2023-06-02 "), "2023-06-02")
        for bad in (None, "", "2023-02-30", "02-06-2023", "2999-01-01"):
            with self.assertRaises(ValueError, msg=bad):
                check_video_date(bad)

    def test_old_jobs_never_borrow_another_day(self):
        pick = JobManager._gallery_date
        self.assertEqual(pick({"video_date": "2023-06-08", "video": "camera_2_2023-06-02-1.avi"}), "2023-06-08")
        self.assertEqual(pick({"video": "camera_2_2023-06-02-111426.avi"}), "2023-06-02")
        self.assertEqual(pick({"video": "clip.mp4"}), NO_STAFF)


class JobDateTests(GalleryFolder):
    def setUp(self):
        super().setUp()
        self.patch = mock.patch.object(JobManager, "_worker", lambda self: None)
        self.patch.start()
        cfg = {"web": {"data_dir": str(Path(self.tmp.name) / "web")},
               "reid": {"model": MODEL, "gallery_file": str(self.default)}}
        self.jobs = JobManager(cfg)

    def tearDown(self):
        self.patch.stop()
        super().tearDown()

    def test_create_stores_date_and_rejects_missing(self):
        job = self.jobs.create("clip.mp4", lambda p: p.write_bytes(b"v"), video_date="2023-06-02",
                               filename_date="2023-07-26")
        self.assertEqual((job["video_date"], job["filename_date"], job["gallery"]), ("2023-06-02", "2023-07-26", None))
        with self.assertRaises(ValueError):
            self.jobs.create("clip.mp4", lambda p: p.write_bytes(b"v"))
        self.assertEqual(len(self.jobs.list()), 1)
        self.assertEqual(len(list(self.jobs.jobs_dir.iterdir())), 1)

    def test_galleries_listing_refreshes_on_change(self):
        self.assertIn("2023-06-02", [d["date"] for d in self.jobs.galleries()])
        write_gallery(self.default.parent / "2024-01-09" / "gallery_prototypes.pkl", ["emp_07"], "2024-01-09")
        self.assertIn("2024-01-09", [d["date"] for d in self.jobs.galleries()])

    def test_staff_by_day(self):
        known = KnownPeople(Path(self.tmp.name) / "known", MODEL, self.default)
        self.assertEqual(known.listing()["staff_day"], "2023-07-26")
        self.assertEqual([s["label"] for s in known.listing()["staff"]], ["emp_04"])
        listing = known.listing("2023-06-02")
        self.assertEqual(([s["label"] for s in listing["staff"]], listing["staff_day"]),
                         (["emp_01", "emp_02"], "2023-06-02"))
        self.assertEqual(listing["staff"][0]["photos"], ["crop1.jpg"])
        self.assertEqual(known.staff_dir("emp_01", "2023-06-02"), self.default.parent / "2023-06-02" / "emp_01")
        self.assertIsNone(known.staff_dir("..", "2023-06-02"))
        self.assertIsNone(known.staff_dir("emp_01", "../x"))
        self.assertEqual(known.listing("2023-05-01")["staff"], [])


if __name__ == "__main__":
    unittest.main()
