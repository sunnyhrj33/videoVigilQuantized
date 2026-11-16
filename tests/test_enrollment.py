"""
Morning check-in enrollment tests: the daily prototype, staff labels, writing a day's
gallery (merge / replace / backup / model mismatch / failure leaves the old one), and the
dashboard flow on a fake finished job (no models loaded).
Run:  python -m unittest discover -s tests -v
"""
import json
import pickle
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.enrollment import (EnrollmentError, check_label, daily_prototype,  # noqa: E402
                             write_day_gallery)
from core.geometry import l2n  # noqa: E402
from core.open_set import load_day_gallery  # noqa: E402
from webapp.pipeline import JobManager  # noqa: E402

try:
    import flask  # noqa: F401
    HAS_FLASK = True
except ImportError:
    HAS_FLASK = False

MODEL = "osnet_ain_x1_0"
DATE = "2023-06-02"


def vec(seed, noise=0.0, rng=np.random.default_rng(1)):
    return l2n(np.random.default_rng(seed).normal(size=512) + noise * rng.normal(size=512))


def views(seed, n, noise=0.3):
    return np.stack([vec(seed, noise) for _ in range(n)]).astype(np.float32)


class PrototypeTests(unittest.TestCase):
    def test_k_spread_samples_and_outlier_rejected(self):
        embs = np.vstack([views(1, 9), vec(99)[None]])          # one view of somebody else
        proto, chosen, outliers = daily_prototype(embs, k=5, min_samples=3)
        self.assertEqual(len(chosen), 5)
        self.assertNotIn(9, chosen)
        self.assertGreaterEqual(outliers, 1)
        self.assertAlmostEqual(float(np.linalg.norm(proto)), 1.0, places=5)
        self.assertGreater(float(proto @ vec(1)), 0.9)

    def test_too_few_views(self):
        with self.assertRaises(EnrollmentError):
            daily_prototype(views(1, 2), k=5, min_samples=3)
        with self.assertRaises(EnrollmentError):
            daily_prototype(np.zeros((0, 512)), k=5, min_samples=3)

    def test_labels(self):
        self.assertEqual(check_label("  Sara   Khan "), "Sara Khan")
        self.assertEqual(check_label("emp_04"), "emp_04")
        for bad in ("", "x", "..", "../x", "a/b", "con", " .hidden", "Sara.", "x" * 41):
            with self.assertRaises(EnrollmentError, msg=bad):
                check_label(bad)


class WriteGalleryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "enrolled"
        self.gfile = self.root / "gallery_prototypes.pkl"
        day = self.root / DATE
        for label in ("emp_01", "emp_02"):
            (day / label).mkdir(parents=True)
            (day / label / "crop1.jpg").write_bytes(b"old")
        with open(day / "gallery_prototypes.pkl", "wb") as f:
            pickle.dump({"prototypes": {"emp_01": vec(1), "emp_02": vec(2)},
                         "meta": {"model": MODEL, "date": DATE, "enroll_cutoff": {"emp_01": ["s", 1], "emp_02": ["s", 2]}}}, f)
        self.photo = Path(self.tmp.name) / "p.jpg"
        self.photo.write_bytes(b"new")

    def tearDown(self):
        self.tmp.cleanup()

    def entry(self, label, seed):
        return {"label": label, "prototype": vec(seed), "photos": [self.photo, self.photo], "samples": 5, "outliers": 0}

    def test_merge_keeps_others_replaces_same_label_and_backs_up(self):
        s = write_day_gallery(self.gfile, DATE, [self.entry("emp_02", 22), self.entry("Sara Khan", 3)], model=MODEL)
        self.assertEqual((s["added"], s["replaced"], s["kept"]), (["Sara Khan"], ["emp_02"], ["emp_01"]))
        g, problem = load_day_gallery(self.gfile, DATE, MODEL)
        self.assertIsNone(problem)
        self.assertEqual(sorted(g.labels), ["Sara Khan", "emp_01", "emp_02"])
        self.assertGreater(float(g.prototypes[g.labels.index("emp_02")] @ vec(22)), 0.99)
        meta = g.meta
        self.assertEqual(meta["folders"]["Sara Khan"], "Sara_Khan")
        self.assertEqual(meta["enroll_cutoff"], {"emp_01": ["s", 1]})
        self.assertEqual((self.root / DATE / "emp_01" / "crop1.jpg").read_bytes(), b"old")
        self.assertEqual((self.root / DATE / "emp_02" / "crop2.jpg").read_bytes(), b"new")
        backup = Path(s["backup"])
        self.assertTrue((backup / "gallery_prototypes.pkl").exists())
        self.assertFalse(any(p.name.startswith(".") for p in self.root.iterdir()))

    def test_replace_drops_others(self):
        s = write_day_gallery(self.gfile, DATE, [self.entry("emp_05", 5)], model=MODEL, mode="replace")
        self.assertEqual((s["staff"], s["removed"]), (1, ["emp_01", "emp_02"]))
        self.assertFalse((self.root / DATE / "emp_01").exists())

    def test_other_model_can_only_be_replaced(self):
        with self.assertRaises(EnrollmentError):
            write_day_gallery(self.gfile, DATE, [self.entry("emp_05", 5)], model="resnet50")
        s = write_day_gallery(self.gfile, DATE, [self.entry("emp_05", 5)], model="resnet50", mode="replace")
        self.assertEqual(s["staff"], 1)

    def test_new_day_and_failure_leaves_old_gallery(self):
        s = write_day_gallery(self.gfile, "2023-06-08", [self.entry("emp_05", 5)], model=MODEL)
        self.assertIsNone(s["backup"])
        bad = self.entry("emp_09", 9)
        bad["photos"] = [Path(self.tmp.name) / "missing.jpg"]
        with self.assertRaises(OSError):
            write_day_gallery(self.gfile, DATE, [bad], model=MODEL)
        g, _ = load_day_gallery(self.gfile, DATE, MODEL)
        self.assertEqual(sorted(g.labels), ["emp_01", "emp_02"])
        self.assertFalse(any(p.name.startswith(".") for p in self.root.iterdir()))
        with mock.patch("pathlib.Path.rename", side_effect=OSError("in use")):
            with self.assertRaises(OSError):
                write_day_gallery(self.gfile, DATE, [self.entry("emp_09", 9)], model=MODEL)
        g, _ = load_day_gallery(self.gfile, DATE, MODEL)
        self.assertEqual(sorted(g.labels), ["emp_01", "emp_02"])


class FakeJob:
    """A finished dashboard job on disk: job.json, people.npz, thumbs/."""

    @staticmethod
    def make(jobs_dir: Path, job_id: str, people, status="done", video_date=DATE):
        d = jobs_dir / job_id
        (d / "thumbs").mkdir(parents=True)
        embs, plist = {}, []
        for pid, st, name, e in people:
            photos = []
            for k in range(1, 4):
                (d / "thumbs" / f"{pid}_{k}.jpg").write_bytes(b"jpg")
                photos.append(f"thumbs/{pid}_{k}.jpg")
            if e is not None:
                embs[pid] = e
            plist.append({"id": pid, "status": st, "name": name, "photos": photos})
        np.savez(d / "people.npz", **embs)
        job = {"id": job_id, "name": "checkin.mp4", "video": "checkin.mp4", "created": "2023-06-02T08:00:00",
               "status": status, "people": plist, "events": [], "video_date": video_date}
        (d / "job.json").write_text(json.dumps(job))


class DashboardEnrollTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.gfile = root / "enrolled" / "gallery_prototypes.pkl"
        self.gfile.parent.mkdir(parents=True)
        self.patch = mock.patch.object(JobManager, "_worker", lambda self: None)
        self.patch.start()
        self.cfg = {"web": {"data_dir": str(root / "web")}, "reid": {"model": MODEL, "gallery_file": str(self.gfile)}}
        jobs_dir = root / "web" / "jobs"
        FakeJob.make(jobs_dir, "J1", [("P01", "UNKNOWN", "Visitor 1", views(1, 8)),
                                      ("P02", "UNKNOWN", "Visitor 2", views(1, 6)),     # same person, split
                                      ("P03", "UNKNOWN", "Visitor 3", views(3, 8)),
                                      ("P04", "UNKNOWN", "Visitor 4", views(4, 2)),     # too few views
                                      ("P05", "SUSPECT", "Suspect 01", views(5, 8))])
        FakeJob.make(jobs_dir, "J2", [("P01", "UNKNOWN", "Visitor 1", views(1, 8))], status="running")
        self.jobs = JobManager(self.cfg)

    def tearDown(self):
        self.patch.stop()
        self.tmp.cleanup()

    def test_preview(self):
        pv = self.jobs.enroll_preview("J1")
        self.assertEqual((pv["date"], pv["existing"]), (DATE, None))
        problems = {p["id"]: p["problem"] for p in pv["people"]}
        self.assertIsNone(problems["P01"])
        self.assertIn("2 clear views", problems["P04"])
        self.assertIn("watch list", problems["P05"])
        with self.assertRaises(RuntimeError):
            self.jobs.enroll_preview("J2")
        with self.assertRaises(KeyError):
            self.jobs.enroll_preview("nope")

    def test_build_merge_split_cards_and_mark_job(self):
        s = self.jobs.enroll("J1", [{"person": "P01", "label": "emp_01"}, {"person": "P02", "label": "emp_01"},
                                    {"person": "P03", "label": "Sara Khan"}, {"person": "P04", "label": ""}])
        self.assertEqual(sorted(s["added"]), ["Sara Khan", "emp_01"])
        self.assertEqual(s["people"]["emp_01"], ["P01", "P02"])
        g, problem = load_day_gallery(self.gfile, DATE, MODEL)
        self.assertIsNone(problem)
        self.assertGreater(float(g.prototypes[g.labels.index("emp_01")] @ vec(1)), 0.9)
        self.assertEqual(self.jobs.get("J1")["enrolled"]["labels"], ["Sara Khan", "emp_01"])
        self.assertIn(DATE, [d["date"] for d in self.jobs.galleries() if d["usable"]])
        staff = self.jobs.known.listing(DATE)["staff"]
        self.assertEqual(sorted(p["label"] for p in staff), ["Sara Khan", "emp_01"])
        self.assertEqual(self.jobs.known.staff_dir("Sara Khan", DATE), self.gfile.parent / DATE / "Sara_Khan")
        self.assertEqual(self.jobs.enroll_preview("J1")["existing"]["staff"], ["Sara Khan", "emp_01"])

    def test_all_problems_reported_and_nothing_written(self):
        self.jobs.known.add("Ali Raza", views(7, 3), [], "x.mp4")
        with self.assertRaises(EnrollmentError) as cm:
            self.jobs.enroll("J1", [{"person": "P01", "label": "emp_01"}, {"person": "P03", "label": "emp_01"},
                                    {"person": "P04", "label": "emp_04"}, {"person": "P05", "label": "emp_05"},
                                    {"person": "P02", "label": "Ali Raza"}, {"person": "P99", "label": "emp_09"},
                                    {"person": "P01", "label": "x"}])
        msg = str(cm.exception)
        for part in ("don't look like the same person", "2 clear views", "watch list", "Ali Raza", "P99", "twice"):
            self.assertIn(part, msg)
        self.assertFalse((self.gfile.parent / DATE).exists())
        with self.assertRaises(EnrollmentError):
            self.jobs.enroll("J1", [{"person": "P01", "label": ""}])

    def test_switched_off(self):
        self.cfg["web"]["staff_enrollment"] = False
        jobs = JobManager(self.cfg)
        with self.assertRaises(EnrollmentError):
            jobs.enroll_preview("J1")


@unittest.skipUnless(HAS_FLASK, "flask is not installed in this Python environment")
class EnrollApiTests(DashboardEnrollTests):
    def setUp(self):
        super().setUp()
        from webapp.server import create_app
        self.client = create_app(self.cfg).test_client()

    def test_api(self):
        self.assertEqual(self.client.get("/api/jobs/J1/enroll").status_code, 200)
        self.assertEqual(self.client.get("/api/jobs/J2/enroll").status_code, 409)
        self.assertEqual(self.client.get("/api/jobs/nope/enroll").status_code, 404)
        r = self.client.post("/api/jobs/J1/enroll", json={"assignments": [{"person": "P04", "label": "emp_04"}]})
        self.assertEqual(r.status_code, 400)
        self.assertIn("Nothing was saved", r.json["error"])
        r = self.client.post("/api/jobs/J1/enroll", json={"assignments": [{"person": "P03", "label": "Sara Khan"}],
                                                         "mode": "merge"})
        self.assertEqual(r.status_code, 200, r.json)
        self.assertEqual(self.client.get(f"/media/staff/Sara%20Khan/crop1.jpg?day={DATE}").status_code, 200)
        self.assertEqual(self.client.get(f"/media/staff/Sara_Khan/crop1.jpg?day={DATE}").status_code, 404)
        r = self.client.post("/api/jobs/J1/enroll", json={"assignments": "nope"})
        self.assertEqual(r.status_code, 400)


if __name__ == "__main__":
    unittest.main()
