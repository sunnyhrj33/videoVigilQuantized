"""
Web dashboard tests: grouping tracks into people, track verdicts, the known-people
store and the HTTP API (with the analysis worker stubbed out – no models loaded).

Run:  python -m unittest discover -s tests -v
"""

import io
import pickle
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.geometry import l2n  # noqa: E402
from core.open_set import PrototypeGallery  # noqa: E402
from webapp.pipeline import (STATUS_SUSPECT, STATUS_UNCLEAR, JobManager,  # noqa: E402
                             KnownPeople, TrackLog, best_spread, group_tracks,
                             group_verdict, segments)

try:
    import flask  # noqa: F401
    HAS_FLASK = True
except ImportError:
    HAS_FLASK = False

RNG = np.random.default_rng(0)


def vec(seed: int, noise: float = 0.0) -> np.ndarray:
    base = np.random.default_rng(seed).normal(size=512)
    return l2n(base + noise * RNG.normal(size=512))


def track(status, name, centroid, frames):
    return {"status": status, "name": name, "centroid": centroid, "frames": list(frames)}


class GroupTracksTests(unittest.TestCase):
    def test_same_appearance_at_different_times_is_one_person(self):
        tracks = [track("UNKNOWN", "UNKNOWN", vec(1), range(0, 50)),
                  track("UNKNOWN", "UNKNOWN", vec(1, 0.3), range(80, 120))]
        self.assertEqual(group_tracks(tracks, 0.75), [[0, 1]])

    def test_different_appearance_stays_apart(self):
        tracks = [track("UNKNOWN", "UNKNOWN", vec(1), range(0, 50)),
                  track("UNKNOWN", "UNKNOWN", vec(2), range(80, 120))]
        self.assertEqual(group_tracks(tracks, 0.75), [[0], [1]])

    def test_tracks_on_screen_together_are_never_merged(self):
        tracks = [track("UNKNOWN", "UNKNOWN", vec(1), range(0, 50)),
                  track("UNKNOWN", "UNKNOWN", vec(1, 0.1), range(40, 90))]
        self.assertEqual(len(group_tracks(tracks, 0.75)), 2)

    def test_same_known_name_merges_even_without_embeddings(self):
        tracks = [track("KNOWN", "Emp 04", None, range(0, 30)),
                  track("KNOWN", "Emp 04", None, range(60, 90))]
        self.assertEqual(group_tracks(tracks, 0.75), [[0, 1]])

    def test_two_different_names_never_join_through_an_unknown(self):
        tracks = [track("KNOWN", "Emp 04", vec(1), range(0, 30)),
                  track("UNKNOWN", "UNKNOWN", vec(1, 0.05), range(40, 60)),
                  track("KNOWN", "Emp 05", vec(1, 0.05), range(70, 90))]
        groups = group_tracks(tracks, 0.75)
        self.assertEqual(len(groups), 2)
        self.assertFalse(any({0, 2} <= set(g) for g in groups))

    def test_groups_ordered_by_first_appearance(self):
        tracks = [track("UNKNOWN", "UNKNOWN", vec(1), range(50, 60)),
                  track("UNKNOWN", "UNKNOWN", vec(2), range(0, 10))]
        self.assertEqual(group_tracks(tracks, 0.75), [[1], [0]])


class VerdictTests(unittest.TestCase):
    def test_track_majority_and_pending_ignored(self):
        log = TrackLog(1)
        for n in range(10):
            log.observe(n, n / 15, {"status": "PENDING", "name": "Identifying…", "confidence": 0.0})
        for n in range(10, 30):
            log.observe(n, n / 15, {"status": "KNOWN", "name": "Emp 04", "confidence": 0.9})
        for n in range(30, 35):
            log.observe(n, n / 15, {"status": "UNKNOWN", "name": "UNKNOWN", "confidence": 0.3})
        status, name, conf, weight = log.verdict()
        self.assertEqual((status, name, weight), ("KNOWN", "Emp 04", 25))
        self.assertAlmostEqual(conf, 0.9)

    def test_track_never_decided_is_unclear(self):
        log = TrackLog(1)
        log.observe(0, 0.0, {"status": "PENDING", "name": "Identifying…", "confidence": 0.0})
        self.assertEqual(log.verdict()[0], STATUS_UNCLEAR)

    def test_suspect_wins_and_unknown_certainty_is_inverted(self):
        items = [{"status": "UNKNOWN", "name": "UNKNOWN", "conf": 0.2, "weight": 50},
                 {"status": STATUS_SUSPECT, "name": "SUSPECT 01", "conf": 0.0, "weight": 5}]
        self.assertEqual(group_verdict(items)[:2], (STATUS_SUSPECT, "SUSPECT 01"))
        status, _, sure = group_verdict(items[:1])
        self.assertEqual(status, "UNKNOWN")
        self.assertAlmostEqual(sure, 0.8)

    def test_segments_split_on_gaps(self):
        times = [0, 0.5, 1.0, 5.0, 5.5]
        self.assertEqual(segments(times, 0.1), [[0.0, 1.1], [5.0, 5.6]])

    def test_best_spread_prefers_spaced_photos(self):
        photos = [(10, 0.0, "a"), (9, 0.1, "b"), (8, 2.0, "c")]
        self.assertEqual([p[2] for p in best_spread(photos, 2)], ["a", "c"])


class KnownPeopleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        staff = root / "enrolled" / "gallery_prototypes.pkl"
        (root / "enrolled" / "2023-07-26" / "emp_04").mkdir(parents=True)
        (root / "enrolled" / "2023-07-26" / "emp_04" / "crop1.jpg").write_bytes(b"jpg")
        with open(staff, "wb") as f:
            pickle.dump({"prototypes": {"emp_04": vec(4)}, "meta": {"date": "2023-07-26"}}, f)
        self.photo = root / "p.jpg"
        self.photo.write_bytes(b"jpg")
        self.root, self.staff = root, staff
        self.known = KnownPeople(root / "known", "osnet_ain_x1_0", staff)

    def tearDown(self):
        self.tmp.cleanup()

    def test_add_list_reload_remove(self):
        embs = np.stack([vec(9, 0.1) for _ in range(5)])
        self.assertEqual(self.known.add("  Sara   Khan ", embs, [self.photo], "clip.mp4"), "Sara Khan")
        listing = self.known.listing()
        self.assertEqual([p["name"] for p in listing["yours"]], ["Sara Khan"])
        self.assertEqual(listing["yours"][0]["photos"], ["photo1.jpg"])
        self.assertEqual(listing["staff"][0]["name"], "Emp 04")
        self.assertEqual(listing["staff"][0]["photos"], ["crop1.jpg"])

        again = KnownPeople(self.root / "known", "osnet_ain_x1_0", self.staff)
        self.assertEqual(again.gallery().labels, ["Sara Khan"])
        again.remove("Sara Khan")
        self.assertIsNone(again.gallery())
        self.assertFalse(any((self.root / "known").glob("Sara*")))

    def test_name_rules(self):
        embs = np.stack([vec(9)])
        for bad in ("", "x", "Emp 04", "emp_07", "Suspect 2", "a/b", "x" * 41):
            with self.assertRaises(ValueError, msg=bad):
                self.known.add(bad, embs, [], "clip.mp4")
        self.known.add("Ali", embs, [], "clip.mp4")
        with self.assertRaises(ValueError):
            self.known.add("ali", embs, [], "clip.mp4")

    def test_needs_embeddings(self):
        with self.assertRaises(ValueError):
            self.known.add("Ali", np.zeros((0, 512)), [], "clip.mp4")

    def test_merged_gallery_recognises_both(self):
        staff = PrototypeGallery.load(self.staff)
        self.known.add("Ali", np.stack([vec(9)]), [], "clip.mp4")
        merged = staff.merged(self.known.gallery())
        self.assertEqual(merged.labels, ["emp_04", "Ali"])
        sims = merged.similarities(np.stack([vec(9)]))[0]
        self.assertEqual(merged.labels[int(np.argmax(sims))], "Ali")


@unittest.skipUnless(HAS_FLASK, "flask is not installed in this Python environment")
class ApiTests(unittest.TestCase):
    """HTTP API with the analysis worker stubbed out (jobs stay queued, no models load)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        staff = root / "gallery_prototypes.pkl"
        with open(staff, "wb") as f:
            pickle.dump({"prototypes": {"emp_04": vec(4)}, "meta": {"date": "x"}}, f)
        cfg = {"web": {"data_dir": str(root / "web"), "max_upload_mb": 1},
               "reid": {"model": "osnet_ain_x1_0", "gallery_file": str(staff)}}
        self.patch = mock.patch.object(JobManager, "_worker", lambda self: None)
        self.patch.start()
        from webapp.server import create_app
        self.app = create_app(cfg)
        self.client = self.app.test_client()

    def tearDown(self):
        self.patch.stop()
        self.tmp.cleanup()

    def upload(self, name, data=b"fake video"):
        return self.client.post("/api/jobs", data={"video": (io.BytesIO(data), name)},
                                content_type="multipart/form-data")

    def test_page_and_status(self):
        self.assertEqual(self.client.get("/").status_code, 200)
        self.assertEqual(self.client.get("/static/app.js").status_code, 200)
        self.assertEqual(self.client.get("/api/status").json["engine"], "loading")

    def test_upload_list_get_delete(self):
        r = self.upload("my clip.mp4")
        self.assertEqual(r.status_code, 201, r.json)
        job = r.json
        self.assertEqual((job["status"], job["name"]), ("queued", "my_clip.mp4"))
        self.assertEqual([j["id"] for j in self.client.get("/api/jobs").json], [job["id"]])
        self.assertEqual(self.client.get(f"/api/jobs/{job['id']}").json["id"], job["id"])
        self.assertEqual(self.client.get(f"/media/jobs/{job['id']}/my_clip.mp4").data, b"fake video")
        self.assertEqual(self.client.delete(f"/api/jobs/{job['id']}").status_code, 200)
        self.assertEqual(self.client.get(f"/api/jobs/{job['id']}").status_code, 404)

    def test_rejects_non_video_and_oversize(self):
        r = self.upload("notes.txt")
        self.assertEqual(r.status_code, 400)
        self.assertIn("isn't a supported video", r.json["error"])
        r = self.upload("big.mp4", b"0" * (2 * 2**20))
        self.assertEqual(r.status_code, 413)
        self.assertIn("too big", r.json["error"])

    def test_no_path_traversal(self):
        job = self.upload("clip.mp4").json
        self.assertEqual(self.client.get(f"/media/jobs/{job['id']}/../../gallery_prototypes.pkl").status_code, 404)

    def test_known_listing_and_unknown_person(self):
        data = self.client.get("/api/known").json
        self.assertEqual(data["yours"], [])
        self.assertEqual(data["watchlist"], [])
        self.assertEqual([s["name"] for s in data["staff"]], ["Emp 04"])
        job = self.upload("clip.mp4").json
        r = self.client.post(f"/api/jobs/{job['id']}/people/P01/remember", json={"name": "Ali"})
        self.assertEqual(r.status_code, 404)


if __name__ == "__main__":
    unittest.main()
