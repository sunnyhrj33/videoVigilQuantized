"""
Recorded-day multi-camera playback tests: finding days / sequences / cameras, time
alignment, the analysis-rate setting, the camera grid, and the multi-camera worker
on tiny generated videos with stub engines (no models loaded).
Run:  python -m unittest discover -s tests -v
"""
import queue
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from console.mosaic import compose, grid_shape, layout, tile_at  # noqa: E402
from console.worker import MultiCameraWorker, analysis_rate, frame_step  # noqa: E402
from core.recordings import CameraVideo, scan_days, sync_offsets  # noqa: E402

COLON = "\uf03a"        # how Windows stores CHIRLA's ':' in file names


def write_video(path: Path, frames: int, fps: float = 10.0, size=(64, 48)):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
    for i in range(frames):
        img = np.zeros((size[1], size[0], 3), np.uint8)
        img[:] = (i * 7) % 255          # frame number encoded in the brightness
        w.write(img)
    w.release()


def cam_name(n: int, date: str, hh: int, mm: int, ss: int) -> str:
    return f"camera_{n}_{date}-{hh:02d}{COLON}{mm:02d}{COLON}{ss:02d}.avi"


class ScanTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = self.root = Path(self.tmp.name)
        a = root / "2023-06-02" / "seq_000"
        a.mkdir(parents=True)
        for n in (1, 2, 3, 4, 6, 7):                       # camera_5 missing
            (a / cam_name(n, "2023-06-02", 11, 14, 25)).write_bytes(b"x")
        (a / cam_name(2, "2023-06-02", 11, 20, 0)).write_bytes(b"x")    # duplicate camera_2
        (a / "notes.txt").write_text("not a video")
        b = root / "2023-06-02" / "seq_001"
        b.mkdir()
        for n in range(1, 8):
            (b / f"camera_{n}_2023-06-02-12_03_33.mp4").write_bytes(b"x")
        (root / "2023-06-02" / "empty_seq").mkdir()
        (root / "not-a-date" / "seq_9").mkdir(parents=True)
        (root / "not-a-date" / "seq_9" / cam_name(1, "2023-06-02", 1, 2, 3)).write_bytes(b"x")

    def tearDown(self):
        self.tmp.cleanup()

    def test_days_sequences_and_missing_cameras(self):
        days = scan_days(self.root, expected=7)
        self.assertEqual(list(days), ["2023-06-02"])
        seq0, seq1 = days["2023-06-02"]
        self.assertEqual((seq0.name, seq1.name), ("seq_000", "seq_001"))
        self.assertEqual([c.camera for c in seq0.cameras], [f"camera_{n}" for n in (1, 2, 3, 4, 6, 7)])
        self.assertEqual(seq0.missing, ["camera_5"])
        self.assertEqual(len(seq0.duplicates), 1)
        self.assertEqual(seq0.cameras[0].cam_id, "Cam_01")
        self.assertEqual(seq0.start, datetime(2023, 6, 2, 11, 14, 25))
        self.assertEqual((len(seq1.cameras), seq1.missing), (7, []))

    def test_missing_root(self):
        self.assertEqual(scan_days(self.root / "nope"), {})


class SyncTests(unittest.TestCase):
    def test_offsets_align_to_latest_start(self):
        cams = [CameraVideo("camera_1", "Cam_01", Path("a"), datetime(2023, 6, 2, 11, 14, 25)),
                CameraVideo("camera_2", "Cam_02", Path("b"), datetime(2023, 6, 2, 11, 14, 27)),
                CameraVideo("camera_3", "Cam_03", Path("c"), None)]
        off = sync_offsets(cams, {"camera_1": 30.0, "camera_2": 25.0, "camera_3": 30.0})
        self.assertEqual(off, {"camera_1": 60, "camera_2": 0, "camera_3": 0})


class RateTests(unittest.TestCase):
    def test_analysis_rate(self):
        cfg = {"multi_camera_fps": 5}
        self.assertIsNone(analysis_rate("auto", 1, cfg))
        self.assertEqual(analysis_rate("auto", 7, cfg), 5.0)
        self.assertIsNone(analysis_rate("all", 7, cfg))
        self.assertEqual(analysis_rate("10", 1, cfg), 10.0)
        self.assertEqual(analysis_rate("3 fps", 7, cfg), 3.0)
        self.assertIsNone(analysis_rate("nonsense", 7, cfg))
        self.assertIsNone(analysis_rate(0, 7, cfg))

    def test_frame_step(self):
        self.assertEqual(frame_step(30, None), 1)
        self.assertEqual(frame_step(30, 10), 3)
        self.assertEqual(frame_step(30, 60), 1)
        self.assertEqual(frame_step(25, 3), 8)


class MosaicTests(unittest.TestCase):
    def test_seven_cameras_fit_and_do_not_overlap(self):
        rects = layout(7, 1200, 800, 1.5)
        self.assertEqual(len(rects), 7)
        for x, y, w, h in rects:
            self.assertTrue(0 <= x and x + w <= 1200 and 0 <= y and y + h <= 800)
        for i, a in enumerate(rects):
            for b in rects[i + 1:]:
                self.assertTrue(a[0] + a[2] <= b[0] or b[0] + b[2] <= a[0] or
                                a[1] + a[3] <= b[1] or b[1] + b[3] <= a[1])
        self.assertEqual(grid_shape(7, 1200, 800, 1.5), (3, 3))
        self.assertEqual(grid_shape(7, 2000, 600, 1.5), (4, 2))

    def test_click_maps_to_tile_and_compose_size(self):
        frames = [np.full((720, 1080, 3), i * 30, np.uint8) for i in range(7)]
        frames[3] = None
        img, rects = compose(frames, [f"Cam_0{i + 1}" for i in range(7)], 900, 600, [False] * 6 + [True])
        self.assertEqual(img.shape, (600, 900, 3))
        x, y, w, h = rects[4]
        self.assertEqual(tile_at(rects, x + w // 2, y + h // 2), 4)
        self.assertIsNone(tile_at(rects, 0, 0))


class StubEngine:
    def __init__(self):
        self.source = self.start = None
        self.read = []
        self.skipped = 0
        self._frame = 0

    def set_source(self, source, fps, start_frame=0, gallery_date=None):
        self.source, self.fps, self.start, self.gallery_date = source, fps, start_frame, gallery_date
        self._frame = start_frame

    def skip_frame(self):
        self.skipped += 1

    def process_frame(self, frame):
        self.read.append(int(frame[0, 0, 0]))
        return frame, {"camera": Path(self.source).name[:8], "people": [], "fps": 5.0, "known": 1}


class MultiCameraWorkerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = Path(self.tmp.name)
        self.cams = []
        for n, ss, frames in ((1, 25, 20), (2, 26, 12)):          # camera_2 starts 1 s later
            p = d / cam_name(n, "2023-06-02", 11, 14, ss)
            write_video(p, frames)
            self.cams.append(CameraVideo(f"camera_{n}", f"Cam_0{n}", p, datetime(2023, 6, 2, 11, 14, ss)))
        self.cams.append(CameraVideo("camera_3", "Cam_03", d / "missing.avi", None))

    def tearDown(self):
        self.tmp.cleanup()

    def test_plays_available_cameras_in_sync(self):
        q = queue.Queue()
        engines = {}
        worker = MultiCameraWorker(self.cams, lambda i: engines.setdefault(i, StubEngine()), q,
                                   "2023-06-02", analysis_fps=5.0, realtime=False)
        worker.start()
        msgs = []
        while not msgs or msgs[-1][0] not in ("done", "error"):    # consume like the UI does
            msgs.append(q.get(timeout=10))
        worker.join(timeout=5)
        kinds = [k for k, _ in msgs]
        self.assertIn(("warning"), kinds)
        self.assertTrue(any("Cam_03" in p for k, p in msgs if k == "warning"))
        self.assertEqual(kinds[-1], "done")
        payloads = [p for k, p in msgs if k == "multi"]
        self.assertTrue(payloads)
        self.assertEqual(payloads[0]["labels"], ["Cam_01", "Cam_02"])
        self.assertEqual(payloads[-1]["ended"], [True, True])
        e1, e2 = engines[0], engines[1]
        self.assertEqual((e1.start, e2.start), (10, 0))            # 1 s × 10 fps head start removed
        self.assertEqual((e1.gallery_date, e2.gallery_date), ("2023-06-02", "2023-06-02"))
        # 5 fps analysed out of 10 fps: every 2nd frame, both cameras on the same moment
        frame_no = lambda v: round(v / 7)                         # noqa: E731 (MJPG is lossy)
        self.assertEqual([frame_no(v) for v in e1.read[:3]], [11, 13, 15])
        self.assertEqual([frame_no(v) for v in e2.read[:3]], [1, 3, 5])
        self.assertGreater(e1.skipped, 0)

    def test_nothing_opens(self):
        q = queue.Queue()
        MultiCameraWorker(self.cams[2:], lambda i: StubEngine(), q, None, None, realtime=False).run()
        kinds = [q.get()[0] for _ in range(q.qsize())]
        self.assertEqual(kinds[-1], "error")


if __name__ == "__main__":
    unittest.main()
