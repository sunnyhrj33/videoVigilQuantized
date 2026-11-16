# VideoVigil — Open-Set Security Console & Web Dashboard

An offline, CPU-optimised AI security surveillance and asset-theft monitoring system using **YOLOv11** (PyTorch FP32/FP16 & OpenVINO INT8), **ByteTrack / BoTSORT** multi-object tracking, **OSNet** OpenVINO person Re-ID, **5-feature tracklet open-set recognition**, spatial security zones, static laptop/bag displacement monitoring, automated suspect watch-list profiling, and cross-camera CCTV search.

---

## Project Structure

```text
videoVigilQuantized/
├── config.yaml                     ← All tunable parameters (models, thresholds, zones, web)
├── requirements.txt                ← Python dependencies
├── run_console.py                  ← Entry point: Desktop Operator Console (Tkinter)
├── run_web.py                      ← Entry point: Upload-and-Analyse Web Dashboard (Flask)
│
├── core/                           ← Shared AI & Security Brain (no UI code)
│   ├── engine.py                   ← DetectionEngine: per-frame orchestration & security state
│   ├── detector.py                 ← YoloDetector, OpenVINO loader, ByteTrack/BoTSORT, SystemSpec
│   ├── reid.py                     ← ReIDExtractor (OSNet/ResNet via torchreid + OpenVINO) & ReIDGallery
│   ├── open_set.py                 ← PrototypeGallery, 5 tracklet features, OpenSetClassifier, TrackletManager
│   ├── security.py                 ← ZoneMonitor, AssetMonitor (displacement state machine), AuditLogger
│   ├── suspects.py                 ← SuspectGallery (persistent watch list) & TrackHistory (backtrack CSV)
│   ├── search.py                   ← VideoIndexer & cosine tracklet search across indexed CCTV
│   ├── overlay.py                  ← All OpenCV drawing (boxes, trails, zones, asset anchors, banners)
│   ├── geometry.py                 ← Shared iou(), overlap_ratio(), centre(), area(), l2n(), sharpness()
│   └── chirla.py                   ← CHIRLA dataset filename parser, recording dates, and crop indexer
│
├── console/                        ← Desktop Operator Console (Tkinter)
│   ├── app.py                      ← Main window shell, header status chips, left sidebar, log panel
│   ├── worker.py                   ← VideoWorker thread, SubprocessRunner, QueueLogHandler
│   ├── theme.py                    ← Dark theme palette, fonts, ScrollFrame, StatTile, Chip, helpers
│   └── tabs/
│       ├── live_tab.py             ← Live Monitor: video canvas, anchor drawing, live people table, alerts
│       ├── suspects_tab.py         ← Suspect gallery cards, forensic backtrack, sightings, play origin
│       ├── search_tab.py           ← Video-wide CCTV search & day index builder
│       ├── pipeline_tab.py         ← 5-step offline pipeline runner with progress & streamed output
│       ├── gallery_tab.py          ← Enrolled employee crops viewer per recording day
│       ├── eval_tab.py             ← Evaluation charts (Naive vs. Tracklet Open-Set)
│       └── audit_tab.py            ← Forensic audit log viewer (output/audit_log.csv)
│
├── webapp/                         ← Web Dashboard (Flask + Vanilla JS SPA)
│   ├── server.py                   ← Flask HTTP server & REST API routes
│   ├── pipeline.py                 ← JobManager, VideoSink (H.264), TrackLog, group_tracks, KnownPeople
│   ├── index.html                  ← Single-page dashboard shell
│   └── static/                     ← Frontend logic (app.js) and styles (app.css)
│
├── scripts/                        ← Offline CLI Tools
│   ├── export_reid_openvino.py     ← Step 1: Convert OSNet Re-ID checkpoint to OpenVINO IR
│   ├── build_gallery.py            ← Step 2: Daily employee enrollment from morning footage
│   ├── train_open_set.py           ← Step 3: Train + evaluate the 5-feature open-set classifier
│   ├── make_demo_clips.py          ← Step 4: Select + export the 3 evaluation demo clips
│   ├── render_demo.py              ← Offline smooth MP4 renderer for presentations
│   └── export_yolo_int8.py         ← YOLO11 INT8 OpenVINO re-quantisation calibrated on CHIRLA
│
├── tests/                          ← Automated Unit Tests
│   ├── test_theft_workflow.py      ← Geometry, asset displacement state machine, suspect gallery, backtrack
│   └── test_web_dashboard.py       ← Track grouping, verdicts, KnownPeople store, and Flask HTTP API
│
├── enrolled_gallery/               ← Daily employee crops, gallery_prototypes.pkl, open_set_lr.json
└── models/                         ← YOLO11 (fp32, fp16, openvino int8) & OSNet Re-ID weights
```

---

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch the Desktop Operator Console
```bash
python run_console.py
# Or jump directly into a demo clip:
python run_console.py --demo demo2 --mode tracklet --start
```

### 3. Launch the Web Dashboard
```bash
python run_web.py          # opens http://127.0.0.1:8000
```

### 4. Run Offline Pipeline Scripts
```bash
python scripts/export_reid_openvino.py   # Convert OSNet-AIN to OpenVINO IR (~7× faster on CPU)
python scripts/build_gallery.py          # Enrol employees per recording day
python scripts/train_open_set.py         # Fit & evaluate the 5-feature open-set classifier
python scripts/make_demo_clips.py        # Export the 3 evaluation clips to output/demo/
python scripts/render_demo.py all        # Render smooth annotated MP4s to output/rendered/
python -m core.search index --day 2023-06-08   # Index a recording day for CCTV search
```

### 5. Run Tests
```bash
python -m unittest discover -s tests -v
```
