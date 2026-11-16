# YOLO Vision — Detection, Tracking & Re-ID

A modular Python application for real-time person/object detection using
YOLOv11 (PyTorch fp32/fp16) and OpenVINO INT8 quantised models, with
ByteTrack/BoTSORT tracking and Re-ID via a built-in EMA gallery.

---

## Project Structure

```
├── app.py                   ← Security console (Tkinter): live monitor, pipeline, gallery,
│                              evaluation, audit log, live log
├── web_app.py               ← Web dashboard (Flask): upload a video → people gallery
├── web_pipeline.py          ← Dashboard back end: job queue, track → person grouping, known people
├── web/                     ← Dashboard page (index.html + static/app.js, app.css; no build step)
├── detection_engine.py      ← Detection + tracking + identity + zones/assets + overlays
├── open_set.py              ← Enrolled gallery, 5 tracklet features, open-set classifier
├── security.py              ← Restricted zones, asset anchors / displacement, CSV audit logger
├── suspects.py              ← Auto-enrolled suspect gallery, track history, forensic backtrack
├── video_search.py          ← Video-wide person search: index recordings + ranked search
├── console_tabs.py          ← Console tabs: Suspects, Video Search
├── ui_theme.py              ← Console colours / fonts / widgets
├── export_yolo_int8.py      ← YOLO INT8 re-quantisation at 480 px (CHIRLA-calibrated)
├── tests/                   ← Theft-workflow + web-dashboard tests (python -m unittest discover -s tests)
├── chirla.py                ← CHIRLA dataset helpers (crops, dates, cameras)
├── export_reid_openvino.py  ← Pipeline 1: Re-ID model → OpenVINO
├── build_gallery.py         ← Pipeline 2: daily employee enrollment
├── train_open_set.py        ← Pipeline 3: train + evaluate the open-set classifier
├── make_demo_clips.py       ← Pipeline 4: pick + export the 3 demo clips
├── render_demo.py           ← Offline smooth render of a demo (for presenting)
├── config.yaml              ← All tunable parameters (edit freely)
├── requirements.txt
├── enrolled_gallery/        ← <date>/emp_XX/crop*.jpg + gallery_prototypes.pkl, open_set_lr.json
└── models/
    ├── fp16/                ← yolo11n/s fp16 .pt weights
    ├── fp32/                ← yolo11n/s fp32 .pt + OpenVINO FP32 .xml
    ├── openvino/            ← yolo11n/s INT8 .xml + .bin
    └── reid/                ← OSNet-AIN Re-ID weights (.pth) + OpenVINO IR
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install --break-system-packages -r requirements.txt
```

For OSNet-based Re-ID (better accuracy):
```bash
pip install --break-system-packages torchreid
```

On Kali and other PEP 668-managed systems, `--break-system-packages` is
required when using the global Python installation.

### 2. Launch the GUI
```bash
python app.py
```

### 3. Workflow
1. **Select a model** from the drop-down (e.g. `yolo11s_int8` for fastest
   inference on CPU, `yolo11s_fp32` for best accuracy).
2. Click **⬡ Load Model** and wait for the status bar to confirm.
3. Select a **source** — leave as `Webcam (device 0)` or **📂 Browse** a
   video file.
4. Adjust **Confidence** / **IoU** sliders and toggle **Show Trail** / **Re-ID**.
5. Click **▶ START**.

---

## Web Dashboard (upload a video → people gallery)

A simplified front end for non-technical users. They upload a recording; the
full pipeline (detection → tracking → open-set identity → asset / suspect
checks) runs on it and they get:

- the **annotated video** (H.264 MP4, seekable in any browser);
- a **people gallery** – one card per person, with their best photos, how long
  they were on screen, and *Recognised / Unrecognised / Watch list / Not enough
  view*. Tracks of the same person (tracker ID switches, leaving and coming
  back) are merged by Re-ID centroid similarity, never when they are on screen
  together;
- an **activity timeline** (appearances, bag/laptop alerts) – click to seek;
- **Remember this person**: naming an unrecognised person adds them to a
  dashboard gallery that is merged into the enrolled gallery for every later
  upload, so they come back as *Recognised*.

```bash
python web_app.py          # opens http://127.0.0.1:8000 (models load in the background)
```

Settings are fixed to the recommended ones (`config.yaml → web`): there are no
model, tracker or threshold controls. Everything the dashboard writes – uploads,
results, its own audit logs, known people and watch list – stays under
`output/web/`, so the console's logs and suspect gallery are never touched.
Videos are analysed one at a time, at `web.analysis_fps` (15) frames per second
of video – about 1.5× the video's length on a 4-core CPU.

---

## Open-Set Security Demo (CHIRLA)

Known **employees** are enrolled into a persistent gallery; everyone else is an
**UNKNOWN visitor**. Each person track keeps a buffer of its last 10 sharp crops,
and 5 tracklet features feed a logistic-regression P(Known):

| # | Feature | Meaning |
|---|---|---|
| 1 | `s_best` | mean cosine similarity to the winning identity |
| 2 | `margin` | top-1 minus top-2 identity similarity |
| 3 | `agreement` | fraction of crops voting for the winner |
| 4 | `consistency` | mean pairwise similarity inside the tracklet |
| 5 | `prototype_fit` | cosine(tracklet centroid, winner prototype) |

P(Known) ≥ 0.5 → green **Emp XX**, else orange **UNKNOWN**. Unknowns inside a
restricted zone raise an intrusion alert; a person overlapping a laptop /
backpack / handbag that moves (or an unknown lingering > 5 s) raises an asset
alert. Every decision and alert is appended to `output/audit_log.csv`.

### Same-day enrollment protocol
CHIRLA spans 7 recording days (Jun–Dec 2023) and people change clothes between
days, so appearance Re-ID cannot match across days (closed-set accuracy 14%
cross-day vs 55–83% same-day). Employees are therefore enrolled **per day** from
their earliest footage ("morning entrance"); the app picks the gallery matching
a CHIRLA video's date automatically.

- Employees: 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 14 · Visitors: 8, 13, 15, 16, 18, 19, 24, 25, 26
- Classifier trained on Jun days (visitors 8, 18, 19, 24), tested on Jul/Dec days
  (visitors 13, 15, 16, 25 – never seen in training)

### Results (held-out days, `output/open_set_report.md`)

| Method | Employee → correct ID ↑ | Employee → wrong ID ↓ | Visitor accepted ↓ | Visitor rejected ↑ |
|---|---|---|---|---|
| Naive single-frame (cos ≥ 0.45) | 62% | 37% | 99% | 1% |
| **Tracklet + logistic regression** | **67%** | **6%** | **27%** | **73%** |

ROC-AUC 0.88. The visitor test set is small (52 tracklets / 4 people).
Known limitation: seated or heavily occluded employees are sometimes rejected
as UNKNOWN (enrollment crops are mostly of people walking).

### Run the pipeline
CHIRLA lives at `chirla.root` in `config.yaml` (default `C:/CHIRLA`).
```bash
python -m gdown 1SigwBE6mPdqiJMqhuIY4aqC7--5CsMal -O models/reid/osnet_ain_x1_0_msmt17.pth
python export_reid_openvino.py   # OSNet-AIN → OpenVINO (~7× faster on CPU)
python build_gallery.py          # daily enrollment
python train_open_set.py         # fit + evaluate the open-set classifier
python make_demo_clips.py        # export the 3 demo clips to output/demo/
python render_demo.py all        # optional: smooth annotated renders in output/rendered/
```
The same steps can be run from the console's **Pipeline** tab.

### Present the demos
```bash
python app.py                                   # pick a demo clip in the sidebar
python app.py --demo demo2 --mode naive --start # open straight into a demo
```
- **Demo 1 – Known persistence:** employees keep a stable green identity.
- **Demo 2 – Unknown rejection:** run in **Naive** (the visitor gets an
  employee's name), then **Tracklet** (the visitor becomes UNKNOWN).
- **Demo 3 – Asset interaction:** an unknown visitor at a laptop → laptop
  ALERT, on-screen banner and audit-log entry.

Live processing runs at ~7 FPS on a dual-core i5-6200U with `yolo11n_int8_480` (crowded
scene; was 2 FPS) and **Real-time playback** keeps videos at true speed; **▶ Play smooth render**
plays the offline-rendered version.

---

## Theft Workflow: Asset Displacement & Suspect Tracking

Implements the FYP-2 "Spatiotemporal Asset Interaction & Automated Suspect Gallery
Enrolment" design (`security.py`, `suspects.py`, `detection_engine.py`).

1. **Static asset anchor** – each laptop / backpack / handbag gets a HOME box: from
   `asset_monitoring.monitored_zones` in `config.yaml`, drawn in the console
   (**✏ Draw laptop anchor**, saved to `output/asset_anchors.json`), or auto-registered
   once a detected asset has been still for `anchor_frames`.
2. **Displacement trigger** – `Δd = ‖centre(box) − centre(home)‖ > 35 px` for 5 frames in a
   row, with a similar box size (so a person covering the laptop is not mistaken for
   movement), while a person overlaps it (`Area(person ∩ asset) / Area(asset) > 0.25`).
3. **Open-set decision** on the mover's tracklet: KNOWN → *"Authorized Relocation by
   Emp_XX"*; UNKNOWN → *"CRITICAL: Unauthorized Asset Displacement"*; still identifying →
   decided when the identity resolves (or treated as UNKNOWN after 6 s).
4. **Auto-enrolment** – the mover's 8 tracklet crops become `SUSPECT_xx`
   (`P_suspect = normalise(1/K Σ eᵢ)`), saved in `output/suspects/`. An unknown person
   lingering at an asset (> 5 s) or a person flagged by the operator (**🚩 Flag suspect**)
   is enrolled the same way.
5. **Two-phase tracking** – while carrying: `SUSPECT 01 | CARRYING LAPTOP` (red) and
   `LAPTOP: WITH SUSPECT 01` (cyan); afterwards `LAPTOP: ABANDONED AT (x, y)` / `MISSING`,
   and the person `SUSPECT 01 | STATUS: UNARMED`. If the track is lost, new tracks are
   matched against suspect prototypes → `SUSPECT 01 (RE-ID MATCH)`, also in other cameras
   (the suspect gallery persists across sources). Suspect labels are continuously
   re-verified and withdrawn if the tracker swaps IDs onto someone else.
6. **Forensic backtrack** – `output/track_history.csv` (per track per second) and each
   suspect's entry point: when and from which image edge they entered the view.
7. **Theft audit log** – `output/theft_audit_log.csv`: THEFT_ALERT, AUTHORIZED_RELOCATION,
   ASSET_ABANDONED / MISSING / SECURED, SUSPECT_ENROLLED, SUSPECT_REID, label withdrawals.

Hand-over between people (asset custody chain) is planned future work.

**Verification.** CHIRLA contains no staged theft, so the displacement state machine is
covered by scripted-trajectory tests: `python -m unittest discover -s tests -v`
(authorised relocation, theft + abandonment, missing, returned, occlusion is not movement,
lingering alert across detection gaps, auto-anchoring, suspect gallery, backtrack).
On Demo 3 the lingering visitor is auto-enrolled and re-identified after his track is
lost; 14 of 15 suspect-labelled one-second samples match CHIRLA ground truth (the one
miss is the instant of a tracker ID swap, after which the label is withdrawn).

## Video-wide Search (find a person in recorded CCTV)

`video_search.py` + console tab **🔎 Video Search**.

- **Index** (once per recording day, ~15 min/day on an i5-6200U):
  `python video_search.py index --day 2023-06-08` – every video is sampled at 2 fps, people
  are tracked (ByteTrack) and stored as tracklets with an OSNet centroid embedding,
  thumbnail, time span and P(Known). Output: `output/index/<date>/*.pkl`.
- **Search** (instant): query = a suspect, an enrolled employee, a person selected in the
  Live Monitor (**🔎 Search person**) or a photo. Results are ranked best-first, one per
  track, with camera / time and **▶ Play from here**. Tracklets recognised as enrolled
  employees can be excluded – the right default when hunting an unknown person.
- **Why tracklets:** on CHIRLA single crops separate same / different people poorly
  (cosine 0.66 vs 0.56 on average); tracklet centroids and ranking work much better.
- **Results:** SUSPECT_01 (visitor 8, enrolled on camera 4) → 4 results on cameras 4, 5
  and 6, all visitor 8 (with employees excluded). Ground-truth retrieval check
  (`python video_search.py eval`, days 2023-06-08 + 2023-07-26, 4 visitor queries,
  written to `output/search_eval.json` and shown in the Evaluation tab):

  | Setting | Top-1 correct | Correct person in top 5 | mAP |
  |---|---|---|---|
  | Plain ranking | 75% | 75% | 24% |
  | Employees excluded (default) | 75% | 100% | 18% |

  The search reliably surfaces the person among the first results, but does not
  retrieve every appearance (low mAP); excluding employees also drops some visitor
  tracklets that are misjudged as employees. Only 4 queries – indicative, not conclusive.
  Same-day only: people change clothes between CHIRLA days.

Launch straight into a search: `python app.py --search SUSPECT_01`.

---

## Model Guide

| Key                     | Format       | Precision | Speed  | Accuracy |
|-------------------------|--------------|-----------|--------|----------|
| `yolo11n_fp32`          | PyTorch .pt  | FP32      | Fast   | Good     |
| `yolo11s_fp32`          | PyTorch .pt  | FP32      | Medium | Better   |
| `yolo11n_fp16`          | PyTorch .pt  | FP16*     | Fast   | Good     |
| `yolo11s_fp16`          | PyTorch .pt  | FP16*     | Medium | Better   |
| `yolo11n_openvino_fp32` | OV .xml      | FP32      | Fast   | Good     |
| `yolo11s_openvino_fp32` | OV .xml      | FP32      | Medium | Better   |
| `yolo11n_int8`          | OV .xml INT8 | INT8      | Fastest| Good     |
| `yolo11s_int8`          | OV .xml INT8 | INT8      | Fast   | Better   |

\* FP16 falls back to FP32 on CPU-only machines.

---

## config.yaml — Key Parameters

```yaml
detection:
  conf_threshold: 0.35   # raise to reduce false positives
  iou_threshold:  0.45   # lower for dense scenes
  target_classes: [0]    # 0 = person; [] = all COCO classes
  imgsz: 640             # 320 = faster, 1280 = more accurate

tracking:
  tracker: bytetrack      # bytetrack | botsort
  track_buffer: 30        # frames to keep a lost track alive

reid:
  enabled: true
  reid_thresh: 0.4        # lower = stricter identity matching
  feature_update_rate: 0.9  # EMA weight for feature update

openvino:
  device: CPU             # CPU | GPU | AUTO
  num_streams: 2          # parallel inference streams
```

---

## Re-ID Notes

Without `torchreid` installed the engine falls back to a simple
64-dimensional HSV colour-histogram feature. This still gives useful
identity persistence but degrades with appearance changes.

Install `torchreid` for production-quality OSNet embeddings:
```bash
pip install torchreid
```

---

## Saving Output Video

Set `output.save_video: true` in `config.yaml` and specify `output_dir`.
Saved files are timestamped MP4s written by OpenCV.
