"""
export_yolo_int8.py
────────────────────────────────────────────────────────────────────────────────
Re-export YOLO11 to OpenVINO INT8 at a smaller input size for faster CPU
inference, calibrated on real CHIRLA office frames (not generic COCO images).

  python export_yolo_int8.py --imgsz 480            # → models/openvino/yolo11n_int8_480_openvino_model
  python export_yolo_int8.py --imgsz 416 --model s  # yolo11s at 416

Adds the new model to config.yaml's `models:` section is left to you (the
script prints the line to paste).
────────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

import cv2

from chirla import parse_video_name
from detection_engine import DetectionEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("export_yolo_int8")

COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush",
]


def build_calibration_set(root: Path, out: Path, per_video: int = 3) -> Path:
    """Sample frames from every CHIRLA video into a YOLO-style dataset folder."""
    img_dir = out / "images" / "val"
    if out.exists():
        shutil.rmtree(out)
    img_dir.mkdir(parents=True)
    (out / "labels" / "val").mkdir(parents=True)
    n = 0
    for v in sorted((root / "videos").glob("seq_*/camera_*.avi")):
        cap = cv2.VideoCapture(str(v))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cam, _ = parse_video_name(v.name)
        for k in range(per_video):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * (k + 1) / (per_video + 1)))
            ok, frame = cap.read()
            if ok:
                cv2.imwrite(str(img_dir / f"{v.parent.name}_{cam}_{k}.jpg"), frame)
                n += 1
        cap.release()
    yaml_path = out / "chirla_calib.yaml"
    yaml_path.write_text(
        f"path: {out.resolve().as_posix()}\ntrain: images/val\nval: images/val\n"
        "names:\n" + "".join(f"  {i}: {name}\n" for i, name in enumerate(COCO_NAMES)))
    logger.info("Calibration set: %d CHIRLA frames → %s", n, out)
    return yaml_path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--imgsz", type=int, default=480)
    ap.add_argument("--model", default="n", choices=["n", "s"])
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    cfg = DetectionEngine.load_config(args.config)
    from ultralytics import YOLO

    pt = Path(cfg["models"][f"yolo11{args.model}_fp32"])
    data = build_calibration_set(Path(cfg["chirla"]["root"]), Path("output/cache/calib"))
    out = YOLO(str(pt)).export(format="openvino", int8=True, imgsz=args.imgsz, data=str(data),
                               fraction=1.0, dynamic=False)
    dst = Path("models/openvino") / f"yolo11{args.model}_int8_{args.imgsz}_openvino_model"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.move(str(out), str(dst))
    key = f"yolo11{args.model}_int8_{args.imgsz}"
    logger.info("✓ Exported %s", dst)
    print(f"\nAdd to config.yaml → models:\n  {key}: {dst.as_posix()}")


if __name__ == "__main__":
    main()
