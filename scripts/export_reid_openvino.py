"""
scripts/export_reid_openvino.py
────────────────────────────────────────────────────────────────────────────────
Convert the Re-ID-trained torchreid checkpoint (models/reid/<model>_*.pth) to
OpenVINO IR so person embeddings run 3–5× faster on CPU.

Usage:  python scripts/export_reid_openvino.py [--config config.yaml]
────────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.engine import DetectionEngine  # noqa: E402
from core.reid import ReIDExtractor  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("export_reid_openvino")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    cfg = DetectionEngine.load_config(args.config)
    reid = cfg["reid"]
    name = reid["model"]
    weights = ReIDExtractor._resolve_weights(name, reid.get("weights", "auto"))
    if not weights:
        raise SystemExit(f"No Re-ID-trained checkpoint for {name} – see config.yaml reid.weights")

    import openvino as ov
    import torch
    import torchreid

    model = torchreid.models.build_model(name, num_classes=1, pretrained=False)
    torchreid.utils.load_pretrained_weights(model, weights)
    model.eval()

    out_dir = Path(weights).parent / f"{name}_openvino"
    out_dir.mkdir(parents=True, exist_ok=True)
    xml = out_dir / f"{name}.xml"
    with torch.no_grad():
        ov_model = ov.convert_model(model, example_input=torch.randn(1, 3, 256, 128),
                                    input=[(-1, 3, 256, 128)])
    ov.save_model(ov_model, str(xml), compress_to_fp16=False)
    logger.info("✓ Saved %s", xml)

    torch_ex = ReIDExtractor(name, "cpu", weights, backend="torch")
    ov_ex = ReIDExtractor(name, "cpu", weights, backend="openvino")
    rng = np.random.default_rng(0)
    crops = [rng.integers(0, 255, (int(rng.integers(120, 300)), int(rng.integers(50, 120)), 3), dtype=np.uint8)
             for _ in range(16)]
    a, b = torch_ex.extract_batch(crops), ov_ex.extract_batch(crops)
    cos = np.sum(a * b, 1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
    logger.info("PyTorch vs OpenVINO embedding cosine: min %.4f, mean %.4f", cos.min(), cos.mean())

    for label, ex in (("PyTorch", torch_ex), ("OpenVINO", ov_ex)):
        ex.extract_batch(crops[:2])
        t = time.time()
        ex.extract_batch(crops)
        bt = time.time() - t
        t = time.time()
        for c in crops[:4]:
            ex.extract_batch([c])
        st = (time.time() - t) / 4
        logger.info("%-8s batch16: %.1f crops/s · single crop: %.0f ms", label, 16 / bt, st * 1000)


if __name__ == "__main__":
    main()
