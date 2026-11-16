"""
scripts/train_open_set.py
────────────────────────────────────────────────────────────────────────────────
Fit and evaluate the open-set (Known employee vs UNKNOWN visitor) classifier.

Usage:  python scripts/train_open_set.py [--config config.yaml]
────────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.chirla import (day_gallery_path, emp_label, employee_ids, imread,  # noqa: E402
                         index_benchmark_crops, sequence_dates)
from core.engine import DetectionEngine  # noqa: E402
from core.open_set import (FEATURE_NAMES, OpenSetClassifier,  # noqa: E402
                           PrototypeGallery, tracklet_features)
from core.reid import ReIDExtractor  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_open_set")

MAX_GAP = 90


def embed_records(records, extractor, cache_file: Path):
    """Embed every crop once and cache the result."""
    keys = [f"{r.seq}|{r.camera}|{r.pid}|{r.frame}" for r in records]
    cache = {}
    if cache_file.exists():
        data = np.load(cache_file, allow_pickle=False)
        cache = dict(zip(data["keys"].tolist(), data["embs"]))
    todo = [i for i, k in enumerate(keys) if k not in cache]
    logger.info("Embeddings: %d cached, %d to compute", len(keys) - len(todo), len(todo))
    cache_file.parent.mkdir(parents=True, exist_ok=True)

    def save():
        if cache:
            np.savez(cache_file, keys=np.array(list(cache.keys())), embs=np.stack(list(cache.values())))

    batch = 32
    t0 = time.time()
    for s in range(0, len(todo), batch):
        idx = todo[s:s + batch]
        imgs = [imread(records[i].path) for i in idx]
        ok = [(i, im) for i, im in zip(idx, imgs) if im is not None and im.size]
        if ok:
            embs = extractor.extract_batch([im for _, im in ok])
            for (i, _), e in zip(ok, embs):
                cache[keys[i]] = e.astype(np.float32)
        done = min(s + batch, len(todo))
        if (s // batch) % 5 == 0 or done == len(todo):
            rate = done / max(time.time() - t0, 1e-6)
            logger.info("  embedded %d / %d  (%.0f crops/s, ~%.0f min left)", done, len(todo),
                        rate, (len(todo) - done) / max(rate, 1e-6) / 60)
        if (s // batch) % 50 == 49:
            save()
    save()
    return [cache.get(k) for k in keys]


def after_enrollment(r, cutoff, gap):
    if cutoff is None:
        return True
    seq, frame = cutoff
    return r.seq > seq or (r.seq == seq and r.frame > frame + gap)


def build_tracklets(records, embs, window: int, min_crops: int, dates, galleries, gap):
    groups = defaultdict(list)
    for r, e in zip(records, embs):
        if e is None:
            continue
        g = galleries.get(dates[r.seq])
        cutoff = g.meta["enroll_cutoff"].get(emp_label(r.pid)) if g else None
        if after_enrollment(r, cutoff, gap):
            groups[(r.seq, r.camera, r.pid)].append((r.frame, e))
    out = []
    for (seq, cam, pid), items in groups.items():
        items.sort(key=lambda t: t[0])
        run = [items[0]]
        for prev, cur in zip(items, items[1:]):
            if cur[0] - prev[0] > MAX_GAP:
                out += _windows(seq, cam, pid, run, window, min_crops, dates[seq])
                run = []
            run.append(cur)
        out += _windows(seq, cam, pid, run, window, min_crops, dates[seq])
    return out


def _windows(seq, cam, pid, run, window, min_crops, day):
    res = []
    for s in range(0, len(run), window):
        chunk = run[s:s + window]
        if len(chunk) >= min_crops:
            res.append({"seq": seq, "camera": cam, "pid": pid, "day": day,
                        "embs": np.stack([e for _, e in chunk])})
    return res


def featurise(tracklets, galleries, employees):
    X, y, enrolled, correct_id = [], [], [], []
    for t in tracklets:
        gallery = galleries[t["day"]]
        f, win = tracklet_features(t["embs"], gallery)
        is_emp = t["pid"] in employees and emp_label(t["pid"]) in gallery.labels
        right = is_emp and gallery.labels[win] == emp_label(t["pid"])
        X.append(f)
        y.append(int(right))
        enrolled.append(is_emp)
        correct_id.append(right)
    return np.array(X), np.array(y), np.array(enrolled), np.array(correct_id)


def open_set_metrics(accept, enrolled, correct_id):
    emp, vis = enrolled, ~enrolled
    return {
        "TAR": float(np.mean(accept[emp] & correct_id[emp])) if emp.any() else float("nan"),
        "MisID": float(np.mean(accept[emp] & ~correct_id[emp])) if emp.any() else float("nan"),
        "FAR": float(np.mean(accept[vis])) if vis.any() else float("nan"),
        "TRR": float(np.mean(~accept[vis])) if vis.any() else float("nan"),
        "n_employee": int(emp.sum()),
        "n_visitor": int(vis.sum()),
    }


def naive_single_frame(tracklets, galleries, employees, thresh):
    accept, enrolled, correct = [], [], []
    for t in tracklets:
        gallery = galleries[t["day"]]
        sims = gallery.similarities(t["embs"])
        for row in sims:
            w = int(np.argmax(row))
            accept.append(thresh is None or row[w] >= thresh)
            is_emp = t["pid"] in employees and emp_label(t["pid"]) in gallery.labels
            enrolled.append(is_emp)
            correct.append(is_emp and gallery.labels[w] == emp_label(t["pid"]))
    return open_set_metrics(np.array(accept), np.array(enrolled), np.array(correct))


def fmt(m):
    return (f"| {m['TAR']:.1%} | {m['MisID']:.1%} | {m['FAR']:.1%} | {m['TRR']:.1%} "
            f"| {m['n_employee']} / {m['n_visitor']} |")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    cfg = DetectionEngine.load_config(args.config)
    ch, reid, trk = cfg["chirla"], cfg["reid"], cfg["tracking"]
    window = int(trk.get("tracklet_window", 10))
    min_crops = int(trk.get("min_tracklet_crops", 4))
    p_thresh = float(reid.get("open_set_p_threshold", 0.5))
    naive_thresh = float(reid.get("reid_thresh", 0.45))
    employees = set(employee_ids(cfg))

    dates = sequence_dates(ch["root"])
    days = list(ch["train_days"]) + list(ch["test_days"])
    galleries = {d: PrototypeGallery.load(day_gallery_path(reid["gallery_file"], d)) for d in days}
    gap = int(ch.get("min_gap_after_enroll", 600))

    records = [r for r in index_benchmark_crops(ch["root"]) if dates.get(r.seq) in galleries]
    extractor = ReIDExtractor(reid["model"], device=reid.get("device", "cpu"),
                              weights=reid.get("weights", "auto"))
    embs = embed_records(records, extractor, Path("output/cache") / f"crop_embs_{reid['model']}.npz")

    tracklets = build_tracklets(records, embs, window, min_crops, dates, galleries, gap)
    train = [t for t in tracklets if t["day"] in ch["train_days"]]
    test = [t for t in tracklets if t["day"] in ch["test_days"]]
    Xtr, ytr, _, _ = featurise(train, galleries, employees)
    Xte, yte, enr_te, cor_te = featurise(test, galleries, employees)
    logger.info("Tracklets: train %d (%d positive), test %d (%d positive)",
                len(ytr), ytr.sum(), len(yte), yte.sum())

    scaler = StandardScaler().fit(Xtr)
    lr = LogisticRegression(class_weight="balanced", max_iter=1000).fit(scaler.transform(Xtr), ytr)
    params = {
        "features": FEATURE_NAMES,
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "coef": lr.coef_[0].tolist(),
        "intercept": float(lr.intercept_[0]),
    }

    out_model = Path(reid["open_set_model"])
    out_model.parent.mkdir(parents=True, exist_ok=True)
    out_model.write_text(json.dumps(params, indent=2))
    clf = OpenSetClassifier(out_model)
    p_te = np.array([clf.p_known(x) for x in Xte])
    tracklet_m = open_set_metrics(p_te >= p_thresh, enr_te, cor_te)
    auc = float(roc_auc_score(yte, p_te)) if 0 < yte.sum() < len(yte) else float("nan")

    rule = OpenSetClassifier(None)
    rule_m = open_set_metrics(np.array([rule.p_known(x) >= p_thresh for x in Xte]), enr_te, cor_te)
    naive_closed = naive_single_frame(test, galleries, employees, None)
    naive_thr = naive_single_frame(test, galleries, employees, naive_thresh)

    params["metrics"] = {"test_tracklet_lr": tracklet_m, "test_auc": auc, "tracklet_rule": rule_m,
                         "naive_closed_set": naive_closed, "naive_threshold": naive_thr,
                         "naive_thresh_value": naive_thresh, "train_days": ch["train_days"],
                         "test_days": ch["test_days"]}
    out_model.write_text(json.dumps(params, indent=2))

    test_visitors = sorted({t["pid"] for t in test if t["pid"] not in employees})
    train_visitors = sorted({t["pid"] for t in train if t["pid"] not in employees})
    report = "\n".join([
        "# Open-set evaluation (CHIRLA)",
        "",
        f"- Re-ID model: `{reid['model']}` (OSNet-AIN, MSMT17 weights)",
        "- Protocol: same-day ('morning entrance') enrollment; each tracklet matched against "
        "its own day's gallery",
        f"- Enrolled per day: " + ", ".join(f"{d}: {len(galleries[d])}" for d in days),
        f"- Train days: {', '.join(ch['train_days'])}  (visitors {train_visitors})",
        f"- Test days: {', '.join(ch['test_days'])}  (visitors {test_visitors}, unseen in training)",
        f"- Tracklets: {window} crops max, {min_crops} min; employee footage before enrollment "
        f"cut-off + {gap / 30:.0f} s excluded",
        f"- Tracklets used: train {len(ytr)}, test {len(yte)}",
        f"- P(Known) threshold: {p_thresh}; naive cosine threshold: {naive_thresh}",
        "",
        "TAR = employee accepted with correct ID · MisID = employee accepted as wrong ID · "
        "FAR = visitor accepted as an employee · TRR = visitor rejected as UNKNOWN",
        "",
        "| Method (test split) | TAR ↑ | MisID ↓ | FAR ↓ | TRR ↑ | samples emp / vis |",
        "|---|---|---|---|---|---|",
        "| Naive single-frame, closed-set (always nearest) " + fmt(naive_closed),
        f"| Naive single-frame, cosine ≥ {naive_thresh} " + fmt(naive_thr),
        "| Tracklet + rule-based fallback " + fmt(rule_m),
        "| **Tracklet + logistic regression** " + fmt(tracklet_m),
        "",
        f"Tracklet LR ROC-AUC (test): {auc:.3f}",
        "",
        "Logistic-regression weights (standardised features):",
        "",
        *[f"- {n}: {w:+.3f}" for n, w in zip(FEATURE_NAMES, params["coef"])],
        f"- intercept: {params['intercept']:+.3f}",
    ])
    Path("output").mkdir(exist_ok=True)
    Path("output/open_set_report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
