#!/usr/bin/env python3
"""
Evaluation + calibration harness for the Deepfake & Image Forensics Toolkit.
Developed by Gourav Shrivastava

Measures how well the engine separates real from fake images on YOUR labelled data,
then (optionally) fits a calibrated combiner and saves it as calibration.json.

Dataset layout (label = first folder; optional sub-folders = source / generator):
    dataset/real/*.jpg                      or  dataset/real/<camera_or_source>/*.jpg
    dataset/fake/*.png                      or  dataset/fake/<generator_name>/*.png

Usage:
    python3 evaluate.py --data dataset                      # measure only
    python3 evaluate.py --data dataset --normalize          # re-encode all to JPEG q90 first (format-bias check)
    python3 evaluate.py --data dataset --calibrate          # + cross-validated logistic calibration report
    python3 evaluate.py --data dataset --apply              # + write calibration.json (app then uses it)
    python3 evaluate.py --data dataset --reuse --apply      # reuse features.csv from a previous run (fast)
    python3 evaluate.py --selftest                          # pipeline smoke test on a synthetic dataset
    python3 evaluate.py --reset-calibration                 # delete calibration.json (back to heuristic)

Positive class = FAKE. Everything is local; no API is called.
"""

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from PIL import Image

import forensics

HERE = os.path.dirname(os.path.abspath(__file__))
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".gif"}
MODULES = ["metadata", "faces", "ela", "noise", "frequency", "copy_move", "jpeg", "ml", "c2pa"]
# ^ NOTE: this must be kept in sync with forensics.py's WEIGHTS dict. It previously listed only
# 8 modules and silently missed "c2pa" (added in step 5) -- every evaluate.py run since then
# extracted features and calibrated weights WITHOUT the C2PA score ever being included, even
# though forensics.py itself was using it. If you add another module to WEIGHTS in forensics.py,
# add it here too or it will suffer the same silent omission.
RAW = {  # raw (unclamped) diagnostics, shown so you can see if the 0-1 clamps hide signal
    "ela_hotspot": lambda r: r["ela"]["hotspot_ratio"],
    "ela_mean": lambda r: r["ela"]["mean"],
    "noise_mean": lambda r: r["noise"].get("mean_noise"),
    "noise_variation": lambda r: r["noise"].get("variation"),
    "fft_peak_z": lambda r: r["frequency"]["peak_z"],
    "fft_peak_count": lambda r: r["frequency"]["peak_count"],
    "fft_hf_ratio": lambda r: r["frequency"]["hf_ratio"],
    "copy_pairs": lambda r: r["copy_move"]["matched_pairs"],
    "jpeg_periodicity": lambda r: r["jpeg"].get("periodicity"),
    "lsb_chi_mean": lambda r: float(np.mean(r["lsb"]["chi_norm"])),
    "c2pa_has_manifest": lambda r: int(bool(r["c2pa"]["has_manifest"])),
    "c2pa_trusted": lambda r: (int(bool(r["c2pa"]["trusted"])) if r["c2pa"]["has_manifest"] else None),
}


# =============================================================================
# Data discovery
# =============================================================================
def discover(root):
    rows = []
    for label in ("real", "fake"):
        base = os.path.join(root, label)
        if not os.path.isdir(base):
            sys.exit(f"Missing folder: {base}  (expected {root}/real and {root}/fake)")
        for dp, _, files in os.walk(base):
            for fn in sorted(files):
                if os.path.splitext(fn)[1].lower() in IMG_EXT:
                    p = os.path.join(dp, fn)
                    rel = os.path.relpath(dp, base)
                    group = label if rel == "." else f"{label}/{rel.split(os.sep)[0]}"
                    rows.append({"path": p, "label": 1 if label == "fake" else 0, "group": group})
    return rows


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dedupe(rows):
    """Drop exact duplicates; drop BOTH copies if the same file appears under real and fake."""
    seen, out, dropped, leaks = {}, [], 0, 0
    for r in rows:
        r["sha"] = sha256_file(r["path"])
        seen.setdefault(r["sha"], []).append(r)
    for sha, lst in seen.items():
        if len({x["label"] for x in lst}) > 1:
            leaks += len(lst)
            continue
        out.append(lst[0])
        dropped += len(lst) - 1
    return out, dropped, leaks


# =============================================================================
# Feature extraction (runs in worker processes)
# =============================================================================
def extract(job):
    forensics.set_calibration(None)  # always score the raw heuristic, never a previous calibration
    path, normalize = job["path"], job["normalize"]
    tmp = tempfile.mkdtemp(prefix="dfd_eval_")
    try:
        orig_struct = forensics.file_structure(path)
        with Image.open(path) as im0:
            width, height = im0.size
        target = path
        if normalize:
            im = forensics.load_working_image(path)[0]
            target = os.path.join(tmp, "norm.jpg")
            im.save(target, "JPEG", quality=90)
        r = forensics.analyze(target, tmp, fast=True)
        ms = r["verdict"]["module_scores"]
        row = {"path": path, "label": job["label"], "group": job["group"],
               "format": orig_struct["real_type"], "width": width, "height": height,
               "bytes": os.path.getsize(path), "has_exif": int(bool(r["metadata"]["exif"]) and not normalize),
               "faces": r["faces"]["count"], "hand_score": r["verdict"]["score"],
               "ai_marker": int(bool(r["metadata"]["ai_markers"]))}
        for m in MODULES:
            row[f"s_{m}"] = ms.get(m)
        for k, fn in RAW.items():
            try:
                row[f"raw_{k}"] = fn(r)
            except Exception:
                row[f"raw_{k}"] = None
        for mm in r["ml"].get("models", []):
            row[f"ml:{mm['name']}"] = mm["score"]
            row[f"mlms:{mm['name']}"] = mm["ms"]
            row[f"mlmode:{mm['name']}"] = mm["mode"]
        row["ml_errors"] = " | ".join(r["ml"].get("errors", [])) or None
        return row
    except Exception as e:
        return {"path": path, "error": f"{type(e).__name__}: {e}"}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def run_extraction(rows, workers, normalize):
    jobs = [{"path": r["path"], "label": r["label"], "group": r["group"], "normalize": normalize} for r in rows]
    out, errs, t0 = [], [], time.time()
    if workers <= 1:
        it = (extract(j) for j in jobs)
        for i, res in enumerate(it, 1):
            (errs if "error" in res else out).append(res)
            _progress(i, len(jobs), t0)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(extract, j) for j in jobs]
            for i, f in enumerate(as_completed(futs), 1):
                res = f.result()
                (errs if "error" in res else out).append(res)
                _progress(i, len(jobs), t0)
    print()
    out.sort(key=lambda r: r["path"])
    return out, errs


def _progress(i, n, t0):
    if not sys.stdout.isatty() and i != n and i % max(1, n // 10):
        return  # keep logs readable when output is piped
    el = time.time() - t0
    eta = el / i * (n - i)
    sys.stdout.write(f"\r  analysed {i}/{n}  ({el:.0f}s elapsed, ~{eta:.0f}s left)   ")
    sys.stdout.flush()


def write_csv(rows, path):
    cols = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def read_csv(path):
    rows = []
    with open(path, newline="") as f:
        for d in csv.DictReader(f):
            r = {}
            for k, v in d.items():
                if v == "" or v is None:
                    r[k] = None
                    continue
                is_str_col = k in ("path", "group", "format", "ml_errors") or k.startswith("mlmode:")
                try:
                    r[k] = v if is_str_col else float(v)
                except ValueError:
                    r[k] = v
            r["label"] = int(r["label"])
            rows.append(r)
    return rows


# =============================================================================
# Metrics (numpy only)
# =============================================================================
def _rank(a):
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a))
    sa = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sa[j + 1] == sa[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1
        i = j + 1
    return ranks


def auc(y, s):
    y, s = np.asarray(y), np.asarray(s, float)
    n1 = int((y == 1).sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = _rank(s)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def auc_ci(y, s, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    y, s = np.asarray(y), np.asarray(s, float)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        a = auc(y[idx], s[idx])
        if not np.isnan(a):
            vals.append(a)
    if not vals:
        return (float("nan"), float("nan"))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def binary_metrics(y, pred):
    y, pred = np.asarray(y), np.asarray(pred)
    tp = int(((y == 1) & (pred == 1)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    div = lambda a, b: a / b if b else float("nan")
    prec, rec = div(tp, tp + fp), div(tp, tp + fn)
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn, "accuracy": div(tp + tn, len(y)),
            "precision": prec, "recall": rec, "fpr": div(fp, fp + tn),
            "f1": div(2 * prec * rec, prec + rec) if prec == prec and rec == rec else float("nan")}


def tpr_at_fpr(y, s, target):
    """Best recall achievable while keeping false-positive rate <= target."""
    y, s = np.asarray(y), np.asarray(s, float)
    n0, n1 = (y == 0).sum(), (y == 1).sum()
    if n0 == 0 or n1 == 0:
        return float("nan"), float("nan")
    best, thr = 0.0, float("inf")
    for t in np.unique(s):
        fpr = ((s >= t) & (y == 0)).sum() / n0
        if fpr <= target:
            tpr = ((s >= t) & (y == 1)).sum() / n1
            if tpr >= best:
                best, thr = float(tpr), float(t)
    return best, thr


def youden_threshold(y, s):
    y, s = np.asarray(y), np.asarray(s, float)
    n0, n1 = (y == 0).sum(), (y == 1).sum()
    best, thr = -1, 0.5
    for t in np.unique(s):
        j = ((s >= t) & (y == 1)).sum() / n1 - ((s >= t) & (y == 0)).sum() / n0
        if j > best:
            best, thr = j, float(t)
    return thr


# =============================================================================
# Logistic regression (numpy Newton/IRLS) with non-negativity + cross validation
# =============================================================================
def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))


def _newton(X, y, sw, l2, iters=60):
    n, d = X.shape
    w, b = np.zeros(d), 0.0
    for _ in range(iters):
        p = _sigmoid(X @ w + b)
        g_w = X.T @ (sw * (p - y)) + l2 * w
        g_b = float((sw * (p - y)).sum())
        v = sw * p * (1 - p) + 1e-9
        H = np.zeros((d + 1, d + 1))
        H[:d, :d] = (X * v[:, None]).T @ X + l2 * np.eye(d)
        H[:d, d] = H[d, :d] = (X * v[:, None]).sum(axis=0)
        H[d, d] = v.sum()
        try:
            step = np.linalg.solve(H + 1e-8 * np.eye(d + 1), np.append(g_w, g_b))
        except np.linalg.LinAlgError:
            break
        w -= step[:d]
        b -= step[d]
        if np.abs(step).max() < 1e-7:
            break
    return w, b


def fit_model(X, y, names, l2=1.0, nonneg=True):
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd[sd < 1e-9] = 1.0
    Z = (X - mu) / sd
    n1, n0 = (y == 1).sum(), (y == 0).sum()
    sw = np.where(y == 1, len(y) / (2 * n1), len(y) / (2 * n0))  # balance classes
    keep = list(range(X.shape[1]))
    while True:
        w_k, b = _newton(Z[:, keep], y, sw, l2)
        if not nonneg or (w_k >= -1e-9).all() or len(keep) == 1:
            break
        keep.pop(int(np.argmin(w_k)))  # drop the most "inverted" feature, refit
    w = np.zeros(X.shape[1])
    w[keep] = np.clip(w_k, 0, None) if nonneg else w_k
    return {"names": list(names), "mu": mu, "sd": sd, "w": w, "b": float(b)}


def predict(model, X):
    return _sigmoid(((X - model["mu"]) / model["sd"]) @ model["w"] + model["b"])


def cross_validate(X, y, names, k=5, seed=0, l2=1.0, nonneg=True):
    rng = np.random.default_rng(seed)
    idx1, idx0 = np.where(y == 1)[0], np.where(y == 0)[0]
    rng.shuffle(idx1)
    rng.shuffle(idx0)
    k = max(2, min(k, len(idx1), len(idx0)))
    folds = [np.concatenate([idx1[i::k], idx0[i::k]]) for i in range(k)]
    oof = np.zeros(len(y))
    aucs = []
    for f in folds:
        train = np.setdiff1d(np.arange(len(y)), f)
        m = fit_model(X[train], y[train], names, l2, nonneg)
        oof[f] = predict(m, X[f])
        aucs.append(auc(y[f], oof[f]))
    return oof, aucs


# =============================================================================
# Analysis + report
# =============================================================================
def col(rows, key):
    return np.array([np.nan if r.get(key) is None else r[key] for r in rows], float)


def module_table(rows, y):
    out = []
    keys = [f"s_{m}" for m in MODULES] + [k for k in rows[0] if k.startswith("raw_")] + \
           sorted({k for r in rows for k in r if k.startswith("ml:")})
    for k in keys:
        v = col(rows, k)
        ok = ~np.isnan(v)
        if ok.sum() < 5 or len(set(y[ok])) < 2 or np.nanstd(v) == 0:
            continue
        a = auc(y[ok], v[ok])
        verdict = ("useful" if a >= 0.65 else "weak" if a >= 0.55 else
                   "no signal" if a > 0.45 else "INVERTED" if a > 0.35 else "INVERTED (strong)")
        out.append({"feature": k, "auc": a, "real_mean": float(v[ok][y[ok] == 0].mean()),
                    "fake_mean": float(v[ok][y[ok] == 1].mean()), "n": int(ok.sum()), "verdict": verdict})
    return out


def confound_warnings(rows, y, normalized=False):
    warns = []
    def share(pred, lab):
        sel = [r for r, yy in zip(rows, y) if yy == lab]
        return sum(pred(r) for r in sel) / max(len(sel), 1)
    jr, jf = share(lambda r: r["format"] == "JPEG", 0), share(lambda r: r["format"] == "JPEG", 1)
    if normalized:
        if abs(jr - jf) > 0.4:
            warns.append(f"Original formats differ ({jr:.0%} vs {jf:.0%} JPEG). --normalize re-encoded everything to "
                         f"JPEG q90, so format bias is neutralised, but metadata signals are gone and the score "
                         f"reflects pixel evidence only.")
    elif abs(jr - jf) > 0.4:
        warns.append(f"FORMAT BIAS: {jr:.0%} of real vs {jf:.0%} of fake images are JPEG. Metadata, ELA and JPEG "
                     f"modules will learn the file format, not authenticity. Re-run with --normalize.")
    er, ef = share(lambda r: r["has_exif"] == 1, 0), share(lambda r: r["has_exif"] == 1, 1)
    if not normalized and abs(er - ef) > 0.5:
        warns.append(f"EXIF BIAS: {er:.0%} of real vs {ef:.0%} of fake images carry EXIF.")
    area = lambda lab: np.median([r["width"] * r["height"] for r, yy in zip(rows, y) if yy == lab])
    ar, af = area(0), area(1)
    if max(ar, af) / max(min(ar, af), 1) > 3:
        warns.append(f"RESOLUTION BIAS: median pixels real={ar:,.0f} vs fake={af:,.0f} (>3x). Noise and FFT "
                     f"features depend on resolution. Resize both classes to a common range.")
    return warns


def fmt(x, p=3):
    return "n/a" if x is None or (isinstance(x, float) and x != x) else f"{x:.{p}f}"


def build_report(rows, y, args, extra):
    L = []
    n, n1 = len(y), int(y.sum())
    hand = col(rows, "hand_score")
    L.append("# Evaluation report\n")
    L.append(f"Developed by Gourav Shrivastava | generated {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    L.append("## 1. Dataset\n")
    L.append(f"- Images analysed: **{n}** (real {n - n1}, fake {n1})" + (" | normalized to JPEG q90" if args.normalize else ""))
    if extra.get("dupes"):
        L.append(f"- Exact duplicates removed: {extra['dupes']}")
    if extra.get("leaks"):
        L.append(f"- Files present in BOTH classes removed (label leak): {extra['leaks']}")
    if extra.get("errors"):
        L.append(f"- Failed to analyse (see errors.txt): {extra['errors']}")
    for lab, name in ((0, "real"), (1, "fake")):
        sel = [r for r, yy in zip(rows, y) if yy == lab]
        fm = {}
        for r in sel:
            fm[r["format"]] = fm.get(r["format"], 0) + 1
        L.append(f"- {name}: formats {fm}, median size {np.median([r['width'] for r in sel]):.0f}x"
                 f"{np.median([r['height'] for r in sel]):.0f}px, median {np.median([r['bytes'] for r in sel]) / 1024:.0f} KB, "
                 f"faces found in {sum(1 for r in sel if r['faces'] > 0)}/{len(sel)}")
    for w in extra["warnings"]:
        L.append(f"\n> WARNING: {w}")
    if n < 100 or min(n1, n - n1) < 50:
        L.append("\n> WARNING: small dataset. Every number below has wide uncertainty. Use the confidence "
                 "intervals, and aim for 200+ images per class before trusting any threshold.")

    L.append("\n## 2. Overall score (current hand-tuned engine)\n")
    lo, hi = auc_ci(y, hand)
    L.append(f"- ROC-AUC: **{fmt(auc(y, hand))}**  (95% bootstrap CI {fmt(lo)} to {fmt(hi)}).  0.5 = coin flip, 1.0 = perfect.")
    for fp_t in (0.05, 0.01):
        t, thr = tpr_at_fpr(y, hand, fp_t)
        L.append(f"- Recall at <= {fp_t:.0%} false positives: **{fmt(t)}** (threshold {fmt(thr)})")
    L.append("\n| Threshold | Accuracy | Precision | Recall | FPR | F1 | TP | FP | TN | FN |\n|---|---|---|---|---|---|---|---|---|---|")
    for name, t in (("MEDIUM+ (>=0.35)", 0.35), ("HIGH (>=0.65)", 0.65), ("Youden-best", youden_threshold(y, hand))):
        m = binary_metrics(y, (hand >= t).astype(int))
        L.append(f"| {name} ({t:.2f}) | {fmt(m['accuracy'])} | {fmt(m['precision'])} | {fmt(m['recall'])} | "
                 f"{fmt(m['fpr'])} | {fmt(m['f1'])} | {m['tp']} | {m['fp']} | {m['tn']} | {m['fn']} |")

    L.append("\n## 3. Which modules actually carry signal\n")
    L.append("AUC per feature, fake = positive. `INVERTED` means the feature points the wrong way on your data.\n")
    L.append("| Feature | AUC | Real mean | Fake mean | N | Verdict |\n|---|---|---|---|---|---|")
    for t in extra["mod_table"]:
        L.append(f"| {t['feature']} | {fmt(t['auc'])} | {fmt(t['real_mean'])} | {fmt(t['fake_mean'])} | {t['n']} | {t['verdict']} |")
    L.append("\n`s_*` = the 0-1 module scores used in the verdict. `raw_*` = unclamped values. If a `raw_*` feature "
             "beats its `s_*` counterpart, the clamp thresholds in forensics.py are hiding signal.")

    L.append("\n## 4. Per source / generator\n")
    L.append("Real groups show how often real images are wrongly flagged (false-positive rate). Fake groups show recall. Flagged = score >= 0.35.\n")
    L.append("| Group | N | Mean score | Flagged |\n|---|---|---|---|")
    groups = sorted({r["group"] for r in rows})
    for g in groups:
        idx = [i for i, r in enumerate(rows) if r["group"] == g]
        L.append(f"| {g} | {len(idx)} | {fmt(float(np.mean(hand[idx])))} | {fmt(float(np.mean(hand[idx] >= 0.35)), 2)} |")

    if extra.get("cal"):
        c = extra["cal"]
        L.append("\n## 5. Calibration (cross-validated)\n")
        L.append(f"- Features used: {', '.join(c['features'])}")
        L.append(f"- Hand-tuned AUC: {fmt(c['hand_auc'])}  |  Calibrated out-of-fold AUC: **{fmt(c['oof_auc'])}** "
                 f"(per-fold mean {fmt(c['fold_mean'])} +/- {fmt(c['fold_std'])})")
        L.append(f"- Thresholds chosen on out-of-fold predictions: MEDIUM >= {fmt(c['thr_medium'])}, HIGH >= {fmt(c['thr_high'])} (<= 5% FPR)")
        L.append(f"- Out-of-fold at MEDIUM: recall {fmt(c['med']['recall'])}, FPR {fmt(c['med']['fpr'])} | at HIGH: recall {fmt(c['high']['recall'])}, FPR {fmt(c['high']['fpr'])}")
        L.append("\n| Feature | Weight (standardised) |\n|---|---|")
        for f, w in sorted(zip(c["features_all"], c["w_all"]), key=lambda x: -x[1]):
            L.append(f"| {f} | {w:.3f}{'  (dropped: pointed the wrong way)' if w == 0 else ''} |")
        gain = c["oof_auc"] - c["hand_auc"]
        L.append(f"\nAUC change vs hand-tuned: **{gain:+.3f}**." +
                 (" Small or negative gain: keep the heuristic or collect more data." if gain < 0.02 else ""))
        L.append("Applied to the app: " + ("**YES, calibration.json written**" if extra.get("applied") else "no (add --apply)"))
    L.append("\n## 6. How to read this honestly\n")
    L.append("- AUC on a dataset like yours does not transfer to other generators or cameras. State the dataset when you quote a number.\n"
             "- If real and fake differ in format, size or source, you measured that difference, not authenticity.\n"
             "- Calibrated thresholds are only as good as the sample. Re-run when you add new generators.")
    return "\n".join(L)


def calibrate(rows, y, args):
    feats = [f"s_{m}" for m in MODULES]
    usable = []
    for f in feats:
        v = col(rows, f)
        if not np.isnan(v).any() and np.nanstd(v) > 0:
            usable.append(f)
    if len(usable) < 2:
        sys.exit("Not enough usable features for calibration.")
    X = np.column_stack([col(rows, f) for f in usable])
    oof, fold_aucs = cross_validate(X, y, usable, k=5, seed=args.seed, l2=args.l2)
    hand = col(rows, "hand_score")
    t_med = youden_threshold(y, oof)
    _, t_high = tpr_at_fpr(y, oof, 0.05)
    if not np.isfinite(t_high) or t_high < t_med:
        t_high = t_med
    model = fit_model(X, y, usable, l2=args.l2)
    med = binary_metrics(y, (oof >= t_med).astype(int))
    high = binary_metrics(y, (oof >= t_high).astype(int))
    info = {"features": [f[2:] for f in usable], "features_all": [f[2:] for f in usable],
            "w_all": [float(w) for w in model["w"]], "hand_auc": auc(y, hand), "oof_auc": auc(y, oof),
            "fold_mean": float(np.nanmean(fold_aucs)), "fold_std": float(np.nanstd(fold_aucs)),
            "thr_medium": t_med, "thr_high": t_high, "med": med, "high": high}
    cal_json = {"version": 1, "created": time.strftime("%Y-%m-%d %H:%M:%S"), "developer": "Gourav Shrivastava",
                "n_images": int(len(y)), "n_real": int((y == 0).sum()), "n_fake": int((y == 1).sum()),
                "normalized": bool(args.normalize), "features": info["features"],
                "mu": [float(m) for m in model["mu"]], "sd": [float(s) for s in model["sd"]],
                "w": [float(w) for w in model["w"]], "b": model["b"],
                "thresholds": {"medium": float(t_med), "high": float(t_high)},
                "cv": {"oof_auc": info["oof_auc"], "fold_mean": info["fold_mean"], "fold_std": info["fold_std"],
                       "hand_auc": info["hand_auc"]}}
    return info, cal_json


# =============================================================================
# Self-test dataset
# =============================================================================
def make_selftest(root, n=40, seed=0):
    import cv2
    rng = np.random.default_rng(seed)

    def base():
        chans = [cv2.resize(rng.random((6, 6)).astype(np.float32), (384, 384), interpolation=cv2.INTER_CUBIC)
                 for _ in range(3)]
        return np.dstack(chans) * 180 + 40

    for lab, groups in (("real", ["cam_a", "cam_b"]), ("fake", ["gen_a", "gen_b"])):
        for g in groups:
            d = os.path.join(root, lab, g)
            os.makedirs(d, exist_ok=True)
            for i in range(n // 2):
                img = base()
                if lab == "real":  # camera-like: sensor noise
                    img = img + rng.normal(0, 5, img.shape)
                else:  # synthetic-like: smoother, tiny periodic upsampling grid
                    img = cv2.GaussianBlur(img, (0, 0), 1.1)
                    yy, xx = np.mgrid[0:384, 0:384]
                    img = img + 4 * np.sin(2 * np.pi * xx / 8)[..., None] + rng.normal(0, 1.5, img.shape)
                Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(
                    os.path.join(d, f"{lab}_{g}_{i:03d}.jpg"), quality=92)


# =============================================================================
def main():
    ap = argparse.ArgumentParser(description="Evaluate and calibrate the forensics engine on a labelled dataset")
    ap.add_argument("--data", help="dataset folder containing real/ and fake/")
    ap.add_argument("--out", default=os.path.join(HERE, "eval_out"))
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--normalize", action="store_true", help="re-encode every image to JPEG q90 (removes format bias)")
    ap.add_argument("--calibrate", action="store_true", help="fit + cross-validate a logistic combiner")
    ap.add_argument("--apply", action="store_true", help="write calibration.json so the app uses it (implies --calibrate)")
    ap.add_argument("--reuse", action="store_true", help="reuse features.csv from --out instead of re-analysing")
    ap.add_argument("--force", action="store_true", help="allow calibration on very small datasets")
    ap.add_argument("--l2", type=float, default=1.0, help="L2 regularisation strength for calibration")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="analyse at most N images per class (quick trial)")
    ap.add_argument("--selftest", action="store_true", help="run on a synthetic dataset to verify the pipeline")
    ap.add_argument("--reset-calibration", action="store_true")
    args = ap.parse_args()

    if args.reset_calibration:
        if os.path.exists(forensics.CAL_PATH):
            os.remove(forensics.CAL_PATH)
            print("Removed", forensics.CAL_PATH)
        else:
            print("No calibration file present.")
        return
    if args.apply:
        args.calibrate = True

    tmp_selftest = None
    if args.selftest:
        tmp_selftest = tempfile.mkdtemp(prefix="dfd_selftest_")
        make_selftest(tmp_selftest)
        args.data = tmp_selftest
        args.out = os.path.join(tmp_selftest, "out")
        forensics.CAL_PATH = os.path.join(tmp_selftest, "calibration.json")  # never touch the real one
        args.calibrate = True
        print("SELFTEST: synthetic dataset. Numbers only prove the pipeline works, not real accuracy.\n")
    if not args.data:
        ap.error("--data is required (or use --selftest)")
    os.makedirs(args.out, exist_ok=True)
    feat_csv = os.path.join(args.out, "features.csv")
    extra = {"dupes": 0, "leaks": 0, "errors": 0}

    if args.reuse and os.path.exists(feat_csv):
        print(f"Reusing {feat_csv}")
        rows = read_csv(feat_csv)
    else:
        rows = discover(args.data)
        if args.limit:
            keep = []
            for lab in (0, 1):
                keep += [r for r in rows if r["label"] == lab][:args.limit]
            rows = keep
        if not rows:
            sys.exit("No images found.")
        rows, extra["dupes"], extra["leaks"] = dedupe(rows)
        print(f"{len(rows)} unique images. Analysing with {args.workers} worker(s)...")
        rows, errs = run_extraction(rows, args.workers, args.normalize)
        extra["errors"] = len(errs)
        if errs:
            with open(os.path.join(args.out, "errors.txt"), "w") as f:
                f.write("\n".join(f"{e['path']}: {e['error']}" for e in errs))
        write_csv(rows, feat_csv)

    y = np.array([r["label"] for r in rows])
    if len(set(y)) < 2:
        sys.exit("Need both real and fake images.")
    extra["warnings"] = confound_warnings(rows, y, args.normalize)
    extra["mod_table"] = module_table(rows, y)

    if args.calibrate:
        if (len(y) < 60 or min((y == 0).sum(), (y == 1).sum()) < 20) and not args.force:
            print("\nCalibration skipped: need >= 60 images and >= 20 per class (use --force to override, "
                  "results will be unreliable).")
        else:
            info, cal_json = calibrate(rows, y, args)
            extra["cal"] = info
            if args.apply:
                with open(forensics.CAL_PATH, "w") as f:
                    json.dump(cal_json, f, indent=2)
                extra["applied"] = True
            with open(os.path.join(args.out, "calibration_candidate.json"), "w") as f:
                json.dump(cal_json, f, indent=2)

    report = build_report(rows, y, args, extra)
    with open(os.path.join(args.out, "report.md"), "w") as f:
        f.write(report)
    metrics = {"n": int(len(y)), "hand_auc": auc(y, col(rows, "hand_score")),
               "modules": extra["mod_table"], "calibration": extra.get("cal")}
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, default=lambda o: None if o != o else float(o))
    print("\n" + report)
    print(f"\nFiles written to {args.out}: report.md, metrics.json, features.csv")
    if tmp_selftest:
        shutil.rmtree(tmp_selftest, ignore_errors=True)


if __name__ == "__main__":
    main()
