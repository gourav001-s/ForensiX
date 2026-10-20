#!/usr/bin/env python3
"""
Evaluation + calibration harness for the VIDEO pipeline (video.py).
Developed by Gourav Shrivastava

The counterpart to evaluate.py, for video instead of images. Reuses evaluate.py's
statistical machinery (AUC, bootstrap CI, cross-validated logistic calibration) rather
than duplicating it -- only the video-specific parts (discovery, feature extraction,
confound checks) are new here. See evaluate.py's docstring for the underlying math.

Dataset layout (same convention as evaluate.py, but video files):
    dataset/real/*.mp4                or  dataset/real/<source>/*.mp4
    dataset/fake/*.mp4                or  dataset/fake/<method>/*.mp4

Usage:
    python3 evaluate_video.py --data dataset                    # measure only
    python3 evaluate_video.py --data dataset --calibrate        # + cross-validated calibration report
    python3 evaluate_video.py --data dataset --apply            # + write video_calibration.json
    python3 evaluate_video.py --data dataset --reuse --apply    # reuse features.csv from a previous run
    python3 evaluate_video.py --selftest                        # pipeline smoke test on synthetic clips
    python3 evaluate_video.py --reset-calibration                # delete video_calibration.json

Positive class = FAKE. Everything is local; no API is called. Video analysis is much
heavier per-file than images (frame extraction + per-frame face tracking + optional ML),
so this defaults to fewer parallel workers and expect it to take noticeably longer.
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

import evaluate as E  # reuse auc/auc_ci/binary_metrics/tpr_at_fpr/youden_threshold/fit_model/cross_validate
import video as V

HERE = os.path.dirname(os.path.abspath(__file__))
VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
MODULES = ["pulse", "blink", "flicker", "ml"]  # keep in sync with video.py's compute_video_verdict weights dict
RAW = {
    "periodicity": lambda r: r["pulse"].get("periodicity"),
    "dominant_bpm": lambda r: r["pulse"].get("dominant_bpm"),
    "blinks": lambda r: r["blink"].get("blinks"),
    "blink_rate": lambda r: r["blink"].get("rate_per_min"),
    "flicker_ratio": lambda r: r["flicker"].get("ratio"),
    "ml_mean": lambda r: r["ml"].get("mean"),
    "ml_max": lambda r: r["ml"].get("max"),
    "ml_std": lambda r: r["ml"].get("std"),
}


# =============================================================================
# Data discovery (video-specific; evaluate.py's discover() is hardcoded to image ext)
# =============================================================================
def discover(root):
    rows = []
    for label in ("real", "fake"):
        base = os.path.join(root, label)
        if not os.path.isdir(base):
            sys.exit(f"Missing folder: {base}  (expected {root}/real and {root}/fake)")
        for dp, _, files in os.walk(base):
            for fn in sorted(files):
                if os.path.splitext(fn)[1].lower() in VIDEO_EXT:
                    p = os.path.join(dp, fn)
                    rel = os.path.relpath(dp, base)
                    group = label if rel == "." else f"{label}/{rel.split(os.sep)[0]}"
                    rows.append({"path": p, "label": 1 if label == "fake" else 0, "group": group})
    return rows


# =============================================================================
# Feature extraction (runs in worker processes)
# =============================================================================
def extract(job):
    V.set_calibration(None)  # always score the raw heuristic, never a previous video calibration
    path = job["path"]
    tmp = tempfile.mkdtemp(prefix="dfd_video_eval_")
    try:
        info = V.probe(path)
        r = V.analyze_video(path, tmp, fast=True)
        if "error" in r:
            return {"path": path, "error": r["error"]}
        ms = r["verdict"]["module_scores"]
        row = {"path": path, "label": job["label"], "group": job["group"],
               "duration_s": info["duration_s"] if info else None,
               "fps": info["fps"] if info else None, "width": info["width"] if info else None,
               "height": info["height"] if info else None, "bytes": os.path.getsize(path),
               "detector": r.get("detector") or "none", "n_frames_sampled": r["n_frames_sampled"],
               "n_frames_tracked": r["n_frames_tracked"], "hand_score": r["verdict"]["score"]}
        for m in MODULES:
            row[f"s_{m}"] = ms.get(m)
        for k, fn in RAW.items():
            try:
                row[f"raw_{k}"] = fn(r)
            except Exception:
                row[f"raw_{k}"] = None
        return row
    except Exception as e:
        return {"path": path, "error": f"{type(e).__name__}: {e}"}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def run_extraction(rows, workers):
    jobs = [{"path": r["path"], "label": r["label"], "group": r["group"]} for r in rows]
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
        return
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
                is_str_col = k in ("path", "group", "detector")
                try:
                    r[k] = v if is_str_col else float(v)
                except ValueError:
                    r[k] = v
            r["label"] = int(r["label"])
            rows.append(r)
    return rows


def dedupe(rows):
    seen = {}
    for r in rows:
        h = hashlib.sha256()
        with open(r["path"], "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        r["sha"] = h.hexdigest()
        seen.setdefault(r["sha"], []).append(r)
    out, dropped, leaks = [], 0, 0
    for sha, lst in seen.items():
        if len({x["label"] for x in lst}) > 1:
            leaks += len(lst)
            continue
        out.append(lst[0])
        dropped += len(lst) - 1
    return out, dropped, leaks


# =============================================================================
# Confound checks (video-specific: duration/fps/detector, not format/EXIF like images)
# =============================================================================
def confound_warnings(rows, y):
    warns = []

    def med(pred, lab):
        vals = [pred(r) for r, yy in zip(rows, y) if yy == lab and pred(r) is not None]
        return float(np.median(vals)) if vals else None

    dr, df = med(lambda r: r["duration_s"], 0), med(lambda r: r["duration_s"], 1)
    if dr and df and max(dr, df) / max(min(dr, df), 0.1) > 2.5:
        warns.append(f"DURATION BIAS: median duration real={dr:.1f}s vs fake={df:.1f}s. Pulse periodicity and "
                     f"blink rate both need enough time to estimate reliably -- a systematic duration gap "
                     f"between classes will look like a detection signal but is really just data quantity.")

    def share_detector(lab, det):
        sel = [r for r, yy in zip(rows, y) if yy == lab]
        return sum(1 for r in sel if r["detector"] == det) / max(len(sel), 1)

    for det in ("haar", "mediapipe"):
        hr, hf = share_detector(0, det), share_detector(1, det)
        if abs(hr - hf) > 0.5:
            warns.append(f"DETECTOR BIAS: {hr:.0%} of real vs {hf:.0%} of fake videos were tracked with the "
                         f"'{det}' detector. Haar vs MediaPipe changes what signals are even available "
                         f"(blink rate needs MediaPipe) -- an uneven split confounds the comparison.")

    fr, ff = med(lambda r: r["fps"], 0), med(lambda r: r["fps"], 1)
    if fr and ff and max(fr, ff) / max(min(fr, ff), 1) > 2:
        warns.append(f"FRAME-RATE BIAS: median fps real={fr:.0f} vs fake={ff:.0f}. Affects how many distinct "
                     f"frames get sampled and the pulse signal's effective sample rate.")
    return warns


# =============================================================================
# Report
# =============================================================================
def col(rows, key):
    return np.array([np.nan if r.get(key) is None else r[key] for r in rows], float)


def module_table(rows, y):
    out = []
    keys = [f"s_{m}" for m in MODULES] + [k for k in rows[0] if k.startswith("raw_")]
    for k in keys:
        v = col(rows, k)
        ok = ~np.isnan(v)
        if ok.sum() < 5 or len(set(y[ok])) < 2 or np.nanstd(v) == 0:
            continue
        a = E.auc(y[ok], v[ok])
        verdict = ("useful" if a >= 0.65 else "weak" if a >= 0.55 else
                   "no signal" if a > 0.45 else "INVERTED" if a > 0.35 else "INVERTED (strong)")
        out.append({"feature": k, "auc": a, "real_mean": float(v[ok][y[ok] == 0].mean()),
                    "fake_mean": float(v[ok][y[ok] == 1].mean()), "n": int(ok.sum()), "verdict": verdict})
    return out


def fmt(x, p=3):
    return "n/a" if x is None or (isinstance(x, float) and x != x) else f"{x:.{p}f}"


def build_report(rows, y, args, extra):
    L = ["# Video evaluation report\n", f"Developed by Gourav Shrivastava | generated {time.strftime('%Y-%m-%d %H:%M:%S')}\n"]
    n, n1 = len(y), int(y.sum())
    hand = col(rows, "hand_score")
    L.append("## 1. Dataset\n")
    L.append(f"- Videos analysed: **{n}** (real {n - n1}, fake {n1})")
    if extra.get("dupes"):
        L.append(f"- Exact duplicates removed: {extra['dupes']}")
    if extra.get("leaks"):
        L.append(f"- Files present in BOTH classes removed (label leak): {extra['leaks']}")
    if extra.get("errors"):
        L.append(f"- Failed to analyse (see errors.txt): {extra['errors']}")
    for lab, name in ((0, "real"), (1, "fake")):
        sel = [r for r, yy in zip(rows, y) if yy == lab]
        med_dur = np.median([r["duration_s"] for r in sel if r["duration_s"]]) if sel else 0
        dets = {}
        for r in sel:
            dets[r["detector"]] = dets.get(r["detector"], 0) + 1
        L.append(f"- {name}: median duration {med_dur:.1f}s, detectors used {dets}, "
                 f"median frames tracked {np.median([r['n_frames_tracked'] for r in sel]):.0f}")
    for w in extra["warnings"]:
        L.append(f"\n> WARNING: {w}")
    if n < 40 or min(n1, n - n1) < 15:
        L.append("\n> WARNING: small dataset. Video ground-truth datasets are expensive to build, but numbers "
                 "below have very wide uncertainty until you have 40+ per class. Read the confidence intervals.")

    L.append("\n## 2. Overall score (current hand-tuned engine)\n")
    lo, hi = E.auc_ci(y, hand)
    L.append(f"- ROC-AUC: **{fmt(E.auc(y, hand))}** (95% bootstrap CI {fmt(lo)} to {fmt(hi)})")
    for fp_t in (0.05, 0.01):
        t, thr = E.tpr_at_fpr(y, hand, fp_t)
        L.append(f"- Recall at <= {fp_t:.0%} false positives: **{fmt(t)}** (threshold {fmt(thr)})")
    L.append("\n| Threshold | Accuracy | Precision | Recall | FPR | F1 |\n|---|---|---|---|---|---|")
    for name, t in (("MEDIUM+ (>=0.32)", 0.32), ("HIGH (>=0.6)", 0.6), ("Youden-best", E.youden_threshold(y, hand))):
        m = E.binary_metrics(y, (hand >= t).astype(int))
        L.append(f"| {name} ({t:.2f}) | {fmt(m['accuracy'])} | {fmt(m['precision'])} | {fmt(m['recall'])} | {fmt(m['fpr'])} | {fmt(m['f1'])} |")

    L.append("\n## 3. Which video signals actually carry signal\n")
    L.append("| Feature | AUC | Real mean | Fake mean | N | Verdict |\n|---|---|---|---|---|---|")
    for t in extra["mod_table"]:
        L.append(f"| {t['feature']} | {fmt(t['auc'])} | {fmt(t['real_mean'])} | {fmt(t['fake_mean'])} | {t['n']} | {t['verdict']} |")
    L.append("\nBlink rate needs the MediaPipe face model; on a dataset where it never loaded, s_blink will be "
             "absent here entirely (constant/missing), not just weak.")

    L.append("\n## 4. Per source / method\n")
    L.append("| Group | N | Mean score | Flagged (>=0.32) |\n|---|---|---|---|")
    for g in sorted({r["group"] for r in rows}):
        idx = [i for i, r in enumerate(rows) if r["group"] == g]
        L.append(f"| {g} | {len(idx)} | {fmt(float(np.mean(hand[idx])))} | {fmt(float(np.mean(hand[idx] >= 0.32)), 2)} |")

    if extra.get("cal"):
        c = extra["cal"]
        L.append("\n## 5. Calibration (cross-validated)\n")
        L.append(f"- Features used: {', '.join(c['features'])}")
        L.append(f"- Hand-tuned AUC: {fmt(c['hand_auc'])}  |  Calibrated out-of-fold AUC: **{fmt(c['oof_auc'])}** "
                 f"(per-fold mean {fmt(c['fold_mean'])} +/- {fmt(c['fold_std'])})")
        L.append(f"- Thresholds: MEDIUM >= {fmt(c['thr_medium'])}, HIGH >= {fmt(c['thr_high'])} (<= 5% FPR)")
        L.append("\n| Feature | Weight (standardised) |\n|---|---|")
        for f, w in sorted(zip(c["features_all"], c["w_all"]), key=lambda x: -x[1]):
            L.append(f"| {f} | {w:.3f}{'  (dropped: pointed the wrong way)' if w == 0 else ''} |")
        gain = c["oof_auc"] - c["hand_auc"]
        L.append(f"\nAUC change vs hand-tuned: **{gain:+.3f}**." +
                 (" Small or negative gain: keep the heuristic or collect more data." if gain < 0.02 else ""))
        L.append("Applied to the app: " + ("**YES, video_calibration.json written**" if extra.get("applied") else "no (add --apply)"))
    L.append("\n## 6. How to read this honestly\n")
    L.append("- Video ground truth is far scarcer than image ground truth. Treat any AUC here as a rough "
             "signal, not a benchmark claim, until the dataset is large and diverse in generators/sources.\n"
             "- rPPG pulse and blink rate are soft signals sensitive to lighting, compression and skin tone. "
             "A model that leans heavily on them will generalise worse than one that doesn't.\n"
             "- Re-run when you add new generation methods; a threshold fit today can go stale fast in this space.")
    return "\n".join(L)


def calibrate(rows, y, args):
    feats = [f"s_{m}" for m in MODULES]
    usable = [f for f in feats if not np.isnan(col(rows, f)).any() and np.nanstd(col(rows, f)) > 0]
    if len(usable) < 2:
        sys.exit(f"Not enough usable features for calibration (found: {usable}). Need at least 2 signals with "
                 f"variance across the dataset -- e.g. install the MediaPipe face model for blink data, or an "
                 f"ML classifier, so more than one video signal actually varies.")
    X = np.column_stack([col(rows, f) for f in usable])
    oof, fold_aucs = E.cross_validate(X, y, usable, k=5, seed=args.seed, l2=args.l2)
    hand = col(rows, "hand_score")
    t_med = E.youden_threshold(y, oof)
    _, t_high = E.tpr_at_fpr(y, oof, 0.05)
    if not np.isfinite(t_high) or t_high < t_med:
        t_high = t_med
    model = E.fit_model(X, y, usable, l2=args.l2)
    info = {"features": [f[2:] for f in usable], "features_all": [f[2:] for f in usable],
            "w_all": [float(w) for w in model["w"]], "hand_auc": E.auc(y, hand), "oof_auc": E.auc(y, oof),
            "fold_mean": float(np.nanmean(fold_aucs)), "fold_std": float(np.nanstd(fold_aucs)),
            "thr_medium": t_med, "thr_high": t_high}
    cal_json = {"version": 1, "created": time.strftime("%Y-%m-%d %H:%M:%S"), "developer": "Gourav Shrivastava",
                "n_videos": int(len(y)), "n_real": int((y == 0).sum()), "n_fake": int((y == 1).sum()),
                "features": info["features"], "mu": [float(m) for m in model["mu"]],
                "sd": [float(s) for s in model["sd"]], "w": [float(w) for w in model["w"]], "b": model["b"],
                "thresholds": {"medium": float(t_med), "high": float(t_high)},
                "cv": {"oof_auc": info["oof_auc"], "fold_mean": info["fold_mean"], "fold_std": info["fold_std"],
                       "hand_auc": info["hand_auc"]}}
    return info, cal_json


# =============================================================================
# Self-test dataset: synthetic clips with controllable pulse/blink/flicker ground truth
# =============================================================================
def make_selftest(root, n_per_class=8, seed=0):
    import cv2
    try:
        from skimage import data as skdata
    except ImportError:
        sys.exit("--selftest needs scikit-image for a real, Haar-detectable test face "
                 "(not in requirements.txt since the app itself never needs it): "
                 "pip install scikit-image")
    rng = np.random.default_rng(seed)
    face = skdata.astronaut()  # a real, Haar-detectable face -- procedural noise patches are NOT
    h, w = face.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    for lab in ("real", "fake"):
        d = os.path.join(root, lab)
        os.makedirs(d, exist_ok=True)
        for i in range(n_per_class):
            path = os.path.join(d, f"{lab}_{i:02d}.mp4")
            vw = cv2.VideoWriter(path, fourcc, 20, (w, h))
            for t in range(160):  # 8 seconds @ 20fps -- needs to clear pulse_analysis's
                                  # 12-sample minimum after 3.0 fps sampling (~24 samples here)
                frame = face.astype(np.int16).copy()
                if lab == "real":
                    # faint periodic green-channel modulation (proxy pulse signal) + normal
                    # camera-like noise, same face content and noise level every frame
                    pulse_mod = 2.5 * np.sin(2 * np.pi * 1.2 * (t / 20.0))
                    frame[:, :, 1] = np.clip(frame[:, :, 1].astype(np.float64) + pulse_mod, 0, 255)
                    frame += rng.integers(-3, 4, frame.shape)
                else:
                    # no periodic modulation at all (flat green channel) + an erratic patch
                    # dropped over roughly the face region every few frames (crude flicker proxy)
                    if t % 4 == 0:
                        fx, fy = w // 3, h // 6
                        frame[fy:fy + h // 2, fx:fx + w // 3] += rng.integers(-35, 35, (h // 2, w // 3, 3))
                    frame += rng.integers(-2, 3, frame.shape)
                vw.write(cv2.cvtColor(np.clip(frame, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
            vw.release()
    print(f"Built {2*n_per_class} synthetic test clips in {root}")


# =============================================================================
def main():
    ap = argparse.ArgumentParser(description="Evaluate and calibrate the video pipeline on a labelled dataset")
    ap.add_argument("--data")
    ap.add_argument("--out", default=os.path.join(HERE, "eval_video_out"))
    ap.add_argument("--workers", type=int, default=max(1, min(4, (os.cpu_count() or 2) - 1)))
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--reuse", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--l2", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--reset-calibration", action="store_true")
    args = ap.parse_args()

    if args.reset_calibration:
        if os.path.exists(V.CAL_PATH):
            os.remove(V.CAL_PATH)
            print("Removed", V.CAL_PATH)
        else:
            print("No video calibration file present.")
        return
    if args.apply:
        args.calibrate = True

    tmp_selftest = None
    if args.selftest:
        tmp_selftest = tempfile.mkdtemp(prefix="dfd_video_selftest_")
        make_selftest(tmp_selftest)
        args.data = tmp_selftest
        args.out = os.path.join(tmp_selftest, "out")
        V.CAL_PATH = os.path.join(tmp_selftest, "video_calibration.json")
        args.calibrate = True
        print("SELFTEST: synthetic clips. Numbers only prove the pipeline works, not real detection accuracy.\n")
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
            sys.exit("No videos found.")
        rows, extra["dupes"], extra["leaks"] = dedupe(rows)
        print(f"{len(rows)} unique videos. Analysing with {args.workers} worker(s) (this is slower than images)...")
        rows, errs = run_extraction(rows, args.workers)
        extra["errors"] = len(errs)
        if errs:
            with open(os.path.join(args.out, "errors.txt"), "w") as f:
                f.write("\n".join(f"{e['path']}: {e['error']}" for e in errs))
        write_csv(rows, feat_csv)

    y = np.array([r["label"] for r in rows])
    if len(set(y)) < 2:
        sys.exit("Need both real and fake videos.")
    extra["warnings"] = confound_warnings(rows, y)
    extra["mod_table"] = module_table(rows, y)

    if args.calibrate:
        if (len(y) < 30 or min((y == 0).sum(), (y == 1).sum()) < 10) and not args.force:
            print("\nCalibration skipped: need >= 30 videos and >= 10 per class (use --force to override).")
        else:
            info, cal_json = calibrate(rows, y, args)
            extra["cal"] = info
            if args.apply:
                with open(V.CAL_PATH, "w") as f:
                    json.dump(cal_json, f, indent=2)
                extra["applied"] = True
            with open(os.path.join(args.out, "calibration_candidate.json"), "w") as f:
                json.dump(cal_json, f, indent=2)

    report = build_report(rows, y, args, extra)
    with open(os.path.join(args.out, "report.md"), "w") as f:
        f.write(report)
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump({"n": int(len(y)), "hand_auc": E.auc(y, col(rows, "hand_score")),
                   "modules": extra["mod_table"], "calibration": extra.get("cal")},
                  f, indent=2, default=lambda o: None if o != o else float(o))
    print("\n" + report)
    print(f"\nFiles written to {args.out}: report.md, metrics.json, features.csv")
    if tmp_selftest:
        shutil.rmtree(tmp_selftest, ignore_errors=True)


if __name__ == "__main__":
    main()
