"""
Video pipeline for the Deepfake & Image Forensics Toolkit.
Developed by Gourav Shrivastava

Frame extraction, per-frame face tracking, and three temporal signals single frames
can't give you:
  - rPPG pulse: real skin shows a faint, periodic brightness modulation from blood flow
    (the basis of remote heart-rate sensors). Many synthesis pipelines don't reproduce
    it. Detects periodicity in the 42-240 bpm band; does not claim an exact heart rate.
  - Blink rate: needs mediapipe_face's landmarks (EAR). Without that model installed,
    this signal is reported unavailable rather than guessed from a face box.
  - Face-region flicker: frame-to-frame volatility in face sharpness/noise, normalised
    against the SAME video's background volatility so ordinary camera motion/compression
    doesn't trigger it on its own.
Frame-level forensics (ELA/noise/FFT/ML) is only run on a sampled subset of frames --
running the full image pipeline on every frame of a real clip is not practical.

Every signal here is heuristic and unvalidated against real deepfake video (see
DATASET_GUIDE.txt) -- treat the video verdict with at least as much caution as the
image one, more so given fewer real-world tests were possible while building this.
"""

import json
import os
import subprocess
import shutil

import cv2
import numpy as np
from PIL import Image

import forensics
import mediapipe_face
import mlclassifier

MAX_FRAMES = 90            # hard cap on frames pulled from the container, regardless of length
SAMPLE_FPS = 3.0           # target sampling rate for face/landmark tracking
ML_SAMPLE_EVERY = 6        # run the (slower) ML classifier on every Nth sampled frame
PULSE_BAND_HZ = (0.7, 4.0)  # 42-240 bpm


# ----------------------------------------------------------------------------
# Frame extraction
# ----------------------------------------------------------------------------
def probe(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        cap.release()
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    # OpenCV "opens" plenty of non-video files (a single image, a corrupt container) as a
    # 1-frame source, sometimes with a garbage/overflowed frame count. Treat anything that
    # doesn't look like a real multi-property video as "could not open", same as isOpened()
    # returning False -- callers already handle probe() returning None as that error path.
    if n <= 0 or w <= 0 or h <= 0 or not (0 < n < 10_000_000):
        return None
    duration = (n / fps) if fps > 0 else None
    return {"fps": round(fps, 3), "frame_count": n, "width": w, "height": h,
            "duration_s": round(duration, 2) if duration else None}


def extract_frames(path, max_frames=MAX_FRAMES, sample_fps=SAMPLE_FPS):
    """Evenly-spaced frames across the whole clip (not just the first N seconds)."""
    info = probe(path)
    if info is None or info["frame_count"] <= 0:
        return [], info
    step_by_rate = max(1, round(info["fps"] / sample_fps)) if info["fps"] > 0 else 1
    n_wanted = min(max_frames, max(1, info["frame_count"] // step_by_rate))
    idxs = np.linspace(0, info["frame_count"] - 1, n_wanted).astype(int)
    idxs = sorted(set(idxs.tolist()))

    cap = cv2.VideoCapture(path)
    frames = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, bgr = cap.read()
        if not ok:
            continue
        t = i / info["fps"] if info["fps"] > 0 else None
        frames.append({"index": int(i), "t": None if t is None else round(t, 3),
                       "image": Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))})
    cap.release()
    return frames, info


def extract_audio(path, out_path):
    """Best-effort mux-out of the audio track via ffmpeg CLI, for future audio analysis.
    Returns None (and leaves no file) if there's no audio track or ffmpeg is unavailable --
    this is optional and never blocks the video pipeline."""
    if not shutil.which("ffmpeg"):
        return None
    try:
        r = subprocess.run(["ffmpeg", "-y", "-i", path, "-vn", "-acodec", "pcm_s16le",
                            "-ar", "16000", "-ac", "1", out_path],
                           capture_output=True, timeout=60)
        return out_path if r.returncode == 0 and os.path.exists(out_path) else None
    except Exception:
        return None


# ----------------------------------------------------------------------------
# Per-frame face tracking (lightweight: box + optional landmarks, not full ELA/FFT)
# ----------------------------------------------------------------------------
def track_faces(frames):
    """Per sampled frame, keep the LARGEST detected face only (assumes a single subject;
    reasonable for the talking-head clips this kind of check targets). Uses mediapipe_face
    when available (adds EAR for blink detection), else Haar."""
    use_mp = mediapipe_face.AVAILABLE
    out = []
    for fr in frames:
        entry = {"index": fr["index"], "t": fr["t"], "face": None}
        try:
            if use_mp:
                faces = mediapipe_face.detect(fr["image"])
            else:
                faces = None
        except Exception:
            faces = None
        if faces:
            best = max(faces, key=lambda f: f["box"][2] * f["box"][3])
            entry["face"] = {"box": best["box"], "ear_left": best.get("ear_left"),
                             "ear_right": best.get("ear_right"), "detector": "mediapipe"}
        else:
            gray = cv2.cvtColor(np.asarray(fr["image"]), cv2.COLOR_RGB2GRAY)
            boxes = forensics._face_cascade().detectMultiScale(gray, 1.1, 6, minSize=(50, 50))
            if len(boxes):
                x, y, w, h = max(boxes, key=lambda b: b[2] * b[3])
                entry["face"] = {"box": [int(x), int(y), int(w), int(h)], "ear_left": None,
                                 "ear_right": None, "detector": "haar"}
        out.append(entry)
    return out


def _crop(img, box, margin=0.0):
    x, y, w, h = box
    W, H = img.width, img.height
    mx, my = int(w * margin), int(h * margin)
    x0, y0 = max(0, x - mx), max(0, y - my)
    x1, y1 = min(W, x + w + mx), min(H, y + h + my)
    if x1 <= x0 or y1 <= y0:
        return None
    return img.crop((x0, y0, x1, y1))


# ----------------------------------------------------------------------------
# Signal 1: rPPG pulse periodicity
# ----------------------------------------------------------------------------
def rppg_signal(frames, tracked):
    """Mean-G-channel signal from a shrunk central skin patch of the tracked face box,
    across all frames that have one. Requires evenly-sampled frames (uses actual
    timestamps, not frame count) to build the periodogram correctly."""
    ts, vals = [], []
    for fr, tr in zip(frames, tracked):
        if tr["face"] is None or fr["t"] is None:
            continue
        box = tr["face"]["box"]
        x, y, w, h = box
        # central 40% patch: avoids hair/background/eyes/mouth, closer to bare skin
        cx0, cy0 = x + int(w * 0.3), y + int(h * 0.25)
        cx1, cy1 = x + int(w * 0.7), y + int(h * 0.55)
        patch = np.asarray(fr["image"])[max(0, cy0):cy1, max(0, cx0):cx1]
        if patch.size == 0:
            continue
        vals.append(float(patch[..., 1].mean()))  # green channel
        ts.append(fr["t"])
    return np.array(ts), np.array(vals)


def pulse_analysis(ts, vals):
    empty = {"available": False, "periodicity": None, "dominant_bpm": None, "note": None}
    if len(ts) < 12:
        empty["note"] = f"Only {len(ts)} tracked frames with a face; need at least 12 for a pulse estimate."
        return empty
    duration = ts[-1] - ts[0]
    if duration < 3.0:
        empty["note"] = f"Tracked span is only {duration:.1f}s; need at least ~3s of continuous face tracking."
        return empty
    # resample to uniform time grid (frame sampling isn't perfectly even)
    fs = max(len(ts) / duration, 2.0 * PULSE_BAND_HZ[1])  # Nyquist-safe resample rate
    fs = min(fs, 15.0)
    grid = np.arange(ts[0], ts[-1], 1.0 / fs)
    if len(grid) < 16:
        empty["note"] = "Not enough samples after resampling."
        return empty
    sig = np.interp(grid, ts, vals)
    sig = sig - np.polyval(np.polyfit(grid, sig, 2), grid)  # detrend (remove slow lighting drift)
    win = np.hanning(len(sig))
    spec = np.abs(np.fft.rfft(sig * win))
    freqs = np.fft.rfftfreq(len(sig), d=1.0 / fs)
    band = (freqs >= PULSE_BAND_HZ[0]) & (freqs <= PULSE_BAND_HZ[1])
    if not band.any() or spec[band].sum() < 1e-9:
        empty["note"] = "No usable signal in the pulse frequency band."
        return empty
    band_power = spec[band]
    peak_i = np.argmax(band_power)
    peak_freq = freqs[band][peak_i]
    periodicity = float(band_power[peak_i] / (band_power.sum() + 1e-9))  # 0..1, how "peaky" the pulse band is
    return {"available": True, "periodicity": round(periodicity, 3),
            "dominant_bpm": round(float(peak_freq * 60), 1), "duration_s": round(float(duration), 1),
            "n_samples": len(ts), "note": None}


# ----------------------------------------------------------------------------
# Signal 2: blink rate (needs mediapipe EAR; unavailable otherwise, not guessed)
# ----------------------------------------------------------------------------
EAR_BLINK_THRESH = 0.19  # standard-ish EAR blink threshold; real faces vary, so treat loosely


def blink_analysis(frames, tracked):
    empty = {"available": False, "blinks": 0, "rate_per_min": None, "note": None}
    ears = [(fr["t"], (tr["face"]["ear_left"] + tr["face"]["ear_right"]) / 2)
            for fr, tr in zip(frames, tracked)
            if tr["face"] and tr["face"]["ear_left"] is not None and tr["face"]["ear_right"] is not None
            and fr["t"] is not None]
    if not ears:
        empty["note"] = ("No EAR data (mediapipe face model not installed, or no face tracked). "
                         "Install with get_facemesh.py to enable blink-rate checks.")
        return empty
    if len(ears) < 10:
        empty["note"] = f"Only {len(ears)} frames with eye landmarks; too few for a reliable blink count."
        return empty
    ts = np.array([t for t, _ in ears])
    vals = np.array([v for _, v in ears])
    duration = ts[-1] - ts[0]
    if duration < 2.0:
        empty["note"] = "Tracked span too short for a blink-rate estimate."
        return empty
    below = vals < EAR_BLINK_THRESH
    blinks = int(np.sum(below[1:] & ~below[:-1]))  # rising edge INTO a blink
    rate = blinks / (duration / 60.0)
    return {"available": True, "blinks": blinks, "rate_per_min": round(rate, 1),
            "duration_s": round(float(duration), 1), "mean_ear": round(float(vals.mean()), 3), "note": None}


# ----------------------------------------------------------------------------
# Signal 3: face-region flicker, normalised against this video's own background
# ----------------------------------------------------------------------------
def flicker_analysis(frames, tracked):
    empty = {"available": False, "face_volatility": None, "bg_volatility": None, "ratio": None, "note": None}
    face_sharp, bg_sharp = [], []
    for fr, tr in zip(frames, tracked):
        if tr["face"] is None:
            continue
        gray = cv2.cvtColor(np.asarray(fr["image"]), cv2.COLOR_RGB2GRAY)
        x, y, w, h = tr["face"]["box"]
        x, y = max(0, x), max(0, y)
        w, h = min(gray.shape[1] - x, w), min(gray.shape[0] - y, h)
        if w < 8 or h < 8:
            continue
        face_sharp.append(float(cv2.Laplacian(gray[y:y + h, x:x + w], cv2.CV_32F).var()))
        mask = np.ones(gray.shape, bool)
        mask[y:y + h, x:x + w] = False
        if mask.sum() > 100:
            # Laplacian of the FULL frame, then variance over only the non-face pixels --
            # NOT a whole-frame Laplacian (that would mix the face region's own volatility
            # back into the "background" measurement and defeat the normalisation below).
            lap_full = cv2.Laplacian(gray, cv2.CV_32F)
            bg_sharp.append(float(lap_full[mask].var()))
    if len(face_sharp) < 6:
        empty["note"] = f"Only {len(face_sharp)} tracked frames; need at least 6 for a flicker estimate."
        return empty
    face_sharp = np.array(face_sharp)
    jerk = np.abs(np.diff(face_sharp, n=2))  # second difference: how erratically sharpness changes frame to frame
    face_vol = float(jerk.mean() / (face_sharp.mean() + 1e-6))
    bg_vol = None
    ratio = None
    if len(bg_sharp) >= 6:
        bg_sharp = np.array(bg_sharp)
        bg_jerk = np.abs(np.diff(bg_sharp, n=2))
        bg_vol = float(bg_jerk.mean() / (bg_sharp.mean() + 1e-6))
        ratio = float(face_vol / (bg_vol + 1e-6))
    return {"available": True, "face_volatility": round(face_vol, 4),
            "bg_volatility": None if bg_vol is None else round(bg_vol, 4),
            "ratio": None if ratio is None else round(ratio, 2), "note": None}


# ----------------------------------------------------------------------------
# Signal 4: temporal ML (sampled frames through mlclassifier)
# ----------------------------------------------------------------------------
def ml_temporal(frames, tracked, every=ML_SAMPLE_EVERY):
    empty = {"available": False, "n_scored": 0, "mean": None, "max": None, "std": None, "per_frame": []}
    if not mlclassifier.list_models():
        return empty
    per_frame = []
    for i, (fr, tr) in enumerate(zip(frames, tracked)):
        if i % every:
            continue
        boxes = [tuple(tr["face"]["box"])] if tr["face"] else None
        try:
            res = mlclassifier.classify(fr["image"], boxes)
        except Exception:
            continue
        if res["available"]:
            per_frame.append({"index": fr["index"], "t": fr["t"], "score": res["score"]})
    if not per_frame:
        return empty
    scores = np.array([p["score"] for p in per_frame])
    return {"available": True, "n_scored": len(scores), "mean": round(float(scores.mean()), 4),
            "max": round(float(scores.max()), 4), "std": round(float(scores.std()), 4), "per_frame": per_frame}


# ----------------------------------------------------------------------------
# Verdict
# ----------------------------------------------------------------------------
def clamp01(x):
    return float(max(0.0, min(1.0, x)))


CAL_PATH = os.environ.get("DFD_VIDEO_CALIBRATION",
                          os.path.join(os.path.dirname(os.path.abspath(__file__)), "video_calibration.json"))
_CAL = "unset"


def _load_calibration():
    try:
        with open(CAL_PATH) as f:
            c = json.load(f)
        if c.get("version") == 1 and c.get("features") and len(c["features"]) == len(c["w"]):
            return c
    except Exception:
        pass
    return None


def set_calibration(cal):
    global _CAL
    _CAL = cal


def get_calibration():
    global _CAL
    if _CAL == "unset":
        _CAL = _load_calibration()
    return _CAL


def compute_video_verdict(pulse, blink, flicker, ml, n_frames, n_tracked):
    parts, reasons = {}, []
    if n_tracked == 0:
        return {"score": 0.0, "percent": 0, "label": "LOW",
                "description": "No face was tracked in any sampled frame; none of the video-specific "
                               "checks could run.", "module_scores": {}, "reasons": [], "calibration": None,
                "explanation": None}

    if pulse["available"]:
        # low periodicity in the pulse band -- no clear heartbeat signal. Weak on its own:
        # compression, lighting and skin tone all affect this; never treated as a hard cutoff.
        parts["pulse"] = clamp01((0.25 - pulse["periodicity"]) / 0.20)
        if parts["pulse"] > 0.4:
            reasons.append(f"No clear pulse periodicity detected (band-power ratio {pulse['periodicity']}). "
                           f"Real skin under stable lighting usually shows a weak but detectable peak; "
                           f"absence is a soft signal, not proof.")

    if blink["available"]:
        minutes = blink["duration_s"] / 60.0
        if minutes >= 0.15:  # ~9s+ needed before "no blinks" means anything
            if blink["blinks"] == 0:
                parts["blink"] = 0.55
                reasons.append(f"Zero blinks detected over {blink['duration_s']}s of tracked eyes. Normal "
                               f"human blink rate is roughly 12-20/min; total absence over several seconds "
                               f"is a known (but not definitive) synthesis tell.")
            elif blink["rate_per_min"] < 4:
                parts["blink"] = 0.3
                reasons.append(f"Unusually low blink rate ({blink['rate_per_min']}/min vs a normal ~12-20/min).")

    if flicker["available"] and flicker["ratio"] is not None:
        parts["flicker"] = clamp01((flicker["ratio"] - 1.8) / 3.0)
        if parts["flicker"] > 0.4:
            reasons.append(f"Face region is {flicker['ratio']}x more frame-to-frame volatile than the rest "
                           f"of the video (sharpness jitter). Can indicate an inconsistently blended/rendered "
                           f"face, but fast motion or autofocus hunting can also cause this.")

    if ml["available"]:
        parts["ml"] = ml["mean"]
        if ml["mean"] >= 0.75:
            reasons.append(f"Local ML classifier: mean P(fake) {ml['mean']} across {ml['n_scored']} sampled "
                           f"frames.")
        if ml["std"] is not None and ml["std"] > 0.3 and ml["n_scored"] >= 4:
            reasons.append(f"ML score is highly inconsistent frame to frame (std {ml['std']}). Could mean "
                           f"the face is only sometimes flagged (worth reviewing those frames), or that the "
                           f"model is unreliable on this footage.")

    if not parts:
        return {"score": 0.0, "percent": 0, "label": "LOW",
                "description": "A face was tracked, but none of the temporal checks had enough data to run "
                               "(short clip, or no MediaPipe/ML model installed). No video-specific signal.",
                "module_scores": {}, "reasons": [], "calibration": None, "explanation": None}

    weights = {"pulse": 0.30, "blink": 0.20, "flicker": 0.20, "ml": 0.45}
    active = list(parts.keys())
    total_w = sum(weights[k] for k in active)
    weighted = sum(weights[k] * parts[k] for k in active) / total_w

    # Same inline-explanation pattern as forensics.py's compute_verdict: build the breakdown
    # from the exact variables the real computation uses, so it cannot drift from the score.
    baseline_terms = [{"feature": k, "value": round(parts[k], 4), "weight": weights[k],
                       "share": round(weights[k] / total_w, 4),
                       "contribution": round(weights[k] * parts[k] / total_w, 4)} for k in active]
    override_steps = []
    if ml.get("available") and ml["mean"] >= 0.9:
        pre = weighted
        binding = 0.8 > pre
        if binding:
            weighted = 0.8
        reason = f"Local ML classifier strongly indicates fake (mean P(fake) {ml['mean']:.2f} across sampled frames)"
        override_steps.append({"name": "ml_strong", "floor": 0.8, "pre_value": round(pre, 4),
                               "binding": binding, "reason": reason})
        reasons.append(reason)

    heuristic_final = weighted
    hi_t, med_t = 0.6, 0.32
    calibrated, calibrated_terms = None, None
    cal = get_calibration()
    if cal and all(f in parts for f in cal["features"]):
        terms = [(f, w, parts[f], mu, sd, w * (parts[f] - mu) / sd)
                for f, w, mu, sd in zip(cal["features"], cal["w"], cal["mu"], cal["sd"])]
        z = cal["b"] + sum(t[5] for t in terms)
        weighted = float(1.0 / (1.0 + np.exp(-z)))
        hi_t, med_t = cal["thresholds"]["high"], cal["thresholds"]["medium"]
        calibrated = {"n_videos": cal.get("n_videos"), "cv_auc": cal.get("cv", {}).get("oof_auc")}
        calibrated_terms = [{"feature": f, "value": round(x, 4), "weight": round(w, 4),
                             "mu": round(mu, 4), "sd": round(sd, 4), "contribution": round(c, 4)}
                            for f, w, x, mu, sd, c in sorted(terms, key=lambda t: -abs(t[5]))]

    label = "HIGH" if weighted >= hi_t else "MEDIUM" if weighted >= med_t else "LOW"
    desc = {"HIGH": "Strong indicators across the video-specific checks.",
           "MEDIUM": "Some anomalies across frames. Manual review recommended.",
           "LOW": "No strong video-specific indicators found."}[label]
    explanation = {"mode": "calibrated" if calibrated else "heuristic", "baseline_terms": baseline_terms,
                   "override_steps": override_steps, "heuristic_score": round(heuristic_final, 4),
                   "calibrated_terms": calibrated_terms, "calibrated_baseline": cal["b"] if calibrated else None,
                   "final_score": round(weighted, 4)}
    return {"score": round(weighted, 3), "percent": int(round(weighted * 100)), "label": label,
            "description": desc, "calibration": calibrated, "explanation": explanation,
            "module_scores": {k: round(v, 3) for k, v in parts.items()}, "reasons": reasons}


# ----------------------------------------------------------------------------
# Visual outputs
# ----------------------------------------------------------------------------
def save_contact_sheet(frames, tracked, out_path, cols=6):
    if not frames:
        return None
    thumb_w = 160
    thumbs = []
    for fr, tr in zip(frames, tracked):
        im = fr["image"].copy()
        arr = np.array(im)  # np.array() copies (writable); np.asarray() would wrap the PIL
                            # buffer read-only and crash cv2.rectangle below
        if tr["face"]:
            x, y, w, h = tr["face"]["box"]
            cv2.rectangle(arr, (x, y), (x + w, y + h), (0, 255, 0) if tr["face"]["detector"] == "mediapipe"
                          else (0, 180, 255), 2)
        im = Image.fromarray(arr)
        ratio = thumb_w / im.width
        im = im.resize((thumb_w, int(im.height * ratio)))
        thumbs.append(im)
    rows = (len(thumbs) + cols - 1) // cols
    th = max(t.height for t in thumbs)
    sheet = Image.new("RGB", (cols * thumb_w, rows * th), (20, 20, 20))
    for i, t in enumerate(thumbs):
        sheet.paste(t, ((i % cols) * thumb_w, (i // cols) * th))
    sheet.save(out_path, quality=85)
    return out_path


def save_signal_plot(out_path, ts, vals, title, ylabel):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 2.2), dpi=110)
    ax.plot(ts, vals, linewidth=1)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("seconds", fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


# ----------------------------------------------------------------------------
def analyze_video(path, case_dir, fast=False):
    """fast=True skips contact-sheet/plot generation (used by evaluate_video.py for batch runs)."""
    out = os.path.join(case_dir, "video")
    os.makedirs(out, exist_ok=True)
    frames, info = extract_frames(path)
    if info is None:
        return {"error": "Could not open this file as a video (unsupported or corrupt container/codec)."}
    if not frames:
        return {"error": "Video opened but no frames could be read.", "info": info}

    tracked = track_faces(frames)
    n_tracked = sum(1 for t in tracked if t["face"] is not None)
    detector = next((t["face"]["detector"] for t in tracked if t["face"]), None)

    ts, rvals = rppg_signal(frames, tracked)
    pulse = pulse_analysis(ts, rvals)
    blink = blink_analysis(frames, tracked)
    flicker = flicker_analysis(frames, tracked)
    ml = ml_temporal(frames, tracked)

    contact_sheet = pulse_plot = ml_plot = None
    if not fast:
        contact_sheet = save_contact_sheet(frames, tracked, os.path.join(out, "contact_sheet.jpg"))
        if pulse["available"]:
            pulse_plot = save_signal_plot(os.path.join(out, "pulse_signal.png"), ts, rvals,
                                          "Face-patch green-channel signal (rPPG input)", "mean G")
        if ml["available"] and len(ml["per_frame"]) >= 2:
            ml_plot = save_signal_plot(os.path.join(out, "ml_scores.png"),
                                       [p["t"] for p in ml["per_frame"]], [p["score"] for p in ml["per_frame"]],
                                       "Per-frame ML P(fake)", "P(fake)")

    verdict = compute_video_verdict(pulse, blink, flicker, ml, len(frames), n_tracked)
    return {"info": info, "n_frames_sampled": len(frames), "n_frames_tracked": n_tracked,
            "detector": detector, "pulse": pulse, "blink": blink, "flicker": flicker, "ml": ml,
            "contact_sheet": os.path.basename(contact_sheet) if contact_sheet else None,
            "pulse_plot": os.path.basename(pulse_plot) if pulse_plot else None,
            "ml_plot": os.path.basename(ml_plot) if ml_plot else None, "verdict": verdict}
