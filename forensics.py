"""
Analysis engine for the Deepfake & Image Forensics Toolkit.
Developed by Gourav Shrivastava

Every module here runs locally, with no API and no ML model download.
Results are HEURISTIC indicators, not proof of manipulation.
"""

import hashlib
import io
import os
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor

import cv2
import exifread
import imagehash
import numpy as np
from PIL import Image, ImageEnhance, ImageOps

import mlclassifier
import c2pa_verify
import mediapipe_face

ANALYSIS_MAX_SIDE = 1024  # cap for heavy operations (speed + stable thresholds)

AI_MARKERS = [
    b"stable diffusion", b"stablediffusion", b"midjourney", b"dall-e", b"dall\xc2\xb7e",
    b"comfyui", b"automatic1111", b"novelai", b"leonardo.ai", b"adobe firefly",
    b"firefly", b"invokeai", b"sdxl", b"runwayml", b"ideogram", b"flux.1",
    b"negative prompt", b"cfg scale", b"sampler:", b"steps:",
]
EDITOR_MARKERS = ["photoshop", "gimp", "lightroom", "snapseed", "pixelmator",
                  "affinity", "paint.net", "canva", "facetune", "faceapp", "remini"]


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def load_working_image(path):
    """Return (PIL RGB image capped at ANALYSIS_MAX_SIDE, original size)."""
    im = Image.open(path)
    orig = im.size
    im = ImageOps.exif_transpose(im).convert("RGB")
    im.thumbnail((ANALYSIS_MAX_SIDE, ANALYSIS_MAX_SIDE), Image.LANCZOS)
    return im, orig


def save_arr(arr, outdir, name):
    fn = f"{name}.png"
    Image.fromarray(arr).save(os.path.join(outdir, fn))
    return fn


def clamp01(x):
    return float(max(0.0, min(1.0, x)))


# ----------------------------------------------------------------------------
# 1. Hashes and file structure
# ----------------------------------------------------------------------------
def compute_hashes(path):
    b = open(path, "rb").read()
    return {"md5": hashlib.md5(b).hexdigest(), "sha1": hashlib.sha1(b).hexdigest(),
            "sha256": hashlib.sha256(b).hexdigest(), "size": len(b)}


def compute_img_hashes(path):
    im = Image.open(path)
    return {"phash": str(imagehash.phash(im)), "ahash": str(imagehash.average_hash(im)),
            "dhash": str(imagehash.dhash(im)), "whash": str(imagehash.whash(im))}


def file_structure(path):
    """Detect real type by magic bytes and any data appended after the image ends."""
    data = open(path, "rb").read()
    res = {"real_type": "unknown", "appended_bytes": 0, "notes": []}
    if data.startswith(b"\xff\xd8\xff"):
        res["real_type"] = "JPEG"
        end = data.rfind(b"\xff\xd9")
        if end != -1 and len(data) - (end + 2) > 0:
            res["appended_bytes"] = len(data) - (end + 2)
    elif data.startswith(b"\x89PNG\r\n\x1a\n"):
        res["real_type"] = "PNG"
        end = data.rfind(b"IEND")
        if end != -1 and len(data) - (end + 8) > 0:
            res["appended_bytes"] = len(data) - (end + 8)
    elif data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        res["real_type"] = "WEBP"
    elif data[:6] in (b"GIF87a", b"GIF89a"):
        res["real_type"] = "GIF"
    elif data[:2] == b"BM":
        res["real_type"] = "BMP"
    elif data[:4] in (b"II*\x00", b"MM\x00*"):
        res["real_type"] = "TIFF"
    ext = os.path.splitext(path)[1].lower().lstrip(".")
    ext_map = {"jpg": "JPEG", "jpeg": "JPEG", "png": "PNG", "webp": "WEBP",
               "gif": "GIF", "bmp": "BMP", "tif": "TIFF", "tiff": "TIFF"}
    if ext_map.get(ext) and res["real_type"] != "unknown" and ext_map[ext] != res["real_type"]:
        res["notes"].append(f"Extension .{ext} does not match real type {res['real_type']}")
    if res["appended_bytes"] > 16:
        res["notes"].append(f"{res['appended_bytes']} bytes appended after image end (possible hidden data)")
    return res


# ----------------------------------------------------------------------------
# 2. Metadata and provenance
# ----------------------------------------------------------------------------
def metadata_analysis(path):
    findings, exif = [], {}
    with open(path, "rb") as f:
        try:
            tags = exifread.process_file(f, details=False)
        except Exception:
            tags = {}
    for k, v in tags.items():
        if k.startswith("Thumbnail") or k == "JPEGThumbnail":
            continue
        exif[k] = str(v)[:200]

    raw = open(path, "rb").read()
    head = raw[:2_000_000].lower()
    hits = {m.decode("latin-1").strip(":") for m in AI_MARKERS if m in head}
    weak = {"steps", "sampler", "cfg scale", "negative prompt", "firefly", "sdxl"}
    strong_hits = hits - weak
    # weak markers are generic words, so require at least two of them together
    ai_hits = sorted(strong_hits | (hits & weak if len(hits & weak) >= 2 else set()))

    # PNG text chunks often hold generator parameters
    png_text = {}
    try:
        im = Image.open(path)
        for k, v in (im.info or {}).items():
            if isinstance(v, str) and k.lower() in ("parameters", "prompt", "workflow", "software",
                                                     "description", "comment", "generation_time"):
                png_text[k] = v[:300]
    except Exception:
        pass

    software = exif.get("Image Software", "")
    make, model = exif.get("Image Make"), exif.get("Image Model")
    editors = [e for e in EDITOR_MARKERS if e in software.lower()]

    score = 0.0
    if ai_hits:
        score += 0.8
        findings.append(f"AI-generation / provenance markers found: {', '.join(ai_hits)}")
    if png_text.get("parameters") or png_text.get("workflow"):
        score = max(score, 0.9)
        findings.append("PNG text chunk contains generation parameters (typical of Stable Diffusion UIs)")
    if editors:
        score += 0.25
        findings.append(f"Editing software recorded in EXIF: {software}")
    if not exif:
        score += 0.15
        findings.append("No EXIF at all (stripped by social media/editing, or synthetic). Weak signal.")
    elif not (make or model):
        score += 0.1
        findings.append("EXIF present but no camera make/model. Weak signal.")
    dt, dto = exif.get("Image DateTime"), exif.get("EXIF DateTimeOriginal")
    if dt and dto and dt != dto:
        score += 0.15
        findings.append(f"Modified time ({dt}) differs from capture time ({dto})")
    return {"exif": exif, "png_text": png_text, "ai_markers": ai_hits, "findings": findings,
            "score": clamp01(score)}


# ----------------------------------------------------------------------------
# 3. Error Level Analysis
# ----------------------------------------------------------------------------
def ela_analysis(im, outdir, quality=90):
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality)
    buf.seek(0)
    resaved = Image.open(buf).convert("RGB")
    diff = np.abs(np.asarray(im, dtype=np.int16) - np.asarray(resaved, dtype=np.int16)).astype(np.float32)
    gray = diff.mean(axis=2)
    scale = 255.0 / max(gray.max(), 1.0)
    vis = np.clip(diff * scale, 0, 255).astype(np.uint8)

    bs = 32
    h, w = gray.shape
    blocks = [gray[y:y + bs, x:x + bs].mean()
              for y in range(0, h - bs + 1, bs) for x in range(0, w - bs + 1, bs)]
    blocks = np.array(blocks) if blocks else np.array([gray.mean()])
    med = float(np.median(blocks)) + 1e-6
    ratio = float(np.percentile(blocks, 98) / med)
    cv = float(blocks.std() / (blocks.mean() + 1e-6))
    score = clamp01((ratio - 2.0) / 5.0)
    return {"file": save_arr(vis, outdir, "ela"), "mean": round(float(gray.mean()), 3),
            "hotspot_ratio": round(ratio, 2), "variation": round(cv, 2), "score": score,
            "note": "Uniform ELA = consistent compression history. Bright isolated regions = "
                    "possible splice/edit. PNG/screenshots and heavily recompressed images give unreliable ELA."}


# ----------------------------------------------------------------------------
# 4. Noise residual consistency
# ----------------------------------------------------------------------------
def noise_analysis(im, outdir):
    gray = np.asarray(im.convert("L"), dtype=np.float32)
    resid = gray - cv2.GaussianBlur(gray, (0, 0), 1.5)
    bs = 32
    h, w = gray.shape
    ys, xs = range(0, h - bs + 1, bs), range(0, w - bs + 1, bs)
    grid = np.array([[resid[y:y + bs, x:x + bs].std() for x in xs] for y in ys])
    if grid.size == 0:
        return {"file": None, "score": 0.0, "note": "Image too small"}
    # ignore near-flat blocks (sky, walls) which always have low noise
    active = grid[grid > 0.5]
    cv = float(active.std() / (active.mean() + 1e-6)) if active.size > 4 else 0.0
    norm = (grid - grid.min()) / (np.ptp(grid) + 1e-6)
    heat = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    heat = cv2.resize(heat, (w, h), interpolation=cv2.INTER_NEAREST)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    mean_noise = float(active.mean()) if active.size else 0.0
    score = clamp01((cv - 0.45) / 0.8)
    if mean_noise < 1.0:
        score = max(score, 0.35)  # unnaturally clean / denoised / synthetic-smooth
    return {"file": save_arr(heat, outdir, "noise_map"), "mean_noise": round(mean_noise, 2),
            "variation": round(cv, 2), "score": score,
            "note": "Real camera noise is fairly uniform. Pasted or generated regions often have a "
                    "different noise level. Very low overall noise suggests denoising or synthesis."}


# ----------------------------------------------------------------------------
# 5. Frequency spectrum (GAN / upsampling artifacts)
# ----------------------------------------------------------------------------
def frequency_analysis(im, outdir):
    gray = np.asarray(im.convert("L").resize((512, 512)), dtype=np.float32)
    win = np.outer(np.hanning(512), np.hanning(512))
    spec = np.fft.fftshift(np.fft.fft2((gray - gray.mean()) * win))
    mag = np.log1p(np.abs(spec))

    yy, xx = np.indices(mag.shape)
    r = np.hypot(yy - 256, xx - 256).astype(int)
    radial = np.bincount(r.ravel(), mag.ravel()) / np.maximum(np.bincount(r.ravel()), 1)
    whitened = mag - radial[r]
    mask = (r > 20) & (r < 250)
    z = whitened[mask]
    peak_z = float((z.max() - z.mean()) / (z.std() + 1e-6))
    n_peaks = int((whitened[mask] > z.mean() + 6 * z.std()).sum())

    hf = float(mag[r > 128].mean() / (mag[r <= 128].mean() + 1e-6))
    vis = ((mag - mag.min()) / (np.ptp(mag) + 1e-6) * 255).astype(np.uint8)
    vis = cv2.cvtColor(cv2.applyColorMap(vis, cv2.COLORMAP_MAGMA), cv2.COLOR_BGR2RGB)
    score = clamp01((peak_z - 9.0) / 8.0 + min(n_peaks, 30) / 120.0)
    return {"file": save_arr(vis, outdir, "fft_spectrum"), "peak_z": round(peak_z, 2),
            "peak_count": n_peaks, "hf_ratio": round(hf, 3), "score": score,
            "note": "Isolated bright dots or a regular grid in the spectrum indicate periodic "
                    "upsampling artifacts common in GAN/diffusion output. Also present in resized or "
                    "sharpened real photos, so treat as supporting evidence only."}


# ----------------------------------------------------------------------------
# 6. Copy-move (clone) detection
# ----------------------------------------------------------------------------
def copy_move_analysis(im, outdir):
    gray = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2GRAY)
    orb = cv2.ORB_create(nfeatures=3000)
    try:
        kps, des = orb.detectAndCompute(gray, None)
    except cv2.error:
        # ORB's internal image pyramid chokes on extremely tiny inputs (e.g. 1x1px) with an
        # unhandled OpenCV assertion. Treat that the same as "no keypoints found" rather than
        # letting it crash the whole analysis.
        kps, des = [], None
    vis = np.asarray(im).copy()
    pairs = 0
    if des is not None and len(kps) > 20:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        matches = bf.knnMatch(des, des, k=3)
        vecs = []
        for m in matches:
            cand = [x for x in m if x.queryIdx != x.trainIdx]
            if len(cand) < 2:
                continue
            a, b = cand[0], cand[1]
            if a.distance < 0.6 * b.distance and a.distance < 40:
                p1, p2 = kps[a.queryIdx].pt, kps[a.trainIdx].pt
                if np.hypot(p1[0] - p2[0], p1[1] - p2[1]) > 40:
                    vecs.append((p1, p2))
        # keep only pairs whose displacement vectors agree (true cloned block)
        if vecs:
            d = np.array([(p2[0] - p1[0], p2[1] - p1[1]) for p1, p2 in vecs])
            d = np.where((d[:, [0]] < 0), -d, d)  # sign-normalise (symmetric matches)
            rounded = np.round(d / 8).astype(int)
            uniq, counts = np.unique(rounded, axis=0, return_counts=True)
            best = uniq[counts.argmax()]
            keep = [i for i in range(len(vecs)) if (rounded[i] == best).all()]
            pairs = len(keep)
            if pairs >= 4:
                for i in keep:
                    p1, p2 = vecs[i]
                    cv2.line(vis, tuple(map(int, p1)), tuple(map(int, p2)), (255, 60, 60), 1)
                    cv2.circle(vis, tuple(map(int, p1)), 3, (60, 255, 60), -1)
    score = clamp01((pairs - 4) / 20.0)
    return {"file": save_arr(vis, outdir, "copy_move"), "matched_pairs": pairs, "score": score,
            "note": "Consistent duplicate regions inside the same image (used to remove or duplicate "
                    "objects). Repetitive textures such as bricks or leaves can cause false positives."}


# ----------------------------------------------------------------------------
# 7. Face-region consistency (core deepfake indicator)
# ----------------------------------------------------------------------------
_cascade = None


def _face_cascade():
    global _cascade
    if _cascade is None:
        _cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    return _cascade


def _dev(r):
    return abs(np.log(max(r, 1e-3)))


def _region_stats(mask, lap, resid, ela):
    if mask.sum() < 4:
        return None
    return float(lap[mask].var()), float(resid[mask].std()) + 1e-9, float(ela[mask].mean()) + 1e-9


def _clip_box(box, W, H):
    x, y, w, h = box
    x1, y1 = x + w, y + h  # true right/bottom edge BEFORE the origin gets clipped
    x, y = max(0, x), max(0, y)
    # width/height must shrink by whatever the origin was clipped by, not just cap at W/H --
    # otherwise a box whose x0/y0 was negative (face near the left/top edge) keeps its full
    # original width/height and overshoots past its own right/bottom edge into background.
    w, h = max(1, min(x1, W) - x), max(1, min(y1, H) - y)
    return x, y, w, h


def _face_analysis_haar(im, rgb, gray, lap, resid, ela, outdir):
    faces = _face_cascade().detectMultiScale(gray, 1.1, 6, minSize=(60, 60))
    vis = rgb.copy()
    results, worst = [], 0.0
    for (x, y, w, h) in faces:
        m = np.zeros(gray.shape, bool)
        m[y:y + h, x:x + w] = True
        bg = ~m
        f_sharp, b_sharp = float(lap[m].var()), float(lap[bg].var()) + 1e-6
        f_noise, b_noise = float(resid[m].std()), float(resid[bg].std()) + 1e-6
        f_ela, b_ela = float(ela[m].mean()), float(ela[bg].mean()) + 1e-6
        sharp_r, noise_r, ela_r = f_sharp / b_sharp, f_noise / b_noise, f_ela / b_ela
        s = clamp01((0.5 * _dev(sharp_r) + 0.8 * _dev(noise_r) + 0.7 * _dev(ela_r)) / 1.6 - 0.25)

        face = gray[y:y + h, x:x + w].astype(np.float32)
        half = w // 2
        lm, rm = face[:, :half].mean(), face[:, w - half:].mean()
        asym = abs(lm - rm) / (face.mean() + 1e-6)

        worst = max(worst, s)
        results.append({"box": [int(x), int(y), int(w), int(h)], "detector": "haar",
                        "sharpness_ratio": round(sharp_r, 2), "noise_ratio": round(noise_r, 2),
                        "ela_ratio": round(ela_r, 2), "lr_brightness_asym": round(float(asym), 2),
                        "score": round(s, 2)})
        col = (255, 60, 60) if s > 0.5 else (60, 255, 60)
        cv2.rectangle(vis, (x, y), (x + w, y + h), col, 2)
        cv2.putText(vis, f"{s:.2f}", (x, max(y - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
    return {"file": save_arr(vis, outdir, "faces"), "faces": results, "count": len(results), "score": worst,
            "detector": "haar",
            "note": "Haar-cascade face detection (frontal-only; profile, tilted or small faces are missed). "
                    "Compares each face with the rest of the image (sharpness, noise, compression). "
                    "Install the MediaPipe face model (get_facemesh.py) for rotation-aware detection and "
                    "eye/mouth-level checks."}


def _face_analysis_mediapipe(im, rgb, gray, lap, resid, ela, outdir, mp_faces):
    H, W = gray.shape
    vis = rgb.copy()
    results, worst = [], 0.0
    for mf in mp_faces:
        x, y, w, h = _clip_box(mf["box"], W, H)
        m = np.zeros(gray.shape, bool)
        m[y:y + h, x:x + w] = True
        bg = ~m
        f_sharp, b_sharp = float(lap[m].var()), float(lap[bg].var()) + 1e-6
        f_noise, b_noise = float(resid[m].std()), float(resid[bg].std()) + 1e-6
        f_ela, b_ela = float(ela[m].mean()), float(ela[bg].mean()) + 1e-6
        sharp_r, noise_r, ela_r = f_sharp / b_sharp, f_noise / b_noise, f_ela / b_ela
        base = clamp01((0.5 * _dev(sharp_r) + 0.8 * _dev(noise_r) + 0.7 * _dev(ela_r)) / 1.6 - 0.25)

        # Eye/mouth sub-regions vs the rest of THIS face (where blending artifacts concentrate)
        face_mask = m.copy()
        region_dev, region_detail = 0.0, {}
        for rname, rbox in mf["regions"].items():
            rx, ry, rw, rh = _clip_box(rbox, W, H)
            rmask = np.zeros(gray.shape, bool)
            rmask[ry:ry + rh, rx:rx + rw] = True
            rest = face_mask & ~rmask
            rs = _region_stats(rmask, lap, resid, ela)
            fs = _region_stats(rest, lap, resid, ela)
            if rs is None or fs is None:
                continue
            r_ela_ratio = rs[2] / fs[2]
            r_noise_ratio = rs[1] / fs[1]
            d = 0.6 * _dev(r_ela_ratio) + 0.5 * _dev(r_noise_ratio)
            region_detail[rname] = {"ela_ratio": round(r_ela_ratio, 2), "noise_ratio": round(r_noise_ratio, 2)}
            region_dev = max(region_dev, d)
        region_score = clamp01(region_dev / 1.6 - 0.25)

        # Geometric tells (weak/supplementary -- see mediapipe_face.py docstring for caveats)
        ear_a = mf.get("ear_asymmetry")
        ear_score = clamp01((ear_a - 0.3) / 1.2) if ear_a is not None else 0.0
        sym = mf.get("symmetry_dev")
        sym_score = clamp01((sym - 0.05) / 0.15) if sym is not None else 0.0

        s = clamp01(0.55 * base + 0.30 * region_score + 0.08 * ear_score + 0.07 * sym_score)
        worst = max(worst, s)
        results.append({"box": [x, y, w, h], "detector": "mediapipe",
                        "sharpness_ratio": round(sharp_r, 2), "noise_ratio": round(noise_r, 2),
                        "ela_ratio": round(ela_r, 2), "region_scores": region_detail,
                        "roll_deg": mf.get("roll_deg"), "ear_left": mf.get("ear_left"),
                        "ear_right": mf.get("ear_right"), "ear_asymmetry": ear_a, "symmetry_dev": sym,
                        "score": round(s, 2)})
        col = (255, 60, 60) if s > 0.5 else (60, 255, 60)
        cv2.rectangle(vis, (x, y), (x + w, y + h), col, 2)
        for rbox in mf["regions"].values():
            rx, ry, rw, rh = _clip_box(rbox, W, H)
            cv2.rectangle(vis, (rx, ry), (rx + rw, ry + rh), (255, 200, 0), 1)
        cv2.putText(vis, f"{s:.2f}", (x, max(y - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
    return {"file": save_arr(vis, outdir, "faces"), "faces": results, "count": len(results), "score": worst,
            "detector": "mediapipe",
            "note": "MediaPipe landmark detection: rotation-aware face box plus separate eye/mouth region "
                    "checks (where face-swap/GAN blending artifacts concentrate), eye-openness (EAR) "
                    "left/right asymmetry, and roll-corrected face symmetry. The geometric checks (EAR "
                    "asymmetry, symmetry) are weak signals: a real turned or expressive face can trigger "
                    "them too, so they contribute only a small share of the score."}


def face_analysis(im, outdir):
    rgb = np.asarray(im)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    resid = gray.astype(np.float32) - cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 1.5)
    ela_buf = io.BytesIO()
    im.save(ela_buf, "JPEG", quality=90)
    ela_buf.seek(0)
    ela = np.abs(rgb.astype(np.int16) - np.asarray(Image.open(ela_buf).convert("RGB"), dtype=np.int16)).mean(axis=2)

    mp_faces = None
    try:
        mp_faces = mediapipe_face.detect(im)
    except Exception:
        mp_faces = None  # any runtime failure falls back to Haar rather than breaking the report
    if mp_faces is not None:
        return _face_analysis_mediapipe(im, rgb, gray, lap, resid, ela, outdir, mp_faces)
    return _face_analysis_haar(im, rgb, gray, lap, resid, ela, outdir)


# ----------------------------------------------------------------------------
# 8. JPEG quantisation / double compression
# ----------------------------------------------------------------------------
def jpeg_analysis(path):
    try:
        im = Image.open(path)
        if im.format != "JPEG":
            return {"applicable": False, "score": 0.0, "note": "Not a JPEG"}
        q = getattr(im, "quantization", {}) or {}
        tables = {k: list(v) for k, v in q.items()}
        est = None
        if 0 in tables:
            avg = sum(tables[0]) / 64.0
            est = int(round(100 - avg * 2)) if avg <= 50 else int(round(5000 / avg))
            est = max(1, min(100, est))
        # Double-compression hint: DCT-domain histogram periodicity on luminance
        gray = np.asarray(im.convert("L"), dtype=np.float32)
        h, w = (gray.shape[0] // 8) * 8, (gray.shape[1] // 8) * 8
        gray = gray[:h, :w] - 128
        blocks = gray.reshape(h // 8, 8, w // 8, 8).swapaxes(1, 2).reshape(-1, 8, 8)
        sample = blocks[:: max(1, len(blocks) // 4000)]
        coef = np.array([cv2.dct(b)[0, 1] for b in sample]).round().astype(int)
        hist = np.bincount(np.abs(coef))[:40]
        periodic = 0.0
        if hist.size > 12 and hist.sum() > 200:
            f = np.abs(np.fft.rfft(hist - hist.mean()))
            periodic = float(f[2:].max() / (f[1:].mean() + 1e-6))
        score = clamp01((periodic - 4.0) / 6.0) * 0.6
        return {"applicable": True, "estimated_quality": est, "tables": len(tables),
                "periodicity": round(periodic, 2), "score": score,
                "note": "Periodic gaps in the DCT histogram suggest the image was JPEG-compressed twice "
                        "(edited and re-saved). Social-media re-uploads also cause this."}
    except Exception as e:
        return {"applicable": False, "score": 0.0, "note": f"Error: {e}"}


# ----------------------------------------------------------------------------
# 9. LSB / steganography (native, no external tool)
# ----------------------------------------------------------------------------
def lsb_analysis(im):
    arr = np.asarray(im)
    lsb = arr & 1
    ratios = [float(lsb[..., c].mean()) for c in range(3)]
    # chi-square style test on pixel-value pairs (2k, 2k+1)
    chi_scores = []
    for c in range(3):
        hist = np.bincount(arr[..., c].ravel(), minlength=256).astype(np.float64)
        even, odd = hist[0::2], hist[1::2]
        exp = (even + odd) / 2
        ok = exp > 5
        chi = float((((even - exp) ** 2)[ok] / exp[ok]).sum())
        dof = int(ok.sum()) - 1
        chi_scores.append(chi / max(dof, 1))
    suspicious = all(abs(r - 0.5) < 0.01 for r in ratios) and np.mean(chi_scores) < 0.3
    return {"lsb_ratios": [round(r, 4) for r in ratios],
            "chi_norm": [round(c, 3) for c in chi_scores],
            "suspicious": bool(suspicious),
            "note": "Embedded payloads make LSB planes near-perfectly random (ratio ~0.5) and flatten "
                    "pair histograms (low chi). Works on lossless formats; JPEG recompression destroys it."}


# ----------------------------------------------------------------------------
# 10. Visual clones (kept from original, extended)
# ----------------------------------------------------------------------------
def generate_clones(path, outdir):
    os.makedirs(outdir, exist_ok=True)
    clones = {}
    im = Image.open(path).convert("RGB").resize((400, 400))

    def save(img, name):
        fn = f"{name}.jpg"
        img.save(os.path.join(outdir, fn), quality=92)
        clones[name] = fn

    save(im, "original")
    save(ImageOps.invert(im), "inverted")
    save(im.convert("L").convert("RGB"), "grayscale")
    r, g, b = im.split()
    for nm, ch in (("red_channel", r), ("green_channel", g), ("blue_channel", b)):
        save(Image.merge("RGB", (ch, ch, ch)), nm)
    save(ImageEnhance.Contrast(im).enhance(2.0), "contrast")
    save(ImageOps.equalize(im), "equalized")
    save(ImageOps.solarize(im, 128), "solarize")
    save(ImageOps.posterize(im, 3), "posterize")
    save(ImageOps.mirror(im), "mirror")
    # bit-plane 0 (LSB) amplified
    lsb = (np.asarray(im) & 1) * 255
    save(Image.fromarray(lsb.astype(np.uint8)), "lsb_plane")
    edges = cv2.Canny(cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2GRAY), 100, 200)
    save(Image.fromarray(edges).convert("RGB"), "edges")
    return clones


# ----------------------------------------------------------------------------
# 11. External CLI tools (optional, auto-skipped when not installed)
# ----------------------------------------------------------------------------
def _run(cmd, timeout=45):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                           stdin=subprocess.DEVNULL, errors="replace")
        return (p.stdout + (("\n" + p.stderr) if p.stderr.strip() else ""))[:60000]
    except subprocess.TimeoutExpired:
        return "Timed out"
    except Exception as e:
        return f"Error: {e}"


def run_external_tools(path, case_dir, real_type):
    jobs = {
        "file": ["file", path],
        "strings": ["strings", "-n", "8", path],
        "exiftool": ["exiftool", path],
        "binwalk": ["binwalk", path],  # signature scan only; no extraction
        "steghide": ["steghide", "info", "-p", "", path],
    }
    if real_type in ("PNG", "BMP"):
        jobs["zsteg"] = ["zsteg", "-a", path]
    out = {}
    avail = {k: v for k, v in jobs.items() if shutil.which(v[0])}
    for k in jobs:
        if k not in avail:
            out[k] = "(not installed, skipped)"
    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = {k: ex.submit(_run, v) for k, v in avail.items()}
        for k, f in futs.items():
            out[k] = f.result()
    return out


# ----------------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# Optional calibration produced by evaluate.py --apply (calibration.json).
# Delete the file to fall back to the hand-tuned heuristic score.
# ----------------------------------------------------------------------------
CAL_PATH = os.environ.get("DFD_CALIBRATION", os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration.json"))
_CAL = "unset"


def _load_calibration():
    try:
        import json
        with open(CAL_PATH) as f:
            c = json.load(f)
        if c.get("version") == 1 and c.get("features") and len(c["features"]) == len(c["w"]):
            return c
    except Exception:
        pass
    return None


def set_calibration(cal):
    """Override the calibration (None disables it). Used by evaluate.py to score the raw heuristic."""
    global _CAL
    _CAL = cal


def get_calibration():
    global _CAL
    if _CAL == "unset":
        _CAL = _load_calibration()
    return _CAL


WEIGHTS = {"metadata": 0.20, "faces": 0.22, "ela": 0.10, "noise": 0.10,
           "frequency": 0.09, "copy_move": 0.09, "jpeg": 0.05, "ml": 0.40, "c2pa": 0.35}
# NOTE: weights are hand-tuned and uncalibrated. Calibrate on a labelled set before relying on them.


def compute_verdict(r):
    parts, reasons = {}, []
    parts["metadata"] = r["metadata"]["score"]
    parts["faces"] = r["faces"]["score"] if r["faces"]["count"] else 0.0
    parts["ela"] = r["ela"]["score"]
    parts["noise"] = r["noise"]["score"]
    parts["frequency"] = r["frequency"]["score"]
    parts["copy_move"] = r["copy_move"]["score"]
    parts["jpeg"] = r["jpeg"]["score"]

    ml_on = r["ml"]["available"]
    if ml_on:
        parts["ml"] = r["ml"]["score"]
    c2pa_on = r["c2pa"]["available"]
    if c2pa_on:
        parts["c2pa"] = r["c2pa"]["score"]
    active = [k for k in WEIGHTS if k in parts and not (k == "faces" and not r["faces"]["count"])]
    total_w = sum(WEIGHTS[k] for k in active)
    weighted = sum(WEIGHTS[k] * parts[k] for k in active) / total_w

    # Per-prediction explanation is built INLINE, from the same variables the real score
    # computation uses, rather than reconstructed afterward from a separate copy of this
    # logic elsewhere -- that would risk the explanation silently drifting from what the
    # score actually did. See explain.py for how this structure gets turned into text.
    baseline_terms = [{"feature": k, "value": round(parts[k], 4), "weight": WEIGHTS[k],
                       "share": round(WEIGHTS[k] / total_w, 4),
                       "contribution": round(WEIGHTS[k] * parts[k] / total_w, 4)} for k in active]
    override_steps = []

    def apply_override(name, condition, floor, reason):
        nonlocal weighted
        if not condition:
            return
        pre = weighted
        binding = floor > pre
        if binding:
            weighted = floor
        override_steps.append({"name": name, "floor": floor, "pre_value": round(pre, 4),
                               "binding": binding, "reason": reason})
        reasons.append(reason)

    # Strong single signals should not be diluted by the average
    apply_override("ai_markers", bool(r["metadata"]["ai_markers"] or r["metadata"]["png_text"].get("parameters")),
                   0.85, "Embedded AI-generation/provenance metadata present (strongest indicator)")
    apply_override("copy_move", r["copy_move"]["matched_pairs"] >= 15, 0.6,
                   f"{r['copy_move']['matched_pairs']} consistent cloned keypoint pairs")
    apply_override("faces", bool(r["faces"]["count"] and r["faces"]["score"] > 0.6), 0.6,
                   "Face region statistically inconsistent with the rest of the image")

    if c2pa_on and r["c2pa"]["has_manifest"]:
        c = r["c2pa"]
        if not c["signature_valid"]:
            apply_override("c2pa_invalid", True, 0.85,
                           "C2PA signature INVALID: file was modified after a credential was attached (strong tamper evidence)")
        elif c["ai_generated"]:
            apply_override("c2pa_ai_generated", True, 0.8 if c["trusted"] else 0.5,
                           f"C2PA manifest declares AI/algorithmic generation ({'trusted' if c['trusted'] else 'untrusted'} signer)")
        elif c["trusted"]:
            reasons.append(f"C2PA manifest VALID and TRUSTED (issuer: {c['issuer']}). Real provenance evidence, "
                           f"but does not by itself rule out later undetected edits outside the signed scope.")
    if ml_on:
        n = len(r["ml"]["models"])
        if r["ml"]["score"] >= 0.9:
            apply_override("ml_strong", True, 0.75,
                           f"Local ML classifier{'s' if n > 1 else ''} strongly indicate fake (mean P(fake) {r['ml']['score']:.2f})")
        elif r["ml"]["score"] >= 0.75:
            apply_override("ml_lean", True, 0.6,
                           f"Local ML classifier{'s' if n > 1 else ''} lean fake (mean P(fake) {r['ml']['score']:.2f})")
        if n > 1:
            sc = [m["score"] for m in r["ml"]["models"]]
            if max(sc) - min(sc) > 0.5:
                reasons.append("ML models disagree strongly. Treat the ML score as unreliable for this image")

    heuristic_final = weighted  # value BEFORE any calibration overwrite -- what "reasons" above actually produced

    hi_t, med_t = 0.65, 0.35
    calibrated = None
    calibrated_terms = None
    cal = get_calibration()
    if cal and all(f in parts for f in cal["features"]):
        terms = [(f, w, parts[f], mu, sd, w * (parts[f] - mu) / sd)
                for f, w, mu, sd in zip(cal["features"], cal["w"], cal["mu"], cal["sd"])]
        z = cal["b"] + sum(t[5] for t in terms)
        weighted = float(1.0 / (1.0 + np.exp(-z)))
        hi_t, med_t = cal["thresholds"]["high"], cal["thresholds"]["medium"]
        calibrated = {"n_images": cal.get("n_images"), "cv_auc": cal.get("cv", {}).get("oof_auc")}
        calibrated_terms = [{"feature": f, "value": round(x, 4), "weight": round(w, 4),
                             "mu": round(mu, 4), "sd": round(sd, 4), "contribution": round(c, 4)}
                            for f, w, x, mu, sd, c in sorted(terms, key=lambda t: -abs(t[5]))]
    label_map = [(hi_t, "HIGH", "Strong indicators of manipulation or synthetic origin"),
                 (med_t, "MEDIUM", "Some anomalies. Manual review recommended"),
                 (0.0, "LOW", "No strong indicators found by these automated checks")]
    label, desc = next((l, d) for t, l, d in label_map if weighted >= t)

    explanation = {"mode": "calibrated" if calibrated else "heuristic", "baseline_terms": baseline_terms,
                   "override_steps": override_steps, "heuristic_score": round(heuristic_final, 4),
                   "calibrated_terms": calibrated_terms, "calibrated_baseline": cal["b"] if calibrated else None,
                   "final_score": round(weighted, 4)}

    for k, v in sorted(parts.items(), key=lambda x: -x[1]):
        if v >= 0.4:
            nm = {"ml": "ML classifier", "c2pa": "C2PA provenance"}.get(k, k.replace('_', ' ').title())
            reasons.append(f"{nm} module flagged (score {v:.2f})")
    reasons += r["metadata"]["findings"]
    reasons += r["structure"]["notes"]
    if c2pa_on:
        reasons += r["c2pa"]["findings"]
    if r["lsb"]["suspicious"]:
        reasons.append("LSB plane statistics look like embedded payload")
    return {"score": round(weighted, 3), "percent": int(round(weighted * 100)), "label": label,
            "description": desc, "calibration": calibrated, "explanation": explanation,
            "module_scores": {k: round(v, 3) for k, v in parts.items()},
            "reasons": list(dict.fromkeys(reasons))}


def analyze(path, case_dir, fast=False):
    """fast=True skips clone generation and external CLI tools (used by evaluate.py)."""
    out = os.path.join(case_dir, "analysis")
    os.makedirs(out, exist_ok=True)
    im, orig = load_working_image(path)
    structure = file_structure(path)
    r = {"original_size": list(orig), "structure": structure,
         "hashes": compute_hashes(path), "img_hashes": compute_img_hashes(path),
         "metadata": metadata_analysis(path), "ela": ela_analysis(im, out),
         "noise": noise_analysis(im, out), "frequency": frequency_analysis(im, out),
         "copy_move": copy_move_analysis(im, out), "faces": face_analysis(im, out),
         "jpeg": jpeg_analysis(path), "lsb": lsb_analysis(im)}
    try:
        r["c2pa"] = c2pa_verify.verify(path)
    except Exception as e:
        r["c2pa"] = c2pa_verify.empty_result(f"C2PA check failed: {e}")
    boxes = [tuple(f["box"]) for f in r["faces"]["faces"]]
    try:
        r["ml"] = mlclassifier.classify(im, boxes)
    except Exception as e:  # never let the ML layer break the report
        r["ml"] = {"available": False, "models": [], "score": None, "errors": [str(e)], "installed": 0}
    r["verdict"] = compute_verdict(r)
    if fast:
        r["clones"], r["tools"] = {}, {}
    else:
        r["clones"] = generate_clones(path, os.path.join(case_dir, "clones"))
        r["tools"] = run_external_tools(path, case_dir, structure["real_type"])
    return r
