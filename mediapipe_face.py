"""
MediaPipe FaceLandmarker integration -- replaces/upgrades Haar-cascade face detection.
Developed by Gourav Shrivastava

Needs a downloaded model file (~4-6 MB, MIT licensed, from Google's MediaPipe model
zoo). Not bundled with the pip package. Get it with:

    python3 get_facemesh.py

which saves to models/face_landmarker.task (or set DFD_FACEMESH_MODEL).
Without the model file, forensics.py automatically falls back to the original
Haar-cascade face box detector -- nothing breaks, you just lose the landmark-based
checks below.

What landmarks buy you over Haar boxes:
  - Works on rotated / tilted / partially turned faces (Haar is frontal-only).
  - A tight, rotation-aware face region instead of an axis-aligned box, so forensic
    sub-scores (ELA/noise/sharpness) aren't diluted by background pixels in the corners.
  - Eye and mouth sub-regions, where face-swap and GAN blending artifacts concentrate,
    can be scored separately from the rest of the face.
  - Two geometric tells specific to landmarks:
      * Left/right eye-openness asymmetry (EAR) -- swapped/regenerated eyes often
        render at slightly different openness than the source face.
      * Face-shape symmetry after roll correction -- warped or blended geometry
        shows up as landmarks that don't mirror cleanly across the face's own axis.
    Both are WEAK, supplementary signals: real faces are asymmetric too, especially
    mid-expression or turned away from the camera. Treat them as one more data point,
    not a verdict on their own.

Region index sets (eyes, lips, iris, oval) are pulled at import time from
mediapipe.tasks.vision.FaceLandmarksConnections -- the library's own authoritative
lists -- rather than hard-coded, so they can't silently drift from whatever
MediaPipe version is installed. The 6-point EAR subsets are the standard ones used
across MediaPipe eye-aspect-ratio implementations; each is checked at import time to
actually be a subset of the library's own eye landmark set, so a version mismatch
fails loudly instead of silently measuring the wrong points.
"""

import os

import numpy as np

MODEL_PATH = os.environ.get("DFD_FACEMESH_MODEL",
                            os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "face_landmarker.task"))

AVAILABLE = False
_INIT_ERROR = None
_landmarker = None
REGIONS = {}
EAR_RIGHT, EAR_LEFT = [33, 160, 158, 133, 153, 144], [362, 385, 387, 263, 373, 380]

try:
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode

    _conn = mp.tasks.vision.FaceLandmarksConnections

    def _idxs(conns):
        s = set()
        for c in conns:
            s.add(c.start)
            s.add(c.end)
        return sorted(s)

    REGIONS = {"left_eye": _idxs(_conn.FACE_LANDMARKS_LEFT_EYE), "right_eye": _idxs(_conn.FACE_LANDMARKS_RIGHT_EYE),
               "lips": _idxs(_conn.FACE_LANDMARKS_LIPS), "oval": _idxs(_conn.FACE_LANDMARKS_FACE_OVAL)}
    for name, subset in (("right eye", EAR_RIGHT), ("left eye", EAR_LEFT)):
        pool = REGIONS["right_eye"] if "right" in name else REGIONS["left_eye"]
        if not set(subset) <= set(pool):
            raise RuntimeError(f"EAR {name} indices are not a subset of this MediaPipe version's landmark set "
                               f"(library changed?). Refusing to guess; landmark analysis disabled.")

    if os.path.isfile(MODEL_PATH):
        opts = FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=MODEL_PATH),
                                     running_mode=RunningMode.IMAGE, num_faces=6,
                                     min_face_detection_confidence=0.5, output_face_blendshapes=False,
                                     output_facial_transformation_matrixes=False)
        _landmarker = FaceLandmarker.create_from_options(opts)
        AVAILABLE = True
    else:
        _INIT_ERROR = f"Model not found at {MODEL_PATH}. Run: python3 get_facemesh.py"
except Exception as e:  # pragma: no cover - exercised when mediapipe/model is absent
    _INIT_ERROR = f"{type(e).__name__}: {e}"


def status():
    return {"available": AVAILABLE, "model_path": MODEL_PATH, "error": None if AVAILABLE else _INIT_ERROR}


def _bbox(points, margin=0.0):
    xs, ys = points[:, 0], points[:, 1]
    w, h = xs.max() - xs.min(), ys.max() - ys.min()
    x0, y0 = xs.min() - w * margin, ys.min() - h * margin
    x1, y1 = xs.max() + w * margin, ys.max() + h * margin
    return [int(round(x0)), int(round(y0)), int(round(x1 - x0)), int(round(y1 - y0))]


def _ear(pts):
    """Eye aspect ratio from the standard 6-point layout: p1,p4 horizontal corners,
    (p2,p6) and (p3,p5) vertical pairs. Returns None if the eye is degenerate (e.g. off-frame)."""
    p1, p2, p3, p4, p5, p6 = pts
    horiz = np.linalg.norm(p1 - p4)
    if horiz < 1e-6:
        return None
    vert = (np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)) / 2.0
    return float(vert / horiz)


def _symmetry_score(all_pts, eye_l_center, eye_r_center):
    """Roll-correct the face, mirror the oval landmarks across the vertical midline,
    and measure how far each point sits from its own mirror image (nearest neighbour),
    normalised by inter-ocular distance. 0 = perfectly symmetric under this correction.
    Only corrects in-plane rotation (roll); a turned (yaw) or expressive face will
    still show elevated asymmetry for reasons that have nothing to do with tampering."""
    iod = np.linalg.norm(eye_r_center - eye_l_center)
    if iod < 1e-6:
        return None
    roll = np.arctan2(eye_l_center[1] - eye_r_center[1], eye_l_center[0] - eye_r_center[0])
    c, s = np.cos(-roll), np.sin(-roll)
    R = np.array([[c, -s], [s, c]])
    centroid = all_pts.mean(axis=0)
    rot = (all_pts - centroid) @ R.T
    oval = rot[REGIONS["oval_local_idx"]]
    mirrored = oval.copy()
    mirrored[:, 0] *= -1
    # nearest-neighbour distance from each mirrored point to the real point set (vectorised)
    d = np.sqrt(((mirrored[:, None, :] - oval[None, :, :]) ** 2).sum(-1)).min(axis=1)
    return float(d.mean() / iod)


def detect(pil_image):
    """Run FaceLandmarker on a PIL RGB image. Returns a list of face dicts, or None
    if the model isn't available (caller should fall back to Haar)."""
    if not AVAILABLE:
        return None
    import mediapipe as mp
    rgb = np.asarray(pil_image.convert("RGB"))
    h, w = rgb.shape[:2]
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = _landmarker.detect(mp_img)
    faces = []
    oval_idx = REGIONS["oval"]
    REGIONS["oval_local_idx"] = list(range(len(oval_idx)))  # set once; same length every call
    for lm_list in result.face_landmarks:
        pts = np.array([[p.x * w, p.y * h] for p in lm_list], dtype=np.float64)
        oval_pts = pts[oval_idx]
        box = _bbox(oval_pts, margin=0.08)
        bx1, by1 = box[0] + box[2], box[1] + box[3]  # true right/bottom edge BEFORE origin clipping
        bx0, by0 = max(0, box[0]), max(0, box[1])
        # width/height must shrink by whatever the origin was clipped by (see forensics.py's
        # _clip_box for the same fix and why) -- otherwise a face near the left/top edge keeps
        # its full unclipped width/height and overshoots into background pixels.
        box = [bx0, by0, min(bx1, w) - bx0, min(by1, h) - by0]
        if box[2] < 8 or box[3] < 8:
            continue
        regions = {"left_eye": _bbox(pts[REGIONS["left_eye"]], margin=0.35),
                  "right_eye": _bbox(pts[REGIONS["right_eye"]], margin=0.35),
                  "lips": _bbox(pts[REGIONS["lips"]], margin=0.25)}
        ear_l, ear_r = _ear(pts[EAR_LEFT]), _ear(pts[EAR_RIGHT])
        ear_asym = abs(ear_l - ear_r) / max((ear_l + ear_r) / 2, 1e-6) if (ear_l is not None and ear_r is not None) else None
        eye_l_c, eye_r_c = pts[REGIONS["left_eye"]].mean(axis=0), pts[REGIONS["right_eye"]].mean(axis=0)
        sym = _symmetry_score(pts[oval_idx], eye_l_c, eye_r_c)
        roll_deg = float(np.degrees(np.arctan2(eye_l_c[1] - eye_r_c[1], eye_l_c[0] - eye_r_c[0])))
        faces.append({"box": box, "regions": regions, "roll_deg": round(roll_deg, 1),
                      "ear_left": None if ear_l is None else round(ear_l, 3),
                      "ear_right": None if ear_r is None else round(ear_r, 3),
                      "ear_asymmetry": None if ear_asym is None else round(ear_asym, 3),
                      "symmetry_dev": None if sym is None else round(sym, 4)})
    return faces
