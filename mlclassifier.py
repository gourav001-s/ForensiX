"""
Local ML deepfake / AI-image classifier (100% offline, no API).
Developed by Gourav Shrivastava

Drop one or more model folders into ./models/ (or set DFD_MODELS_DIR).
Each folder is one model. Supported layouts:

  models/<name>/model.onnx            -> ONNX backend  (needs: onnxruntime)
  models/<name>/onnx/model.onnx       -> ONNX backend  (Hugging Face export layout)
  models/<name>/config.json (+ weights) -> Transformers backend (needs: torch, transformers)

Optional models/<name>/classifier.json overrides auto-detection:
  {
    "labels": ["Real", "Fake"],          # class order of the model output
    "fake_labels": ["Fake"],             # which labels mean "fake" (auto-detected if omitted)
    "input_size": 224,                   # int or [h, w]
    "mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5],
    "mode": "both"                       # "full" | "face" | "both"  (see below)
  }

mode: "full" scores the whole image, "face" scores detected face crops (falls back to
the whole image when no face is found), "both" (default) takes the higher of the two.
Use "face" for models trained on face crops (FaceForensics++ style) and "full" for
generic AI-generated-image detectors.

Every score is P(fake) in [0, 1]. Scores are UNCALIBRATED model outputs, not
probabilities of guilt. Always validate a model on your own labelled samples.
"""

import json
import os
import re
import threading
import time

import numpy as np
from PIL import Image

MODELS_DIR = os.environ.get("DFD_MODELS_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "models"))

FAKE_TOKENS = {"fake", "deepfake", "ai", "artificial", "synthetic", "generated", "spoof",
               "manipulated", "forged", "aigenerated", "sd", "gan", "diffusion"}
REAL_TOKENS = {"real", "authentic", "genuine", "human", "natural", "original", "pristine", "live", "bonafide"}
IMAGENET_MEAN, IMAGENET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

_lock = threading.Lock()
_cache = {}  # folder -> loaded model object or Exception


# ----------------------------------------------------------------------------
# Metadata resolution
# ----------------------------------------------------------------------------
def _read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


def _tokens(label):
    return set(re.findall(r"[a-z0-9]+", str(label).lower()))


def resolve_fake_indices(labels, override=None):
    """Return the class indices that mean 'fake'. Raises ValueError if ambiguous."""
    if override:
        want = {str(o).lower() for o in override}
        idx = [i for i, l in enumerate(labels) if str(l).lower() in want]
        if idx:
            return idx
        raise ValueError(f"fake_labels {override} not found in labels {labels}")
    fake = [i for i, l in enumerate(labels) if _tokens(l) & FAKE_TOKENS and not (_tokens(l) & REAL_TOKENS)]
    if fake:
        return fake
    if len(labels) == 2:
        real = [i for i, l in enumerate(labels) if _tokens(l) & REAL_TOKENS]
        if len(real) == 1:
            return [1 - real[0]]
    raise ValueError(f"Cannot tell which label means 'fake' from {labels}. "
                     f"Add a classifier.json with \"fake_labels\".")


def _size_from_pre(pre):
    s = pre.get("size")
    if isinstance(s, int):
        return (s, s)
    if isinstance(s, dict):
        if "height" in s and "width" in s:
            return (int(s["height"]), int(s["width"]))
        if "shortest_edge" in s:
            return (int(s["shortest_edge"]),) * 2
    return None


def _resolve_meta(folder, session_shape=None):
    cj = _read_json(os.path.join(folder, "classifier.json"))
    pre = _read_json(os.path.join(folder, "preprocessor_config.json"))
    cfg = _read_json(os.path.join(folder, "config.json"))

    labels = cj.get("labels")
    if not labels and cfg.get("id2label"):
        id2 = cfg["id2label"]
        labels = [id2[str(i)] if str(i) in id2 else id2[i] for i in range(len(id2))]

    size = cj.get("input_size")
    if isinstance(size, int):
        size = (size, size)
    elif isinstance(size, list):
        size = tuple(size)
    size = size or _size_from_pre(pre) or session_shape or (224, 224)

    mean = cj.get("mean") or pre.get("image_mean") or IMAGENET_MEAN
    std = cj.get("std") or pre.get("image_std") or IMAGENET_STD
    return {"labels": labels, "fake_labels": cj.get("fake_labels"), "size": tuple(size),
            "mean": np.array(mean, np.float32).reshape(1, 3, 1, 1),
            "std": np.array(std, np.float32).reshape(1, 3, 1, 1),
            "mode": cj.get("mode", "both"), "tta": bool(cj.get("tta", True))}


def _preprocess(pil_img, meta):
    h, w = meta["size"]
    im = pil_img.convert("RGB").resize((w, h), Image.BILINEAR)
    x = np.asarray(im, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0
    return (x - meta["mean"]) / meta["std"]


def _softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


# ----------------------------------------------------------------------------
# Backends
# ----------------------------------------------------------------------------
class _OnnxModel:
    backend = "onnx"

    def __init__(self, folder, onnx_path):
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = max(1, (os.cpu_count() or 2) // 2)
        self.sess = ort.InferenceSession(onnx_path, opts, providers=["CPUExecutionProvider"])
        inp = self.sess.get_inputs()[0]
        self.input_name = inp.name
        shp = inp.shape
        sess_shape = (shp[2], shp[3]) if len(shp) == 4 and all(isinstance(d, int) for d in shp[2:]) else None
        self.meta = _resolve_meta(folder, sess_shape)
        out_dim = self.sess.get_outputs()[0].shape[-1]
        if not self.meta["labels"]:
            if out_dim == 2:
                raise ValueError("Model has no label info. Add classifier.json with \"labels\" and \"fake_labels\".")
            if out_dim == 1:
                self.meta["labels"] = ["fake"]
            else:
                raise ValueError("Add classifier.json with \"labels\" for this model.")
        self.fake_idx = resolve_fake_indices(self.meta["labels"], self.meta["fake_labels"])

    def _p(self, x):
        out = self.sess.run(None, {self.input_name: x})[0][0].astype(np.float64)
        if out.size == 1:  # single logit
            return float(1 / (1 + np.exp(-out[0])))
        return float(_softmax(out)[self.fake_idx].sum())

    def predict(self, pil_img):
        x = _preprocess(pil_img, self.meta)
        p = self._p(x)
        if self.meta["tta"]:
            p = (p + self._p(x[..., ::-1].copy())) / 2.0
        return p


class _HFModel:
    backend = "transformers"

    def __init__(self, folder):
        import torch
        from transformers import AutoImageProcessor, AutoModelForImageClassification
        self.torch = torch
        self.proc = AutoImageProcessor.from_pretrained(folder)
        self.model = AutoModelForImageClassification.from_pretrained(folder).eval()
        id2 = self.model.config.id2label
        labels = [id2[i] for i in range(len(id2))]
        cj = _read_json(os.path.join(folder, "classifier.json"))
        self.meta = {"labels": cj.get("labels") or labels, "mode": cj.get("mode", "both"),
                     "tta": bool(cj.get("tta", True)), "size": (0, 0)}
        self.fake_idx = resolve_fake_indices(self.meta["labels"], cj.get("fake_labels"))

    def _p(self, pil_img):
        with self.torch.no_grad():
            inputs = self.proc(images=pil_img.convert("RGB"), return_tensors="pt")
            logits = self.model(**inputs).logits[0].double().numpy()
        return float(_softmax(logits)[self.fake_idx].sum())

    def predict(self, pil_img):
        p = self._p(pil_img)
        if self.meta["tta"]:
            p = (p + self._p(pil_img.transpose(Image.FLIP_LEFT_RIGHT))) / 2.0
        return p


def _find_onnx(folder):
    for rel in ("model.onnx", os.path.join("onnx", "model.onnx")):
        p = os.path.join(folder, rel)
        if os.path.isfile(p):
            return p
    for f in sorted(os.listdir(folder)):
        if f.endswith(".onnx"):
            return os.path.join(folder, f)
    return None


def _load(folder):
    onnx_path = _find_onnx(folder)
    if onnx_path:
        return _OnnxModel(folder, onnx_path)
    if os.path.isfile(os.path.join(folder, "config.json")):
        return _HFModel(folder)
    raise FileNotFoundError("No model.onnx or config.json found")


# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------
def list_models():
    """Installed model folders with load status (loads lazily, caches result)."""
    out = []
    if not os.path.isdir(MODELS_DIR):
        return out
    for name in sorted(os.listdir(MODELS_DIR)):
        folder = os.path.join(MODELS_DIR, name)
        if not os.path.isdir(folder) or name.startswith("."):
            continue
        if os.path.exists(os.path.join(folder, "DISABLED")):
            out.append({"name": name, "ok": False, "error": "Disabled (see models/<name>/DISABLED). "
                       "Delete that file, or run compare_models.py --keep, to re-enable."})
            continue
        with _lock:
            if folder not in _cache:
                try:
                    _cache[folder] = _load(folder)
                except Exception as e:  # keep the error so the UI can show it
                    _cache[folder] = e
            m = _cache[folder]
        if isinstance(m, Exception):
            out.append({"name": name, "ok": False, "error": f"{type(m).__name__}: {m}"})
        else:
            out.append({"name": name, "ok": True, "backend": m.backend, "labels": m.meta["labels"],
                        "fake_labels": [m.meta["labels"][i] for i in m.fake_idx], "mode": m.meta["mode"]})
    return out


def reload_models():
    with _lock:
        _cache.clear()


def _crop_face(im, box, margin=0.3):
    x, y, w, h = box
    mx, my = int(w * margin), int(h * margin)
    return im.crop((max(0, x - mx), max(0, y - my), min(im.width, x + w + mx), min(im.height, y + h + my)))


def classify(im, face_boxes=None):
    """
    Run every installed model. Returns:
      {"available": bool, "models": [...], "score": float|None, "errors": [...]}
    """
    infos = list_models()
    res = {"available": False, "models": [], "score": None, "errors": [], "installed": len(infos)}
    scores = []
    for info in infos:
        if not info["ok"]:
            res["errors"].append(f"{info['name']}: {info['error']}")
            continue
        m = _cache[os.path.join(MODELS_DIR, info["name"])]
        t0 = time.time()
        try:
            mode = m.meta["mode"]
            full = m.predict(im) if mode in ("full", "both") or not face_boxes else None
            faces = []
            if mode in ("face", "both") and face_boxes:
                faces = [round(m.predict(_crop_face(im, b)), 4) for b in face_boxes[:5]]
            cands = [p for p in ([full] + faces) if p is not None]
            score = max(cands) if cands else None
            if score is None:
                continue
            res["models"].append({"name": info["name"], "backend": m.backend, "mode": mode,
                                  "full": None if full is None else round(full, 4),
                                  "faces": faces, "score": round(score, 4),
                                  "ms": int((time.time() - t0) * 1000)})
            scores.append(score)
        except Exception as e:
            res["errors"].append(f"{info['name']}: inference failed: {e}")
    if scores:
        res["available"] = True
        res["score"] = round(float(np.mean(scores)), 4)
    return res
