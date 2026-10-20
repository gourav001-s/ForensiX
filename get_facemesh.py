#!/usr/bin/env python3
"""
Download the MediaPipe FaceLandmarker model for mediapipe_face.py.
Developed by Gourav Shrivastava

Run this on YOUR machine (needs internet once). The build environment that wrote
this toolkit could not reach storage.googleapis.com to fetch or test the model
directly -- verify it works with --selftest below before relying on it.

Usage:
    python3 get_facemesh.py                    # download the default (float16) model
    python3 get_facemesh.py --variant float32  # slightly larger, more precise
    python3 get_facemesh.py --selftest         # download, then run it on a synthetic
                                                # image and print what it detects
"""
import argparse
import hashlib
import os
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
DEST = os.path.join(HERE, "models", "face_landmarker.task")
# Official Google MediaPipe model zoo. MIT licensed. See:
# https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker
URLS = {"float16": "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
                   "face_landmarker/float16/1/face_landmarker.task",
        "float32": "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
                   "face_landmarker/float32/1/face_landmarker.task"}


def download(url, dest):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f"Downloading {url}")
    tmp = dest + ".part"

    def hook(count, block, total):
        if total > 0:
            pct = min(100, count * block * 100 // total)
            sys.stdout.write(f"\r  {pct}%")
            sys.stdout.flush()
    urllib.request.urlretrieve(url, tmp, reporthook=hook)
    print()
    os.replace(tmp, dest)
    size = os.path.getsize(dest)
    print(f"Saved to {dest} ({size / 1e6:.1f} MB, sha256 {hashlib.sha256(open(dest,'rb').read()).hexdigest()[:16]}...)")


def selftest():
    sys.path.insert(0, HERE)
    import importlib
    import numpy as np
    from PIL import Image
    import mediapipe_face
    importlib.reload(mediapipe_face)  # pick up the just-downloaded model
    st = mediapipe_face.status()
    print("status():", st)
    if not st["available"]:
        sys.exit("Model did not load. See the error above.")
    # Synthetic image: no real face, so this just proves the pipeline runs without
    # crashing end to end. Test with a real photo of a face for a meaningful check.
    im = Image.fromarray(np.zeros((200, 200, 3), np.uint8))
    faces = mediapipe_face.detect(im)
    print(f"detect() ran without error, found {len(faces)} face(s) on a blank test image "
         f"(0 is expected here). Run the actual app and upload a real photo to confirm "
         f"detection quality.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=list(URLS), default="float16")
    ap.add_argument("--dest", default=DEST)
    ap.add_argument("--selftest", action="store_true", help="download, then smoke-test it")
    args = ap.parse_args()
    if os.path.exists(args.dest) and not args.selftest:
        print(f"Already present: {args.dest} (delete it first to re-download)")
    else:
        download(URLS[args.variant], args.dest)
    if args.selftest:
        selftest()
    print("\nRestart the app (or open a new case) to use it. The 'Per-face results' table "
         "will switch from 'haar detector' to 'mediapipe detector' once it's picked up.")
