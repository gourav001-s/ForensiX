#!/usr/bin/env python3
"""
Download a Hugging Face image-classification model into ./models/<name>
Developed by Gourav Shrivastava

Usage:  python3 get_model.py --repo <hf-repo-id> --name mymodel [--mode face|full|both]
Run this on YOUR machine (needs internet once). Afterwards everything is offline.
Read the model card first: check license, training data and the label names.
"""
import argparse
import json
import os
import sys

try:
    from huggingface_hub import snapshot_download
except ImportError:
    sys.exit("Install first:  pip install -r requirements-ml.txt")

ap = argparse.ArgumentParser()
ap.add_argument("--repo", required=True, help="Hugging Face repo id, e.g. owner/model-name")
ap.add_argument("--name", required=True, help="folder name under ./models")
ap.add_argument("--mode", choices=["full", "face", "both"], default="both")
ap.add_argument("--revision", default=None, help="pin a commit/tag for reproducibility")
args = ap.parse_args()

dest = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", args.name)
os.makedirs(dest, exist_ok=True)
snapshot_download(repo_id=args.repo, revision=args.revision, local_dir=dest,
                  allow_patterns=["*.json", "*.onnx", "*.safetensors", "*.txt", "*.md"])

cj_path = os.path.join(dest, "classifier.json")
cj = json.load(open(cj_path)) if os.path.exists(cj_path) else {}
cj["mode"] = args.mode
json.dump(cj, open(cj_path, "w"), indent=2)

has_onnx = any(f.endswith(".onnx") for _, _, fs in os.walk(dest) for f in fs)
print(f"Downloaded to {dest}")
print("Backend:", "ONNX" if has_onnx else "PyTorch (requires: pip install torch transformers)")
print("Next: start the app, open 'ML Models' and check that labels/fake mapping look correct.")
