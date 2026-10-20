# AGENTS.md — CommunityForensics-DeepfakeDet-ViT

## What this repo is
Hugging Face model repo for `buildborderless/CommunityForensics-DeepfakeDet-ViT` — a ViT-Small classifier for deepfake image detection. Trained on 2.7M samples across 4,803 generators. This is a model distribution repo (no app, no build, no tests).

## Key files
- **`model.safetensors`** — HF-format weights (Git LFS — ensure `git lfs pull` after clone)
- **`config.json`** — `ViTForImageClassification` config (384×384, 6 heads, num_labels=1, sigmoid output: real/fake)
- **`preprocessor_config.json`** — CLIP-style normalization, resize to shortest_edge=440, center-crop to 384
- **`modeling_vit_classifier.py`** — **DEPRECATED** (moved to `scripts/`). Use standard HF path below.
- **`pretrained_weights/`** — original `.pt` checkpoints from training (also LFS)
- **`onnx/`** — 5 pre-exported ONNX variants (15MB–84MB) for CPU/GPU deployment. See README for variant guide.

## Usage
The model is hosted on Hugging Face. The standard way to load it is via `transformers`:
```python
from transformers import ViTForImageClassification, AutoImageProcessor
model = ViTForImageClassification.from_pretrained("buildborderless/CommunityForensics-DeepfakeDet-ViT")
processor = AutoImageProcessor.from_pretrained("buildborderless/CommunityForensics-DeepfakeDet-ViT")

# Preprocess without overriding size (preserves aspect ratio via shortest_edge=440 + center_crop=384)
inputs = processor(images=image, return_tensors="pt")
logits = model(**inputs).logits
```

> [!IMPORTANT]
> **Image Preprocessing**: Do NOT pass `size={"height": 384, "width": 384}` to the processor. Squashing non-square images directly distorts facial features and destroys deepfake detection accuracy. The processor automatically handles `shortest_edge=440` and `center_crop=384` defined in `preprocessor_config.json`.

## Multi-Threading & XAI Hook Safety
- When deploying this model in multi-threaded web servers (FastAPI/Gradio), wrap forward/backward passes and Captum/XAI hook registrations with a `threading.Lock()`. PyTorch hooks registered on model singletons are NOT thread-safe and will contaminate concurrent inferences if run simultaneously.

The custom wrapper (`modeling_vit_classifier.py`) uses `timm.create_model` with a sigmoid output and `pretrained_weights/model_v11_ViT_384_base_ckpt.pt`. This is for standalone (non-HF-pipeline) inference requiring both `timm` and `transformers`.

## Dependencies
- `transformers >= 5.4.0` (required — older versions lack `shortest_edge` resize and will squash images)
- `timm` (for the deprecated ViTClassifier wrapper only)
- `torch`, `torchvision`, `Pillow`
- `onnxruntime >= 1.27` (for ONNX models)

## Scripts (in `scripts/`)
Data processing utilities for the eval dataset — not needed for inference:
- `convert_to_pytorch.py` — convert timm checkpoints to HuggingFace format
- `resample_evalset.py` — face-detection-based dataset filtering
- `restructure.py` — reorganize real/generated image directories
- `quick_analysis.py` — dataset statistics report

## Git LFS
All weight files (`.safetensors`, `.pt`, `.ckpt`, `.onnx`) are stored via Git LFS. Always run `git lfs pull` after cloning or the model files will be pointer stubs. The full ONNX model alone is 138MB — pull selectively with `git lfs pull --include="onnx/model_int8.onnx"` if you only need one variant.

## Remote
This repo is pushed to `https://huggingface.co/buildborderless/CommunityForensics-DeepfakeDet-ViT`, not GitHub. Standard `gh` CLI commands will not work.
