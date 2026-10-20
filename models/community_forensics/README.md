---
base_model:
- timm/vit_small_patch16_384.augreg_in21k_ft_in1k
library_name: transformers
license: mit
pipeline_tag: image-classification
tags:
- image-classification
- timm
- transformers
- detection
- deepfake
- forensics
- deepfake_detection
- community
- opensight
- onnx
---

# CommunityForensics DeepfakeDet-ViT

Vision Transformer (ViT-Small) trained on 2.7M samples across 4,803 generators for detecting AI-generated images. Presented in [Community Forensics: Using Thousands of Generators to Train Fake Image Detectors](https://huggingface.co/papers/2411.04125) (CVPR 2025).

**Uploaded for community validation as part of OpenSight** — An upcoming open-source framework for adaptive deepfake detection.

**Project OpenSight HF Spaces coming soon with an eval playground and eventually a leaderboard. Preview:**

![image/png](https://cdn-uploads.huggingface.co/production/uploads/639daf827270667011153fbc/AUmW697OefKN83BClM1ae.png)

## IMPORTANT — Configuration Fix (July 2026)

**If you downloaded this model before July 22, 2026, your local copy has incorrect config and weights.** Apologies for the mess — this model was originally hastily put together as an internal proof-of-concept for a hackathon, and we never imagined it would quietly become one of the top image classification models on Hugging Face. This update is long overdue.

The `model.safetensors` has been regenerated from the correct training checkpoint and all metadata has been fixed. For a detailed breakdown of every change, see [CHANGELOG.md](CHANGELOG.md). If you use LLM-based coding agents (Claude Code, Cursor, GitHub Copilot, etc.), the repo includes an [AGENTS.md](AGENTS.md) to help your agent ramp up quickly.

| Bug | Effect | Fixed Value |
|---|---|---|
| Wrong `model.safetensors` | Weights from different model (intermediate_size=3072, wrong classifier) | Regenerated from `pretrained_weights/model_v11_ViT_384_base_ckpt.pt` |
| `num_attention_heads: 12` | **Silently wrong** — attention sliced 12×32d instead of 6×64d | `6` |
| Preprocessor `size` | Squashed non-square images or no center-crop | `shortest_edge: 440` + `do_center_crop` |
| `num_classes: 2` / no `num_labels` | Wrong output format for single-class classifier — `num_classes=1` maps to 2 labels internally | `num_labels: 1` (sigmoid output) |

### ⚠️ Breaking change for older transformers versions

This model now requires `transformers >= 5.4.0` for correct image preprocessing. Versions older than 5.4.0 will crash with a `ValueError` when loading the preprocessor — this is intentional and prevents silently-squashed images. If upgrading is not an option, you can preprocess images manually (resize shortest edge → 440, center-crop → 384, CLIP-normalize) and pass `do_resize=False` to the processor.

### How to verify you have the fix

```python
import json
with open("path/to/config.json") as f:
    cfg = json.load(f)
assert cfg["num_labels"] == 1, "Still broken — re-download the model"
assert cfg["num_attention_heads"] == 6, "Still broken — re-download the model"
assert cfg["intermediate_size"] == 1536, "Still broken — re-download the model"
```

### If you were using the old custom wrapper (`modeling_vit_classifier.py`)

It has been moved to `scripts/` and marked deprecated. Switch to the standard HuggingFace path:

```python
from transformers import ViTForImageClassification, ViTImageProcessor
model = ViTForImageClassification.from_pretrained("buildborderless/CommunityForensics-DeepfakeDet-ViT")
processor = ViTImageProcessor.from_pretrained("buildborderless/CommunityForensics-DeepfakeDet-ViT")
```

### If you were using the ONNX repo

The separate [`buildborderless/CommunityForensics-DeepfakeDet-ViT-ONNX`](https://huggingface.co/buildborderless/CommunityForensics-DeepfakeDet-ViT-ONNX) repo is now deprecated. All ONNX models are included here in `onnx/` with corrected weights. Old exports are archived in `onnx_legacy/`.

### Archived files

- `model_legacy.safetensors` — previous (incorrect) weights, frozen for reference
- `model_fixed.safetensors` — identical copy of the current `model.safetensors`
- `onnx_legacy/` — previous ONNX exports from the incorrect weights

---

## Quick Start

```python
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
import torch

model = ViTForImageClassification.from_pretrained("buildborderless/CommunityForensics-DeepfakeDet-ViT")
processor = ViTImageProcessor.from_pretrained("buildborderless/CommunityForensics-DeepfakeDet-ViT")

image = Image.open("suspicious_image.jpg")
inputs = processor(image, return_tensors="pt")
outputs = model(**inputs)

fake_prob = torch.sigmoid(outputs.logits).item()
print(f"fake: {fake_prob:.4f}, real: {1 - fake_prob:.4f}")
print(f"verdict: {'fake' if fake_prob > 0.5 else 'real'}")
```

## Dependencies

- `transformers >= 5.4.0` (**required** — older versions lack `shortest_edge` resize and will crash. Do not downgrade below 5.4.0 or images will be silently squashed.)
- `torch`, `torchvision`, `Pillow`
- `onnxruntime >= 1.27` (for ONNX models — install `onnxruntime` for CPU or `onnxruntime-gpu` for GPU)

---

## ONNX Variants (v1.1)

Five pre-exported ONNX models with different size/speed trade-offs. All use the corrected config (single-class sigmoid output).

| Variant | Size | Speed (CPU) | Fidelity vs FP32 | Best For |
|---|---|---|---|---|
| `model.onnx` (full) | 83 MB | ★★★ | Reference (FP32) | Maximum accuracy, server-side baseline |
| `model_int8.onnx` | 22 MB | ★★★★★ | High fidelity on standard inputs; may diverge on OOD generators | **Fastest CPU**, general deployment |
| `model_uint8.onnx` | 22 MB | ★★★★★ | Alternative dynamic quantization error profile | Fast CPU deployment |
| `model_quantized.onnx` | 22 MB | ★★★★★ | Identical to `model_int8.onnx` | Drop-in INT8 alias |
| `model_q4.onnx` | 15 MB | ★★★ | Aggressive weight quantization; high variance on subtle inputs | Smallest disk/RAM footprint |

**Which variant should I use?**

| Use case | Recommended variant | Why |
|---|---|---|
| Server-side, maximum accuracy | `model.onnx` (full) | No quantization loss, FP32 precision — reference baseline |
| General CPU deployment | `model_int8.onnx` | Fastest CPU inference, matches FP32 on clear-cut inputs |
| Disk/RAM constrained | `model_q4.onnx` | Smallest file size (15 MB), low disk/RAM footprint |

> **Quantization note**: Dynamic per-tensor quantization without calibration causes quantized variants to diverge from FP32 on certain inputs (up to 10–70 percentage points) — particularly images from generators outside the training set. **Significant disagreement between FP32 and INT8/Q4 indicates the input is near the model's decision boundary or out-of-distribution.** For maximum single-model consistency, use `model.onnx` (FP32).

```python
import onnxruntime as ort, numpy as np
from PIL import Image

session = ort.InferenceSession("onnx/model_int8.onnx")

# Preprocess: shortest edge → 440 (maintain aspect ratio), center-crop → 384, CLIP normalize
image = Image.open("image.jpg")
w, h = image.size
scale = 440 / min(w, h)
img = image.resize((int(w * scale), int(h * scale)))
left = (img.size[0] - 384) // 2
top = (img.size[1] - 384) // 2
img = img.crop((left, top, left + 384, top + 384))
arr = np.array(img, dtype=np.float32) / 255.0
arr = (arr - np.array([0.4815, 0.4578, 0.4082])) / np.array([0.2686, 0.2613, 0.2758])
arr = np.expand_dims(arr.transpose(2, 0, 1), 0)

logit = session.run(None, {"pixel_values": arr})[0][0, 0]
fake_prob = 1 / (1 + np.exp(-logit))
```

---

## Benchmark & Comparison Space

A companion Gradio Space lets you test every variant side by side — upload your own images and compare PyTorch vs ONNX performance in real time.

**What it does:**

| Tab | Description |
|---|---|
| **Compare** | Upload a single image, see PyTorch and all selected ONNX variants side by side with timing |
| **Benchmark** | Upload multiple images for batch processing, compare inference speed across all variants |
| **Help** | Variant selection guide and preprocessing details |

**Use it to:**
- See how quantization affects prediction confidence on your own images
- Measure real-world inference speed across variants (CPU/GPU)
- Verify the corrected model produces results consistent with the original timm pipeline

> Link coming soon — deploying as a separate Space. Follow the repo for updates.

---

## Model Details

- **Developed by**: Jeongsoo Park and Andrew Owens, University of Michigan
- **HF integration + ONNX**: Han Yoon, Borderless / Ethix R&D
- **Model type**: Vision Transformer (ViT-Small)
- **License**: MIT
- **Input**: RGB image, shortest edge resized to 440 (aspect ratio preserved), center-cropped to 384×384, CLIP-normalized
- **Output**: single logit → sigmoid → fake probability
- **Architecture**: hidden_size=384, 6 attention heads, 12 layers, patch_size=16, intermediate_size=1536

### Links

- **Original paper**: [arXiv:2411.04125](https://arxiv.org/pdf/2411.04125)
- **Original repository**: [JeongsooP/Community-Forensics](https://github.com/JeongsooP/Community-Forensics)
- **Project page**: https://jespark.net/projects/2024/community_forensics
- **Datasets**: [Full (1.1TB)](https://huggingface.co/datasets/OwensLab/CommunityForensics), [Small (278GB)](https://huggingface.co/datasets/OwensLab/CommunityForensics-Small), [Eval (206GB)](https://huggingface.co/datasets/OwensLab/CommunityForensics-Eval)

---

## Coming Soon — v2

We're actively working on a significantly stronger model with an expanded dataset and novel detection concepts. Follow the repo for updates in the coming months.

---

## Citation

```bibtex
@InProceedings{Park_2025_CVPR,
    author    = {Park, Jeongsoo and Owens, Andrew},
    title     = {Community Forensics: Using Thousands of Generators to Train Fake Image Detectors},
    booktitle = {Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {8245-8257}
}
```
