# Changelog — July 2026 Overhaul

This documents every change in [PR #5](https://huggingface.co/buildborderless/CommunityForensics-DeepfakeDet-ViT/discussions/5). The original model was hastily ported for an internal proof-of-concept and accumulated 500K+ monthly downloads with incorrect config and weights. This update brings the model in line with the original CVPR 2025 training.

---

## Critical fixes

### `model.safetensors` — regenerated from correct checkpoint

The previous safetensors was converted from **different weights** (intermediate_size=3072, 8× MLP ratio, wrong classifier head). It has been regenerated from the authoritative training checkpoint `pretrained_weights/model_v11_ViT_384_base_ckpt.pt` with the correct ViT-Small architecture.

- `intermediate_size`: 3072 → **1536**
- `classifier`: weight `[2, 384]` → **`[1, 384]`**
- Backbone weights: now match the original training checkpoint exactly

### `config.json` — corrected for ViT-Small

| Field | Before | After |
|---|---|---|
| `num_attention_heads` | 12 (ViT-Base) | **6** (ViT-Small) |
| `intermediate_size` | 3072 | **1536** |
| `num_classes` | 2 (wrong) | **1** (single-class sigmoid) |
| `num_heads` / `num_layers` | present (redundant) | removed |
| `id2label` / `label2id` | missing then present | removed (not meaningful for sigmoid) |
| `mlp_ratio` / `encoder_stride` | present (non-standard) | removed |
| `input_size` | missing | **384** (HF inference widget) |
| `transformers_version` | `4.50.0.dev0` | **`5.4.0`** (minimum for `shortest_edge`) |

### `preprocessor_config.json` — fixed image preprocessing

| Field | Before | After |
|---|---|---|
| `size` | `384` (int) or `{height: 440, width: 440}` | **`{shortest_edge: 440}`** |
| `do_center_crop` | missing or implicit | **`true`** |
| `crop_size` | n/a | **`384`** |

The previous configs either squashed non-square images (force-resize to 440×440) or skipped center-cropping entirely. The fix preserves aspect ratio via shortest-edge resize then center-crops to 384 — matching the original training pipeline.

---

## ONNX models — re-exported from corrected weights

### New exports (`onnx/`)

Five variants exported via `optimum-cli` + `onnxruntime` quantization:

| Variant | Size | Method |
|---|---|---|
| `model.onnx` | 84 MB | FP32 base (optimum-cli) |
| `model_int8.onnx` | 22 MB | Dynamic QInt8 quantization |
| `model_uint8.onnx` | 22 MB | Dynamic QUInt8 quantization |
| `model_quantized.onnx` | 22 MB | Alias of INT8 |
| `model_q4.onnx` | 16 MB | 4-bit MatMul quantization (block_size=32) |

### Legacy exports (`onnx_legacy/`)

The previous 8 ONNX variants (21–138 MB) — exported from the incorrect weights — have been moved to `onnx_legacy/` and are no longer recommended.

---

## Housekeeping

### Licensing
- `LICENSE` added with original MIT copyright (2025 Jeongsoo Park)
- Additional copyright lines for Han Yoon / Borderless / Ethix R&D (2025–2026) covering HF integration, ONNX exports, and configuration fixes

### File reorganization

| Before | After |
|---|---|
| `modeling_vit_classifier.py` (root) | `scripts/modeling_vit_classifier.py` (marked deprecated) |
| `onnx/` (8 old variants) | New `onnx/` (5 correct variants) + `onnx_legacy/` (old archive) |
| No legacy safetensors | `model_legacy.safetensors` (old weights) + `model_fixed.safetensors` (alias) |
| No `.gitignore` | `.gitignore` added |

### Documentation
- `README.md` — fully rewritten with fix notice, breaking change warning, ONNX guide, v2 teaser
- `AGENTS.md` — created for AI coding agent context
- `CHANGELOG.md` — this file

### Dependencies
- `transformers >= 5.4.0` now required (for `shortest_edge` resize support)

---

## Verification

The corrected model produces results closely matching the original timm-based training pipeline. Old vs new comparison across test images:

| Image | Old (broken) | New (fixed) | Diff |
|---|---|---|---|
| synthetic (600×400) | 0.15% fake | 0.14% fake | 0.0001% |
| real photo (portrait, 720×1280) | 6.58% fake | 3.31% fake | 0.0327% |
| AI-generated sample | 99.97% fake | 99.48% fake | 0.0049% |

The portrait image (720×1280) shows the largest deviation (0.033%) due to PIL vs torchvision interpolation differences on extreme aspect ratios during the resize-to-440 step. Both paths classify every image identically (same verdict). For square and near-square images, diffs are consistently <0.005%.

---

## Known limitations — ONNX

The FP16 and BNB4 quantized variants were dropped due to onnxconverter-common and onnxruntime API incompatibilities in the current toolchain. They will be added back when the tooling stabilizes. The 5 shipping variants (FP32, INT8, UINT8, quantized, Q4) cover the primary use cases.
