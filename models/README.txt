Local ML models  (Developed by Gourav Shrivastava)
==================================================
One sub-folder per model. Everything here runs offline.

Layouts
-------
1) ONNX (light, recommended)
     models/mymodel/model.onnx
     models/mymodel/classifier.json      (optional if HF metadata files are present)
2) Hugging Face / PyTorch (needs torch + transformers)
     models/mymodel/config.json, preprocessor_config.json, model.safetensors

classifier.json (all keys optional; auto-detected from config.json / preprocessor_config.json)
-----------------------------------------------------------------------------------------------
{
  "labels": ["Real", "Fake"],     # order of the model's output classes
  "fake_labels": ["Fake"],        # which of those mean fake
  "input_size": 224,              # or [height, width]
  "mean": [0.5, 0.5, 0.5],
  "std":  [0.5, 0.5, 0.5],
  "mode": "both",                 # "full" | "face" | "both"
  "tta": true                     # average with horizontal flip
}

mode
----
full : whole image only        -> generic AI-generated-image detectors
face : face crops only         -> face-swap / FaceForensics++-style models
both : max of the two (default)

Rules
-----
- Scores are UNCALIBRATED. Test every model on your own real and fake images.
- Models trained on one dataset often fail on newer generators.
- Multiple models are averaged; a large disagreement is flagged in the report.
- Check each model's license before redistributing or using commercially.

## Model

This project uses CommunityForensics DeepfakeDet-ViT (ViT-Small, MIT license),
from "Community Forensics: Using Thousands of Generators to Train Fake Image
Detectors" (CVPR 2025), https://huggingface.co/papers/2411.04125

Download the model with:

    python get_model.py