---
title: SAM3 Cutout Studio
emoji: 🎬
colorFrom: gray
colorTo: yellow
sdk: gradio
sdk_version: 6.20.0
app_file: app.py
python_version: 3.12.12
fullWidth: true
license: mit
models:
  - facebook/sam3.1
  - hustvl/vitmatte-small-composition-1k
---

# SAM3 Cutout Studio

Text-guided video cutouts using SAM 3.1 Object Multiplex tracking and ViTMatte alpha refinement. The application is a Gradio-only studio designed for local CUDA, Hugging Face ZeroGPU, and an MPS pathway for high-memory Apple-silicon Macs.

The source of truth is this public GitHub repository. Model weights are downloaded from their upstream repositories and retain their upstream licenses.

## Planned output contract

- H.264 MP4 preview composited over black, preserving source frame rate and audio.
- Downloadable alpha matte.
- Downloadable transparent master when the selected encoder supports alpha.

## Architecture

```text
Gradio
  -> media probe and frame decoder
  -> platform-selected SAM backend
  -> mask union and trimap generation
  -> Transformers ViTMatte refinement
  -> preview/master encoders and audio remux
```

ComfyUI is not an application dependency. The finalized ComfyUI workflow is committed under `reference/` solely as the behavioral reference used for parity testing.

## Development workstation

CUDA development and testing take place on the Hinode RTX PRO 6000 Blackwell workstation using Python 3.12.12 and PyTorch 2.11.0. MPS validation follows CUDA end-to-end validation and uses the same repository through the backend/device seam.

## Licenses

The application code is MIT licensed. SAM 3.1 code and checkpoints are governed by Meta's separate SAM License. ViTMatte upstream code and weights retain their respective upstream licenses. See `THIRD_PARTY_NOTICES.md` and `licenses/`.
