---
title: SAM3 Cutout Studio
short_description: Text-guided SAM 3.1 video cutouts with ViTMatte alpha.
emoji: 🎬
colorFrom: gray
colorTo: yellow
sdk: gradio
sdk_version: 6.20.0
app_file: app.py
python_version: 3.12.12
fullWidth: true
license: mit
hf_oauth: true
startup_duration_timeout: 1h
models:
  - Comfy-Org/sam3.1
  - hustvl/vitmatte-small-composition-1k
preload_from_hub:
  - Comfy-Org/sam3.1 checkpoints/sam3.1_multiplex_fp16.safetensors ba901fbc9701054c359ed5240c4d76f83a178108
  - hustvl/vitmatte-small-composition-1k config.json,model.safetensors,preprocessor_config.json 6a58ad7646403c1df626fbd746900aec7361ea1d
---

# SAM3 Cutout Studio

SAM3 Cutout Studio turns short videos into text-guided cutouts using SAM 3.1 Object Multiplex tracking and ViTMatte alpha refinement. It produces an opaque preview, a transparent editing master, and an alpha matte from one upload.

The public source of truth is [`techfreakworm/SAM3-cutout-studio`](https://github.com/techfreakworm/SAM3-cutout-studio) on GitHub. The hosted release uses the same public identifier at [`techfreakworm/SAM3-cutout-studio`](https://huggingface.co/spaces/techfreakworm/SAM3-cutout-studio) on Hugging Face. The Space remains private while a release is staged and is made public only after the complete deployed test gate passes.

CUDA and Hugging Face ZeroGPU are the implemented runtimes. Apple MPS is planned, but it is not supported or verified yet.

## Outputs

- **Preview:** H.264/yuv420p MP4 composited over black, with source audio stream-copied when present.
- **Transparent master:** ProRes 4444 MOV with source audio stream-copied when present.
- **Alpha matte:** silent H.264/yuv420p MP4 for inspection and downstream use.

Every output is constant frame rate (CFR) at the probed source rate. Frames receive sequential timestamps; arbitrary variable-frame-rate timestamps are not preserved.

ViTMatte produces a soft floating-point alpha. The preview is composited from that soft alpha, while the master alpha and standalone matte are rounded to `uint8` before encoding. ProRes 4444 preserves transparency, but its high-bit-depth pixel format does not turn the 8-bit source alpha into genuine 16-bit alpha.

## Hosted ZeroGPU contract

The hosted app requires Hugging Face sign-in. It uses an xlarge ZeroGPU allocation (the full 96 GB RTX PRO 6000 Blackwell slice) with a 60-second GPU duration request. One inference runs at a time, at most eight jobs may wait, and public Gradio API exposure is disabled with `api_open=False`; the supported hosted entry point is the browser UI.

Hosted input is rejected before model inference unless it satisfies all of these limits:

| Limit | Hosted value |
| --- | ---: |
| Prompt clauses | At most 3 comma-delimited clauses |
| Duration | At most 2.0 seconds |
| Frames | At most 60 |
| Frame rate | At most 30 fps |
| Canvas | Long edge at most 1920 px and short edge at most 1080 px |
| Upload | At most 100 MiB (104,857,600 bytes) |

The 100 MiB ceiling is enforced both by the Gradio launch boundary and by application validation. A successful upload within these limits is not a promise that every possible scene will complete in a fixed time.

## Local CUDA contract

Local execution accepts at most four prompt clauses. Its Gradio launch and validation cap is 2 GiB (2,147,483,648 bytes). The local validation envelope is at most 120 seconds, 7,200 frames, 60 fps, and 4096 x 4096 pixels. These are admission limits, not performance guarantees.

### Prerequisites

- Python 3.12 (the project requires `>=3.12,<3.13`);
- an NVIDIA CUDA GPU with a driver compatible with the pinned PyTorch build;
- `ffmpeg` and `ffprobe` available on `PATH`;
- Git, network access for the pinned model artifacts, and enough disk space for model caches and rendered outputs; and
- acceptance of the upstream model licenses described in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

### Install, run, and test

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python app.py
```

Gradio serves locally on its configured address (normally port 7860). Install the development tools and run the regression checks with:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
python -m ruff check .
```

`SAM3_CHECKPOINT` may point to an operator-provided checkpoint file. `SAM3_OUTPUT_DIR` may point to an operator-managed output directory. Standard Gradio environment variables can be used to configure the server address and port.

## Pinned models and startup

| Runtime artifact | Pin |
| --- | --- |
| SAM model code | `facebookresearch/sam3@46957e47805eaa273f4aa7bbbd25a88bca9108ce` |
| SAM checkpoint | `Comfy-Org/sam3.1`, `checkpoints/sam3.1_multiplex_fp16.safetensors`, revision `ba901fbc9701054c359ed5240c4d76f83a178108` |
| SAM checkpoint SHA-256 | `9ba99c92703c2e8b4f47de2d34a539bb8e18923049e238b780d70dbe6368eb03` |
| ViTMatte | `hustvl/vitmatte-small-composition-1k@6a58ad7646403c1df626fbd746900aec7361ea1d` |

Space startup preloads the exact SAM checkpoint and ViTMatte `config.json`, `model.safetensors`, and `preprocessor_config.json` into the Hub disk cache. That file staging does not instantiate a model. On ZeroGPU, model construction happens lazily inside the persistent GPU worker on the first request, because built models cannot cross the worker's pickle boundary; the first request therefore includes one-time model construction, and later requests reuse the resident models. Local CUDA startup instead eagerly materializes one process-wide SAM backend and one ViTMatte refiner for queued requests. `startup_duration_timeout: 1h` gives a cold Space enough time to warm up dependencies and caches.

## Output retention and privacy

Without `SAM3_OUTPUT_DIR`, accepted jobs write beneath Gradio's temporary/cache root (`GRADIO_TEMP_DIR`, or the system temporary Gradio directory) in a `sam3-cutout-studio` subdirectory. The app configures `delete_cache=(600, 3600)`: Gradio checks its cache every 600 seconds and removes cached files older than 3,600 seconds. Failed jobs remove their working directory immediately; successful output paths remain available for download until cache retention removes them or the service restarts.

Setting `SAM3_OUTPUT_DIR` opts out of the default Gradio-managed location. The operator is then responsible for access controls, retention, backup, and deletion for that directory.

Uploads are transmitted to and processed on the host. Login limits who can use the hosted UI, but neither login nor eventual cache cleanup is a confidential-storage or immediate-deletion guarantee. Do not upload sensitive media without an appropriate deployment and retention policy.

## Architecture and validation

```text
Gradio upload and validation
  -> CFR frame decoder
  -> CUDA SAM 3.1 multiplex tracking
  -> mask union and trimap generation
  -> Transformers ViTMatte refinement
  -> preview, transparent master, and matte encoders
  -> source-audio remux for preview and master
```

ComfyUI is not an application dependency. The finalized workflow under [`reference/`](reference/) is the behavioral reference documented in [`docs/WORKFLOW_PARITY.md`](docs/WORKFLOW_PARITY.md).

See the dated [`docs/CUDA_VALIDATION.md`](docs/CUDA_VALIDATION.md) snapshot for measured CUDA and browser evidence. Those timings, memory figures, and artifact sizes describe one verified run, not a service-level guarantee.

## Licenses

The repository MIT license covers original SAM3 Cutout Studio application code only. It does not relicense third-party model code, weights, codecs, or other dependencies. Review [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) and [`licenses/`](licenses/) before use or redistribution.
