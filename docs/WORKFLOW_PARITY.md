# SAM3 Cutout Studio workflow parity map

| ComfyUI stage | Final value or behavior | Direct Python equivalent |
| --- | --- | --- |
| `VHS_LoadVideo` | Native dimensions/rate; all frames; source audio | PyAV/FFmpeg decode plus `ffprobe` metadata |
| `CheckpointLoaderSimple` | `sam3.1_multiplex_fp16.safetensors` | `Comfy-Org/sam3.1` file `checkpoints/sam3.1_multiplex_fp16.safetensors` at revision `ba901fbc9701054c359ed5240c4d76f83a178108` |
| `CLIPTextEncode` | User text; reference value `man,hair,collar mic` | Split into clauses and normalize standalone `mic` to `microphone`; effective prompts `man`, `hair`, `collar microphone` |
| `SAM3_VideoTrack` | threshold `0.5`, max objects `8`, interval `1` | Backend tracking configuration |
| `SAM3_TrackToMask` | all tracked objects | Per-frame logical union |
| `VITMatteRefine` | erode `6`, dilate `6`, five iterations; black `0.15`, white `0.99`, max `2 MP` | OpenCV trimap plus Transformers ViTMatte soft alpha and histogram remap |
| `EmptyImage` | black, source dimensions | Zero-valued RGB background |
| `ImageCompositeMasked` | foreground over black | `round(foreground * soft_alpha)` for the preview |
| `VHS_VideoInfoSource` | source FPS | Probed rational FPS used for sequential CFR timestamps |
| `VHS_VideoCombine` | H.264, yuv420p, CRF 19, audio | CFR H.264 preview plus source-audio stream copy |

## Pinned runtime identity

- SAM code: `facebookresearch/sam3@46957e47805eaa273f4aa7bbbd25a88bca9108ce`.
- SAM checkpoint: `Comfy-Org/sam3.1`, `checkpoints/sam3.1_multiplex_fp16.safetensors`, revision `ba901fbc9701054c359ed5240c4d76f83a178108`.
- SAM checkpoint SHA-256: `9ba99c92703c2e8b4f47de2d34a539bb8e18923049e238b780d70dbe6368eb03`.
- ViTMatte: `hustvl/vitmatte-small-composition-1k@6a58ad7646403c1df626fbd746900aec7361ea1d`.

The Space preloads the exact SAM checkpoint and the ViTMatte `config.json`, `model.safetensors`, and `preprocessor_config.json` files. That describes Hub artifact caching, not per-request construction. On ZeroGPU, the first request builds one SAM predictor and one ViTMatte model inside the persistent GPU worker, and later requests reuse those resident instances; local CUDA startup eagerly materializes the same pair in module-level resources instead.

## Hosted and local deltas

Parity settings are shared, but deployment admission policy differs deliberately:

| Policy | Hosted ZeroGPU | Local CUDA |
| --- | --- | --- |
| Prompt clauses | 3 maximum | 4 maximum |
| Upload cap | 100 MiB | 2 GiB |
| Input envelope | 2.0 s, 60 frames, 30 fps, long edge <= 1920 and short edge <= 1080 | 120 s, 7,200 frames, 60 fps, width/height <= 4096 |
| Login | Hugging Face OAuth profile required | none at app level |
| Gradio API | disabled with `api_open=False` | disabled with `api_open=False` |
| GPU boundary | xlarge 96 GB ZeroGPU, 120-second duration request | operator CUDA device |

These limits determine whether a request is admitted; they do not alter the accepted request's tracking, trimap, ViTMatte, compositing, or encoding values.

## Additive outputs

The reference workflow produces an opaque black-background MP4. The Python pipeline adds two downloads without changing that parity preview:

- A ProRes 4444 transparent master receives RGBA `uint8` frames. ViTMatte soft alpha is rounded to 8-bit before it reaches the encoder, so a high-bit-depth ProRes pixel format is not evidence of genuine 16-bit source alpha.
- A silent H.264/yuv420p matte receives the same 8-bit alpha repeated across RGB channels. H.264 is lossy, so decoded matte values need not be byte-identical to the master alpha plane.

All three outputs are CFR at the probed rational source rate. Source audio is stream-copied only into the preview and master; the matte intentionally has no audio.

See [`CUDA_VALIDATION.md`](CUDA_VALIDATION.md) for one dated, measured parity run. Its timings, memory use, artifact sizes, and quality comparisons are evidence for that run rather than universal guarantees.
