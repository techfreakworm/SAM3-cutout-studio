# SAM3 Cutout Studio design

## Priorities

1. Preserve the behavior of the finalized ComfyUI background-removal workflow.
2. Provide a production-shaped Gradio studio for local CUDA and hosted ZeroGPU use.
3. Keep one implemented pipeline behind a backend seam that can accept a future MPS adapter.
4. Make runtime limits, model pins, output precision, storage behavior, and license boundaries explicit.

## Pipeline

```text
Video upload
  -> probe and validate metadata
  -> decode RGB frames at a selected rational CFR
  -> text-prompted SAM 3.1 multi-object tracking
  -> union selected object masks
  -> create trimap (threshold, erosion, dilation)
  -> ViTMatte per-frame soft-alpha refinement
  -> floating-point histogram remap
  -> soft-alpha preview composite and 8-bit alpha quantization
  -> CFR preview, matte, and ProRes 4444 master
  -> source-audio remux into preview and master
```

The workflow values are compatibility inputs: detection threshold `0.5`, maximum objects `8`, detection interval `1`, erosion kernel `6`, dilation kernel `6`, five morphology iterations, black point `0.15`, white point `0.99`, and a `2.0` megapixel ViTMatte processing budget.

## Platform seam

The orchestration layer depends on a small segmentation-backend protocol rather than on a model imported by request code.

- `MetaSam31Backend` implements CUDA and ZeroGPU using Meta's multiplex video predictor.
- `runtime.gpu` applies `spaces.GPU(size="xlarge", duration=90)` only when the Space reports ZeroGPU; it is a no-op for local CUDA.
- ViTMatte, trimap construction, media handling, compositing, validation, cleanup, and the Gradio UI are shared by both implemented paths.

MPS is planned, not supported. Device probing can identify MPS, but `MetaSam31Backend` is explicitly CUDA-only. A separate adapter and successful end-to-end Apple-silicon validation are required before public MPS support can be claimed.

## Artifact preload and model residency

The Space frontmatter pins and preloads these files into the Hugging Face Hub disk cache:

- `Comfy-Org/sam3.1` file `checkpoints/sam3.1_multiplex_fp16.safetensors` at revision `ba901fbc9701054c359ed5240c4d76f83a178108`;
- `hustvl/vitmatte-small-composition-1k` files `config.json`, `model.safetensors`, and `preprocessor_config.json` at revision `6a58ad7646403c1df626fbd746900aec7361ea1d`.

Hub preload is file staging only. Local CUDA startup creates `ApplicationResources` with eager preload enabled, materializes one SAM predictor and one ViTMatte model, and retains both in module-level `RESOURCES`; requests reuse these process-wide instances. Per-request ViTMatte settings are applied under the shared resource lock and restored afterward.

ZeroGPU differs deliberately. GPU tasks execute in a persistent worker process and only pickled values cross its boundary, so built models cannot leave it. The first request constructs both models inside that worker through `sam3_matting.zerogpu_worker`, and later requests reuse the resident instances. Startup performs no eager preload on ZeroGPU, and the request lease duration is sized to cover one-time construction plus one inference.

A one-hour Space `startup_duration_timeout` accommodates cold dependency and cache warmup. It is a startup ceiling, not an expected boot time.

## Hosted and local policy

| Boundary | Hosted ZeroGPU | Local CUDA |
| --- | --- | --- |
| Authentication | Hugging Face OAuth profile required | No app-level login |
| Prompt clauses | 3 maximum | 4 maximum |
| GPU allocation | xlarge, full 96 GB RTX PRO 6000 Blackwell slice | Operator GPU |
| GPU duration request | 90 seconds | Not applicable |
| Upload launch/validation cap | 100 MiB | 2 GiB |
| Video duration | 2.0 seconds | 120 seconds |
| Frame count | 60 | 7,200 |
| Frame rate | 30 fps | 60 fps |
| Canvas | long edge <= 1920, short edge <= 1080 | width and height <= 4096 |
| Public Gradio API | disabled (`api_open=False`) | disabled (`api_open=False`) |

The UI admits one inference at a time and holds at most eight queued jobs. Hosted metadata validation and OAuth-profile validation complete before a job workspace is created or SAM/ViTMatte inference begins. Gradio also rejects oversized uploads at the launch boundary.

Only the inference entry point receives the ZeroGPU decorator. Values crossing the queue boundary are ordinary control values plus uploaded or generated file paths. Process-wide models and locks may exist in the serving process or a forked ZeroGPU worker; no stronger process-isolation claim is made. SAM sessions and media file handles are created and closed inside inference, including `finally` cleanup. Live model, session, decoder, or encoder objects are never returned through the queue.

## Timing and alpha precision

The encoder assigns sequential integer frame numbers at one probed rational rate. Generated videos are CFR and retain frame count and rate, not arbitrary variable-frame-rate timestamps.

ViTMatte produces floating-point soft alpha and histogram remapping remains floating point. The black-background preview is composited from soft alpha. The standalone matte and RGBA master receive `round(alpha * 255)` as `uint8`; a high-bit-depth ProRes 4444 pixel format must therefore not be described as genuine 16-bit source alpha.

## Output lifecycle and privacy

Each accepted request receives a unique `job-*` directory. By default that directory lives in `sam3-cutout-studio` beneath `GRADIO_TEMP_DIR`, or beneath the system temporary Gradio cache directory when `GRADIO_TEMP_DIR` is unset. This keeps default outputs inside Gradio's temporary/cache root.

`gr.Blocks(delete_cache=(600, 3600))` asks Gradio to check every 600 seconds and remove cached files older than 3,600 seconds. Failed jobs remove their working directory immediately. Successful paths remain available for download until cache retention or server restart removes them; cleanup is eventual, not immediate.

An explicit `SAM3_OUTPUT_DIR` moves application outputs outside that default location. Its access controls, retention, backup, cleanup, and deletion are entirely operator-managed.

Uploads are processed on the serving host. OAuth controls access to the hosted UI, but it does not make temporary storage confidential, guarantee immediate deletion, or replace a deployment-specific data policy.

## Release topology

[`techfreakworm/SAM3-cutout-studio`](https://github.com/techfreakworm/SAM3-cutout-studio) is the public source of truth. The matching Hugging Face Space ID is [`techfreakworm/SAM3-cutout-studio`](https://huggingface.co/spaces/techfreakworm/SAM3-cutout-studio). A candidate Space remains private during staging; public visibility is enabled only after the full deployed browser, API-boundary, artifact, and runtime checks pass.

## Verification gates

- Dated CUDA, artifact, unit/integration, and Playwright evidence is recorded in [`CUDA_VALIDATION.md`](CUDA_VALIDATION.md).
- Pure and synthetic-media tests cover device policy, prompt handling, validation, trimap generation, timing, dimensions, audio, preview, master, matte, and cleanup behavior.
- CUDA golden-run comparison uses the finalized ComfyUI reference workflow.
- A staged private Space must pass hosted OAuth, browser upload, decoded-output, viewport, console, and disabled-API checks before it is made public.
- A representative MPS render is required before MPS can move from roadmap to supported status.
