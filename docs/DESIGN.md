# Design

## Priorities

1. Fidelity to the finalized `sam3_pro_vitmatte_bg_removal` workflow.
2. A complete, production-shaped Gradio experience rather than a demo wrapper.
3. One shared pipeline across CUDA/ZeroGPU and MPS, with only the SAM backend and device policy varying.
4. Public, reproducible source with pinned upstream revisions and explicit license provenance.

## Pipeline

```text
Video upload
  -> probe and validate metadata
  -> decode RGB frames and retain source timing/audio
  -> text-prompted SAM 3.1 multi-object tracking
  -> union selected object masks
  -> create trimap (threshold, erosion, dilation)
  -> ViTMatte per-frame alpha refinement
  -> histogram remap
  -> preview composite and downloadable matte/master
  -> remux source audio
```

The workflow values are part of the compatibility contract: detection threshold `0.5`, maximum objects `8`, detection interval `1`, erosion kernel `6`, dilation kernel `6`, five morphology iterations, black point `0.15`, white point `0.99`, and a `2.0` megapixel ViTMatte processing budget.

## Platform seam

The orchestration layer depends on a small segmentation-backend protocol rather than importing a concrete model globally.

- `MetaSam31Backend`: primary CUDA/ZeroGPU implementation using Meta's multiplex video predictor.
- MPS backend: implemented after CUDA parity. Meta's current predictor contains CUDA-specific operations, so MPS is a distinct adapter rather than a string replacement.
- ViTMatte, trimap, media, compositing, validation, cleanup, and Gradio code remain shared.

Device selection order for `auto` is CUDA, then MPS, then CPU. Explicit unavailable devices fail with an actionable message.

## ZeroGPU boundary

Only the inference entry point receives the `spaces.GPU` decorator. Metadata probing and input validation run before requesting a GPU. Jobs use unique temporary directories, concurrency one, dynamic GPU duration based on decoded workload, and cleanup in `finally`.

No live CUDA tensor, predictor session, file handle, or lock crosses the ZeroGPU process boundary. Outputs are CPU data or paths beneath the request workspace.

## Studio UI

The interface follows the local LTX and Qwen studio family without copying either application: a dark optical-workbench surface, warm amber action signal, native Gradio controls, two-column input/output rhythm, a compact machine-status rail, and progressive disclosure for tracking/matting controls.

## Verification gates

- Pure unit tests for device policy, trimap generation, duration estimation, validation, and cleanup.
- Synthetic media integration tests for frame count, FPS, dimensions, audio, preview, and alpha outputs.
- CUDA golden-run comparison against the finalized ComfyUI workflow.
- Playwright validation of the live Hinode Gradio application.
- MPS smoke and representative render on the 128 GB MacBook.
- Deployed ZeroGPU API and browser smoke tests before release.
