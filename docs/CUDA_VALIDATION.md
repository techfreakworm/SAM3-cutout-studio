# CUDA validation snapshot — 2026-07-28

This document records one verified end-to-end CUDA run of SAM3 Cutout Studio. It is dated evidence for a specific input, host, source commit, dependency state, model state, and cache condition—not a throughput, latency, memory, quality, availability, or output-size guarantee.

## Source and model identity

| Item | Verified value |
| --- | --- |
| Application source commit | `1f3dc19e5c97b0eec4e3e19a33cfc8f09dcc6bf2` |
| SAM code commit | `46957e47805eaa273f4aa7bbbd25a88bca9108ce` |
| SAM repository | `Comfy-Org/sam3.1` |
| SAM checkpoint | `checkpoints/sam3.1_multiplex_fp16.safetensors` |
| SAM revision | `ba901fbc9701054c359ed5240c4d76f83a178108` |
| SAM checkpoint SHA-256 | `9ba99c92703c2e8b4f47de2d34a539bb8e18923049e238b780d70dbe6368eb03` |
| ViTMatte repository | `hustvl/vitmatte-small-composition-1k` |
| ViTMatte revision | `6a58ad7646403c1df626fbd746900aec7361ea1d` |

The repository MIT license applies to original application code only. SAM code and checkpoint use the SAM License, and ViTMatte code and weights retain their upstream licenses.

## Runtime identity

| Item | Verified value |
| --- | --- |
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| GPU memory | 97,887 MiB |
| NVIDIA driver | 580.159.03 |
| Python | 3.12.12 |
| PyTorch | 2.11.0+cu128 |
| CUDA runtime | 12.8 |
| cuDNN | 91900 |
| FFmpeg | 8.1.2 |
| ffprobe | 8.1.2 |

## Benchmark conditions

The application was started in a fresh process. Required Hub files were already warm in the on-disk cache. SAM and ViTMatte were then eagerly preloaded and kept resident before the first request. The 10.244-second model preload is reported separately and excluded from both request timings below.

| Input property | Verified value |
| --- | --- |
| SHA-256 | `6e79f17491936dbeac68cc220c4886834ac7623ac6618c872604ac84ba78ee77` |
| Dimensions | 1080 x 1920 |
| Duration | 2.000 seconds |
| Frame rate | 30 fps |
| Frame count | 60 |
| Audio | AAC |
| Entered prompt | `man,hair,collar mic` |
| Effective SAM clauses | `man`, `hair`, `collar microphone` |

The prompt parser split the comma-delimited value and normalized the standalone word `mic` to `microphone` before inference.

## Measured execution

| Metric | Measured value |
| --- | ---: |
| Eager model preload, excluded from request | 10.244 s |
| First-request wall time | 42.806 s |
| First-request pipeline time | 42.40 s |
| Peak CUDA memory allocated | 22,133,509,120 B (20.613 GiB) |
| Peak CUDA memory reserved | 29,408,362,496 B (27.389 GiB) |

These values describe the stated fresh-process/warm-disk-cache condition and this input. Cache state, drivers, GPU allocation, scene content, resolution, frame count, and service contention can change every measurement.

## Artifact evidence

| Artifact | Video stream | Timing | Size | Audio |
| --- | --- | --- | ---: | --- |
| Preview | H.264, `yuv420p` | 2.000 s, 30 fps CFR, 60 frames | 1,586,907 B | AAC |
| Transparent master | ProRes, `yuva444p12le` | 2.000 s, 30 fps CFR, 60 frames | 85,499,621 B | AAC |
| Alpha matte | H.264, `yuv420p` | 2.000 s, 30 fps CFR, 60 frames | 231,115 B | silent |

The source, preview, and transparent-master AAC packet payloads all had SHA-256 `27459971ff3ed37a56f400e91c260174cade210b22f22711505527471d0931b2`, recording byte-identical source-audio remuxing for this run. All three video artifacts completed a full decode without reported errors.

The decoded master alpha plane and decoded standalone matte measured **51.164 dB PSNR**. This checks consistency between two encoded derivatives; it is not a ground-truth segmentation or matting quality score. The decoded master alpha spanned 0 through 255 and contained 255 distinct levels. ViTMatte generated soft floating-point alpha, but the pipeline rounded it to `uint8` before master and matte encoding; `yuva444p12le` therefore describes the encoded representation, not 12-bit or 16-bit source-alpha precision.

## Automated and browser verification

The repository suite reported:

```text
271 passed, 1 skipped
```

The skip is an explicitly opt-in test. The count is a dated regression signal, not a substitute for rerunning tests after source, dependency, driver, model, or platform changes.

A Playwright run against the real application recorded:

- a real upload completed in 43.41 seconds;
- desktop and 390-pixel mobile viewports had no horizontal overflow;
- the browser console reported zero errors;
- a direct processing POST returned HTTP 404, consistent with `api_open=False`; and
- all browser-produced outputs completed a full decode.

## Scope boundaries

- This run validates the local CUDA path and stated artifacts for one input.
- It does not guarantee hosted queue time, ZeroGPU allocation latency, lease completion, or performance on other inputs.
- Hosted acceptance limits are 2.0 seconds, 60 frames, 30 fps, and an orientation-neutral 1080 x 1920 canvas; admission within them is not an SLA.
- MPS remains planned and unverified. This snapshot provides no Apple-silicon compatibility evidence.
- CFR output preserves the selected rate and frame count through sequential timestamps; it does not preserve arbitrary variable-frame-rate source timestamps.
