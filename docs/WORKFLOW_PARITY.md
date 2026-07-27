# Finalized workflow parity map

| ComfyUI stage | Final value/behavior | Direct Python equivalent |
| --- | --- | --- |
| `VHS_LoadVideo` | Native dimensions/rate; all frames; source audio | PyAV/FFmpeg decode plus `ffprobe` metadata |
| `CheckpointLoaderSimple` | `sam3.1_multiplex_fp16.safetensors` | Pinned Meta SAM 3.1 multiplex backend/checkpoint adapter |
| `CLIPTextEncode` | User text, reference value `man,hair,collar mic` | SAM 3.1 predictor text prompt |
| `SAM3_VideoTrack` | threshold `0.5`, max objects `8`, interval `1` | Backend tracking configuration |
| `SAM3_TrackToMask` | all tracked objects | Per-frame logical union |
| `VITMatteRefine` | erode `6`, dilate `6`, black `0.15`, white `0.99`, max `2 MP` | OpenCV trimap plus Transformers ViTMatte |
| `EmptyImage` | black, source dimensions | Zero-valued RGB background |
| `ImageCompositeMasked` | foreground over black | `foreground * alpha` |
| `VHS_VideoInfoSource` | source FPS | Probed rational FPS |
| `VHS_VideoCombine` | H.264, yuv420p, CRF 19, audio | FFmpeg preview encode/remux |

The reference workflow produces an opaque black-background MP4. Alpha-matte and transparent-master downloads are additive outputs; they must not change the parity preview.
