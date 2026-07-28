# SAM3 Cutout Studio third-party notices

The repository-level MIT License applies only to original SAM3 Cutout Studio application code and project-authored material. It does not relicense third-party models, model code, checkpoints, codecs, fonts, or other dependencies.

| Component | Exact upstream identity | License | Use and distribution note |
| --- | --- | --- | --- |
| SAM 3.1 model code | `facebookresearch/sam3@46957e47805eaa273f4aa7bbbd25a88bca9108ce` | SAM License | Pinned source dependency; upstream license copied under `licenses/` |
| SAM 3.1 runtime checkpoint | `Comfy-Org/sam3.1`, `checkpoints/sam3.1_multiplex_fp16.safetensors`, revision `ba901fbc9701054c359ed5240c4d76f83a178108` | SAM License | Preloaded/resolved through the Hub disk cache; never committed to this repository |
| ViTMatte code | `hustvl/ViTMatte` | MIT | Used through the Transformers integration |
| ViTMatte small Composition-1k weights | `hustvl/vitmatte-small-composition-1k@6a58ad7646403c1df626fbd746900aec7361ea1d` | Apache-2.0 | Exact `config.json`, `model.safetensors`, and `preprocessor_config.json` are preloaded from the Hub; never committed to this repository |
| FFmpeg and ffprobe | `ffmpeg.org` | Build-dependent LGPL/GPL | External executables invoked for probing, media I/O, encoding, and remuxing |
| Finalized ComfyUI workflow JSON | Project-authored workflow | MIT | Behavioral reference only; ComfyUI is not bundled or required at runtime |

The pinned SAM checkpoint SHA-256 is:

```text
9ba99c92703c2e8b4f47de2d34a539bb8e18923049e238b780d70dbe6368eb03
```

Hub preload, local caching, or eager in-memory model residency does not change any upstream license. Users and deployers are responsible for reviewing and complying with the licenses, acceptable-use terms, attribution requirements, and restrictions that apply to their chosen models, codec build, and deployment.
