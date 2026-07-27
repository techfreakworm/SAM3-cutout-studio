# SAM3 Cutout Studio naming design

## Decision

The public product name is **SAM3 Cutout Studio**.

The canonical public repository identifier is **`techfreakworm/SAM3-cutout-studio`** on both GitHub and Hugging Face. Matching identifiers make commit-to-deployment traceability obvious and prevent the Space from drifting into a second product identity.

## Naming matrix

| Surface | Canonical value |
| --- | --- |
| Product title | `SAM3 Cutout Studio` |
| GitHub repository | `techfreakworm/SAM3-cutout-studio` |
| Hugging Face Space | `techfreakworm/SAM3-cutout-studio` |
| Python distribution | `sam3-cutout-studio` |
| Python import package | `sam3_matting` |
| Hinode checkout | `/home/wakeuser/Projects/llm/SAM3-cutout-studio` |

The import package remains `sam3_matting`: it describes the implementation domain, follows Python naming rules, and avoids breaking established module imports for a branding-only change.

## Rationale

`SAM3 Cutout Studio` follows the same model-plus-capability-plus-studio structure as Qwen Voice Studio while using a familiar user-facing term. “Cutout” describes the result without exposing the internal ViTMatte implementation or the original ComfyUI workflow filename. The title remains short enough for a Space card, browser tab, mobile header, and repository slug.

## Migration

The existing public GitHub repository is renamed in place so GitHub preserves redirects and history. The Hinode checkout directory, Git remote, README front matter, package metadata, design documents, UI copy, tests, and future Space deployment target are updated atomically. Historical workflow filenames under `reference/` remain unchanged because they document provenance rather than public branding.
