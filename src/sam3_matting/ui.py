"""Production Gradio shell for SAM3 Cutout Studio."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from html import escape
from pathlib import Path
from typing import Any

import gradio as gr

StudioProcessFn = Callable[
    [str, str, float, int, int, int, int, float, float, float],
    tuple[str, str, str, str],
]

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_STUDIO_CSS = _PROJECT_ROOT / "assets" / "studio.css"

_FONT_HEAD = """
<meta name="color-scheme" content="dark">
<!-- Display: Bricolage Grotesque · Interface: IBM Plex Sans / Mono -->
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link
  href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,600;12..96,700&family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap"
  rel="stylesheet"
>
"""

_DEFAULT_RUNTIME_STATUS = {
    "device": "AUTO · accelerator selected per host",
    "cuda": "Primary",
    "mps": "Portable",
    "zerogpu": "Compatible",
}


def _runtime_status_html(runtime_status: Mapping[str, str] | None) -> str:
    status = {**_DEFAULT_RUNTIME_STATUS, **(runtime_status or {})}
    device = escape(str(status["device"]))
    cuda = escape(str(status["cuda"]))
    mps = escape(str(status["mps"]))
    zerogpu = escape(str(status["zerogpu"]))
    return f"""
    <section class="runtime-rail" aria-label="Runtime compatibility">
      <div class="runtime-device">
        <span class="signal-lamp" aria-hidden="true"></span>
        <span class="runtime-label">Active device</span>
        <strong>{device}</strong>
      </div>
      <div class="runtime-readout">
        <span class="runtime-label">CUDA</span>
        <strong>{cuda}</strong>
      </div>
      <div class="runtime-readout">
        <span class="runtime-label">MPS</span>
        <strong>{mps}</strong>
      </div>
      <div class="runtime-readout">
        <span class="runtime-label">ZeroGPU</span>
        <strong>{zerogpu}</strong>
      </div>
      <div class="runtime-tick" aria-hidden="true">01 / 03</div>
    </section>
    """


def _studio_theme() -> gr.Theme:
    return gr.themes.Base(
        primary_hue=gr.themes.colors.amber,
        secondary_hue=gr.themes.colors.orange,
        neutral_hue=gr.themes.colors.slate,
        radius_size=gr.themes.sizes.radius_sm,
        spacing_size=gr.themes.sizes.spacing_lg,
        font=("IBM Plex Sans", "ui-sans-serif", "sans-serif"),
        font_mono=("IBM Plex Mono", "ui-monospace", "monospace"),
    )


def studio_launch_kwargs() -> dict[str, Any]:
    """Return Gradio 6 launch-time presentation settings."""
    return {
        "theme": _studio_theme(),
        "css_paths": _STUDIO_CSS,
        "head": _FONT_HEAD,
        "footer_links": [],
    }


def build_ui(
    process_fn: StudioProcessFn,
    runtime_status: Mapping[str, str] | None = None,
) -> gr.Blocks:
    """Build the studio and bind its sole inference event to ``process_fn``."""
    with gr.Blocks(
        title="SAM3 Cutout Studio",
        analytics_enabled=False,
        fill_width=True,
        delete_cache=(86_400, 86_400),
    ) as demo:
        gr.HTML(
            """
            <header class="darkroom-masthead">
              <div class="optical-mark" aria-hidden="true">
                <span class="optical-iris"></span>
                <span class="optical-cross optical-cross-x"></span>
                <span class="optical-cross optical-cross-y"></span>
              </div>
              <div class="masthead-copy">
                <div class="frame-code">
                  <span>SAM 3.1 / VITMATTE</span>
                  <span>OPTICAL WORKBENCH · 001</span>
                </div>
                <h1>SAM3<br><em>Cutout Studio</em></h1>
                <p>Text-guided separation for difficult edges, motion, and fine detail.</p>
              </div>
              <div class="masthead-spec" aria-label="Output specification">
                <span>TRACK</span><strong>MULTIPLEX</strong>
                <span>REFINE</span><strong>ALPHA / 16-BIT</strong>
                <span>MASTER</span><strong>RGBA + MATTE</strong>
              </div>
            </header>
            """,
            elem_id="studio-masthead",
            container=False,
        )
        gr.HTML(
            _runtime_status_html(runtime_status),
            elem_id="runtime-status",
            container=False,
        )

        with gr.Row(elem_id="studio-deck", elem_classes=["studio-deck"], equal_height=False):
            with gr.Column(
                scale=5,
                min_width=340,
                elem_id="input-bay",
                elem_classes=["studio-panel", "input-panel"],
            ):
                gr.HTML(
                    """
                    <div class="section-heading">
                      <span class="section-index">01</span>
                      <div><p>Source &amp; direction</p><h2>Acquire subject</h2></div>
                      <span class="section-rule" aria-hidden="true"></span>
                    </div>
                    """,
                    container=False,
                )
                source_video = gr.Video(
                    label="Source footage",
                    sources=["upload"],
                    include_audio=True,
                    format=None,
                    elem_id="studio-video",
                    elem_classes=["optical-input"],
                )
                subject_prompt = gr.Textbox(
                    label="Subject prompt",
                    placeholder="person, hair, collar mic",
                    info="Name the foreground subject and edge details worth preserving.",
                    lines=2,
                    max_lines=4,
                    elem_id="studio-prompt",
                )
                run_button = gr.Button(
                    "Extract foreground  →",
                    variant="primary",
                    size="lg",
                    elem_id="studio-run",
                )
                job_status = gr.Markdown(
                    "**READY** · Load a clip and describe the foreground subject.",
                    elem_id="job-status",
                    elem_classes=["job-status"],
                    container=False,
                )
                gr.HTML(
                    """
                    <div class="process-track" aria-label="Processing stages">
                      <span><b>01</b> Track</span>
                      <i aria-hidden="true"></i>
                      <span><b>02</b> Refine</span>
                      <i aria-hidden="true"></i>
                      <span><b>03</b> Export</span>
                    </div>
                    """,
                    container=False,
                )

                with gr.Accordion(
                    "Advanced · tracking and edge optics",
                    open=False,
                    elem_id="advanced-controls",
                ):
                    gr.Markdown(
                        "Tracking values mirror the finalized ComfyUI workflow. "
                        "Change them only when the shot needs intervention.",
                        elem_classes=["control-note"],
                    )
                    with gr.Row():
                        detection_threshold = gr.Slider(
                            0.05,
                            0.95,
                            value=0.5,
                            step=0.01,
                            label="Detection threshold",
                            elem_id="detection-threshold",
                        )
                        max_objects = gr.Slider(
                            1,
                            8,
                            value=8,
                            step=1,
                            label="Maximum objects",
                            elem_id="max-objects",
                        )
                    detect_interval = gr.Slider(
                        1,
                        30,
                        value=1,
                        step=1,
                        label="Detection interval",
                        info="Frames between fresh text detections; 1 gives maximum fidelity.",
                        elem_id="detect-interval",
                    )
                    with gr.Row():
                        erode_kernel = gr.Slider(
                            1,
                            31,
                            value=6,
                            step=1,
                            label="Erode kernel",
                            elem_id="erode-kernel",
                        )
                        dilate_kernel = gr.Slider(
                            1,
                            31,
                            value=6,
                            step=1,
                            label="Dilate kernel",
                            elem_id="dilate-kernel",
                        )
                    with gr.Row():
                        black_point = gr.Slider(
                            0.0,
                            0.9,
                            value=0.15,
                            step=0.01,
                            label="Black point",
                            elem_id="black-point",
                        )
                        white_point = gr.Slider(
                            0.1,
                            1.0,
                            value=0.99,
                            step=0.01,
                            label="White point",
                            elem_id="white-point",
                        )
                    max_megapixels = gr.Slider(
                        0.25,
                        4.0,
                        value=2.0,
                        step=0.25,
                        label="ViTMatte megapixel budget",
                        info="Higher values preserve more edge detail and use more accelerator memory.",
                        elem_id="max-megapixels",
                    )

            with gr.Column(
                scale=7,
                min_width=380,
                elem_id="output-bay",
                elem_classes=["studio-panel", "output-panel"],
            ):
                gr.HTML(
                    """
                    <div class="section-heading output-heading">
                      <span class="section-index">02</span>
                      <div><p>Review &amp; delivery</p><h2>Inspect the separation</h2></div>
                      <span class="section-rule" aria-hidden="true"></span>
                    </div>
                    """,
                    container=False,
                )
                preview = gr.Video(
                    label="Composite preview",
                    interactive=False,
                    autoplay=False,
                    buttons=["download"],
                    elem_id="result-preview",
                    elem_classes=["matte-preview"],
                )
                gr.HTML(
                    """
                    <div class="preview-legend" aria-label="Preview legend">
                      <span><i class="legend-chip legend-alpha"></i> refined alpha edge</span>
                      <span><i class="legend-chip legend-black"></i> black inspection field</span>
                      <span>source timing + audio retained</span>
                    </div>
                    """,
                    container=False,
                )
                with gr.Row(elem_id="delivery-row", elem_classes=["delivery-row"]):
                    transparent_master = gr.File(
                        label="Transparent master",
                        interactive=False,
                        elem_id="result-master",
                        elem_classes=["delivery-file", "master-file"],
                    )
                    alpha_matte = gr.File(
                        label="Alpha matte",
                        interactive=False,
                        elem_id="result-matte",
                        elem_classes=["delivery-file", "matte-file"],
                    )
                gr.HTML(
                    """
                    <div class="delivery-notes">
                      <p><span>MASTER</span> Straight-alpha RGBA for finishing and compositing.</p>
                      <p><span>MATTE</span> Standalone monochrome alpha for inspection or relighting.</p>
                    </div>
                    """,
                    container=False,
                )

        gr.HTML(
            """
            <footer class="studio-footer">
              <span>SAM3 CUTOUT STUDIO</span>
              <span>PUBLIC SOURCE · GRADIO · CUDA / MPS / ZEROGPU</span>
              <span>FRAME ACCURACY OVER SPEED</span>
            </footer>
            """,
            container=False,
        )

        run_button.click(
            fn=process_fn,
            inputs=[
                source_video,
                subject_prompt,
                detection_threshold,
                max_objects,
                detect_interval,
                erode_kernel,
                dilate_kernel,
                black_point,
                white_point,
                max_megapixels,
            ],
            outputs=[preview, transparent_master, alpha_matte, job_status],
            api_name="remove_background",
            api_description="Track a text-described subject and export a refined alpha matte.",
            scroll_to_output=True,
            show_progress="full",
            show_progress_on=[preview, job_status],
            concurrency_limit=1,
            concurrency_id="sam3-cutout-inference",
        )

    return demo.queue(max_size=8, default_concurrency_limit=1)
