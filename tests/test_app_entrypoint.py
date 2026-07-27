from __future__ import annotations

import importlib
import os
import subprocess
import sys
import textwrap
import tomllib
from pathlib import Path


def test_importing_entrypoint_has_no_model_or_ui_side_effects() -> None:
    app = importlib.import_module("app")

    assert app.RESOURCES is None
    assert app.DEMO is None


def test_bootstrap_preloads_then_decorates_exact_callback_and_builds_runtime_chrome() -> None:
    app = importlib.import_module("app")
    events: list[tuple[str, object]] = []
    resources = object()
    raw_callback = object()
    validator = object()
    decorated_callback = object()
    demo = object()

    def resolve_checkpoint() -> Path:
        events.append(("checkpoint", None))
        return Path("/models/sam.safetensors")

    def device() -> str:
        events.append(("device", None))
        return "cuda"

    def is_zerogpu() -> bool:
        events.append(("zerogpu", None))
        return True

    def make_resources(checkpoint: Path, **kwargs: object) -> object:
        events.append(("resources", (checkpoint, kwargs)))
        return resources

    def make_callback(active_resources: object, **kwargs: object) -> object:
        events.append(("callback", (active_resources, kwargs)))
        return raw_callback

    def make_validator(**kwargs: object) -> object:
        events.append(("validator", kwargs))
        return validator

    def gpu_factory(*, duration: int, size: str):
        events.append(("gpu", (duration, size)))

        def decorate(callback: object) -> object:
            events.append(("decorate", callback))
            return decorated_callback

        return decorate

    def make_ui(
        callback: object,
        *,
        validator_fn: object,
        hosted: bool,
        runtime_status: dict[str, str],
    ) -> object:
        events.append(("ui", (callback, validator_fn, hosted, runtime_status)))
        return demo

    built_resources, built_demo = app.bootstrap_application(
        resolve_checkpoint_fn=resolve_checkpoint,
        target_device_fn=device,
        zerogpu_detector=is_zerogpu,
        build_resources_fn=make_resources,
        create_callback_fn=make_callback,
        create_validator_fn=make_validator,
        gpu_factory=gpu_factory,
        build_ui_fn=make_ui,
    )

    assert built_resources is resources
    assert built_demo is demo
    assert events == [
        ("checkpoint", None),
        ("device", None),
        ("zerogpu", None),
        (
            "resources",
            (
                Path("/models/sam.safetensors"),
                {"device": "cuda", "preload": True},
            ),
        ),
        ("callback", (resources, {"zerogpu": True})),
        ("validator", {"zerogpu": True}),
        ("gpu", (90, "xlarge")),
        ("decorate", raw_callback),
        (
            "ui",
            (
                decorated_callback,
                validator,
                True,
                {
                    "device": "CUDA",
                    "cuda": "Active",
                    "mps": "Next phase",
                    "zerogpu": "96 GB xlarge",
                },
            ),
        ),
    ]


def test_launch_uses_environment_host_port_and_studio_presentation() -> None:
    app = importlib.import_module("app")

    class Demo:
        def __init__(self) -> None:
            self.kwargs: dict[str, object] | None = None

        def launch(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    demo = Demo()
    app.launch_application(
        demo,
        environ={
            "GRADIO_SERVER_NAME": "127.0.0.7",
            "GRADIO_SERVER_PORT": "9876",
            "SPACES_ZERO_GPU": "1",
        },
        launch_kwargs_fn=lambda: {
            "theme": "studio-theme",
            "css_paths": Path("/assets/studio.css"),
            "head": "<meta>",
            "footer_links": [],
        },
    )

    assert demo.kwargs == {
        "server_name": "127.0.0.7",
        "server_port": 9876,
        "max_file_size": 100 * 1024 * 1024,
        "show_error": False,
        "theme": "studio-theme",
        "css_paths": Path("/assets/studio.css"),
        "head": "<meta>",
        "footer_links": [],
    }


def test_local_launch_allows_two_gibibyte_uploads_without_exposing_tracebacks() -> None:
    app = importlib.import_module("app")

    class Demo:
        def __init__(self) -> None:
            self.kwargs: dict[str, object] | None = None

        def launch(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    demo = Demo()
    app.launch_application(
        demo,
        environ={},
        launch_kwargs_fn=lambda: {},
    )

    assert demo.kwargs == {
        "server_name": "0.0.0.0",
        "server_port": 7860,
        "max_file_size": 2 * 1024 * 1024 * 1024,
        "show_error": False,
    }


def test_space_requirements_install_the_src_layout_package() -> None:
    requirements = Path(__file__).resolve().parents[1] / "requirements.txt"
    entries = {
        line.split("#", maxsplit=1)[0].strip()
        for line in requirements.read_text().splitlines()
        if line.split("#", maxsplit=1)[0].strip()
    }

    assert entries == {"-e ."}


def test_pyproject_pins_runtime_and_contributor_dependencies() -> None:
    project_root = Path(__file__).resolve().parents[1]
    pyproject = tomllib.loads((project_root / "pyproject.toml").read_text())

    assert set(pyproject["project"]["dependencies"]) == {
        "torch==2.11.0",
        "torchvision==0.26.0",
        "numpy==1.26.4",
        "pillow==12.3.0",
        "opencv-python-headless==4.10.0.84",
        "av==18.0.0",
        "transformers==5.14.1",
        "huggingface-hub==1.25.1",
        "safetensors==0.8.0",
        "gradio[oauth]==6.20.0",
        "spaces==0.51.1",
        "psutil==7.2.2",
        "scipy==1.17.0",
        "timm==1.0.24",
        "einops==0.8.1",
        "pycocotools==2.0.10",
        "ftfy==6.1.1",
        "iopath==0.1.10",
        ("sam3 @ git+https://github.com/facebookresearch/sam3.git@46957e47805eaa273f4aa7bbbd25a88bca9108ce"),
    }
    assert set(pyproject["project"]["optional-dependencies"]["dev"]) == {
        "pytest==8.4.2",
        "pytest-cov==7.0.0",
        "ruff==0.14.14",
    }
    assert pyproject["tool"]["hatch"]["metadata"]["allow-direct-references"] is True


def test_zerogpu_entrypoint_imports_spaces_before_project_or_accelerator_startup(
    tmp_path: Path,
) -> None:
    project_root = Path(__file__).resolve().parents[1]
    event_log = tmp_path / "startup-events.log"
    fake_package = tmp_path / "sam3_matting"
    fake_package.mkdir()
    (fake_package / "__init__.py").write_text("")

    (tmp_path / "spaces.py").write_text(
        textwrap.dedent(
            """
            import os
            from pathlib import Path

            with Path(os.environ["STARTUP_EVENT_LOG"]).open("a") as event_log:
                event_log.write("spaces-import\\n")
            """
        )
    )
    (fake_package / "application.py").write_text(
        textwrap.dedent(
            """
            import os
            from pathlib import Path

            def record(event):
                with Path(os.environ["STARTUP_EVENT_LOG"]).open("a") as event_log:
                    event_log.write(f"{event}\\n")

            record("application-import")

            class ApplicationResources:
                pass

            def build_resources(_checkpoint, *, device, preload):
                assert device == "cuda"
                assert preload is True
                record("model-preload")
                return object()

            def create_process_callback(_resources, *, zerogpu):
                assert zerogpu is True
                return lambda *_args: None

            def create_request_validator(*, zerogpu):
                assert zerogpu is True
                return lambda *_args: None
            """
        )
    )
    (fake_package / "runtime.py").write_text(
        textwrap.dedent(
            """
            import os
            from pathlib import Path

            def record(event):
                with Path(os.environ["STARTUP_EVENT_LOG"]).open("a") as event_log:
                    event_log.write(f"{event}\\n")

            record("runtime-import")

            def gpu(*, duration, size):
                assert duration == 90
                assert size == "xlarge"
                return lambda function: function

            def on_zerogpu():
                return True

            def resolve_sam_checkpoint():
                return Path("/models/sam.safetensors")

            def target_device():
                record("torch-cuda-init")
                return "cuda"
            """
        )
    )
    (fake_package / "ui.py").write_text(
        textwrap.dedent(
            """
            import os
            from pathlib import Path

            def record(event):
                with Path(os.environ["STARTUP_EVENT_LOG"]).open("a") as event_log:
                    event_log.write(f"{event}\\n")

            record("ui-import")

            class Demo:
                def launch(self, **_kwargs):
                    record("launch")

            def build_ui(_process, *, validator_fn, hosted, runtime_status):
                assert callable(validator_fn)
                assert hosted is True
                assert runtime_status["zerogpu"] == "96 GB xlarge"
                record("ui-build")
                return Demo()

            def studio_launch_kwargs():
                return {}
            """
        )
    )

    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONPATH": str(tmp_path),
            "SPACES_ZERO_GPU": "1",
            "STARTUP_EVENT_LOG": str(event_log),
        }
    )
    subprocess.run(
        [sys.executable, str(project_root / "app.py")],
        cwd=project_root,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    events = event_log.read_text().splitlines()
    assert events.count("spaces-import") == 1
    assert events[0] == "spaces-import"
    for later_event in (
        "application-import",
        "runtime-import",
        "ui-import",
        "torch-cuda-init",
        "model-preload",
    ):
        assert events.index("spaces-import") < events.index(later_event)
    assert events.count("model-preload") == 1
    assert events.index("model-preload") < events.index("launch")
