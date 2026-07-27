import pytest


def test_auto_device_prefers_cuda_over_mps() -> None:
    try:
        from sam3_matting.devices import DeviceAvailability, choose_device
    except ModuleNotFoundError:
        pytest.fail("device policy has not been implemented")

    availability = DeviceAvailability(cuda=True, mps=True)

    selected = choose_device("auto", availability)

    assert selected == "cuda"


def test_auto_device_uses_mps_when_cuda_is_unavailable() -> None:
    try:
        from sam3_matting.devices import DeviceAvailability, choose_device
    except ModuleNotFoundError:
        pytest.fail("device policy has not been implemented")

    availability = DeviceAvailability(cuda=False, mps=True)

    selected = choose_device("auto", availability)

    assert selected == "mps"


def test_auto_device_falls_back_to_cpu() -> None:
    try:
        from sam3_matting.devices import DeviceAvailability, choose_device
    except ModuleNotFoundError:
        pytest.fail("device policy has not been implemented")

    availability = DeviceAvailability(cuda=False, mps=False)

    selected = choose_device("auto", availability)

    assert selected == "cpu"


def test_explicit_mps_is_selected_when_available() -> None:
    try:
        from sam3_matting.devices import DeviceAvailability, choose_device
    except ModuleNotFoundError:
        pytest.fail("device policy has not been implemented")

    availability = DeviceAvailability(cuda=True, mps=True)

    selected = choose_device("mps", availability)

    assert selected == "mps"


def test_explicit_unavailable_device_is_rejected() -> None:
    try:
        from sam3_matting.devices import DeviceAvailability, choose_device
    except ModuleNotFoundError:
        pytest.fail("device policy has not been implemented")

    availability = DeviceAvailability(cuda=True, mps=False)

    with pytest.raises(RuntimeError, match="mps is not available"):
        choose_device("mps", availability)


def test_explicit_cpu_is_always_available() -> None:
    try:
        from sam3_matting.devices import DeviceAvailability, choose_device
    except ModuleNotFoundError:
        pytest.fail("device policy has not been implemented")

    availability = DeviceAvailability(cuda=True, mps=True)

    selected = choose_device("cpu", availability)

    assert selected == "cpu"


def test_explicit_cuda_is_selected_when_available() -> None:
    try:
        from sam3_matting.devices import DeviceAvailability, choose_device
    except ModuleNotFoundError:
        pytest.fail("device policy has not been implemented")

    availability = DeviceAvailability(cuda=True, mps=False)

    selected = choose_device("cuda", availability)

    assert selected == "cuda"


def test_explicit_unavailable_cuda_is_rejected() -> None:
    try:
        from sam3_matting.devices import DeviceAvailability, choose_device
    except ModuleNotFoundError:
        pytest.fail("device policy has not been implemented")

    availability = DeviceAvailability(cuda=False, mps=True)

    with pytest.raises(RuntimeError, match="cuda is not available"):
        choose_device("cuda", availability)


def test_unknown_device_preference_is_rejected() -> None:
    try:
        from sam3_matting.devices import DeviceAvailability, choose_device
    except ModuleNotFoundError:
        pytest.fail("device policy has not been implemented")

    availability = DeviceAvailability(cuda=True, mps=True)

    with pytest.raises(ValueError, match="unknown device preference"):
        choose_device("tpu", availability)


def test_runtime_detection_matches_pytorch() -> None:
    import torch

    try:
        from sam3_matting.devices import detect_availability
    except ImportError:
        pytest.fail("runtime device detection has not been implemented")

    detected = detect_availability()

    expected_mps = bool(
        hasattr(torch.backends, "mps") and torch.backends.mps.is_built() and torch.backends.mps.is_available()
    )
    assert detected.cuda is torch.cuda.is_available()
    assert detected.mps is expected_mps
