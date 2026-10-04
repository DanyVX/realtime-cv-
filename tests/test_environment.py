from rcs.bench.environment import capture_environment


def test_environment_capture_has_reproducibility_fields() -> None:
    environment = capture_environment()

    assert environment.python
    assert environment.platform
    assert "onnxruntime" in environment.packages
