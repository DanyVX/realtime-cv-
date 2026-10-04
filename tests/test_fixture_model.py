import subprocess
from pathlib import Path

import numpy as np

from rcs.runtime.onnx import OnnxModel


def test_generated_model_is_batch_dynamic() -> None:
    subprocess.run([".venv/Scripts/python.exe", "scripts/generate_fixture_model.py"], check=True)
    model = OnnxModel(Path("models/fixture-detector.onnx"))
    output = model.infer([np.zeros((3, 64, 64), np.float32), np.zeros((3, 64, 64), np.float32)])
    assert len(output) == 2 and output[0].shape == (1, 6)
