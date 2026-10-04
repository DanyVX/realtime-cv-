"""ONNX Runtime session ownership and provider selection."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort


class OnnxModel:
    def __init__(self, path: Path) -> None:
        if not path.is_file():
            raise FileNotFoundError(f"model not found: {path}")
        providers = ort.get_available_providers()
        self.session = ort.InferenceSession(
            str(path),
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
            if "CUDAExecutionProvider" in providers
            else ["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name

    def infer(self, images: list[np.ndarray]) -> list[np.ndarray]:
        output = self.session.run(None, {self.input_name: np.stack(images)})[0]
        return [output[index] for index in range(len(images))]
