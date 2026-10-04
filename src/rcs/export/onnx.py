"""Torch-to-ONNX export entry point; requires the optional export extra."""

from __future__ import annotations

from pathlib import Path


def export_dynamic(model: object, example: object, output: Path) -> None:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("install with: uv sync --extra export") from error
    torch.onnx.export(
        model,
        example,
        output,
        opset_version=17,
        input_names=["images"],
        output_names=["predictions"],
        dynamic_axes={"images": {0: "batch"}, "predictions": {0: "batch"}},
    )
