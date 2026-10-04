"""Generate a tiny deterministic ONNX detector-shaped model for CI and demos.

This is deliberately not a trained detector. It returns one fixed normalized box per image so
the full serving pipeline can be exercised without distributing a checkpoint or personal data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def main() -> None:
    output = Path("models/fixture-detector.onnx")
    output.parent.mkdir(exist_ok=True)
    images = helper.make_tensor_value_info("images", TensorProto.FLOAT, ["batch", 3, 64, 64])
    predictions = helper.make_tensor_value_info("predictions", TensorProto.FLOAT, ["batch", 1, 6])
    value = numpy_helper.from_array(np.array([[[8, 8, 48, 48, 0.9, 0]]], np.float32), "template")
    shape = numpy_helper.from_array(np.array([0, 1, 1], np.int64), "shape")
    graph = helper.make_graph(
        [
            helper.make_node("Shape", ["images"], ["input_shape"]),
            helper.make_node("Gather", ["input_shape", "shape"], ["batch_shape"], axis=0),
            helper.make_node("Expand", ["template", "batch_shape"], ["predictions"]),
        ],
        "fixture",
        [images],
        [predictions],
        [value, shape],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    onnx.save(model, output)
    print(output)


if __name__ == "__main__":
    main()
