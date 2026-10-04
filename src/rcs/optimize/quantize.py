"""Static INT8 quantization with an explicit calibration-data requirement."""

from __future__ import annotations

from pathlib import Path

import onnxruntime.quantization as ortq


def quantize_int8(
    model: Path, output: Path, calibration_reader: ortq.CalibrationDataReader
) -> None:
    ortq.quantize_static(
        str(model),
        str(output),
        calibration_reader,
        per_channel=True,
        weight_type=ortq.QuantType.QInt8,
        activation_type=ortq.QuantType.QUInt8,
    )
