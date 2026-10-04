import numpy as np

from rcs.postprocess.nms import nms


def test_nms_removes_overlap_and_nan() -> None:
    kept = nms(
        np.array([[0, 0, 10, 10], [1, 1, 9, 9], [20, 20, 30, 30]], float),
        np.array([0.9, 0.8, np.nan]),
        0.5,
        10,
    )
    assert kept.tolist() == [0]
