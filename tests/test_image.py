import numpy as np

from rcs.preprocess.image import decode_image, letterbox


def test_letterbox_preserves_shape_and_scale() -> None:
    tensor, scale, padding = letterbox(np.zeros((8, 100, 3), np.uint8), (64, 64))
    assert tensor.shape == (3, 64, 64) and scale == 0.64 and padding == (0, 29)


def test_decode_rejects_non_image() -> None:
    try:
        decode_image(b"bad", 10, 10)
    except ValueError:
        return
    raise AssertionError("invalid payload accepted")
