"""Safe image decoding and letterbox preprocessing."""

from __future__ import annotations

import cv2
import numpy as np

MAGIC = (b"\xff\xd8\xff", b"\x89PNG\r\n\x1a\n", b"GIF87a", b"GIF89a", b"RIFF")


def decode_image(payload: bytes, max_bytes: int, max_pixels: int) -> np.ndarray:
    if not payload or len(payload) > max_bytes or not payload.startswith(MAGIC):
        raise ValueError("invalid or oversized image payload")
    image = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None or image.size == 0 or image.shape[0] * image.shape[1] > max_pixels:
        raise ValueError("unreadable image or decompression bomb")
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    if image.dtype != np.uint8:
        image = cv2.convertScaleAbs(image, alpha=255.0 / max(1, image.max()))
    return image


def letterbox(
    image: np.ndarray, size: tuple[int, int]
) -> tuple[np.ndarray, float, tuple[int, int]]:
    height, width = image.shape[:2]
    target_h, target_w = size
    scale = min(target_w / width, target_h / height)
    resized = cv2.resize(image, (round(width * scale), round(height * scale)))
    pad_x, pad_y = (target_w - resized.shape[1]) // 2, (target_h - resized.shape[0]) // 2
    canvas = np.full((target_h, target_w, 3), 114, np.uint8)
    canvas[pad_y : pad_y + resized.shape[0], pad_x : pad_x + resized.shape[1]] = resized
    return canvas[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255, scale, (pad_x, pad_y)
