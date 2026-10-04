"""Numerically defensive postprocessing primitives."""

from __future__ import annotations

import numpy as np


def iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    left_top = np.maximum(box[:2], boxes[:, :2])
    right_bottom = np.minimum(box[2:], boxes[:, 2:])
    inter = np.prod(np.maximum(0, right_bottom - left_top), axis=1)
    union = (
        np.prod(np.maximum(0, box[2:] - box[:2]))
        + np.prod(np.maximum(0, boxes[:, 2:] - boxes[:, :2]), axis=1)
        - inter
    )
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def nms(boxes: np.ndarray, scores: np.ndarray, threshold: float, max_det: int) -> np.ndarray:
    valid = np.isfinite(scores)
    order = np.argsort(scores[valid])[::-1]
    indices = np.flatnonzero(valid)[order]
    keep: list[int] = []
    while indices.size and len(keep) < max_det:
        current = indices[0]
        keep.append(int(current))
        remaining = indices[1:]
        indices = remaining[iou(boxes[current], boxes[remaining]) <= threshold]
    return np.asarray(keep, dtype=np.int64)  # type: ignore[no-any-return]
