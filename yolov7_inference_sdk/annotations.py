from __future__ import annotations

from typing import Iterable
from yolov7_inference_sdk.detector import Detection

import cv2
import numpy as np


def draw_boxes(image: np.ndarray, detections: Iterable[Detection]) -> np.ndarray:
    output = image.copy()
    for item in detections:
        color = _color_for_class(item.class_id)
        start = (int(item.x0), int(item.y0))
        end = (int(item.x1), int(item.y1))
        label = f"{item.class_name} {item.confidence:.2f}"
        cv2.rectangle(output, start, end, color, 2)
        cv2.putText(output, label, (start[0], max(16, start[1] - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return output


def _color_for_class(class_id: int) -> tuple[int, int, int]:
    palette = [
        (56, 142, 60),
        (25, 118, 210),
        (230, 81, 0),
        (123, 31, 162),
        (0, 121, 107),
        (198, 40, 40),
    ]
    return palette[class_id % len(palette)]
