"""Small YOLOv7 inference adapter package."""

from yolov7_inference_sdk.annotations import draw_boxes
from yolov7_inference_sdk.detector import Detection, YoloV7Detector

__all__ = ["Detection", "YoloV7Detector", "draw_boxes"]

