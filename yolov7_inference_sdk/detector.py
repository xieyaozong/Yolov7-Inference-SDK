from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import sys


@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    x0: int
    y0: int
    x1: int
    y1: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class YoloV7Detector:
    def __init__(
        self,
        yolov7_repo: Path,
        weights: Path,
        image_size: int = 640,
        confidence: float = 0.25,
        iou: float = 0.45,
        device: str = "",
    ) -> None:
        self.yolov7_repo = Path(yolov7_repo)
        self.weights = Path(weights)
        self.image_size = image_size
        self.confidence = confidence
        self.iou = iou
        self.device_name = device
        self._backend: Optional[dict[str, Any]] = None

    def detect_file(self, image_path: Path) -> list[Detection]:
        import cv2

        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        return self.detect_array(image)

    def detect_array(self, image: Any) -> list[Detection]:
        import numpy as np
        import torch

        backend = self._load_backend()
        letterbox = backend["letterbox"]
        non_max_suppression = backend["non_max_suppression"]
        scale_coords = backend["scale_coords"]
        time_synchronized = backend["time_synchronized"]

        model = backend["model"]
        device = backend["device"]
        names = backend["names"]
        stride = backend["stride"]
        image_size = backend["image_size"]

        resized = letterbox(image, image_size, stride=stride)[0]
        tensor_image = resized[:, :, ::-1].transpose(2, 0, 1)
        tensor_image = np.ascontiguousarray(tensor_image)
        tensor_image = torch.from_numpy(tensor_image).to(device).float() / 255.0
        if tensor_image.ndimension() == 3:
            tensor_image = tensor_image.unsqueeze(0)

        time_synchronized()
        with torch.no_grad():
            prediction = model(tensor_image, augment=False)[0]
        filtered = non_max_suppression(prediction, self.confidence, self.iou)

        detections: list[Detection] = []
        for detection in filtered:
            if len(detection):
                detection[:, :4] = scale_coords(tensor_image.shape[2:], detection[:, :4], image.shape).round()
                for *xyxy, conf, cls in reversed(detection):
                    class_id = int(cls)
                    detections.append(
                        Detection(
                            class_id=class_id,
                            class_name=str(names[class_id]),
                            confidence=float(conf),
                            x0=int(xyxy[0]),
                            y0=int(xyxy[1]),
                            x1=int(xyxy[2]),
                            y1=int(xyxy[3]),
                        )
                    )
        return detections

    def _load_backend(self) -> dict[str, Any]:
        if self._backend is not None:
            return self._backend
        if not self.yolov7_repo.exists():
            raise FileNotFoundError(f"YOLOv7 repository not found: {self.yolov7_repo}")
        if not self.weights.exists():
            raise FileNotFoundError(f"YOLOv7 weights not found: {self.weights}")

        sys.path.insert(0, str(self.yolov7_repo))

        from models.experimental import attempt_load
        from utils.datasets import letterbox
        from utils.general import check_img_size, non_max_suppression, scale_coords, set_logging
        from utils.torch_utils import select_device, time_synchronized

        import torch

        set_logging()
        device = select_device(self.device_name)
        model = attempt_load(str(self.weights), map_location=device)
        stride = int(model.stride.max())
        image_size = check_img_size(self.image_size, s=stride)
        names = model.module.names if hasattr(model, "module") else model.names
        if device.type != "cpu":
            model(torch.zeros(1, 3, image_size, image_size).to(device).type_as(next(model.parameters())))

        self._backend = {
            "check_img_size": check_img_size,
            "device": device,
            "image_size": image_size,
            "letterbox": letterbox,
            "model": model,
            "names": names,
            "non_max_suppression": non_max_suppression,
            "scale_coords": scale_coords,
            "stride": stride,
            "time_synchronized": time_synchronized,
        }
        return self._backend
