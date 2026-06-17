from __future__ import annotations

from pathlib import Path

import argparse
import json
import os
import sys

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from yolov7_inference_sdk import YoloV7Detector, draw_boxes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLOv7 inference through the SDK adapter.")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--weights", type=Path, default=PROJECT_ROOT / "models" / "yolov7.pt")
    parser.add_argument("--yolov7-repo", type=Path, default=os.environ.get("YOLOV7_REPO"))
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs")
    parser.add_argument("--confidence", type=float, default=0.25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.yolov7_repo is None:
        raise ValueError("Set --yolov7-repo or YOLOV7_REPO to an external YOLOv7 checkout.")

    image_path = args.image if args.image.is_absolute() else PROJECT_ROOT / args.image
    weights_path = args.weights if args.weights.is_absolute() else PROJECT_ROOT / args.weights
    repo_path = args.yolov7_repo if args.yolov7_repo.is_absolute() else PROJECT_ROOT / args.yolov7_repo

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    detector = YoloV7Detector(repo_path, weights_path, confidence=args.confidence)
    detections = detector.detect_array(image)
    annotated = draw_boxes(image, detections)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output_dir / "detected.jpg"), annotated)
    records = [item.to_dict() for item in detections]
    (args.output_dir / "detections.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"Saved {len(records)} detections to {args.output_dir}")


if __name__ == "__main__":
    main()
