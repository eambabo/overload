"""Garbage can detector using fine-tuned YOLOv8 or zero-shot YOLO-World."""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

import cv2
from ultralytics import YOLO


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent.parent.parent


class Detection(TypedDict):
    bbox: list[int]  # [x1, y1, x2, y2]
    confidence: float
    class_name: str


class GarbageCanDetector:
    """Detects garbage cans in images and videos.

    Uses fine-tuned YOLOv8 model when available, falls back to YOLO-World zero-shot.
    """

    DEFAULT_CLASSES = ["garbage can", "trash can", "waste bin", "dumpster"]
    FINETUNED_MODEL_PATH = "models/garbage_detector.pt"

    def __init__(
        self,
        model_name: str | None = None,
        confidence: float = 0.3,
        use_finetuned: bool = True,
    ):
        """Initialize the detector.

        Args:
            model_name: Override model path. If None, uses fine-tuned or YOLO-World.
            confidence: Minimum confidence threshold for detections.
            use_finetuned: If True, prefer fine-tuned model when available.
        """
        self.confidence = confidence
        self.is_finetuned = False

        if model_name:
            self.model = YOLO(model_name)
            self.is_finetuned = "garbage_detector" in model_name
        else:
            finetuned_path = get_project_root() / self.FINETUNED_MODEL_PATH
            if use_finetuned and finetuned_path.exists():
                print(f"Using fine-tuned model: {finetuned_path}")
                self.model = YOLO(str(finetuned_path))
                self.is_finetuned = True
            else:
                print("Using YOLO-World zero-shot model")
                self.model = YOLO("yolov8s-world.pt")
                self.model.set_classes(self.DEFAULT_CLASSES)

    def detect_image(self, image_path: str | Path) -> list[Detection]:
        """Detect garbage cans in an image.

        Args:
            image_path: Path to the image file.

        Returns:
            List of detections with bounding boxes and confidence scores.
        """
        results = self.model(str(image_path), conf=self.confidence, verbose=False)
        return self._parse_results(results[0])

    def detect_video(
        self, video_path: str | Path, output_path: str | Path | None = None
    ) -> list[list[Detection]]:
        """Detect garbage cans in a video.

        Args:
            video_path: Path to the video file.
            output_path: Optional path to save annotated video.

        Returns:
            List of detections per frame.
        """
        video_path = Path(video_path)
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        writer = None
        if output_path:
            output_path = Path(output_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        all_detections = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = self.model(frame, conf=self.confidence, verbose=False)
            detections = self._parse_results(results[0])
            all_detections.append(detections)

            if writer:
                annotated = results[0].plot()
                writer.write(annotated)

        cap.release()
        if writer:
            writer.release()

        return all_detections

    def _parse_results(self, result) -> list[Detection]:
        """Parse YOLO results into Detection format."""
        detections = []
        boxes = result.boxes

        if boxes is None:
            return detections

        for box in boxes:
            bbox = box.xyxy[0].tolist()
            cls_idx = int(box.cls[0])

            if self.is_finetuned:
                # Fine-tuned model has single class
                class_name = "garbage_can"
            else:
                # YOLO-World uses our custom class list
                class_name = self.DEFAULT_CLASSES[cls_idx]

            detections.append(
                Detection(
                    bbox=[int(coord) for coord in bbox],
                    confidence=float(box.conf[0]),
                    class_name=class_name,
                )
            )

        return detections
