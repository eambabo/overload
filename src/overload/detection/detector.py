"""Garbage can detector using fine-tuned YOLOv8 or zero-shot YOLO-World."""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

import cv2
import numpy as np
from ultralytics import YOLO


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent.parent.parent


def compute_iou(box1: list[int], box2: list[int]) -> float:
    """Compute Intersection over Union between two bounding boxes.

    Args:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]

    Returns:
        IoU score between 0 and 1.
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0


class SimpleTracker:
    """Simple IoU-based object tracker."""

    def __init__(self, iou_threshold: float = 0.3, max_age: int = 30):
        """Initialize tracker.

        Args:
            iou_threshold: Minimum IoU to consider a match.
            max_age: Frames to keep a track alive without matches.
        """
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.tracks: dict[int, dict] = {}  # track_id -> {bbox, age, class_name}
        self.next_id = 1

    def update(self, detections: list[dict]) -> list[tuple[dict, int]]:
        """Update tracks with new detections.

        Args:
            detections: List of detections with 'bbox' and 'class_name'.

        Returns:
            List of (detection, track_id) tuples.
        """
        results = []

        # Match detections to existing tracks using IoU
        unmatched_detections = list(range(len(detections)))
        matched_tracks = set()

        for det_idx, det in enumerate(detections):
            best_iou = 0
            best_track_id = None

            for track_id, track in self.tracks.items():
                if track_id in matched_tracks:
                    continue
                iou = compute_iou(det["bbox"], track["bbox"])
                if iou > best_iou and iou >= self.iou_threshold:
                    best_iou = iou
                    best_track_id = track_id

            if best_track_id is not None:
                # Update existing track
                self.tracks[best_track_id]["bbox"] = det["bbox"]
                self.tracks[best_track_id]["age"] = 0
                matched_tracks.add(best_track_id)
                unmatched_detections.remove(det_idx)
                results.append((det, best_track_id))

        # Create new tracks for unmatched detections
        for det_idx in unmatched_detections:
            det = detections[det_idx]
            track_id = self.next_id
            self.next_id += 1
            self.tracks[track_id] = {
                "bbox": det["bbox"],
                "age": 0,
                "class_name": det["class_name"],
            }
            results.append((det, track_id))

        # Age out old tracks
        tracks_to_remove = []
        for track_id, track in self.tracks.items():
            if track_id not in matched_tracks:
                track["age"] += 1
                if track["age"] > self.max_age:
                    tracks_to_remove.append(track_id)

        for track_id in tracks_to_remove:
            del self.tracks[track_id]

        return results


class Detection(TypedDict):
    bbox: list[int]  # [x1, y1, x2, y2]
    confidence: float
    class_name: str


class TrackedDetection(TypedDict):
    bbox: list[int]  # [x1, y1, x2, y2]
    confidence: float
    class_name: str
    track_id: int  # Unique ID for this object across frames


class TrackingResult(TypedDict):
    frames: list[list[TrackedDetection]]  # Detections per frame
    unique_ids: set[int]  # All unique track IDs seen
    total_detections: int  # Total detection count across all frames


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

    def detect_video_with_tracking(
        self, video_path: str | Path, output_path: str | Path | None = None
    ) -> TrackingResult:
        """Detect and track garbage cans in a video.

        Uses IoU-based tracking to assign persistent IDs to objects across frames.

        Args:
            video_path: Path to the video file.
            output_path: Optional path to save annotated video.

        Returns:
            TrackingResult with per-frame detections, unique IDs, and total count.
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

        tracker = SimpleTracker(iou_threshold=0.3, max_age=30)
        all_detections: list[list[TrackedDetection]] = []
        unique_ids: set[int] = set()
        total_detections = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Run detection
            results = self.model(frame, conf=self.confidence, verbose=False)
            detections = self._parse_results(results[0])

            # Update tracker and get track IDs
            tracked = tracker.update(detections)
            frame_detections = []
            for det, track_id in tracked:
                frame_detections.append(
                    TrackedDetection(
                        bbox=det["bbox"],
                        confidence=det["confidence"],
                        class_name=det["class_name"],
                        track_id=track_id,
                    )
                )
                unique_ids.add(track_id)

            all_detections.append(frame_detections)
            total_detections += len(frame_detections)

            if writer:
                # Draw bounding boxes with track IDs
                annotated = frame.copy()
                for det in frame_detections:
                    x1, y1, x2, y2 = det["bbox"]
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"ID:{det['track_id']} {det['confidence']:.2f}"
                    cv2.putText(
                        annotated, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
                    )
                writer.write(annotated)

        cap.release()
        if writer:
            writer.release()

        return TrackingResult(
            frames=all_detections,
            unique_ids=unique_ids,
            total_detections=total_detections,
        )

    def _parse_track_results(self, result) -> list[TrackedDetection]:
        """Parse YOLO tracking results into TrackedDetection format."""
        detections = []
        boxes = result.boxes

        if boxes is None:
            return detections

        for box in boxes:
            bbox = box.xyxy[0].tolist()
            cls_idx = int(box.cls[0])

            # Get track ID (-1 if not assigned)
            track_id = int(box.id[0]) if box.id is not None else -1

            if self.is_finetuned:
                class_name = "garbage_can"
            else:
                class_name = self.DEFAULT_CLASSES[cls_idx]

            detections.append(
                TrackedDetection(
                    bbox=[int(coord) for coord in bbox],
                    confidence=float(box.conf[0]),
                    class_name=class_name,
                    track_id=track_id,
                )
            )

        return detections

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
