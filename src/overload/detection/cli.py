"""Command-line interface for garbage can detection."""

import argparse
import sys
from pathlib import Path

from .detector import GarbageCanDetector

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def main():
    parser = argparse.ArgumentParser(description="Detect garbage cans in images or videos")
    parser.add_argument("input", help="Path to image or video file")
    parser.add_argument("-o", "--output", help="Path to save annotated output (video only)")
    parser.add_argument(
        "-c", "--confidence", type=float, default=0.3, help="Confidence threshold (default: 0.3)"
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: File not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    detector = GarbageCanDetector(confidence=args.confidence)
    suffix = input_path.suffix.lower()

    if suffix in IMAGE_EXTENSIONS:
        detections = detector.detect_image(input_path)
        print(f"Found {len(detections)} garbage can(s) in {input_path.name}:")
        for i, det in enumerate(detections, 1):
            print(f"  {i}. {det['class_name']} (confidence: {det['confidence']:.2f})")
            print(f"     bbox: {det['bbox']}")

    elif suffix in VIDEO_EXTENSIONS:
        print(f"Processing video: {input_path.name}")
        all_detections = detector.detect_video(input_path, args.output)
        total = sum(len(frame_dets) for frame_dets in all_detections)
        print(f"Processed {len(all_detections)} frames, found {total} total detections")
        if args.output:
            print(f"Annotated video saved to: {args.output}")

    else:
        print(f"Error: Unsupported file type: {suffix}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
