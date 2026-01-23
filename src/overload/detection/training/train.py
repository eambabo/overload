"""Train YOLOv8 model for garbage can detection."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent.parent.parent.parent


def train(
    epochs: int = 50,
    batch_size: int = 16,
    image_size: int = 640,
    base_model: str = "yolov8n.pt",
    patience: int = 10,
) -> Path:
    """Train YOLOv8 model on garbage can dataset.

    Args:
        epochs: Number of training epochs.
        batch_size: Training batch size.
        image_size: Input image size.
        base_model: Base YOLO model to fine-tune.
        patience: Early stopping patience.

    Returns:
        Path to the best trained model.
    """
    project_root = get_project_root()
    data_yaml = project_root / "data" / "garbage_cans.yaml"
    models_dir = project_root / "models"
    models_dir.mkdir(exist_ok=True)

    if not data_yaml.exists():
        raise FileNotFoundError(
            f"Dataset config not found at {data_yaml}. "
            "Run download_data.py first to prepare the dataset."
        )

    print(f"Loading base model: {base_model}")
    model = YOLO(base_model)

    print(f"Training on data: {data_yaml}")
    print(f"Epochs: {epochs}, Batch size: {batch_size}, Image size: {image_size}")

    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        batch=batch_size,
        imgsz=image_size,
        patience=patience,
        save=True,
        project=str(models_dir),
        name="garbage_detector",
        exist_ok=True,
        verbose=True,
    )

    # Copy best model to models directory with standard name
    best_model_src = Path(results.save_dir) / "weights" / "best.pt"
    best_model_dst = models_dir / "garbage_detector.pt"

    if best_model_src.exists():
        import shutil

        shutil.copy(best_model_src, best_model_dst)
        print(f"\nBest model saved to: {best_model_dst}")
    else:
        print(f"\nWarning: Best model not found at {best_model_src}")

    return best_model_dst


def main() -> None:
    """Main training entry point."""
    parser = argparse.ArgumentParser(description="Train garbage can detector")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--image-size", type=int, default=640, help="Image size")
    parser.add_argument(
        "--base-model", type=str, default="yolov8n.pt", help="Base YOLO model"
    )
    parser.add_argument(
        "--patience", type=int, default=10, help="Early stopping patience"
    )
    args = parser.parse_args()

    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        image_size=args.image_size,
        base_model=args.base_model,
        patience=args.patience,
    )


if __name__ == "__main__":
    main()
