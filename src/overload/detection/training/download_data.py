"""Download and prepare training data for garbage can detection."""

from __future__ import annotations

import json
import random
import shutil
import subprocess
import sys
from pathlib import Path

import fiftyone as fo
import fiftyone.zoo as foz


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent.parent.parent.parent


def download_taco(data_dir: Path) -> Path:
    """Download TACO dataset from GitHub.

    Args:
        data_dir: Directory to download data to.

    Returns:
        Path to the downloaded TACO directory.
    """
    taco_dir = data_dir / "taco"

    if taco_dir.exists():
        print("TACO dataset already downloaded")
        return taco_dir

    print("Downloading TACO dataset...")
    taco_dir.mkdir(parents=True, exist_ok=True)

    # Clone TACO repository
    subprocess.run(
        ["git", "clone", "--depth", "1", "https://github.com/pedropro/TACO.git", str(taco_dir)],
        check=True,
    )

    # Download the actual images using TACO's download script
    print("Downloading TACO images...")
    subprocess.run(
        [sys.executable, str(taco_dir / "download.py")],
        cwd=taco_dir,
        check=True,
    )

    return taco_dir


def download_open_images(data_dir: Path, max_samples: int = 3000) -> fo.Dataset:
    """Download Open Images waste container subset using FiftyOne.

    Args:
        data_dir: Directory to store data.
        max_samples: Maximum number of samples to download.

    Returns:
        FiftyOne dataset with waste container images.
    """
    print(f"Downloading Open Images waste container images (max {max_samples})...")

    dataset = foz.load_zoo_dataset(
        "open-images-v7",
        split="train",
        label_types=["detections"],
        classes=["Waste container"],
        max_samples=max_samples,
        dataset_dir=str(data_dir / "open_images"),
    )

    return dataset


def convert_taco_to_yolo(taco_dir: Path, output_dir: Path) -> list[tuple[Path, Path]]:
    """Convert TACO COCO format annotations to YOLO format.

    Args:
        taco_dir: Path to TACO dataset.
        output_dir: Output directory for YOLO format data.

    Returns:
        List of (image_path, label_path) tuples.
    """
    print("Converting TACO annotations to YOLO format...")

    annotations_file = taco_dir / "data" / "annotations.json"
    if not annotations_file.exists():
        raise FileNotFoundError(f"TACO annotations not found at {annotations_file}")

    with open(annotations_file) as f:
        coco_data = json.load(f)

    # Build category mapping - we'll map all trash-related categories to garbage_can (0)
    # TACO has various litter categories, but we want bins/containers
    bin_categories = set()
    for cat in coco_data["categories"]:
        name_lower = cat["name"].lower()
        if any(term in name_lower for term in ["bin", "container", "bag", "bucket"]):
            bin_categories.add(cat["id"])

    # Build image ID to info mapping
    image_info = {img["id"]: img for img in coco_data["images"]}

    # Group annotations by image
    image_annotations: dict[int, list] = {}
    for ann in coco_data["annotations"]:
        if ann["category_id"] in bin_categories:
            img_id = ann["image_id"]
            if img_id not in image_annotations:
                image_annotations[img_id] = []
            image_annotations[img_id].append(ann)

    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"
    images_dir.mkdir(exist_ok=True)
    labels_dir.mkdir(exist_ok=True)

    pairs = []
    for img_id, annotations in image_annotations.items():
        img_info = image_info[img_id]
        src_image = taco_dir / "data" / img_info["file_name"]

        if not src_image.exists():
            continue

        # Copy image
        dst_image = images_dir / f"taco_{img_id}.jpg"
        shutil.copy(src_image, dst_image)

        # Convert annotations to YOLO format
        img_width = img_info["width"]
        img_height = img_info["height"]

        label_file = labels_dir / f"taco_{img_id}.txt"
        with open(label_file, "w") as f:
            for ann in annotations:
                x, y, w, h = ann["bbox"]
                # Convert to YOLO format (center_x, center_y, width, height) normalized
                center_x = (x + w / 2) / img_width
                center_y = (y + h / 2) / img_height
                norm_w = w / img_width
                norm_h = h / img_height
                f.write(f"0 {center_x:.6f} {center_y:.6f} {norm_w:.6f} {norm_h:.6f}\n")

        pairs.append((dst_image, label_file))

    print(f"Converted {len(pairs)} TACO images with bin annotations")
    return pairs


def convert_open_images_to_yolo(
    dataset: fo.Dataset, output_dir: Path
) -> list[tuple[Path, Path]]:
    """Convert Open Images FiftyOne dataset to YOLO format.

    Args:
        dataset: FiftyOne dataset with Open Images data.
        output_dir: Output directory for YOLO format data.

    Returns:
        List of (image_path, label_path) tuples.
    """
    print("Converting Open Images to YOLO format...")

    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"
    images_dir.mkdir(exist_ok=True)
    labels_dir.mkdir(exist_ok=True)

    pairs = []
    for idx, sample in enumerate(dataset):
        if sample.detections is None:
            continue

        # Filter to only waste container detections
        waste_detections = [
            det for det in sample.detections.detections if det.label == "Waste container"
        ]

        if not waste_detections:
            continue

        # Copy image
        src_image = Path(sample.filepath)
        dst_image = images_dir / f"openimages_{idx}.jpg"
        shutil.copy(src_image, dst_image)

        # Write YOLO format labels
        label_file = labels_dir / f"openimages_{idx}.txt"
        with open(label_file, "w") as f:
            for det in waste_detections:
                # FiftyOne stores bounding boxes as [x, y, width, height] normalized
                x, y, w, h = det.bounding_box
                center_x = x + w / 2
                center_y = y + h / 2
                f.write(f"0 {center_x:.6f} {center_y:.6f} {w:.6f} {h:.6f}\n")

        pairs.append((dst_image, label_file))

    print(f"Converted {len(pairs)} Open Images samples")
    return pairs


def split_dataset(
    all_pairs: list[tuple[Path, Path]], output_dir: Path, train_ratio: float = 0.8
) -> None:
    """Split dataset into train and validation sets.

    Args:
        all_pairs: List of (image_path, label_path) tuples.
        output_dir: Output directory for final dataset.
        train_ratio: Ratio of training samples (default 0.8).
    """
    print(f"Splitting {len(all_pairs)} samples into train/val...")

    random.shuffle(all_pairs)
    split_idx = int(len(all_pairs) * train_ratio)
    train_pairs = all_pairs[:split_idx]
    val_pairs = all_pairs[split_idx:]

    for split_name, pairs in [("train", train_pairs), ("val", val_pairs)]:
        split_images = output_dir / split_name / "images"
        split_labels = output_dir / split_name / "labels"
        split_images.mkdir(parents=True, exist_ok=True)
        split_labels.mkdir(parents=True, exist_ok=True)

        for img_path, label_path in pairs:
            shutil.move(str(img_path), str(split_images / img_path.name))
            shutil.move(str(label_path), str(split_labels / label_path.name))

    print(f"Train: {len(train_pairs)} samples, Val: {len(val_pairs)} samples")


def main() -> None:
    """Download and prepare all training data."""
    project_root = get_project_root()
    data_dir = project_root / "data"
    raw_dir = data_dir / "raw"
    final_dir = data_dir / "garbage_cans"

    # Clean up any existing processed data
    if final_dir.exists():
        print(f"Removing existing data at {final_dir}")
        shutil.rmtree(final_dir)

    # Download datasets
    taco_dir = download_taco(raw_dir)
    oi_dataset = download_open_images(raw_dir)

    # Convert to YOLO format
    temp_dir = data_dir / "temp_yolo"
    taco_pairs = convert_taco_to_yolo(taco_dir, temp_dir / "taco")
    oi_pairs = convert_open_images_to_yolo(oi_dataset, temp_dir / "openimages")

    # Merge and split
    all_pairs = taco_pairs + oi_pairs
    print(f"Total samples: {len(all_pairs)}")

    split_dataset(all_pairs, final_dir)

    # Clean up temp directory
    if temp_dir.exists():
        shutil.rmtree(temp_dir)

    print(f"\nDataset prepared at {final_dir}")
    print("You can now run training with: python -m overload.detection.training.train")


if __name__ == "__main__":
    main()
