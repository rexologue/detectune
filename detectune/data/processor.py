"""Dataset processing utilities for MaskDINO experiments."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from pycocotools.coco import COCO


DATASET_STRUCTURE = {
    "train": "train/_annotations.coco.json",
    "valid": "valid/_annotations.coco.json",
    "test": "test/_annotations.coco.json",
}


@dataclass
class DatasetSummary:
    path: Path
    splits: Dict[str, Path]
    categories: Dict[int, str]
    num_images: Dict[str, int]
    num_annotations: Dict[str, int]


def validate_dataset(directory: str | Path) -> DatasetSummary:
    """Validate that the dataset directory matches the expected structure."""

    dataset_path = Path(directory).expanduser().resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")

    splits: Dict[str, Path] = {}
    num_images: Dict[str, int] = {}
    num_annotations: Dict[str, int] = {}
    categories: Dict[int, str] = {}

    for split, rel_path in DATASET_STRUCTURE.items():
        annotation_path = dataset_path / rel_path
        if not annotation_path.exists():
            raise FileNotFoundError(f"Missing annotation file for split '{split}': {annotation_path}")
        splits[split] = annotation_path

        coco = COCO(annotation_path)
        img_ids = coco.getImgIds()
        ann_ids = coco.getAnnIds()
        num_images[split] = len(img_ids)
        num_annotations[split] = len(ann_ids)

        if not categories:
            cats = coco.loadCats(coco.getCatIds())
            categories = {cat["id"]: cat["name"] for cat in cats}

    return DatasetSummary(
        path=dataset_path,
        splits=splits,
        categories=categories,
        num_images=num_images,
        num_annotations=num_annotations,
    )


def export_label_mapping(summary: DatasetSummary, output_path: str | Path) -> None:
    """Export the dataset label mapping as a JSON file."""

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    label_map = {str(k): v for k, v in summary.categories.items()}
    with open(output_file, "w", encoding="utf-8") as fp:
        json.dump({"id2label": label_map}, fp, indent=2)


def load_label_mapping(path: Optional[str | Path]) -> Optional[Dict[int, str]]:
    """Load a label mapping JSON produced by :func:`export_label_mapping`."""

    if path is None:
        return None

    with open(path, "r", encoding="utf-8") as fp:
        payload = json.load(fp)

    id2label = payload.get("id2label")
    if id2label is None:
        raise ValueError("Label mapping file is missing 'id2label' key")

    return {int(k): str(v) for k, v in id2label.items()}
