"""COCO dataset wrapper tailored for MaskDINO fine-tuning."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset


class CocoSegmentationDataset(Dataset):
    """PyTorch dataset returning raw images and annotations."""

    def __init__(
        self,
        dataset_dir: str | Path,
        split: str,
        remove_empty: bool = True,
    ) -> None:
        super().__init__()
        self.dataset_dir = Path(dataset_dir)
        self.split = split
        self.remove_empty = remove_empty

        annotation_path = self.dataset_dir / split / "_annotations.coco.json"
        if not annotation_path.exists():
            raise FileNotFoundError(f"Annotation file not found: {annotation_path}")

        self.coco = COCO(annotation_path)
        self.image_ids = self._collect_image_ids()

        cats = self.coco.loadCats(self.coco.getCatIds())
        id_to_name = {cat["id"]: cat["name"] for cat in cats}
        sorted_cat_ids = sorted(id_to_name.keys())
        self.cat_id_to_contiguous = {cat_id: idx for idx, cat_id in enumerate(sorted_cat_ids)}
        self.id2label = {idx: id_to_name[cat_id] for idx, cat_id in enumerate(sorted_cat_ids)}
        self.label2id = {label: idx for idx, label in self.id2label.items()}

    def _collect_image_ids(self) -> List[int]:
        image_ids = []
        for img_id in sorted(self.coco.getImgIds()):
            ann_ids = self.coco.getAnnIds(imgIds=[img_id], iscrowd=None)
            if self.remove_empty and len(ann_ids) == 0:
                continue
            image_ids.append(img_id)
        return image_ids

    def __len__(self) -> int:
        return len(self.image_ids)

    def _load_image(self, img_info: Dict) -> Image.Image:
        image_path = self.dataset_dir / self.split / img_info["file_name"]
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        return Image.open(image_path).convert("RGB")

    def _build_target(self, ann_list: List[Dict], img_info: Dict) -> Dict:
        boxes: List[List[float]] = []
        classes: List[int] = []
        masks: List[np.ndarray] = []

        for ann in ann_list:
            if ann.get("iscrowd", 0) == 1:
                continue
            bbox = ann.get("bbox")
            segmentation = ann.get("segmentation")
            category_id = ann.get("category_id")
            if bbox is None or segmentation is None or category_id is None:
                continue
            mask = self.coco.annToMask(ann)
            if mask.sum() == 0:
                continue
            boxes.append(bbox)
            classes.append(self.cat_id_to_contiguous[category_id])
            masks.append(mask)

        return {
            "image_id": img_info["id"],
            "class_labels": classes,
            "boxes": boxes,
            "masks": masks,
            "size": [img_info["height"], img_info["width"]],
        }

    def __getitem__(self, idx: int):
        img_id = self.image_ids[idx]
        img_info = self.coco.loadImgs([img_id])[0]
        ann_ids = self.coco.getAnnIds(imgIds=[img_id], iscrowd=None)
        ann_list = self.coco.loadAnns(ann_ids)

        image = self._load_image(img_info)
        target = self._build_target(ann_list, img_info)

        return image, target


def build_collate_fn(processor) -> Callable:
    """Return a collate function that leverages the Hugging Face image processor."""

    def collate_fn(batch: List[Tuple]) -> Dict:
        images, annotations = zip(*batch)
        encoded = processor(
            list(images),
            annotations=list(annotations),
            return_tensors="pt",
        )
        encoded = dict(encoded)
        labels = []
        for label in encoded["labels"]:
            labels.append({k: v for k, v in label.items()})
        encoded["labels"] = labels
        return encoded

    return collate_fn
