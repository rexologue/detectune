"""Utility helpers for computing object detection and classification metrics."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def _to_numpy(array: Any) -> np.ndarray:
    """Convert tensors or lists to ``np.ndarray`` for easier manipulation."""

    if isinstance(array, torch.Tensor):
        return array.detach().cpu().numpy()
    if isinstance(array, np.ndarray):
        return array
    return np.asarray(array)


def _xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """Convert ``[x, y, w, h]`` boxes to ``[x1, y1, x2, y2]`` format."""

    if boxes.size == 0:
        return boxes.reshape(0, 4)
    xyxy = boxes.copy()
    xyxy[:, 2] = xyxy[:, 0] + xyxy[:, 2]
    xyxy[:, 3] = xyxy[:, 1] + xyxy[:, 3]
    return xyxy


def _box_iou(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """Compute IoU matrix between two sets of ``[x1, y1, x2, y2]`` boxes."""

    if boxes1.size == 0 or boxes2.size == 0:
        return np.zeros((boxes1.shape[0], boxes2.shape[0]), dtype=np.float32)

    area1 = (boxes1[:, 2] - boxes1[:, 0]).clip(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clip(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clip(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clip(min=0)

    lt = np.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = np.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])

    wh = (rb - lt).clip(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    union = area1[:, None] + area2[None, :] - inter
    union = np.clip(union, a_min=1e-6, a_max=None)
    return inter / union


def _infer_device(outputs: Any) -> torch.device:
    """Best-effort attempt at retrieving the device of model outputs."""

    for attr in ("pred_masks", "pred_boxes", "logits"):
        tensor = getattr(outputs, attr, None)
        if isinstance(tensor, torch.Tensor):
            return tensor.device
    raise AttributeError("Unable to infer device from model outputs.")


class CocoMetrics:
    """Accumulates predictions to produce COCO-style metrics and class accuracy."""

    def __init__(
        self,
        coco: COCO,
        contiguous_id_to_cat_id: Dict[int, int],
        id2label: Dict[int, str],
        processor: Any,
        classification_iou_threshold: float = 0.5,
    ) -> None:
        self.coco = coco
        self.contiguous_id_to_cat_id = contiguous_id_to_cat_id
        self.id2label = id2label
        self.processor = processor
        self.classification_iou_threshold = classification_iou_threshold
        self.cat_id_to_contiguous = {cat_id: cont_id for cont_id, cat_id in contiguous_id_to_cat_id.items()}
        self.reset()

    def reset(self) -> None:
        """Reset internal buffers for a new evaluation run."""

        self._detections: List[Dict[str, float]] = []
        self._class_correct = defaultdict(int)
        self._class_totals = defaultdict(int)
        self._total_correct = 0
        self._total_gt = 0

    def update(self, outputs: Any, labels: Sequence[Dict[str, Any]]) -> None:
        """Update metrics buffers with a batch of model outputs."""

        device = _infer_device(outputs)
        target_sizes: List[torch.Tensor] = []
        for label in labels:
            size = label.get("size")
            if isinstance(size, torch.Tensor):
                target_sizes.append(size.to(device))
            else:
                target_sizes.append(torch.tensor(size, device=device))
        target_sizes_tensor = torch.stack(target_sizes).to(device=device, dtype=torch.int64)

        processed = self.processor.post_process_instance_segmentation(
            outputs,
            threshold=0.0,
            target_sizes=target_sizes_tensor,
        )

        for label, prediction in zip(labels, processed):
            image_id_tensor = label.get("image_id")
            if isinstance(image_id_tensor, torch.Tensor):
                image_id = int(image_id_tensor.item())
            else:
                image_id = int(image_id_tensor)

            ann_ids = self.coco.getAnnIds(imgIds=[image_id], iscrowd=None)
            annotations = self.coco.loadAnns(ann_ids)
            gt_boxes_list: List[List[float]] = []
            gt_labels: List[int] = []
            for ann in annotations:
                if ann.get("iscrowd", 0) == 1:
                    continue
                bbox = ann.get("bbox")
                category_id = ann.get("category_id")
                if bbox is None or category_id is None:
                    continue
                contiguous_label = self.cat_id_to_contiguous.get(int(category_id))
                if contiguous_label is None:
                    continue
                gt_boxes_list.append(bbox)
                gt_labels.append(contiguous_label)
                self._class_totals[contiguous_label] += 1
                self._total_gt += 1
            gt_boxes = np.asarray(gt_boxes_list, dtype=np.float32)

            pred_boxes = prediction.get("boxes")
            pred_scores = prediction.get("scores")
            pred_labels = prediction.get("labels")

            if pred_boxes is None or pred_scores is None or pred_labels is None:
                continue

            pred_boxes_np = _to_numpy(pred_boxes).reshape(-1, 4)
            pred_scores_np = _to_numpy(pred_scores).reshape(-1)
            pred_labels_np = _to_numpy(pred_labels).astype(np.int64)

            # Detection metrics (COCO bbox format requires xywh)
            for box, score, label_idx in zip(pred_boxes_np, pred_scores_np, pred_labels_np):
                x_min, y_min, x_max, y_max = box.tolist()
                bbox_xywh = [x_min, y_min, max(x_max - x_min, 0.0), max(y_max - y_min, 0.0)]
                category_id = self.contiguous_id_to_cat_id.get(int(label_idx))
                if category_id is None:
                    continue
                self._detections.append(
                    {
                        "image_id": image_id,
                        "category_id": int(category_id),
                        "bbox": [float(f"{coord:.3f}") for coord in bbox_xywh],
                        "score": float(f"{score:.5f}"),
                    }
                )

            # Classification accuracy via IoU matching
            if gt_boxes.size == 0 or pred_boxes_np.size == 0:
                continue

            gt_boxes_xyxy = _xywh_to_xyxy(gt_boxes)
            iou_matrix = _box_iou(pred_boxes_np, gt_boxes_xyxy)
            matched_gt = set()

            for pred_idx in np.argsort(-pred_scores_np):
                gt_idx = int(np.argmax(iou_matrix[pred_idx]))
                best_iou = float(iou_matrix[pred_idx, gt_idx])
                if best_iou < self.classification_iou_threshold or gt_idx in matched_gt:
                    continue
                matched_gt.add(gt_idx)

                predicted_label = int(pred_labels_np[pred_idx])
                gt_label = int(gt_labels[gt_idx])
                if predicted_label == gt_label:
                    self._class_correct[gt_label] += 1
                    self._total_correct += 1

    def compute(self) -> Dict[str, Any]:
        """Compute detection and classification metrics from accumulated data."""

        detection_metrics = {
            "mAP": 0.0,
            "mAP50": 0.0,
            "mAP75": 0.0,
            "mean_recall": 0.0,
        }

        if self._detections:
            coco_dt = self.coco.loadRes(self._detections)
            evaluator = COCOeval(self.coco, coco_dt, iouType="bbox")
            evaluator.evaluate()
            evaluator.accumulate()
            evaluator.summarize()
            stats = evaluator.stats
            detection_metrics = {
                "mAP": float(stats[0]),
                "mAP50": float(stats[1]),
                "mAP75": float(stats[2]),
                "mean_recall": float(stats[8]),
            }

        per_class_accuracy: Dict[str, float] = {}
        for class_idx, total in self._class_totals.items():
            label_name = self.id2label.get(class_idx, str(class_idx))
            correct = self._class_correct.get(class_idx, 0)
            per_class_accuracy[label_name] = float(correct / total) if total else 0.0

        overall_accuracy = float(self._total_correct / self._total_gt) if self._total_gt else 0.0

        return {
            "detection": detection_metrics,
            "classification": {
                "overall_accuracy": overall_accuracy,
                "per_class_accuracy": per_class_accuracy,
            },
        }

