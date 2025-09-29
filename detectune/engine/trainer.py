"""Training loop for MaskDINO."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from detectune.engine.checkpoint import CheckpointManager
from detectune.engine.metrics import CocoMetrics
from detectune.logging import NullLogger


@dataclass
class TrainState:
    epoch: int = 0
    global_step: int = 0


class Trainer:
    """Orchestrates the training and validation loops."""

    def __init__(
        self,
        model,
        optimizer,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        device: torch.device,
        max_grad_norm: Optional[float],
        grad_accumulation_steps: int,
        processor,
        scheduler=None,
        logger=None,
        checkpoint_manager: Optional[CheckpointManager] = None,
        log_every_n_steps: int = 10,
        save_every_n_epochs: int = 1,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.max_grad_norm = max_grad_norm
        self.grad_accumulation_steps = grad_accumulation_steps
        self.processor = processor
        self.scheduler = scheduler
        self.logger = logger or NullLogger()
        self.checkpoint_manager = checkpoint_manager
        self.log_every_n_steps = log_every_n_steps
        self.save_every_n_epochs = max(save_every_n_epochs, 1)
        self.val_dataset = getattr(val_loader, "dataset", None) if val_loader is not None else None

        self.state = TrainState()
        self.model.to(self.device)

    @staticmethod
    def _sanitize_metric_name(name: str) -> str:
        """Normalize metric keys to be filesystem- and logger-friendly."""

        return name.replace("/", "-").replace(" ", "_")

    def _flatten_metrics(self, metrics: Dict[str, Any], prefix: str = "") -> Dict[str, float]:
        """Flatten nested metric dictionaries into dot-delimited keys."""

        flat: Dict[str, float] = {}
        for key, value in metrics.items():
            key_path = f"{prefix}{key}" if not prefix else f"{prefix}.{key}"
            if isinstance(value, dict):
                flat.update(self._flatten_metrics(value, key_path))
            else:
                try:
                    flat[key_path] = float(value)
                except (TypeError, ValueError):
                    continue
        return flat

    def _prepare_epoch_metrics(
        self,
        train_loss: float,
        val_loss: Optional[float],
        val_metrics: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Aggregate train/validation metrics into a structured dictionary."""

        epoch_metrics: Dict[str, Any] = {"train": {"loss": train_loss}}

        if val_loss is not None or val_metrics is not None:
            epoch_metrics.setdefault("valid", {})

        if val_loss is not None:
            epoch_metrics["valid"]["loss"] = val_loss

        if val_metrics is None:
            return epoch_metrics

        detection_metrics = val_metrics.get("detection")
        if isinstance(detection_metrics, dict) and detection_metrics:
            epoch_metrics["valid"]["detection"] = {
                self._sanitize_metric_name(name): value
                for name, value in detection_metrics.items()
            }

        classification_metrics = val_metrics.get("classification")
        if isinstance(classification_metrics, dict) and classification_metrics:
            valid_classification: Dict[str, Any] = {}

            overall_accuracy = classification_metrics.get("overall_accuracy")
            if overall_accuracy is not None:
                valid_classification["overall_accuracy"] = overall_accuracy

            per_class_accuracy = classification_metrics.get("per_class_accuracy")
            if isinstance(per_class_accuracy, dict) and per_class_accuracy:
                valid_classification["per_class_accuracy"] = {
                    self._sanitize_metric_name(class_name): accuracy
                    for class_name, accuracy in per_class_accuracy.items()
                }

            if valid_classification:
                epoch_metrics["valid"]["classification"] = valid_classification

        return epoch_metrics

    def _move_to_device(self, batch: Dict[str, torch.Tensor]):
        def move(value):
            if isinstance(value, torch.Tensor):
                return value.to(self.device)
            if isinstance(value, list):
                return [move(item) for item in value]
            if isinstance(value, dict):
                return {k: move(v) for k, v in value.items()}
            return value

        return {key: move(value) for key, value in batch.items()}

    def _forward_backward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        batch = self._move_to_device(batch)
        outputs = self.model(**batch)
        loss = outputs.loss / self.grad_accumulation_steps
        loss.backward()
        return outputs.loss.detach()

    def train_epoch(self) -> float:
        self.model.train()
        running_loss = 0.0
        progress = tqdm(self.train_loader, desc=f"Epoch {self.state.epoch + 1}", leave=False)
        self.optimizer.zero_grad()

        for step, batch in enumerate(progress, start=1):
            loss = self._forward_backward(batch)
            running_loss += loss.item()

            if step % self.grad_accumulation_steps == 0:
                if self.max_grad_norm is not None:
                    clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                self.optimizer.zero_grad()
                self.state.global_step += 1

                if self.state.global_step % self.log_every_n_steps == 0:
                    avg_loss = running_loss / step
                    self.logger.save_metrics("train", "loss", avg_loss, step=self.state.global_step)

        return running_loss / max(len(self.train_loader), 1)

    @torch.no_grad()
    def evaluate(self) -> Tuple[Optional[float], Optional[Dict[str, Any]]]:
        if self.val_loader is None:
            return None, None

        self.model.eval()
        running_loss = 0.0
        metrics_tracker = None
        if self.val_dataset is not None:
            metrics_tracker = CocoMetrics(
                coco=self.val_dataset.coco,
                contiguous_id_to_cat_id=self.val_dataset.contiguous_id_to_cat_id,
                id2label=self.val_dataset.id2label,
                processor=self.processor,
            )
        for batch in self.val_loader:
            batch = self._move_to_device(batch)
            outputs = self.model(**batch)
            running_loss += outputs.loss.item()
            if metrics_tracker is not None:
                metrics_tracker.update(outputs, batch["labels"])

        metrics = metrics_tracker.compute() if metrics_tracker is not None else None
        return running_loss / max(len(self.val_loader), 1), metrics

    def fit(self, num_epochs: int, eval_every_n_epochs: int = 1) -> None:
        for epoch in range(num_epochs):
            self.state.epoch = epoch
            train_loss = self.train_epoch()
            self.logger.save_metrics("train", "epoch_loss", train_loss, step=self.state.global_step)

            val_loss = None
            val_metrics = None
            if self.val_loader is not None and ((epoch + 1) % max(eval_every_n_epochs, 1) == 0):
                val_loss, val_metrics = self.evaluate()
                if val_loss is not None:
                    self.logger.save_metrics("valid", "loss", val_loss, step=self.state.global_step)
                if val_metrics is not None:
                    detection_metrics = val_metrics.get("detection", {})
                    if detection_metrics:
                        names = [f"detection_{name}" for name in detection_metrics.keys()]
                        values = list(detection_metrics.values())
                        self.logger.save_metrics("valid", names, values, step=self.state.global_step)

                    classification_metrics = val_metrics.get("classification", {})
                    overall_accuracy = classification_metrics.get("overall_accuracy")
                    if overall_accuracy is not None:
                        self.logger.save_metrics(
                            "valid",
                            "classification_overall_accuracy",
                            overall_accuracy,
                            step=self.state.global_step,
                        )

                    per_class_accuracy = classification_metrics.get("per_class_accuracy", {})
                    if per_class_accuracy:
                        metric_names = []
                        metric_values = []
                        for class_name, accuracy in per_class_accuracy.items():
                            safe_name = class_name.replace("/", "-").replace(" ", "_")
                            metric_names.append(f"class_accuracy_{safe_name}")
                            metric_values.append(accuracy)
                        self.logger.save_metrics("valid", metric_names, metric_values, step=self.state.global_step)

            epoch_metrics = self._prepare_epoch_metrics(train_loss, val_loss, val_metrics)
            flattened_metrics = self._flatten_metrics(epoch_metrics)
            # Backwards compatibility aliases
            if "train.loss" in flattened_metrics:
                flattened_metrics.setdefault("train_loss", flattened_metrics["train.loss"])
            if "valid.loss" in flattened_metrics:
                flattened_metrics.setdefault("val_loss", flattened_metrics["valid.loss"])
                flattened_metrics.setdefault("valid_loss", flattened_metrics["valid.loss"])

            if self.checkpoint_manager is not None and ((epoch + 1) % self.save_every_n_epochs == 0):
                monitor_key = self.checkpoint_manager.monitor
                monitor_value = flattened_metrics.get(monitor_key)
                if monitor_value is None and monitor_key in {"val_loss", "valid.loss", "valid_loss"}:
                    monitor_value = flattened_metrics.get("train.loss")
                self.checkpoint_manager.save_checkpoint(
                    epoch=epoch + 1,
                    model=self.model,
                    processor=self.processor,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    monitor_value=monitor_value,
                )

        self.logger.stop()
