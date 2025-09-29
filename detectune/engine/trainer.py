"""Training loop for MaskDINO."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from detectune.engine.checkpoint import CheckpointManager
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

        self.state = TrainState()
        self.model.to(self.device)

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
    def evaluate(self) -> Optional[float]:
        if self.val_loader is None:
            return None

        self.model.eval()
        running_loss = 0.0
        for batch in self.val_loader:
            batch = self._move_to_device(batch)
            outputs = self.model(**batch)
            running_loss += outputs.loss.item()

        return running_loss / max(len(self.val_loader), 1)

    def fit(self, num_epochs: int, eval_every_n_epochs: int = 1) -> None:
        for epoch in range(num_epochs):
            self.state.epoch = epoch
            train_loss = self.train_epoch()
            self.logger.save_metrics("train", "epoch_loss", train_loss, step=self.state.global_step)

            val_loss = None
            if self.val_loader is not None and ((epoch + 1) % max(eval_every_n_epochs, 1) == 0):
                val_loss = self.evaluate()
                if val_loss is not None:
                    self.logger.save_metrics("valid", "loss", val_loss, step=self.state.global_step)

            if self.checkpoint_manager is not None and ((epoch + 1) % self.save_every_n_epochs == 0):
                monitor_value = val_loss if val_loss is not None else train_loss
                self.checkpoint_manager.save_checkpoint(
                    epoch=epoch + 1,
                    model=self.model,
                    processor=self.processor,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    monitor_value=monitor_value,
                )

        self.logger.stop()
