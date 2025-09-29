"""Checkpoint management utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import shutil
from typing import Optional

import torch


@dataclass
class CheckpointState:
    epoch: int
    monitor_value: Optional[float]
    path: Path


class CheckpointManager:
    """Utility that handles periodic and best-model checkpointing."""

    def __init__(
        self,
        directory: str | Path,
        monitor: str = "val_loss",
        mode: str = "min",
        keep_last_n: int = 3,
    ) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.best_dir = self.directory / "best"
        self.best_dir.mkdir(exist_ok=True)

        if mode not in {"min", "max"}:
            raise ValueError("mode must be either 'min' or 'max'")
        self.monitor = monitor
        self.mode = mode
        self.keep_last_n = keep_last_n

        self._best_state: Optional[CheckpointState] = None
        self._history: list[CheckpointState] = []

    def _is_improvement(self, value: Optional[float]) -> bool:
        if value is None:
            return False
        if self._best_state is None:
            return True
        if self.mode == "min":
            return value < self._best_state.monitor_value
        return value > self._best_state.monitor_value

    def save_checkpoint(
        self,
        epoch: int,
        model,
        processor,
        optimizer,
        scheduler,
        monitor_value: Optional[float],
    ) -> Path:
        checkpoint_dir = self.directory / f"epoch_{epoch:04d}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        model.save_pretrained(checkpoint_dir)
        processor.save_pretrained(checkpoint_dir)
        torch.save(
            {
                "epoch": epoch,
                "monitor": monitor_value,
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict() if scheduler else None,
            },
            checkpoint_dir / "training_state.pt",
        )

        state = CheckpointState(epoch=epoch, monitor_value=monitor_value, path=checkpoint_dir)
        self._history.append(state)

        if self.keep_last_n and len(self._history) > self.keep_last_n:
            to_remove = self._history.pop(0)
            shutil.rmtree(to_remove.path, ignore_errors=True)

        if self._is_improvement(monitor_value):
            self._best_state = state
            model.save_pretrained(self.best_dir)
            processor.save_pretrained(self.best_dir)
            torch.save(
                {
                    "epoch": epoch,
                    "monitor": monitor_value,
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict() if scheduler else None,
                },
                self.best_dir / "training_state.pt",
            )
            meta = {
                "best_epoch": epoch,
                "monitor": self.monitor,
                "monitor_value": monitor_value,
            }
            with open(self.best_dir / "meta.json", "w", encoding="utf-8") as fp:
                json.dump(meta, fp, indent=2)

        return checkpoint_dir

    @property
    def best_state(self) -> Optional[CheckpointState]:
        return self._best_state
