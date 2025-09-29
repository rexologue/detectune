#!/usr/bin/env python
"""Entry-point for training MaskDINO models using Detectune."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from detectune.config import load_config
from detectune.data.dataset import CocoSegmentationDataset, build_collate_fn
from detectune.data.processor import validate_dataset
from detectune.engine import CheckpointManager, Trainer
from detectune.logging import NeptuneLogger, NullLogger
from detectune.models import load_model_and_processor
from detectune.utils import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to experiment YAML configuration")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    set_seed(config.experiment.seed)

    output_dir = Path(config.experiment.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_config_path = output_dir / "config.yaml"
    resolved_config_path.write_text(Path(args.config).read_text())

    requested_device = config.training.device
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(requested_device)

    validate_dataset(config.data.dataset_dir)

    train_dataset = CocoSegmentationDataset(
        dataset_dir=config.data.dataset_dir,
        split=config.data.train_split,
        remove_empty=config.data.remove_empty_annotations,
    )
    val_dataset = None
    if config.data.valid_split:
        val_dataset = CocoSegmentationDataset(
            dataset_dir=config.data.dataset_dir,
            split=config.data.valid_split,
            remove_empty=config.data.remove_empty_annotations,
        )

    model, processor = load_model_and_processor(
        config.model.name_or_path,
        models_dir=config.model.models_dir,
        id2label=train_dataset.id2label,
        ignore_mismatched_sizes=config.model.ignore_mismatched_sizes,
    )

    if config.data.image_size:
        processor.size = {"shortest_edge": config.data.image_size}
        if hasattr(processor, "max_size") and processor.max_size is not None:
            processor.max_size = config.data.image_size

    collate_fn = build_collate_fn(processor)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        collate_fn=collate_fn,
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.training.batch_size,
            shuffle=False,
            num_workers=config.data.num_workers,
            collate_fn=collate_fn,
        )

    optimizer = AdamW(
        model.parameters(),
        lr=config.optimizer.lr,
        betas=tuple(config.optimizer.betas),
        weight_decay=config.optimizer.weight_decay,
        eps=config.optimizer.eps,
    )

    total_steps = config.scheduler.total_training_steps
    if total_steps is None:
        steps_per_epoch = math.ceil(len(train_loader) / config.training.gradient_accumulation_steps)
        steps_per_epoch = max(steps_per_epoch, 1)
        total_steps = steps_per_epoch * config.training.num_epochs

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.scheduler.warmup_steps,
        num_training_steps=total_steps,
    )

    if config.neptune.enabled:
        logger = NeptuneLogger(config.neptune)
    else:
        logger = NullLogger()

    logger.log_hyperparameters(
        {
            "model": config.model.name_or_path,
            "optimizer": {
                "lr": config.optimizer.lr,
                "betas": config.optimizer.betas,
                "weight_decay": config.optimizer.weight_decay,
                "eps": config.optimizer.eps,
            },
            "training": {
                "batch_size": config.training.batch_size,
                "gradient_accumulation_steps": config.training.gradient_accumulation_steps,
                "num_epochs": config.training.num_epochs,
                "max_grad_norm": config.training.max_grad_norm,
            },
            "data": {
                "dataset_dir": str(Path(config.data.dataset_dir).resolve()),
                "train_split": config.data.train_split,
                "valid_split": config.data.valid_split,
                "image_size": config.data.image_size,
            },
        }
    )

    checkpoint_manager = CheckpointManager(
        directory=config.checkpointing.dir,
        monitor=config.checkpointing.monitor,
        mode=config.checkpointing.mode,
        keep_last_n=config.checkpointing.keep_last_n,
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        max_grad_norm=config.training.max_grad_norm,
        grad_accumulation_steps=config.training.gradient_accumulation_steps,
        processor=processor,
        scheduler=scheduler,
        logger=logger,
        checkpoint_manager=checkpoint_manager,
        log_every_n_steps=config.training.log_every_n_steps,
        save_every_n_epochs=config.checkpointing.save_every_n_epochs,
    )

    trainer.fit(config.training.num_epochs, eval_every_n_epochs=config.training.eval_every_n_epochs)


if __name__ == "__main__":
    main()
