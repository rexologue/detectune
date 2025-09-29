"""Experiment configuration dataclasses and loader."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class ExperimentConfig:
    seed: int = 42
    output_dir: str = "outputs"


@dataclass
class ModelConfig:
    name_or_path: str
    models_dir: Optional[str] = None
    ignore_mismatched_sizes: bool = False


@dataclass
class OptimizerConfig:
    lr: float
    betas: List[float]
    weight_decay: float = 0.0
    eps: float = 1e-8


@dataclass
class SchedulerConfig:
    warmup_steps: int = 0
    total_training_steps: Optional[int] = None


@dataclass
class TrainingConfig:
    batch_size: int
    gradient_accumulation_steps: int = 1
    num_epochs: int = 10
    max_grad_norm: Optional[float] = None
    log_every_n_steps: int = 10
    eval_every_n_epochs: int = 1
    device: str = "cuda"


@dataclass
class CheckpointConfig:
    dir: str
    save_every_n_epochs: int = 1
    monitor: str = "val_loss"
    mode: str = "min"
    keep_last_n: int = 3


@dataclass
class NeptuneConfig:
    enabled: bool = False
    project: Optional[str] = None
    api_token: Optional[str] = None
    run_id: Optional[str] = None
    experiment_name: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    env_path: Optional[str] = None
    dependencies_path: Optional[str] = None


@dataclass
class DataConfig:
    dataset_dir: str
    train_split: str = "train"
    valid_split: str = "valid"
    test_split: str = "test"
    num_workers: int = 4
    image_size: Optional[int] = None
    remove_empty_annotations: bool = True


@dataclass
class Config:
    experiment: ExperimentConfig
    model: ModelConfig
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    training: TrainingConfig
    checkpointing: CheckpointConfig
    neptune: NeptuneConfig
    data: DataConfig


def _build_dataclass(cls, data: Dict[str, Any]):
    field_names = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    filtered = {k: v for k, v in data.items() if k in field_names}
    return cls(**filtered)


def load_config(path: str | Path) -> Config:
    """Load configuration from YAML file into dataclasses."""

    with open(path, "r", encoding="utf-8") as fp:
        raw_cfg = yaml.safe_load(fp)

    experiment = _build_dataclass(ExperimentConfig, raw_cfg.get("experiment", {}))
    model = _build_dataclass(ModelConfig, raw_cfg.get("model", {}))
    optimizer = _build_dataclass(OptimizerConfig, raw_cfg.get("optimizer", {}))
    scheduler = _build_dataclass(SchedulerConfig, raw_cfg.get("scheduler", {}))
    training = _build_dataclass(TrainingConfig, raw_cfg.get("training", {}))
    checkpointing = _build_dataclass(CheckpointConfig, raw_cfg.get("checkpointing", {}))
    neptune = _build_dataclass(NeptuneConfig, raw_cfg.get("neptune", {}))
    data = _build_dataclass(DataConfig, raw_cfg.get("data", {}))

    return Config(
        experiment=experiment,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        training=training,
        checkpointing=checkpointing,
        neptune=neptune,
        data=data,
    )
