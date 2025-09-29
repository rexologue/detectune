"""Shared helpers for working with MMDetection configuration objects."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional

from mmengine.config import Config


def _maybe_wrap_dataset(dataloader: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
    dataset_cfg = dataloader.get('dataset', dataloader)
    if isinstance(dataset_cfg, MutableMapping):
        return dataset_cfg
    raise TypeError(
        'Expected dataloader to expose a dataset mapping, '
        f'but received type {type(dataset_cfg)!r}.'
    )


def propagate_data_root(cfg: Config, data_root: Optional[str]) -> None:
    """Propagate ``data_root`` to all dataset and evaluator sections."""

    if not data_root:
        return

    cfg.data_root = data_root

    def _update_dataset(dataset: MutableMapping[str, Any]) -> None:
        dataset['data_root'] = data_root
        ann_file = dataset.get('ann_file')
        if ann_file:
            dataset['ann_file'] = str(ann_file)
        if 'metainfo' not in dataset and hasattr(cfg, 'metainfo') and cfg.metainfo:
            dataset['metainfo'] = cfg.metainfo

    for loader_key in ['train_dataloader', 'val_dataloader', 'test_dataloader']:
        if loader_key not in cfg:
            continue
        dataloader = cfg[loader_key]
        if isinstance(dataloader, MutableMapping):
            dataset_cfg = _maybe_wrap_dataset(dataloader)
            _update_dataset(dataset_cfg)

    for evaluator_key in ['val_evaluator', 'test_evaluator']:
        if evaluator_key not in cfg:
            continue
        evaluator = cfg[evaluator_key]
        if isinstance(evaluator, MutableMapping) and evaluator.get('ann_file'):
            ann_path = Path(evaluator['ann_file'])
            if not ann_path.is_absolute():
                evaluator['ann_file'] = str(Path(data_root) / ann_path)


def _resolve_ann_path(cfg: Config, dataset_cfg: Mapping[str, Any]) -> Optional[Path]:
    ann_file = dataset_cfg.get('ann_file')
    if not ann_file:
        return None

    ann_path = Path(str(ann_file))
    if ann_path.is_absolute():
        return ann_path

    data_root = dataset_cfg.get('data_root') or getattr(cfg, 'data_root', None)
    if data_root:
        candidate = Path(str(data_root)) / ann_path
        if candidate.exists():
            return candidate

    # Fall back to interpreting the annotation file relative to the working dir.
    return ann_path if ann_path.exists() else None


def infer_and_populate_classes(cfg: Config) -> None:
    """Infer dataset classes from the training annotations if missing."""

    if not hasattr(cfg, 'train_dataloader'):
        return

    train_loader = cfg.train_dataloader
    if not isinstance(train_loader, MutableMapping):
        return

    dataset_cfg = _maybe_wrap_dataset(train_loader)
    current_meta = dataset_cfg.get('metainfo') or getattr(cfg, 'metainfo', {})
    classes = current_meta.get('classes') if isinstance(current_meta, Mapping) else None
    if classes:
        return

    ann_path = _resolve_ann_path(cfg, dataset_cfg)
    if not ann_path or not ann_path.exists():
        return

    with ann_path.open('r', encoding='utf-8') as f:
        data = json.load(f)

    categories: Iterable[Mapping[str, Any]] = data.get('categories', [])
    if not categories:
        return

    sorted_categories = sorted(categories, key=lambda c: c.get('id', c.get('name', '')))
    inferred_classes = tuple(cat.get('name', str(cat.get('id'))) for cat in sorted_categories)

    cfg.metainfo = dict(getattr(cfg, 'metainfo', {}), classes=inferred_classes)

    for loader_key in ['train_dataloader', 'val_dataloader', 'test_dataloader']:
        if loader_key not in cfg:
            continue
        dataloader = cfg[loader_key]
        if not isinstance(dataloader, MutableMapping):
            continue
        dataset = _maybe_wrap_dataset(dataloader)
        existing = dataset.get('metainfo', {})
        dataset['metainfo'] = dict(existing, classes=inferred_classes)


def load_settings(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}

    settings_path = Path(path)
    if not settings_path.exists():
        raise FileNotFoundError(f'Settings file not found: {settings_path}')

    import yaml

    with settings_path.open('r', encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError('Settings YAML must define a mapping at the top level.')
    return data
