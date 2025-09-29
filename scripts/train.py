#!/usr/bin/env python3
"""Train an MMDetection model on a COCO-style dataset."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict

from mmengine.config import Config, DictAction
from mmengine.runner import Runner

from mmdet.utils import register_all_modules


def _propagate_data_root(cfg: Config, data_root: str | None) -> None:
    if data_root is None:
        return
    cfg.data_root = data_root

    def _update_dataset(dataset: Dict[str, Any]) -> None:
        dataset['data_root'] = data_root
        if 'ann_file' in dataset:
            dataset['ann_file'] = str(dataset['ann_file'])
        if 'metainfo' not in dataset and hasattr(cfg, 'metainfo'):
            dataset['metainfo'] = cfg.metainfo

    for loader_key in ['train_dataloader', 'val_dataloader', 'test_dataloader']:
        if loader_key not in cfg:
            continue
        dataloader = cfg[loader_key]
        dataset_cfg = dataloader.get('dataset', dataloader)
        if isinstance(dataset_cfg, dict):
            _update_dataset(dataset_cfg)

    for evaluator_key in ['val_evaluator', 'test_evaluator']:
        if evaluator_key not in cfg:
            continue
        evaluator = cfg[evaluator_key]
        if isinstance(evaluator, dict) and evaluator.get('ann_file'):
            ann_path = Path(evaluator['ann_file'])
            if not ann_path.is_absolute():
                evaluator['ann_file'] = str(Path(data_root) / ann_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--config',
        default='configs/custom_dataset/faster-rcnn_r50_fpn_custom.py',
        help='Path to the MMDetection config file.',
    )
    parser.add_argument(
        '--work-dir',
        default=None,
        help='Directory for logs and checkpoints. Defaults to config.work_dir or work_dirs/<config_name>.',
    )
    parser.add_argument(
        '--data-root',
        default=None,
        help='Override the dataset root directory declared in the config.',
    )
    parser.add_argument(
        '--auto-resume',
        action='store_true',
        help='Resume automatically from the latest checkpoint in the work directory.',
    )
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        default=None,
        help='Override config options, e.g. train_dataloader.batch_size=4 optim_wrapper.optimizer.lr=0.01',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)

    if args.work_dir:
        cfg.work_dir = args.work_dir
    elif not cfg.get('work_dir'):
        cfg.work_dir = os.path.join('work_dirs', Path(args.config).stem)

    cfg.auto_resume = args.auto_resume

    _propagate_data_root(cfg, args.data_root)

    register_all_modules(init_default_scope=False)

    os.makedirs(cfg.work_dir, exist_ok=True)

    runner = Runner.from_cfg(cfg)
    runner.train()


if __name__ == '__main__':
    main()
