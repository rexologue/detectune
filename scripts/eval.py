#!/usr/bin/env python3
"""Evaluate an MMDetection checkpoint on the validation or test split."""

from __future__ import annotations

import argparse
import json
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

    for loader_key in ['val_dataloader', 'test_dataloader']:
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
        required=True,
        help='Path to the MMDetection config file.',
    )
    parser.add_argument(
        '--checkpoint',
        required=True,
        help='Checkpoint file to evaluate.',
    )
    parser.add_argument(
        '--data-root',
        default=None,
        help='Override the dataset root directory declared in the config.',
    )
    parser.add_argument(
        '--work-dir',
        default=None,
        help='Directory for evaluation artifacts (defaults to the checkpoint directory).',
    )
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        default=None,
        help='Override config options, e.g. test_dataloader.batch_size=2',
    )
    parser.add_argument(
        '--split',
        choices=['val', 'test'],
        default='val',
        help='Which split to evaluate.',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)

    if args.work_dir:
        eval_dir = Path(args.work_dir)
    else:
        eval_dir = Path(args.checkpoint).resolve().parent
    eval_dir.mkdir(parents=True, exist_ok=True)

    _propagate_data_root(cfg, args.data_root)

    register_all_modules(init_default_scope=False)

    cfg.resume = False
    cfg.load_from = None

    runner = Runner.from_cfg(cfg)
    runner.load_checkpoint(args.checkpoint)

    if args.split == 'test':
        results = runner.test()
        metrics = results[0] if isinstance(results, list) else results
    else:
        results = runner.val()
        metrics = results

    metrics_path = eval_dir / f'eval_metrics_{args.split}.json'
    with metrics_path.open('w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
