#!/usr/bin/env python3
"""Evaluate an MMDetection checkpoint on the validation or test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mmengine.config import Config, DictAction
from mmengine.runner import Runner

from mmdet.utils import register_all_modules

from _config_utils import infer_and_populate_classes, load_settings, propagate_data_root


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
        '--settings',
        default=None,
        help='Optional YAML file to provide data_root, work_dir, or cfg_options defaults.',
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

    settings = load_settings(args.settings)

    data_root = args.data_root or settings.get('data_root')
    work_dir = args.work_dir or settings.get('work_dir')

    cfg = Config.fromfile(args.config)

    if 'cfg_options' in settings:
        cfg_options = settings['cfg_options']
        if not isinstance(cfg_options, dict):
            raise TypeError('cfg_options in the settings file must be a mapping.')
        cfg.merge_from_dict(cfg_options)

    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)

    if work_dir:
        eval_dir = Path(work_dir)
    else:
        eval_dir = Path(args.checkpoint).resolve().parent
    eval_dir.mkdir(parents=True, exist_ok=True)

    propagate_data_root(cfg, data_root)
    infer_and_populate_classes(cfg)

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
