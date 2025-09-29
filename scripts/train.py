#!/usr/bin/env python3
"""Train an MMDetection model on a COCO-style dataset."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from mmengine.config import Config, DictAction
from mmengine.runner import Runner

from mmdet.utils import register_all_modules

from _config_utils import infer_and_populate_classes, load_settings, propagate_data_root

DEFAULT_CONFIG = 'configs/custom_dataset/faster-rcnn_r50_fpn_custom.py'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--config',
        default=DEFAULT_CONFIG,
        help='Path to the MMDetection config file.',
    )
    parser.add_argument(
        '--settings',
        default=None,
        help='Optional YAML file with defaults for config, data_root, work_dir, etc.',
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


def _select_value(cli_value: Any, default: Any, settings: dict, key: str) -> Any:
    if cli_value is not None and cli_value != default:
        return cli_value
    if key in settings:
        return settings[key]
    return cli_value if cli_value is not None else default


def main() -> None:
    args = parse_args()

    settings = load_settings(args.settings)

    config_path = _select_value(args.config, DEFAULT_CONFIG, settings, 'config')
    work_dir = _select_value(args.work_dir, None, settings, 'work_dir')
    data_root = _select_value(args.data_root, None, settings, 'data_root')

    auto_resume = args.auto_resume or bool(settings.get('auto_resume', False))

    cfg = Config.fromfile(config_path)

    if 'cfg_options' in settings:
        cfg_options = settings['cfg_options']
        if not isinstance(cfg_options, dict):
            raise TypeError('cfg_options in the settings file must be a mapping.')
        cfg.merge_from_dict(cfg_options)

    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)

    if work_dir:
        cfg.work_dir = work_dir
    elif not cfg.get('work_dir'):
        cfg.work_dir = os.path.join('work_dirs', Path(config_path).stem)

    cfg.auto_resume = auto_resume

    propagate_data_root(cfg, data_root)
    infer_and_populate_classes(cfg)

    register_all_modules(init_default_scope=False)

    os.makedirs(cfg.work_dir, exist_ok=True)

    runner = Runner.from_cfg(cfg)
    runner.train()


if __name__ == '__main__':
    main()
