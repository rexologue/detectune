#!/usr/bin/env python3
"""Run a simple grid-search sweep over MMDetection configs."""

from __future__ import annotations

import argparse
import itertools
import subprocess
from collections.abc import Iterable
from pathlib import Path
from typing import Dict, List, Tuple

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--config',
        required=True,
        help='Base config to use for every run.',
    )
    parser.add_argument(
        '--search-space',
        required=True,
        help='YAML file describing the search space.',
    )
    parser.add_argument(
        '--data-root',
        required=True,
        help='Dataset root in COCO format.',
    )
    parser.add_argument(
        '--work-dir-base',
        default='work_dirs/tuning',
        help='Directory that will contain the sweep runs.',
    )
    parser.add_argument(
        '--max-runs',
        type=int,
        default=None,
        help='Optional cap on the number of runs (useful for smoke tests).',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print the commands without executing them.',
    )
    parser.add_argument(
        '--extra-train-args',
        nargs=argparse.REMAINDER,
        help='Extra arguments to forward to scripts/train.py after "--".',
    )
    return parser.parse_args()


def _load_search_space(path: Path) -> Dict[str, Iterable]:
    with path.open('r', encoding='utf-8') as f:
        space = yaml.safe_load(f) or {}
    processed = {}
    for key, values in space.items():
        if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
            raise ValueError(
                f'Search space values must be non-string iterables. '
                f'Key "{key}" has invalid type {type(values)!r}.'
            )
        processed[key] = list(values)
    return processed


def _iter_combinations(space: Dict[str, List]) -> Iterable[Tuple[Tuple[str, object], ...]]:
    keys = sorted(space.keys())
    values = [space[k] for k in keys]
    for combo in itertools.product(*values):
        yield tuple(zip(keys, combo))


def _format_run_name(combo: Tuple[Tuple[str, object], ...]) -> str:
    parts = []
    for key, value in combo:
        clean_value = str(value).replace('/', '-').replace(' ', '_')
        parts.append(f"{key}-{clean_value}")
    return '__'.join(parts) if parts else 'default'


def _build_cfg_options(combo: Tuple[Tuple[str, object], ...]) -> List[str]:
    mapping = {
        'learning_rate': 'optim_wrapper.optimizer.lr',
        'batch_size': 'train_dataloader.batch_size',
        'max_epochs': 'train_cfg.max_epochs',
        'weight_decay': 'optim_wrapper.optimizer.weight_decay',
    }
    options = []
    for key, value in combo:
        target = mapping.get(key, key)
        options.append(f'{target}={value}')
    return options


def main() -> None:
    args = parse_args()

    search_space = _load_search_space(Path(args.search_space))
    combinations = list(_iter_combinations(search_space))
    if args.max_runs is not None:
        combinations = combinations[: args.max_runs]

    base_work_dir = Path(args.work_dir_base)
    base_work_dir.mkdir(parents=True, exist_ok=True)

    for combo in combinations:
        run_name = _format_run_name(combo)
        work_dir = base_work_dir / run_name
        work_dir.mkdir(parents=True, exist_ok=True)
        cfg_options = _build_cfg_options(combo)

        latest_ckpt = work_dir / 'latest.pth'
        if latest_ckpt.exists():
            print(f'[skip] {run_name} - checkpoint already exists at {latest_ckpt}')
            continue

        cmd = [
            'python',
            'scripts/train.py',
            '--config',
            args.config,
            '--data-root',
            args.data_root,
            '--work-dir',
            str(work_dir),
        ]
        if cfg_options:
            cmd.extend(['--cfg-options', *cfg_options])
        if args.extra_train_args:
            cmd.append('--')
            cmd.extend(args.extra_train_args)

        print('[run]', ' '.join(cmd))
        if args.dry_run:
            continue

        subprocess.run(cmd, check=True)


if __name__ == '__main__':
    main()
