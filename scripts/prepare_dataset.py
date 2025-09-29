#!/usr/bin/env python
"""Validate dataset structure and optionally export label mappings."""

from __future__ import annotations

import argparse
from pathlib import Path

from detectune.data.processor import export_label_mapping, validate_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True, help="Path to the dataset root directory")
    parser.add_argument(
        "--output-labels",
        type=str,
        default=None,
        help="Optional path to write the inferred id2label mapping as JSON",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = validate_dataset(args.dataset_dir)

    print(f"Dataset: {summary.path}")
    print("Splits:")
    for split, path in summary.splits.items():
        print(f"  - {split}: {path}")

    print("Images per split:")
    for split, count in summary.num_images.items():
        print(f"  - {split}: {count}")

    print("Annotations per split:")
    for split, count in summary.num_annotations.items():
        print(f"  - {split}: {count}")

    print("Categories:")
    for cat_id, name in summary.categories.items():
        print(f"  - {cat_id}: {name}")

    if args.output_labels:
        export_label_mapping(summary, args.output_labels)
        print(f"Label mapping written to {args.output_labels}")


if __name__ == "__main__":
    main()
