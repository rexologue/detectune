"""Utilities for loading MaskDINO models and processors."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

from transformers import MaskDinoForUniversalSegmentation, MaskDinoImageProcessor


def load_model_and_processor(
    name_or_path: str,
    models_dir: Optional[str],
    id2label: Dict[int, str],
    ignore_mismatched_sizes: bool = False,
):
    """Load MaskDINO model and image processor with the provided label mapping."""

    cache_kwargs = {"cache_dir": models_dir} if models_dir else {}

    processor = MaskDinoImageProcessor.from_pretrained(name_or_path, **cache_kwargs)
    model = MaskDinoForUniversalSegmentation.from_pretrained(
        name_or_path,
        id2label=id2label,
        label2id={label: idx for idx, label in id2label.items()},
        ignore_mismatched_sizes=ignore_mismatched_sizes,
        **cache_kwargs,
    )

    return model, processor
