# Detectune MaskDINO Training Suite

Detectune is a clean-room rewrite focused on fine-tuning [MaskDINO](https://huggingface.co/docs/transformers/model_doc/maskdino)
models for instance and semantic segmentation. The project takes inspiration from
[`vit_tune`](https://github.com/rexologue/vit_tune) and follows a similar
configuration-first philosophy while targeting MaskDINO use-cases.

## Features

- **YAML-first configuration** for every experiment – from optimizer hyperparameters
  to dataset, logging, and checkpointing options.
- **Dataset processor** that validates Roboflow-style COCO exports with the following
  directory structure:

  ```
  trees_diseases/
  ├── README.dataset.txt
  ├── README.roboflow.txt
  ├── train/
  │   └── _annotations.coco.json
  ├── valid/
  │   └── _annotations.coco.json
  └── test/
      └── _annotations.coco.json
  ```

- **Training engine** for MaskDINO models powered by PyTorch + Hugging Face
  Transformers.
- **Checkpoint management** with epoch-based checkpoints and automatic best-model
  tracking.
- **Experiment tracking with Neptune**, mirroring the ergonomics of the
  `vit_tune` Neptune logger.

## Getting Started

### 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure your experiment

Update `configs/default.yaml` or create a copy for your run. The configuration file
controls dataset paths, optimizer settings, Neptune logging, and checkpoint
behaviour.

### 3. Prepare the dataset metadata

Validate a dataset directory and export label metadata using the dataset processor:

```bash
python scripts/prepare_dataset.py \
  --dataset-dir /path/to/trees_diseases \
  --output-labels configs/labels.yaml
```

The processor checks folder integrity, reports basic statistics, and optionally
writes the inferred `id2label` mapping that can be reused across experiments.

### 4. Launch training

```bash
python train.py --config configs/default.yaml
```

The trainer will download the requested MaskDINO checkpoint into the configured
`models_dir`, start logging to Neptune if enabled, and save checkpoints within
`checkpointing.dir` every `checkpointing.save_every_n_epochs` epochs. The best
model (based on validation loss) is stored under
`checkpointing.dir/best`.

### Repository Layout

```
configs/            # YAML experiment definitions
scripts/            # CLI utilities (dataset processor, evaluation helpers)
detectune/
├── config.py       # Dataclasses + loader for experiment configuration
├── data/           # Dataset utilities and dataloaders
├── engine/         # Training and checkpointing logic
├── logging/        # Neptune experiment logger
├── models/         # MaskDINO model & processor factory
└── utils/          # Helper functions (seeding, distributed helpers)
train.py            # Single-entry training script
```

## Neptune Logging

Set `neptune.enabled: true` inside your config and provide either `api_token` or an
environment variable `NEPTUNE_API_TOKEN`. Additional metadata such as custom tags
or a run name can also be specified.

## License

This project is distributed under the terms of the MIT License. See `LICENSE`
for details.
