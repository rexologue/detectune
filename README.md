# Detectune

Tools and reference configuration for fine-tuning MMDetection models on custom datasets that follow the COCO detection format. The repository ships with:

- A ready-to-run Faster R-CNN configuration that plugs in a COCO-style dataset.
- Lightweight training, evaluation, and hyper-parameter sweep scripts powered by MMEngine.
- Documentation for preparing data, customizing experiments, and tracking runs.

The project keeps the moving parts that MMDetection expects (config, work directories, datasets) but trims the setup down so you can focus on iterating on the model.

## Repository structure

```
.
├── configs/
│   └── custom_dataset/
│       └── faster-rcnn_r50_fpn_custom.py   # Base configuration you can extend.
├── scripts/
│   ├── eval.py                             # Evaluate a trained checkpoint.
│   ├── train.py                            # Fine-tune using the provided config.
│   └── tune.py                             # Run a simple grid search over hyper-parameters.
├── tuning/
│   └── search_space.yaml                   # Example hyper-parameter combinations.
├── requirements.txt                        # Python package requirements.
└── README.md
```

You are encouraged to fork the repository and keep your own experiment-specific configs inside `configs/` (e.g. `configs/<project>/<model>.py`).

## Prerequisites

- Python 3.9 or newer.
- PyTorch with CUDA support that matches your GPU drivers. The requirements file assumes the official PyTorch wheels are already installed. If you do not have PyTorch installed yet, follow the instructions at [pytorch.org](https://pytorch.org/) before installing the rest of the dependencies.
- A COCO-style dataset with `train`, `val`, and (optionally) `test` splits.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

The requirements pull the latest stable MMDetection and MMEngine packages. If you prefer to pin exact versions for reproducibility, update `requirements.txt` accordingly.

## Dataset layout

Place or symlink your dataset under a folder of your choice (e.g. `data/my_dataset`). The expected structure mirrors the COCO layout used by MMDetection:

```
/path/to/dataset/
├── annotations/
│   ├── train.json
│   └── val.json
├── train/               # Training images
│   ├── xxx.jpg
│   └── ...
└── val/                 # Validation images
    ├── xxx.jpg
    └── ...
```

The configuration also looks for an optional `test.json`/`test/` pair when running evaluation or inference. Update the `metainfo` section in `configs/custom_dataset/faster-rcnn_r50_fpn_custom.py` so that the `classes` tuple matches your label names and ordering.

## Training

The minimal training command is:

```bash
python scripts/train.py \
  --config configs/custom_dataset/faster-rcnn_r50_fpn_custom.py \
  --data-root /path/to/dataset \
  --work-dir work_dirs/faster_rcnn_baseline
```

- `--config` points to any MMDetection config file.
- `--data-root` overrides `data_root` inside the config so you can keep dataset paths out of version control.
- `--work-dir` is where logs, checkpoints, and tensorboard summaries will be stored (`work_dirs/<config_name>` by default).
- `--cfg-options` allows inline overrides, e.g. `--cfg-options train_dataloader.batch_size=4 optim_wrapper.optimizer.lr=0.01`.

By default, automatic resume is disabled. Pass `--auto-resume` to continue from the latest checkpoint inside the work directory when training restarts.

## Evaluation

After training, evaluate a checkpoint on the validation split:

```bash
python scripts/eval.py \
  --config configs/custom_dataset/faster-rcnn_r50_fpn_custom.py \
  --checkpoint work_dirs/faster_rcnn_baseline/epoch_12.pth \
  --data-root /path/to/dataset
```

The script prints the evaluation metrics to the terminal and saves them as JSON inside the work directory for easy experiment tracking.

## Hyper-parameter sweeps

`tune.py` implements a small grid-search utility that iterates over a YAML-defined search space. Edit `tuning/search_space.yaml` to your liking, then launch:

```bash
python scripts/tune.py \
  --config configs/custom_dataset/faster-rcnn_r50_fpn_custom.py \
  --data-root /path/to/dataset \
  --search-space tuning/search_space.yaml \
  --work-dir-base work_dirs/tuning_runs
```

Each configuration combination gets its own sub-directory under `work_dirs/tuning_runs`. Completed runs are skipped automatically when their checkpoint already exists, letting you resume interrupted sweeps.

## Tips

- Keep separate configs per experiment and inherit from the provided base to ensure consistency.
- Track your GPU memory usage when bumping `batch_size`. Adjust `auto_scale_lr.base_batch_size` inside the config to enable automatic learning rate scaling.
- Use TensorBoard (`tensorboard --logdir work_dirs`) or MMDetection's built-in visualization hooks to monitor training curves.

Happy training!
