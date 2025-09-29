"""Neptune experiment logger following the vit_tune ergonomics."""

from __future__ import annotations

import os
from typing import Any, Dict, Iterable, Optional

import neptune
from neptune.utils import stringify_unsupported

try:  # pragma: no cover - optional dependency
    from dotenv import load_dotenv
    USE_DOT_ENV = True
except ImportError:  # pragma: no cover - optional dependency
    USE_DOT_ENV = False


class NullLogger:
    """Fallback logger that performs no operations."""

    def log_hyperparameters(self, params: Dict[str, Any]):  # noqa: D401 - documented in interface
        pass

    def save_metrics(self, *args, **kwargs):  # noqa: D401 - documented in interface
        pass

    def save_plot(self, *args, **kwargs):  # noqa: D401 - documented in interface
        pass

    def add_tag(self, *args, **kwargs):  # noqa: D401 - documented in interface
        pass

    def stop(self):  # noqa: D401 - documented in interface
        pass


class NeptuneLogger:
    """Wrapper around :mod:`neptune` mirroring vit_tune behaviour."""

    def __init__(self, config) -> None:
        env_path = getattr(config, "env_path", None)
        if USE_DOT_ENV and env_path:
            load_dotenv(env_path)

        api_token = getattr(config, "api_token", None) or os.environ.get("NEPTUNE_API_TOKEN")
        if api_token:
            os.environ.setdefault("NEPTUNE_API_TOKEN", api_token)

        run_kwargs = {
            "project": getattr(config, "project", None),
            "api_token": os.environ.get("NEPTUNE_API_TOKEN"),
            "name": getattr(config, "experiment_name", None),
            "dependencies": getattr(config, "dependencies_path", None),
            "with_id": getattr(config, "run_id", None),
            "tags": getattr(config, "tags", None),
        }
        run_kwargs = {k: v for k, v in run_kwargs.items() if v is not None}
        self.run = neptune.init_run(**run_kwargs)

    def log_hyperparameters(self, params: Dict[str, Any]):
        self.run["hyperparameters"] = stringify_unsupported(params)

    def save_metrics(
        self,
        type_set: str,
        metric_name,
        metric_value,
        step: Optional[int] = None,
    ) -> None:
        if isinstance(metric_name, Iterable) and not isinstance(metric_name, (str, bytes)):
            for name, value in zip(metric_name, metric_value):
                self.run[f"{type_set}/{name}"].log(value, step=step)
        else:
            self.run[f"{type_set}/{metric_name}"].log(metric_value, step=step)

    def save_plot(self, type_set: str, plot_name: str, plt_fig) -> None:
        self.run[f"{type_set}/{plot_name}"].append(plt_fig)

    def add_tag(self, tag: str) -> None:
        self.run["sys/tags"].add(tag)

    def stop(self) -> None:
        self.run.stop()
