import json
import os
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Mapping, MutableMapping, NamedTuple

import jax
import wandb
from llmrl.config import Config
from llmrl.experiment import Experiment
from rich.console import Console
from rich.live import Live
from rich.table import Table
from tensorboardX import SummaryWriter

Metrics = dict[str, jax.Array | float | int]


def json_normalize(data: dict, sep: str = ".") -> dict:
    out = {}

    def flatten(x, name=""):
        if isinstance(x, dict):
            for a in x:
                flatten(x[a], name + a + sep)
        else:
            if isinstance(x, jax.Array):
                x = x.item()

            out[name[:-len(sep)]] = x

    flatten(data)
    return out


class BaseLogger(ABC):
    @abstractmethod
    def __init__(self, unique_token: str):
        pass

    def log_dict(self, data: Metrics, step: int) -> None:
        pass

    def start(self) -> None:
        pass

    def close(self) -> None:
        pass


class MultiLogger(BaseLogger):
    def __init__(self, loggers: list[BaseLogger]) -> None:
        self.loggers = loggers

    def log_dict(self, data: dict, step: int) -> None:
        for logger in self.loggers:
            logger.log_dict(data, step)

    def start(self):
        for logger in self.loggers:
            logger.start()

    def close(self) -> None:
        for logger in self.loggers:
            logger.close()


class TensorboardLogger(BaseLogger):
    def __init__(self, unique_token: str) -> None:
        log_path = Path("./logs/tensorboard") / unique_token
        os.makedirs(log_path, exist_ok=True)
        self.writer = SummaryWriter(log_path.as_posix())

    def log_dict(self, data: Metrics, step: int) -> None:
        data = json_normalize(data, sep="/")

        for key, value in data.items():
            self.writer.add_scalar(key, value, step)

    def close(self) -> None:
        self.writer.close()


class ConsoleLogger(BaseLogger):
    def __init__(self, unique_token: str, console: Console) -> None:
        self._live = Live(console=console)

    def start(self):
        self._live.start()

    def close(self):
        self._live.stop()

    def log_dict(self, data: Metrics, step: int) -> None:
        data = json_normalize(data, sep=".")

        keys = sorted(data.keys())
        values = []
        for key in keys:
            value = data[key]
            if hasattr(value, "item"):
                 value = value.item()
            values.append(f"{value:.6f}" if isinstance(value, float) else value)

        table = Table()
        table.add_column("Key")
        table.add_column("Value")
        for key, value in zip(keys, values):
            table.add_row(key, str(value))

        table.add_row("Step", str(step))
        self._live.update(table)


class JsonLogger(BaseLogger):
    def __init__(self, experiment_path: str) -> None:
        self._file = None
        self._experiment_path = f"{experiment_path}/logs.jsonl"

    def start(self):
        self._file = open(f"{self._experiment_path}.jsonl", "w")

    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None

    def log_dict(self, data: Metrics, step: int) -> None:
        if self._file is not None:
            self._file.write(json.dumps(data) + "\n")
            self._file.flush()


class WandbLogger(BaseLogger):
    def __init__(self, unique_token: str, settings: Config):
        wandb.init(project=settings.logger.project_name, name=unique_token, config=settings.model_dump())

    def log_dict(self, data: Metrics, step: int) -> None:
        normalized_data = json_normalize(data)

        wandb.log(normalized_data, step=step)

    def close(self) -> None:
        wandb.finish()


def create_logger(experiment: Experiment, console: Console) -> BaseLogger:
    logger_config = experiment.config.logger
    loggers: list[BaseLogger] = []

    if logger_config.use_tb:
        loggers.append(TensorboardLogger(experiment.unique_token))
    if logger_config.use_console:
        loggers.append(ConsoleLogger(experiment.unique_token, console))
    if logger_config.use_wandb:
        loggers.append(WandbLogger(experiment.unique_token, experiment.config))
    if logger_config.use_jsonl:
        loggers.append(JsonLogger(experiment.root))

    return MultiLogger(loggers)


class MetricAccum(NamedTuple):
    total: int | float
    count: int


def _accum_merge(dst: MutableMapping[str, Any], update: Mapping[str, Any]) -> None:
    for key, value in update.items():
        if isinstance(value, dict):
            child = dst.setdefault(key, {})
            _accum_merge(dst[key], value)
        else:
            if hasattr(value, 'item'):
                value = value.item()

            acc = dst.get(key)
            dst[key] = (
                MetricAccum(value, 1) if acc is None else MetricAccum(acc.total + value, acc.count + 1)
            )


def _tree_map(tree: Any, fn: Callable[[Any], Any]) -> Any:
    if isinstance(tree, Mapping):
        return {key: _tree_map(value, fn) for key, value in tree.items()}
    return fn(tree)


class MetricsAccumulator:
    def __init__(self, logger: BaseLogger):
        self._metrics = {}
        self._counts = {}
        self._logger = logger
        self._last_metrics = None
        self._last_counts = None

    def add(self, metrics: Metrics):
        if self._last_metrics is not None:
            _accum_merge(self._metrics, self._last_metrics)
        self._last_metrics = metrics

    def add_counts(self, counts: Metrics):
        if self._last_counts is not None:
            _accum_merge(self._counts, self._last_counts)
        self._last_counts = counts

    def flush(self, step: int):
        current_time = time.perf_counter()
        delta_time = current_time - self._last_flush_time
        # self._last_flush_time = current_time

        self._metrics = _tree_map(self._metrics, lambda x: x.total / x.count)
        self._counts = _tree_map(self._counts, lambda x: x.total / delta_time)
        self._logger.log_dict(self._metrics | self._counts, step)
        self._metrics.clear()
        self._counts.clear()

    def start(self):
        self._last_flush_time = time.perf_counter()
        self._logger.start()

    def close(self):
        self._logger.close()
