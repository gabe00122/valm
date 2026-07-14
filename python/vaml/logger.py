import json
import os
from abc import ABC, abstractmethod
from pathlib import Path

import jax
import wandb
from rich.console import Console
from rich.live import Live
from rich.table import Table
from tensorboardX import SummaryWriter
from vaml.config import Config
from vaml.experiment import Experiment

Metrics = dict[str, jax.Array | float | int]


def json_normalize(data: dict, sep: str = ".") -> dict:
    out = {}

    def flatten(x, name=""):
        if isinstance(x, dict):
            for a in x:
                flatten(x[a], name + a + sep)
        else:
            if hasattr(x, "item"):
                x = x.item()

            out[name[: -len(sep)]] = x

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
        self._experiment_file_path = f"{experiment_path}/logs.jsonl"

    def start(self):
        self._file = open(self._experiment_file_path, "w")

    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None

    def log_dict(self, data: Metrics, step: int) -> None:
        if self._file is not None:
            # this really probably shouldn't be flattened, we need to extract the logic that converts jax arrays to floats from the flattening logic
            self._file.write(json.dumps(json_normalize(data)) + "\n")
            self._file.flush()


class WandbLogger(BaseLogger):
    def __init__(
        self, unique_token: str, settings: Config, tags: list[str] | None = None
    ):
        wandb.init(
            project=settings.logger.project_name,
            name=unique_token,
            config={"hypers": settings.model_dump()},
            tags=tags,
        )

    def log_dict(self, data: Metrics, step: int) -> None:
        normalized_data = json_normalize(data)

        wandb.log(normalized_data, step=step)

    def close(self) -> None:
        wandb.finish()


def create_logger(
    experiment: Experiment,
    console: Console,
    wandb_tags: list[str] | None = None,
) -> BaseLogger:
    logger_config = experiment.config.logger
    loggers: list[BaseLogger] = []

    if logger_config.use_tb:
        loggers.append(TensorboardLogger(experiment.unique_token))
    if logger_config.use_console:
        loggers.append(ConsoleLogger(experiment.unique_token, console))
    if logger_config.use_wandb:
        loggers.append(
            WandbLogger(experiment.unique_token, experiment.config, wandb_tags)
        )
    if logger_config.use_jsonl:
        loggers.append(JsonLogger(experiment.root))

    return MultiLogger(loggers)
