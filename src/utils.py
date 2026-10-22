from __future__ import annotations
import numpy as np
import random
import torch
import time
from pathlib import Path
from pydantic import TypeAdapter
from pydantic.dataclasses import dataclass


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.elapsed = time.perf_counter() - self.start
        return False

    @property
    def ms_elasped(self) -> float:
        return round(self.elapsed / 1000, 3)


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    epochs: int
    train_size: int
    test_size: int
    lr: float
    seed: int
    decay: float
    train_logics: list[str]
    test_logics: list[str]
    loss: str
    data_selection: str = "all"
    # Training
    label_smoothing: float = 0.0
    scheduler: str = "cosine"
    warmup_steps: int = 0

    @classmethod
    def from_json(cls, config_path: Path) -> ExperimentConfig:
        assert config_path.is_file(), "Experiment config file required"
        return TypeAdapter(cls).validate_json(config_path.read_text())
