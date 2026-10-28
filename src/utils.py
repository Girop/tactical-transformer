from __future__ import annotations

import random
import time
from pathlib import Path

import numpy as np
import torch
from pydantic import TypeAdapter
from pydantic.dataclasses import dataclass

from smt_embed import GIN_DIM
from tactics import CATALOG


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
    gin_model: str = "matched"
    architecture: str = "enc-dec"  # "enc-dec" | "dec" | "llm-lora"
    label_smoothing: float = 0.0
    scheduler: str = "cosine"
    warmup_steps: int = 0
    pretrained_model_name: str = "Qwen/Qwen2.5-0.5B"  # llm-lora only

    @classmethod
    def from_json(cls, config_path: Path) -> ExperimentConfig:
        assert config_path.is_file(), "Experiment config file required"
        return TypeAdapter(cls).validate_json(config_path.read_text())


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    input_dimension: int
    model_dimension: int
    max_strat_len: int
    attention_heads: int
    layers: int
    feedforward: int
    architecture: str = "enc-dec"  # "enc-dec" | "dec" | "llm-lora"
    pretrained_model_name: str = "Qwen/Qwen2.5-0.5B"  # llm-lora only

    @classmethod
    def default(cls, architecture: str, **overrides):
        return cls(
            vocab_size=CATALOG.vocab_size(),
            input_dimension=GIN_DIM,
            model_dimension=768,
            max_strat_len=101,
            attention_heads=8,
            layers=12,
            feedforward=2048,
            architecture=architecture,
            **overrides,
        )

