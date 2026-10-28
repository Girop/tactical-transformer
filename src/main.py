from __future__ import annotations

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import argparse
import logging
from pathlib import Path
from test import test

import torch
from torch import nn

from dataloader import Data
from train import Trainer
from transformer import ModelConfig, build_model
from utils import ExperimentConfig, seed_everything
from z3alpha.config import setup_logging

log = logging.getLogger(__name__)


def get_args() -> argparse.Namespace:
    arg = argparse.ArgumentParser()
    arg.add_argument('--data', type=Path, default=Path("experiments/train/all.csv"))
    arg.add_argument('--excluded', type=Path, default=Path("data/excluded.txt"))
    arg.add_argument('--experiment', type=Path, default=Path("experiment.json"))
    arg.add_argument('--skip-train', action="store_true")
    arg.add_argument('--z3', type=str, default="z3")
    arg.add_argument('--compile', action="store_true", help="torch.compile the model")
    return arg.parse_args()


def load_model(config: ModelConfig, name: str) -> nn.Module:
    model = build_model(config)
    model.load_state_dict(torch.load(f'models/{name}.pt', weights_only=True))
    model.eval()
    return model


def testing_set(excluded: Path) -> set[str]:
    return set(excluded.read_text().splitlines()) if excluded.exists() else set()


def main():
    setup_logging()
    args = get_args()
    experiment = ExperimentConfig.from_json(args.experiment)
    log.info(f"Training model: {experiment.name}")
    seed_everything(experiment.seed)
    model_config = ModelConfig.default(
        architecture=experiment.architecture,
        pretrained_model_name=experiment.pretrained_model_name,
    )
    excluded = testing_set(args.excluded)
    log.info("Loading train data...")
    loader = Data(args.data, model_config, gin_model=experiment.gin_model)
    train, val = loader.train_splits(experiment.train_size, 0.15, excluded, experiment.train_logics)
    log.info("Loading test data...")
    test_loader = loader.test_split(experiment.test_size, excluded, experiment.test_logics)
    log.info(f"Loaded samples: {len(train.dataset)} train, {len(val.dataset)} validation, {len(test_loader.dataset)} test")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(model_config).to(device)
    if args.compile:
        model = torch.compile(model)
    if not args.skip_train:
        Trainer(model, train, val, experiment).execute()
    model = load_model(model_config, experiment.name).to(device)
    test(model, test_loader, args.z3, model_config, experiment)


if __name__ == '__main__':
    main()
