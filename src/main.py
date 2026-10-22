from __future__ import annotations
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

from dataloader import Data
from pathlib import Path
from transformer import ModelConfig, build_model
import argparse
from torch import nn
from z3alpha.config import setup_logging
import logging
from train import Trainer
from utils import seed_everything, ExperimentConfig
import torch
from test import test

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


def load_model(config: ModelConfig, name: str, architecture: str = "enc-dec") -> nn.Module:
    model = build_model(config, architecture)
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
    model_config = ModelConfig.default()
    excluded = testing_set(args.excluded)
    log.info("Loading train data...")
    loader = Data(args.data, model_config, gin_model=experiment.gin_model)
    train, val = loader.train_splits(experiment.train_size, 0.15, excluded, experiment.train_logics)
    log.info("Loading test data...")
    test_loader = loader.test_split(experiment.test_size, excluded, experiment.test_logics)
    log.info(f"Loaded samples: {len(train.dataset)} train, {len(val.dataset)} validation, {len(test_loader.dataset)} test")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(model_config, experiment.architecture).to(device)
    if args.compile:
        model = torch.compile(model)
    if not args.skip_train:
        Trainer(model, train, val, experiment).execute()
    model = load_model(model_config, experiment.name, experiment.architecture).to(device)
    test(model, test_loader, args.z3, model_config, experiment)


if __name__ == '__main__':
    main()
