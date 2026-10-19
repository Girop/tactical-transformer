import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

from dataloader import Data, Status, Benchmark
from pathlib import Path
from transformer import TacticTransformer, ModelConfig
from z3alpha.evaluator import SolverRunner
from tqdm import tqdm
import argparse
from tactics import SpecialTacticsTokens, CATALOG
from smt_embed import GIN_DIM
from dataclasses import dataclass
from pydantic import BaseModel
from z3alpha.config import setup_logging
import logging

log = logging.getLogger(__name__)
from utils import seed_everything
import json

from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch


def get_args() -> argparse.Namespace:
    arg = argparse.ArgumentParser()
    arg.add_argument('--data', type=Path, default=Path("experiments/train/all.csv"))
    arg.add_argument('--excluded', type=Path, default=Path("data/excluded.txt"))
    arg.add_argument('--experiment', type=Path, default=Path("experiment.json"))
    arg.add_argument('--skip-train', action="store_true")
    arg.add_argument('--z3', type=str, default="z3")
    arg.add_argument('--compile', action="store_true", help="torch.compile the model")
    return arg.parse_args()


@dataclass
class Metrics:
    loss: float
    par2: float
    solve_rate: float

    @classmethod
    @torch.no_grad()
    def collect(cls, model: TacticTransformer, loader: DataLoader, z3path: Path | str, timeout: float) -> Metrics:
        strat_bench = generate_strategies(model, loader)
        benches = run_solver_on_benchmarks(strat_bench, z3path, timeout)
        n = len(benches)
        loss = mean_loss(model, loader)
        return cls(loss, sum(par2(b.time_s, b.solved, timeout) for b in benches) / n, sum(b.solved for b in benches) / n)


class ExperimentConfig(BaseModel, frozen=True):
    name: str
    epochs: int
    train_size: int
    test_size: int
    lr: float
    seed: int
    decay: float

    @classmethod
    def from_json(cls, config_path: Path) -> ExperimentConfig:
        assert config_path.is_file(), "Experiment config file required"
        return cls.model_validate_json(config_path.read_text())


def generate_strategies(model: TacticTransformer, data: DataLoader) -> list[tuple[Path, str]]:
    device = model.device
    strats = []
    for smt, _, bench in tqdm(data, desc="Generating"):
        ids = model.generate(smt.to(device)).tolist()
        strategies = [CATALOG.tactics_to_text(id_) for id_ in ids]
        strats.extend([(b.benchmark, s) for b, s in zip(bench, strategies)])
    return strats


# Cache by: (benchmark, strategy, timeout), values: (status, runtime)
def run_solver_on_benchmarks(benchmarks: list[tuple[Path, str]], z3path: Path | str, timeout=10.0, cache: Path = Path("data/benchmark_cache")) -> list[Benchmark]:
    cache.mkdir(exist_ok=True)
    cache_file = cache / "results.json"
    store: dict = json.loads(cache_file.read_text()) if cache_file.exists() else {}

    results = []
    for idx, (bench_path, strategy) in enumerate(tqdm(benchmarks, desc="Testing")):
        key = f"{bench_path}|{strategy}|{timeout}"
        if key in store:
            status, runtime = store[key]["status"], store[key]["runtime"]
        else:
            _, status, runtime, _ = SolverRunner(
                z3path,
                str(bench_path),
                timeout,
                run_id=idx,
                z3_strategy=strategy
            ).execute()
            store[key] = {"status": status, "runtime": runtime}
            cache_file.write_text(json.dumps(store, indent=2))
        results.append(Benchmark(strategy, bench_path, Status[status], float(runtime), status in ("sat", "unsat")))
    return results


@torch.no_grad()
def mean_loss(model: TacticTransformer, loader: DataLoader) -> float:
    model.eval()
    total = 0.0
    count = 0
    for smt, strat, _ in loader:
        smt, strat = smt.to(model.device), strat.to(model.device)
        logits = model(smt, strat[:, :-1])
        bs = smt.size(0)
        total += F.cross_entropy(logits.transpose(1, 2), strat[:, 1:],
                                 ignore_index=SpecialTacticsTokens.PAD_ID).item() * bs
        count += bs
    return total / count


def train_model(model: TacticTransformer, train: DataLoader, validation: DataLoader, config: ExperimentConfig):
    Path("models").mkdir(exist_ok=True)
    device = model.device
    opt = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=config.epochs)
    for epoch in range(config.epochs):
        model.train()
        train_loss = 0.0
        for smt, strat, _ in tqdm(train, desc=f"Epoch {epoch}"):
            opt.zero_grad()
            smt, strat = smt.to(device), strat.to(device)
            logits = model(smt, strat[:, :-1])
            loss = F.cross_entropy(logits.transpose(1, 2), strat[:, 1:], ignore_index=SpecialTacticsTokens.PAD_ID)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_loss += loss.item()

        loss = mean_loss(model, validation)
        scheduler.step()
        log.info(f"Epoch {epoch}: train loss - {train_loss / len(train):.4f}, validation loss - {loss:.4f}, lr - {scheduler.get_last_lr()[0]:.6f}")
        torch.save(model.state_dict(), f'models/{config.name}-epoch-{epoch}.pt')
    torch.save(model.state_dict(), f'models/{config.name}.pt')


def make_config():
    return ModelConfig(
        vocab_size=CATALOG.vocab_size(),
        input_dimension=GIN_DIM,
        model_dimension=1024,
        max_strat_len=101,
        attention_heads=16,
        layers=24,
        feedforward=2048,
    )


def get_baseline(test: DataLoader) -> list[Benchmark]:
    baseline = []
    for _, _, bench_data in test:
        baseline.extend(bench_data)
    return baseline


def par2(time_s: float, solved: bool, timeout: float) -> float:
    return time_s if solved else timeout * 2


def run_test(model: TacticTransformer, test: DataLoader, z3path: Path | str, timeout: float = 10.0):
    metrics = Metrics.collect(model, test, z3path, timeout)

    baseline = get_baseline(test)
    baseline_sr = sum(b.solved for b in baseline) / len(baseline)
    baseline_par2 = sum(par2(b.time_s, b.solved, timeout) for b in baseline) / len(baseline)

    # TODO This is not a fair comparison, I'm using supbar solutions for smt problems (transformer is data hungry),
    # training data is not good representative of peak capabilities of z3alpha.
    # TODO additionally to that, I should compare average scores from multiple seeds
    log.info(f"\n{'─' * 44}")
    log.info(f"  {'':20} {'Model':>5}  {'Z3Alpha':>7}")
    log.info(f"{'─' * 44}")
    log.info(f"  {'Solve rate':20} {metrics.solve_rate:>5.2f}  {baseline_sr:>7.2f}")
    log.info(f"  {'Avg PAR-2':20} {metrics.par2:>5.2f}  {baseline_par2:>7.2f}")
    log.info(f"{'─' * 44}")


def load_model(config: ModelConfig, name: str) -> TacticTransformer:
    model = TacticTransformer(config)
    model.load_state_dict(torch.load(f'models/{name}.pt', weights_only=True))
    model.eval()
    return model


def exclusion_set(excluded: Path) -> set[str]:
    return set(excluded.read_text().splitlines()) if excluded.exists() else set()


def main():
    setup_logging()
    args = get_args()
    experiment = ExperimentConfig.from_json(args.experiment)
    log.info(f"Training model: {experiment.name}")
    seed_everything(experiment.seed)
    model_config = make_config()
    excluded = exclusion_set(args.excluded)
    log.info("Loading train data...")
    loader = Data(args.data, model_config)
    train, val = loader.train_splits(experiment.train_size, 0.15, excluded)
    log.info("Loading test data...")
    test = loader.test_split(experiment.test_size, excluded)
    log.info(f"Loaded samples: {len(train.dataset)} train, {len(val.dataset)} validation, {len(test.dataset)} test")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TacticTransformer(model_config).to(device)
    if args.compile:
        model = torch.compile(model)
    if not args.skip_train:
        train_model(model, train, val, experiment)
    model = load_model(model_config, args.name).to(device)
    run_test(model, test, args.z3)


if __name__ == '__main__':
    main()
