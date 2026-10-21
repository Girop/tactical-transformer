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
from pydantic import TypeAdapter
from pydantic.dataclasses import dataclass
from dataclasses import asdict
from collections import defaultdict
from z3alpha.config import setup_logging
import logging
from math import inf

log = logging.getLogger(__name__)
from utils import seed_everything
import json

from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch
import torch.nn as nn


def get_args() -> argparse.Namespace:
    arg = argparse.ArgumentParser()
    arg.add_argument('--data', type=Path, default=Path("experiments/train/all.csv"))
    arg.add_argument('--excluded', type=Path, default=Path("data/excluded.txt"))
    arg.add_argument('--experiment', type=Path, default=Path("experiment.json"))
    arg.add_argument('--skip-train', action="store_true")
    arg.add_argument('--z3', type=str, default="z3")
    arg.add_argument('--compile', action="store_true", help="torch.compile the model")
    return arg.parse_args()


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

    @classmethod
    def from_json(cls, config_path: Path) -> ExperimentConfig:
        assert config_path.is_file(), "Experiment config file required"
        return TypeAdapter(cls).validate_json(config_path.read_text())


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


class Par2Loss(nn.Module):
    def __init__(self, timeout: float, pad_id: int):
        super().__init__()
        self.timeout = timeout
        self.ce = nn.CrossEntropyLoss(ignore_index=pad_id, reduction='none')

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, benchmarks: list) -> torch.Tensor:
        per_sample = self.ce(logits.transpose(1, 2), targets).mean(dim=1)
        weights = torch.tensor([1.0 - b.time_s / self.timeout if b.solved else 0.0 for b in benchmarks], device=logits.device)
        return (per_sample * weights).mean()


class CELoss(nn.Module):
    def __init__(self, pad_id: int):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=pad_id)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, benchmarks: list) -> torch.Tensor:
        return self.ce(logits.transpose(1, 2), targets)


LOSS_FUNCTIONS = {
    "par2": lambda timeout, pad_id: Par2Loss(timeout, pad_id),
    "ce": lambda timeout, pad_id: CELoss(pad_id),
}


def train_model(model: TacticTransformer, train: DataLoader, validation: DataLoader, config: ExperimentConfig, timeout: float = 10.0):
    Path("models").mkdir(exist_ok=True)
    device = model.device
    opt = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=config.epochs)
    assert config.loss in LOSS_FUNCTIONS, f"Unknown loss: {config.loss}, expected one of {list(LOSS_FUNCTIONS)}"
    criterion = LOSS_FUNCTIONS[config.loss](timeout, SpecialTacticsTokens.PAD_ID)
    min_loss = inf
    for epoch in range(config.epochs):
        model.train()
        train_loss = 0.0
        for smt, strat, bench in tqdm(train, desc=f"Epoch {epoch}"):
            opt.zero_grad()
            smt, strat = smt.to(device), strat.to(device)
            logits = model(smt, strat[:, :-1])
            loss = criterion(logits, strat[:, 1:], bench)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_loss += loss.item()

        val_loss = mean_loss(model, validation)
        scheduler.step()
        log.info(f"Epoch {epoch}: train loss - {train_loss / len(train):.4f}, validation loss - {val_loss:.4f}, lr - {scheduler.get_last_lr()[0]:.6f}")
        if val_loss < min_loss: # TODO nice metric would come in handy...
            torch.save(model.state_dict(), f'models/epoch-{epoch}-{config.name}.pt')
            min_loss = val_loss
    torch.save(model.state_dict(), f'models/{config.name}.pt')


def make_config():
    return ModelConfig(
        vocab_size=CATALOG.vocab_size(),
        input_dimension=GIN_DIM,
        model_dimension=768,
        max_strat_len=101,
        attention_heads=8,
        layers=12,
        feedforward=2048,
    )


def par2(time_s: float, solved: bool, timeout: float) -> float:
    return time_s if solved else timeout * 2


@dataclass(frozen=True)
class LogicStats:
    logic: str
    model_sr: float
    baseline_sr: float
    model_par2: float
    baseline_par2: float
    n: int

    @classmethod
    def from_benchmarks(cls, model_benchmarks: list[Benchmark], base_benchmarks: list[Benchmark], logic: str, timeout: float) -> LogicStats:
        def sr_par2(benchmarks: list[Benchmark]):
            n = max(len(benchmarks), 1)
            return sum(b.solved for b in benchmarks) / n, sum(par2(b.time_s, b.solved, timeout) for b in benchmarks) / n
        m_sr, m_p2 = sr_par2(model_benchmarks)
        b_sr, b_p2 = sr_par2(base_benchmarks)
        return cls(logic=logic, model_sr=m_sr, baseline_sr=b_sr, model_par2=m_p2, baseline_par2=b_p2, n=len(model_benchmarks))


@dataclass(frozen=True)
class TestResults:
    overall: LogicStats
    per_logic: list[LogicStats]

    @classmethod
    def from_benchmarks(cls, model_results: list[Benchmark], baseline: list[Benchmark], timeout: float = 10.0) -> TestResults:
        overall = LogicStats.from_benchmarks(model_results, baseline, "Overall", timeout)

        model_by_logic: dict[str, list[Benchmark]] = defaultdict(list)
        baseline_by_logic: dict[str, list[Benchmark]] = defaultdict(list)
        for b in model_results:
            model_by_logic[b.logic].append(b)
        for b in baseline:
            baseline_by_logic[b.logic].append(b)

        per_logic = [
            LogicStats.from_benchmarks(model_by_logic.get(logic, []), baseline_by_logic.get(logic, []), logic, timeout)
            for logic in sorted(set(model_by_logic) | set(baseline_by_logic))
        ]
        return cls(overall=overall, per_logic=per_logic)

    def display(self):
        rows = [self.overall, *self.per_logic]
        w = max((len(r.logic) for r in rows), default=10) + 2
        header = f"{'─' * (w + 42)}"
        log.info(f"\n{header}")
        log.info(f"  {'':>{w}} {'Model SR':>9} {'Model P2':>9} {'Base SR':>9} {'Base P2':>9}  {'n':>4}")
        log.info(header)
        for r in rows:
            log.info(f"  {r.logic:>{w}} {r.model_sr:>9.2f} {r.model_par2:>9.2f} {r.baseline_sr:>9.2f} {r.baseline_par2:>9.2f}  {r.n:>4}")
        log.info(header)


    def to_json(self, path: Path, model: ModelConfig, experiment: ExperimentConfig):
        data = {
            "model": str(model),
            "experiment": str(experiment),
            "overall": asdict(self.overall),
            "per_logic": [asdict(r) for r in self.per_logic],
        }
        path.write_text(json.dumps(data, indent=2))


def run_test(model: TacticTransformer, test: DataLoader, z3path: Path | str, timeout: float = 10.0) -> TestResults:
    baseline = test.dataset.bench_data
    assert len(baseline) == len(set(b.benchmark for b in baseline))
    strat_bench = generate_strategies(model, test)
    model_results = run_solver_on_benchmarks(strat_bench, z3path, timeout)
    return TestResults.from_benchmarks(model_results, baseline, timeout)


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
    train, val = loader.train_splits(experiment.train_size, 0.15, excluded, experiment.train_logics)
    log.info("Loading test data...")
    test = loader.test_split(experiment.test_size, excluded, experiment.test_logics)
    log.info(f"Loaded samples: {len(train.dataset)} train, {len(val.dataset)} validation, {len(test.dataset)} test")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TacticTransformer(model_config).to(device)
    if args.compile:
        # TODO Doesn't work
        model = torch.compile(model)
    if not args.skip_train:
        train_model(model, train, val, experiment)
    model = load_model(model_config, experiment.name).to(device)
    results = run_test(model, test, args.z3)
    results.display()
    results.to_json(Path(f"experiment-{experiment.name}-out.json"), model_config, experiment)


if __name__ == '__main__':
    main()
