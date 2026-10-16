import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

from dataloader import make_loaders, Status, Benchmark
from pathlib import Path
from transformer import TacticTransformer, ModelConfig
from z3alpha.evaluator import SolverRunner
from tqdm import tqdm
import argparse
from tactics import SpecialTacticsTokens, CATALOG
from smt_embed import GIN_DIM
from dataclasses import dataclass
from utils import seed_everything

from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch

@dataclass
class Metrics:
    loss: float
    par2: float
    solve_rate: float

    @classmethod
    @torch.no_grad()
    def collect(cls, model: TacticTransformer, loader: DataLoader, z3path, timeout) -> Metrics:
        benches = model_strats_on_benchmarks(model, loader, z3path)
        n = len(benches)
        loss = mean_loss(model, loader)
        return cls(loss, sum(par2(b.time_s, b.solved, timeout) for b in benches) / n, sum(b.solved for b in benches) / n)


def model_strats_on_benchmarks(model: TacticTransformer, test: DataLoader, z3path: Path | str, timeout=10.0, tqdm_disable=False) -> list[Benchmark]:
    results = []
    device = model.device
    idx = 0
    for smt, _, bench_data in tqdm(test, desc="Testing", disable=tqdm_disable):
        strats = model.generate(smt.to(device))
        for bench, strat in zip(bench_data, strats):
            strat = CATALOG.tactics_to_text(strat.tolist())
            _, status, runtime, _ = SolverRunner(
                z3path,
                str(bench.benchmark),
                timeout,
                run_id=idx,
                z3_strategy=strat
            ).execute()
            results.append(Benchmark(strat, bench.benchmark, Status[status], float(runtime), status in ("sat", "unsat")))
            idx += 1
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


def train_model(model: TacticTransformer, train: DataLoader, validation: DataLoader, name: str, epochs: int, lr=3e-4):
    Path("models").mkdir(exist_ok=True)
    device = model.device
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for epoch in range(epochs):
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
        print(f"epoch {epoch}: train loss - {train_loss / len(train):.4f}, validation loss - {loss:.4f}") # avg par2 - {metrics.par2:.2f}, solve rate % - {metrics.solve_rate:.2f}")
        torch.save(model.state_dict(), f'models/{name}-epoch-{epoch}.pt')
    torch.save(model.state_dict(), f'models/{name}.pt')


def make_config():
    return ModelConfig(
        vocab_size=CATALOG.vocab_size(),
        model_dimension=GIN_DIM,
        max_strat_len=101,
        attention_heads=8,
        layers=6,
        feedforward=1024,
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

    print("=" * 10 + "\nRun summary, Model vs Z3Alpha:")
    print(f"Solve rate: {metrics.solve_rate:.2f}\t{baseline_sr:.2f}")
    print(f"Avg par2: {metrics.par2:.2f}\t{baseline_par2:.2f}")


def load_model(config: ModelConfig, name: str) -> TacticTransformer:
    model = TacticTransformer(config)
    model.load_state_dict(torch.load(f'models/{name}.pt', weights_only=True))
    model.eval()
    return model


def exclusion_set(excluded: Path) -> set[str]:
    return set(excluded.read_text().splitlines()) if excluded.exists() else set()


def get_args() -> argparse.Namespace:
    arg = argparse.ArgumentParser()
    arg.add_argument('--data', type=Path, default=Path("experiments/train"))
    arg.add_argument('--excluded', type=Path, default=Path("data/excluded.txt"))
    arg.add_argument('--name', type=str, default="dev")
    arg.add_argument('--epochs', type=int, default=10)
    arg.add_argument('--sample-count', type=int, default=None)
    arg.add_argument('--skip-train', action="store_true")
    arg.add_argument('--z3', type=str, default="z3")
    arg.add_argument('--seed', type=int, default=40)
    return arg.parse_args()


def main():
    args = get_args()
    print(f"Training model: {args.name}")
    seed_everything(args.seed)
    config = make_config()
    excluded = exclusion_set(args.excluded)
    print("Loading data...")
    loaders = make_loaders(args.data, config, excluded, args.sample_count)
    train, val, test = loaders["train"], loaders["validation"], loaders["test"]
    print(f"Loaded samples: {len(train.dataset)} train, {len(val.dataset)} validation, {len(test.dataset)} test")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TacticTransformer(config).to(device)
    if not args.skip_train:
        train_model(model, train, val, args.name, args.epochs)
    model = load_model(config, args.name).to(device)
    run_test(model, test, args.z3)


if __name__ == '__main__':
    main()
