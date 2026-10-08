from dataloader import make_loaders, Status, Benchmark
from pathlib import Path
from transformer import TacticTransformer, ModelConfig
from z3alpha.evaluator import SolverRunner
from smt_graph import NUM_OPS
from tqdm import tqdm
import argparse
from tactics import SpecialTacticsTokens, CATALOG

from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch


def train_model(model: TacticTransformer, train: DataLoader, validation: DataLoader, name: str, epochs=20, lr=3e-4):
    Path("models").mkdir(exist_ok=True)
    device = model.device
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    for epoch in tqdm(range(epochs), desc=f"Training"):
        model.train()
        train_loss = 0.0
        for smt, strat, _ in train:
            smt, strat = smt.to(device), strat.to(device)
            logits = model(smt, strat[:, :-1])
            loss = F.cross_entropy(logits.transpose(1, 2), strat[:, 1:], ignore_index=SpecialTacticsTokens.PAD_ID)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_loss += loss.item()
        print(f"epoch {epoch}: train loss - {train_loss / len(train):.4f}, validation loss - {validate(model, validation, device):.4f}")
        torch.save(model.state_dict(), f'models/{name}-epoch-{epoch}.pt')
    torch.save(model.state_dict(), f'models/{name}.pt')
    return model


@torch.no_grad()
def validate(model: TacticTransformer, loader: DataLoader, device) -> float:
    """Mean per-token loss over the loader, padding excluded."""
    model.eval()
    total = tokens = 0
    for smt, strat, _ in loader:
        smt, strat = smt.to(device), strat.to(device)
        logits = model(smt, strat[:, :-1])
        total += F.cross_entropy(logits.transpose(1, 2), strat[:, 1:],
                                 ignore_index=SpecialTacticsTokens.PAD_ID, reduction="sum").item()
        tokens += (strat[:, 1:] != SpecialTacticsTokens.PAD_ID).sum().item()
    return total / tokens


def make_config():
    return ModelConfig(
        graph_tensor_size=NUM_OPS + 1,
        max_graph_size=2_000,
        vocab_size=CATALOG.vocab_size(),
        model_dimension=256,
        max_strat_len=100,
        attention_heads=8,
        layers=6,
        feedforward=1024,
    )



def test_model(model, test, z3path, device, timeout=10.0) -> list[Benchmark]:
    results = []
    idx = 0
    for smt, _, bench_data in tqdm(test, desc="Testing"):
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


def get_baseline(test: DataLoader) -> list[Benchmark]:
    baseline = []
    for _, _, bench_data in test:
        baseline.extend(bench_data)
    return baseline


def summarize(results: list[Benchmark], baseline: list[Benchmark]) -> dict:
    assert len(results) == len(baseline)
    n = len(results)

    return {
        "example_count": n,
        "solve_rate_baseline": sum(b.solved for b in baseline) / n,
        "solve_rate_model": sum(b.solved for b in results) / n,
        "mean_runtime_baseline": sum(b.time_s for b in baseline) / n,
        "mean_runtime_model": sum(b.time_s for b in results) / n,
    }


def get_args() -> argparse.Namespace:
    arg = argparse.ArgumentParser()
    arg.add_argument('--name', type=str, default="tactics-model")
    arg.add_argument('--data', type=Path, required=True)
    arg.add_argument('--sample-count', type=int, default=None)
    arg.add_argument('--only-test', type=bool, action="store_true")
    return arg.parse_args()


def main(args):
    print("Tactic selection transformer")
    config = make_config()
    print("Loading data...")
    loaders = make_loaders(args.data, config, args.sample_count)
    train, val, test = loaders["train"], loaders["validation"], loaders["test"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TacticTransformer(config, device).to(device)
    if args.only_test:
        print("Training")
        model = train_model(model, train, val, args.name)
    print("Testing")
    results = test_model(model, test, z3path="z3", device=device)
    print(summarize(results, get_baseline(test)))


if __name__ == '__main__':
    args = get_args()
    main(args)
