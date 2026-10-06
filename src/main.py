from dataclasses import dataclass
from dataloader import make_loaders, Status
from pathlib import Path
from transformer import TacticTransformer, SpecialTacticsTokens, ModelConfig, SRC_PAD_ID
from z3alpha.parser import parse_linear_strategy
from z3alpha.tactics.catalog import NAME_TO_ID, SOLVER_TACTICS, PREPROCESS_TACTICS, SOLVER_CATALOG, PREPROCESS_CATALOG, tactic_name_for_action
from z3alpha.evaluator import SolverRunner
from smt_graph import parse_graph, NUM_OPS
from tqdm import tqdm

from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch


# TODO rethink, redesign
def encode_smt_file(path: Path, max_graph_size=2_000) -> torch.Tensor:
    graph = parse_graph(path, max_nodes=max_graph_size)
    ops = torch.from_numpy(graph.op.astype("int64"))
    return F.pad(ops, (0, max_graph_size - len(ops)), value=SRC_PAD_ID)


# TODO use config values
def encode_strats(contents: str, max_length=100):
    strats = [
        NAME_TO_ID[name] for (name, _params)
        in parse_linear_strategy(contents)
        if name in PREPROCESS_TACTICS or name in SOLVER_TACTICS
    ]
    assert strats[-1] in SOLVER_CATALOG.keys()
    strats = [SpecialTacticsTokens.BOS_ID.value, *strats]
    repr = torch.tensor(strats, dtype=torch.int32)
    assert len(strats) < max_length
    return F.pad(repr, (0, max_length - len(strats)), value=SpecialTacticsTokens.PAD_ID.value)


def train_model(model: TacticTransformer, train: DataLoader, validation: DataLoader):
    return model


def make_config():
    return ModelConfig(
        graph_tensor_size=NUM_OPS + 1,
        max_graph_size=2_000,
        vocab_size=SpecialTacticsTokens.vocab_size(),
        model_dimension=256,
        max_strat_len=100,
        attention_heads=8,
        layers=6,
        feedforward=1024,
    )


@dataclass
class Comparison:
    benchmark: Path
    baseline_strat: str
    baseline_status: Status
    baseline_time_s: float
    baseline_solved: bool
    model_strat: str
    model_status: Status
    model_time_s: float
    model_solved: bool


def test_model(model, test, z3path, device, timeout=10.0) -> list[Comparison]:
    results = []
    idx = 0
    for smt, _, bench_data in tqdm(test, desc="Testing"):
        strats = model.generate(smt.to(device))
        for bench, strat in zip(bench_data, strats):
            ids = strat.tolist()[1:]  # drop BOS
            if SpecialTacticsTokens.EOS_ID.value in ids:
                ids = ids[:ids.index(SpecialTacticsTokens.EOS_ID.value)]
            names = [tactic_name_for_action(i) for i in ids if i in SOLVER_CATALOG or i in PREPROCESS_CATALOG]
            strat = f"(then {' '.join(names)})" if len(names) > 1 else (names[0] if names else "smt")

            _, status, runtime, _ = SolverRunner(
                z3path,
                str(bench.benchmark),
                timeout,
                run_id=idx,
                z3_strategy=strat
            ).execute()
            results.append(Comparison(
                benchmark=bench.benchmark,
                baseline_strat=bench.strat,
                baseline_status=bench.status,
                baseline_time_s=bench.time_s,
                baseline_solved=bench.solved,
                model_strat=strat,
                model_status=Status[status],
                model_time_s=runtime,
                model_solved=status in ("sat", "unsat"),
            ))
            idx += 1
    return results

# TODO get something more interpretable
def summarize(results: list[Comparison]) -> dict:
    n = len(results)
    both_solved = [r for r in results if r.baseline_solved and r.model_solved]
    return {
        "n": n,
        "baseline_solve_rate": sum(r.baseline_solved for r in results) / n,
        "model_solve_rate": sum(r.model_solved for r in results) / n,
        "both_solved": len(both_solved),
        "model_only_solved": sum(r.model_solved and not r.baseline_solved for r in results),
        "baseline_only_solved": sum(r.baseline_solved and not r.model_solved for r in results),
        "neither_solved": sum(not r.baseline_solved and not r.model_solved for r in results),
        "status_agreement_rate": sum(r.baseline_status == r.model_status for r in results) / n,
        "strategy_exact_match_rate": sum(r.baseline_strat == r.model_strat for r in results) / n,
        "mean_runtime_ratio_model_over_baseline": (
            sum(r.model_time_s / r.baseline_time_s for r in both_solved) / len(both_solved)
            if both_solved else float("nan")
        ),
        "model_faster_count": sum(r.model_time_s < r.baseline_time_s for r in both_solved),
    }


if __name__ == '__main__':
    p = Path("experiments/labels/qfnia_sample2k")
    loaders = make_loaders(p, 20, encode_smt_file, encode_strats)
    train, val, test = loaders["train"], loaders["validation"], loaders["test"]

    device = torch.device('cuda')
    config = make_config()
    model = TacticTransformer(config, device).to(device)
    model = train_model(model, train, val)
    results = test_model(model, test, z3path="z3", device=device)
    print(summarize(results))

