from dataclasses import dataclass
from pathlib import Path
import csv
from typing import Callable
from enum import Enum, auto
from tqdm import tqdm
from random import choices

from transformer import SpecialTacticsTokens, ModelConfig, SRC_PAD_ID
from z3alpha.tactics.catalog import NAME_TO_ID, SOLVER_TACTICS, PREPROCESS_TACTICS, SOLVER_CATALOG
from z3alpha.parser import parse_linear_strategy
from smt_graph import parse_graph

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split


class Status(Enum):
    sat = auto()
    unsat = auto()
    timeout = auto()
    unknown = auto()
    error = auto()


@dataclass
class Benchmark:
    strat: str
    benchmark: Path
    status: Status
    time_s: float
    solved: bool


# TODO rethink, redesign
def encode_smt_file(path: Path, max_graph_size) -> torch.Tensor:
    graph = parse_graph(path, max_nodes=max_graph_size)
    ops = torch.from_numpy(graph.op.astype("int64"))
    return F.pad(ops, (0, max_graph_size - len(ops)), value=SRC_PAD_ID)


# TODO use config values
def encode_strats(contents: str, max_length):
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


class TacticExample(Dataset):

    def __init__(self, benchmarks: list[Benchmark], config: ModelConfig):
        self.smt = [encode_smt_file(p.benchmark, config.max_graph_size) for p in tqdm(benchmarks, desc="Encoding SMT problems")] # TODO a) Discarding too much info b) the benchmarks themselves can be interned
        self.strat = [encode_strats(b.strat, config.max_strat_len) for b in tqdm(benchmarks, desc="Encoding strategies")]
        self.bench_data = benchmarks

    def __len__(self):
        assert len(self.smt) == len(self.strat) == len(self.bench_data)
        return len(self.smt)

    def __getitem__(self, idx):
        return self.smt[idx], self.strat[idx], self.bench_data[idx]

    @staticmethod
    def collate_fn(batch: list[tuple[torch.Tensor, torch.Tensor, Benchmark]]) -> tuple[torch.Tensor, torch.Tensor, list[Benchmark]]:
        smts, strats, bench_data = zip(*batch)
        return torch.stack(smts), torch.stack(strats), list(bench_data)


def make_splits(benchmarks: TacticExample, batch_size=16, validation: float = 0.1, test: float = 0.1) -> dict[str, DataLoader]:
    train_size = 1.0 - validation - test
    train_idx, val_idx, test_idx = random_split(benchmarks, [train_size, validation, test])
    return {
        "train": DataLoader(train_idx, batch_size=batch_size, shuffle=True, pin_memory=True, collate_fn=TacticExample.collate_fn),
        "validation": DataLoader(val_idx, batch_size=batch_size, shuffle=False, pin_memory=True, collate_fn=TacticExample.collate_fn),
        "test": DataLoader(test_idx, batch_size=batch_size, shuffle=False, pin_memory=True, collate_fn=TacticExample.collate_fn)
    }


def load_examples(dirpath: Path) -> list[Benchmark]:
    assert dirpath.is_dir()
    res = []
    for p in dirpath.rglob("*.csv"):
        with open(p, newline="") as fp:
            for row in csv.DictReader(fp):
                res.append(Benchmark(row["strat"], Path(row["benchmark"]), Status[row["status"]], float(row["time_s"]), row["solved"] == "True"))
    return res


def make_loaders(benchmarks_out: Path, count: int, config: ModelConfig) -> dict[str, DataLoader]:
    bench_results = choices(load_examples(benchmarks_out), k=count)
    dataset = TacticExample(bench_results, config)
    return make_splits(dataset)
