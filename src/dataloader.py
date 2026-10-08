from dataclasses import dataclass
from pathlib import Path
import csv
from enum import Enum, auto
from tqdm import tqdm
from random import choices
from typing import Optional

from transformer import SpecialTacticsTokens, ModelConfig, SRC_PAD_ID
from z3alpha.tactics.catalog import SOLVER_TACTICS, PREPROCESS_TACTICS, NAME_TO_ID
from z3alpha.parser import parse_linear_strategy
from smt_graph import parse_graph
from tactics import CATALOG

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


VALID_TACTIC_NAMES = [*PREPROCESS_TACTICS, *SOLVER_TACTICS]

# TODO rethink, redesign
def encode_smt_file(path: Path, max_graph_size) -> torch.Tensor:
    graph = parse_graph(path, max_nodes=max_graph_size)
    ops = torch.from_numpy(graph.op.astype("int64"))
    return F.pad(ops, (0, max_graph_size - len(ops)), value=SRC_PAD_ID)


def encode_strats(contents: str, max_length):
    strats = [
        CATALOG.name_to_id(name) for (name, _params)
        in parse_linear_strategy(contents)
        if name in VALID_TACTIC_NAMES
    ]
    strats = [SpecialTacticsTokens.BOS_ID.value, *strats, SpecialTacticsTokens.EOS_ID.value]
    repr = torch.tensor(strats, dtype=torch.long)
    assert len(strats) <= max_length
    return F.pad(repr, (0, max_length - len(strats)), value=SpecialTacticsTokens.PAD_ID.value)


class TacticExample(Dataset):

    def __init__(self, benchmarks: list[Benchmark], config: ModelConfig):
        encoded = {p: encode_smt_file(p, config.max_graph_size) for p in tqdm({b.benchmark for b in benchmarks}, desc="Encoding SMT problems")} # TODO Discarding too much info
        self.smt = [encoded[b.benchmark] for b in benchmarks]
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


def make_loaders(benchmarks_out: Path, config: ModelConfig, count: Optional[int] = None) -> dict[str, DataLoader]:
    bench_results = load_examples(benchmarks_out)
    if count is not None:
        bench_results = choices(bench_results, k=count)
    dataset = TacticExample(bench_results, config)
    return make_splits(dataset)
