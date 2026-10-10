from dataclasses import dataclass
from pathlib import Path
import csv
import warnings
from enum import Enum, auto
from random import sample
from typing import Optional

from transformer import SpecialTacticsTokens, ModelConfig
from z3alpha.parser import parse_linear_strategy
from smt_embed import embed_benchmarks
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


def encode_strats(contents: str, max_length: int):
    strats = [
        CATALOG.name_to_id(name) for (name, _params)
        in parse_linear_strategy(contents)
        if name in CATALOG.valid_tactic_names
    ]
    strats = [SpecialTacticsTokens.BOS_ID.value, *strats, SpecialTacticsTokens.EOS_ID.value]
    if len(strats) > max_length:
        warnings.warn(f"Strategy of length {len(strats)} truncated to {max_length}, strategy: {contents}")
        strats = strats[:max_length]
        strats[-1] = SpecialTacticsTokens.EOS_ID.value
    repr = torch.tensor(strats, dtype=torch.long)
    return F.pad(repr, (0, max_length - len(strats)), value=SpecialTacticsTokens.PAD_ID.value)


class TacticExample(Dataset):

    def __init__(self, benchmarks: list[Benchmark], config: ModelConfig):
        encoded = embed_benchmarks({b.benchmark for b in benchmarks})
        benchmarks = [b for b in benchmarks if b.benchmark in encoded]   # skip files the GIN cannot encode
        self.smt = [encoded[b.benchmark] for b in benchmarks]
        self.strat = [encode_strats(b.strat, config.max_strat_len) for b in benchmarks]
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


# TODO the best shot at improving things is now to generate more, high quality data.
# a) more best linear strategies found by z3alpha
# b) sample fairly (round-robin style?) from each of the families
# c) use much more compute and leave it for longer
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
        bench_results = sample(bench_results, k=count)
    dataset = TacticExample(bench_results, config)
    return make_splits(dataset)
