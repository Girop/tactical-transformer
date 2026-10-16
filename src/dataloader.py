from dataclasses import dataclass
from pathlib import Path
import csv
import warnings
from enum import Enum, auto
from collections import defaultdict
from random import sample, shuffle
from typing import Optional, Callable

from transformer import SpecialTacticsTokens, ModelConfig
from z3alpha.parser import parse_linear_strategy
from smt_embed import embed_benchmarks
from tactics import CATALOG

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset


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

    @classmethod
    def from_row(cls, row):
        return cls(row["strat"], Path(row["benchmark"]), Status[row["status"]], float(row["time_s"]), row["solved"] == "True")

    # TODO:
    # @property
    # def family(self)


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
        benchmarks = [b for b in benchmarks if b.benchmark in encoded]
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


def load_examples(dirpath: Path, should_skip: Callable) -> list[Benchmark]:
    assert dirpath.is_dir()
    res = []
    with open(dirpath / "all.csv", newline="") as fp:
        for row in csv.DictReader(fp):
            if should_skip(row):
                continue
            res.append(Benchmark.from_row(row))
    return res


def load_final_predictions(dirpath: Path, excluded: set[str], config: ModelConfig, batch_size: int = 16) -> DataLoader:
    should_skip = lambda row: row["benchmark"] not in excluded
    res = load_examples(dirpath, should_skip)
    dataset = TacticExample(res, config)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True, collate_fn=TacticExample.collate_fn)


def make_splits(dataset: Subset, batch_size=16, validation: float = 0.1, test: float = 0.1) -> dict[str, DataLoader]:
    groups = defaultdict(list)
    for i in range(len(dataset)):
        _, _, b = dataset[i]
        groups[b.benchmark].append(i)

    keys = list(groups.keys())
    shuffle(keys)

    n_test = int(len(keys) * test)
    n_val = int(len(keys) * validation)

    test_indices = [i for k in keys[:n_test] for i in groups[k]]
    val_indices = [i for k in keys[n_test:n_test + n_val] for i in groups[k]]
    train_indices = [i for k in keys[n_test + n_val:] for i in groups[k]]

    return {
        "train": DataLoader(Subset(dataset, train_indices), batch_size=batch_size, shuffle=True, pin_memory=True, collate_fn=TacticExample.collate_fn),
        "validation": DataLoader(Subset(dataset, val_indices), batch_size=batch_size, shuffle=False, pin_memory=True, collate_fn=TacticExample.collate_fn),
        "test": DataLoader(Subset(dataset, test_indices), batch_size=batch_size, shuffle=False, pin_memory=True, collate_fn=TacticExample.collate_fn),
    }


def make_loaders(benchmarks_out: Path, config: ModelConfig, excluded: set[str], count: Optional[int] = None) -> dict[str, DataLoader]:
    should_skip = lambda row: row["benchmark"] in excluded
    bench_results = load_examples(benchmarks_out, should_skip)

    if count is not None and len(bench_results) > count:
        bench_results= sample(bench_results, count * 4) # TODO quick hack, ensures that we will have more or less enough benchmarks if some fail.

    # TODO GIN is failing in encoding way too many benchmarks.
    # print("Before: ", len(bench_results))
    dataset = TacticExample(bench_results, config)
    # print("After: ", len(dataset))
    if count is not None and count < len(dataset):
        indices = sample(range(len(dataset)), k=count)
        dataset = Subset(dataset, indices)
    return make_splits(dataset)
