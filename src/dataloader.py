from dataclasses import dataclass
from pathlib import Path
import csv
from typing import Callable
from enum import Enum, auto
from tqdm import tqdm
from random import choices

import torch
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


class TacticExample(Dataset):

    def __init__(self, benchmarks: list[Benchmark], smt_encoder: Callable, strat_encoder: Callable):
        self.smt = [smt_encoder(p.benchmark) for p in tqdm(benchmarks, desc="Encoding SMT problems")] # TODO a) Discarding too much info b) the benchmarks themselves can be interned
        self.strat = [strat_encoder(b.strat) for b in tqdm(benchmarks, desc="Encoding strategies")]
        self.bench_data = benchmarks

    def __len__(self):
        assert len(self.smt) == len(self.strat) == len(self.bench_data)
        return len(self.smt)

    def __getitem__(self, idx):
        return self.smt[idx], self.strat[idx], self.bench_data[idx]

    @staticmethod
    def collate_examples(batch: list[tuple[torch.Tensor, torch.Tensor, Benchmark]]) -> tuple[torch.Tensor, torch.Tensor, list[Benchmark]]:
        smts, strats, bench_data = zip(*batch)
        return torch.stack(smts), torch.stack(strats), list(bench_data)


def make_splits(benchmarks: TacticExample, batch_size=16, validation: float = 0.1, test: float = 0.1) -> dict[str, DataLoader]:
    train_size = 1.0 - validation - test
    train_idx, val_idx, test_idx = random_split(benchmarks, [train_size, validation, test])
    return {
        "train": DataLoader(train_idx, batch_size=batch_size, shuffle=True, pin_memory=True, collate_fn=TacticExample.collate_examples),
        "validation": DataLoader(val_idx, batch_size=batch_size, shuffle=False, pin_memory=True, collate_fn=TacticExample.collate_examples),
        "test": DataLoader(test_idx, batch_size=batch_size, shuffle=False, pin_memory=True, collate_fn=TacticExample.collate_examples)
    }


def load_examples(dirpath: Path) -> list[Benchmark]:
    assert dirpath.is_dir()
    res = []
    for p in dirpath.rglob("*.csv"):
        with open(p, newline="") as fp:
            for row in csv.DictReader(fp):
                res.append(Benchmark(row["strat"], Path(row["benchmark"]), Status[row["status"]], float(row["time_s"]), row["solved"] == "True"))
    return res


def make_loaders(benchmarks_out: Path, count: int, smt_encoder: Callable, str_encoder: Callable) -> dict[str, DataLoader]:
    bench_results = choices(load_examples(benchmarks_out), k=count)
    dataset = TacticExample(bench_results, smt_encoder, str_encoder)
    return make_splits(dataset)
