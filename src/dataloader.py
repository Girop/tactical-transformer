from dataclasses import dataclass
from pathlib import Path
import csv
import warnings
from enum import Enum, auto
from collections import defaultdict
from random import shuffle
from typing import Callable
from tqdm import tqdm

from transformer import SpecialTacticsTokens, ModelConfig
from z3alpha.parser import parse_linear_strategy
from smt_embed import embed_benchmarks
from tactics import CATALOG

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


class Status(Enum):
    sat = auto()
    unsat = auto()
    timeout = auto()
    unknown = auto()
    error = auto()

    def is_solved(self):
        return self in (Status.sat, Status.unsat)


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

    @property
    def logic(self) -> str:
        return self.benchmark.parts[1]


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

# Cutoff

class TacticExample(Dataset):

    def __init__(self, smt_encodings: dict[Path, torch.Tensor], benchmarks: list[Benchmark], config: ModelConfig):
        self.smt = [smt_encodings[b.benchmark] for b in benchmarks]
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


class Data:

    def __init__(self, allcsv: Path, config: ModelConfig, batch_size: int = 16):
        assert allcsv.is_file(), 'Expected single all.csv file'
        self.allcsv = allcsv
        self.config = config
        self.batch_size = batch_size

    def __load_examples(self, should_skip: Callable) -> list[Benchmark]:
        res = []
        with open(self.allcsv, newline="") as fp:
            for row in csv.DictReader(fp):
                if should_skip(row):
                    continue
                res.append(Benchmark.from_row(row))
        return res


    def __embed(self, pool: list[Benchmark], needed: int) -> dict[Path, torch.Tensor]:
        processed = 0
        embeddings: dict[Path, torch.Tensor] = {}
        pbar = tqdm(total=needed, desc="GIN embedding")

        while len(embeddings) < needed:
            batch = pool[processed:processed + self.batch_size]
            if len(batch) == 0:
                break
            batch_emb = embed_benchmarks({b.benchmark for b in batch})
            embeddings.update(batch_emb)
            processed += len(batch)
            pbar.update(len(batch_emb))
        pbar.close()
        return embeddings


    def train_splits(self, train_size: int, validation: float, excluded: set[str]) -> tuple[DataLoader, DataLoader]:
        pool = self.__load_examples(lambda row: row["benchmark"] not in excluded and not Status[row["status"]].is_solved())
        shuffle(pool)
        total_needed = int(train_size + train_size * validation)
        embeddings = self.__embed(pool, total_needed)
        pool = [p for p in pool if p.benchmark in embeddings.keys()]
        if len(pool) < total_needed:
            warnings.warn(f"Only {len(pool)}/{total_needed} samples survived GIN encoding")

        by_benchmark = defaultdict(list)
        for i, b in enumerate(pool):
            by_benchmark[b.benchmark].append(i)

        keys = list(by_benchmark.keys())
        shuffle(keys)
        n_val = int(len(keys) * validation)

        val_indices = [i for k in keys[:n_val] for i in by_benchmark[k]]
        train_indices = [i for k in keys[n_val:] for i in by_benchmark[k]]

        train_benchmarks = [pool[i] for i in train_indices][:train_size]
        val_benchmarks = [pool[i] for i in val_indices][:int(train_size * validation)]

        train_ds = TacticExample(embeddings, train_benchmarks, self.config)
        val_ds = TacticExample(embeddings, val_benchmarks, self.config)

        return (
            DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, pin_memory=True, collate_fn=TacticExample.collate_fn),
            DataLoader(val_ds, batch_size=self.batch_size, shuffle=False, pin_memory=True, collate_fn=TacticExample.collate_fn)
        )


    def test_split(self, size: int, excluded: set[str], batch_size: int = 16) -> DataLoader:
        pool = self.__load_examples(lambda row: row["benchmark"] not in excluded)
        shuffle(pool)
        embeddings = self.__embed(pool, size)
        pool = [p for p in pool if p.benchmark in embeddings]
        if len(pool) < size:
            warnings.warn(f"Only {len(pool)}/{size} test samples survived GIN encoding")

        dataset = TacticExample(embeddings, pool[:size], self.config)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            collate_fn=TacticExample.collate_fn
        )

