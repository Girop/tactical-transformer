import time
from pathlib import Path
from typing import Any
import re
import argparse
import torch
from torch import nn

import z3


class ASTTransformer(nn.Module):
    def __init__(self, vocab_size: int, out_size: int, d_model=256, n_heads=8, n_layers=4, max_len=2_000):
        super().__init__()

        self.ast_embedding = nn.Embedding(vocab_size, d_model)
        self.out_embedding = nn.Embedding(out_size, d_model)
        self.src_pos_embedding = nn.Embedding(max_len, d_model)
        self.tgt_pos_embedding = nn.Embedding(max_len, d_model)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=n_heads,
            num_encoder_layers=n_layers,
            num_decoder_layers=n_layers,
            dim_feedforward=1024,
            batch_first=True
        )

        self.out_proj = nn.Linear(d_model, out_size)


    def forward(self, src_ids, tgt_ids):
        src_pos = torch.arange(src_ids.size(1), device=src_ids.device)
        tgt_pos = torch.arange(tgt_ids.size(1), device=tgt_ids.device)

        src = self.ast_embedding(src_ids) + self.src_pos_embedding(src_pos)
        tgt = self.out_embedding(tgt_ids) + self.tgt_pos_embedding(tgt_pos)

        causal_mask = torch.triu(
            torch.ones(
                tgt_ids.size(1),
                tgt_ids.size(1),
                device=tgt_ids.device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )

        x = self.transformer(src, tgt, tgt_mask=causal_mask)

        return self.out_proj(x)


class Timer:
    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc_info):
        self.elapsed = time.perf_counter() - self._start

    @property
    def elapsed_ms(self) -> float:
        return round(self.elapsed * 1000, 3)


def run_solver(source: str, tactics: list[str], timeout_ms: int = 10_000) -> dict[str, Any]:
    assertions = z3.parse_smt2_string(source)

    solver = z3.Then(*tactics).solver()
    solver.set(timeout=timeout_ms)
    solver.add(*assertions)

    status = solver.check()
    stats = solver.statistics()
    return {
        "status": str(status),
        "statistics": {key: stats.get_key_value(key) for key in stats.keys()},
    }


LOGIC_RE = re.compile(r"\(\s*set-logic\s+([^\s()]+)", re.IGNORECASE)


def extract_features(source: str) -> dict[str, Any]:
    """Parse `formulation` and return its logic, AST/operator/sort counts, and Z3 probe values."""
    assertions = z3.parse_smt2_string(source)
    goal = z3.Goal()
    goal.add(*assertions)

    seen: set[int] = set()
    operators: dict[str, int] = {}
    sorts: dict[str, int] = {}
    max_depth = 0
    stack = [(expr, 1) for expr in goal]
    while stack:
        expr, depth = stack.pop()
        max_depth = max(max_depth, depth)
        seen.add(expr.get_id())
        if z3.is_app(expr):
            name = expr.decl().name()
            operators[name] = operators.get(name, 0) + 1
            sorts[expr.sort().sexpr()] = sorts.get(expr.sort().sexpr(), 0) + 1
            stack.extend((expr.arg(i), depth + 1) for i in range(expr.num_args()))

    logic_match = LOGIC_RE.search(source)
    return {
        "logic": logic_match.group(1) if logic_match else None,
        "assertion_count": len(goal),
        "ast_unique_nodes": len(seen),
        "ast_max_depth": max_depth,
        "operator_counts": dict(sorted(operators.items())),
        "sort_counts": dict(sorted(sorts.items())),
        "z3_probes": {name: float(z3.Probe(name)(goal)) for name in sorted(z3.probes())},
    }


def get_args() -> argparse.Namespace:
    args = argparse.ArgumentParser()
    args.add_argument("--smt-file", type=Path, required=True)
    return args.parse_args()


def discover_formulations(root: Path = Path("./smtlib/")) -> list[Path]:
    return list(root.rglob("*.smt2"))


def load_formulations(count: int) -> list[str]:
    res = []
    for file in discover_formulations()[:count]:
        source = file.read_text(encoding="utf-8")
        assertions = z3.parse_smt2_string(source)
        ast_text = "\n".join(f"(assert {a.sexpr()})" for a in assertions)
        res.append(ast_text)
    return res
