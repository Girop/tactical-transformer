import time
from pathlib import Path
from typing import Any
import re
import argparse
from transformers import AutoTokenizer

import z3


class Timer:
    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc_info):
        self.elapsed = time.perf_counter() - self._start

    @property
    def elapsed_ms(self) -> float:
        return round(self.elapsed * 1000, 3)


def benchmark(source: str, tactics: list[str], timeout_ms: int = 10_000,) -> dict[str, Any]:
    assertions = z3.parse_smt2_string(source)

    solver = (z3.Tactic(tactics[0]) if len(tactics) == 1 else z3.Then(*tactics)).solver()
    solver.set(timeout=timeout_ms)
    solver.add(*assertions)

    with Timer() as t:
        status = solver.check()
    stats = solver.statistics()
    return {
        "status": str(status),
        "elapsed_ms": t.elapsed_ms,
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


# Design plan - prototype
# Encoder-Decoder, sequence generation
# Output: for now flat list of tactics, assumed to be in a single Then(...), later expand on different combinators
# Input: try out different encoders
# Training dataset: z3alpha generated tactics, discard complex ones, use only sequences of Then(...) + sat/unsat/unknown 
# Base model: ???


def tokenize(source: str) -> list[int]:
    TOKENIZER_NAME = "google/byt5-small"
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    return tokenizer(file, padding=True)


if __name__ == '__main__':
    args = get_args()
    file = args.smt_file.read_text(encoding="utf-8")
    input_ids, att = tokenize(file).values()
    print("Token count: ", sum(att))
    print("Tokens: ", input_ids)
