"""Helpers shared by the data-generation scripts (see scripts/generate_data.sh)."""

import csv
from pathlib import Path

# Columns of z3alpha's linear_strategy_per_benchmark.csv; src/dataloader.py:load_examples reads them.
LABEL_COLUMNS = ["strat", "benchmark", "status", "time_s", "solved"]
STATUSES = {"sat", "unsat", "timeout", "unknown", "error"}

# Cheap reference strategies: used to probe benchmark difficulty and always kept as cross-eval anchors.
PROBE_STRATEGIES = ["smt", "qfnia", "(then simplify nla2bv smt)"]

QF_NIA_DIR = "smtlib/non-incremental/QF_NIA"


def family_of(bench: str, root: str = QF_NIA_DIR) -> str:
    return Path(bench).relative_to(root).parts[0]


def read_lines(path: Path) -> list[str]:
    return [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]


def write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{line}\n" for line in lines))


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
