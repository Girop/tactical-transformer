"""Merge stage-1 and cross-eval results into one table, print coverage stats, write the training set.

  <labels>/all.csv   every (strategy, benchmark) result once, plus source and run_id columns
  <train>/best.csv   per benchmark, the solved rows within max(1.5 x best, best + 0.5s) of the fastest;
                     same columns as z3alpha's per-benchmark CSV, so src/dataloader.py:load_examples reads it
"""

import argparse
import csv
import json
import logging
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from z3alpha.config import setup_logging
from z3alpha.parser import parse_linear_strategy

from common import LABEL_COLUMNS, STATUSES, read_csv

log = logging.getLogger(__name__)


def sources(stage1_root: Path, cross_root: Path):
    """Yield (source, run_id, csv path): finished stage-1 runs first, then unfinished ones, then cross-eval."""
    runs = sorted(p for p in stage1_root.glob("*/linear_strategy_per_benchmark.csv"))

    def finished(p):
        meta = p.with_name("run.json")
        return meta.exists() and json.loads(meta.read_text()).get("finished", False)

    for p in sorted(runs, key=lambda p: not finished(p)):
        yield "stage1", p.parent.name, p
    for p in sorted(cross_root.glob("shard-*.csv")):
        yield "cross", p.stem, p


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1-root", type=Path, default=Path("experiments/stage1"))
    parser.add_argument("--cross-root", type=Path, default=Path("experiments/cross"))
    parser.add_argument("--sample", type=Path, default=Path("data/qfnia_sample.txt"))
    parser.add_argument("--labels", type=Path, default=Path("experiments/labels"))
    parser.add_argument("--train", type=Path, default=Path("experiments/train"))
    parser.add_argument("--best-ratio", type=float, default=1.5)
    parser.add_argument("--best-slack", type=float, default=0.5)
    args = parser.parse_args()
    setup_logging()

    meta_path = args.sample.with_suffix(".meta.csv")
    family = {r["benchmark"]: r["family"] for r in read_csv(meta_path)} if meta_path.exists() else {}

    rows = []  # (strat, bench, status, time, solved, source, run_id)
    seen = set()
    dupes = 0
    strat_runs = defaultdict(set)  # stage-1 runs that evaluated each strategy
    for source, run_id, path in sources(args.stage1_root, args.cross_root):
        for r in read_csv(path):
            if source == "stage1":
                strat_runs[r["strat"]].add(run_id)
            key = (r["strat"], r["benchmark"])
            if key in seen:
                dupes += 1
                continue
            seen.add(key)
            status = r["status"] if r["status"] in STATUSES else "error"
            rows.append((r["strat"], r["benchmark"], status, float(r["time_s"]), status in ("sat", "unsat"), source, run_id))
    if not rows:
        log.error("No results found")
        return

    args.labels.mkdir(parents=True, exist_ok=True)
    with open(args.labels / "all.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(LABEL_COLUMNS + ["source", "run_id"])
        writer.writerows(rows)

    by_bench = defaultdict(list)
    for row in rows:
        by_bench[row[1]].append(row)
    best = []
    for bench_rows in by_bench.values():
        solved = [r for r in bench_rows if r[4]]
        if not solved:
            continue
        t_best = min(r[3] for r in solved)
        limit = max(args.best_ratio * t_best, t_best + args.best_slack)
        best += [r[:5] for r in solved if r[3] <= limit]
    args.train.mkdir(parents=True, exist_ok=True)
    with open(args.train / "best.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(LABEL_COLUMNS)
        writer.writerows(best)

    # --- stats
    length = {s: len(parse_linear_strategy(s)) for s in {r[0] for r in rows}}
    per_bench = [len({r[0] for r in bench_rows}) for bench_rows in by_bench.values()]
    runs = {run for rs in strat_runs.values() for run in rs}
    unique_to_run = Counter(next(iter(rs)) for s, rs in strat_runs.items() if len(rs) == 1)

    print(f"\nRows: {len(rows)} ({dupes} duplicate pairs dropped), sources: {dict(Counter(r[5] for r in rows))}")
    print(f"Distinct strategies: {len(length)}, benchmarks: {len(by_bench)}")
    print(f"Strategies per benchmark: min {min(per_bench)}, median {statistics.median(per_bench)}, max {max(per_bench)}")
    print(f"Status: {dict(Counter(r[2] for r in rows))}")
    print(f"Stage-1 runs: {len(runs)}; strategies found by only one run: median "
          f"{statistics.median([unique_to_run[run] for run in runs]) if runs else 0} per run")
    print(f"Benchmarks solved by some strategy: {sum(any(r[4] for r in b) for b in by_bench.values())}")
    print(f"Training rows (best.csv): {len(best)}")

    print("\nSolve rate by strategy length:")
    by_len = defaultdict(list)
    for r in rows:
        by_len[length[r[0]]].append(r[4])
    for n in sorted(by_len):
        print(f"  {n:2d}: {statistics.mean(by_len[n]):.3f}  ({len(by_len[n])} rows, "
              f"{sum(v == n for v in length.values())} strategies)")

    if family:
        print("\nSolve rate by family:")
        by_fam = defaultdict(list)
        for r in rows:
            by_fam[family.get(r[1], "?")].append(r[4])
        for fam in sorted(by_fam):
            print(f"  {fam:40s} {statistics.mean(by_fam[fam]):.3f}  ({len(by_fam[fam])} rows)")

    log.info(f"Wrote {args.labels / 'all.csv'} and {args.train / 'best.csv'}")


if __name__ == "__main__":
    main()
