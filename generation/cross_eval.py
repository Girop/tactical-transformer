"""Optional second pass: run strategies found by stage 1 on benchmarks whose block never tried them.

  plan  (once, after stage 1 finished): pick anchor strategies, assign every benchmark the anchors plus
        --per-bench strategies it has not seen, write the shuffled pair list.
  run   (one call per array task): evaluate pairs[shard::num_shards]; resumes from its own output CSV.
"""

import argparse
import csv
import logging
import random
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from z3alpha.config import setup_logging
from z3alpha.evaluator import SolverRunner

from common import LABEL_COLUMNS, PROBE_STRATEGIES, read_csv, read_lines, write_csv

log = logging.getLogger(__name__)


def load_stage1(stage1_root: Path):
    """Return {benchmark: strategies already run on it} and per-run top strategies."""
    seen = defaultdict(set)
    top_counts = Counter()
    for per_bench in sorted(stage1_root.glob("*/linear_strategy_per_benchmark.csv")):
        for row in read_csv(per_bench):
            seen[row["benchmark"]].add(row["strat"])
        summary = read_csv(per_bench.with_name("linear_strategy_summary.csv"))
        summary.sort(key=lambda r: (-int(r["n_solved"]), float(r["par2_avg"])))
        top_counts.update(r["strategy"] for r in summary[:10] if int(r["n_solved"]) > 0)
    return seen, top_counts


def plan(args):
    rng = random.Random(args.seed)
    seen, top_counts = load_stage1(args.stage1_root)
    all_strats = sorted(set().union(*seen.values()))
    benches = read_lines(args.sample)
    log.info(f"{len(all_strats)} distinct stage-1 strategies, {len(seen)} benchmarks with stage-1 results")

    anchors = list(PROBE_STRATEGIES)
    for strat, _ in top_counts.most_common():
        if len(anchors) >= args.anchors:
            break
        if strat not in anchors:
            anchors.append(strat)

    anchor_set = set(anchors)
    pool = [s for s in all_strats if s not in anchor_set]
    rng.shuffle(pool)
    order = list(benches)
    rng.shuffle(order)
    pairs = []
    cursor = 0  # one cursor over the pool for all benchmarks, so coverage per strategy stays even
    for bench in order:
        pairs += [(s, bench) for s in anchors if s not in seen[bench]]
        chosen = []
        for _ in range(len(pool)):
            if len(chosen) == args.per_bench:
                break
            strat = pool[cursor % len(pool)]
            cursor += 1
            if strat not in seen[bench]:
                chosen.append(strat)
        pairs += [(s, bench) for s in chosen]
    rng.shuffle(pairs)

    write_csv(args.pairs, ["strat", "benchmark"], [{"strat": s, "benchmark": b} for s, b in pairs])
    log.info(f"{len(anchors)} anchors, {len(pairs)} pairs for {len(benches)} benchmarks -> {args.pairs}")


def run(args):
    pairs = [(r["strat"], r["benchmark"]) for r in read_csv(args.pairs)][args.shard :: args.num_shards]
    out = args.out_root / f"shard-{args.shard:03d}.csv"
    done = {(r["strat"], r["benchmark"]) for r in read_csv(out)} if out.exists() else set()
    todo = [p for p in pairs if p not in done]
    log.info(f"Shard {args.shard}/{args.num_shards}: {len(pairs)} pairs, {len(done)} done, {len(todo)} to run")

    out.parent.mkdir(parents=True, exist_ok=True)
    new_file = not out.exists()
    extra = [f"-memory:{args.memory_mb}"] if args.memory_mb else []
    with open(out, "a", newline="") as f, ThreadPoolExecutor(max_workers=args.workers) as pool:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(LABEL_COLUMNS)
        futures = [
            pool.submit(SolverRunner(args.z3, bench, args.timeout, i, strat, extra).execute)
            for i, (strat, bench) in enumerate(todo)
        ]
        for n, future in enumerate(as_completed(futures), start=1):
            i, status, runtime, _ = future.result()
            strat, bench = todo[i]
            writer.writerow([strat, bench, status, runtime, status in ("sat", "unsat")])
            f.flush()
            if n % 1000 == 0 or n == len(todo):
                log.info(f"  {n}/{len(todo)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--pairs", type=Path, default=Path("data/cross_pairs.csv"))
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("plan", parents=[shared])
    p.add_argument("--stage1-root", type=Path, default=Path("experiments/stage1"))
    p.add_argument("--sample", type=Path, default=Path("data/qfnia_sample.txt"))
    p.add_argument("--anchors", type=int, default=30)
    p.add_argument("--per-bench", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)

    r = sub.add_parser("run", parents=[shared])
    r.add_argument("--shard", type=int, required=True)
    r.add_argument("--num-shards", type=int, required=True)
    r.add_argument("--timeout", type=int, default=10)
    r.add_argument("--workers", type=int, default=4)
    r.add_argument("--memory-mb", type=int, default=0, help="z3 -memory limit per process (0: none)")
    r.add_argument("--z3", default="z3")
    r.add_argument("--out-root", type=Path, default=Path("experiments/cross"))

    args = parser.parse_args()
    setup_logging()
    {"plan": plan, "run": run}[args.cmd](args)


if __name__ == "__main__":
    main()
