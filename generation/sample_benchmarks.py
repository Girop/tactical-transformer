"""Pick a family-stratified QF_NIA subset, probe its difficulty, keep a difficulty-balanced sample.

Writes <out> (one benchmark path per line) and <out stem>.meta.csv (family, probe class, probe times).
"""

import argparse
import logging
import random
import statistics
from collections import defaultdict
from itertools import zip_longest
from pathlib import Path

from z3alpha.config import setup_logging
from z3alpha.evaluator import SolverEvaluator
from z3alpha.utils import create_benchmark_list

from common import PROBE_STRATEGIES, QF_NIA_DIR, family_of, write_csv, write_lines

log = logging.getLogger(__name__)

TRIVIAL_TIME = 0.1


def round_robin(groups: list[list[str]], limit: int) -> list[str]:
    """Interleave the groups (one item from each in turn) and take the first `limit` items."""
    out = [x for row in zip_longest(*groups) for x in row if x is not None]
    return out[:limit]


def stratified_candidates(benches: list[str], root: str, n: int, cap: int, rng: random.Random) -> list[str]:
    by_family = defaultdict(list)
    for b in benches:
        by_family[family_of(b, root)].append(b)
    groups = []
    for fam in sorted(by_family):
        files = by_family[fam]
        rng.shuffle(files)
        groups.append(files[:cap])
        log.info(f"{fam}: {len(files)} files, {min(len(files), cap)} eligible")
    return round_robin(groups, n)


def probe(candidates: list[str], z3: str, timeout: float, workers: int) -> dict[str, list[tuple]]:
    evaluator = SolverEvaluator(z3, candidates, timeout, workers)
    results = defaultdict(list)
    for strat in PROBE_STRATEGIES:
        log.info(f"Probing {strat} on {len(candidates)} benchmarks")

        def progress(done, total, solved):
            if done % 500 == 0 or done == total:
                log.info(f"  {done}/{total}, solved {solved}")

        for bench, res in zip(candidates, evaluator.evaluate(strat, progress)):
            results[bench].append(res)  # (solved, time, status)
    return results


def classify(res: list[tuple]) -> str:
    if all(solved and t < TRIVIAL_TIME for solved, t, _ in res):
        return "trivial"
    if any(solved for solved, _, _ in res):
        return "medium"
    return "unsolved"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-dir", default=QF_NIA_DIR)
    parser.add_argument("--candidates", type=int, default=5000, help="benchmarks to probe")
    parser.add_argument("--per-family-cap", type=int, default=1200)
    parser.add_argument("--keep", type=int, default=2400, help="benchmarks in the final sample")
    parser.add_argument("--trivial-frac", type=float, default=0.10)
    parser.add_argument("--unsolved-frac", type=float, default=0.25)
    parser.add_argument("--probe-timeout", type=float, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--z3", default="z3")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("data/qfnia_sample.txt"))
    args = parser.parse_args()
    setup_logging()
    rng = random.Random(args.seed)

    benches = create_benchmark_list([args.benchmark_dir])
    candidates = stratified_candidates(benches, args.benchmark_dir, args.candidates, args.per_family_cap, rng)
    log.info(f"{len(candidates)} candidates from {len(benches)} benchmarks")

    results = probe(candidates, args.z3, args.probe_timeout, args.workers)
    classes = {b: classify(results[b]) for b in candidates}
    times = [t for b in candidates for _, t, _ in results[b]]
    log.info(f"Mean probe runtime {statistics.mean(times):.2f}s (timeout {args.probe_timeout}s)")

    # Fill difficulty quotas; each class keeps the family round-robin order of `candidates`.
    by_class = {c: [b for b in candidates if classes[b] == c] for c in ("trivial", "medium", "unsolved")}
    quota = {
        "trivial": int(args.keep * args.trivial_frac),
        "unsolved": int(args.keep * args.unsolved_frac),
    }
    quota["medium"] = args.keep - quota["trivial"] - quota["unsolved"]
    sample = []
    for c in ("trivial", "unsolved", "medium"):
        sample += by_class[c][: quota[c]]
    # A class short of its quota is topped up from the leftovers, medium first.
    taken = set(sample)
    leftovers = [b for c in ("medium", "unsolved", "trivial") for b in by_class[c] if b not in taken]
    sample += leftovers[: args.keep - len(sample)]

    for c, bs in by_class.items():
        log.info(f"{c}: {len(bs)} candidates, {sum(classes[b] == c for b in sample)} kept")
    log.info(f"Sample size {len(sample)}")

    write_lines(args.out, sample)
    meta = []
    for b in sample:
        row = {"benchmark": b, "family": family_of(b, args.benchmark_dir), "probe_class": classes[b]}
        for i, (_, t, status) in enumerate(results[b]):
            row[f"probe{i}_status"], row[f"probe{i}_time"] = status, f"{t:.3f}"
        meta.append(row)
    columns = ["benchmark", "family", "probe_class"] + [
        f"probe{i}_{k}" for i in range(len(PROBE_STRATEGIES)) for k in ("status", "time")
    ]
    write_csv(args.out.with_suffix(".meta.csv"), columns, meta)
    log.info(f"Wrote {args.out} and {args.out.with_suffix('.meta.csv')}")


if __name__ == "__main__":
    main()
