"""Cut the benchmark sample into blocks and write the stage-1 task table (one MCTS run per row).

Part of the blocks hold a single family (specialist strategies), the rest mix families. Each block gets
one task; a fraction of blocks get a second task with another seed. c_uct and is_mean vary per task so
the runs explore differently.
"""

import argparse
import logging
import random
from collections import defaultdict
from pathlib import Path

from z3alpha.config import setup_logging

from common import read_csv, write_csv, write_lines

log = logging.getLogger(__name__)


def chunk(items: list[str], size: int) -> list[list[str]]:
    blocks = [items[i : i + size] for i in range(0, len(items), size)]
    if len(blocks) > 1 and len(blocks[-1]) < size // 2:
        blocks[-2] += blocks.pop()
    return blocks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=Path, default=Path("data/qfnia_sample.txt"))
    parser.add_argument("--block-size", type=int, default=40)
    parser.add_argument("--family-frac", type=float, default=0.33, help="share of benchmarks in single-family blocks")
    parser.add_argument("--extra-seed-frac", type=float, default=0.5, help="share of blocks run with a second seed")
    parser.add_argument("--c-uct", type=float, nargs="+", default=[0.5, 1, 2, 4])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--blocks-dir", type=Path, default=Path("data/blocks"))
    parser.add_argument("--tasks", type=Path, default=Path("data/stage1_tasks.csv"))
    args = parser.parse_args()
    setup_logging()
    rng = random.Random(args.seed)

    meta = read_csv(args.sample.with_suffix(".meta.csv"))
    by_family = defaultdict(list)
    for row in meta:
        by_family[row["family"]].append(row["benchmark"])

    blocks: list[tuple[str, list[str]]] = []
    budget = int(len(meta) * args.family_frac)
    used = set()
    for fam in sorted(by_family, key=lambda f: -len(by_family[f])):
        files = by_family[fam]
        rng.shuffle(files)
        n_blocks = min(len(files) // args.block_size, (budget - len(used)) // args.block_size)
        for i in range(n_blocks):
            block = files[i * args.block_size : (i + 1) * args.block_size]
            blocks.append((f"family:{fam}", block))
            used.update(block)

    rest = [row["benchmark"] for row in meta if row["benchmark"] not in used]
    rng.shuffle(rest)
    blocks += [("mixed", block) for block in chunk(rest, args.block_size)]

    if args.blocks_dir.exists():
        for old in args.blocks_dir.glob("block-*.txt"):
            old.unlink()
    for i, (_, block) in enumerate(blocks):
        write_lines(args.blocks_dir / f"block-{i:03d}.txt", block)

    tasks = []
    extra = rng.sample(range(len(blocks)), int(len(blocks) * args.extra_seed_frac))
    for b in list(range(len(blocks))) + sorted(extra):
        tasks.append({
            "task_id": len(tasks),
            "block": str(args.blocks_dir / f"block-{b:03d}.txt"),
            "kind": blocks[b][0],
            "seed": args.seed * 100_000 + len(tasks),
            "c_uct": rng.choice(args.c_uct),
            "is_mean": len(tasks) % 2 == 1,
        })
    write_csv(args.tasks, list(tasks[0]), tasks)

    n_family = sum(kind != "mixed" for kind, _ in blocks)
    log.info(f"{len(blocks)} blocks ({n_family} single-family, {len(blocks) - n_family} mixed), {len(tasks)} tasks")
    log.info(f"Submit with: sbatch --array=0-{len(tasks) - 1} scripts/generate_data.sh stage1")


if __name__ == "__main__":
    main()
