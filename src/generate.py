"""Data-generation pipeline for the tactic transformer.

Subcommands (run in order):
  prepare   Discover SMT-LIB benchmarks, partition into blocks, write task table.
  stage1    Run one MCTS search per task (one Slurm array element each).
  merge     Merge stage-1 results, enrich with logic/family, write training CSVs.
  split     Draw held-out test benchmarks per logic/family, write them to an exclusion list.

See scripts/generate_data.sh for cluster and local invocation.
"""

import argparse
import csv
import json
import logging
import random
import shutil
import statistics
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path

from utils import seed_everything
from z3alpha.config import setup_logging
from z3alpha.mcts import LinearStrategySearchRun
from z3alpha.mcts.param_selection import ParamSelectionConfig
from z3alpha.mcts.run import MctsConfig
from z3alpha.tactics.logic_config import load_logic_config
from z3alpha.utils import create_benchmark_list

log = logging.getLogger(__name__)


LABEL_COLUMNS = ["strat", "logic", "family", "benchmark", "status", "time_s", "solved"]
STATUSES = {"sat", "unsat", "timeout", "unknown", "error"}


def family_of(bench: str, root: str) -> str:
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


def build_blocks(benches: list[str], block_size: int) -> list[list[str]]:
    """Partition benchmarks into fixed-size blocks (in discovery order)."""
    blocks: list[list[str]] = []
    for i in range(0, len(benches), block_size):
        block = benches[i:i + block_size]
        if blocks and len(block) < block_size // 2:
            blocks[-1].extend(block)
        else:
            blocks.append(block)
    return blocks


def build_tasks(
    all_blocks: list[tuple[str, list[str]]], blocks_dir: Path, base_seed: int,
    c_uct_choices: list[float],
) -> list[dict]:
    """Assign MCTS hyperparameters to blocks."""
    tasks = []
    for b, (logic, _) in enumerate(all_blocks):
        tasks.append({
            "task_id": b,
            "block": str(blocks_dir / f"block-{b:03d}.txt"),
            "logic": logic,
            "seed": base_seed * 100_000 + b,
            "c_uct": random.choice(c_uct_choices),
        })
    return tasks


def cmd_prepare(args):
    seed_everything(args.seed)

    all_blocks: list[tuple[str, list[str]]] = []  # (logic, benches)

    for logic in args.logics:
        logic_dir = args.smtlib_root / logic
        if not logic_dir.exists():
            log.warning(f"Logic directory not found, skipping: {logic_dir}")
            continue
        benches = create_benchmark_list([str(logic_dir)])
        allowed = set(args.benchmarks)
        benches = [b for b in benches if family_of(b, str(logic_dir)) in allowed]
        log.info(f"{logic}: {len(benches)} benchmarks")
        blocks = build_blocks(benches, args.block_size)
        log.info(f"  {len(blocks)} blocks of ~{args.block_size}")
        all_blocks.extend((logic, block) for block in blocks)

    if not all_blocks:
        log.error("No benchmarks found — check --smtlib-root and --logics")
        return

    tasks = build_tasks(all_blocks, args.blocks_dir, args.seed, args.c_uct)

    if args.blocks_dir.exists():
        for old in args.blocks_dir.glob("block-*.txt"):
            old.unlink()
    for i, (_, block) in enumerate(all_blocks):
        write_lines(args.blocks_dir / f"block-{i:03d}.txt", block)
    write_csv(args.tasks, list(tasks[0]), tasks)

    log.info(f"{len(all_blocks)} blocks, {len(tasks)} tasks written to {args.tasks}")
    log.info(f"Submit with: sbatch --array=0-{len(tasks) - 1} scripts/generate_data.sh stage1")


def load_task(args: argparse.Namespace) -> dict | None:
    """Resolve task from --task-id. Returns None if task_id is past end of table."""
    rows = read_csv(args.tasks)
    if args.task_id >= len(rows):
        return None
    row = rows[args.task_id]
    assert int(row["task_id"]) == args.task_id
    return {"block": Path(row["block"]), "logic": row["logic"], "seed": int(row["seed"]),
            "c_uct": float(row["c_uct"])}


def prepare_out_dir(out: Path) -> bool:
    """Return False if a finished run is already there; clear an unfinished one."""
    meta = out / "run.json"
    if meta.exists() and json.loads(meta.read_text()).get("finished"):
        return False
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    return True


def cmd_stage1(args):
    task = load_task(args)
    if task is None:
        log.info(f"Task {args.task_id} is past the end of {args.tasks}, nothing to do")
        return

    out = args.out_root / f"{Path(task['block']).stem}-s{task['seed']}"
    if not prepare_out_dir(out):
        log.info(f"{out} already finished, nothing to do")
        return

    benches = read_lines(task["block"])
    random.seed(task["seed"])
    config = MctsConfig(
        sim_num=args.sims, timeout=args.timeout, c_uct=task["c_uct"],
        random_seed=task["seed"], llm_prior=None,
        param_selector=ParamSelectionConfig(enabled=False),
    )
    run = LinearStrategySearchRun(
        config, benches, task["logic"], args.z3, "par2", out,
        batch_size=args.workers, logic_config=load_logic_config(task["logic"]),
    )
    if not args.debug_trace:
        run.trace_log.setLevel(logging.INFO)

    meta = {
        **{k: str(v) if isinstance(v, Path) else v for k, v in task.items()},
        "benchmarks": len(benches), "sims": args.sims, "timeout": args.timeout,
        "workers": args.workers,
        "z3_version": subprocess.check_output([args.z3, "--version"], text=True).strip(),
        "finished": False,
    }
    (out / "run.json").write_text(json.dumps(meta, indent=2))

    start = time.time()
    run.start()
    elapsed = time.time() - start

    meta.update(finished=True, sims_done=args.sims, distinct_strategies=len(run.res_database),
                elapsed_s=round(elapsed), best_strategy=run.get_best_strat())
    (out / "run.json").write_text(json.dumps(meta, indent=2))
    log.info(f"Done: {len(run.res_database)} distinct strategies x {len(benches)} benchmarks "
             f"in {args.sims} simulations, {elapsed / 3600:.2f}h")


def cmd_merge(args):
    rows: list[tuple] = []
    for run_dir in sorted(args.stage1_root.iterdir()):
        run_meta = run_dir / "run.json"
        results = run_dir / "linear_strategy_per_benchmark.csv"
        if not results.exists():
            continue
        logic = json.loads(run_meta.read_text())["logic"] if run_meta.exists() else ""
        smtlib_logic_root = str(args.smtlib_root / logic) if logic else ""
        for r in read_csv(results):
            bench = r["benchmark"]
            family = family_of(bench, smtlib_logic_root) if smtlib_logic_root else ""
            status = r["status"] if r["status"] in STATUSES else "error"
            rows.append((r["strat"], logic, family, bench, status,
                         float(r["time_s"]), status in ("sat", "unsat")))

    if not rows:
        log.error("No results found")
        return

    args.train.mkdir(parents=True, exist_ok=True)
    with open(args.train / "all.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(LABEL_COLUMNS)
        writer.writerows(rows)

    # --- stats ---
    by_bench: dict[str, list] = defaultdict(list)
    for row in rows:
        by_bench[row[3]].append(row)
    per_bench = [len({r[0] for r in b}) for b in by_bench.values()]

    log.info(f"Rows: {len(rows)}")
    log.info(f"Status: {dict(Counter(r[4] for r in rows))}")
    log.info(f"Distinct strategies: {len({r[0] for r in rows})}, benchmarks: {len(by_bench)}")
    log.info(f"Strategies per benchmark: min {min(per_bench)}, median {statistics.median(per_bench):.0f}, max {max(per_bench)}")
    log.info(f"Benchmarks solved by at least one strategy: {sum(any(r[6] for r in b) for b in by_bench.values())}")

    log.info("Solve rate by family:")
    by_fam: dict[str, list] = defaultdict(list)
    for r in rows:
        by_fam[r[2] or "?"].append(r[6])
    for fam in sorted(by_fam):
        log.info(f"  {fam:40s} {statistics.mean(by_fam[fam]):.3f}  ({len(by_fam[fam])} rows)")

    log.info(f"Wrote {args.train / 'all.csv'}")


def cmd_split(args):
    all_csv = args.train / "all.csv"
    if not all_csv.exists():
        log.error(f"all.csv not found at {all_csv} — run merge first")
        return

    logic_families: dict[str, set[str]] = defaultdict(set)
    for r in read_csv(all_csv):
        logic_families[r["logic"]].add(r["family"])

    if not logic_families:
        log.error("No logic/family pairs found in all.csv")
        return

    seed_everything(args.seed)
    excluded: list[str] = []

    for logic, families in sorted(logic_families.items()):
        logic_dir = args.smtlib_root / logic
        if not logic_dir.exists():
            log.warning(f"Logic directory not found, skipping: {logic_dir}")
            continue
        benches = create_benchmark_list([str(logic_dir)])
        by_family: dict[str, list[str]] = defaultdict(list)
        for b in benches:
            family = family_of(b, str(logic_dir))
            if family in families:
                by_family[family].append(b)
        # Draw per family so small families still get test benchmarks.
        for family in sorted(by_family):
            members = by_family[family]
            k = max(1, round(len(members) * args.test_fraction))
            excluded.extend(random.sample(members, k))
            log.info(f"{logic}/{family}: {k} of {len(members)} held out")

    if not excluded:
        log.error("No benchmarks found — check --smtlib-root and --train")
        return

    write_lines(args.out, sorted(excluded))
    log.info(f"{len(excluded)} excluded benchmarks written to {args.out}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    # --- prepare ---
    p = sub.add_parser("prepare", help="Discover benchmarks, partition into blocks, write task table")
    p.add_argument("--smtlib-root", type=Path, default=Path("smtlib/non-incremental"),
                    help="root directory containing one subdirectory per logic (e.g. smtlib/non-incremental/)")
    p.add_argument("--logics", nargs="+", required=True,
                    help="logic names to process (subdirectories of --smtlib-root)")
    p.add_argument("--benchmarks", nargs="+", required=True,
                    help="benchmark families to include (e.g. leipzig AproVE)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--block-size", type=int, default=40)
    p.add_argument("--c-uct", type=float, nargs="+", default=[0.5, 1, 2, 4])
    p.add_argument("--blocks-dir", type=Path, default=Path("data/blocks"))
    p.add_argument("--tasks", type=Path, default=Path("data/stage1_tasks.csv"))

    # --- stage1 ---
    p = sub.add_parser("stage1", help="Run one MCTS search per task")
    p.add_argument("--tasks", type=Path, default=Path("data/stage1_tasks.csv"))
    p.add_argument("--task-id", type=int, required=True, help="row of --tasks to run")
    p.add_argument("--sims", type=int, default=300)
    p.add_argument("--timeout", type=int, default=10)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--z3", default="z3")
    p.add_argument("--out-root", type=Path, default=Path("experiments/stage1"))
    p.add_argument("--debug-trace", action="store_true", help="keep z3alpha's per-step DEBUG trace")

    # --- merge ---
    p = sub.add_parser("merge", help="Merge stage-1 results, enrich with logic/family, write training CSVs")
    p.add_argument("--smtlib-root", type=Path, default=Path("smtlib/non-incremental"),
                    help="same --smtlib-root used in prepare (needed to derive family from benchmark paths)")
    p.add_argument("--stage1-root", type=Path, default=Path("experiments/stage1"))
    p.add_argument("--train", type=Path, default=Path("experiments/train"))

    # --- split ---
    p = sub.add_parser("split", help="Draw held-out test benchmarks, write them to an exclusion list")
    p.add_argument("--smtlib-root", type=Path, default=Path("smtlib/non-incremental"),
                    help="same --smtlib-root used in prepare, so paths match the benchmark column of all.csv")
    p.add_argument("--train", type=Path, default=Path("experiments/train"),
                    help="directory containing all.csv produced by merge")
    p.add_argument("--test-fraction", type=float, default=0.2,
                    help="fraction held out per logic/family (at least one benchmark each)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=Path("data/excluded.txt"))

    return parser


def main():
    args = build_parser().parse_args()
    setup_logging()
    {"prepare": cmd_prepare, "stage1": cmd_stage1, "merge": cmd_merge,
     "split": cmd_split}[args.command](args)


if __name__ == "__main__":
    main()
