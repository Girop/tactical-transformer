"""Run one z3alpha stage-1 (linear) MCTS search on a block of benchmarks.

Every strategy the search evaluates is run on every benchmark of the block and logged by z3alpha to
<out>/linear_strategy_per_benchmark.csv. That file is the labelled data; the search is only a way of
choosing which strategies to evaluate. Tactic parameters are off (the dataloader encodes tactic names only).
"""

import argparse
import json
import logging
import random
import subprocess
import time
from pathlib import Path

from z3alpha.config import setup_logging
from z3alpha.mcts import LinearStrategySearchRun
from z3alpha.mcts.param_selection import ParamSelectionConfig
from z3alpha.mcts.run import MctsConfig
from z3alpha.tactics.logic_config import load_logic_config

from common import read_csv, read_lines

log = logging.getLogger(__name__)

LOGIC = "QF_NIA"


class DiverseLinearSearch(LinearStrategySearchRun):
    """Stage-1 search with a wall-clock budget and optional pure-random simulations.

    With probability `random_frac` a simulation skips tree selection and evaluates a random rollout
    from the root, which counters MCTS converging on one region of the strategy space.
    """

    def __init__(self, *args, random_frac: float = 0.0, deadline: float | None = None, **kwargs):
        self.random_frac = random_frac
        self.deadline = deadline
        super().__init__(*args, **kwargs)

    def _one_simulation(self) -> None:
        if self.random_frac and random.random() < self.random_frac:
            self.env = self._create_env()
            self._rollout()
            value = self.env.get_value(self.res_database, self.value_type)
            self.trace_log.info(f"Random Strategy: {self.env}\nFinal Return: {value}\n")
            self._update_top_strategies(value, str(self.env))
            return
        super()._one_simulation()

    def start(self) -> None:
        for i in range(self.num_simulations):
            if self.deadline is not None and time.time() > self.deadline:
                log.info(f"Wall-clock budget reached after {i} simulations")
                break
            self.num_sim = i
            self.trace_log.info(f"Simulation {i} starts")
            self._one_simulation()
            self._after_simulation()
            if i % 10 == 0:
                log.info(f"Simulation {i}: {len(self.res_database)} distinct strategies")
        else:
            i = self.num_simulations
        self.sims_done = i


def task_from_table(path: Path, task_id: int) -> dict | None:
    """Row `task_id` of the task table, or None past its end (arrays may be submitted larger than the table)."""
    rows = read_csv(path)
    if task_id >= len(rows):
        return None
    row = rows[task_id]
    assert int(row["task_id"]) == task_id
    return {
        "block": Path(row["block"]),
        "seed": int(row["seed"]),
        "c_uct": float(row["c_uct"]),
        "is_mean": row["is_mean"] == "True",
        "kind": row["kind"],
    }


def prepare_out_dir(out: Path) -> bool:
    """Return False if a finished run is already there; move an unfinished one aside."""
    meta = out / "run.json"
    if meta.exists() and json.loads(meta.read_text()).get("finished"):
        return False
    if out.exists():
        n = 0
        while (partial := out.with_name(f"{out.name}.partial-{n}")).exists():
            n += 1
        out.rename(partial)
        log.warning(f"Moved unfinished run to {partial}")
    out.mkdir(parents=True)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, default=Path("data/stage1_tasks.csv"))
    parser.add_argument("--task-id", type=int, help="row of --tasks to run")
    parser.add_argument("--block", type=Path, help="run on this block instead of a task row")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--c-uct", type=float, default=1.0)
    parser.add_argument("--is-mean", action="store_true")
    parser.add_argument("--sims", type=int, default=300)
    parser.add_argument("--timeout", type=int, default=10)
    parser.add_argument("--random-frac", type=float, default=0.0)
    parser.add_argument("--max-hours", type=float, help="stop starting new simulations after this long")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--z3", default="z3")
    parser.add_argument("--out-root", type=Path, default=Path("experiments/stage1"))
    parser.add_argument("--debug-trace", action="store_true", help="keep z3alpha's per-step DEBUG trace")
    args = parser.parse_args()
    setup_logging()

    if args.block is not None:
        task = {"block": args.block, "seed": args.seed, "c_uct": args.c_uct, "is_mean": args.is_mean, "kind": "manual"}
    elif args.task_id is not None:
        task = task_from_table(args.tasks, args.task_id)
        if task is None:
            log.info(f"Task {args.task_id} is past the end of {args.tasks}, nothing to do")
            return
    else:
        parser.error("give --task-id or --block")

    out = args.out_root / f"{task['block'].stem}-s{task['seed']}"
    if not prepare_out_dir(out):
        log.info(f"{out} already finished, nothing to do")
        return

    benches = read_lines(task["block"])
    random.seed(task["seed"])
    config = MctsConfig(
        sim_num=args.sims,
        timeout=args.timeout,
        c_uct=task["c_uct"],
        random_seed=task["seed"],
        is_mean=task["is_mean"],
        llm_prior=None,
        param_selector=ParamSelectionConfig(enabled=False),
    )
    start = time.time()
    deadline = start + args.max_hours * 3600 if args.max_hours else None
    run = DiverseLinearSearch(
        config,
        benches,
        LOGIC,
        args.z3,
        "par10",
        out,
        batch_size=args.workers,
        logic_config=load_logic_config(LOGIC),
        random_frac=args.random_frac,
        deadline=deadline,
    )
    if not args.debug_trace:
        run.trace_log.setLevel(logging.INFO)

    meta = {
        **{k: str(v) if isinstance(v, Path) else v for k, v in task.items()},
        "benchmarks": len(benches),
        "sims": args.sims,
        "timeout": args.timeout,
        "random_frac": args.random_frac,
        "workers": args.workers,
        "z3_version": subprocess.check_output([args.z3, "--version"], text=True).strip(),
        "finished": False,
    }
    (out / "run.json").write_text(json.dumps(meta, indent=2))

    run.start()

    elapsed = time.time() - start
    meta.update(finished=True, sims_done=run.sims_done, distinct_strategies=len(run.res_database),
                elapsed_s=round(elapsed), best_strategy=run.get_best_strat())
    (out / "run.json").write_text(json.dumps(meta, indent=2))
    log.info(f"Done: {len(run.res_database)} distinct strategies x {len(benches)} benchmarks "
             f"in {run.sims_done} simulations, {elapsed / 3600:.2f}h")


if __name__ == "__main__":
    main()
