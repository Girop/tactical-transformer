"""Run z3alpha's stage-1 (linear-only) MCTS strategy search on a chosen SMT-LIB benchmark set.

Stage 1 searches over *linear* tactic sequences only (``Then(t1, t2, ..., solver)``,
no branching). This script runs stage 1 in isolation -- it never invokes z3alpha's
stage-2 branched search -- and shortlists the best linear strategies it found.

Example:
    python generate_linear_tactics.py \\
        --benchmark-dir smtlib/non-incremental/QF_NIA \\
        --logic QF_NIA \\
        --mcts-sims 200 \\
        --timeout 10

Output (written under --out-dir/out-<timestamp>/):
    linear_strategy_summary.csv      -- per-strategy: id, strategy, n_solved, par2_avg, par10_avg
    linear_strategy_per_benchmark.csv -- per-instance outcome for every tried strategy
    linear_selected_strategies.csv   -- the --max-ln-strategies shortlist (the actual "tactics" output)
    linear_strategy_mcts.log          -- MCTS trace
"""

import argparse
import datetime
import logging
from pathlib import Path

from z3alpha.config import (
    DEFAULT_C_UCT,
    DEFAULT_RANDOM_SEED,
    ExperimentConfig,
    SynthesisRun,
    resolve_mcts_config,
    setup_logging,
)
from z3alpha.config.env import check_z3_version, load_env_config
from z3alpha.synthesize import synthesize_linear_strategies

log = logging.getLogger(__name__)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark-dir",
        required=True,
        help="Directory of .smt2 files to search over, e.g. smtlib/non-incremental/QF_NIA",
    )
    parser.add_argument(
        "--logic",
        required=True,
        help="SMT-LIB logic name; must match a z3alpha/z3alpha/tactics/logic_configs/<LOGIC>.json",
    )
    parser.add_argument(
        "--timeout", type=int, default=10, help="Per-benchmark solver timeout in seconds (default: 10)"
    )
    parser.add_argument(
        "--mcts-sims", type=int, default=200, help="Number of stage-1 MCTS simulations (default: 200)"
    )
    parser.add_argument(
        "--max-ln-strategies",
        type=int,
        default=10,
        help="Number of top linear strategies to shortlist into linear_selected_strategies.csv (default: 10)",
    )
    parser.add_argument(
        "--value-type", default="par10", help="Reward metric for MCTS backup (default: par10)"
    )
    parser.add_argument("--c-uct", type=float, default=DEFAULT_C_UCT, help="PUCT exploration constant")
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED, help="Random seed")
    parser.add_argument(
        "--logic-config-dir",
        default=None,
        help="Override directory of tactic/parameter JSON configs (default: z3alpha's built-in configs)",
    )
    parser.add_argument(
        "--out-dir",
        default="experiments/linear_tactics",
        help="Parent directory for timestamped output logs/CSVs (default: experiments/linear_tactics)",
    )
    parser.add_argument(
        "--llm-prior",
        action="store_true",
        help="Score legal tactics with an LLM as PUCT priors (needs OPENAI_API_KEY)",
    )
    parser.add_argument("--llm-model", default="gpt-5.4-mini")
    parser.add_argument("--llm-base-url", default=None)
    parser.add_argument("--llm-timeout", type=float, default=None)
    parser.add_argument("--llm-temperature", type=float, default=None)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = get_args()

    env = load_env_config()
    check_z3_version(env)
    setup_logging(level=args.log_level)

    experiment = ExperimentConfig(
        logic=args.logic,
        train_dir=args.benchmark_dir,
        timeout=args.timeout,
        mcts_sims=args.mcts_sims,
        branched_sims=0,
        max_ln_strategies=args.max_ln_strategies,
        value_type=args.value_type,
        logic_config_dir=args.logic_config_dir,
    )
    mcts_config = resolve_mcts_config(args, experiment)
    run = SynthesisRun(experiment=experiment, mcts=mcts_config)

    log_folder = Path(args.out_dir) / f"out-{datetime.datetime.now():%Y-%m-%d_%H-%M-%S}"
    log_folder.mkdir(parents=True)

    _, shortlist = synthesize_linear_strategies(run, log_folder, env=env)

    log.info("Stage-1 linear tactic search complete; results in %s", log_folder)
    for strat, _ in shortlist:
        log.info("  %s", strat)


if __name__ == "__main__":
    main()
