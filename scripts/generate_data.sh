#!/usr/bin/env bash
#SBATCH --job-name=gen-data
#SBATCH --partition=scc-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=2G
#SBATCH --time=08:00:00
#SBATCH --output=experiments/slurm/%x-%A_%a.out

# Training-data generation for the tactic transformer: many small z3alpha stage-1 runs (see generation/).
# Works under sbatch and locally; extra arguments are passed on to the Python script.
#
#   scripts/generate_data.sh sample [args]      pick + probe the QF_NIA subset   (generation/sample_benchmarks.py)
#   scripts/generate_data.sh blocks [args]      cut it into blocks, write tasks  (generation/make_blocks.py)
#   scripts/generate_data.sh stage1 [args]      one MCTS run per array task      (generation/run_stage1.py)
#   scripts/generate_data.sh cross-plan [args]  assign cross-eval pairs, once    (generation/cross_eval.py plan)
#   scripts/generate_data.sh cross [args]       one cross-eval shard per task    (generation/cross_eval.py run)
#   scripts/generate_data.sh merge [args]       all.csv, best.csv and stats      (generation/merge_labels.py)
#
# On the cluster (submit from the repo root; Slurm needs the log directory to exist):
#   mkdir -p experiments/slurm
#   sbatch -c 32 -t 2:00:00 scripts/generate_data.sh sample
#   scripts/generate_data.sh blocks                         # prints the --array range
#   sbatch --array=0-1 scripts/generate_data.sh stage1      # calibrate, then the rest of the range
#   scripts/generate_data.sh cross-plan
#   sbatch --array=0-19 -c 32 scripts/generate_data.sh cross
#   scripts/generate_data.sh merge
# Locally, set the task/shard yourself: TASK_ID=0 scripts/generate_data.sh stage1
#
# Environment: SIMS (300), TIMEOUT (10), MAX_HOURS (7.5, keep below --time), WORKERS (allocated cores),
# MEMORY_MB (4000, cross-eval z3 limit), NUM_SHARDS (array size), VENV (venv).

set -euo pipefail

CMD="${1:?usage: $0 sample|blocks|stage1|cross-plan|cross|merge [args]}"
shift

cd "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")/..}"   # repo root

SIMS="${SIMS:-300}"
TIMEOUT="${TIMEOUT:-10}"
MAX_HOURS="${MAX_HOURS:-7.5}"
WORKERS="${WORKERS:-${SLURM_CPUS_PER_TASK:-$(nproc)}}"
MEMORY_MB="${MEMORY_MB:-4000}"
VENV="${VENV:-venv}"

# No module load: the venv is built on the system python3.12, which every node has.
source "${VENV}/bin/activate"
# z3alpha calls "z3" from PATH; the venv's z3-solver ships the binary.
command -v z3 >/dev/null || { echo "z3 binary not found in PATH" >&2; exit 1; }
export PYTHONUNBUFFERED=1

task_id() { echo "${SLURM_ARRAY_TASK_ID:-${TASK_ID:?set TASK_ID when running outside a Slurm array}}"; }

case "${CMD}" in
    sample)
        python generation/sample_benchmarks.py --workers "${WORKERS}" "$@" ;;
    blocks)
        python generation/make_blocks.py "$@" ;;
    stage1)
        python generation/run_stage1.py --task-id "$(task_id)" --sims "${SIMS}" --timeout "${TIMEOUT}" \
            --max-hours "${MAX_HOURS}" --workers "${WORKERS}" "$@" ;;
    cross-plan)
        python generation/cross_eval.py plan "$@" ;;
    cross)
        python generation/cross_eval.py run --shard "$(task_id)" \
            --num-shards "${NUM_SHARDS:-${SLURM_ARRAY_TASK_COUNT:?set NUM_SHARDS}}" \
            --timeout "${TIMEOUT}" --workers "${WORKERS}" --memory-mb "${MEMORY_MB}" "$@" ;;
    merge)
        python generation/merge_labels.py "$@" ;;
    *)
        echo "unknown command: ${CMD}" >&2; exit 1 ;;
esac
