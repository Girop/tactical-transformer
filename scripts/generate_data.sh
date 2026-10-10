#!/usr/bin/env bash
#SBATCH --job-name=gen-data
#SBATCH --partition=scc-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=2G
#SBATCH --time=08:00:00
#SBATCH --output=experiments/slurm/%x-%A_%a.out

# Training-data generation for the tactic transformer (see src/generate.py).
# Works under sbatch and locally; extra arguments are passed on to the Python script.
#
#   scripts/generate_data.sh prepare [args]  discover benchmarks, make blocks + task table
#   scripts/generate_data.sh stage1  [args]  one MCTS run per array task
#   scripts/generate_data.sh merge   [args]  all.csv, solved.csv and stats
#
# On the cluster (submit from the repo root; Slurm needs the log directory to exist):
#   mkdir -p experiments/slurm
#   sbatch -c 32 -t 2:00:00 scripts/generate_data.sh prepare
#   sbatch --array=0-99%60 scripts/generate_data.sh stage1
#   scripts/generate_data.sh merge
# Locally, set the task/shard yourself: TASK_ID=0 scripts/generate_data.sh stage1
#
# Environment: SIMS (300), TIMEOUT (10), WORKERS (allocated cores), VENV (venv).

set -euo pipefail

CMD="${1:?usage: $0 prepare|stage1|merge [args]}"
shift

cd "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")/..}"   # repo root

SIMS="${SIMS:-300}"
TIMEOUT="${TIMEOUT:-10}"
WORKERS="${WORKERS:-${SLURM_CPUS_PER_TASK:-$(nproc)}}"
VENV="${VENV:-venv}"

# No module load: the venv is built on the system python3.12, which every node has.
source "${VENV}/bin/activate"
# z3alpha calls "z3" from PATH; the venv's z3-solver ships the binary.
command -v z3 >/dev/null || { echo "z3 binary not found in PATH" >&2; exit 1; }
export PYTHONUNBUFFERED=1

task_id() { echo "${SLURM_ARRAY_TASK_ID:-${TASK_ID:?set TASK_ID when running outside a Slurm array}}"; }

case "${CMD}" in
    prepare)
        python src/generate.py prepare "$@" ;;
    stage1)
        python src/generate.py stage1 --task-id "$(task_id)" --sims "${SIMS}" \
            --timeout "${TIMEOUT}" --workers "${WORKERS}" "$@" ;;
    merge)
        python src/generate.py merge "$@" ;;
    *)
        echo "unknown command: ${CMD}" >&2; exit 1 ;;
esac
