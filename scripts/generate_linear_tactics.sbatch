#!/bin/bash
#SBATCH --job-name=linear-tactics
#SBATCH --partition=scc-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=2G
#SBATCH --time=12:00:00
#SBATCH --output=experiments/slurm/%x-%j.out
#SBATCH --mail-type=END,FAIL

# Stage-1 (linear-only) z3alpha strategy search on GWDG's scc-cpu partition.
#
# One-time setup on the cluster (log in to glogin-p3, where scc-cpu lives):
#   git clone --recurse-submodules <this repo> && cd tactics
#   module load python/3.12
#   ./setup.sh                               # venv, pinned deps, QF_NIA benchmarks
#
# Submit from the repo root (paths below are relative to it):
#   sbatch scripts/generate_linear_tactics.sbatch
# Override search parameters via environment variables, e.g.:
#   LOGIC=QF_NIA MCTS_SIMS=500 TIMEOUT=20 sbatch scripts/generate_linear_tactics.sbatch
# Change cores/walltime on the command line (flags override #SBATCH lines):
#   sbatch -c 64 -t 24:00:00 scripts/generate_linear_tactics.sbatch

set -euo pipefail

LOGIC="${LOGIC:-QF_NIA}"
BENCHMARK_DIR="${BENCHMARK_DIR:-smtlib/non-incremental/${LOGIC}}"
MCTS_SIMS="${MCTS_SIMS:-200}"
TIMEOUT="${TIMEOUT:-10}"
MAX_LN_STRATEGIES="${MAX_LN_STRATEGIES:-10}"
SEED="${SEED:-}"
OUT_DIR="${OUT_DIR:-experiments/linear_tactics}"
VENV="${VENV:-venv}"

cd "${SLURM_SUBMIT_DIR}"

module purge
module load python/3.12
source "${VENV}/bin/activate"

# z3alpha resolves "z3" from PATH; the venv's z3-solver ships the binary.
command -v z3 >/dev/null || { echo "z3 binary not found in PATH" >&2; exit 1; }

echo "Job ${SLURM_JOB_ID} on $(hostname): ${LOGIC}, ${MCTS_SIMS} sims," \
     "${TIMEOUT}s timeout, ${SLURM_CPUS_PER_TASK} workers"

# One z3 process per allocated core.
python -u generate_linear_tactics.py \
    --benchmark-dir "${BENCHMARK_DIR}" \
    --logic "${LOGIC}" \
    --mcts-sims "${MCTS_SIMS}" \
    --timeout "${TIMEOUT}" \
    --max-ln-strategies "${MAX_LN_STRATEGIES}" \
    --workers "${SLURM_CPUS_PER_TASK}" \
    --out-dir "${OUT_DIR}" \
    ${SEED:+--random-seed "${SEED}"}
