#!/usr/bin/env bash
#SBATCH --job-name=train
#SBATCH --partition=scc-gpu
#SBATCH -G A100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=2-00:00:00
#SBATCH --output=experiments/slurm/%x-%j.out

# Train + test the tactic transformer. Submit from the repo root: sbatch scripts/train.sbatch [--data DIR] [--sample-count N]
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")/..}"
source "${VENV:-venv}/bin/activate"
command -v z3 >/dev/null || { echo "z3 binary not found in PATH" >&2; exit 1; }
export PYTHONUNBUFFERED=1
nvidia-smi -L
python src/main.py --data experiments/train "$@"
