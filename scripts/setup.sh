#!/usr/bin/env bash
# Usage: ./setup.sh [--logic LOGIC]... [--no-benchmarks] [--tokenizer]
#   --logic LOGIC     SMT-LIB logic to download (repeatable; default: QF_NIA)
#   --no-benchmarks   skip the benchmark download
#   --tokenizer       also train the BPE tokenizer into tokenizer/smtlib-bpe

set -euo pipefail

LOGICS=()
DOWNLOAD_BENCHMARKS=1
TRAIN_TOKENIZER=0
PYTHON="${PYTHON:-python3.12}"
PYTHON_VERSION="3.12"   # requirements.txt is pinned against this version

while [[ $# -gt 0 ]]; do
  case "$1" in
    --logic) LOGICS+=("$2"); shift 2 ;;
    --no-benchmarks) DOWNLOAD_BENCHMARKS=0; shift ;;
    --tokenizer) TRAIN_TOKENIZER=1; shift ;;
    -h|--help) sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "Unknown argument: $1 (see --help)" >&2; exit 1 ;;
  esac
done
[[ ${#LOGICS[@]} -eq 0 ]] && LOGICS=(QF_NIA)

cd "$(dirname "${BASH_SOURCE[0]}")"

step() { echo; echo "==> $*"; }
die() { echo "Error: $*" >&2; exit 1; }

step "Checking prerequisites"
command -v git >/dev/null || die "git not found"
command -v "$PYTHON" >/dev/null \
  || die "$PYTHON not found. Install Python $PYTHON_VERSION, or set PYTHON=... (on GWDG: module load python/$PYTHON_VERSION)"
if [[ $DOWNLOAD_BENCHMARKS -eq 1 ]]; then
  command -v wget >/dev/null || die "wget not found (needed to download benchmarks)"
  command -v zstd >/dev/null || die "zstd not found (needed to unpack benchmarks; on GWDG try: module load zstd)"
fi

step "Initialising z3alpha submodule"
git submodule update --init --recursive

step "Creating virtual environment in venv/"
if [[ -x venv/bin/python ]]; then
  have="$(venv/bin/python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
  [[ "$have" == "$PYTHON_VERSION" ]] \
    || die "venv/ uses Python $have, expected $PYTHON_VERSION. Remove venv/ and re-run."
  echo "venv/ already exists (Python $have)"
else
  "$PYTHON" -m venv venv
fi

step "Installing pinned dependencies"
venv/bin/pip install -r requirements.txt

step "Checking z3 matches z3alpha's env_config.json"
PATH="$PWD/venv/bin:$PATH" venv/bin/python -c \
  'from z3alpha.config.env import load_env_config, check_z3_version; check_z3_version(load_env_config())'
echo "ok"

if [[ $DOWNLOAD_BENCHMARKS -eq 1 ]]; then
  for logic in "${LOGICS[@]}"; do
    step "Benchmarks: $logic"
    if [[ -d "smtlib/non-incremental/$logic" ]]; then
      echo "smtlib/non-incremental/$logic already present, skipping"
    else
      bash scripts/download_smtlib.sh "$logic"
    fi
  done
fi

if [[ $TRAIN_TOKENIZER -eq 1 ]]; then
  step "Training BPE tokenizer"
  if [[ -f tokenizer/smtlib-bpe/tokenizer.json ]]; then
    echo "tokenizer/smtlib-bpe already present, skipping"
  else
    venv/bin/python scripts/train_bpe_tokenizer.py
  fi
fi

mkdir -p experiments/slurm   # Slurm needs the --output dir to exist

step "Done. Activate with: source venv/bin/activate"
