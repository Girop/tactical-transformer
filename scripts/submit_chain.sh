#!/usr/bin/env bash
# Submit the whole generation pipeline as a Slurm dependency chain (run on a login node).
# The 2h QOS allows only 10 submitted jobs per user (counting all of them), so the arrays use the normal QOS.
set -euo pipefail
cd ~/tactics-gen
mkdir -p experiments/slurm
G=scripts/generate_data.sh
O=experiments/slurm/%x-%j.out

J1=$(sbatch --parsable -J gen-prepare  -c 32 --mem-per-cpu=2G -t 1:30:00 --qos=2h -o $O $G prepare)
# About 90 tasks expected; tasks past the end of data/stage1_tasks.csv exit immediately.
J2=$(sbatch --parsable -J gen-stage1 -d afterok:$J1  --array=0-99%60 -c 8  --mem-per-cpu=3G -t 3:30:00 $G stage1)
# afterany: a few failed stage-1 tasks should not block merge.
J3=$(sbatch --parsable -J gen-merge   -d afterany:$J2 -c 2  --mem-per-cpu=8G -t 0:45:00 --qos=2h -o $O $G merge)

echo "prepare=$J1 stage1=$J2 merge=$J3" | tee experiments/slurm/chain.txt
