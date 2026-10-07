#!/usr/bin/env bash
# Submit the whole generation pipeline as a Slurm dependency chain (run on a login node).
# The 2h QOS allows only 10 submitted jobs per user (counting all of them), so the arrays use the normal QOS.
set -euo pipefail
cd ~/tactics-gen
mkdir -p experiments/slurm
G=scripts/generate_data.sh
O=experiments/slurm/%x-%j.out

J1=$(sbatch --parsable -J gen-sample -c 32 --mem-per-cpu=2G -t 1:30:00 --qos=2h -o $O $G sample)
J2=$(sbatch --parsable -J gen-blocks -d afterok:$J1 -c 1 --mem-per-cpu=2G -t 0:15:00 --qos=2h -o $O $G blocks)
# About 90 tasks expected; tasks past the end of data/stage1_tasks.csv exit immediately.
J3=$(MAX_HOURS=3 sbatch --parsable -J gen-stage1 -d afterok:$J2 --array=0-99%60 -c 8 --mem-per-cpu=3G -t 3:30:00 $G stage1)
# afterany: a few failed stage-1 tasks should not block the rest of the pipeline.
J4=$(sbatch --parsable -J gen-crossplan -d afterany:$J3 -c 1 --mem-per-cpu=8G -t 0:30:00 --qos=2h -o $O $G cross-plan)
J5=$(NUM_SHARDS=10 MEMORY_MB=2500 sbatch --parsable -J gen-cross -d afterok:$J4 --array=0-9 -c 48 --mem-per-cpu=3G -t 3:00:00 $G cross)
J6=$(sbatch --parsable -J gen-merge -d afterany:$J5 -c 2 --mem-per-cpu=8G -t 0:45:00 --qos=2h -o $O $G merge)

echo "sample=$J1 blocks=$J2 stage1=$J3 crossplan=$J4 cross=$J5 merge=$J6" | tee experiments/slurm/chain.txt
