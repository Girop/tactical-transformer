#!/usr/bin/env bash
# Submit the whole generation pipeline as a Slurm dependency chain (run on a login node).
# The 2h QOS allows only 10 submitted jobs per user (counting all of them), so the arrays use the normal QOS.
set -euo pipefail
cd ~/tactical-transformer
mkdir -p experiments/slurm
G=scripts/generate_data.sh
O=experiments/slurm/%x-%j.out

# Uncovered families (new run uses separate blocks/tasks to avoid clobbering existing ones).
LOGICS="QF_BV QF_NIA QF_NRA"
BENCHMARKS="2017-BuchwaldFried 20170501-Heizmann-UltimateAutomizer 20170531-Hansen-Check
  2018-Goel-hwbench 2018-Mann 2019-Mann 2019-Wolf-fmbench
  20190311-bv-term-small-rw-Noetzli 20190429-UltimateAutomizerSvcomp2019
  2020-Weber 20200328-Favaro 20200415-Yurichev 20210219-Sydr 20210312-Bouvier
  20220315-ecrw 20221012-MCMPC 20221214-p4dfa-XiaoqiChen
  20230221-oisc-gurtner 20230224-grsbits-truby 20230321-UltimateAutomizerSvcomp2023
  RWS VS3 asp bench_ab bmc-bv bmc-bv-svcomp14
  brummayerbiere brummayerbiere2 brummayerbiere3 brummayerbiere4
  calypto challenge check2 crafted dwp_formulas ecc fft float galois
  gulwani-pldi08 log-slicing mcm pipe pspace rubik sage spear
  stp stp_samples tacas07 uclid uclid_contrib_smtcomp09 uum wienand-cav2008
  20170427-VeryMax 2019-ezsmt 20210219-Dartagnan 20220315-MathProblems
  20230328-sqrtmodinv-hoenicke 20250331-elster LCTES LassoRanker
  UltimateAutomizer UltimateLassoRanker
  20161105-Sturm-MBO 20161105-Sturm-MGC 20170501-Heizmann-UltimateInvariantSynthesis
  20180501-Economics-Mulligan 20200911-Pine 20211101-Geogebra 20220314-Uncu
  20240407-pPDA-Chiari-Pontiggia-Winkler hong kissing meti-tarski zankl"

J1=$(sbatch --parsable -J gen-prepare -c 32 --mem-per-cpu=2G -t 1:30:00 --qos=2h -o $O $G prepare \
  --logics $LOGICS --benchmarks $BENCHMARKS --seed 1 \
  --blocks-dir data/blocks-new --tasks data/stage1_tasks_new.csv)
# Tasks past the end of data/stage1_tasks_new.csv exit immediately.
J2=$(sbatch --parsable -J gen-stage1 -d afterok:$J1 --array=0-2000%60 -c 8 --mem-per-cpu=3G -t 3:30:00 $G stage1 \
  --tasks data/stage1_tasks_new.csv)
# afterany: a few failed stage-1 tasks should not block merge.
J3=$(sbatch --parsable -J gen-merge  -d afterany:$J2 -c 2  --mem-per-cpu=8G -t 0:45:00 --qos=2h -o $O $G merge)

echo "prepare=$J1 stage1=$J2 merge=$J3" | tee experiments/slurm/chain.txt
