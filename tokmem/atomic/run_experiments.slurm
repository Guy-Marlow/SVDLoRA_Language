#!/bin/bash
#SBATCH --job-name=svdlora_lang_1000t
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:2
#SBATCH --open-mode=append
#SBATCH --output=/data/gmar762/research/continuous_learning/svd_sketching_language/tokmem/atomic/run_logs/slurm_experiments_%j.out
#SBATCH --error=/data/gmar762/research/continuous_learning/svd_sketching_language/tokmem/atomic/run_logs/slurm_experiments_%j.err
#
# Paper-worthy language CL comparison on Super-NaturalInstructions atomic recall.
#   4 methods x 3 seeds = 12 runs, 400 tasks (500 train / 50 test, 1 epoch).
#   Methods: seqlora, svdlora (adaptive eps=0.005), olora, inflora.
#   3 seeds (1993/1994/1995) each define a FIXED task ORDER (--order_seed); the 400-task
#   SET is constant across all runs, only the order varies. Same order across methods.
#   SEED-MAJOR: seed N's methods are submitted (and so run) before seed N+1's, so you get
#   every method for >=1 ordering early. Logs stream live and are readable mid-run.
#
# We request 2 GPUs (defensive: if the cluster's busted card is one of them, the job still
# runs on the good one). Work is launched as SLURM job steps and SLURM decides placement --
# we do NOT pin CUDA_VISIBLE_DEVICES. With 2 good GPUs, 2 runs go at once; with 1, sequentially.
# Each run is wrapped so a crash/OOM cannot abort the whole job (the harness also OOM-guards
# internally and exits 0, writing partial metrics).
#
# NOTE: create run_logs/ before submitting (SBATCH opens --output before the script runs):
#   mkdir -p .../tokmem/atomic/run_logs   &&   sbatch run_experiments.slurm

ATOMIC=/data/gmar762/research/continuous_learning/svd_sketching_language/tokmem/atomic
cd "$ATOMIC"
mkdir -p run_logs

source /data/gmar762/anaconda3/etc/profile.d/conda.sh
conda activate treelora
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

COMMON="--num_tasks 400 --train_size 500 --val_size 10 --test_size 50 \
  --model_name meta-llama/Llama-3.2-3B-Instruct --num_epochs 1 --batch_size 4 \
  --max_length 1024 --max_instruction_tokens 1024 --eval_batch_size 16 \
  --lr 5e-5 --lora_r 8 --lora_alpha 32"

# label | method-specific args  (label only affects the per-run log filename)
RUNS=(
  "seqlora|--method seqlora"
  "svdlora_adapt|--method svdlora --svd_energy_target 0.005"
  "olora|--method olora"
  "inflora|--method inflora"
)

# SEED-MAJOR submission: all of seed 1993's methods are submitted before 1994's, etc., so
# SLURM (FIFO across the allocated GPUs) finishes config 1993's methods first -> you get every
# method for >=1 ordering ASAP. No barrier between seeds: if a GPU frees, the next config's
# method may start early (fine). Each run streams to its own log, readable live without
# interrupting anything.
for SEED in 1993 1994 1995; do
  for entry in "${RUNS[@]}"; do
    LABEL="${entry%%|*}"
    EXTRA="${entry#*|}"
    LOG="run_logs/exp_${LABEL}_${SEED}.out"
    # one SLURM job step per run; SLURM schedules these across the allocated GPUs.
    srun --exclusive --gres=gpu:1 --ntasks=1 bash -lc "
      cd '$ATOMIC'
      conda activate treelora 2>/dev/null || true
      export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
      echo \"==== START ${LABEL} seed ${SEED} \$(date) ====\"
      python main_svdlora_baseline.py $COMMON --order_seed ${SEED} --seed ${SEED} ${EXTRA} > '$LOG' 2>&1 || true
      echo \"==== DONE  ${LABEL} seed ${SEED} \$(date) ====\"
      true
    " &
  done
done
wait
echo "ALL RUNS COMPLETE $(date)"
