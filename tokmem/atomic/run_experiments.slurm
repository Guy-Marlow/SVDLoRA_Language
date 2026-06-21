#!/bin/bash
#SBATCH --job-name=svdlora_lang_1000t
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:2
#SBATCH --open-mode=append
#SBATCH --output=/data/gmar762/research/continuous_learning/svd_sketching_language/tokmem/atomic/run_logs/slurm_experiments_%j.out
#SBATCH --error=/data/gmar762/research/continuous_learning/svd_sketching_language/tokmem/atomic/run_logs/slurm_experiments_%j.err
#
# Paper-worthy language CL comparison on Super-NaturalInstructions atomic recall.
#   4 methods x 3 seeds = 12 runs, 1000 tasks (500 train / 50 test, 1 epoch).
#   Methods: seqlora, svdlora (fixed r_hat=8), svdlora (adaptive eps=0.005), olora.
#   3 seeds (1993/1994/1995) each define a FIXED task ORDER (--order_seed); the 1000-task
#   SET is constant across all runs, only the order varies. Same order across methods.
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

COMMON="--num_tasks 1000 --train_size 500 --val_size 10 --test_size 50 \
  --model_name meta-llama/Llama-3.2-3B-Instruct --num_epochs 1 --batch_size 4 \
  --max_length 1024 --max_instruction_tokens 1024 --eval_batch_size 16 \
  --lr 5e-5 --lora_r 8 --lora_alpha 32"

# label | method-specific args  (label only affects the per-run log filename)
RUNS=(
  "seqlora|--method seqlora"
  "svdlora_fixed|--method svdlora --svd_rank 8"
  "svdlora_adapt|--method svdlora --svd_energy_target 0.005"
  "olora|--method olora"
)

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
echo "ALL 12 RUNS COMPLETE $(date)"
