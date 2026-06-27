#!/bin/bash
#SBATCH --job-name=lh750
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --open-mode=append
#SBATCH --array=0-35%12
#SBATCH --output=/data/gmar762/research/continuous_learning/svd_sketching_language/tokmem/atomic/run_logs/longhorizon/slurm_lh750_%A_%a.out
#SBATCH --error=/data/gmar762/research/continuous_learning/svd_sketching_language/tokmem/atomic/run_logs/longhorizon/slurm_lh750_%A_%a.err
#
# Long-horizon CL on Super-NaturalInstructions, Qwen2.5-0.5B, 750 tasks.
#   Fixed 750-task SET drawn from the 766 English tasks with >=560 instances (min_instances 560,
#   set_seed 0 -> identical task SET for every run); ORDER permuted per seed (order_seed) to show
#   robustness to task ordering. Cumulative eval every 50 tasks on ALL tasks seen so far
#   (test 50/task) -> accuracy-vs-tasks-seen curve + full task x checkpoint ROUGE matrix.
#
#   6 configs x 3 seeds x 2 boundary modes = 36 runs (one per array index 0..35):
#     methods : seqlora (fresh optimizer per boundary), svdlora (adaptive eps=0.005 sketch),
#               olora (per-task slot bank, orthogonality lamda_1=0.5)
#     lrs     : 5e-5, 8e-5
#     modes   : task   = boundary at every task end (standard; boundaries are GIVEN)
#               sample = boundary every 2.5*train_size = 1250 streamed samples, regardless of
#                        task identity (learner does NOT know a task ended) -- tests whether
#                        SVDLoRA degrades more gracefully than seqlora/olora without given
#                        task boundaries. Eval still at 50-TASK increments (experimenter knows).
#
# One run per GPU per array task (independent), %12 = at most 12 concurrent. Mirrors the
# cluster's other comparison jobs: request gpu ONCE, plain `python`, no srun / CVD juggling.

ATOMIC=/data/gmar762/research/continuous_learning/svd_sketching_language/tokmem/atomic
cd "$ATOMIC"
mkdir -p run_logs/longhorizon

source /data/gmar762/anaconda3/etc/profile.d/conda.sh
conda activate treelora
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

COMMON="--num_tasks 750 --min_instances 560 --train_size 500 --val_size 10 --test_size 50 \
  --model_name Qwen/Qwen2.5-0.5B-Instruct --num_epochs 1 --batch_size 4 \
  --max_length 1024 --max_instruction_tokens 1024 --eval_batch_size 32 \
  --lora_r 8 --lora_alpha 32 --target_modules q_proj,v_proj \
  --eval_every_tasks 50 --boundary_samples_mult 2.5 \
  --svd_energy_target 0.005 --lamda_1 0.5 --out_dir run_logs/longhorizon"

# Build the 36-run grid deterministically: mode x lr x method x seed.
RUNS=()
for MODE in task sample; do
  for LR in 5e-5 8e-5; do
    for METHOD in seqlora svdlora olora; do
      for SEED in 1993 1994 1995; do
        RUNS+=("${MODE}|${LR}|${METHOD}|${SEED}")
      done
    done
  done
done

ENTRY="${RUNS[$SLURM_ARRAY_TASK_ID]}"
MODE="$(echo "$ENTRY" | cut -d'|' -f1)"
LR="$(echo   "$ENTRY" | cut -d'|' -f2)"
METHOD="$(echo "$ENTRY" | cut -d'|' -f3)"
SEED="$(echo "$ENTRY" | cut -d'|' -f4)"

LABEL="${METHOD}_lr${LR}_${MODE}_s${SEED}"
echo "==== START [${SLURM_ARRAY_TASK_ID}] ${LABEL} $(date) ===="
python main_cl_longhorizon.py $COMMON \
  --method "${METHOD}" --lr "${LR}" --boundary_mode "${MODE}" \
  --order_seed "${SEED}" --seed "${SEED}" \
  > "run_logs/longhorizon/${LABEL}.out" 2>&1
echo "==== DONE  [${SLURM_ARRAY_TASK_ID}] ${LABEL} $(date) ===="
