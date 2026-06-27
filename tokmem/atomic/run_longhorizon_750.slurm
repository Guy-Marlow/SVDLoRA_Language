#!/bin/bash
#SBATCH --job-name=lh750
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:2
#SBATCH --open-mode=append
#SBATCH --output=/data/gmar762/research/continuous_learning/svd_sketching_language/tokmem/atomic/run_logs/longhorizon/slurm_lh750_%j.out
#SBATCH --error=/data/gmar762/research/continuous_learning/svd_sketching_language/tokmem/atomic/run_logs/longhorizon/slurm_lh750_%j.err
#
# Long-horizon CL on Super-NaturalInstructions, Qwen2.5-0.5B, 750 tasks.
#   Fixed 750-task SET from the 766 English tasks with >=560 instances (set_seed 0 -> same SET
#   every run; order_seed permutes per seed -> ordering robustness). Cumulative eval every 50
#   tasks on ALL tasks seen so far (test 50/task) -> accuracy-vs-tasks-seen + task x checkpoint
#   matrix + per-checkpoint compute metrics.
#
#   6 configs x 3 seeds x 2 boundary modes = 36 runs:
#     methods seqlora/svdlora(adapt eps=0.005)/olora(lamda_1=0.5); lr {5e-5,8e-5};
#     boundary {task = boundary per task end ; sample = boundary every 2.5*train_size=1250
#     streamed samples, learner unaware of task ends}. Eval still at 50-TASK increments.
#
# HARD CAP: 2 GPUs, 48h. The 36 runs (~3-4h each; eval-heavy at test_size 50) do NOT all fit
# in one 48h job (~96 GPU-h available vs ~110+ needed). So this script:
#   * runs TWO runs concurrently, one pinned to each allocated GPU (a 2-slot pool);
#   * is RESUMABLE -- it SKIPS any run whose metrics JSON already says "status": "done", so
#     re-submitting the SAME script (a second 48h allocation) completes the remainder;
#   * orders SEED-MAJOR (1993, then 1994, then 1995) so a COMPLETE single-seed sweep -- a full
#     standalone experiment -- lands first if the job is cut off at the wall.
# Each run writes its own JSON/.out incrementally, so a wall-clock kill loses only the
# in-flight runs (their partial JSON is kept), never the finished ones.

ATOMIC=/data/gmar762/research/continuous_learning/svd_sketching_language/tokmem/atomic
cd "$ATOMIC"
mkdir -p run_logs/longhorizon

source /data/gmar762/anaconda3/etc/profile.d/conda.sh
conda activate treelora
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_DEVICE_ORDER=PCI_BUS_ID

COMMON="--num_tasks 750 --min_instances 560 --train_size 500 --val_size 10 --test_size 50 \
  --model_name Qwen/Qwen2.5-0.5B-Instruct --num_epochs 1 --batch_size 4 \
  --max_length 1024 --max_instruction_tokens 1024 --eval_batch_size 32 \
  --lora_r 8 --lora_alpha 32 --target_modules q_proj,v_proj \
  --eval_every_tasks 50 --boundary_samples_mult 2.5 \
  --svd_energy_target 0.005 --lamda_1 0.5 --out_dir run_logs/longhorizon"

# Build the 36-run grid SEED-MAJOR (complete one seed before advancing).
RUNS=()
for SEED in 1993 1994 1995; do
  for MODE in task sample; do
    for LR in 5e-5 8e-5; do
      for METHOD in seqlora svdlora olora; do
        RUNS+=("${MODE}|${LR}|${METHOD}|${SEED}")
      done
    done
  done
done

# Run one grid entry on a given GPU, skipping it if already finished.
run_one() {
  local gpu="$1" entry="$2"
  local MODE LR METHOD SEED LABEL JSON LRPY
  MODE="${entry%%|*}"; entry="${entry#*|}"
  LR="${entry%%|*}";   entry="${entry#*|}"
  METHOD="${entry%%|*}"; SEED="${entry#*|}"
  LABEL="${METHOD}_lr${LR}_${MODE}_s${SEED}"
  # the orchestrator's JSON tag formats lr as Python str(float(lr)) -> "5e-5" becomes "5e-05".
  # Mirror that EXACTLY so the resume/skip check finds finished runs.
  LRPY="$(python -c "print(float('${LR}'))")"
  JSON="run_logs/longhorizon/longhorizon_${METHOD}_lr${LRPY}_${MODE}_750t_s${SEED}.json"
  if [ -f "$JSON" ] && grep -q '"status": "done"' "$JSON"; then
    echo "==== SKIP (done) ${LABEL} $(date) ===="
    return 0
  fi
  echo "==== START [gpu ${gpu}] ${LABEL} $(date) ===="
  CUDA_VISIBLE_DEVICES="${gpu}" python main_cl_longhorizon.py $COMMON \
    --method "${METHOD}" --lr "${LR}" --boundary_mode "${MODE}" \
    --order_seed "${SEED}" --seed "${SEED}" \
    > "run_logs/longhorizon/${LABEL}.out" 2>&1 || true
  echo "==== DONE  [gpu ${gpu}] ${LABEL} $(date) ===="
}

# 2-slot pool: GPU 0 takes even-index entries, GPU 1 odd-index. Each worker runs its half
# sequentially in the background; `wait` blocks until the 48h wall (or both finish).
worker() {
  local gpu="$1" start="$2"
  local i
  for (( i=start; i<${#RUNS[@]}; i+=2 )); do
    run_one "$gpu" "${RUNS[$i]}"
  done
}

worker 0 0 &
worker 1 1 &
wait
echo "ALL ASSIGNED RUNS COMPLETE (or wall reached) $(date)"
