#!/bin/bash
#SBATCH --job-name=lh750v2
#SBATCH --time=120:00:00
#SBATCH --gres=gpu:2
#SBATCH --open-mode=append
#SBATCH --output=/data/gmar762/research/continuous_learning/svd_sketching_language/tokmem/atomic/run_logs/longhorizon/slurm_lh750v2_%j.out
#SBATCH --error=/data/gmar762/research/continuous_learning/svd_sketching_language/tokmem/atomic/run_logs/longhorizon/slurm_lh750v2_%j.err
#
# Long-horizon CL on Super-NaturalInstructions, 750 tasks, follow-up sweep (SVDLoRA + SeqLoRA only).
#   Same fixed 750-task SET (set_seed 0 -> identical SET every run; order_seed permutes per seed).
#   Cumulative eval every 50 tasks on ALL tasks seen so far (test_size 50/task) -> accuracy-vs-
#   tasks-seen curve + per-task x checkpoint matrix + per-checkpoint compute metrics.
#
#   Metrics recorded per checkpoint (every 50 tasks): rougeL, exact_match, deployed_rank_total,
#   adapter_mb, inference_flops_per_token, train_seconds_cumulative, train_seconds_interval,
#   eval_seconds, inference_seconds_per_example, inference_ms_per_example, peak_vram_mb, per_task.
#   Plus a per-task (== per-epoch, num_epochs=1) wall-clock list meta["per_task_train_seconds"],
#   and final meta["train_seconds_total"]. (Logging identical to the prior H200 grid + the two
#   new timing fields requested: per-epoch train wall-clock and per-example inference time.)
#
#   24 runs = methods {svdlora(adapt eps=0.005), seqlora, olora(lamda_1=0.5)} x boundary
#   {task, sample}, lr 5e-5:
#     * Qwen2.5-0.5B-Instruct, seed 1995 ONLY (1993/1994 already done last time; 1995 didn't
#       finish)            -> 3 methods x 2 modes            =  6 runs
#     * Llama-3.2-3B-Instruct, seeds 1993/1994/1995 (all)    -> 3 seeds x 3 methods x 2 modes = 18 runs
#
#   boundary task   = boundary at each task end (learner sees task structure);
#   boundary sample = boundary every 2.5*train_size = 1250 streamed samples (learner UNAWARE of
#                     task ends). Eval still at 50-TASK increments in both modes.
#
# HARD CAP: 2 GPUs, 120h. This script:
#   * runs TWO runs concurrently, one pinned per allocated GPU (a 2-slot pool);
#   * is RESUMABLE -- SKIPS any run whose metrics JSON already says "status": "done", so
#     re-submitting the SAME script (a second allocation) completes the remainder;
#   * orders priority-first then SEED-MAJOR (Qwen-1995, then Llama 1993, 1994, 1995) so the
#     unfinished Qwen run and then a COMPLETE single-seed Llama sweep land first if cut off.
#   * NOTE: the run tag (method_lr_mode_750t_sSEED) does NOT encode the model, so Llama writes
#     into its own out_dir subdir (run_logs/longhorizon/llama3b) to avoid colliding with the
#     Qwen JSONs in run_logs/longhorizon. Qwen 1993/1994 JSONs there are preserved (skip-if-done
#     ignores them since they're not in the grid; 1995 re-runs because it's not marked done).
# Each run writes its own JSON/.out incrementally, so a wall kill loses only in-flight runs.

ATOMIC=/data/gmar762/research/continuous_learning/svd_sketching_language/tokmem/atomic
cd "$ATOMIC"
mkdir -p run_logs/longhorizon run_logs/longhorizon/llama3b

source /data/gmar762/anaconda3/etc/profile.d/conda.sh
conda activate treelora
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_DEVICE_ORDER=PCI_BUS_ID

LR=5e-5
COMMON="--num_tasks 750 --min_instances 560 --train_size 500 --val_size 10 --test_size 50 \
  --num_epochs 1 --batch_size 4 --max_length 1024 --max_instruction_tokens 1024 \
  --lora_r 8 --lora_alpha 32 --target_modules q_proj,v_proj \
  --eval_every_tasks 50 --boundary_samples_mult 2.5 --svd_energy_target 0.005 \
  --lamda_1 0.5 --lr ${LR}"

# Build the 24-run grid. Priority-first (unfinished Qwen 1995), then Llama SEED-MAJOR.
# Entry format: MODELKEY|METHOD|MODE|SEED
RUNS=()
for METHOD in svdlora seqlora olora; do
  for MODE in task sample; do
    RUNS+=("qwen|${METHOD}|${MODE}|1995")
  done
done
for SEED in 1993 1994 1995; do
  for METHOD in svdlora seqlora olora; do
    for MODE in task sample; do
      RUNS+=("llama|${METHOD}|${MODE}|${SEED}")
    done
  done
done

# the orchestrator tags lr as Python str(float(lr)) -> "5e-5" becomes "5e-05". Mirror EXACTLY
# so the resume/skip check finds finished runs.
LRPY="$(python -c "print(float('${LR}'))")"

run_one() {
  local gpu="$1" entry="$2"
  local MODELKEY METHOD MODE SEED MODEL OUTDIR EBS LABEL JSON
  MODELKEY="${entry%%|*}"; entry="${entry#*|}"
  METHOD="${entry%%|*}";   entry="${entry#*|}"
  MODE="${entry%%|*}";     SEED="${entry#*|}"
  if [ "$MODELKEY" = "qwen" ]; then
    MODEL="Qwen/Qwen2.5-0.5B-Instruct"; OUTDIR="run_logs/longhorizon";        EBS=32
  else
    MODEL="meta-llama/Llama-3.2-3B-Instruct"; OUTDIR="run_logs/longhorizon/llama3b"; EBS=16
  fi
  LABEL="${MODELKEY}_${METHOD}_lr${LR}_${MODE}_s${SEED}"
  JSON="${OUTDIR}/longhorizon_${METHOD}_lr${LRPY}_${MODE}_750t_s${SEED}.json"
  if [ -f "$JSON" ] && grep -q '"status": "done"' "$JSON"; then
    echo "==== SKIP (done) ${LABEL} $(date) ===="
    return 0
  fi
  echo "==== START [gpu ${gpu}] ${LABEL} $(date) ===="
  CUDA_VISIBLE_DEVICES="${gpu}" python main_cl_longhorizon.py $COMMON \
    --model_name "${MODEL}" --eval_batch_size "${EBS}" --out_dir "${OUTDIR}" \
    --method "${METHOD}" --boundary_mode "${MODE}" \
    --order_seed "${SEED}" --seed "${SEED}" \
    > "run_logs/longhorizon/${LABEL}.out" 2>&1 || true
  echo "==== DONE  [gpu ${gpu}] ${LABEL} $(date) ===="
}

# 2-slot pool: GPU 0 takes even-index entries, GPU 1 odd-index. Each worker runs its half
# sequentially; `wait` blocks until the 120h wall (or both finish).
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