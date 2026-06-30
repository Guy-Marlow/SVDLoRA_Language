#!/bin/bash
#SBATCH --job-name=epshoriz
#SBATCH --time=120:00:00
#SBATCH --gres=gpu:2
#SBATCH --open-mode=append
#SBATCH --output=/data/gmar762/research/continuous_learning/svd_sketching_language/tokmem/atomic/run_logs/sensitivity_tasks_llama3b/slurm_epshoriz_%j.out
#SBATCH --error=/data/gmar762/research/continuous_learning/svd_sketching_language/tokmem/atomic/run_logs/sensitivity_tasks_llama3b/slurm_epshoriz_%j.err
#
# Adaptive-SVDLoRA epsilon x horizon sensitivity -- Llama-3.2-3B-Instruct, seed 1993.
#   Same EVAL SCHEME and METRICS as the three 100-task A100 seeds (main_svdlora_baseline.py):
#   train all N tasks, then ONE final pooled greedy-generation eval over every task's test set
#   (NI ROUGE-L). Metrics per run (via --sketch_cost_probe): rougeL, exact_match, adapter mem MB,
#   inference FLOPs/token, deployed_rank (nominal + SVD-measured numerical rank), sketch_cost
#   (ideal-vs-sketched ROUGE drop), peak_vram_mb, per-task JSON. Identical flags to the 100-task
#   runs except num_tasks varies.
#
#   15 runs = num_tasks {250, 500, 750} x energy_target {0.005, 0.0075, 0.01, 0.0125, 0.015}.
#   Ordered num_tasks-MAJOR (the lighter 250-task sweep -- a complete horizon -- lands first;
#   then 500, then 750) so a wall/kill still yields whole sub-experiments.
#
#   Results land in a SEPARATE dir (run_logs/sensitivity_tasks_llama3b) -- does NOT touch the
#   100-task run_logs/sensitivity outputs.
#
# HARD CAP: 120h. This script:
#   * is GPU-count-aware -- with NGPU allocated slots, GPU g runs entries g, g+NGPU, ...; on ONE
#     GPU it collapses to a single worker walking all 15 runs one-at-a-time (no co-tenant 3B
#     jobs -> no OOM/contention). Request --gres=gpu:1 (or more) and it adapts; no edit needed.
#   * is RESUMABLE -- SKIPS any run whose metrics JSON already says "status": "done", so
#     re-submitting the SAME script completes the remainder.
# Each run writes its own JSON/.out incrementally, so a wall kill loses only in-flight runs.

ATOMIC=/data/gmar762/research/continuous_learning/svd_sketching_language/tokmem/atomic
cd "$ATOMIC"

source /data/gmar762/anaconda3/etc/profile.d/conda.sh
conda activate treelora
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_DEVICE_ORDER=PCI_BUS_ID

OUT=run_logs/sensitivity_tasks_llama3b
mkdir -p "$OUT"
SEED=1993
MODEL=meta-llama/Llama-3.2-3B-Instruct
TASKS=(250 500 750)
EPS=(0.005 0.0075 0.01 0.0125 0.015)

COMMON="--method svdlora --train_size 500 --val_size 10 --test_size 50 --min_instances 560 \
  --model_name ${MODEL} --num_epochs 1 --batch_size 4 \
  --max_length 1024 --max_instruction_tokens 1024 --eval_batch_size 16 \
  --lr 5e-5 --lora_r 8 --lora_alpha 32 \
  --seed ${SEED} --set_seed ${SEED} --order_seed ${SEED} --sketch_cost_probe --out_dir ${OUT}"

# Build the 15-run grid, num_tasks-major then eps ascending.
RUNS=()
for N in "${TASKS[@]}"; do
  for E in "${EPS[@]}"; do
    RUNS+=("${N}|${E}")
  done
done

run_one() {
  local gpu="$1" entry="$2"
  local N="${entry%%|*}" E="${entry#*|}" EPY JSON
  # mirror main_svdlora_baseline's tag: energy_target is str(float(E)) -> compute it exactly
  EPY="$(python -c "print(float('${E}'))")"
  JSON="$OUT/metrics_svdlora_all_${N}t_s${SEED}_adapt${EPY}.json"
  if [ -f "$JSON" ] && grep -q '"status": "done"' "$JSON"; then
    echo "==== SKIP (done) N=${N} eps=${E} $(date) ====" >> "$OUT/_driver.log"
    return 0
  fi
  echo "==== START [gpu ${gpu}] N=${N} eps=${E} $(date) ====" >> "$OUT/_driver.log"
  CUDA_VISIBLE_DEVICES="${gpu}" python main_svdlora_baseline.py $COMMON \
    --num_tasks "${N}" --svd_energy_target "${E}" \
    > "$OUT/run_${N}t_eps${E}_s${SEED}.out" 2>&1 || true
  echo "==== DONE  [gpu ${gpu}] N=${N} eps=${E} $(date) ====" >> "$OUT/_driver.log"
}

# GPU-count-aware pool. slot g -> GPU g handles entries g, g+NGPU, g+2*NGPU, ... sequentially.
NGPU="${SLURM_GPUS_ON_NODE:-}"
[ -z "$NGPU" ] && NGPU="$(nvidia-smi -L 2>/dev/null | wc -l)"
case "$NGPU" in (''|*[!0-9]*) NGPU=1 ;; esac   # non-numeric/empty -> 1
[ "$NGPU" -lt 1 ] && NGPU=1
echo "==== epshoriz: ${#RUNS[@]} runs on NGPU=${NGPU} slot(s) $(date) ====" >> "$OUT/_driver.log"

worker() {
  local gpu="$1" i
  for (( i=gpu; i<${#RUNS[@]}; i+=NGPU )); do
    run_one "$gpu" "${RUNS[$i]}"
  done
}

for (( g=0; g<NGPU; g++ )); do
  worker "$g" &
done
wait
echo "ALL SENSITIVITY_TASKS_LLAMA3B RUNS COMPLETE $(date)" >> "$OUT/_driver.log"