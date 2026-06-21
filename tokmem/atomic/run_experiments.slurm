#!/bin/bash
#SBATCH --job-name=svdlora_lang_400t
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --open-mode=append
#SBATCH --output=/data/gmar762/research/continuous_learning/svd_sketching_language/tokmem/atomic/run_logs/slurm_experiments_%j.out
#SBATCH --error=/data/gmar762/research/continuous_learning/svd_sketching_language/tokmem/atomic/run_logs/slurm_experiments_%j.err
#
# Language CL comparison on Super-NaturalInstructions atomic recall.
#   5 methods x 3 seeds = 15 runs, 400 tasks (500 train / 50 test, 1 epoch).
#   Methods: seqlora, svdlora (fixed r_hat=8), svdlora (adaptive eps=0.005), olora, inflora.
#
# GPUs are requested ONCE, here in the #SBATCH header. The job body does NOT allocate GPUs
# (no `srun --gres`): SLURM hands us the allocated device(s) via $CUDA_VISIBLE_DEVICES, and we
# just call python on them. If you bump `--gres=gpu:N` above, the script auto-detects the N
# devices and runs N lanes in parallel; with gpu:1 (default) everything runs sequentially on
# the single allocated GPU.
#
# Ordering: the fast O(1) methods (seqlora, svdlora) run FIRST across all seeds, so usable
# results land in hours; the slow O(K) bank methods (olora, inflora) run LAST. No gradient
# checkpointing -- on H200 (141GB) the un-checkpointed per-step peak (~40GB) fits with huge
# headroom, and skipping recompute is several times faster (the point of the big card).

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

# Devices SLURM allocated to THIS job (it populates CUDA_VISIBLE_DEVICES; default to 0 if unset).
IFS=',' read -ra GPUS <<< "${CUDA_VISIBLE_DEVICES:-0}"
NGPU=${#GPUS[@]}
echo "Allocated GPUs: ${GPUS[*]}  (N=$NGPU)  $(date)"

# fast O(1) methods FIRST (hours each), slow O(K) banks LAST (days each, checkpointed -> no OOM)
RUNS=(
  "seqlora|--method seqlora"
  "svdlora_fixed|--method svdlora --svd_rank 8"
  "svdlora_adapt|--method svdlora --svd_energy_target 0.005"
  "olora|--method olora"
  "inflora|--method inflora"
)
SEEDS=(1993 1994 1995)

# Flat job list, method-major so all fast methods (every seed) precede the banks.
JOBS=()
for entry in "${RUNS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    JOBS+=("${SEED}|${entry}")
  done
done

# One lane per allocated GPU; each lane runs its share of JOBS sequentially on its device.
run_lane () {   # $1 = physical gpu id (a value from CUDA_VISIBLE_DEVICES)   $2 = lane index
  local gpu="$1" lane="$2"
  for idx in "${!JOBS[@]}"; do
    (( idx % NGPU == lane )) || continue
    local job="${JOBS[idx]}"
    local SEED="${job%%|*}"; local rest="${job#*|}"
    local LABEL="${rest%%|*}"; local EXTRA="${rest#*|}"
    echo "==== START ${LABEL} seed ${SEED} on gpu ${gpu} $(date) ===="
    CUDA_VISIBLE_DEVICES="${gpu}" python main_svdlora_baseline.py $COMMON \
      --order_seed "${SEED}" --seed "${SEED}" ${EXTRA} \
      > "run_logs/exp_${LABEL}_${SEED}.out" 2>&1 || true
    echo "==== DONE  ${LABEL} seed ${SEED} $(date) ===="
  done
}

for lane in "${!GPUS[@]}"; do
  run_lane "${GPUS[lane]}" "${lane}" &
done
wait
echo "ALL RUNS COMPLETE $(date)"
