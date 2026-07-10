#!/bin/bash
#SBATCH --job-name=stride750
#SBATCH --time=144:00:00
#SBATCH --gres=gpu:2
#SBATCH --open-mode=append
#SBATCH --output=/data/gmar762/research/continuous_learning/svd_sketching_language/tokmem/atomic/run_logs/slurm_stride750_%j.out
#SBATCH --error=/data/gmar762/research/continuous_learning/svd_sketching_language/tokmem/atomic/run_logs/slurm_stride750_%j.err
#
# 750-task strided-checkpoint CL study, two backbones: Llama-3.2-3B-Instruct and
# Qwen2.5-0.5B-Instruct. main_cl_longhorizon.py, boundary_mode=task. Every 75 tasks the
# DEPLOYED adapter (seqlora: the single drifting adapter; svdlora: the frozen sketch after
# that task's compression; olora: the merged composite of all per-task adapters) is
# evaluated on the prior tasks at EVEN 1-indexed training positions (k % 2 == 0) of that
# seed's ordering. Per checkpoint the JSON records: rougeL, exact_match, per_task scores,
# deployed rank, adapter MB, inference FLOPs/token, cumulative+interval train seconds,
# eval seconds, per-example inference time, peak VRAM; plus per-task train wall-clock and
# the full task ordering (meta["task_order"], position k = index+1).
#
#   24 runs = 2 backbones x 12:
#     base x {1993,1994,1995}  (raw model, single stride-eval -- the baseline row)
#     + {svdlora eps=0.005, seqlora, olora lamda_1=0.5} x {1993,1994,1995}
#   Seeds are FULLY paired (--seed=--set_seed=--order_seed=S), the same convention as the
#   sensitivity grid + cs-ablation runs, so each seed shares its task set AND order with
#   the existing 750t svdlora runs (same for both backbones, since the task order only
#   depends on set_seed/order_seed, not the model). Baselines run first (cheap), then
#   seed-major method trios. Qwen (0.5B, cheap) entries are listed before the equivalent
#   Llama (3B, slow) entries in the flat run list, so under the round-robin GPU-slot
#   assignment below most Qwen runs land earlier than most Llama runs (not a strict
#   per-slot guarantee, but the full Qwen sweep should finish well ahead of Llama).
#
#   Each backbone writes to its OWN subdir (run_logs/strideeval_750t/{llama3b,qwen05b})
#   since the run tag does not encode the model name.
#
#   RESUMABLE: skips any run whose JSON says "status": "done" -- resubmit this same script
#   to finish the remainder if 144h is not enough (the Llama-3B trios are the long pole;
#   expect several resubmissions for the full 24-run grid).
#   Writes ONLY into run_logs/strideeval_750t/ (new dir; no existing logs touched).

ATOMIC=/data/gmar762/research/continuous_learning/svd_sketching_language/tokmem/atomic
cd "$ATOMIC"

source /data/gmar762/anaconda3/etc/profile.d/conda.sh
conda activate treelora
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_DEVICE_ORDER=PCI_BUS_ID

OUT_BASE=run_logs/strideeval_750t
OUT_LLAMA="$OUT_BASE/llama3b"
OUT_QWEN="$OUT_BASE/qwen05b"
mkdir -p "$OUT_LLAMA" "$OUT_QWEN"
DRV="$OUT_BASE/_driver.log"
N=750

COMMON="--num_tasks ${N} --min_instances 560 \
  --train_size 500 --val_size 10 --test_size 50 \
  --num_epochs 1 --batch_size 4 --eval_batch_size 16 \
  --max_length 1024 --max_instruction_tokens 1024 \
  --lora_r 8 --lora_alpha 32 --target_modules q_proj,v_proj \
  --lr 5e-5 --svd_energy_target 0.005 --lamda_1 0.5 \
  --boundary_mode task --eval_every_tasks 75 --eval_stride 2"

# entries: model|method|seed. Qwen (fast) interleaved before Llama (slow) per method/seed
# group; within each backbone, baselines first, then seed-major method trios.
RUNS=(
  "qwen|base|1993"    "qwen|base|1994"    "qwen|base|1995"
  "llama|base|1993"   "llama|base|1994"   "llama|base|1995"
  "qwen|svdlora|1993" "qwen|seqlora|1993" "qwen|olora|1993"
  "qwen|svdlora|1994" "qwen|seqlora|1994" "qwen|olora|1994"
  "qwen|svdlora|1995" "qwen|seqlora|1995" "qwen|olora|1995"
  "llama|svdlora|1993" "llama|seqlora|1993" "llama|olora|1993"
  "llama|svdlora|1994" "llama|seqlora|1994" "llama|olora|1994"
  "llama|svdlora|1995" "llama|seqlora|1995" "llama|olora|1995"
)

run_one() {
  local gpu="$1" entry="$2"
  local BK="${entry%%|*}" rest="${entry#*|}"
  local M="${rest%%|*}" S="${rest#*|}"
  local MODEL OUTDIR
  if [ "$BK" = "llama" ]; then
    MODEL=meta-llama/Llama-3.2-3B-Instruct; OUTDIR="$OUT_LLAMA"
  else
    MODEL=Qwen/Qwen2.5-0.5B-Instruct; OUTDIR="$OUT_QWEN"
  fi
  local JSON="$OUTDIR/longhorizon_${M}_lr5e-05_task_${N}t_s${S}.json"
  local LOG="$OUTDIR/run_${M}_${N}t_s${S}.out"
  if [ -f "$JSON" ] && grep -q '"status": "done"' "$JSON"; then
    echo "==== SKIP (done) ${BK} ${M} s${S} $(date) ====" >> "$DRV"
    return 0
  fi
  echo "==== START [gpu ${gpu}] ${BK} ${M} s${S} $(date) ====" >> "$DRV"
  CUDA_VISIBLE_DEVICES="${gpu}" python main_cl_longhorizon.py $COMMON \
    --model_name "$MODEL" --out_dir "$OUTDIR" \
    --method "$M" --seed "$S" --set_seed "$S" --order_seed "$S" \
    > "$LOG" 2>&1 || true
  echo "==== DONE  [gpu ${gpu}] ${BK} ${M} s${S} $(date) ====" >> "$DRV"
}

NGPU="${SLURM_GPUS_ON_NODE:-}"
[ -z "$NGPU" ] && NGPU="$(nvidia-smi -L 2>/dev/null | wc -l)"
case "$NGPU" in (''|*[!0-9]*) NGPU=1 ;; esac
[ "$NGPU" -lt 1 ] && NGPU=1
echo "==== stride750: ${#RUNS[@]} runs on NGPU=${NGPU} slot(s) $(date) ====" >> "$DRV"

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
echo "ALL STRIDEEVAL-750T RUNS COMPLETE $(date)" >> "$DRV"
