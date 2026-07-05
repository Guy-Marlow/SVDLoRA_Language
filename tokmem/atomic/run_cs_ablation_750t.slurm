#!/bin/bash
#SBATCH --job-name=cs750
#SBATCH --time=96:00:00
#SBATCH --gres=gpu:2
#SBATCH --open-mode=append
#SBATCH --output=/data/gmar762/research/continuous_learning/svd_sketching_language/tokmem/atomic/run_logs/slurm_cs750_%j.out
#SBATCH --error=/data/gmar762/research/continuous_learning/svd_sketching_language/tokmem/atomic/run_logs/slurm_cs750_%j.err
#
# CountSketch merge-op ablation at 750 tasks + the two missing adaptive-SVDLoRA eps=0.005 seeds.
#   Llama-3.2-3B-Instruct, main_svdlora_baseline.py (same eval scheme + metrics as the 100-task
#   A100 seeds and the eps-x-horizon sensitivity grid: train all 750, ONE final pooled
#   greedy-generation eval, --sketch_cost_probe per-boundary ideal-vs-kept cost).
#
#   5 runs:
#     countsketch k=8   x seeds {1993, 1994, 1995}   -> run_logs/cs_ablation_750t
#     svdlora eps=0.005 x seeds {1994, 1995}         -> run_logs/sensitivity_tasks_llama3b
#   (svdlora eps=0.005 seed 1993 at 750t ALREADY EXISTS in the sensitivity grid -- not rerun.
#    The SVD seed-fills land in the sensitivity dir so that grid gains error bars; the CS runs
#    get their own dir. Each seed uses --seed/--set_seed/--order_seed = S, EXACTLY like the
#    sensitivity script, so every CS seed shares its task set+order with its SVD sibling.)
#
#   CS arm: --merge_op countsketch --cs_rank 8 = parameter- and FLOP-matched to fixed rank-8
#   (k(d_in+d_out) bytes). Boundary op cost is negligible (~60ms/boundary measured); runtime is
#   training+eval bound, same as any fixed-rank run. 10-task probe: measured collision error =
#   theory (m2/p2 = 0.98); drift vs true accumulated delta compounds ~+0.44/task -> registered
#   prediction: cs8 lands at/below seqlora at long horizon. This run tests that at 750.
#
#   Run order interleaves arms so seed-PAIRS complete earliest (cs1993 pairs with the existing
#   sensitivity svd1993 as soon as it lands). GPU-count-aware (slot g takes entries g, g+NGPU,
#   ...; 1 GPU -> sequential). RESUMABLE: skips runs whose metrics JSON says "status": "done".

ATOMIC=/data/gmar762/research/continuous_learning/svd_sketching_language/tokmem/atomic
cd "$ATOMIC"

source /data/gmar762/anaconda3/etc/profile.d/conda.sh
conda activate treelora
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_DEVICE_ORDER=PCI_BUS_ID

OUT_CS=run_logs/cs_ablation_750t
OUT_SVD=run_logs/sensitivity_tasks_llama3b
mkdir -p "$OUT_CS" "$OUT_SVD"
DRV="$OUT_CS/_driver.log"
MODEL=meta-llama/Llama-3.2-3B-Instruct
N=750
CS_K=8

common() {   # $1 = seed, $2 = out_dir
  echo "--method svdlora --num_tasks ${N} --train_size 500 --val_size 10 --test_size 50 \
    --min_instances 560 --model_name ${MODEL} --num_epochs 1 --batch_size 4 \
    --max_length 1024 --max_instruction_tokens 1024 --eval_batch_size 16 \
    --lr 5e-5 --lora_r 8 --lora_alpha 32 \
    --seed $1 --set_seed $1 --order_seed $1 --sketch_cost_probe --out_dir $2"
}

# entries: kind|seed  (kind: cs / svd). Interleaved so seed-pairs complete earliest.
RUNS=("cs|1993" "svd|1994" "cs|1994" "svd|1995" "cs|1995")

run_one() {
  local gpu="$1" entry="$2"
  local kind="${entry%%|*}" S="${entry#*|}" JSON OUTDIR EXTRA LOG
  if [ "$kind" = "cs" ]; then
    OUTDIR="$OUT_CS"
    EXTRA="--svd_rank 8 --merge_op countsketch --cs_rank ${CS_K}"
    JSON="$OUTDIR/metrics_svdlora_all_${N}t_s${S}_cs${CS_K}.json"
    LOG="$OUTDIR/run_cs${CS_K}_${N}t_s${S}.out"
  else
    OUTDIR="$OUT_SVD"
    EXTRA="--svd_energy_target 0.005"
    JSON="$OUTDIR/metrics_svdlora_all_${N}t_s${S}_adapt0.005.json"
    LOG="$OUTDIR/run_${N}t_eps0.005_s${S}.out"
  fi
  if [ -f "$JSON" ] && grep -q '"status": "done"' "$JSON"; then
    echo "==== SKIP (done) ${kind} s${S} $(date) ====" >> "$DRV"
    return 0
  fi
  echo "==== START [gpu ${gpu}] ${kind} s${S} $(date) ====" >> "$DRV"
  CUDA_VISIBLE_DEVICES="${gpu}" python main_svdlora_baseline.py $(common "$S" "$OUTDIR") $EXTRA \
    > "$LOG" 2>&1 || true
  echo "==== DONE  [gpu ${gpu}] ${kind} s${S} $(date) ====" >> "$DRV"
}

NGPU="${SLURM_GPUS_ON_NODE:-}"
[ -z "$NGPU" ] && NGPU="$(nvidia-smi -L 2>/dev/null | wc -l)"
case "$NGPU" in (''|*[!0-9]*) NGPU=1 ;; esac
[ "$NGPU" -lt 1 ] && NGPU=1
echo "==== cs750: ${#RUNS[@]} runs on NGPU=${NGPU} slot(s) $(date) ====" >> "$DRV"

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
echo "ALL CS-ABLATION-750T RUNS COMPLETE $(date)" >> "$DRV"