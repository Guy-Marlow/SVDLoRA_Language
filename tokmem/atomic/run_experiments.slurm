#!/bin/bash
#SBATCH --job-name=svdlora_lang_400t
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:2
#SBATCH --open-mode=append
#SBATCH --output=/data/gmar762/research/continuous_learning/svd_sketching_language/tokmem/atomic/run_logs/slurm_experiments_%j.out
#SBATCH --error=/data/gmar762/research/continuous_learning/svd_sketching_language/tokmem/atomic/run_logs/slurm_experiments_%j.err
#
# Language CL comparison on Super-NaturalInstructions atomic recall.
#   5 methods x 3 seeds = 15 runs, 400 tasks (500 train / 50 test, 1 epoch).
#   Methods: seqlora, svdlora (fixed r_hat=8), svdlora (adaptive eps=0.005), olora, inflora.
#
# GPU handling mirrors the cluster's other comparison jobs
# (treelora/scripts/run_method_comparison_seeds.slurm, svd_sketching/scripts/run_comparison.slurm):
# request the GPU(s) ONCE in the #SBATCH header, then run plain `python` calls sequentially --
# no srun, no CUDA_VISIBLE_DEVICES juggling, no backgrounding. The training script is
# single-device (cuda:0), so gpu:2 is just defensive (run on the good card if one is busted),
# exactly as the sibling jobs request it; one run executes at a time.
#
# Ordering: fast O(1) methods (seqlora, svdlora) run FIRST across all seeds so usable results
# land early; slow O(K) banks (olora, inflora) run LAST. No gradient checkpointing -- on H200
# (141GB) the un-checkpointed peak (~40GB) fits with huge headroom and runs several x faster.

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

# label | method-specific args  (fast O(1) methods first, O(K) banks last)
RUNS=(
  "seqlora|--method seqlora"
  "svdlora_fixed|--method svdlora --svd_rank 8"
  "svdlora_adapt|--method svdlora --svd_energy_target 0.005"
  "olora|--method olora"
  "inflora|--method inflora"
)

# Method-major over seeds: every fast method (all 3 seeds) finishes before the banks start.
# Sequential on the allocated GPU. `|| true` so one OOM/crash cannot abort the rest of the
# sweep (the harness also OOM-guards internally and exits 0, writing partial metrics).
for entry in "${RUNS[@]}"; do
  LABEL="${entry%%|*}"
  EXTRA="${entry#*|}"
  for SEED in 1993 1994 1995; do
    echo "==== START ${LABEL} seed ${SEED} $(date) ===="
    python main_svdlora_baseline.py $COMMON \
      --order_seed "${SEED}" --seed "${SEED}" ${EXTRA} \
      > "run_logs/exp_${LABEL}_${SEED}.out" 2>&1 || true
    echo "==== DONE  ${LABEL} seed ${SEED} $(date) ===="
  done
done
echo "ALL RUNS COMPLETE $(date)"