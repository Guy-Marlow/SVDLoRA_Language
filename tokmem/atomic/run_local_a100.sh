#!/bin/bash
# Local A100 hedge run: same experiment as run_experiments.slurm (5 methods x 3 seeds,
# 400 tasks), split over TWO local A100s so we get results regardless of the H200 job.
#   GPU 0 lane: seqlora, svdlora (fixed r_hat=8), svdlora (adaptive eps=0.005)
#   GPU 1 lane: olora, inflora
# Seed-major within each lane (seed 1993's methods finish before 1994's, etc.).
# All outputs (stdout logs + metrics/diag/pertask JSONs) land in run_logs/a100_400t/.
ATOMIC="$(cd "$(dirname "$0")" && pwd)"
cd "$ATOMIC"
PY=/home/gmar762/anaconda3/envs/treelora/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

OUT=run_logs/a100_400t
mkdir -p "$OUT"

COMMON="--num_tasks 400 --train_size 500 --val_size 10 --test_size 50 \
  --model_name meta-llama/Llama-3.2-3B-Instruct --num_epochs 1 --batch_size 4 \
  --max_length 1024 --max_instruction_tokens 1024 --eval_batch_size 16 \
  --lr 5e-5 --lora_r 8 --lora_alpha 32 --out_dir $OUT"

run_one () {  # $1=gpu  $2=label  $3=method-args  $4=seed
  echo "==== START $2 seed $4 (gpu $1) $(date) ===="
  CUDA_VISIBLE_DEVICES=$1 $PY main_svdlora_baseline.py $COMMON \
    --order_seed $4 --seed $4 $3 > "$OUT/exp_$2_$4.out" 2>&1 || true
  echo "==== DONE  $2 seed $4 $(date) ===="
}

# GPU 0 lane: the bounded-memory methods (seed-major)
lane0 () {
  for SEED in 1993 1994 1995; do
    run_one 0 seqlora       "--method seqlora" $SEED
    run_one 0 svdlora_fixed "--method svdlora --svd_rank 8" $SEED
    run_one 0 svdlora_adapt "--method svdlora --svd_energy_target 0.005" $SEED
  done
  echo "LANE0 (gpu0) COMPLETE $(date)"
}

# GPU 1 lane: the O(K) banks (seed-major)
lane1 () {
  for SEED in 1993 1994 1995; do
    run_one 1 olora   "--method olora" $SEED
    run_one 1 inflora "--method inflora" $SEED
  done
  echo "LANE1 (gpu1) COMPLETE $(date)"
}

lane0 &
lane1 &
wait
echo "LOCAL A100 RUN COMPLETE $(date)"
