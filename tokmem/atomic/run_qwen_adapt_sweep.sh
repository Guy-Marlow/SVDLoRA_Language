#!/bin/bash
# Adaptive-SVDLoRA hyperparameter sweep on Qwen2.5-0.5B-Instruct.
#   epsilon (energy_target) in {0.01, 0.005, 0.001}  x  sketch period in {5, 10, 20}  = 9 runs.
# 400 tasks (train 500 / test 50, 1 epoch), order_seed 1993. Per-task (per-compression)
# diagnostics ON via --svd_diag -- FREE for adaptive (the spectrum comes from the same
# randomized SVD that does the compression; no extra full SVD), so no training slowdown.
# Outputs (metrics/diag/pertask JSONs + stdout) -> run_logs/qwen_adapt_sweep/.
export CUDA_VISIBLE_DEVICES=${1:-2}
PY=/home/gmar762/anaconda3/envs/treelora/bin/python
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

OUT=run_logs/qwen_adapt_sweep
mkdir -p "$OUT"
COMMON="--method svdlora --model_name Qwen/Qwen2.5-0.5B-Instruct \
  --num_tasks 400 --train_size 500 --val_size 10 --test_size 50 --num_epochs 1 \
  --batch_size 4 --eval_batch_size 32 --max_length 1024 --max_instruction_tokens 1024 \
  --lr 5e-5 --lora_r 8 --lora_alpha 32 --target_modules q_proj,v_proj \
  --order_seed 1993 --seed 1993 --svd_diag --out_dir $OUT"

for EPS in 0.01 0.005 0.001; do
  for P in 5 10 20; do
    echo "================ START eps=${EPS} period=${P} $(date) ================"
    $PY main_svdlora_baseline.py $COMMON --svd_energy_target ${EPS} --svd_period ${P} \
      > "$OUT/qwen_adapt_eps${EPS}_P${P}.out" 2>&1 || true
    echo "================ DONE  eps=${EPS} period=${P} $(date) ================"
  done
done
echo "QWEN ADAPTIVE SWEEP COMPLETE $(date)"
