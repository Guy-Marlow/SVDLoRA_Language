#!/bin/bash
# Adaptive-rank SVDLoRA energy-target sweep, 100 tasks, sequential on one GPU.
# Same config as the fixed-rank baseline (alpha=32 -> s=4, all layers) so the only
# variable is fixed-rank-8 vs adaptive rank. Usage: bash run_adapt_sweep.sh <gpu> <eps> [<eps> ...]
export CUDA_VISIBLE_DEVICES=${1:-2}; shift
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/gmar762/anaconda3/envs/treelora/bin/python
cd "$(dirname "$0")"; mkdir -p run_logs
COMMON="--method svdlora --num_tasks 100 --train_size 200 --val_size 10 --test_size 50 \
  --model_name meta-llama/Llama-3.2-3B-Instruct --num_epochs 1 --batch_size 4 \
  --max_length 1024 --max_instruction_tokens 1024 --eval_batch_size 16 \
  --lr 5e-5 --lora_r 8 --lora_alpha 32 --lora_dropout 0.1 \
  --svd_rank 8 --svd_oversampling 10 --seed 42 --svd_layers all"

for EPS in "$@"; do
  echo "================ START adaptive eps=${EPS} (100 tasks) $(date) ================"
  $PY main_svdlora_baseline.py $COMMON --svd_energy_target ${EPS} 2>&1 \
    | tee "run_logs/run100_svdlora_adapt${EPS}.out"
  echo "================ DONE  adaptive eps=${EPS} $(date) ================"
done
echo "ADAPTIVE sweep COMPLETE for: $* $(date)"
