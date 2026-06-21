#!/bin/bash
# 100-task smoke: run the given methods sequentially on one GPU.
# Usage: bash run_100task_smoke.sh <gpu> <method> [<method> ...]
#   methods: seqlora svdlora olora inflora
export CUDA_VISIBLE_DEVICES=${1:-0}; shift
PY=/home/gmar762/anaconda3/envs/treelora/bin/python
cd "$(dirname "$0")"; mkdir -p run_logs
COMMON="--num_tasks 100 --train_size 200 --val_size 10 --test_size 50 \
  --model_name meta-llama/Llama-3.2-3B-Instruct --num_epochs 1 --batch_size 4 \
  --max_length 1024 --max_instruction_tokens 1024 --eval_batch_size 16 \
  --lr 5e-5 --lora_r 8 --lora_alpha 32 --lora_dropout 0.1 \
  --svd_rank 8 --svd_oversampling 10 --seed 42"

for METHOD in "$@"; do
  echo "================ START ${METHOD} (100 tasks) $(date) ================"
  $PY main_svdlora_baseline.py --method ${METHOD} ${COMMON} 2>&1 | tee "run_logs/smoke100_${METHOD}.out"
  echo "================ DONE  ${METHOD} $(date) ================"
done
echo "100-task smoke COMPLETE for: $* $(date)"
