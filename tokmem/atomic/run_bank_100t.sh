#!/bin/bash
# O(K)-bank methods at 100 tasks, sequential on one GPU, with the memory fixes
# (gradient checkpointing + dynamic padding + expandable_segments).
# Usage: bash run_bank_100t.sh <gpu>
export CUDA_VISIBLE_DEVICES=${1:-0}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/gmar762/anaconda3/envs/treelora/bin/python
cd "$(dirname "$0")"; mkdir -p run_logs
COMMON="--num_tasks 100 --train_size 200 --val_size 10 --test_size 50 \
  --model_name meta-llama/Llama-3.2-3B-Instruct --num_epochs 1 --batch_size 4 \
  --max_length 1024 --max_instruction_tokens 1024 --eval_batch_size 16 \
  --lr 5e-5 --lora_r 8 --lora_alpha 32 --lora_dropout 0.1 \
  --svd_rank 8 --svd_oversampling 10 --seed 42 --gradient_checkpointing"

for METHOD in inflora olora; do
  echo "================ START ${METHOD} (100 tasks) $(date) ================"
  $PY main_svdlora_baseline.py --method ${METHOD} ${COMMON} 2>&1 | tee "run_logs/run100_${METHOD}.out"
  echo "================ DONE  ${METHOD} $(date) ================"
done
echo "BANK 100-task runs COMPLETE $(date)"
