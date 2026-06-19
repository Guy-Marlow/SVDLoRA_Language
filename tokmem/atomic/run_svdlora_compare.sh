#!/bin/bash
# SVDLoRA vs SeqLoRA on TokMem atomic recall (10 tasks), matched HPs.
# Single adapter, no routing, P=1 (compress every task for svdlora). Effective lr 5e-5.
export CUDA_VISIBLE_DEVICES=${1:-0}
PY=/home/gmar762/anaconda3/envs/treelora/bin/python
cd "$(dirname "$0")"; mkdir -p run_logs
NT=${2:-10}
COMMON="--num_tasks ${NT} --train_size 500 --val_size 10 --test_size 50 \
  --model_name meta-llama/Llama-3.2-3B-Instruct --num_epochs 1 --batch_size 4 \
  --max_length 1024 --max_instruction_tokens 1024 --eval_batch_size 16 \
  --lr 5e-5 --lora_r 8 --lora_alpha 32 --lora_dropout 0.1 \
  --svd_rank 8 --svd_oversampling 10 --seed 42"

for METHOD in svdlora seqlora; do
  echo "================ START ${METHOD} (${NT} tasks) $(date) ================"
  $PY main_svdlora_baseline.py --method ${METHOD} ${COMMON} 2>&1 | tee "run_logs/${METHOD}_${NT}t.out"
  echo "================ DONE  ${METHOD} $(date) ================"
done
echo "SVDLORA/SEQLORA ${NT}-task COMPARISON COMPLETE $(date)"
