#!/bin/bash
# Re-run the O(K) bank methods at 400 tasks with the WEIGHT-FOLDED forward (frozen adapters merged
# into one dense delta -> O(1) per-step forward instead of the K-branch Python loop). No gradient
# checkpointing: the fold makes each step cheap and the un-checkpointed step (~27GB incl. the dense
# delta) fits on a dedicated 40GB card. Mathematically identical at eval (proven: smoke_fold_test.py).
#
# olora on a free GPU, inflora on another free GPU (NOT GPU1 -- that has an unrelated co-tenant).
ATOMIC="$(cd "$(dirname "$0")" && pwd)"; cd "$ATOMIC"
PY=/home/gmar762/anaconda3/envs/treelora/bin/python
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT=run_logs/a100_400t; mkdir -p "$OUT"

COMMON="--num_tasks 400 --train_size 500 --val_size 10 --test_size 50 \
  --model_name meta-llama/Llama-3.2-3B-Instruct --num_epochs 1 --batch_size 4 \
  --max_length 1024 --max_instruction_tokens 1024 --eval_batch_size 16 \
  --lr 5e-5 --lora_r 8 --lora_alpha 32 --out_dir $OUT"

run_one () {  # $1=gpu  $2=method  $3=seed
  echo "==== START $2 seed $3 (gpu $1, folded) $(date) ===="
  CUDA_VISIBLE_DEVICES=$1 $PY main_svdlora_baseline.py $COMMON \
    --order_seed $3 --seed $3 --method $2 > "$OUT/exp_$2_$3.out" 2>&1 || true
  echo "==== DONE  $2 seed $3 $(date) ===="
}

olora_lane ()  { for S in 1993 1994 1995; do run_one 2 olora   $S; done; echo "OLORA  LANE DONE $(date)"; }
inflora_lane () { for S in 1993 1994 1995; do run_one 4 inflora $S; done; echo "INFLORA LANE DONE $(date)"; }

olora_lane &
inflora_lane &
wait
echo "FOLDED BANKS COMPLETE $(date)"