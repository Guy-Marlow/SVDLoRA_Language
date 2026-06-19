#!/bin/bash
# Faithfulness check of the actual TokMem method against the paper's Table 1.
# Llama-3.2-3B-Instruct, Atomic recall on SNI, predicted-task routing eval.
# Target ROUGE-L by num_tasks: 10 -> 68.0  50 -> 62.3  1000 -> 61.5
# Matches paper TokMem HPs: lr 5e-3 (stock main_tokmem.sh omits --lr -> 0.001!),
# coupled embeddings (plain TokMem, not +DC), 1 epoch, bs 4, max_len 1024.
export CUDA_VISIBLE_DEVICES=${2:-2}
NUM_TASKS=${1:-10}
LR=${3:-0.005}

python main_in_domain.py \
    --num_tasks ${NUM_TASKS} \
    --train_size 500 \
    --val_size 10 \
    --test_size 50 \
    --model_name "meta-llama/Llama-3.2-3B-Instruct" \
    --num_epochs 1 \
    --batch_size 4 \
    --gradient_accumulation_steps 1 \
    --max_length 1024 \
    --max_instruction_tokens 1024 \
    --eval_batch_size 16 \
    --validate_every_n_steps 1000 \
    --lr ${LR} \
    --seed 42
