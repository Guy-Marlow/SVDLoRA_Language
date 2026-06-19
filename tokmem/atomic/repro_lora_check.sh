#!/bin/bash
# Faithfulness check of the stock TokMem LoRA baseline against the paper's Table 1.
# Llama-3.2-3B-Instruct, Atomic recall on SNI. Target ROUGE-L by num_tasks:
#   10 -> 67.1   50 -> 59.1(FT)/61.1(replay)   1000 -> 57.9(FT)/60.0(replay)
# All HPs match main_lora_baseline.sh except num_tasks (set via $1, default 10).
export CUDA_VISIBLE_DEVICES=${2:-0}
NUM_TASKS=${1:-10}
# NOTE: train_lora_model divides lr by gradient_accumulation_steps (=4), so the
# EFFECTIVE lr = $LR/4. Pass 2e-4 to hit the paper's 5e-5; default 5e-5 -> 1.25e-5.
LR=${3:-5e-5}

python main_lora_baseline.py \
    --num_tasks ${NUM_TASKS} \
    --train_size 500 \
    --val_size 10 \
    --test_size 50 \
    --model_name "meta-llama/Llama-3.2-3B-Instruct" \
    --num_epochs 1 \
    --batch_size 1 \
    --gradient_accumulation_steps 4 \
    --max_length 1280 \
    --max_instruction_tokens 1024 \
    --eval_batch_size 8 \
    --validate_every_n_steps 1000 \
    --lr ${LR} \
    --lora_r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --target_modules "q_proj,v_proj" \
    --save_path "repro_lora_model_${NUM_TASKS}t" \
    --continual_replay \
    --block_size 10 \
    --continual_replay_ratio 0.1