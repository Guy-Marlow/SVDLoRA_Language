#!/usr/bin/env python3
"""
SVDLoRA (and SeqLoRA) baseline for TokMem Atomic Memory Recall.

Fair drop-in for `main_lora_baseline.py`'s single PEFT adapter:
  - raw AutoModelForCausalLM (NO TaskCallingModel, NO reserved task tokens, NO routing)
  - one deployed adapter on q_proj/v_proj; eval = greedy generation from
    instruction+query over the POOLED test set, NI ROUGE-L  (identical to the baseline:
    we reuse evaluate_with_generation from main_lora_baseline verbatim).

The only structural difference between the two methods this script supports:
  - method=seqlora : one residual adapter trained sequentially over the tasks
                     (no compression). The single-drifting-adapter lower bound.
  - method=svdlora : after EVERY task (SVD period P=1) the accumulated effective
                     delta-W is compressed into a frozen rank-r_hat sketch and the
                     residual is reset.  Bounded O(1) memory.
Both train task-by-task with a fresh optimiser per task, so they differ ONLY in the
compression step -> isolates the value of the sketch.

LR NOTE: we set the optimiser LR directly (no division). The stock baseline's
train_lora_model silently divides LR by grad_accum; to match its *effective* LR of
5e-5 we simply pass --lr 5e-5 here.
"""

import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

from task_dataset import sample_natural_instructions_tasks
# reuse the baseline's evaluation + seeding VERBATIM so eval is identical
from main_lora_baseline import (
    set_random_seed,
    lora_collate_fn,
    evaluate_with_generation,
)
from natural_instructions_eval import print_evaluation_results
from svdlora_layer import (
    inject_svdlora,
    svdlora_trainable_parameters,
    compress_all,
)


class LoRAInstructionsDataset(Dataset):
    """Identical rendering to main_lora_baseline.create_lora_dataloaders' inner class:
    instruction+query -> response, loss on response tokens only, left padding."""

    def __init__(self, data, tokenizer, max_length=1024):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        instruction = item.get('instruction', '')
        query = item.get('query', '')
        response = item['responses'][0] if item['responses'] else ""

        is_qwen = 'qwen' in self.tokenizer.name_or_path.lower()
        parts = []
        if is_qwen:
            parts.append(f"<|im_start|>user\n{instruction}\n\n{query}<|im_end|>\n")
            parts.append(f"<|im_start|>assistant\n{response}<|im_end|>\n")
        else:
            parts.append("<|begin_of_text|>")
            parts.append(f"<|start_header_id|>user<|end_header_id|>\n{instruction}\n\n{query}<|eot_id|>")
            parts.append(f"<|start_header_id|>assistant<|end_header_id|>\n{response}<|eot_id|>")
        text = "".join(parts)

        enc = self.tokenizer(text, truncation=True, max_length=self.max_length,
                             padding="max_length", return_tensors="pt")
        input_ids = enc.input_ids.squeeze()
        attention_mask = enc.attention_mask.squeeze()
        labels = input_ids.clone()

        if is_qwen:
            assistant_tok = self.tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
        else:
            assistant_tok = self.tokenizer.encode("<|start_header_id|>assistant<|end_header_id|>", add_special_tokens=False)
        assistant_start = None
        for i in range(len(input_ids) - len(assistant_tok)):
            if input_ids[i:i+len(assistant_tok)].tolist() == assistant_tok:
                assistant_start = i + len(assistant_tok)
                break
        if assistant_start:
            labels[:assistant_start] = -100
        labels[attention_mask == 0] = -100
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}


def group_by_task(train_data):
    """Group samples by task, preserving first-appearance (sampling) order."""
    order, seen, buckets = [], set(), {}
    for item in train_data:
        t = item['tasks'][0]
        if t not in seen:
            seen.add(t); order.append(t); buckets[t] = []
        buckets[t].append(item)
    return [(t, buckets[t]) for t in order]


def train_one_task(model, modules, task_samples, tokenizer, args):
    """Fresh optimiser, constant LR (matches baseline: AdamW, no scheduler), N epochs."""
    ds = LoRAInstructionsDataset(task_samples, tokenizer, max_length=args.max_length)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        collate_fn=lambda b: lora_collate_fn(b, tokenizer))
    optim = torch.optim.AdamW(svdlora_trainable_parameters(modules), lr=args.lr)
    model.train()
    for epoch in range(args.num_epochs):
        optim.zero_grad()
        for step, batch in enumerate(tqdm(loader, desc=f"  epoch {epoch+1}/{args.num_epochs}", leave=False)):
            batch = {k: v.to(args.device) for k, v in batch.items()}
            loss = model(**batch).loss / args.gradient_accumulation_steps
            loss.backward()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optim.step(); optim.zero_grad()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--method', choices=['svdlora', 'seqlora'], default='svdlora')
    p.add_argument('--tasks_dir', default='natural-instructions-2.8/tasks')
    p.add_argument('--model_name', default="meta-llama/Llama-3.2-3B-Instruct")
    p.add_argument('--num_tasks', type=int, default=10)
    p.add_argument('--train_size', type=int, default=500)
    p.add_argument('--val_size', type=int, default=10)
    p.add_argument('--test_size', type=int, default=50)
    p.add_argument('--max_length', type=int, default=1024)
    p.add_argument('--max_instruction_tokens', type=int, default=1024)
    p.add_argument('--num_epochs', type=int, default=1)
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--eval_batch_size', type=int, default=16)
    p.add_argument('--gradient_accumulation_steps', type=int, default=1)
    p.add_argument('--lr', type=float, default=5e-5)          # effective LR; no division
    p.add_argument('--device', default="cuda")
    p.add_argument('--seed', type=int, default=42)
    # adapter HPs (match baseline: r=8, alpha=32, q_proj+v_proj)
    p.add_argument('--lora_r', type=int, default=8)
    p.add_argument('--lora_alpha', type=float, default=32.0)
    p.add_argument('--lora_dropout', type=float, default=0.1)
    p.add_argument('--target_modules', default="q_proj,v_proj")
    # sketch HPs
    p.add_argument('--svd_rank', type=int, default=8)        # r_hat
    p.add_argument('--svd_oversampling', type=int, default=10)
    args = p.parse_args()

    set_random_seed(args.seed)
    print("=" * 60)
    print(f"{args.method.upper()} for Natural Instructions (atomic recall)")
    print(f"Model: {args.model_name} | tasks: {args.num_tasks} | P=1 (compress every task)")
    print(f"adapter r={args.lora_r} alpha={args.lora_alpha} -> r_hat={args.svd_rank} "
          f"oversampling={args.svd_oversampling} | lr={args.lr}")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.bos_token
    tokenizer.padding_side = "left"

    train_data, val_data, test_data, _ = sample_natural_instructions_tasks(
        tasks_dir=args.tasks_dir, num_tasks=args.num_tasks,
        max_instruction_tokens=args.max_instruction_tokens, tokenizer=tokenizer,
        stable_test_split=True, train_size=args.train_size,
        val_size=args.val_size, test_size=args.test_size, few_shot=False)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16, device_map=args.device)
    modules = inject_svdlora(
        model, target_modules=tuple(args.target_modules.split(",")),
        r=args.lora_r, r_hat=args.svd_rank, alpha=args.lora_alpha,
        oversampling=args.svd_oversampling, dropout=args.lora_dropout)
    n_train = sum(p.numel() for p in svdlora_trainable_parameters(modules))
    print(f"Injected {len(modules)} SVDLoRA modules | trainable residual params: {n_train:,}")

    tasks = group_by_task(train_data)
    print(f"Training sequentially over {len(tasks)} tasks "
          f"({'compress after each' if args.method=='svdlora' else 'no compression'})")
    for i, (task_name, samples) in enumerate(tasks):
        print(f"[task {i+1}/{len(tasks)}] {task_name}  ({len(samples)} samples)")
        train_one_task(model, modules, samples, tokenizer, args)
        if args.method == 'svdlora':
            compress_all(modules)   # P = 1

    # ---- evaluation: identical path to the LoRA baseline ----
    print("\nEvaluating (greedy generation, pooled test set, NI ROUGE-L)...")
    results, _ = evaluate_with_generation(
        model=model, tokenizer=tokenizer, test_examples=test_data,
        device=args.device, max_new_tokens=256, batch_size=args.eval_batch_size)
    print_evaluation_results(results)
    print(f"\n{args.method.upper()} DONE | ROUGE-L {results['rougeL']:.2f} | EM {results['exact_match']:.2f}")


if __name__ == "__main__":
    main()
