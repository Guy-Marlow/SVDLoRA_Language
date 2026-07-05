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
import json
import os
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
import report
from svdlora_layer import (
    inject_svdlora,
    svdlora_trainable_parameters,
    compress_all,
    stash_all,
)
from multislot_lora import (
    inject_multislot_lora,
    trainable_parameters as multislot_trainable_parameters,
    orthogonality_penalty,
    collect_covariance,
    init_lora_A_dualgpm,
    update_dualgpm,
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

        # No padding here: pad dynamically per-batch in svd_collate_fn (exactly
        # equivalent to max_length padding -- pad tokens are masked -- but far
        # cheaper, since most SNI samples are well under max_length).
        enc = self.tokenizer(text, truncation=True, max_length=self.max_length,
                             padding=False, return_tensors="pt")
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


def svd_collate_fn(batch, tokenizer):
    """Dynamic padding to the batch-longest sequence (left side, matching
    tokenizer.padding_side='left'). Pads input_ids with pad_token_id, attention
    with 0, labels with -100 -> identical masked loss to max_length padding."""
    maxlen = max(item['input_ids'].size(0) for item in batch)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    left = (getattr(tokenizer, "padding_side", "right") == "left")
    ids_b, am_b, lb_b = [], [], []
    for item in batch:
        ids, am, lb = item['input_ids'], item['attention_mask'], item['labels']
        n = maxlen - ids.size(0)
        p_ids = torch.full((n,), pad_id, dtype=ids.dtype)
        p_am = torch.zeros(n, dtype=am.dtype)
        p_lb = torch.full((n,), -100, dtype=lb.dtype)
        if left:
            ids_b.append(torch.cat([p_ids, ids])); am_b.append(torch.cat([p_am, am])); lb_b.append(torch.cat([p_lb, lb]))
        else:
            ids_b.append(torch.cat([ids, p_ids])); am_b.append(torch.cat([am, p_am])); lb_b.append(torch.cat([lb, p_lb]))
    return {'input_ids': torch.stack(ids_b), 'attention_mask': torch.stack(am_b), 'labels': torch.stack(lb_b)}


def build_loader(samples, tokenizer, args, shuffle):
    ds = LoRAInstructionsDataset(samples, tokenizer, max_length=args.max_length)
    return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle,
                      collate_fn=lambda b: svd_collate_fn(b, tokenizer))


def train_one_task(model, trainable_params, loader, args, extra_loss_fn=None, optim=None):
    """Constant LR (matches baseline: AdamW, no scheduler), N epochs.
    extra_loss_fn() (e.g. O-LoRA's orthogonality penalty) is added to each step's loss.
    If `optim` is given it is reused (persistent Adam momentum across tasks); otherwise a
    fresh optimiser is built. Returns the optimiser so callers can persist it."""
    # Reference TokMem LoRA baseline (main_lora_baseline.py) divides BOTH the loss and the
    # LR by grad_accum -- a double 1/G. Under Adam only the lr/G bites (Adam is invariant to
    # the loss/G magnitude scaling), so the reference's effective Adam LR is lr/grad_accum.
    # Our default does loss/G only (line below in the step loop) and leaves the LR alone.
    # --ref_lr_div replicates the reference exactly so we can faithfully reproduce its
    # bs1/ga4 effective LR (and its sequence-weighted gradient when batch_size=1).
    eff_lr = args.lr / args.gradient_accumulation_steps if args.ref_lr_div else args.lr
    if optim is None:
        optim = torch.optim.AdamW(trainable_params, lr=eff_lr)
    model.train()
    for epoch in range(args.num_epochs):
        optim.zero_grad()
        for step, batch in enumerate(tqdm(loader, desc=f"  epoch {epoch+1}/{args.num_epochs}", leave=False)):
            batch = {k: v.to(args.device) for k, v in batch.items()}
            loss = model(**batch).loss
            if extra_loss_fn is not None:
                loss = loss + extra_loss_fn()
            (loss / args.gradient_accumulation_steps).backward()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optim.step(); optim.zero_grad()
    return optim


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--method', choices=['svdlora', 'seqlora', 'olora', 'inflora'], default='svdlora')
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
    p.add_argument('--lr', type=float, default=5e-5)          # effective LR; no division (unless --ref_lr_div)
    p.add_argument('--ref_lr_div', action='store_true',
                   help="replicate reference grad-accum LR division: effective LR = lr / grad_accum")
    p.add_argument('--persist_optimizer', action='store_true',
                   help="retain a single AdamW across all tasks (persistent Adam momentum) "
                        "instead of building a fresh optimiser per task")
    p.add_argument('--device', default="cuda")
    p.add_argument('--seed', type=int, default=42)
    # experiment ordering: fixed task set (set_seed) permuted per run (order_seed).
    # order_seed=None -> legacy global-shuffle selection.
    p.add_argument('--order_seed', type=int, default=None)
    p.add_argument('--set_seed', type=int, default=0)
    p.add_argument('--out_dir', default="run_logs",
                   help='where metrics/diag/pertask JSONs are written (default run_logs/)')
    # adapter HPs (match baseline: r=8, alpha=32, q_proj+v_proj)
    p.add_argument('--lora_r', type=int, default=8)
    p.add_argument('--lora_alpha', type=float, default=32.0)
    p.add_argument('--lora_dropout', type=float, default=0.1)
    p.add_argument('--target_modules', default="q_proj,v_proj")
    # sketch HPs (svdlora)
    p.add_argument('--svd_rank', type=int, default=8)        # r_hat (fixed-rank mode)
    p.add_argument('--svd_oversampling', type=int, default=10)
    p.add_argument('--svd_period', type=int, default=1)
    # adaptive-rank mode: keep smallest rank retaining (1 - eps) energy each task.
    # None/unset => original fixed-rank path. e.g. 0.01 keeps 99% energy.
    p.add_argument('--svd_energy_target', type=float, default=None)
    p.add_argument('--svd_warmup_fixed_rank', type=int, default=None,
                   help="warm-up: compress to this FIXED target rank for the first "
                        "--svd_warmup_tasks tasks (lossless while raw rank < it), then switch "
                        "to the adaptive --svd_energy_target path. e.g. 160")
    p.add_argument('--svd_warmup_tasks', type=int, default=0,
                   help="number of initial tasks to use the fixed warm-up rank before adaptive")
    # per-task retained-energy diagnostics (extra full SVD/module/task -> EXPENSIVE). OFF for real runs.
    p.add_argument('--svd_diag', action='store_true')
    # ---- merge-op ablation: how the boundary folds sketch+residual back into the sketch ----
    # randsvd (default) = spectral truncation (the method). countsketch = unbiased random
    # signed merge of the concatenated factors' rank dimension into --cs_rank buckets
    # (naive-merge ablation; fixed-rank only, incompatible with --svd_energy_target/warm-up).
    p.add_argument('--merge_op', choices=['randsvd', 'countsketch'], default='randsvd')
    p.add_argument('--cs_rank', type=int, default=None,
                   help='CountSketch bucket count k (deployed sketch rank); default = --svd_rank '
                        '(parameter-matched to fixed-rank SVDLoRA)')
    # O-LoRA HPs
    p.add_argument('--lamda_1', type=float, default=0.5)     # orthogonality weight
    p.add_argument('--lamda_2', type=float, default=0.0)     # L2 on current LoRA
    # InfLoRA / DualGPM HPs
    p.add_argument('--lamb', type=float, default=0.95)       # threshold lower bound
    p.add_argument('--lame', type=float, default=1.0)        # threshold upper bound
    # InfLoRA ablation: disable the data-aligned analytical (DualGPM null-space) A init.
    # A is left at its random kaiming init and frozen; only B trains (InfLoRA's bank
    # structure preserved). Isolates whether InfLoRA's edge is the data-alignment or just
    # the frozen-A/trained-B per-task bank. No covariance collection / no null-space growth.
    p.add_argument('--inflora_no_align', action='store_true')
    # memory: gradient checkpointing (exact; recompute activations in backward)
    p.add_argument('--gradient_checkpointing', action='store_true')
    # svdlora layer restriction: all blocks / bottom (low-level) / top (high-level) half
    p.add_argument('--svd_layers', choices=['all', 'bottom', 'top'], default='all')
    # ---- forgetting probe ----
    # Track the FIRST `forgetting_probe` trained tasks and re-evaluate their (fixed) test
    # sets at several training checkpoints to measure catastrophic forgetting directly:
    # if accuracy on those tasks is highest right after they are trained and decays as more
    # tasks are learned, forgetting is occurring. 0 = off (default behaviour unchanged).
    p.add_argument('--forgetting_probe', type=int, default=0,
                   help='number of earliest-trained tasks to track for forgetting (e.g. 20)')
    p.add_argument('--probe_checkpoints', default='20,40,60,80,100',
                   help='comma list of #tasks-trained at which to re-eval the probe set; the '
                        'final task is always added')
    # restrict the sampling pool to English tasks with >= this many Instances (the
    # 500/10/50 footprint needs >=560 -> the ~750-task long-horizon pool). 0 = all English.
    p.add_argument('--min_instances', type=int, default=0)
    # ---- per-task sketch-cost probe (adaptive SVDLoRA, period P=1) ----
    # After training each task's residual, evaluate that task TWICE: once with the residual
    # still live (W + B_hat A_hat + s B A)  -> the "ideal" pre-compression accuracy, and again
    # after the residual is sketched into the frozen B_hat A_hat -> the deployed accuracy.
    # The gap is exactly the per-task accuracy cost of the SVD sketch.
    p.add_argument('--sketch_cost_probe', action='store_true')
    args = p.parse_args()

    set_random_seed(args.seed)
    if args.merge_op == 'countsketch':
        assert args.method == 'svdlora', "--merge_op countsketch only applies to --method svdlora"
        assert args.svd_energy_target is None and args.svd_warmup_fixed_rank is None, \
            "countsketch merge is fixed-rank only (no spectrum): drop --svd_energy_target/--svd_warmup_*"
    _cs_k = args.cs_rank if args.cs_rank is not None else args.svd_rank
    print("=" * 60)
    detail = {
        'svdlora': (f"period P={args.svd_period} compress, r_hat={args.svd_rank}, oversampling={args.svd_oversampling}"
                    + (f" | merge_op=COUNTSKETCH k={_cs_k} (naive-merge ablation)"
                       if args.merge_op == 'countsketch' else "")),
        'seqlora': "single drifting adapter (no compression)",
        'olora':   f"per-task adapters, orth penalty lamda_1={args.lamda_1} lamda_2={args.lamda_2}",
        'inflora': (f"per-task adapters, RANDOM frozen-A (DualGPM align DISABLED), train B only"
                    if args.inflora_no_align else
                    f"per-task adapters, DualGPM lamb={args.lamb} lame={args.lame}"),
    }[args.method]
    print(f"{args.method.upper()} for Natural Instructions (atomic recall)")
    print(f"Model: {args.model_name} | tasks: {args.num_tasks} | {detail}")
    print(f"adapter r={args.lora_r} alpha={args.lora_alpha} q/v | lr={args.lr} (no division)")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.bos_token
    tokenizer.padding_side = "left"

    train_data, val_data, test_data, _ = sample_natural_instructions_tasks(
        tasks_dir=args.tasks_dir, num_tasks=args.num_tasks,
        max_instruction_tokens=args.max_instruction_tokens, tokenizer=tokenizer,
        stable_test_split=True, train_size=args.train_size,
        val_size=args.val_size, test_size=args.test_size, few_shot=False,
        set_seed=args.set_seed, order_seed=args.order_seed,
        min_instances=args.min_instances)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16, device_map=args.device)
    if args.gradient_checkpointing:
        # exact (recompute, not approximate). use_reentrant=False handles the
        # frozen base; enable_input_require_grads lets grad reach the adapters.
        model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
        print("Gradient checkpointing ENABLED (use_reentrant=False, use_cache=False)")
    targets = tuple(args.target_modules.split(","))
    multislot = args.method in ('olora', 'inflora')
    if multislot:
        # per-task adapters added lazily (grow one slot per task)
        modules = inject_multislot_lora(model, target_modules=targets, r=args.lora_r,
                                        alpha=args.lora_alpha, dropout=args.lora_dropout,
                                        collect_cov=(args.method == 'inflora' and not args.inflora_no_align))
        if args.method == 'inflora' and not args.inflora_no_align:
            # q_proj/v_proj alternate per layer (q first); q is square (out==in), pair them
            pairs = list(zip(modules[0::2], modules[1::2]))
            assert pairs[0][0].out_features == pairs[0][0].in_features, "unexpected q/v order"
            q_modules = [q for q, _ in pairs]
    else:
        # block-range restriction (Llama-3.2-3B has 28 blocks; 14/14 split)
        layer_indices = {'all': None,
                         'bottom': set(range(0, 14)),
                         'top': set(range(14, 28))}[args.svd_layers]
        modules = inject_svdlora(model, target_modules=targets, r=args.lora_r,
                                 r_hat=args.svd_rank, alpha=args.lora_alpha,
                                 oversampling=args.svd_oversampling, dropout=args.lora_dropout,
                                 layer_indices=layer_indices,
                                 energy_target=args.svd_energy_target, diag=args.svd_diag,
                                 warmup_fixed_rank=args.svd_warmup_fixed_rank,
                                 warmup_tasks=args.svd_warmup_tasks,
                                 merge_op=args.merge_op, cs_rank=args.cs_rank,
                                 cs_seed=args.seed)
    print(f"Injected {len(modules)} adapter modules ({args.method}, svd_layers={args.svd_layers})")

    # InfLoRA DualGPM state (per-module)
    feature_list, project_type, feature_mat = [], [], []

    tasks = group_by_task(train_data)
    print(f"Training sequentially over {len(tasks)} tasks")

    # run id + incremental metrics (survive any crash)
    os.makedirs(args.out_dir, exist_ok=True)
    _seed_tag = f"_s{args.order_seed}" if args.order_seed is not None else ""
    _tag = f"{args.svd_layers}_{args.num_tasks}t{_seed_tag}" + (
        f"_adapt{args.svd_energy_target}" if args.svd_energy_target is not None else "") + (
        f"_P{args.svd_period}" if args.svd_period != 1 else "") + (
        f"_warm{args.svd_warmup_fixed_rank}x{args.svd_warmup_tasks}"
        if args.svd_warmup_fixed_rank is not None else "") + (
        "_noalign" if (args.method == 'inflora' and args.inflora_no_align) else "") + (
        f"_cs{_cs_k}" if args.merge_op == 'countsketch' else "")
    svd_diag_path = f"{args.out_dir}/svdlora_diag_{args.method}_{_tag}.json"
    metrics_path = f"{args.out_dir}/metrics_{args.method}_{_tag}.json"
    metrics = {"method": args.method, "num_tasks": args.num_tasks, "order_seed": args.order_seed,
               "seed": args.seed, "train_size": args.train_size, "lora_r": args.lora_r,
               "energy_target": args.svd_energy_target, "svd_rank": args.svd_rank,
               "status": "running", "tasks_done": 0, "mem_curve": []}
    report.write_metrics(metrics_path, metrics)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    svd_diag_records = []
    # per-task sketch-cost probe records (svdlora only)
    sketch_cost_records = []
    sketch_cost_path = f"{args.out_dir}/sketchcost_{args.method}_{_tag}.json"
    if args.sketch_cost_probe and args.method == 'svdlora' and args.svd_period != 1:
        print("[sketch-cost] WARNING: probe assumes period P=1 (compress every task); "
              f"svd_period={args.svd_period} -> 'ideal' includes stashed adapters.", flush=True)

    # opt-in (MEM_SNAPSHOT=1): record allocation call-stacks so an OOM dumps a
    # snapshot pinning exactly which tensors/call-sites hold the memory.
    _mem_snap = os.environ.get("MEM_SNAPSHOT") == "1"
    if _mem_snap and torch.cuda.is_available():
        torch.cuda.memory._record_memory_history(max_entries=200000)

    # ---- forgetting probe setup ----
    # The probe set = pooled test examples of the first `forgetting_probe` trained tasks
    # (fixed across checkpoints). We re-evaluate it after the requested #tasks are trained.
    probe_on = args.forgetting_probe > 0
    probe_records = None
    if probe_on:
        n_probe = min(args.forgetting_probe, len(tasks))
        probe_task_names = {tasks[k][0] for k in range(n_probe)}
        probe_test = [ex for ex in test_data if ex['tasks'][0] in probe_task_names]
        checkpoints = {int(c) for c in args.probe_checkpoints.split(',') if c.strip()}
        checkpoints = {c for c in checkpoints if c <= len(tasks)}
        checkpoints.add(len(tasks))   # always probe at completion
        probe_records = {"method": args.method, "order_seed": args.order_seed,
                         "seed": args.seed, "num_tasks": args.num_tasks,
                         "n_probe_tasks": n_probe, "probe_tasks": sorted(probe_task_names),
                         "checkpoints": sorted(checkpoints), "n_probe_examples": len(probe_test),
                         "evals": {}}   # evals[checkpoint] = {"overall": x, "per_task": {...}}
        probe_path = f"{args.out_dir}/forgetting_{args.method}_{_tag}.json"
        print(f"[forgetting-probe] tracking {n_probe} tasks ({len(probe_test)} test ex) "
              f"at checkpoints {sorted(checkpoints)} -> {probe_path}")

    persist_optim = None   # reused across tasks only when --persist_optimizer
    try:
        for i, (task_name, samples) in enumerate(tasks):
            print(f"[task {i+1}/{len(tasks)}] {task_name}  ({len(samples)} samples)")
            loader = build_loader(samples, tokenizer, args, shuffle=True)

            if multislot:
                for m in modules:
                    m.add_task()

            extra_loss = None
            if args.method == 'inflora':
                if not args.inflora_no_align:
                    # collect input covariance with the accumulated model, set A analytically, freeze A
                    collect_covariance(model, q_modules, build_loader(samples, tokenizer, args, shuffle=False), args.device)
                    init_lora_A_dualgpm(pairs, feature_mat, project_type, i, args.lora_r, args.device)
                # ablation: A stays at its random kaiming init (set in add_task); freeze A, train B only
                for m in modules:
                    m.set_trainable(i, train_a=False)
                trainable = multislot_trainable_parameters(modules)
            elif args.method == 'olora':
                for m in modules:
                    m.set_trainable(i, train_a=True)
                trainable = multislot_trainable_parameters(modules)

                def extra_loss(idx=i):
                    pen = args.lamda_1 * orthogonality_penalty(modules, idx)
                    if args.lamda_2 > 0:
                        pen = pen + args.lamda_2 * sum(
                            torch.norm(m.lora_A[idx]) + torch.norm(m.lora_B[idx]) for m in modules)
                    return pen
            else:  # svdlora / seqlora
                trainable = svdlora_trainable_parameters(modules)

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            persist_optim = train_one_task(model, trainable, loader, args,
                                           extra_loss_fn=extra_loss,
                                           optim=persist_optim if args.persist_optimizer else None)
            if not args.persist_optimizer:
                persist_optim = None
            if torch.cuda.is_available():
                _alloc = torch.cuda.memory_allocated() / 1e9
                _peak = torch.cuda.max_memory_allocated() / 1e9
                print(f"[mem] task {i}: live_boundary={_alloc:.2f}GB  step_peak={_peak:.2f}GB", flush=True)

            if args.method == 'svdlora':
                # Sketch period: within a period each task trains an INDEPENDENT adapter over
                # W+sketch (see svdlora_layer); we stash it and only fold the whole period into
                # the sketch at a boundary. Always compress on the final task so the deployed
                # sketch contains every task.
                boundary = ((i + 1) % args.svd_period == 0) or (i + 1 == len(tasks))
                if boundary:
                    # ---- per-task sketch-cost probe: eval THIS task before vs after sketching ----
                    if args.sketch_cost_probe:
                        cur_test = [ex for ex in test_data if ex['tasks'][0] == task_name]
                        ideal_res, _ = evaluate_with_generation(
                            model=model, tokenizer=tokenizer, test_examples=cur_test,
                            device=args.device, max_new_tokens=256, batch_size=args.eval_batch_size)
                        model.train()
                    diag = compress_all(modules, task_idx=i)
                    if args.sketch_cost_probe:
                        # residual now folded into the frozen sketch + reset -> deployed state
                        sketched_res, _ = evaluate_with_generation(
                            model=model, tokenizer=tokenizer, test_examples=cur_test,
                            device=args.device, max_new_tokens=256, batch_size=args.eval_batch_size)
                        model.train()
                        ranks = [m.sketch_B.shape[1] for m in modules]
                        rec_sc = {"task_idx": i, "task": task_name, "n_test": len(cur_test),
                                  "ideal_rougeL": ideal_res["rougeL"], "sketched_rougeL": sketched_res["rougeL"],
                                  "rougeL_drop": round(ideal_res["rougeL"] - sketched_res["rougeL"], 4),
                                  "ideal_em": ideal_res["exact_match"], "sketched_em": sketched_res["exact_match"],
                                  "em_drop": round(ideal_res["exact_match"] - sketched_res["exact_match"], 4),
                                  "sketch_rank_total": int(sum(ranks)),
                                  "sketch_rank_mean": round(sum(ranks) / len(ranks), 3)}
                        sketch_cost_records.append(rec_sc)
                        with open(sketch_cost_path, "w") as f:
                            json.dump(sketch_cost_records, f, indent=2)
                        print(f"[sketch-cost] task {i+1} {task_name}: ideal {ideal_res['rougeL']:.2f} "
                              f"-> sketched {sketched_res['rougeL']:.2f} (drop {rec_sc['rougeL_drop']:+.2f}) "
                              f"| sketch rank tot {rec_sc['sketch_rank_total']}", flush=True)
                    if diag is not None and args.svd_diag:   # diagnostics only when requested
                        rec = {"task": i, **{k: diag[k] for k in
                               ("retained_mean", "retained_min", "sigma_next_mean", "sigma_next_max",
                                "fro_mean", "fro_max", "residual_fro_mean", "residual_fro_max",
                                "r_hat_mean", "r_hat_max", "r_hat_total",
                                "merge_relerr_mean", "merge_relerr_max")},
                               "r_hat_per_module": diag.get("r_hat"),
                               "merge_relerr_per_module": diag.get("merge_relerr")}
                        svd_diag_records.append(rec)
                        with open(svd_diag_path, "w") as f:
                            json.dump(svd_diag_records, f, indent=2)
                else:
                    stash_all(modules)
            elif args.method == 'inflora' and not args.inflora_no_align:
                collect_covariance(model, q_modules, build_loader(samples, tokenizer, args, shuffle=False), args.device)
                feature_mat = update_dualgpm(q_modules, feature_list, project_type, i, len(tasks),
                                             args.lamb, args.lame)

            # cheap progress + adapter-memory curve flush (partial results survive failure)
            metrics["tasks_done"] = i + 1
            if (i + 1) % 25 == 0 or (i + 1) == len(tasks):
                metrics["mem_curve"].append({"task": i + 1, "adapter_mb": round(
                    report.adapter_memory_bytes(modules, args.method) / 1024 / 1024, 4)})
                report.write_metrics(metrics_path, metrics)

            # ---- forgetting probe: re-eval the fixed first-N-task test set at checkpoints ----
            # Runs on the DEPLOYED state of this task (after svdlora compression / inflora
            # DualGPM update above), so it measures exactly what is deployed at task i+1.
            if probe_on and (i + 1) in checkpoints:
                _was_ckpt = args.gradient_checkpointing
                if _was_ckpt:
                    model.gradient_checkpointing_disable(); model.config.use_cache = True
                pr, _ = evaluate_with_generation(
                    model=model, tokenizer=tokenizer, test_examples=probe_test,
                    device=args.device, max_new_tokens=256, batch_size=args.eval_batch_size)
                probe_records["evals"][i + 1] = {"overall": pr["rougeL"],
                                                 "exact_match": pr.get("exact_match"),
                                                 "per_task": pr.get("per_task", {})}
                with open(probe_path, "w") as f:
                    json.dump(probe_records, f, indent=2)
                print(f"[forgetting-probe] after {i+1} tasks: probe ROUGE-L {pr['rougeL']:.2f} "
                      f"(first {probe_records['n_probe_tasks']} tasks)", flush=True)
                if _was_ckpt:
                    model.gradient_checkpointing_enable(
                        gradient_checkpointing_kwargs={"use_reentrant": False})
                    model.config.use_cache = False
                model.train()   # resume training mode for the next task

        # ---- evaluation: identical path to the LoRA baseline ----
        if args.gradient_checkpointing:
            model.gradient_checkpointing_disable()
            model.config.use_cache = True   # restore KV cache for fast generation
        if probe_on:
            # The forgetting probe already ran the completion-checkpoint eval (the first-N
            # tasks). Skip the full pooled eval so total eval cost stays ~one full eval.
            print("\n[forgetting-probe] run complete; skipping full pooled eval.")
            _final = probe_records["evals"].get(len(tasks), {})
            results = {"rougeL": _final.get("overall"), "exact_match": _final.get("exact_match")}
        else:
            print("\nEvaluating (greedy generation, pooled test set, NI ROUGE-L)...")
            results, _ = evaluate_with_generation(
                model=model, tokenizer=tokenizer, test_examples=test_data,
                device=args.device, max_new_tokens=256, batch_size=args.eval_batch_size)
            print_evaluation_results(results)
            if "per_task" in results:
                with open(f"{args.out_dir}/pertask_{args.method}_{_tag}.json", "w") as f:
                    json.dump(results["per_task"], f, indent=2)

        # ---- final performance report (memory / FLOPs / accuracy / peak VRAM) ----
        metrics["status"] = "done"
        metrics["rougeL"] = results["rougeL"]
        metrics["exact_match"] = results["exact_match"]
        metrics["adapter"] = report.adapter_report(modules, args.method, args.lora_r)
        # MEASURE (via SVD) the actual numerical rank of the deployed adapter each method
        # uses at inference -- nominal (sum of block ranks) vs measured, never conflated.
        print("Measuring deployed-adapter rank (SVD per module)...", flush=True)
        rank_meas = report.measure_deployed_rank(modules, args.method)
        metrics["deployed_rank_measured"] = rank_meas
        print(f"[deployed-rank] {args.method}: nominal_total={rank_meas['nominal_rank_total']} "
              f"measured_total={rank_meas['measured_rank_total']} "
              f"(exact={rank_meas['measured_rank_exact_total']}, "
              f"E99={rank_meas['energy99_rank_total']}, E99.9={rank_meas['energy999_rank_total']}) | "
              f"q_proj {rank_meas['q_proj']['shape']} cap{rank_meas['q_proj']['cap_each']} "
              f"nominal{rank_meas['q_proj']['nominal_each']} measured~{rank_meas['q_proj']['measured_mean']} | "
              f"v_proj {rank_meas['v_proj']['shape']} cap{rank_meas['v_proj']['cap_each']} "
              f"nominal{rank_meas['v_proj']['nominal_each']} measured~{rank_meas['v_proj']['measured_mean']}",
              flush=True)
        # per-task sketch-cost summary (mean/max accuracy lost when the residual is sketched)
        if sketch_cost_records:
            drops = [r["rougeL_drop"] for r in sketch_cost_records]
            ideals = [r["ideal_rougeL"] for r in sketch_cost_records]
            skets = [r["sketched_rougeL"] for r in sketch_cost_records]
            metrics["sketch_cost"] = {
                "n_tasks": len(drops),
                "mean_ideal_rougeL": round(sum(ideals) / len(ideals), 3),
                "mean_sketched_rougeL": round(sum(skets) / len(skets), 3),
                "mean_rougeL_drop": round(sum(drops) / len(drops), 3),
                "max_rougeL_drop": round(max(drops), 3),
                "records_file": os.path.basename(sketch_cost_path)}
            print(f"[sketch-cost] mean ideal {metrics['sketch_cost']['mean_ideal_rougeL']:.2f} "
                  f"-> mean sketched {metrics['sketch_cost']['mean_sketched_rougeL']:.2f} "
                  f"(mean drop {metrics['sketch_cost']['mean_rougeL_drop']:+.2f}, "
                  f"max {metrics['sketch_cost']['max_rougeL_drop']:+.2f})", flush=True)
        if torch.cuda.is_available():
            metrics["peak_vram_mb"] = round(torch.cuda.max_memory_allocated() / 1024 / 1024, 2)
        report.write_metrics(metrics_path, metrics)
        ad = metrics["adapter"]
        print(f"\n{args.method.upper()} DONE | ROUGE-L {results['rougeL']:.2f} | EM {results['exact_match']:.2f}")
        print(f"  adapter mem {ad['adapter_memory_mb']:.3f} MB | inference FLOPs/token {ad['adapter_inference_flops_per_token']:,} "
              f"(x{ad['adapter_flops_x_one_rank_r']} one rank-{args.lora_r}) | "
              f"deployed rank tot {ad['deployed_rank_total']} | peak VRAM {metrics.get('peak_vram_mb')} MB")

    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if not isinstance(e, torch.cuda.OutOfMemoryError) and "out of memory" not in str(e).lower():
            raise   # a real bug, not OOM -> surface it
        print(f"\n[OOM] {args.method} order_seed={args.order_seed} ended early at "
              f"task {metrics['tasks_done']}/{len(tasks)} -- ending this run cleanly.")
        metrics["status"] = "oom"
        metrics["error"] = str(e)[:300]
        if torch.cuda.is_available():
            metrics["peak_vram_mb"] = round(torch.cuda.max_memory_allocated() / 1024 / 1024, 2)
            if _mem_snap:
                snap = os.path.join(args.out_dir, f"oom_snapshot_{args.method}_{args.order_seed}.pickle")
                try:
                    torch.cuda.memory._dump_snapshot(snap)
                    print(f"[mem-snapshot] dumped allocation history -> {snap}")
                    print(torch.cuda.memory_summary())
                except Exception as _se:
                    print(f"[mem-snapshot] dump failed: {_se}")
            torch.cuda.empty_cache()
        report.write_metrics(metrics_path, metrics)
        import sys
        sys.exit(0)   # clean exit so the experiment loop proceeds to the next run


if __name__ == "__main__":
    main()
