#!/usr/bin/env python3
"""
Long-horizon CL orchestrator for the 750-task SNI study (Qwen2.5-0.5B).

Adds TWO things the per-task baseline (main_svdlora_baseline.py) does not, in a separate
codepath so the validated task-loop scripts are untouched:

  1. CUMULATIVE EVAL every --eval_every_tasks tasks: the single deployed (composite) adapter
     is evaluated on the POOLED test set of ALL tasks seen so far -> an accuracy-vs-tasks-seen
     curve (and the full task x checkpoint ROUGE matrix for forgetting analysis). No routing.

  2. A SAMPLE-DRIVEN BOUNDARY mode (--boundary_mode sample). The method's "boundary"
     (svdlora: SVD compression; olora: a new per-task adapter slot; seqlora: a fresh optimizer)
     is normally fired at every TASK end (--boundary_mode task, the standard setup). In sample
     mode it instead fires every B = round(boundary_samples_mult * train_size) STREAMED samples,
     regardless of task identity -- so the learner never "knows" a task ended, only that some
     number of samples elapsed. This tests whether SVDLoRA degrades more gracefully than
     seqlora/olora when task boundaries are NOT given at training time.

     NOTE: eval still happens at --eval_every_tasks TASK increments (the experimenter knows the
     task structure for measurement; the *learner* does not, in sample mode).

Methods reuse the existing layers verbatim:
  seqlora : one drifting residual (inject_svdlora, never compressed); boundary = new optimizer.
  svdlora : adaptive-rank sketch (inject_svdlora, energy_target); boundary = compress_all.
  olora   : per-task bank (inject_multislot_lora) + orthogonality penalty; boundary = new slot.
"""
import argparse
import json
import os
import time
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM

from task_dataset import sample_natural_instructions_tasks
from main_lora_baseline import set_random_seed, evaluate_with_generation
from natural_instructions_eval import print_evaluation_results
from main_svdlora_baseline import group_by_task, LoRAInstructionsDataset, svd_collate_fn
import report
from svdlora_layer import inject_svdlora, svdlora_trainable_parameters, compress_all
from multislot_lora import (
    inject_multislot_lora, trainable_parameters as multislot_trainable_parameters,
    orthogonality_penalty,
)


def build_loader(samples, tokenizer, args, shuffle):
    ds = LoRAInstructionsDataset(samples, tokenizer, max_length=args.max_length)
    return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle,
                      collate_fn=lambda b: svd_collate_fn(b, tokenizer))


@torch.no_grad()
def deployed_now(modules, method, lora_r):
    """Actual deployed-adapter cost at the current (possibly mid-chunk) state. For svdlora it
    counts the frozen sketch PLUS any live (non-zero) residual, so a sample-mode checkpoint
    that falls mid-chunk is reported honestly. Returns (rank_total, bytes, flops_per_token)."""
    rank = byts = flops = 0
    for m in modules:
        if method == 'svdlora':
            rs = m.sketch_B.shape[1] if float(m.sketch_B.abs().sum()) > 0 else 0  # 0 before 1st compress
            res = m.r if float(m.lora_B.abs().sum()) > 0 else 0
            r_tot = rs + res
            byts += (m.sketch_B.numel() + m.sketch_A.numel()) * 2
            if res:
                byts += res * (m.in_features + m.out_features) * 2
        elif method == 'seqlora':
            r_tot = m.lora_A.shape[0]
            byts += (m.lora_A.numel() + m.lora_B.numel()) * 2
        else:  # olora
            r_tot = sum(a.shape[0] for a in m.lora_A)
            byts += sum(t.numel() * 2 for A, B in zip(m.lora_A, m.lora_B) for t in (A, B))
        rank += r_tot
        flops += 2 * r_tot * (m.in_features + m.out_features)
    return rank, byts, flops


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--method', choices=['seqlora', 'svdlora', 'olora'], default='svdlora')
    p.add_argument('--tasks_dir', default='natural-instructions-2.8/tasks')
    p.add_argument('--model_name', default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument('--num_tasks', type=int, default=750)
    p.add_argument('--min_instances', type=int, default=560,
                   help="only sample tasks with >= this many instances (fixes the pool)")
    p.add_argument('--train_size', type=int, default=500)
    p.add_argument('--val_size', type=int, default=10)
    p.add_argument('--test_size', type=int, default=50)
    p.add_argument('--max_length', type=int, default=1024)
    p.add_argument('--max_instruction_tokens', type=int, default=1024)
    p.add_argument('--num_epochs', type=int, default=1)
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--eval_batch_size', type=int, default=32)
    p.add_argument('--lr', type=float, default=5e-5)
    p.add_argument('--device', default="cuda")
    p.add_argument('--seed', type=int, default=1993)
    p.add_argument('--order_seed', type=int, default=1993)
    p.add_argument('--set_seed', type=int, default=0)
    p.add_argument('--out_dir', default="run_logs/longhorizon")
    p.add_argument('--lora_r', type=int, default=8)
    p.add_argument('--lora_alpha', type=float, default=32.0)
    p.add_argument('--lora_dropout', type=float, default=0.1)
    p.add_argument('--target_modules', default="q_proj,v_proj")
    # svdlora
    p.add_argument('--svd_energy_target', type=float, default=0.005)
    p.add_argument('--svd_oversampling', type=int, default=10)
    # olora
    p.add_argument('--lamda_1', type=float, default=0.5)
    # cumulative eval cadence (tasks)
    p.add_argument('--eval_every_tasks', type=int, default=50)
    # boundary regime
    p.add_argument('--boundary_mode', choices=['task', 'sample'], default='task')
    p.add_argument('--boundary_samples_mult', type=float, default=2.5,
                   help="sample mode: boundary every round(mult*train_size) streamed samples")
    args = p.parse_args()

    set_random_seed(args.seed)
    B = int(round(args.boundary_samples_mult * args.train_size))   # sample-mode boundary size
    print("=" * 72)
    print(f"LONG-HORIZON CL | {args.method} | lr={args.lr} | boundary={args.boundary_mode}"
          + (f" (every {B} samples)" if args.boundary_mode == 'sample' else " (every task)"))
    print(f"{args.model_name} | {args.num_tasks} tasks (>= {args.min_instances} inst) | "
          f"seed/order {args.seed}/{args.order_seed} | eval every {args.eval_every_tasks} tasks")
    print("=" * 72, flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.bos_token
    tokenizer.padding_side = "left"

    train_data, val_data, test_data, _ = sample_natural_instructions_tasks(
        tasks_dir=args.tasks_dir, num_tasks=args.num_tasks,
        max_instruction_tokens=args.max_instruction_tokens, tokenizer=tokenizer,
        stable_test_split=True, train_size=args.train_size, val_size=args.val_size,
        test_size=args.test_size, few_shot=False,
        set_seed=args.set_seed, order_seed=args.order_seed, min_instances=args.min_instances)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16, device_map=args.device)
    targets = tuple(args.target_modules.split(","))
    if args.method == 'olora':
        modules = inject_multislot_lora(model, target_modules=targets, r=args.lora_r,
                                        alpha=args.lora_alpha, dropout=args.lora_dropout,
                                        collect_cov=False)
    else:
        modules = inject_svdlora(model, target_modules=targets, r=args.lora_r,
                                 r_hat=args.lora_r, alpha=args.lora_alpha,
                                 oversampling=args.svd_oversampling, dropout=args.lora_dropout,
                                 energy_target=(args.svd_energy_target if args.method == 'svdlora' else None))
    print(f"Injected {len(modules)} adapter modules", flush=True)

    train_tasks = group_by_task(train_data)
    n_tasks = len(train_tasks)
    # ordered cumulative test groups (same order as training tasks)
    test_by_name = {}
    for ex in test_data:
        test_by_name.setdefault(ex['tasks'][0], []).append(ex)
    ordered_test = [test_by_name.get(name, []) for name, _ in train_tasks]
    print(f"Training over {n_tasks} tasks (boundary={args.boundary_mode})", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    tag = (f"{args.method}_lr{args.lr}_{args.boundary_mode}_{n_tasks}t_s{args.order_seed}")
    out_path = f"{args.out_dir}/longhorizon_{tag}.json"
    meta = {"method": args.method, "lr": args.lr, "boundary_mode": args.boundary_mode,
            "boundary_samples": B if args.boundary_mode == 'sample' else None,
            "boundary_samples_mult": args.boundary_samples_mult,
            "model": args.model_name, "n_tasks": n_tasks, "order_seed": args.order_seed,
            "seed": args.seed, "train_size": args.train_size, "test_size": args.test_size,
            "lora_r": args.lora_r, "lora_alpha": args.lora_alpha,
            "svd_energy_target": args.svd_energy_target if args.method == 'svdlora' else None,
            "lamda_1": args.lamda_1 if args.method == 'olora' else None,
            "eval_every_tasks": args.eval_every_tasks, "status": "running",
            "per_task_train_seconds": [], "checkpoints": []}
    report.write_metrics(out_path, meta)

    # ---- chunk machinery (a "chunk" = the span between two boundaries) ----
    state = {"optim": None, "cur_slot": -1, "n_boundaries": 0}

    def begin_chunk():
        if args.method == 'olora':
            for m in modules:
                m.add_task()
            state["cur_slot"] += 1
            for m in modules:
                m.set_trainable(state["cur_slot"], train_a=True)
            trainable = multislot_trainable_parameters(modules)
        else:
            trainable = svdlora_trainable_parameters(modules)
        state["optim"] = torch.optim.AdamW(trainable, lr=args.lr)

    def end_chunk():
        if args.method == 'svdlora':
            compress_all(modules)            # fold residual -> sketch, reset residual
        # seqlora: drift continues (no compress); olora: fold happens at next set_trainable
        state["n_boundaries"] += 1

    def olora_extra_loss():
        return args.lamda_1 * orthogonality_penalty(modules, state["cur_slot"])

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    train_seconds = 0.0
    last_ckpt_train_seconds = 0.0   # train_seconds at the previous checkpoint (for per-interval delta)

    def do_eval(tasks_seen):
        nonlocal train_seconds, last_ckpt_train_seconds
        seen = [ex for grp in ordered_test[:tasks_seen] for ex in grp]
        model.eval()
        t0 = time.time()
        results, _ = evaluate_with_generation(
            model=model, tokenizer=tokenizer, test_examples=seen, device=args.device,
            max_new_tokens=256, batch_size=args.eval_batch_size)
        eval_s = time.time() - t0
        model.train()
        rank, byts, flops = deployed_now(modules, args.method, args.lora_r)
        n_seen = max(len(seen), 1)
        train_delta = train_seconds - last_ckpt_train_seconds
        last_ckpt_train_seconds = train_seconds
        rec = {"tasks_seen": tasks_seen, "n_test": len(seen),
               "samples_trained": state.get("samples", 0), "n_boundaries": state["n_boundaries"],
               "rougeL": results.get("rougeL"), "exact_match": results.get("exact_match"),
               "deployed_rank_total": rank, "adapter_mb": round(byts / 1024 / 1024, 4),
               "inference_flops_per_token": flops,
               "train_seconds_cumulative": round(train_seconds, 1),
               "train_seconds_interval": round(train_delta, 1),
               "eval_seconds": round(eval_s, 1),
               "inference_seconds_per_example": round(eval_s / n_seen, 4),
               "inference_ms_per_example": round(1000.0 * eval_s / n_seen, 2),
               "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 1024 / 1024, 1)
               if torch.cuda.is_available() else None,
               "per_task": results.get("per_task")}
        meta["checkpoints"].append(rec)
        report.write_metrics(out_path, meta)
        print(f"  [eval @ {tasks_seen} tasks] ROUGE-L {rec['rougeL']:.2f} | EM {rec['exact_match']:.2f} "
              f"| rank {rank} | {rec['adapter_mb']:.2f}MB | boundaries {state['n_boundaries']} "
              f"| train {train_seconds/60:.1f}m | eval {eval_s/60:.1f}m", flush=True)

    # ---- stream ----
    try:
        begin_chunk()
        samples_seen = 0
        next_boundary = B
        for ti, (task_name, samples) in enumerate(train_tasks):
            loader = build_loader(samples, tokenizer, args, shuffle=True)
            task_train_seconds = 0.0      # wall-clock of this task's single epoch
            for batch in loader:
                t0 = time.time()
                batch = {k: v.to(args.device) for k, v in batch.items()}
                loss = model(**batch).loss
                if args.method == 'olora':
                    loss = loss + olora_extra_loss()
                loss.backward()
                state["optim"].step(); state["optim"].zero_grad()
                samples_seen += batch["input_ids"].size(0)
                state["samples"] = samples_seen
                dt = time.time() - t0
                train_seconds += dt
                task_train_seconds += dt
                # sample-mode boundary may fire mid-task
                if args.boundary_mode == 'sample' and samples_seen >= next_boundary:
                    end_chunk(); begin_chunk()
                    next_boundary += B
            meta["per_task_train_seconds"].append(round(task_train_seconds, 2))
            if args.boundary_mode == 'task':
                end_chunk()
            # cumulative eval at task increments (learner unaware in sample mode)
            if (ti + 1) % args.eval_every_tasks == 0 or (ti + 1) == n_tasks:
                do_eval(ti + 1)
            if args.boundary_mode == 'task' and (ti + 1) != n_tasks:
                begin_chunk()

        meta["status"] = "done"
        meta["train_seconds_total"] = round(train_seconds, 1)
        report.write_metrics(out_path, meta)
        final = meta["checkpoints"][-1]
        print(f"\nDONE | {tag} | final ROUGE-L {final['rougeL']:.2f} EM {final['exact_match']:.2f} "
              f"| deployed rank {final['deployed_rank_total']} | {final['adapter_mb']:.2f}MB "
              f"| train {train_seconds/3600:.2f}h", flush=True)

    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if not isinstance(e, torch.cuda.OutOfMemoryError) and "out of memory" not in str(e).lower():
            raise
        print(f"\n[OOM] {tag} ended early; {len(meta['checkpoints'])} checkpoints saved.")
        meta["status"] = "oom"; meta["error"] = str(e)[:300]
        report.write_metrics(out_path, meta)
        import sys
        sys.exit(0)


if __name__ == "__main__":
    main()
