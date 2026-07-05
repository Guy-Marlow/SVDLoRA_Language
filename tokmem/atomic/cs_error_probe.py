"""CountSketch error probe: 10 real tasks, measured vs THEORY error + per-op timing.

For each boundary and module, records:
  measured   ||dW - kept||_F / ||dW||_F          (one hash draw)
  predicted  sqrt( (1/k) sum_{i!=j}(g_i g_j + Gb_ij Ga_ij) ) / ||dW||_F   (closed form)
  truth      ||kept - TRUE accumulated delta|| / ||TRUE||   (compounding across boundaries)
  cs_ms      core sketch op wall time (concat+rebalance+hash+scatter, cuda-synced)

Theory check: measured^2 is a random variable with mean predicted^2, so over
56 modules x 10 boundaries the ratio  mean(measured^2)/mean(predicted^2)  should be ~1.
Also times a randsvd compress_all on the same trained state for the ablation ETA.

Run: CUDA_VISIBLE_DEVICES=<gpu> python cs_error_probe.py
Output: run_logs/cs_smoke/cs_probe_10t.json + printed table.
"""
import argparse
import json
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from svdlora_layer import inject_svdlora, svdlora_trainable_parameters, compress_all
from task_dataset import sample_natural_instructions_tasks
from main_svdlora_baseline import group_by_task, build_loader, train_one_task


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--num_tasks', type=int, default=10)
    p.add_argument('--train_size', type=int, default=500)
    p.add_argument('--cs_rank', type=int, default=8)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--order_seed', type=int, default=1993)
    p.add_argument('--out', default="run_logs/cs_smoke/cs_probe_10t.json")
    # harness-compat args consumed by build_loader / train_one_task
    p.add_argument('--model_name', default="meta-llama/Llama-3.2-3B-Instruct")
    p.add_argument('--tasks_dir', default="natural-instructions-2.8/tasks")
    p.add_argument('--max_instruction_tokens', type=int, default=1000)
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--gradient_accumulation_steps', type=int, default=1)
    p.add_argument('--num_epochs', type=int, default=1)
    p.add_argument('--lr', type=float, default=5e-5)
    p.add_argument('--ref_lr_div', action='store_true')
    p.add_argument('--max_length', type=int, default=1024)
    p.add_argument('--device', default="cuda")
    args = p.parse_args()
    torch.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.bos_token
    tokenizer.padding_side = "left"

    train_data, _, _, _ = sample_natural_instructions_tasks(
        tasks_dir=args.tasks_dir, num_tasks=args.num_tasks,
        max_instruction_tokens=args.max_instruction_tokens, tokenizer=tokenizer,
        stable_test_split=True, train_size=args.train_size, val_size=10, test_size=10,
        few_shot=False, set_seed=0, order_seed=args.order_seed)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16, device_map=args.device)
    modules = inject_svdlora(model, r=8, r_hat=8, alpha=32.0, oversampling=10,
                             dropout=0.1, merge_op="countsketch",
                             cs_rank=args.cs_rank, cs_seed=args.seed)
    for m in modules:
        m.cs_probe = True
    print(f"injected {len(modules)} modules | countsketch k={args.cs_rank} | probe ON")

    tasks = group_by_task(train_data)
    records = []
    for i, (task_name, samples) in enumerate(tasks):
        loader = build_loader(samples, tokenizer, args, shuffle=True)
        t0 = time.perf_counter()
        train_one_task(model, svdlora_trainable_parameters(modules), loader, args)
        train_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        compress_all(modules, task_idx=i)
        boundary_s = time.perf_counter() - t0

        meas = np.array([m.last_merge_relerr for m in modules])
        pred = np.array([m.last_cs_pred_relerr for m in modules])
        truth = np.array([m.last_truth_relerr for m in modules])
        csms = np.array([m.last_cs_ms for m in modules])
        ratio = float((meas ** 2).mean() / (pred ** 2).mean())
        rec = {"task": i, "name": task_name, "train_s": round(train_s, 1),
               "boundary_s": round(boundary_s, 3),
               "cs_core_ms_sum": round(float(csms.sum()), 2),
               "cs_core_ms_per_module": round(float(csms.mean()), 4),
               "measured_relerr_mean": round(float(meas.mean()), 4),
               "predicted_relerr_mean": round(float(pred.mean()), 4),
               "meas2_over_pred2": round(ratio, 4),
               "truth_relerr_mean": round(float(truth.mean()), 4),
               "truth_relerr_max": round(float(truth.max()), 4),
               "measured": meas.round(4).tolist(), "predicted": pred.round(4).tolist(),
               "truth": truth.round(4).tolist()}
        records.append(rec)
        print(f"[task {i}] train {train_s:5.1f}s | boundary {boundary_s*1000:7.1f}ms "
              f"(cs core {csms.sum():6.1f}ms) | meas {meas.mean():.3f} pred {pred.mean():.3f} "
              f"ratio(m2/p2) {ratio:.3f} | vs-truth {truth.mean():.3f}", flush=True)

    # overall theory check across all boundaries/modules
    all_m = np.concatenate([np.array(r["measured"]) for r in records])
    all_p = np.concatenate([np.array(r["predicted"]) for r in records])
    overall = float((all_m ** 2).mean() / (all_p ** 2).mean())
    print(f"\nOVERALL theory check: mean(measured^2)/mean(predicted^2) = {overall:.4f} "
          f"over {all_m.size} (module,boundary) samples  [expect ~1]")

    # randsvd timing reference on the SAME final state (one compress_all)
    for m in modules:
        m.merge_op = "randsvd"
        m.cs_probe = False
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    compress_all(modules, task_idx=len(tasks))
    torch.cuda.synchronize()
    randsvd_s = time.perf_counter() - t0
    print(f"randsvd compress_all reference (same state): {randsvd_s*1000:.1f}ms total")

    out = {"config": {"num_tasks": args.num_tasks, "train_size": args.train_size,
                      "cs_rank": args.cs_rank, "seed": args.seed,
                      "order_seed": args.order_seed},
           "overall_meas2_over_pred2": overall,
           "randsvd_compress_all_ms": randsvd_s * 1000.0,
           "boundaries": records}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()