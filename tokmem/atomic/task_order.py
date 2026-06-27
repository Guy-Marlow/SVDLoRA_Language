"""
Deterministic task selection + ordering for the multi-seed experiments.

We want 3 fixed task ORDERS (A/B/C) that are:
  - permutations of the SAME fixed task SET (so all runs/methods see identical tasks),
  - unique across the 3 runs,
  - identical across methods for a given run.

`ordered_task_names` decouples SET selection from ORDER:
  1. select the fixed `num_tasks` set with a constant `set_seed` (run-independent),
  2. permute that fixed set with `order_seed` (= the run seed 1993/1994/1995).
Both stages use local RNGs, so the order does not depend on the global training seed
(which still seeds dropout / data shuffling during training).

`order_seed=None` reproduces the original behaviour (hierarchical global shuffle) for
backward compatibility.

CLI (verification): `python task_order.py --tasks_dir <dir> --num_tasks 1000`
prints the first tasks of orders A/B/C and asserts: same set, distinct orders.
"""
import argparse
import json
import os
import random

DEFAULT_SET_SEED = 0   # constant -> the fixed 1000-task set is the same for every run


def english_task_files(tasks_dir, min_instances=0):
    """Sorted list of task json filenames whose input AND output languages are English.

    min_instances>0 additionally requires the task to have at least that many Instances
    (used to fix a pool of tasks large enough for a given train+val+test footprint, e.g.
    min_instances=560 for train 500 / val 10 / test 50). Backward compatible (default 0)."""
    if not os.path.exists(tasks_dir):
        raise FileNotFoundError(f"Tasks directory not found: {tasks_dir}")
    out = []
    for fn in os.listdir(tasks_dir):
        if not (fn.startswith("task") and fn.endswith(".json")):
            continue
        try:
            with open(os.path.join(tasks_dir, fn)) as f:
                d = json.load(f)
        except Exception:
            continue
        il, ol = d.get("Input_language", []), d.get("Output_language", [])
        if not (("English" in il or il == ["English"]) and ("English" in ol or ol == ["English"])):
            continue
        if min_instances and len(d.get("Instances", [])) < min_instances:
            continue
        out.append(fn)
    return sorted(out)


def ordered_task_names(tasks_dir, num_tasks, set_seed=DEFAULT_SET_SEED, order_seed=None,
                       english_files=None, min_instances=0):
    """Return the ordered list of task json filenames for a run.

    english_files: optional precomputed list from english_task_files() to avoid re-scanning.
    min_instances: only used when english_files is None (forwarded to english_task_files).
    """
    files = english_files if english_files is not None else english_task_files(tasks_dir, min_instances)
    files = sorted(files)
    n = min(num_tasks, len(files))
    if order_seed is None:
        # backward-compatible: hierarchical shuffle via the global RNG (seeded in main)
        shuffled = list(files)
        random.shuffle(shuffled)
        return shuffled[:n]
    # experiment mode: fixed set (set_seed) then per-run permutation (order_seed)
    base = list(files)
    random.Random(set_seed).shuffle(base)
    fixed_set = base[:n]
    random.Random(order_seed).shuffle(fixed_set)
    return fixed_set


def _main():
    p = argparse.ArgumentParser()
    p.add_argument("--tasks_dir", default="natural-instructions-2.8/tasks")
    p.add_argument("--num_tasks", type=int, default=1000)
    p.add_argument("--set_seed", type=int, default=DEFAULT_SET_SEED)
    p.add_argument("--seeds", type=int, nargs="+", default=[1993, 1994, 1995])
    args = p.parse_args()

    eng = english_task_files(args.tasks_dir)
    print(f"English tasks available: {len(eng)} | requesting {args.num_tasks}")
    orders = {s: ordered_task_names(args.tasks_dir, args.num_tasks, args.set_seed, s, eng)
              for s in args.seeds}
    for s, o in orders.items():
        print(f"order seed {s}: first6={[n[:18] for n in o[:6]]}")
    # checks
    sets = [frozenset(o) for o in orders.values()]
    assert all(x == sets[0] for x in sets), "SET differs across seeds!"
    seqs = [tuple(o) for o in orders.values()]
    assert len(set(seqs)) == len(seqs), "orders not all distinct!"
    print(f"OK: all {len(args.seeds)} orders are permutations of the SAME "
          f"{len(sets[0])}-task set, and all distinct.")


if __name__ == "__main__":
    _main()
