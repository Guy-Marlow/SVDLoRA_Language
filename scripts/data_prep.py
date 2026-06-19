#!/usr/bin/env python3
"""
Download the Natural Instructions (Super-NaturalInstructions) dataset into ./data,
matching the layout the tokmem atomic loaders expect.

The atomic code reads task JSONs from `natural-instructions-2.8/tasks/` (a relative
symlink -> ../../data/natural-instructions-2.8). This script materializes that target
by cloning the official allenai repo (download-to-local; loaders untouched).

Usage:
  python scripts/data_prep.py                       # shallow clone into ./data
  python scripts/data_prep.py --data_root /scratch/$USER/data
  python scripts/data_prep.py --full                # full history (large; default is --depth 1)
Requires: git
"""
import argparse
import os
import subprocess
import sys

REPO = "https://github.com/allenai/natural-instructions"
DIRNAME = "natural-instructions-2.8"   # name the atomic symlink points at


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="./data")
    p.add_argument("--full", action="store_true", help="full clone (default is --depth 1 shallow)")
    args = p.parse_args()

    os.makedirs(args.data_root, exist_ok=True)
    dest = os.path.join(args.data_root, DIRNAME)
    tasks = os.path.join(dest, "tasks")
    if os.path.isdir(tasks) and len(os.listdir(tasks)) > 1000:
        print(f"[ni] already present at {dest} ({len(os.listdir(tasks))} task files). Skipping.")
        return

    cmd = ["git", "clone"]
    if not args.full:
        cmd += ["--depth", "1"]
    cmd += [REPO, dest]
    print(f"[ni] {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        sys.exit(f"clone failed: {e}")
    n = len(os.listdir(tasks)) if os.path.isdir(tasks) else 0
    print(f"[ni] done -> {dest} ({n} task files)")


if __name__ == "__main__":
    main()
