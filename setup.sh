#!/bin/bash
# One-shot cluster setup for svd_sketching_language.
#   1. install deps   2. cache the (gated) Llama-3.2-3B checkpoint   3. fetch the NI dataset
#
# Prereqs:
#   - conda env active (e.g. `conda activate treelora`)
#   - Llama-3.2 access granted on HuggingFace; authenticate first:
#       export HF_TOKEN=hf_xxx        (or: huggingface-cli login)
#   - optionally point the HF cache at scratch: export HF_HOME=/scratch/$USER/hf_cache
set -e
cd "$(dirname "$0")"

echo "[1/3] installing requirements..."
pip install -r requirements.txt

echo "[2/3] pre-fetching Llama-3.2-3B-Instruct (gated; needs HF_TOKEN)..."
python scripts/prefetch_models.py --models 3B

echo "[3/3] fetching Super-NaturalInstructions into ./data ..."
python scripts/data_prep.py

# SBATCH opens --output (under run_logs/) before the job script runs, so it must pre-exist.
mkdir -p tokmem/atomic/run_logs

echo
echo "Setup complete."
echo "  NI tasks:  data/natural-instructions-2.8/tasks  (atomic harness reads it via the relative symlink)"
echo "  Run the 1000-task experiment:  cd tokmem/atomic && mkdir -p run_logs && sbatch run_experiments.slurm"
