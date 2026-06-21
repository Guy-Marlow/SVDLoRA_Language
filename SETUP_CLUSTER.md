# Cluster setup — svd_sketching_language

Long-horizon language CL bench: SVDLoRA vs the TokMem atomic-recall LoRA baseline on
Super-NaturalInstructions. Active code is in `tokmem/atomic/`. No data or model weights
are committed. `internbootcamp/` (a second, not-yet-used benchmark) is gitignored.

## 0. One-shot setup
After `conda activate <env>` and authenticating to HuggingFace (`export HF_TOKEN=hf_xxx`,
Llama-3.2 access granted), run:
```bash
bash setup.sh      # installs deps, caches Llama-3.2-3B, fetches Super-NaturalInstructions
```
Then launch the headline experiment (12 runs = 4 methods × 3 seeds, 1000 tasks):
```bash
cd tokmem/atomic && mkdir -p run_logs && sbatch run_experiments.slurm
```
Results land in `tokmem/atomic/run_logs/`: `exp_<method>_<seed>.out` (logs) and
`metrics_<method>_<tag>.json` (final ROUGE-L/EM, deployed-adapter memory bytes, inference
FLOPs/token, peak VRAM, per-task memory curve, run `status`). The manual steps below are the
breakdown of `setup.sh`.

## 1. Environment
```bash
pip install -r requirements.txt
```

## 2. Model checkpoints (gated Llama-3.2)
Llama-3.2 is gated on HuggingFace — request access, then authenticate and pre-fetch:
```bash
export HF_TOKEN=hf_xxx                  # token with Llama-3.2 access (or: huggingface-cli login)
export HF_HOME=/scratch/$USER/hf_cache  # persist the download cache
python scripts/prefetch_models.py --models 3B     # or: 1B 3B
```

## 3. Dataset (Super-NaturalInstructions)
```bash
python scripts/data_prep.py                        # shallow-clones allenai/natural-instructions into ./data
# custom location: --data_root /scratch/$USER/data
```
`tokmem/atomic/natural-instructions-2.8` is a relative symlink to `data/natural-instructions-2.8`,
so the atomic loaders (`--tasks_dir natural-instructions-2.8/tasks`) work once ./data is populated.

## 4. Run (from tokmem/atomic/)
```bash
cd tokmem/atomic
# LoRA baseline (paper-faithful: pass lr 2e-4 -> stock code's /grad_accum yields effective 5e-5):
bash repro_lora_check.sh 10 0 2e-4
# SVDLoRA vs SeqLoRA (single adapter, no routing, P=1 compression, effective lr 5e-5):
bash run_svdlora_compare.sh 0 10
```
Notes:
- The stock LoRA `train_lora_model` divides LR by gradient_accumulation_steps; SVDLoRA sets
  LR directly. Use effective 5e-5 for all LoRA-baseline comparisons.
- GPU gotcha: `CUDA_VISIBLE_DEVICES` enumeration may not match `nvidia-smi` indices; verify
  the selected device is the intended GPU (or set `CUDA_DEVICE_ORDER=PCI_BUS_ID`).
