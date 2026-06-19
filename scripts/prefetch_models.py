#!/usr/bin/env python3
"""
Pre-download the (gated) Llama-3.2 Instruct checkpoints into the HF cache, so compute
nodes don't fetch at training start.

Llama-3.2 is GATED on HuggingFace. Authenticate first (one of):
    export HF_TOKEN=hf_xxx                 # read-token with Llama-3.2 access granted
    # or: huggingface-cli login
Point the cache at shared/scratch storage:
    export HF_HOME=/scratch/$USER/hf_cache
Then:
    python scripts/prefetch_models.py                       # 3B (default)
    python scripts/prefetch_models.py --models 1B 3B        # both

The atomic code loads these via AutoModelForCausalLM.from_pretrained(model_name).
"""
import argparse
import os

IDS = {
    "1B": "meta-llama/Llama-3.2-1B-Instruct",
    "3B": "meta-llama/Llama-3.2-3B-Instruct",
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=["3B"], choices=list(IDS))
    args = p.parse_args()

    from huggingface_hub import snapshot_download
    token = os.environ.get("HF_TOKEN")
    if token is None:
        print("WARNING: HF_TOKEN not set; relying on cached `huggingface-cli login`. "
              "Llama-3.2 is gated and will fail without access.")
    print(f"HF_HOME={os.environ.get('HF_HOME', '~/.cache/huggingface')}")
    for m in args.models:
        repo = IDS[m]
        print(f"prefetching {repo} ...")
        snapshot_download(repo_id=repo, token=token,
                          allow_patterns=["*.json", "*.safetensors", "*.model", "tokenizer*"])
        print(f"  cached {repo}")
    print("done.")


if __name__ == "__main__":
    main()
