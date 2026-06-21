"""
Performance reporting for the language CL experiments: how much memory each method's
adapter occupies, and how many FLOPs the adapter adds at inference.

Memory and FLOPs are computed **from the deployed adapter** — the part that actually
contributes to the output at inference — NOT merely from how many parameters are stored.
This matters because some methods keep a large adapter bank but do not all of it at
inference (e.g. a bank kept only for a training regularizer): for those the inference
FLOPs should reflect what is *applied*, not what is *stored*.

Per method (current roster), the deployed/contributing adapter is:
  - seqlora : the single residual adapter (rank r)                  -> SVDLoRALinear.lora_A/B
  - svdlora : the frozen sketch (rank r_hat; per-module for adaptive) -> SVDLoRALinear.sketch_*
              (the residual is reset to 0 after every task, so it contributes nothing at eval)
  - olora   : the full per-task bank, summed at inference (rank K*r) -> MultiSlotLoRALinear.lora_A/B
  - inflora : same bank structure as olora
NOTE for future methods (TreeLoRA/HiDeLoRA): if the bank is NOT applied at inference, add a
branch returning the *deployed* contributing rank (e.g. 0 or the single routed adapter), not K*r.
"""
import json


def _module_deployed_rank(m, method):
    """Per-module contributing rank of the deployed adapter at inference."""
    if method == "svdlora":
        return m.sketch_B.shape[1]                       # r_hat (adaptive: per module)
    if method == "seqlora":
        return m.lora_A.shape[0]                         # residual rank r
    # olora / inflora: forward sums the whole bank -> all slots contribute
    return sum(a.shape[0] for a in m.lora_A)             # K * r


def _module_deployed_tensors(m, method):
    """The tensors that constitute the deployed adapter (for byte accounting)."""
    if method == "svdlora":
        return [m.sketch_B, m.sketch_A]
    if method == "seqlora":
        return [m.lora_A, m.lora_B]
    return [t for A, B in zip(m.lora_A, m.lora_B) for t in (A, B)]


def adapter_memory_bytes(modules, method):
    """Bytes occupied by the deployed adapter (sum over modules)."""
    return sum(t.numel() * t.element_size()
               for m in modules for t in _module_deployed_tensors(m, method))


def adapter_inference_flops_per_token(modules, method):
    """FLOPs the adapter adds per token at inference. A rank-r factor pair applied to a
    [d_in]->[d_out] projection costs 2*r*(d_in+d_out) (two matmuls: x@A^T then (·)@B^T)."""
    return sum(2 * _module_deployed_rank(m, method) * (m.in_features + m.out_features)
               for m in modules)


def adapter_report(modules, method, lora_r, dtype_bytes=2):
    """Build the adapter memory/FLOPs report dict for one run's final state."""
    mem = adapter_memory_bytes(modules, method)
    flops = adapter_inference_flops_per_token(modules, method)
    # baseline: a single rank-`lora_r` adapter across the same modules -> express as a multiple
    one_adapter_flops = sum(2 * lora_r * (m.in_features + m.out_features) for m in modules)
    ranks = [_module_deployed_rank(m, method) for m in modules]
    return {
        "method": method,
        "adapter_memory_bytes": int(mem),
        "adapter_memory_mb": round(mem / 1024 / 1024, 4),
        "adapter_inference_flops_per_token": int(flops),
        "adapter_flops_x_one_rank_r": round(flops / one_adapter_flops, 3) if one_adapter_flops else None,
        "deployed_rank_total": int(sum(ranks)),
        "deployed_rank_mean": round(sum(ranks) / len(ranks), 2) if ranks else 0,
        "n_modules": len(modules),
    }


def write_metrics(path, payload):
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
