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


def _module_deployed_delta(m, method):
    """Reconstruct the actual [out, in] weight delta the module ADDS at inference.
    seqlora : scale * B A            (single residual; sketch is zero)
    svdlora : sketch_B @ sketch_A    (scale baked into the factors; residual reset to no-op)
    olora/inflora : frozen_delta + scale * B_cur A_cur   (the full summed bank; frozen_delta
                    already holds every folded prior adapter)."""
    import torch
    if method == "svdlora":
        return m.sketch_B.float() @ m.sketch_A.float()
    if method == "seqlora":
        return m.scale * (m.lora_B.float() @ m.lora_A.float())
    # olora / inflora
    D = m.frozen_delta.float().clone()
    if getattr(m, "_cur_idx", None) is not None and m._cur_idx < len(m.lora_A):
        D = D + m.scale * (m.lora_B[m._cur_idx].float() @ m.lora_A[m._cur_idx].float())
    return D


def measure_deployed_rank(modules, method, rtol=1e-3, abstol=1e-6):
    """MEASURE (not estimate) the actual numerical rank of each module's deployed weight
    delta via SVD, and aggregate. Returns nominal (sum-of-block-ranks) vs measured rank so
    the two are never conflated. Run once at end of training (a full SVD per module)."""
    import torch
    per = []   # (out, in, cap, nominal, rank_rel, rank_abs, r99, r999, fro)
    with torch.no_grad():
        for m in modules:
            D = _module_deployed_delta(m, method)
            out, inn = D.shape
            cap = min(out, inn)
            s = torch.linalg.svdvals(D.to(torch.float32))      # descending
            smax = float(s[0]) if s.numel() else 0.0
            rank_rel = int((s > rtol * smax).sum().item()) if smax > 0 else 0
            rank_abs = int((s > abstol).sum().item())
            e = s.pow(2); tot = float(e.sum())
            if tot > 0:
                cum = torch.cumsum(e, 0) / tot
                r99 = int((cum < 0.99).sum().item()) + 1
                r999 = int((cum < 0.999).sum().item()) + 1
            else:
                r99 = r999 = 0
            nominal = _module_deployed_rank(m, method)
            per.append((out, inn, cap, nominal, rank_rel, rank_abs, r99, r999, tot ** 0.5))
    def _tot(i): return int(sum(p[i] for p in per))
    # split by projection shape (q_proj is square out==in; v_proj is the GQA-narrowed one)
    q = [p for p in per if p[0] == p[1]]; v = [p for p in per if p[0] != p[1]]
    return {
        "method": method, "n_modules": len(per), "rtol": rtol,
        "nominal_rank_total": _tot(3),                 # sum of K*r block ranks (what deployed_rank_total reports)
        "measured_rank_total": _tot(4),                # sum of numerical ranks (sigma > rtol*sigma_max)
        "measured_rank_exact_total": _tot(5),          # sigma > abstol
        "energy99_rank_total": _tot(6), "energy999_rank_total": _tot(7),
        "measured_rank_mean": round(_tot(4) / len(per), 2) if per else 0,
        "q_proj": {"n": len(q), "shape": (q[0][0], q[0][1]) if q else None,
                   "cap_each": q[0][2] if q else None,
                   "nominal_each": q[0][3] if q else None,
                   "measured_mean": round(sum(p[4] for p in q) / len(q), 2) if q else None,
                   "measured_max": max((p[4] for p in q), default=None)},
        "v_proj": {"n": len(v), "shape": (v[0][0], v[0][1]) if v else None,
                   "cap_each": v[0][2] if v else None,
                   "nominal_each": v[0][3] if v else None,
                   "measured_mean": round(sum(p[4] for p in v) / len(v), 2) if v else None,
                   "measured_max": max((p[4] for p in v), default=None)},
    }


def write_metrics(path, payload):
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
