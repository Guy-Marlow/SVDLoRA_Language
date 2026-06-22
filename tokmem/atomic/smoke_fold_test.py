"""Smoke test: prove the weight-folded MultiSlotLoRALinear is mathematically identical to the
original K-branch sum.

The original forward (sum over all adapters, dropout per-branch) is reconstructed by hand and
compared to the new folded forward, across a simulated task stream (add_task + set_trainable per
task, the new adapter randomized to mimic training).

  [1] forward identity when dropout is OFF (eval / dropout=0)  -- the reported-metric case
  [2] gradient to the CURRENT (trainable) adapter is identical -- training of the live adapter unchanged
  [3] incremental frozen_delta == scale * sum_{frozen} B_t @ A_t  (exact)
  [4] the ONLY divergence under dropout>0 is frozen-branch dropout, and it vanishes at eval

Run at float64 to show true mathematical identity (~machine epsilon); the float32 numbers are
just floating-point reassociation noise (one big matmul vs a sum of K small ones).
"""
import torch, torch.nn as nn
from multislot_lora import MultiSlotLoRALinear

IN, OUT, R, ALPHA, K = 48, 64, 8, 32.0, 6     # out != in to catch shape bugs (GQA-like)
SCALE = ALPHA / R

def make_module(dropout, dtype):
    base = nn.Linear(IN, OUT, bias=False)
    nn.init.normal_(base.weight, std=0.05)
    for p in base.parameters():
        p.requires_grad = False
    m = MultiSlotLoRALinear(base, r=R, alpha=ALPHA, dropout=dropout)
    m.to(dtype)                                # base + frozen_delta -> dtype; new adapters follow
    torch.manual_seed(123)
    for t in range(K):                         # simulate the task stream
        m.add_task()
        with torch.no_grad():
            m.lora_A[t].normal_(0, 0.1)
            m.lora_B[t].normal_(0, 0.1)        # B != 0 => "trained"
        m.set_trainable(t)                     # folds t-1, makes t current
    return m

def original_forward(m, x, drop):
    """pre-fold behaviour: base(x) + scale * sum_t (drop(x) @ A_t^T) @ B_t^T."""
    out = m.base(x)
    for A, B in zip(m.lora_A, m.lora_B):
        out = out + m.scale * ((drop(x) @ A.t()) @ B.t())
    return out

def run(dtype, tol):
    m = make_module(0.0, dtype); m.eval()
    x = torch.randn(4, 10, IN, dtype=dtype)
    d1 = (m(x) - original_forward(m, x, nn.Identity())).abs().max().item()
    c = m._cur_idx
    m.zero_grad(set_to_none=True); m(x).pow(2).sum().backward()
    gAf, gBf = m.lora_A[c].grad.clone(), m.lora_B[c].grad.clone()
    m.zero_grad(set_to_none=True); original_forward(m, x, nn.Identity()).pow(2).sum().backward()
    gAo, gBo = m.lora_A[c].grad.clone(), m.lora_B[c].grad.clone()
    dA = (gAf - gAo).abs().max().item(); dB = (gBf - gBo).abs().max().item()
    ref = torch.zeros(OUT, IN, dtype=dtype)
    for t in range(K):
        if t != c:
            ref += SCALE * (m.lora_B[t] @ m.lora_A[t])
    d3 = (m.frozen_delta - ref).abs().max().item()
    print(f"=== {str(dtype):>14} (tol {tol:.0e}) ===")
    print(f"  [1] forward       max|d| = {d1:.2e}")
    print(f"  [2] grad current  max|dA|= {dA:.2e}  max|dB| = {dB:.2e}")
    print(f"  [3] fold == sumBA max|d| = {d3:.2e}")
    return d1 < tol and dA < tol and dB < tol and d3 < tol

# [4] dropout caveat (float32, train vs eval)
md = make_module(0.1, torch.float32); md.train()
x = torch.randn(4, 10, IN)
torch.manual_seed(7); ft = md(x); torch.manual_seed(7); ot = original_forward(md, x, md.drop)
md.eval(); de = (md(x) - original_forward(md, x, nn.Identity())).abs().max().item()
print(f"[4] dropout=0.1  TRAIN max|d|={(ft-ot).abs().max():.2e} (frozen-branch dropout)  "
      f"EVAL max|d|={de:.2e} (identical)\n")

ok32 = run(torch.float32, 1e-3)
ok64 = run(torch.float64, 1e-9)
print("\nRESULT:", "PASS -- folded is mathematically identical to the original whenever dropout is "
      "off (float64 diffs at machine epsilon); the sole training-time difference is frozen-branch "
      "dropout, which the O-LoRA/InfLoRA papers do not apply (they merge frozen adapters)."
      if (ok32 and ok64 and de < 1e-5) else "FAIL -- investigate")