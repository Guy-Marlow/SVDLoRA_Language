"""Smoke tests for the CountSketch merge-op ablation (svdlora_layer._compress_countsketch).

Validates, on a small SVDLoRALinear (no model download):
  1. shapes / dtypes / trainable flags after a countsketch compress; randsvd path untouched.
  2. determinism: same (cs_seed, task_idx, module_idx) -> bit-identical sketches.
  3. unbiasedness: E[B'A'] over many hash draws -> the true accumulated dW
     (relative error of the Monte-Carlo mean shrinks ~1/sqrt(N)).
  4. error decreases with bucket count k (~1/sqrt(k) trend on a fixed dW).
  5. multi-boundary accumulation: after 2 boundaries the sketch approximates the SUM of
     both tasks' deltas (linearity of the merge), not just the last one.

Run: python cs_smoke_test.py
"""
import copy
import math
import torch
import torch.nn as nn

from svdlora_layer import SVDLoRALinear

torch.manual_seed(0)
OUT, IN, R = 32, 48, 4


def make_module(merge_op="countsketch", cs_rank=8, cs_seed=7, module_idx=3, r_hat=8):
    base = nn.Linear(IN, OUT, bias=False)
    m = SVDLoRALinear(base, r=R, r_hat=r_hat, alpha=32.0, oversampling=10,
                      merge_op=merge_op, cs_rank=cs_rank, cs_seed=cs_seed,
                      module_idx=module_idx)
    return m


def set_residual(m, seed):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        m.lora_A.copy_(torch.randn(m.lora_A.shape, generator=g))
        m.lora_B.copy_(torch.randn(m.lora_B.shape, generator=g))


def true_delta(m):
    return m.scale * (m.lora_B.float() @ m.lora_A.float())


def sketch_delta(m):
    return m.sketch_B.float() @ m.sketch_A.float()


# ---------- 1. shapes / flags / randsvd untouched ----------
m = make_module()
set_residual(m, 1)
dW = true_delta(m).clone()
m.compress(task_idx=0)
assert m.sketch_B.shape == (OUT, 8) and m.sketch_A.shape == (8, IN), m.sketch_B.shape
assert m.r_hat == 8 and m.last_r_hat == 8
assert not m.sketch_B.requires_grad and not m.sketch_A.requires_grad
assert m.lora_A.requires_grad and m.lora_B.requires_grad
assert float(m.lora_B.abs().sum()) == 0.0            # residual reset (B=0)
assert m.last_merge_relerr is not None
# rank <= r_hat case must be EXACT for randsvd (regression: original path untouched)
m2 = make_module(merge_op="randsvd")
set_residual(m2, 1)
dW2 = true_delta(m2).clone()
m2.compress(task_idx=0)
exact_err = (sketch_delta(m2) - dW2).norm() / dW2.norm()
assert exact_err < 1e-5, f"randsvd rank<=r_hat should be exact, got {exact_err:.2e}"
print(f"[1] shapes/flags OK | randsvd exactness {exact_err:.2e} | cs relerr {m.last_merge_relerr:.3f}")

# ---------- 2. determinism ----------
a, b = make_module(), make_module()
set_residual(a, 2), set_residual(b, 2)
a.compress(task_idx=5), b.compress(task_idx=5)
assert torch.equal(a.sketch_B, b.sketch_B) and torch.equal(a.sketch_A, b.sketch_A)
c = make_module(module_idx=4)                        # different module index -> different hash
set_residual(c, 2)
c.compress(task_idx=5)
assert not torch.equal(a.sketch_B, c.sketch_B)
print("[2] determinism OK (same seeds identical; different module_idx differs)")

# ---------- 3. unbiasedness over hash draws ----------
N = 400
acc = torch.zeros(OUT, IN)
for t in range(N):
    mm = make_module(cs_seed=1234)
    set_residual(mm, 3)
    if t == 0:
        dW = true_delta(mm).clone()
    mm.compress(task_idx=t)                          # task_idx varies the hash draw
    acc += sketch_delta(mm)
mean_relerr = ((acc / N) - dW).norm() / dW.norm()
single = make_module(cs_seed=1234); set_residual(single, 3); single.compress(task_idx=0)
single_relerr = (sketch_delta(single) - dW).norm() / dW.norm()
assert mean_relerr < single_relerr / 3, (mean_relerr, single_relerr)
print(f"[3] unbiasedness OK: single-draw relerr {single_relerr:.3f} -> mean-of-{N} {mean_relerr:.3f}")

# ---------- 4. error vs k ----------
errs = {}
for k in (2, 8, 32):
    es = []
    for t in range(30):
        mm = make_module(cs_rank=k, cs_seed=99)
        set_residual(mm, 4)
        mm.compress(task_idx=t)
        es.append(mm.last_merge_relerr)
    errs[k] = sum(es) / len(es)
assert errs[2] > errs[8] > errs[32], errs
print(f"[4] error decreases with k OK: {errs}")

# ---------- 5. multi-boundary accumulation (linearity) ----------
mm = make_module(cs_rank=64)                          # large k -> low noise, isolate linearity
set_residual(mm, 10)
d1 = true_delta(mm).clone()
mm.compress(task_idx=0)
set_residual(mm, 11)
d2 = true_delta(mm).clone()
mm.compress(task_idx=1)
tot_relerr = (sketch_delta(mm) - (d1 + d2)).norm() / (d1 + d2).norm()
only_last = (sketch_delta(mm) - d2).norm() / d2.norm()
assert tot_relerr < only_last, (tot_relerr, only_last)
print(f"[5] accumulation OK: vs d1+d2 relerr {tot_relerr:.3f} < vs d2-only {only_last:.3f}")

print("ALL COUNTSKETCH SMOKE TESTS PASSED")