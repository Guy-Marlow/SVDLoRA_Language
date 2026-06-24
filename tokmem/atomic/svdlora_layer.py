"""
SVDLoRA layer for the TokMem atomic-recall LoRA comparison.

Design goal: a *drop-in replacement for the single PEFT LoRA adapter* used by
`main_lora_baseline.py`, so the comparison is apples-to-apples:
  - ONE deployed adapter, NO task routing, NO task identity at inference.
  - same q_proj / v_proj target modules, same rank/alpha scaling as the baseline.
  - the only structural addition over a single drifting adapter ("SeqLoRA") is a
    periodic SVD compression of the *accumulated* weight delta.

Two slots per wrapped Linear (bounded, O(1) memory regardless of #tasks):
  - SKETCH  : frozen rank-r_hat factors (B_hat, A_hat) that store the *effective*
              accumulated delta-W.  Scale is baked in -> forward scale = 1.
  - RESIDUAL: trainable rank-r LoRA factors (lora_A, lora_B) for the current task,
              applied with the baseline's scale = alpha / r (PEFT convention).

Compression (call once per task -> SVD period P = 1):
    dW_eff = sketch_B @ sketch_A  +  (alpha/r) * lora_B @ lora_A      # [out, in]
    B_hat, A_hat = rand_svd(dW_eff, r_hat, oversampling)              # rank-r_hat
    sketch <- (B_hat, A_hat);  residual <- reset (A kaiming, B zero)
So at eval the residual is a no-op and the model deploys the single sketch.

rand_svd is embedded here (a clean copy of the trusted
svd_sketching_vision/rand_svd_impl/randsvd.py torch path) so this language
harness has no cross-project import dependency.
"""

import math
import torch
import torch.nn as nn


@torch.no_grad()
def rand_svd(M: torch.Tensor, target_rank: int, oversampling: int,
             return_singular_values: bool = False):
    """Randomised SVD factorisation B_hat @ A_hat ~= M for M of shape [m, n].

    Returns (B_hat [m, target_rank], A_hat [target_rank, n]).  Computed in the
    dtype of M (cast to float32 by the caller for numerical stability).
    If ``return_singular_values`` also returns S (the singular values of M_bar,
    length target_rank+oversampling); when target_rank+oversampling >= rank(M)
    these are the exact singular values of M -- used by the adaptive-rank path
    to pick the target rank for free (no separate full SVD).
    """
    omega = torch.randn(M.shape[1], target_rank + oversampling, device=M.device, dtype=M.dtype)
    Y = M @ omega
    Q, _ = torch.linalg.qr(Y)
    M_bar = Q.t() @ M
    U_bar, S, Vh = torch.linalg.svd(M_bar, full_matrices=False)
    S_root = torch.diag(torch.sqrt(S))[:target_rank, :target_rank]
    U = (Q @ U_bar)[:, :target_rank]
    B_hat = U @ S_root
    A_hat = S_root @ Vh[:target_rank, :]
    if return_singular_values:
        return B_hat, A_hat, S
    return B_hat, A_hat


class SVDLoRALinear(nn.Module):
    """Wraps a frozen nn.Linear with a frozen SVD sketch + a trainable residual."""

    def __init__(self, base_linear: nn.Linear, r: int, r_hat: int,
                 alpha: float, oversampling: int, dropout: float = 0.0,
                 energy_target: float = None, diag: bool = False):
        super().__init__()
        # diag: log per-task retained-energy/sigma diagnostics. For the FIXED-rank path this
        # needs an extra full svdvals(dW) per module per task -> EXPENSIVE; default OFF for real
        # runs. (The adaptive path gets these for free from its rand_svd spectrum.)
        self.diag = diag
        self.base = base_linear
        for p in self.base.parameters():
            p.requires_grad = False

        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.r = r
        self.r_hat = r_hat
        # adaptive-rank mode: if set (e.g. 0.01), each compress keeps the SMALLEST
        # rank that retains (1 - energy_target) of the accumulated delta's energy,
        # growing the sketch only as much as the data's intrinsic rank demands.
        # None => the original fixed-rank-r_hat path (untouched).
        self.energy_target = energy_target
        self.scale = alpha / r                      # PEFT convention, matches baseline
        self.oversampling = oversampling
        self.last_retained = None         # retained-energy frac at last compress
        self.last_sigma_next = None       # σ_{r̂+1} (largest deleted value) at last compress
        self.last_fro = None              # ||ΔW||_F of accumulated effective delta
        self.last_residual_fro = None     # ||s·B_rA_r||_F of this task's residual
        self.last_r_hat = None            # sketch rank chosen at last compress (adaptive)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        dtype = base_linear.weight.dtype
        device = base_linear.weight.device

        # trainable residual (current task); standard LoRA init (B=0 -> no-op start)
        self.lora_A = nn.Parameter(torch.empty(r, self.in_features, dtype=dtype, device=device))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, r, dtype=dtype, device=device))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        # frozen sketch (accumulated effective delta); starts at zero (no-op)
        self.register_buffer("sketch_B", torch.zeros(self.out_features, r_hat, dtype=dtype, device=device))
        self.register_buffer("sketch_A", torch.zeros(r_hat, self.in_features, dtype=dtype, device=device))

        # Sketch-period support: with period P>1, the completed INDEPENDENT per-task adapters
        # of the current period are stashed here (NOT applied in forward) until the next
        # boundary folds them all into the sketch. Each was trained as a delta over W+sketch,
        # never over each other. _period_adapters = how many are folded at the next compress
        # (= len(stash)+1, the current residual), used as the adaptive rank budget.
        self._stash_A, self._stash_B = [], []
        self._period_adapters = 1

    def forward(self, x):
        out = self.base(x)
        # sketch term (scale = 1; effective delta is baked into the factors)
        out = out + (x @ self.sketch_A.t()) @ self.sketch_B.t()
        # residual term (scale = alpha / r, exactly like the PEFT baseline)
        out = out + self.scale * ((self.drop(x) @ self.lora_A.t()) @ self.lora_B.t())
        return out

    @torch.no_grad()
    def reset_residual(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.lora_B.zero_()

    @torch.no_grad()
    def stash_task(self):
        """End a NON-boundary task within a period: store this task's (independent) adapter
        for the next compression and reset the residual, so the next task trains a fresh
        delta over W+sketch without seeing this one."""
        self._stash_A.append(self.lora_A.detach().clone())
        self._stash_B.append(self.lora_B.detach().clone())
        self.reset_residual()

    @torch.no_grad()
    def compress(self):
        """Sketch boundary: fold the prior sketch + this period's INDEPENDENT per-task
        adapters (stashed + current residual, summed) into a new sketch via rand_svd, then
        clear the period stash and reset the residual."""
        dtype = self.sketch_B.dtype
        resid = self.scale * (self.lora_B.float() @ self.lora_A.float())
        if self.diag:
            # this task's residual contribution norm (diagnostic)
            self.last_residual_fro = resid.norm().item()
        # sum of the period's independent adapters: current residual + any stashed ones
        period_delta = resid
        for A_j, B_j in zip(self._stash_A, self._stash_B):
            period_delta = period_delta + self.scale * (B_j.float() @ A_j.float())
        self._period_adapters = len(self._stash_A) + 1   # rank budget for adaptive
        dW = self.sketch_B.float() @ self.sketch_A.float() + period_delta

        if self.energy_target is None:
            self._compress_fixed(dW, dtype)
        else:
            self._compress_adaptive(dW, dtype)
        self.last_r_hat = self.r_hat
        self._stash_A, self._stash_B = [], []
        self.reset_residual()

    def _compress_fixed(self, dW, dtype):
        """Original fixed-rank-r_hat compression (behaviour unchanged)."""
        if self.diag:
            # --- diagnostics on the pre-truncation accumulated ΔW (EXTRA full SVD) ---
            S = torch.linalg.svdvals(dW)                   # singular values, descending
            e = S.pow(2); tot = e.sum()
            self.last_retained = (e[:self.r_hat].sum() / tot).item() if tot > 0 else 1.0
            self.last_sigma_next = S[self.r_hat].item() if S.numel() > self.r_hat else 0.0
            self.last_fro = tot.sqrt().item()
        B_hat, A_hat = rand_svd(dW, self.r_hat, self.oversampling)
        self.sketch_B = B_hat.to(dtype)
        self.sketch_A = A_hat.to(dtype)

    def _compress_adaptive(self, dW, dtype):
        """Adaptive-rank compression: keep the smallest rank retaining
        (1 - energy_target) of the energy.  rank(dW) <= r_hat + (period adapters)*r, so one
        randomized SVD to that bound gives the EXACT spectrum -- the target rank
        is then a cumsum on those singular values (no separate full SVD)."""
        max_rank = min(self.sketch_B.shape[1] + self._period_adapters * self.r, min(dW.shape))
        B_full, A_full, S = rand_svd(dW, max_rank, self.oversampling,
                                     return_singular_values=True)
        e = S.pow(2); tot = e.sum()
        if tot > 0:
            cum = torch.cumsum(e, 0) / tot
            r_hat_t = int((cum < (1.0 - self.energy_target)).sum().item()) + 1
        else:
            r_hat_t = 1
        r_hat_t = max(1, min(r_hat_t, max_rank))
        # diagnostics from the SAME spectrum (no extra svdvals)
        self.last_retained = (e[:r_hat_t].sum() / tot).item() if tot > 0 else 1.0
        self.last_sigma_next = S[r_hat_t].item() if S.numel() > r_hat_t else 0.0
        self.last_fro = tot.sqrt().item()
        # truncate the (rank-ordered) factors to the chosen rank
        self.sketch_B = B_full[:, :r_hat_t].to(dtype).contiguous()
        self.sketch_A = A_full[:r_hat_t, :].to(dtype).contiguous()
        self.r_hat = r_hat_t


def _layer_index_of(name):
    """Parse the transformer block index from a module name like
    'model.layers.13.self_attn.q_proj' -> 13 (returns -1 if not found)."""
    parts = name.split(".")
    for i, p in enumerate(parts):
        if p == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
            return int(parts[i + 1])
    return -1


def inject_svdlora(model, target_modules=("q_proj", "v_proj"), r=8, r_hat=8,
                   alpha=32.0, oversampling=10, dropout=0.0, layer_indices=None,
                   energy_target=None, diag=False):
    """Replace each target nn.Linear in `model` with an SVDLoRALinear wrapper.

    Freezes every base parameter; only the residual lora_A/lora_B remain trainable.
    ``layer_indices`` (a set of block indices) restricts injection to those blocks;
    None = every block.  ``energy_target`` (None = fixed-rank) enables adaptive
    rank.  Returns the list of inserted SVDLoRALinear modules.
    """
    for p in model.parameters():
        p.requires_grad = False

    target_modules = set(target_modules)
    replace = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and name.split(".")[-1] in target_modules:
            if layer_indices is not None and _layer_index_of(name) not in layer_indices:
                continue
            replace.append(name)

    inserted = []
    for name in replace:
        parent = model.get_submodule(name.rsplit(".", 1)[0])
        child_name = name.rsplit(".", 1)[1]
        base_linear = getattr(parent, child_name)
        wrapper = SVDLoRALinear(base_linear, r=r, r_hat=r_hat, alpha=alpha,
                                oversampling=oversampling, dropout=dropout,
                                energy_target=energy_target, diag=diag)
        setattr(parent, child_name, wrapper)
        inserted.append(wrapper)
    return inserted


def svdlora_trainable_parameters(modules):
    params = []
    for m in modules:
        params += [m.lora_A, m.lora_B]
    return params


def stash_all(modules):
    """Non-boundary task within a period: stash each module's adapter, reset its residual."""
    for m in modules:
        m.stash_task()


def compress_all(modules):
    for m in modules:
        m.compress()
    # aggregate the per-module compression diagnostics
    ret = [m.last_retained for m in modules if m.last_retained is not None]
    sig = [m.last_sigma_next for m in modules if m.last_sigma_next is not None]
    fro = [m.last_fro for m in modules if m.last_fro is not None]
    rfro = [m.last_residual_fro for m in modules if m.last_residual_fro is not None]
    rhat = [m.last_r_hat for m in modules if m.last_r_hat is not None]
    if not ret:
        return None
    import numpy as _np
    # each diagnostic list may be empty depending on which fields were populated
    # (the fixed path only fills these when diag=True; the adaptive path fills
    # retained/sigma/fro for free but not residual_fro) -> guard every reduction.
    def _m(v, fn):
        return float(fn(v)) if len(v) else None
    out = {
        "retained_mean": _m(ret, _np.mean), "retained_min": _m(ret, _np.min),
        "sigma_next_mean": _m(sig, _np.mean), "sigma_next_max": _m(sig, _np.max),
        "fro_mean": _m(fro, _np.mean), "fro_max": _m(fro, _np.max),
        "residual_fro_mean": _m(rfro, _np.mean), "residual_fro_max": _m(rfro, _np.max),
        # adaptive-rank: the sketch rank chosen this task (mean/max/total across modules)
        "r_hat_mean": _m(rhat, _np.mean), "r_hat_max": _m(rhat, _np.max),
        "r_hat_total": int(_np.sum(rhat)) if rhat else None,
        "retained": ret, "sigma_next": sig, "fro": fro, "residual_fro": rfro,
        "r_hat": rhat,   # per-module rank (module order = layer0.q, layer0.v, layer1.q, ...)
    }
    return out
