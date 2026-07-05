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
import time as _time
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
                 energy_target: float = None, diag: bool = False,
                 warmup_fixed_rank: int = None, warmup_tasks: int = 0,
                 merge_op: str = "randsvd", cs_rank: int = None,
                 cs_seed: int = 0, module_idx: int = 0):
        super().__init__()
        # merge_op: how the boundary folds (sketch + residual) back into the sketch slot.
        #   "randsvd"     -> spectral truncation (original SVDLoRA; optimal rank-k, Eckart-Young)
        #   "countsketch" -> CountSketch of the CONCATENATED FACTORS along the rank dimension:
        #                    dW = [B_hat | sqrt(s)B][A_hat ; sqrt(s)A] is hashed from inner dim
        #                    m = r_hat + p*r down to k buckets with random signs. E[B'A'] = dW
        #                    (unbiased), error = random collision cross-terms ~ O(1/k) of TOTAL
        #                    component energy -- the "naive merge" ablation vs the SVD's
        #                    principled tail-truncation. Fixed-rank only (no spectrum exists).
        self.merge_op = merge_op
        if merge_op == "countsketch":
            assert energy_target is None and warmup_fixed_rank is None, \
                "countsketch merge is fixed-rank only (no spectrum to threshold on)"
        self.cs_rank = cs_rank if cs_rank is not None else r_hat
        self.cs_seed = cs_seed
        self.module_idx = module_idx
        self.last_merge_relerr = None     # ||dW - kept||_F / ||dW||_F at last compress (CS path)
        # ---- CS probe (diagnostic runs only; off for real ablations) ----
        # cs_probe=True additionally tracks, per boundary:
        #   last_cs_pred_relerr : closed-form E||err||_F prediction from the component Grams
        #                         E||E||^2 = (1/k) sum_{i!=j} (g_i g_j + Gb_ij Ga_ij)
        #   last_truth_relerr   : ||kept - TRUE accumulated delta||/||TRUE|| where the true
        #                         delta (exact sum of every task's s*B@A) accumulates in a
        #                         CPU float32 buffer -> exposes compounding across boundaries.
        # last_cs_ms (always on for CS) times the CORE sketch op (concat+rebalance+hash+scatter),
        # excluding diagnostics, so it reflects real-ablation cost.
        self.cs_probe = False
        self.last_cs_ms = None
        self.last_cs_pred_relerr = None
        self.last_truth_relerr = None
        self._truth_delta = None          # CPU [out, in] float32, allocated lazily when probing
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
        # warm-up: for the first `warmup_tasks` compressions, sketch with a FIXED target
        # rank `warmup_fixed_rank` (lossless while the accumulated raw rank < that target),
        # then switch to the adaptive energy_target path. None => warm-up disabled.
        self.warmup_fixed_rank = warmup_fixed_rank
        self.warmup_tasks = warmup_tasks
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
    def compress(self, task_idx=None):
        """Sketch boundary: fold the prior sketch + this period's INDEPENDENT per-task
        adapters (stashed + current residual, summed) into a new sketch via rand_svd, then
        clear the period stash and reset the residual.

        `task_idx` (0-based index of the task just finished) selects warm-up vs adaptive:
        while it is < warmup_tasks and a warmup_fixed_rank is set, compress to that fixed
        rank (lossless padding while the raw rank is below it); afterwards use energy_target."""
        dtype = self.sketch_B.dtype
        if self.merge_op == "countsketch":
            self._compress_countsketch(dtype, task_idx=task_idx)
            self.last_r_hat = self.r_hat
            self._stash_A, self._stash_B = [], []
            self.reset_residual()
            return
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

        in_warmup = (self.warmup_fixed_rank is not None and task_idx is not None
                     and task_idx < self.warmup_tasks)
        if in_warmup:
            self._compress_fixed(dW, dtype, target_rank=self.warmup_fixed_rank)
        elif self.energy_target is None:
            self._compress_fixed(dW, dtype)
        else:
            self._compress_adaptive(dW, dtype)
        self.last_r_hat = self.r_hat
        self._stash_A, self._stash_B = [], []
        self.reset_residual()

    def _compress_fixed(self, dW, dtype, target_rank=None):
        """Fixed-rank compression. target_rank=None uses self.r_hat (original behaviour);
        the warm-up path passes an explicit (larger) rank. The target is capped at the
        matrix's own dimensions -- beyond the true rank the extra slots are zero-energy."""
        rk = self.r_hat if target_rank is None else min(target_rank, min(dW.shape))
        if self.diag:
            # --- diagnostics on the pre-truncation accumulated ΔW (EXTRA full SVD) ---
            S = torch.linalg.svdvals(dW)                   # singular values, descending
            e = S.pow(2); tot = e.sum()
            self.last_retained = (e[:rk].sum() / tot).item() if tot > 0 else 1.0
            self.last_sigma_next = S[rk].item() if S.numel() > rk else 0.0
            self.last_fro = tot.sqrt().item()
        B_hat, A_hat = rand_svd(dW, rk, self.oversampling)
        self.sketch_B = B_hat.to(dtype).contiguous()
        self.sketch_A = A_hat.to(dtype).contiguous()
        self.r_hat = rk

    def _compress_countsketch(self, dtype, task_idx=None):
        """CountSketch merge: hash the rank dimension of the concatenated factors down to
        k = cs_rank buckets.

        The accumulated delta has the exact factored form
            dW = [B_hat | sqrt(s)B_1 | ... | sqrt(s)B_p] @ [A_hat ; sqrt(s)A_1 ; ... ; sqrt(s)A_p]
                 = F_B @ F_A,   inner dim m = r_hat + p*r  (s = alpha/r folded in symmetrically).
        Each rank-1 column pair (F_B[:,i], F_A[i,:]) is REBALANCED to equal norms (canonical
        form -- product unchanged; makes the hash-collision error independent of how LoRA
        happened to split magnitude between A and B, and treats sketch/residual components
        symmetrically). Then a fresh CountSketch S (hash h:[m]->[k], signs +-1; one nonzero
        per column) is applied to the INNER dimension:
            B' = F_B S^T   [out, k]      (index_add over columns: bucket h[i] += sgn[i]*col_i)
            A' = S  F_A    [k, in]       (index_add over rows)
        E[S^T S] = I  =>  E[B'A'] = dW (unbiased). The error is exactly the colliding
        cross-terms sum_{i!=j, h_i=h_j} sgn_i sgn_j F_B[:,i] F_A[j,:]  -- a random signed
        merge of components, NOT a spectral truncation. Hash/signs are re-drawn each
        boundary, seeded by (cs_seed, task_idx, module_idx) for reproducibility.
        Deployed factors land in the same sketch_B/sketch_A slots (LoRA form preserved)."""
        # probe: accumulate the TRUE delta added this boundary (exact, pre-sketch)
        if self.cs_probe:
            new_true = self.scale * (self.lora_B.float() @ self.lora_A.float())
            for A_j, B_j in zip(self._stash_A, self._stash_B):
                new_true = new_true + self.scale * (B_j.float() @ A_j.float())
            if self._truth_delta is None:
                self._truth_delta = torch.zeros(self.out_features, self.in_features,
                                                dtype=torch.float32, device="cpu")
            self._truth_delta += new_true.cpu()

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        _t0 = _time.perf_counter()
        rs = math.sqrt(self.scale)
        comps_B = [self.sketch_B.float(), rs * self.lora_B.float()]
        comps_A = [self.sketch_A.float(), rs * self.lora_A.float()]
        for A_j, B_j in zip(self._stash_A, self._stash_B):
            comps_B.append(rs * B_j.float())
            comps_A.append(rs * A_j.float())
        F_B = torch.cat(comps_B, dim=1)          # [out, m]
        F_A = torch.cat(comps_A, dim=0)          # [m, in]
        # drop zero components (initial all-zero sketch columns, untrained residuals)
        nb, na = F_B.norm(dim=0), F_A.norm(dim=1)
        keep = (nb * na) > 0
        k = min(self.cs_rank, self.out_features, self.in_features)
        if not bool(keep.any()):                 # nothing accumulated yet -> zero sketch
            self.sketch_B = torch.zeros(self.out_features, k, dtype=dtype,
                                        device=self.sketch_B.device)
            self.sketch_A = torch.zeros(k, self.in_features, dtype=dtype,
                                        device=self.sketch_A.device)
            self.r_hat = k
            self.last_merge_relerr = 0.0
            self.last_retained = 1.0
            self.last_fro = 0.0
            self.last_cs_ms = (_time.perf_counter() - _t0) * 1000.0
            return
        F_B, F_A = F_B[:, keep], F_A[keep, :]
        nb, na = nb[keep], na[keep]
        # rebalance: ||F_B[:,i]|| == ||F_A[i,:]|| == sqrt(nb_i * na_i); product exact
        t = torch.sqrt(na / nb)
        F_B = F_B * t.unsqueeze(0)
        F_A = F_A / t.unsqueeze(1)
        m = F_B.shape[1]
        # fresh CountSketch per boundary, deterministic per (run seed, task, module)
        g = torch.Generator(device="cpu")
        g.manual_seed((int(self.cs_seed) * 1000003
                       + (0 if task_idx is None else int(task_idx) + 1) * 9176
                       + int(self.module_idx)) % (2 ** 63 - 1))
        h = torch.randint(0, k, (m,), generator=g).to(F_B.device)
        sgn = (torch.randint(0, 2, (m,), generator=g) * 2 - 1).to(F_B.dtype).to(F_B.device)
        B_new = torch.zeros(self.out_features, k, dtype=F_B.dtype, device=F_B.device)
        A_new = torch.zeros(k, self.in_features, dtype=F_A.dtype, device=F_A.device)
        B_new.index_add_(1, h, F_B * sgn.unsqueeze(0))
        A_new.index_add_(0, h, F_A * sgn.unsqueeze(1))
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.last_cs_ms = (_time.perf_counter() - _t0) * 1000.0
        # probe: closed-form expected error from the component Grams (m x m, tiny)
        if self.cs_probe:
            Gb = F_B.t() @ F_B                   # [m, m]; diag = g_i (post-rebalance)
            Ga = F_A @ F_A.t()
            g = torch.diagonal(Gb)
            off = ~torch.eye(m, dtype=torch.bool, device=F_B.device)
            pred_sq = ((g.outer(g) + Gb * Ga)[off]).sum() / k
            fro_dw = (F_B @ F_A).norm()
            self.last_cs_pred_relerr = (pred_sq.sqrt() / fro_dw).item() if fro_dw > 0 else 0.0
        # merge-error diagnostic (one dense pass; dW is exact from the factors).
        # last_retained stores the Frobenius energy-captured ANALOG 1 - relerr^2 (== the SVD
        # path's retained-energy when the kept matrix is an orthogonal projection; for CS it
        # can go negative under bad collisions -- that is signal, not a bug).
        dW = F_B @ F_A
        fro = dW.norm()
        relerr = ((dW - B_new @ A_new).norm() / fro).item() if fro > 0 else 0.0
        self.last_merge_relerr = relerr
        self.last_retained = 1.0 - relerr ** 2
        self.last_fro = fro.item()
        # probe: drift vs the TRUE accumulated delta (compounding across boundaries)
        if self.cs_probe and self._truth_delta is not None:
            kept_cpu = (B_new @ A_new).float().cpu()
            tn = self._truth_delta.norm()
            self.last_truth_relerr = ((kept_cpu - self._truth_delta).norm() / tn).item() \
                if tn > 0 else 0.0
        self.sketch_B = B_new.to(dtype).contiguous()
        self.sketch_A = A_new.to(dtype).contiguous()
        self.r_hat = k

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
                   energy_target=None, diag=False, warmup_fixed_rank=None, warmup_tasks=0,
                   merge_op="randsvd", cs_rank=None, cs_seed=0):
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
    for idx, name in enumerate(replace):
        parent = model.get_submodule(name.rsplit(".", 1)[0])
        child_name = name.rsplit(".", 1)[1]
        base_linear = getattr(parent, child_name)
        wrapper = SVDLoRALinear(base_linear, r=r, r_hat=r_hat, alpha=alpha,
                                oversampling=oversampling, dropout=dropout,
                                energy_target=energy_target, diag=diag,
                                warmup_fixed_rank=warmup_fixed_rank, warmup_tasks=warmup_tasks,
                                merge_op=merge_op, cs_rank=cs_rank, cs_seed=cs_seed,
                                module_idx=idx)
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


def compress_all(modules, task_idx=None):
    for m in modules:
        m.compress(task_idx=task_idx)
    # aggregate the per-module compression diagnostics
    ret = [m.last_retained for m in modules if m.last_retained is not None]
    sig = [m.last_sigma_next for m in modules if m.last_sigma_next is not None]
    fro = [m.last_fro for m in modules if m.last_fro is not None]
    rfro = [m.last_residual_fro for m in modules if m.last_residual_fro is not None]
    rhat = [m.last_r_hat for m in modules if m.last_r_hat is not None]
    cserr = [m.last_merge_relerr for m in modules if m.last_merge_relerr is not None]
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
        # countsketch merge path only: ||dW - kept||_F / ||dW||_F per module
        "merge_relerr_mean": _m(cserr, _np.mean), "merge_relerr_max": _m(cserr, _np.max),
        "merge_relerr": cserr,
    }
    return out
