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
def rand_svd(M: torch.Tensor, target_rank: int, oversampling: int):
    """Randomised SVD factorisation B_hat @ A_hat ~= M for M of shape [m, n].

    Returns (B_hat [m, target_rank], A_hat [target_rank, n]).  Computed in the
    dtype of M (cast to float32 by the caller for numerical stability).
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
    return B_hat, A_hat


class SVDLoRALinear(nn.Module):
    """Wraps a frozen nn.Linear with a frozen SVD sketch + a trainable residual."""

    def __init__(self, base_linear: nn.Linear, r: int, r_hat: int,
                 alpha: float, oversampling: int, dropout: float = 0.0):
        super().__init__()
        self.base = base_linear
        for p in self.base.parameters():
            p.requires_grad = False

        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.r = r
        self.r_hat = r_hat
        self.scale = alpha / r                      # PEFT convention, matches baseline
        self.oversampling = oversampling
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
    def compress(self):
        """Fold residual into the sketch via rand_svd, then reset the residual."""
        dtype = self.sketch_B.dtype
        dW = (self.sketch_B.float() @ self.sketch_A.float()
              + self.scale * (self.lora_B.float() @ self.lora_A.float()))
        B_hat, A_hat = rand_svd(dW, self.r_hat, self.oversampling)
        self.sketch_B = B_hat.to(dtype)
        self.sketch_A = A_hat.to(dtype)
        self.reset_residual()


def inject_svdlora(model, target_modules=("q_proj", "v_proj"), r=8, r_hat=8,
                   alpha=32.0, oversampling=10, dropout=0.0):
    """Replace each target nn.Linear in `model` with an SVDLoRALinear wrapper.

    Freezes every base parameter; only the residual lora_A/lora_B remain trainable.
    Returns the list of inserted SVDLoRALinear modules (for compression / param collection).
    """
    for p in model.parameters():
        p.requires_grad = False

    target_modules = set(target_modules)
    replace = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and name.split(".")[-1] in target_modules:
            replace.append(name)

    inserted = []
    for name in replace:
        parent = model.get_submodule(name.rsplit(".", 1)[0])
        child_name = name.rsplit(".", 1)[1]
        base_linear = getattr(parent, child_name)
        wrapper = SVDLoRALinear(base_linear, r=r, r_hat=r_hat, alpha=alpha,
                                oversampling=oversampling, dropout=dropout)
        setattr(parent, child_name, wrapper)
        inserted.append(wrapper)
    return inserted


def svdlora_trainable_parameters(modules):
    params = []
    for m in modules:
        params += [m.lora_A, m.lora_B]
    return params


def compress_all(modules):
    for m in modules:
        m.compress()
