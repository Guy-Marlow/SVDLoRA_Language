"""
Multi-slot per-task LoRA for the O-LoRA and InfLoRA baselines on TokMem atomic recall.

Shared structure for both methods (mirrors the vision ports in
svd_sketching_vision/LAMDA-PILOT/models/{olora,inflora}.py):
  - one LoRA adapter per task on each q_proj/v_proj (a growing ParameterList).
  - MERGED forward: base(x) + (alpha/r) * sum_{k<=t} B_k A_k x.  This is inherently
    a single deployed composite with NO routing / NO task id -> matches the tokmem
    LoRA-baseline eval (greedy generation over the pooled test set) exactly.

O-LoRA: trains slot t's (A_t, B_t), freezes prior slots, adds an orthogonality
        penalty between A_t and the frozen A_{<t}.
InfLoRA: collects the input second-moment E[x x^T] per module, sets A_t analytically
        from the SVD of that covariance projected to REMOVE the past subspace
        (DualGPM), freezes A_t, trains only B_t, then grows the DualGPM memory.

Covariance is collected over the residual-stream input to q_proj/v_proj (dim = hidden,
3072 for Llama-3.2-3B). q and v share that input, so their per-module covariances are
identical (harmless redundancy). NOTE: collection accumulates over all token positions,
including left-padding (the module only sees x, not the attention mask) -- an
approximation; content tokens dominate the top singular directions.
"""
import math
import numpy as np
import torch
import torch.nn as nn


class MultiSlotLoRALinear(nn.Module):
    def __init__(self, base_linear: nn.Linear, r: int, alpha: float,
                 dropout: float = 0.0, collect_cov: bool = False):
        super().__init__()
        self.base = base_linear
        for p in self.base.parameters():
            p.requires_grad = False
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.r = r
        self.scale = alpha / r
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.lora_A = nn.ParameterList()   # each [r, in]
        self.lora_B = nn.ParameterList()   # each [out, r]
        self.collect_cov = collect_cov
        self.collect = False
        if collect_cov:
            self.register_buffer("cur_matrix",
                                 torch.zeros(self.in_features, self.in_features, dtype=torch.float32,
                                             device=base_linear.weight.device))

    @property
    def n_tasks(self):
        return len(self.lora_A)

    def add_task(self):
        dtype, device = self.base.weight.dtype, self.base.weight.device
        A = nn.Parameter(torch.empty(self.r, self.in_features, dtype=dtype, device=device))
        B = nn.Parameter(torch.zeros(self.out_features, self.r, dtype=dtype, device=device))
        nn.init.kaiming_uniform_(A, a=math.sqrt(5))
        self.lora_A.append(A)
        self.lora_B.append(B)

    def forward(self, x):
        out = self.base(x)
        if self.collect and self.collect_cov:
            xf = x.reshape(-1, self.in_features).float()
            self.cur_matrix += xf.t() @ xf
        for A, B in zip(self.lora_A, self.lora_B):
            out = out + self.scale * ((self.drop(x) @ A.t()) @ B.t())
        return out

    def set_trainable(self, task_idx, train_a=True):
        """Freeze every slot, then enable slot task_idx (A only if train_a)."""
        for i in range(self.n_tasks):
            self.lora_A[i].requires_grad_(i == task_idx and train_a)
            self.lora_B[i].requires_grad_(i == task_idx)

    def set_collect(self, flag):
        self.collect = flag

    def reset_cur_matrix(self):
        if self.collect_cov:
            self.cur_matrix.zero_()


def inject_multislot_lora(model, target_modules=("q_proj", "v_proj"), r=8,
                          alpha=32.0, dropout=0.0, collect_cov=False):
    for p in model.parameters():
        p.requires_grad = False
    targets = set(target_modules)
    names = [n for n, m in model.named_modules()
             if isinstance(m, nn.Linear) and n.split(".")[-1] in targets]
    inserted = []
    for name in names:
        parent = model.get_submodule(name.rsplit(".", 1)[0])
        child = name.rsplit(".", 1)[1]
        wrapper = MultiSlotLoRALinear(getattr(parent, child), r=r, alpha=alpha,
                                      dropout=dropout, collect_cov=collect_cov)
        setattr(parent, child, wrapper)
        inserted.append(wrapper)
    return inserted


def trainable_parameters(modules):
    return [p for m in modules for p in m.parameters() if p.requires_grad]


# ----------------------- O-LoRA penalty -----------------------
def orthogonality_penalty(modules, cur_task):
    """sum_modules sum_{s<t} | A_t @ A_s^T |.sum()  (current vs frozen previous)."""
    orth = 0.0
    for m in modules:
        A_t = m.lora_A[cur_task]
        for s in range(cur_task):
            orth = orth + torch.abs(A_t @ m.lora_A[s].t()).sum()
    return orth


# ----------------------- InfLoRA / DualGPM -----------------------
# Per-LAYER (q & v share the residual-stream input -> identical covariance), and all
# decompositions on GPU in float32. The covariance is symmetric PSD, so we use eigh
# (cheap) for it; the projected matrix act_hat is not symmetric, so it uses svd.
# This is the same DualGPM logic as the verbatim ViT port, just deduplicated to 28
# layers and moved off CPU-float64 numpy (which was infeasible at d=3072 x 56 modules).
def _sym_eig_desc(A):
    """Eigendecomposition of symmetric PSD A [d,d]; eigenvectors/values in DESC order."""
    evals, evecs = torch.linalg.eigh(A)          # ascending
    return evecs.flip(1), evals.flip(0).clamp(min=0)


@torch.no_grad()
def collect_covariance(model, q_modules, loader, device):
    """Accumulate input covariance on the q-side module of each layer (q & v share input)."""
    for m in q_modules:
        m.reset_cur_matrix(); m.set_collect(True)
    model.eval()
    for batch in loader:
        model(input_ids=batch["input_ids"].to(device),
              attention_mask=batch["attention_mask"].to(device))
    for m in q_modules:
        m.set_collect(False)


@torch.no_grad()
def init_lora_A_dualgpm(pairs, feature_mat, project_type, cur_task, rank, device):
    """Set A_t from the (DualGPM-projected) input covariance; write to BOTH q and v; freeze."""
    for kk, (q, v) in enumerate(pairs):
        cur = q.cur_matrix.float()                       # [in, in], symmetric PSD
        if cur_task == 0:
            U, _ = _sym_eig_desc(cur)
            basis = U[:, :rank]
        else:
            fmat = feature_mat[kk].to(cur.device)
            cur = (cur - fmat @ cur) if project_type[kk] == "remove" else (fmat @ cur)
            U, _, _ = torch.linalg.svd(cur, full_matrices=False)   # projected -> not symmetric
            basis = U[:, :rank]
        A = (basis.t() / math.sqrt(3)).to(q.lora_A[cur_task].dtype)  # [rank, in]
        q.lora_A[cur_task].data.copy_(A)
        v.lora_A[cur_task].data.copy_(A)
        q.reset_cur_matrix()


@torch.no_grad()
def update_dualgpm(q_modules, feature_list, project_type, cur_task, total_tasks, lamb, lame):
    """Grow the per-layer DualGPM feature memory (GPU float32); returns recomputed feature_mat.
    Same logic as InfLoRA/methods/inflora.py::update_DualGPM, deduplicated to one entry/layer."""
    threshold = (lame - lamb) * cur_task / total_tasks + lamb
    mats = [m.cur_matrix.float().clone() for m in q_modules]
    for m in q_modules:
        m.reset_cur_matrix()

    if len(feature_list) == 0:
        for activation in mats:
            U, S = _sym_eig_desc(activation)
            sval_ratio = (S ** 2) / (S ** 2).sum()
            r = int((torch.cumsum(sval_ratio, 0) < threshold).sum().item())
            feature_list.append(U[:, :max(r, 1)].clone())
            project_type.append('remove' if r < (activation.shape[0] / 2) else 'retain')
    else:
        for i, activation in enumerate(mats):
            F = feature_list[i]
            if project_type[i] == 'remove':
                _, S1 = _sym_eig_desc(activation)
                sval_total = (S1 ** 2).sum()
                act_hat = activation - F @ (F.t() @ activation)
                U, S, _ = torch.linalg.svd(act_hat, full_matrices=False)
                sval_ratio = (S ** 2) / sval_total
                accumulated = (sval_total - (S ** 2).sum()) / sval_total
                r = 0
                for ii in range(sval_ratio.shape[0]):
                    if accumulated < threshold:
                        accumulated += sval_ratio[ii]; r += 1
                    else:
                        break
                if r == 0:
                    continue
                Ui = torch.cat([F, U[:, :r]], dim=1)
                feature_list[i] = Ui[:, :Ui.shape[0]] if Ui.shape[1] > Ui.shape[0] else Ui
            else:
                _, S1 = _sym_eig_desc(activation)
                sval_total = (S1 ** 2).sum()
                act_hat = F @ (F.t() @ activation)
                U, S, _ = torch.linalg.svd(act_hat, full_matrices=False)
                sval_ratio = (S ** 2) / sval_total
                accumulated = (S ** 2).sum() / sval_total
                r = 0
                for ii in range(sval_ratio.shape[0]):
                    if accumulated >= (1 - threshold):
                        accumulated -= sval_ratio[ii]; r += 1
                    else:
                        break
                if r == 0:
                    continue
                act_feature = F - U[:, :r] @ (U[:, :r].t() @ F)
                Ui, _, _ = torch.linalg.svd(act_feature, full_matrices=False)
                feature_list[i] = Ui[:, :F.shape[1] - r]

    for i in range(len(feature_list)):
        F = feature_list[i]
        if project_type[i] == 'remove' and (F.shape[1] > (F.shape[0] / 2)):
            U, _, _ = torch.linalg.svd(F, full_matrices=True)
            feature_list[i] = U[:, F.shape[1]:]
            project_type[i] = 'retain'

    # null-space fill diagnostic: fraction of each layer's d input dims consumed by
    # the accumulated subspace (-> 1.0 = orthogonal room exhausted, interference returns).
    # 'remove' stores the spanned subspace; 'retain' stores its complement.
    consumed = [(F.shape[1] if project_type[i] == 'remove' else F.shape[0] - F.shape[1])
                for i, F in enumerate(feature_list)]
    d = feature_list[0].shape[0]
    frac = [c / d for c in consumed]
    import numpy as _np
    print(f"  [DualGPM fill] task {cur_task}: consumed/d mean={_np.mean(frac):.4f} "
          f"max={_np.max(frac):.4f} min={_np.min(frac):.4f} (d={d}, threshold={threshold:.4f})",
          flush=True)

    return [F @ F.t() for F in feature_list]
