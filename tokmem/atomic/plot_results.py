"""Two bar charts for the 400-task language CL comparison:
  1) final ROUGE-L and exact-match per method (mean over finished seeds, sd error bars)
  2) deployed adapter memory per method (log scale), with a dotted reference bar for the
     O-LoRA / InfLoRA O(K) bank projected to num_tasks * per-task-adapter-size.
"""
import glob, json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

HERE = os.path.dirname(os.path.abspath(__file__))
MET = os.path.join(HERE, "run_logs", "a100_400t")
OUT = os.path.join(HERE, "run_logs")

def label_of(d):
    if d["method"] == "svdlora":
        return "SVDLoRA-adaptive" if d.get("energy_target") else "SVDLoRA-fixed"
    return {"seqlora": "SeqLoRA", "olora": "O-LoRA", "inflora": "InfLoRA"}.get(d["method"], d["method"])

rows, mem = [], {}
per_task_mb, num_tasks = None, 400
for f in glob.glob(os.path.join(MET, "metrics_*_400t_*.json")):
    d = json.load(open(f))
    if d.get("status") != "done":
        continue
    lab = label_of(d)
    rows.append({"method": lab, "metric": "ROUGE-L", "value": d["rougeL"]})
    rows.append({"method": lab, "metric": "Exact Match", "value": d["exact_match"]})
    a = d.get("adapter", {})
    mem[lab] = a.get("adapter_memory_mb")
    num_tasks = d.get("num_tasks", num_tasks)
    if d["method"] == "seqlora":              # one rank-r adapter = the per-task bank increment
        per_task_mb = a.get("adapter_memory_mb")

df = pd.DataFrame(rows)
order = [m for m in ["SeqLoRA", "SVDLoRA-fixed", "SVDLoRA-adaptive", "O-LoRA", "InfLoRA"] if m in set(df["method"])]
sns.set_theme(style="whitegrid", context="talk")

# ---- Chart 1: per-metric groups; 3 method bars each, distinct colors, descending by score ----
method_order = list(df[df.metric == "ROUGE-L"].groupby("method")["value"].mean()
                    .sort_values(ascending=False).index)        # adaptive > seqlora > fixed
fig, ax = plt.subplots(figsize=(9.5, 6))
sns.barplot(data=df, x="metric", y="value", hue="method", hue_order=method_order,
            order=["ROUGE-L", "Exact Match"], errorbar="sd", capsize=.1,
            err_kws={"linewidth": 1.5}, ax=ax, palette="Set2")
ax.set_xlabel(""); ax.set_ylabel("Score (%)")
ax.set_title(f"Final atomic-recall performance @ {num_tasks} tasks\n(mean over finished seeds, sd error bars)")
for c in ax.containers:
    ax.bar_label(c, fmt="%.1f", padding=2, fontsize=10)
ax.legend(title="", loc="upper right", frameon=True)
ax.set_ylim(0, max(df["value"]) * 1.18)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "chart_rouge_em.png"), dpi=150)

# ---- Chart 2: adapter memory (log) + O(K) bank reference ----
bank_mb = per_task_mb * num_tasks                      # task * adapter-size
mrows = [{"method": m, "mem_mb": mem[m]} for m in order if m in mem]
mdf = pd.DataFrame(mrows)
fig2, ax2 = plt.subplots(figsize=(9, 6))
sns.barplot(data=mdf, x="method", y="mem_mb", order=[r["method"] for r in mrows],
            color="#55A868", ax=ax2)
for c in ax2.containers:
    ax2.bar_label(c, fmt="%.1f MB", padding=3, fontsize=11)
# dotted reference bar for O-LoRA / InfLoRA O(K) bank
xs = ax2.get_xlim()
ax2.axhline(bank_mb, ls=":", lw=2.5, color="#C44E52")
ax2.text(0.02, bank_mb * 1.18, f"O-LoRA / InfLoRA bank  =  {num_tasks} tasks x {per_task_mb:.3f} MB  =  {bank_mb:.0f} MB",
         va="bottom", ha="left", color="#C44E52", fontsize=11, clip_on=False)
ax2.bar(len(mrows), bank_mb, color="#C44E52", alpha=0.25, hatch="//", edgecolor="#C44E52",
        linestyle=":", linewidth=2, label="O-LoRA/InfLoRA (projected)")
ax2.set_xticks(list(range(len(mrows) + 1)))
ax2.set_xticklabels([r["method"] for r in mrows] + ["O-LoRA/\nInfLoRA"], rotation=12, ha="right")
ax2.set_yscale("log")
ax2.set_ylim(top=bank_mb * 2.2)
ax2.set_xlabel(""); ax2.set_ylabel("Deployed adapter memory (MB, log scale)")
ax2.set_title(f"Deployed adapter memory @ {num_tasks} tasks\n(bounded O(1) sketch vs projected O(K) bank)")
fig2.tight_layout(); fig2.savefig(os.path.join(OUT, "chart_memory.png"), dpi=150)

print("methods with finished results:", order)
print("memory (MB):", {m: round(mem[m], 3) for m in mem})
print(f"bank reference: {num_tasks} x {per_task_mb:.3f} = {bank_mb:.1f} MB")
print("saved:", os.path.join(OUT, "chart_rouge_em.png"), "and", os.path.join(OUT, "chart_memory.png"))
