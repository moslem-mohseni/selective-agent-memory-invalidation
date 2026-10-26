"""Tables and figures for Paper 1."""
from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from stats_utils import holm, mean_ci, paired_diff_ci, sign_flip_p

RES = os.path.join(os.path.dirname(__file__), "..", "results")
FIG = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
ORDER = ["keep_all", "clear_all", "ttl", "revalidate_all", "selective_tool", "selective_clause",
         "selective_clause_audit0.1", "selective_clause_tiered0.5"]
LABEL = {"keep_all": "Keep all", "clear_all": "Clear all", "ttl": "TTL (75)", "revalidate_all": "Revalidate all",
         "selective_tool": "Selective (tool-level)", "selective_clause": "Selective (clause-level)",
         "selective_clause_audit0.1": "Clause + 10% audit", "selective_clause_tiered0.5": "Clause + tiered probe"}


def main():
    d = pd.read_csv(os.path.join(RES, "e1_main.csv"))
    rows = []
    for p in ORDER:
        g = d[d.policy == p]
        s = mean_ci(g.success.values)
        c = mean_ci(g.cost_per_task.values)
        rows.append(dict(policy=p, success=s[0], s_lo=s[1], s_hi=s[2], cost=c[0], c_lo=c[1], c_hi=c[2],
                         probe_units=g.probe_cost.mean(), silent_stale=g.stale_silent.mean(), visible_stale=g.stale_visible.mean(),
                         missed_after_maintenance=g.missed_stale.mean(), healthy_retained=g.healthy_retention.mean(),
                         cost_per_success=(g.cost_per_task / g.success).mean()))
    t1 = pd.DataFrame(rows)
    t1.to_csv(os.path.join(RES, "table1_main.csv"), index=False)
    print(t1.round(3).to_string())

    pairs = [("selective_tool", "revalidate_all"), ("selective_clause", "revalidate_all"),
             ("selective_clause_tiered0.5", "revalidate_all"), ("selective_clause", "selective_tool"),
             ("selective_tool", "clear_all"), ("selective_clause_tiered0.5", "clear_all"),
             ("selective_clause", "keep_all"), ("selective_clause_audit0.1", "selective_clause")]
    pr = []
    for a, b in pairs:
        for met in ("success", "cost_per_task", "probe_cost"):
            x = d[d.policy == a].sort_values("seed")[met].values
            y = d[d.policy == b].sort_values("seed")[met].values
            m, lo, hi, dz = paired_diff_ci(x, y)
            pr.append(dict(a=a, b=b, metric=met, diff=m, lo=lo, hi=hi, dz=dz, p=sign_flip_p(x, y, n_perm=3000)))
    pdf = pd.DataFrame(pr)
    pdf["p_holm"] = holm(pdf.p.values)
    pdf.to_csv(os.path.join(RES, "table2_paired.csv"), index=False)
    print(pdf.round(4).to_string())

    # harm-weighted cost
    hw = []
    for w in (0, 2, 5, 10):
        d["eff"] = d.cost_per_task + w * d.stale_silent / d.n_post
        hw.append(dict(w=w, **d.groupby("policy").eff.mean().to_dict()))
    pd.DataFrame(hw).to_csv(os.path.join(RES, "table3_harm_weighted_cost.csv"), index=False)

    # Fig 1: Pareto
    fig, ax = plt.subplots(figsize=(6.0, 3.4), dpi=350)
    cols = {"keep_all": "#999999", "clear_all": "#d62728", "ttl": "#8c564b", "revalidate_all": "#ff7f0e",
            "selective_tool": "#2ca02c", "selective_clause": "#1f77b4", "selective_clause_audit0.1": "#17becf",
            "selective_clause_tiered0.5": "#9467bd"}
    for _, r in t1.iterrows():
        ax.errorbar(r.cost, r.success, xerr=[[r.cost - r.c_lo], [r.c_hi - r.cost]], yerr=[[r.success - r.s_lo], [r.s_hi - r.success]],
                    fmt="o", color=cols[r.policy], ms=5, capsize=2, label=LABEL[r.policy])
    ax.set_xlabel("Mean cost per post-change task (LLM-call units)")
    ax.set_ylabel("Post-change task success")
    ax.legend(frameon=False, fontsize=6, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig1_pareto.png"))
    plt.close(fig)

    # Fig 2: recall sweep
    r2 = pd.read_csv(os.path.join(RES, "e2_recall.csv"))
    r2["recall"] = r2.cond.str.split("=").str[1].astype(float)
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.0), dpi=300)
    for p in ["revalidate_all", "clear_all", "selective_tool", "selective_clause", "selective_clause_tiered0.5"]:
        g = r2[r2.policy == p].groupby("recall")
        ax[0].plot(g.success.mean().index, g.success.mean().values, "-o", ms=3, color=cols[p], label=LABEL[p])
        ax[1].plot(g.cost_per_task.mean().index, g.cost_per_task.mean().values, "-o", ms=3, color=cols[p])
    ax[0].set_xlabel("Dependency-recording recall")
    ax[0].set_ylabel("Post-change task success")
    ax[1].set_xlabel("Dependency-recording recall")
    ax[1].set_ylabel("Mean cost per task")
    ax[0].legend(frameon=False, fontsize=6)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig2_recall.png"))
    plt.close(fig)

    # Fig 3: by change type
    r4 = pd.read_csv(os.path.join(RES, "e4_type.csv"))
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.0), dpi=300)
    types = ["benign", "rename", "precond", "unit"]
    pols = ["keep_all", "clear_all", "revalidate_all", "selective_tool", "selective_clause"]
    w = 0.16
    for i, p in enumerate(pols):
        g = r4[r4.policy == p].groupby("cond")
        ax[0].bar(np.arange(4) + i * w, [g.success.mean()[t] for t in types], w, color=cols[p], label=LABEL[p])
        ax[1].bar(np.arange(4) + i * w, [g.cost_per_task.mean()[t] for t in types], w, color=cols[p])
    for a in ax:
        a.set_xticks(np.arange(4) + 2 * w)
        a.set_xticklabels(["benign\n(optional field)", "renamed\nargument", "requested\nprecondition", "silent unit\nchange"], fontsize=7)
    ax[0].set_ylabel("Post-change task success")
    ax[1].set_ylabel("Mean cost per task")
    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=7, ncol=3,
               loc="upper center", bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.83))
    fig.savefig(os.path.join(FIG, "fig3_change_type.png"))
    plt.close(fig)

    # sensitivity table
    e5 = pd.read_csv(os.path.join(RES, "e5_sensitivity.csv"))
    s5 = e5.groupby(["cond", "policy"]).agg(success=("success", "mean"), cost=("cost_per_task", "mean")).unstack()
    s5.to_csv(os.path.join(RES, "table4_sensitivity.csv"))
    e6 = pd.read_csv(os.path.join(RES, "e6_audit.csv"))
    s6 = e6.groupby(["cond", "policy"]).agg(success=("success", "mean"), cost=("cost_per_task", "mean"), missed=("missed_stale", "mean")).reset_index()
    s6.to_csv(os.path.join(RES, "table5_audit.csv"), index=False)
    e3 = pd.read_csv(os.path.join(RES, "e3_magnitude.csv"))
    e3.groupby(["cond", "policy"]).agg(success=("success", "mean"), cost=("cost_per_task", "mean")).unstack().to_csv(os.path.join(RES, "table6_magnitude.csv"))
    print("ok")


if __name__ == "__main__":
    main()
