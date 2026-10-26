"""Run all Paper 1 experiments and write tidy CSVs to ../results."""
from __future__ import annotations

import argparse
import dataclasses
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from sim import (KeepAll, Params, Selective, build_world, run, standard_policies)

RES = os.path.join(os.path.dirname(__file__), "..", "results")


def one_seed(args):
    exp, label, P, seed, audit = args
    world = build_world(P, seed)
    rows = []
    for pol in standard_policies(audit=audit):
        r = run(P, world, pol)
        r.update(exp=exp, cond=label, policy=pol.name, seed=seed)
        rows.append(r)
    return rows


def sweep(exp, conds, seeds, audit=0.1, workers=8, seed0=0):
    jobs = [(exp, label, P, s, audit) for label, P in conds for s in range(seed0, seed0 + seeds)]
    out = []
    for j in jobs:  # serial: the simulator is cheap and avoids multiprocessing pickling issues
        out.extend(one_seed(j))
    df = pd.DataFrame(out)
    # attach swept parameters
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=300)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    os.makedirs(RES, exist_ok=True)
    base = Params()
    S, W = a.seeds, a.workers

    # E1 main condition
    df = sweep("E1_main", [("main", base)], S, workers=W)
    df.to_csv(os.path.join(RES, "e1_main.csv"), index=False)

    # E2 dependency-recording recall
    conds = [(f"recall={r}", dataclasses.replace(base, recall=r)) for r in (1.0, 0.95, 0.9, 0.8, 0.7, 0.5)]
    sweep("E2_recall", conds, S, workers=W).to_csv(os.path.join(RES, "e2_recall.csv"), index=False)

    # E3 change magnitude
    conds = [(f"k={k}", dataclasses.replace(base, n_changed_tools=k)) for k in (1, 2, 3, 4, 6)]
    sweep("E3_magnitude", conds, S, workers=W).to_csv(os.path.join(RES, "e3_magnitude.csv"), index=False)

    # E4 change type (pure mixes)
    pure = {"rename": (1, 0, 0, 0), "precond": (0, 1, 0, 0), "unit": (0, 0, 1, 0), "benign": (0, 0, 0, 1)}
    conds = [(k, dataclasses.replace(base, mix=m)) for k, m in pure.items()]
    sweep("E4_type", conds, S, workers=W).to_csv(os.path.join(RES, "e4_type.csv"), index=False)

    # E5 cost/accuracy sensitivity: exploration success and probe sensitivity
    conds = [(f"p_explore={p}", dataclasses.replace(base, p_explore=p)) for p in (0.5, 0.6, 0.75, 0.9)]
    conds += [(f"q_det={q}", dataclasses.replace(base, q_det=q)) for q in (0.3, 0.5, 0.9, 1.0)]
    conds += [(f"p_feedback={f}", dataclasses.replace(base, p_feedback=f)) for f in (0.2, 0.5)]
    conds += [(f"c_probe={c}", dataclasses.replace(base, c_probe=c)) for c in (0.1, 1.0, 2.0)]
    sweep("E5_sensitivity", conds, S, workers=W).to_csv(os.path.join(RES, "e5_sensitivity.csv"), index=False)

    # E6 audit-rate ablation under imperfect recall (recall 0.8 and 0.6)
    rows = []
    for rec in (0.8, 0.6):
        for audit in (0.0, 0.05, 0.1, 0.2, 0.4, 0.7):
            P = dataclasses.replace(base, recall=rec)
            for s in range(S):
                world = build_world(P, s)
                r = run(P, world, Selective("clause", audit=audit))
                r.update(exp="E6_audit", cond=f"recall={rec}", policy=f"audit={audit}", seed=s)
                rows.append(r)
    pd.DataFrame(rows).to_csv(os.path.join(RES, "e6_audit.csv"), index=False)
    # E7 confirmation on fresh seeds (the tiered policy was added after inspecting a first run)
    conds = [("main", base), ("recall=0.7", dataclasses.replace(base, recall=0.7)), ("recall=0.5", dataclasses.replace(base, recall=0.5))]
    sweep("E7_confirm", conds, S, workers=W, seed0=10000).to_csv(os.path.join(RES, "e7_confirm.csv"), index=False)
    # Exploratory robustness additions, fixed before inspecting these results.
    conds = [("scale_100_30", dataclasses.replace(base, n_types=100, n_tools=30)),
             ("trace_full", dataclasses.replace(base, recorder="trace_full")),
             ("trace_arguments", dataclasses.replace(base, recorder="trace_arguments"))]
    sweep("E8_scale_trace", conds, S, seed0=20000).to_csv(os.path.join(RES, "e8_scale_trace.csv"), index=False)
    import random
    from env import synthesize
    rows = []
    for label, P in conds[1:]:
        for seed in range(20000, 20000 + S):
            world = build_world(P, seed)
            for tt, tools in enumerate(world["task_tools"]):
                skill = synthesize(tt, tools, world["contract"], 0, P.recall, random.Random(seed + tt), P.recorder)
                rows.append(dict(recorder=label, seed=seed, skill=tt,
                                 recall=len(skill.deps & skill.true_deps) / len(skill.true_deps),
                                 precision=len(skill.deps & skill.true_deps) / len(skill.deps)))
    pd.DataFrame(rows).to_csv(os.path.join(RES, "e9_trace_recall.csv"), index=False)
    print("done")


if __name__ == "__main__":
    main()
