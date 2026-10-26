"""Study 2 protocol: LLM execution/repair with exact contract-ledger scoring.

An episode is one post-change task under one maintenance policy. The fixed
protocol has 100 independent worlds and five paired policies (500 scored
policy episodes). A mock run exercises the harness, never supplies evidence
about a language model. No simulated success probability is used here.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from env import Skill, Step, execute, synthesize
from sim import ClearAll, KeepAll, Params, RevalidateAll, Selective, build_world
from stats_utils import mean_ci, paired_diff_ci, sign_flip_p, holm
from client import LLMClient, discover

OUT = HERE.parents[1] / "results" / "llm"
POLICIES = [KeepAll(), ClearAll(), RevalidateAll(), Selective("tool"), Selective("clause")]


def prompt(world, tools, stored):
    contract = world["new"]
    public = {str(t): dict(names=list(spec.args.values()), canonical=list(spec.args),
                          factors=list(spec.unit.values()), requires=sorted(spec.pre))
              for t, spec in contract.tools.items()}
    memory = None if stored is None else [dict(tool=s.tool, names=list(s.names.values()), units=list(s.units.values())) for s in stored.steps]
    return [{"role": "user", "content":
        "Execute a synthetic task under the current tool contract. Return only JSON with key steps, a list of objects "
        "with tool (integer), names (accepted argument names in canonical order), units (current unit factors in canonical order). "
        "Include prerequisite tools before their dependents and all requested tools. A saved plan may be stale; repair it if needed. "
        + json.dumps(dict(requested_tools=tools, contract=public, saved_plan=memory), sort_keys=True)}]


def parse_plan(text, contract, tt):
    data = json.loads(text)
    steps = []
    if not isinstance(data["steps"], list):
        raise ValueError("steps must be a list")
    for s in data["steps"]:
        tool = s["tool"]
        canonical = list(contract.tools[tool].args)
        if len(s["names"]) != len(canonical) or len(s["units"]) != len(canonical):
            raise ValueError("argument cardinality")
        units = dict(zip(canonical, map(float, s["units"])))
        if any(not (0 < v <= 1e9) for v in units.values()):
            raise ValueError("invalid units")
        steps.append(Step(tool, dict(zip(canonical, s["names"])), units))
    return Skill(tt, steps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["mock", "ollama", "llamacpp", "openai", "anthropic"], default="mock")
    ap.add_argument("--model", default="mock")
    ap.add_argument("--episodes", type=int, default=100)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    client = LLMClient(a.backend, a.model, OUT / "cache", max_tokens=768,
                       price_input_per_million=float(os.environ["LLM_INPUT_USD_PER_MILLION"]) if os.environ.get("LLM_INPUT_USD_PER_MILLION") else None,
                       price_output_per_million=float(os.environ["LLM_OUTPUT_USD_PER_MILLION"]) if os.environ.get("LLM_OUTPUT_USD_PER_MILLION") else None)
    rows = []
    for seed in range(30000, 30000 + a.episodes):
        p = Params()
        world = build_world(p, seed)
        tt = seed % p.n_types
        for policy in POLICIES:
            rng = random.Random(seed)
            mem = {i: synthesize(i, tools, world["contract"], 0, p.recall, rng) for i, tools in enumerate(world["task_tools"])}
            context = dict(P=p, rng=rng, task_tools=world["task_tools"])
            # Study 2 probes use the exact executable postcondition, avoiding
            # the Study 1 parametric sensitivity assumption.
            context["P"] = Params(q_det=1.0)
            probe_units = policy.on_change(mem, world["diff"], world["contract"], world["new"], context)
            messages = prompt(world, world["task_tools"][tt], mem.get(tt))
            text = client.complete(messages, seed, temperature=0)
            try:
                skill = parse_plan(text, world["new"], tt)
                outcome, detail = execute(skill, world["task_tools"][tt], world["new"], {})
            except (ValueError, KeyError, TypeError, ZeroDivisionError) as exc:
                outcome, detail = "invalid_plan", str(exc)
            rows.append(dict(seed=seed, policy=policy.name, success=int(outcome == "ok"),
                             outcome=outcome, detail=detail, llm_calls=1, probe_units=probe_units,
                             prompt_sha256=client.last["prompt_sha256"], mock=a.backend == "mock"))
    filename = "dry_run.csv" if a.backend == "mock" else "episodes.csv"
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / filename, index=False)
    status = dict(status="dry_run_only" if a.backend == "mock" else "completed", model=a.model, backend=a.backend,
                  paired_worlds=a.episodes, seed_start=30000, temperature=0,
                  real_llm_run=a.backend != "mock", availability=discover())
    acquisition = HERE / "availability_attempt.json"
    if acquisition.exists():
        status["local_runtime_attempt"] = json.loads(acquisition.read_text())
    (OUT / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    if a.backend != "mock":
        summary, comparisons = [], []
        for policy, group in frame.groupby("policy", sort=True):
            mean, lo, hi = mean_ci(group.success)
            summary.append(dict(policy=policy, success=mean, lo=lo, hi=hi, llm_calls=group.llm_calls.mean(), probe_units=group.probe_units.mean()))
            if policy != "keep_all":
                base = frame[frame.policy == "keep_all"].sort_values("seed").success.values
                values = group.sort_values("seed").success.values
                diff, low, high, dz = paired_diff_ci(values, base)
                comparisons.append(dict(policy=policy, diff=diff, lo=low, hi=high, dz=dz, p=sign_flip_p(values, base)))
        pd.DataFrame(summary).to_csv(OUT / "summary.csv", index=False)
        paired = pd.DataFrame(comparisons)
        paired["p_holm"] = holm(paired.p)
        paired.to_csv(OUT / "paired.csv", index=False)
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
