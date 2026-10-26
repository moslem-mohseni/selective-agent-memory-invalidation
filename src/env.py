"""Executable tool-contract environment for procedural-memory experiments.

A *contract* describes N_TOOLS tools: accepted argument names, unit factors
applied to argument values, and preconditions (other tools that must already
have run). A *skill* is a recorded plan (list of tool steps) learned under one
contract version. Executing a skill against a *different* contract can fail
visibly (renamed argument, unmet precondition) or silently (unit change).
Ground truth is obtained by actually executing the skill.
"""
from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass, field

N_TOOLS = 12

# Clause kinds. A clause is a tuple: ('args', t) | ('pre', t) | ('unit', t, arg) | ('opt', t)
RENAME, PRECOND, UNIT, BENIGN = "rename", "precond", "unit", "benign"
CHANGE_TYPES = (RENAME, PRECOND, UNIT, BENIGN)


@dataclass
class ToolSpec:
    args: dict            # canonical arg -> currently accepted name
    pre: set              # tool ids that must be done first
    unit: dict            # canonical arg -> unit factor
    n_optional: int = 0   # benign optional fields (never breaks a skill)


@dataclass
class Contract:
    tools: dict
    version: int = 1

    def copy(self):
        return copy.deepcopy(self)


def make_contract(rng: random.Random, n_tools: int = N_TOOLS) -> Contract:
    tools = {}
    for t in range(n_tools):
        n = rng.choice([2, 3])
        args = {f"t{t}_p{i}": f"t{t}_p{i}" for i in range(n)}
        unit = {a: 1.0 for a in args}
        tools[t] = ToolSpec(args=args, pre=set(), unit=unit)
    return Contract(tools=tools)


@dataclass
class Step:
    tool: int
    names: dict     # canonical arg -> name used when learned
    units: dict     # canonical arg -> unit factor assumed when learned


@dataclass
class Skill:
    ttype: int
    steps: list
    learned_at: int = 0
    deps: set = field(default_factory=set)       # recorded (possibly incomplete) clauses
    true_deps: set = field(default_factory=set)  # all clauses the plan actually relies on


def _closure_order(tools, contract: Contract):
    """Add transitive precondition tools, ordered so that prerequisites come first."""
    seen, order = set(), []

    def visit(t):
        if t in seen:
            return
        seen.add(t)
        for p in sorted(contract.tools[t].pre):
            visit(p)
        order.append(t)

    for t in tools:
        visit(t)
    return order


def synthesize(ttype: int, task_tools: list, contract: Contract, learned_at: int,
               recall: float, rng: random.Random, recorder: str = "bernoulli") -> Skill:
    """Plan a skill valid under `contract` (what a successful exploration/repair yields)."""
    order = _closure_order(task_tools, contract)
    steps, true_deps = [], set()
    for t in order:
        spec = contract.tools[t]
        steps.append(Step(tool=t, names=dict(spec.args), units=dict(spec.unit)))
        true_deps.add(("args", t))
        true_deps.add(("pre", t))
        for a in spec.args:
            true_deps.add(("unit", t, a))
    # Sorting is essential: Python hash randomisation must not change the
    # assignment of random draws to clauses across processes or platforms.
    if recorder == "bernoulli":
        deps = {c for c in sorted(true_deps) if rng.random() < recall}
    else:
        deps = trace_dependencies(steps, include_preconditions=recorder == "trace_full")
    return Skill(ttype=ttype, steps=steps, learned_at=learned_at, deps=deps, true_deps=true_deps)


def trace_dependencies(steps, include_preconditions=True):
    """Extract clause use from an executed plan's structured call trace.

    This deliberately simple extractor handles the simulator grammar only.
    An arguments-only ablation omits prerequisite-contract reads.
    """
    deps = set()
    for step in steps:
        deps.add(("args", step.tool))
        if include_preconditions:
            deps.add(("pre", step.tool))
        deps.update(("unit", step.tool, arg) for arg in step.units)
    return deps


def execute(skill: Skill, task_tools: list, contract: Contract, values: dict):
    """Run skill on `contract`. Returns ('ok'|'error'|'silent_wrong', detail)."""
    done, state = set(), {}
    for s in skill.steps:
        spec = contract.tools[s.tool]
        accepted = set(spec.args.values())
        if set(s.names.values()) != accepted:
            return "error", f"unknown/missing argument in tool {s.tool}"
        if not spec.pre <= done:
            return "error", f"precondition unmet for tool {s.tool}"
        for a in spec.args:
            raw = values.get((s.tool, a), 1.0) / s.units[a]  # value as the agent sends it
            state[(s.tool, a)] = raw * spec.unit[a]          # value as the tool interprets it
        done.add(s.tool)
    for t in task_tools:
        for a in contract.tools[t].args:
            want = values.get((t, a), 1.0)
            got = state.get((t, a), float("nan"))
            if not math.isfinite(got) or abs(got - want) > 1e-9 * max(1.0, abs(want)):
                return "silent_wrong", f"tool {t} arg {a} misinterpreted"
    return "ok", ""


def truly_broken(skill: Skill, task_tools: list, contract: Contract) -> bool:
    return execute(skill, task_tools, contract, {})[0] != "ok"


def apply_change(contract: Contract, rng: random.Random, kinds: list, tools: list):
    """Apply one change per (kind, tool); return new contract and the clause diff."""
    new = contract.copy()
    diff = set()
    for kind, t in zip(kinds, tools):
        spec = new.tools[t]
        if kind == RENAME:
            a = rng.choice(sorted(spec.args))
            spec.args[a] = spec.args[a] + "_v2"
            diff.add(("args", t))
        elif kind == PRECOND:
            cands = [u for u in range(t) if u not in spec.pre]
            if not cands:  # tool 0 cannot gain a lower-index prerequisite
                kind = RENAME
                a = rng.choice(sorted(spec.args))
                spec.args[a] = spec.args[a] + "_v2"
                diff.add(("args", t))
            else:
                spec.pre.add(rng.choice(cands))
                diff.add(("pre", t))
        elif kind == UNIT:
            a = rng.choice(sorted(spec.args))
            spec.unit[a] *= rng.choice([100.0, 1000.0])
            diff.add(("unit", t, a))
        else:
            spec.n_optional += 1
            diff.add(("opt", t))
    new.version += 1
    return new, diff
