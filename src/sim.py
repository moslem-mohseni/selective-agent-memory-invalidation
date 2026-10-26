"""Task-stream simulation and memory-maintenance policies for Paper 1."""
from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np

from env import (BENIGN, CHANGE_TYPES, N_TOOLS, PRECOND, RENAME, UNIT, apply_change,
                 execute, make_contract, synthesize, truly_broken)


@dataclass
class Params:
    n_tools: int = 12
    recorder: str = "bernoulli"
    n_types: int = 40
    horizon: int = 400
    change_at: int = 100
    n_changed_tools: int = 2
    mix: tuple = (0.25, 0.25, 0.25, 0.25)  # rename, precond, unit, benign
    recall: float = 0.9          # dependency-recording recall
    q_det: float = 0.7           # probe detects a silent semantic break
    p_exec: float = 0.95         # agent executes a valid skill correctly
    p_explore: float = 0.75      # exploration from scratch yields a valid plan
    p_repair: float = 0.85       # repair after a visible error succeeds
    c_exec: float = 1.0          # LLM calls to run a stored skill
    c_repair: float = 3.0
    c_probe: float = 0.5
    zipf_s: float = 1.0
    ttl: int = 75
    p_feedback: float = 0.0      # a silent wrong result is later noticed and the skill dropped


def c_explore(n_steps: int) -> float:
    return 2.0 * n_steps + 2.0


class Policy:
    name = "base"
    uses_diff = False

    def on_change(self, mem, diff, old, new, ctx):
        """Return probe cost spent; may delete entries from mem."""
        return 0.0

    def pre_step(self, mem, t, ctx):
        return


class KeepAll(Policy):
    name = "keep_all"


class ClearAll(Policy):
    name = "clear_all"

    def on_change(self, mem, diff, old, new, ctx):
        mem.clear()
        return 0.0


class TTL(Policy):
    name = "ttl"

    def pre_step(self, mem, t, ctx):
        ttl = ctx["P"].ttl
        for k in [k for k, s in mem.items() if t - s.learned_at > ttl]:
            del mem[k]


def _probe(skill, ctx, new):
    """Probe a skill against the new contract: visible errors always found, silent
    breaks found with probability q_det. Returns True if the skill is judged broken."""
    task_tools = ctx["task_tools"][skill.ttype]
    res, _ = _outcome(skill, new, ctx)
    if res == "error":
        return True
    if res == "silent_wrong":
        return ctx["rng"].random() < ctx["P"].q_det
    return False


def _outcome(skill, contract, ctx):
    """Memoise pure execution; hold the skill reference to prevent id reuse."""
    cache = ctx.setdefault("execution_cache", {})
    key = (id(skill), contract.version)
    if key not in cache:
        cache[key] = (skill, execute(skill, ctx["task_tools"][skill.ttype], contract, {}))
    return cache[key][1]


class RevalidateAll(Policy):
    name = "revalidate_all"

    def on_change(self, mem, diff, old, new, ctx):
        cost = 0.0
        for k in list(mem):
            cost += ctx["P"].c_probe
            if _probe(mem[k], ctx, new):
                del mem[k]
        return cost


class Selective(Policy):
    """Flag skills whose recorded dependencies intersect the contract diff, then probe
    only the flagged ones. `granularity`='clause' uses clause-level diffs; 'tool' flags
    every skill that uses any changed tool. `audit` probes that fraction of unflagged
    skills to catch dependencies the recorder missed."""

    uses_diff = True

    def __init__(self, granularity="clause", audit=0.0, probe_flagged=True, name=None, tier2=0.0):
        self.granularity = granularity
        self.audit = audit
        self.tier2 = tier2  # probability of probing tool-level-only hits (tiered selection)
        self.probe_flagged = probe_flagged
        self.name = name or f"selective_{granularity}" + (f"_audit{audit:g}" if audit else "") + (f"_tiered{tier2:g}" if tier2 else "")

    def on_change(self, mem, diff, old, new, ctx):
        cost = 0.0
        rng = ctx["rng"]
        changed_tools = {c[1] for c in diff}
        flagged, unflagged = [], []
        for k, s in mem.items():
            tool_hit = any(c[1] in changed_tools for c in s.deps)
            if self.granularity == "clause":
                hit = bool(s.deps & diff)
                if not hit and self.tier2 and tool_hit and rng.random() < self.tier2:
                    hit = True
            else:
                hit = tool_hit
            (flagged if hit else unflagged).append(k)
        for k in flagged:
            if not self.probe_flagged:
                del mem[k]
                continue
            cost += ctx["P"].c_probe
            if _probe(mem[k], ctx, new):
                del mem[k]
        if self.audit > 0:
            for k in unflagged:
                if rng.random() < self.audit:
                    cost += ctx["P"].c_probe
                    if _probe(mem[k], ctx, new):
                        del mem[k]
        return cost


def standard_policies(audit=0.1):
    return [KeepAll(), ClearAll(), TTL(), RevalidateAll(),
            Selective("tool"), Selective("clause"), Selective("clause", audit=audit),
            Selective("clause", tier2=0.5)]


def zipf_weights(n, s):
    w = np.array([1.0 / (i + 1) ** s for i in range(n)])
    return w / w.sum()


def build_world(P: Params, seed: int):
    """Everything shared across policies for one seed (common random numbers)."""
    rng = random.Random(seed)
    contract = make_contract(rng, P.n_tools)
    task_tools = []
    for _ in range(P.n_types):
        k = rng.choice([3, 4, 5, 6])
        task_tools.append(sorted(rng.sample(range(P.n_tools), k)))
    nprng = np.random.default_rng(seed + 1)
    w = zipf_weights(P.n_types, P.zipf_s)
    stream = nprng.choice(P.n_types, size=P.horizon, p=w)
    u = nprng.random((P.horizon, 3))  # exec, explore/repair, spare
    kinds = list(rng.choices(CHANGE_TYPES, weights=P.mix, k=P.n_changed_tools))
    tools = rng.sample(range(P.n_tools), P.n_changed_tools)
    new_contract, diff = apply_change(contract, rng, kinds, tools)
    return dict(contract=contract, new=new_contract, diff=diff, task_tools=task_tools,
                stream=stream, u=u, kinds=kinds, tools=tools, seed=seed)


def run(P: Params, world, policy: Policy, seed_offset=0):
    rng = random.Random(world["seed"] * 7919 + 13 + seed_offset)
    task_tools = world["task_tools"]
    contract = world["contract"]
    mem = {}
    # warm start: every skill learned under the old contract
    for tt in range(P.n_types):
        # learning times are staggered over one TTL window so expiry is not aligned with the change
        mem[tt] = synthesize(tt, task_tools[tt], contract, -rng.randint(0, P.ttl), P.recall, rng, P.recorder)
    ctx = dict(P=P, rng=rng, task_tools=task_tools)

    cost = probe_cost = 0.0
    n = succ = stale_visible = stale_silent = explores = repairs = 0
    retained_ok = retained_n = missed_stale = flagged_healthy_dropped = 0
    cur = contract
    post_cost = 0.0
    for t in range(P.horizon):
        if t == P.change_at:
            before = set(mem)
            healthy_before = {k for k in before
                              if _outcome(mem[k], world["new"], ctx)[0] == "ok"}
            pc = policy.on_change(mem, world["diff"], contract, world["new"], ctx)
            probe_cost += pc
            cost += pc
            cur = world["new"]
            # diagnostics at the moment right after maintenance
            retained_ok = len(healthy_before & set(mem))
            retained_n = len(healthy_before)
            missed_stale = sum(1 for k in mem if _outcome(mem[k], cur, ctx)[0] != "ok")
        policy.pre_step(mem, t, ctx)
        tt = int(world["stream"][t])
        tools = task_tools[tt]
        u_exec, u_other, _ = world["u"][t]
        step_cost = 0.0
        ok = False
        if tt in mem:
            sk = mem[tt]
            step_cost += P.c_exec
            res, _ = _outcome(sk, cur, ctx)
            if res == "ok":
                ok = u_exec < P.p_exec
            elif res == "error":
                if t >= P.change_at:
                    stale_visible += 1
                step_cost += P.c_repair
                repairs += 1
                if u_other < P.p_repair:
                    mem[tt] = synthesize(tt, tools, cur, t, P.recall, rng, P.recorder)
                    ok = u_exec < P.p_exec
                else:
                    del mem[tt]
            else:  # silent wrong result, agent believes it succeeded
                if t >= P.change_at:
                    stale_silent += 1
                ok = False
                if world["u"][t][2] < P.p_feedback:
                    del mem[tt]
        else:
            step_cost += c_explore(len(tools))
            explores += 1
            if u_other < P.p_explore:
                mem[tt] = synthesize(tt, tools, cur, t, P.recall, rng, P.recorder)
                ok = True
        if t >= P.change_at:
            n += 1
            succ += ok
            cost += step_cost
            post_cost += step_cost
    return dict(
        success=succ / n, cost_per_task=(post_cost + probe_cost) / n,
        probe_cost=probe_cost, stale_visible=stale_visible, stale_silent=stale_silent,
        explores=explores, repairs=repairs,
        healthy_retention=(retained_ok / retained_n) if retained_n else 1.0,
        missed_stale=missed_stale, n_post=n,
    )
