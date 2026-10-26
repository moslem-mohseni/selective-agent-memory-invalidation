import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from env import (BENIGN, PRECOND, RENAME, UNIT, apply_change, execute, make_contract,
                 synthesize, truly_broken)
from sim import KeepAll, Params, Selective, build_world, run


def _setup(kind, tool=3, seed=0):
    rng = random.Random(seed)
    c = make_contract(rng)
    tools = [tool, 5, 7]
    sk = synthesize(0, tools, c, 0, 1.0, rng)
    new, diff = apply_change(c, rng, [kind], [tool])
    return c, new, diff, sk, tools


def test_fresh_skill_runs_ok():
    c, _, _, sk, tools = _setup(BENIGN)
    assert execute(sk, tools, c, {})[0] == "ok"


def test_rename_visible_error():
    c, new, diff, sk, tools = _setup(RENAME)
    assert execute(sk, tools, new, {})[0] == "error"
    assert ("args", 3) in diff


def test_unit_change_is_silent():
    c, new, diff, sk, tools = _setup(UNIT)
    assert execute(sk, tools, new, {})[0] == "silent_wrong"


def test_benign_does_not_break():
    c, new, diff, sk, tools = _setup(BENIGN)
    assert not truly_broken(sk, tools, new)


def test_precondition_breaks_unless_prereq_already_present():
    c, new, diff, sk, tools = _setup(PRECOND, tool=7)
    pre = next(iter(new.tools[7].pre))
    present_before = pre in [s.tool for s in sk.steps if True][: [s.tool for s in sk.steps].index(7)]
    assert truly_broken(sk, tools, new) == (not present_before)


def test_perfect_recall_selective_keeps_all_healthy_skills():
    P = Params(recall=1.0, q_det=1.0)
    for seed in range(5):
        w = build_world(P, seed)
        r = run(P, w, Selective("clause"))
        assert r["healthy_retention"] == 1.0
        assert r["missed_stale"] == 0


def test_keep_all_never_probes():
    P = Params()
    r = run(P, build_world(P, 1), KeepAll())
    assert r["probe_cost"] == 0.0


def test_determinism():
    P = Params()
    a = run(P, build_world(P, 3), Selective("clause"))
    b = run(P, build_world(P, 3), Selective("clause"))
    assert a == b
