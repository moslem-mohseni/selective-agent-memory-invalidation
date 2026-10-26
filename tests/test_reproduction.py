import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from env import Skill, execute, make_contract, synthesize, trace_dependencies
import random


def test_hash_seed_does_not_change_simulation():
    code = "from sim import *; import json; p=Params(); print(json.dumps(run(p,build_world(p,8),Selective('clause')),sort_keys=True))"
    outputs = []
    for hash_seed in ("1", "983"):
        env = dict(os.environ, PYTHONHASHSEED=hash_seed)
        outputs.append(subprocess.check_output([sys.executable, "-c", code], cwd=Path(__file__).parents[1] / "src", env=env))
    assert outputs[0] == outputs[1]


def test_omitted_required_tool_cannot_succeed():
    contract = make_contract(random.Random(1))
    assert execute(Skill(0, []), [3], contract, {})[0] == "silent_wrong"


def test_trace_extractor_measures_missing_precondition_reads():
    rng = random.Random(2)
    contract = make_contract(rng, 30)
    skill = synthesize(0, [2, 4, 18], contract, 0, 1, rng)
    assert trace_dependencies(skill.steps) == skill.true_deps
    partial = trace_dependencies(skill.steps, False)
    assert partial < skill.true_deps


def test_larger_world_executes():
    from sim import Params, Selective, build_world, run
    p = Params(n_tools=30, n_types=100, recorder="trace_full")
    world = build_world(p, 7)
    assert len(world["contract"].tools) == 30
    assert 0 <= run(p, world, Selective("clause"))["success"] <= 1


def test_execution_cache_preserves_outcomes(monkeypatch):
    import sim
    p = sim.Params()
    expected = [sim.run(p, sim.build_world(p, seed), sim.Selective("clause")) for seed in range(3)]
    monkeypatch.setattr(sim, "_outcome", lambda skill, contract, ctx: execute(skill, ctx["task_tools"][skill.ttype], contract, {}))
    actual = [sim.run(p, sim.build_world(p, seed), sim.Selective("clause")) for seed in range(3)]
    assert actual == expected
