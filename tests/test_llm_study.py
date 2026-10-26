import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src" / "llm_study"))
from run import parse_plan, prompt
from env import execute, make_contract
from sim import Params, build_world


def test_parse_exact_plan_and_missing_tool():
    contract = make_contract(random.Random(1))
    spec = contract.tools[3]
    plan = json.dumps({"steps": [{"tool": 3, "names": list(spec.args.values()), "units": list(spec.unit.values())}]})
    assert execute(parse_plan(plan, contract, 0), [3], contract, {})[0] == "ok"
    assert execute(parse_plan('{"steps":[]}', contract, 0), [3], contract, {})[0] != "ok"


def test_prompt_uses_public_contract_without_outcome_labels():
    world = build_world(Params(), 30001)
    text = prompt(world, [3], None)[0]["content"]
    assert "requested_tools" in text and "saved_plan" in text
    assert "success" not in text and "true_deps" not in text
