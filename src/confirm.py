"""Run only the fresh-seed confirmation experiment (E7)."""
import dataclasses, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from run_experiments import RES, sweep
from sim import Params
base = Params()
conds = [("main", base), ("recall=0.7", dataclasses.replace(base, recall=0.7)), ("recall=0.5", dataclasses.replace(base, recall=0.5))]
sweep("E7_confirm", conds, 300, seed0=10000).to_csv(os.path.join(RES, "e7_confirm.csv"), index=False)
print("done")
