"""Run a planning job through plan.py (same hydra args, same seed) and also dump the planned actions.

plan.py does not save the actions it executes; the per-episode analyses (repro/eval/pusht_fulldemo_analyze.py,
pusht_cost_landscape.py) replay them, so the job is run through this wrapper instead of plan.py directly.
Nothing else changes: logs.json, plan_targets.pkl and the videos are what plan.py writes.

usage (after `source repro/env.sh`, same DINO_WM_* env as repro/queue/run_plan.sh):
    CUDA_VISIBLE_DEVICES=<gpu> python repro/eval/plan_dump_actions.py <out_dir> <plan.py hydra args...> ckpt_base_path=$DINO_CKPT
Writes <out_dir>/diag_actions.pkl: MPC -> actions (all executed, normalized), per-iteration taken actions,
is_success, action_len; open-loop GD/CEM -> final actions.
"""
import os, sys, pickle
THREAD0 = os.environ["THREAD0"]
sys.path.insert(0, THREAD0)
out_dir = os.path.abspath(sys.argv[1])
os.makedirs(out_dir, exist_ok=True)
os.chdir(THREAD0)
sys.argv = ["plan.py"] + sys.argv[2:] + [f"hydra.run.dir={out_dir}"]

import runpy
from planning.mpc import MPCPlanner

_orig_plan = {}

def _wrap(cls):
    orig = cls.plan
    def plan(self, obs_0, obs_g, actions=None):
        res = orig(self, obs_0, obs_g, actions)
        if not getattr(self, "_diag_is_sub", False):
            d = {"actions": res[0].detach().cpu(), "action_len": res[1]}
            if isinstance(self, MPCPlanner):
                d["planned_actions"] = [a.detach().cpu() for a in self.planned_actions]
                d["is_success"] = self.is_success.copy()
            with open(os.path.join(out_dir, "diag_actions.pkl"), "wb") as f:
                pickle.dump(d, f)
            print("diag: dumped planned actions", flush=True)
        return res
    cls.plan = plan

from planning.cem import CEMPlanner
from planning.gd import GDPlanner
for c in (MPCPlanner, CEMPlanner, GDPlanner):
    _wrap(c)
_orig_init = MPCPlanner.__init__
def _mpc_init(self, *a, **k):
    _orig_init(self, *a, **k)
    self.sub_planner._diag_is_sub = True
MPCPlanner.__init__ = _mpc_init

runpy.run_path(os.path.join(THREAD0, "plan.py"), run_name="__main__")
