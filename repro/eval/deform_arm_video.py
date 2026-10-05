"""Re-run a deformable (rope / granular) planning job through plan.py and also write videos in which the
xArm6 pusher is seen executing every action.

Why: the DINO-WM data and the evaluator's videos keep ONE frame per action, rendered after the arm has been
reset to its home pose and the particles have settled (FlexEnv.step: reset_robot() + 200 settle steps), so in
them the arm never moves. FlexEnv.step(save_data=True) already renders intermediate frames (arm at its IK pose,
every 40 / 80 sub-steps) into imgs_list, but FlexEnvWrapper.step_multiple keeps only the last one. This tool
keeps them and writes them as video.

It does not change the simulation: no extra pyflex.step(), no extra render, same call order, so the metrics,
planned actions and standard videos of this process are what plan.py alone would produce. The whole job is re-run
because the planned actions are not saved anywhere, and replaying saved actions in a long-lived process would not
match either: a second rollout of the same actions in one process differs from the first (particle positions up to
~0.04 after one push) because state is left over between rollouts.
A re-run is a replicate of the evaluated job (same model, arguments, seed and start/goal states), not a copy: FleX
rollouts are not bit-reproducible across processes and GPUs, so per-episode results can differ from the evaluated
run (our Rope re-run differed from MPC iteration 1 on, the Granular one from iteration 3 on).

usage (after `source repro/env.sh`, same DINO_WM_* env as the original job):
    CUDA_VISIBLE_DEVICES=<gpu> python repro/eval/deform_arm_video.py <out_dir> <plan.py hydra args...> ckpt_base_path=$DINO_TRAIN
(repro/queue/run_plan.sh appends ckpt_base_path itself, so it is not in the job's recorded args.)
<out_dir> gets everything plan.py writes, plus
    arm_video/{plan<i>,output_final}_<idx>_<tag>.mp4   video with the arm moving (fps 24), same panels and labels as
                                                        the evaluator's videos
    diag_actions.pkl                                    planned actions (per MPC iteration, normalized)
and the log line "settled-vs-standard frame mismatches 0/N" checks that the last frame kept for each action equals
the frame the evaluator uses. THREAD0 (set by repro/env.sh) is the checkout whose code runs. Only for deformable_env
(needs SerialVectorEnv).
"""
import os
import sys
import pickle

THREAD0 = os.environ["THREAD0"]
sys.path.insert(0, THREAD0)
out_dir = os.path.abspath(sys.argv[1])
os.makedirs(out_dir, exist_ok=True)
os.chdir(THREAD0)
sys.argv = ["plan.py"] + sys.argv[2:] + [f"hydra.run.dir={out_dir}"]

import runpy
import cv2
import imageio
import numpy as np
import torch
from einops import rearrange

from env.deformable_env.FlexEnvWrapper import FlexEnvWrapper
from planning.evaluator import PlanEvaluator, label_video_panels, VIDEO_HOLD_SEC
from planning.mpc import MPCPlanner
from planning.cem import CEMPlanner
from planning.gd import GDPlanner

FPS = 24
HOLD_START = FPS // 2      # frames: initial state before the first action
HOLD_SETTLED = FPS // 4    # frames: settled state after each action (arm back at home pose)
HOLD_END = VIDEO_HOLD_SEC * FPS  # frames: last frame (same hold as the other eval videos)
CORRECTION = 0.3           # same constant as PlanEvaluator._plot_rollout_compare (darkens env / goal panels)


# --- keep every frame FlexEnv.step() renders, not just the last -------------------------------------------------
_orig_step = FlexEnvWrapper.step      # inherited from FlexEnv
_orig_rollout = FlexEnvWrapper.rollout


def _step(self, action, save_data=False, data=None):
    res = _orig_step(self, action, save_data=save_data, data=data)
    if save_data and res is not None:
        imgs_list = res[1][0]  # (n_frames) x (n_cams, H, W, 5): arm-moving frames + the settled final frame
        # kept as lossless PNG bytes: 10 envs x 25 actions x tens of frames would be GBs of host RAM for granular
        self.arm_acc.append(
            [
                cv2.imencode(".png", np.ascontiguousarray(im[self.camera_view][..., :3][..., ::-1]).astype(np.uint8))[1].tobytes()
                for im in imgs_list
            ]
        )
    return res


def _rollout(self, seed, init_state, actions):
    self.arm_acc = []
    out = _orig_rollout(self, seed, init_state, actions)
    self.arm_frames, self.arm_acc = self.arm_acc, []  # per action: [moving frames..., settled frame] (PNG bytes)
    return out


FlexEnvWrapper.step = _step
FlexEnvWrapper.rollout = _rollout


# --- video ------------------------------------------------------------------------------------------------------
def _compose(e_obs, i_obs, goal):
    """(c, h, w) tensors in [-1, 1] -> uint8 frame [[real, goal], [model, goal]], as in PlanEvaluator."""
    top = torch.cat([e_obs, goal - CORRECTION], dim=2) - CORRECTION
    bottom = torch.cat([i_obs, goal - CORRECTION], dim=2)
    frame = rearrange(torch.cat([top, bottom], dim=1), "c h w -> h w c").detach().cpu().numpy()
    frame = frame * 2 - 1 if frame.min() >= 0 else frame
    return (((np.clip(frame, -1, 1) + 1) / 2) * 255).astype(np.uint8)


_n_mismatch = [0, 0]  # settled frame != standard frame (should stay 0), checked


def _write_arm_videos(ev, e_visuals, i_visuals, successes, filename):
    assert ev.frameskip == 1, "deformable envs use frameskip 1"
    n = min(ev.n_plot_samples, e_visuals.shape[0])
    goal = ev.preprocessor.transform_obs_visual(ev.obs_g["visual"][:n])  # (n, 1, c, h, w)
    os.makedirs("arm_video", exist_ok=True)
    for idx in range(n):
        env = getattr(ev.env.envs[idx], "unwrapped", ev.env.envs[idx])
        acts = env.arm_frames
        assert len(acts) == e_visuals.shape[1] - 1, (len(acts), e_visuals.shape)
        ticks = [(e_visuals[idx, 0], i_visuals[idx, 0], HOLD_START)]
        for k, frames in enumerate(acts, start=1):
            frames = [cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR) for b in frames]  # lossless
            dense = ev.preprocessor.transform_obs_visual(np.stack(frames)[None])[0].float()  # (m, c, h, w)
            _n_mismatch[1] += 1
            _n_mismatch[0] += int(not torch.allclose(dense[-1], e_visuals[idx, k].float(), atol=1e-6))
            # while the arm executes action k the model panel still shows its state after action k-1;
            # it advances together with the settled real frame
            ticks += [(dense[j], i_visuals[idx, k - 1], 1) for j in range(len(frames) - 1)]
            ticks.append((dense[-1], i_visuals[idx, k], HOLD_SETTLED))
        ticks[-1] = (ticks[-1][0], ticks[-1][1], HOLD_END)
        tag = "success" if successes[idx] else "failure"
        path = os.path.join("arm_video", f"{filename}_{idx}_{tag}.mp4")
        writer = imageio.get_writer(path, fps=FPS)
        for e_obs, i_obs, rep in ticks:
            frame = label_video_panels(_compose(e_obs.cpu(), i_obs.cpu(), goal[idx, 0]))
            for _ in range(rep):
                writer.append_data(frame)
        writer.close()
    print(f"arm video: wrote {n} x {filename}; settled-vs-standard frame mismatches {_n_mismatch[0]}/{_n_mismatch[1]}",
          flush=True)


_orig_plot = PlanEvaluator._plot_rollout_compare


def _plot(self, e_visuals, i_visuals, successes, save_video=False, filename="", **kw):
    _orig_plot(self, e_visuals, i_visuals, successes, save_video=save_video, filename=filename, **kw)
    if save_video:
        _write_arm_videos(self, e_visuals, i_visuals, successes, filename)


PlanEvaluator._plot_rollout_compare = _plot


# --- dump the planned actions (not saved by plan.py) -----------------------------------------------------------
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
            print("dumped planned actions", flush=True)
        return res

    cls.plan = plan


for _c in (MPCPlanner, CEMPlanner, GDPlanner):
    _wrap(_c)
_orig_init = MPCPlanner.__init__


def _mpc_init(self, *a, **k):
    _orig_init(self, *a, **k)
    self.sub_planner._diag_is_sub = True


MPCPlanner.__init__ = _mpc_init

runpy.run_path(os.path.join(THREAD0, "plan.py"), run_name="__main__")
