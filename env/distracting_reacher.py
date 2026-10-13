"""DMC Reacher with Distracting-Control-Suite style video backgrounds.

Implements the DINO-WM env interface (see env/pointmaze/point_maze_wrapper.py and
env/pusht/pusht_wrapper.py) so this is a drop-in target for plan.py /
planning.evaluator.PlanEvaluator / env.venv.SubprocVectorEnv:

    action_dim                              (attribute)
    sample_random_init_goal_states(seed) -> (init_state, goal_state)
    update_env(env_info)
    eval_state(goal_state, cur_state)    -> {'success', 'state_dist', ...}
    prepare(seed, init_state)            -> (obs, state)
    step(action)                         -> obs, reward, done, info  (info['state'])
    step_multiple(actions)               -> obses, rewards, dones, infos
    rollout(seed, init_state, actions)   -> (obses, states)

State layout (8 dims, STATE_KEYS):
    [shoulder, wrist, shoulder_vel, wrist_vel, finger_x, finger_y, target_x, target_y]
Only the first 4 dims are exposed as `proprio`: the target position must be read
out of the image, otherwise the distractor experiment is vacuous (the planner
could solve the task from proprio alone and never consult the visual latent).

Background replacement
----------------------
The Distracting Control Suite replaces the *skybox* texture with a video frame and
makes the ground plane transparent so the skybox is visible from the overhead
reacher camera. The upstream package (google-research/distracting_control) writes
to `mjModel.tex_rgb`, which MuJoCo 3.x renamed to `tex_data`, so it breaks against
the mujoco==3.2.7 pinned by this repo. We therefore implement the same mechanism
here (handling both attribute names) rather than depending on that package.
Color and camera distractions are deliberately NOT implemented: this experiment
varies the background only.

MUJOCO_GL is read from the environment (default 'egl'); set MUJOCO_GL=osmesa on
CPU-only machines. It must be set before dm_control is imported, hence the lazy
import in _lazy_dm_control().
"""

import os
import glob
import numpy as np

STATE_KEYS = [
    "shoulder",
    "wrist",
    "shoulder_vel",
    "wrist_vel",
    "finger_x",
    "finger_y",
    "target_x",
    "target_y",
]
STATE_DIM = len(STATE_KEYS)
PROPRIO_DIM = 4  # joint angles + joint velocities only

# dm_control reacher target radii (dm_control/suite/reacher.py)
TARGET_SIZE = {"easy": 0.05, "hard": 0.015}
TARGET_RADIUS_RANGE = (0.05, 0.20)
# dm_control reacher joint ranges: shoulder is unlimited, wrist is +-160 deg
WRIST_LIMIT = np.deg2rad(160.0)

_DM = {}


def _lazy_dm_control():
    """Import dm_control/mujoco only after MUJOCO_GL has been fixed up."""
    if _DM:
        return _DM
    os.environ.setdefault("MUJOCO_GL", "egl")
    if os.environ["MUJOCO_GL"] == "egl":
        # which GPU renders; set per worker by scripts/collect_data.py
        os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")
    import mujoco
    from dm_control import suite

    _DM["mujoco"] = mujoco
    _DM["suite"] = suite
    return _DM


# --------------------------------------------------------------------------- #
# background video sources
# --------------------------------------------------------------------------- #


class BackgroundSource:
    """Frames of one background 'video', loaded lazily and cached.

    Two sources are supported:
      davis     : $DAVIS_PATH/JPEGImages/480p/<video>/*.jpg, split according to
                  ImageSets/2017/{train,val}.txt
      synthetic : deterministic procedurally generated frames, so the pipeline can
                  be smoke-tested without the ~4GB DAVIS download.
    """

    def __init__(self, name, frame_paths=None, synthetic_seed=None, n_frames=60):
        self.name = name
        self.frame_paths = frame_paths
        self.synthetic_seed = synthetic_seed
        self.n_frames = len(frame_paths) if frame_paths is not None else n_frames
        self._cache = {}

    def __len__(self):
        return self.n_frames

    def get(self, idx, out_hw):
        """Frame `idx` (wrapped), resized to out_hw, as uint8 HWC."""
        idx = int(idx) % self.n_frames
        key = (idx, out_hw)
        if key not in self._cache:
            if self.frame_paths is not None:
                from PIL import Image

                with Image.open(self.frame_paths[idx]) as im:
                    im = im.convert("RGB").resize((out_hw[1], out_hw[0]), Image.BILINEAR)
                    frame = np.asarray(im, dtype=np.uint8)
            else:
                frame = _synthetic_frame(self.synthetic_seed, idx, out_hw, self.n_frames)
            if len(self._cache) > 8:  # reads are sequential; keep the cache tiny
                self._cache.clear()
            self._cache[key] = frame
        return self._cache[key]


def _synthetic_frame(seed, idx, out_hw, n_frames):
    """Deterministic moving-pattern frame, for runs without DAVIS."""
    h, w = out_hw
    rng = np.random.RandomState(seed)
    c1 = rng.randint(0, 255, size=3).astype(np.float32)
    c2 = rng.randint(0, 255, size=3).astype(np.float32)
    freq = rng.uniform(2.0, 6.0)
    # sin(x + p) * cos(y - p) repeats every pi of phase, so this makes the video
    # repeat exactly every n_frames, as the wrap in BackgroundSource.get assumes.
    # (A 2*pi/30 step repeated every 15 frames: shifts of 15, 30, ... frames,
    # e.g. a T4 mismatch of half a 60-frame video, were no shift at all.)
    phase = np.pi * idx / n_frames
    yy, xx = np.meshgrid(
        np.linspace(0, 1, h, dtype=np.float32),
        np.linspace(0, 1, w, dtype=np.float32),
        indexing="ij",
    )
    m = 0.5 * (
        1.0
        + np.sin(freq * 2 * np.pi * xx + phase) * np.cos(freq * 2 * np.pi * yy - phase)
    )
    frame = c1[None, None, :] * m[..., None] + c2[None, None, :] * (1 - m[..., None])
    return frame.astype(np.uint8)


def davis_video_names(davis_path, split):
    """Video ids of a DAVIS 2017 split ('train' or 'val')."""
    list_file = os.path.join(davis_path, "ImageSets", "2017", f"{split}.txt")
    if not os.path.exists(list_file):
        raise FileNotFoundError(
            f"DAVIS split list not found: {list_file}\n"
            "Expected the DAVIS 2017 trainval layout "
            "($DAVIS_PATH/{ImageSets,JPEGImages}). Run scripts/setup_instance.sh."
        )
    with open(list_file) as f:
        return sorted(ln.strip() for ln in f if ln.strip())


def build_background_sources(bg_source, split, davis_path=None, n_synthetic=30):
    """Ordered BackgroundSource list for a split. The order is stable, so a seed
    maps to the same video on every machine."""
    if bg_source == "none":
        return []
    if bg_source == "synthetic":
        # disjoint seed ranges so train and val backgrounds never coincide
        base = 10_000 if split == "train" else 20_000
        return [
            BackgroundSource(f"synth_{split}_{i:03d}", synthetic_seed=base + i)
            for i in range(n_synthetic)
        ]
    if bg_source == "davis":
        davis_path = davis_path or os.environ.get("DAVIS_PATH")
        if not davis_path:
            raise ValueError("bg_source='davis' requires davis_path or $DAVIS_PATH")
        sources = []
        for name in davis_video_names(davis_path, split):
            frames = sorted(
                glob.glob(os.path.join(davis_path, "JPEGImages", "480p", name, "*.jpg"))
            )
            if frames:
                sources.append(BackgroundSource(name, frame_paths=frames))
        if not sources:
            raise FileNotFoundError(
                f"No DAVIS frames under {davis_path}/JPEGImages/480p for split "
                f"'{split}'."
            )
        return sources
    raise ValueError(f"Unknown bg_source: {bg_source}")


# --------------------------------------------------------------------------- #
# skybox texture plumbing
# --------------------------------------------------------------------------- #


def _tex_buffer(model):
    """MuJoCo renamed tex_rgb -> tex_data in 3.x."""
    if hasattr(model, "tex_data"):
        return model.tex_data
    if hasattr(model, "tex_rgb"):
        return model.tex_rgb
    raise RuntimeError(
        "mjModel has neither tex_data nor tex_rgb; unsupported mujoco version."
    )


class _SkyboxPainter:
    """Writes images into the skybox texture of a dm_control Physics."""

    def __init__(self, physics):
        mujoco = _lazy_dm_control()["mujoco"]
        self.physics = physics
        model = physics.model
        tex_id = None
        for i in range(model.ntex):
            if model.tex_type[i] == mujoco.mjtTexture.mjTEXTURE_SKYBOX:
                tex_id = i
                break
        if tex_id is None:
            raise RuntimeError("No skybox texture in this model.")
        self.tex_id = tex_id
        self.height = int(model.tex_height[tex_id])
        self.width = int(model.tex_width[tex_id])
        nchannel = getattr(model, "tex_nchannel", None)
        self.nchannel = int(nchannel[tex_id]) if nchannel is not None else 3
        self.adr = int(model.tex_adr[tex_id])
        self.size = self.height * self.width * self.nchannel
        # a skybox is a cube map stored as 6 stacked square faces
        self.n_faces = (
            6 if (self.height % self.width == 0 and self.height // self.width == 6) else 1
        )
        self.face_hw = (
            (self.width, self.width) if self.n_faces == 6 else (self.height, self.width)
        )

    def _upload(self):
        mujoco = _lazy_dm_control()["mujoco"]
        with self.physics.contexts.gl.make_current() as ctx:
            ctx.call(
                mujoco.mjr_uploadTexture,
                self.physics.model.ptr,
                self.physics.contexts.mujoco.ptr,
                self.tex_id,
            )

    def paint(self, frame_hwc):
        """frame_hwc: uint8 (face_h, face_w, 3)."""
        if self.nchannel == 3:
            tile = frame_hwc
        else:  # pad, e.g. RGBA textures
            pad = np.full(
                (*frame_hwc.shape[:2], self.nchannel - 3), 255, dtype=np.uint8
            )
            tile = np.concatenate([frame_hwc, pad], axis=-1)
        buf = np.repeat(tile[None], self.n_faces, axis=0).reshape(-1)
        if buf.size != self.size:
            raise RuntimeError(
                f"skybox size mismatch: produced {buf.size}, texture holds {self.size}"
            )
        _tex_buffer(self.physics.model)[self.adr : self.adr + self.size] = buf
        self._upload()  # otherwise render() keeps using the old texture

    def restore(self, default_buf):
        _tex_buffer(self.physics.model)[self.adr : self.adr + self.size] = default_buf
        self._upload()


# --------------------------------------------------------------------------- #
# the env
# --------------------------------------------------------------------------- #


class DistractingReacherWrapper:
    """DMC Reacher + background distractor, in the DINO-WM env interface.

    Background behaviour is driven by plain attributes, so it can be set through
    SubprocVectorEnv.set_env_attr() without modifying env/venv.py:

        bg_enabled      bool  master switch (False => default clean skybox)
        bg_split        str   'train' | 'val'  (which DAVIS split to draw from)
        bg_video_idx    int   index into that split's video list (wrapped)
        bg_frame_offset int   first video frame to show
        bg_dynamic      bool  advance one frame per env step

    Changes take effect on the next prepare()/reset().
    """

    # gym.make() (0.23.1) does `env.unwrapped.spec = spec` and then wraps the env
    # in TimeLimit/OrderEnforcing; venv workers use `env.unwrapped` to set the
    # background attributes. This class is not a gym.Env subclass, so provide both.
    spec = None

    @property
    def unwrapped(self):
        return self

    def __init__(
        self,
        difficulty="easy",
        img_size=224,
        action_repeat=2,
        camera_id=0,
        bg_enabled=False,
        bg_source="davis",
        bg_split="train",
        bg_video_idx=0,
        bg_frame_offset=0,
        bg_dynamic=True,
        ground_plane_alpha=0.0,
        davis_path=None,
        n_synthetic=30,
        time_limit=float("inf"),
    ):
        dm = _lazy_dm_control()
        self.difficulty = difficulty
        self.target_size = TARGET_SIZE[difficulty]
        self.img_size = img_size
        self.action_repeat = action_repeat
        self.camera_id = camera_id
        self.state_dim = STATE_DIM
        self.proprio_dim = PROPRIO_DIM

        self.bg_enabled = bg_enabled
        self.bg_source = bg_source
        self.bg_split = bg_split
        self.bg_video_idx = int(bg_video_idx)
        self.bg_frame_offset = int(bg_frame_offset)
        self.bg_dynamic = bg_dynamic
        self.ground_plane_alpha = ground_plane_alpha
        self.davis_path = davis_path or os.environ.get("DAVIS_PATH")
        self.n_synthetic = n_synthetic

        self._dm_env = dm["suite"].load(
            "reacher", difficulty, task_kwargs={"time_limit": time_limit}
        )
        self.physics = self._dm_env.physics
        self.action_dim = int(self._dm_env.action_spec().shape[0])

        self._bg_sources = {}
        self._painter = None
        self._default_skybox = None
        self._default_ground_alpha = None
        self._frame_counter = 0
        self._step_count = 0
        self.reset_to_state = None
        self.random_state = np.random.RandomState(0)
        self._last_goal_ik_err = None  # residual of the last goal-pose IK solve

        # force GL context creation so the painter can upload textures later
        self.physics.render(height=16, width=16, camera_id=self.camera_id)

    # ---------------- background helpers ----------------

    def _sources(self, split):
        if split not in self._bg_sources:
            self._bg_sources[split] = build_background_sources(
                self.bg_source, split, self.davis_path, self.n_synthetic
            )
        return self._bg_sources[split]

    def n_background_videos(self, split=None):
        return len(self._sources(split or self.bg_split))

    def current_background_info(self):
        """What is actually on screen; recorded into the result jsonl."""
        if not self.bg_enabled:
            return {"enabled": False, "split": None, "video": None, "offset": None}
        srcs = self._sources(self.bg_split)
        if not srcs:
            return {"enabled": False, "split": self.bg_split, "video": None, "offset": None}
        src = srcs[self.bg_video_idx % len(srcs)]
        return {
            "enabled": True,
            "split": self.bg_split,
            "video": src.name,
            "offset": int(self.bg_frame_offset),
            "n_frames": len(src),
            "dynamic": bool(self.bg_dynamic),
        }

    def _ground_rgba(self):
        """The floor material's rgba row, as a writable view.

        Tries the material named 'grid' (dm_control's common/materials.xml) and
        falls back to whatever material the 'ground' geom actually uses, so this
        keeps working if the scene is renamed.
        """
        model = self.physics.model
        try:
            return self.physics.named.model.mat_rgba["grid"]
        except (KeyError, IndexError):
            pass
        mujoco = _lazy_dm_control()["mujoco"]
        for geom_name in ("ground", "floor"):
            try:
                gid = mujoco.mj_name2id(
                    model.ptr if hasattr(model, "ptr") else model,
                    mujoco.mjtObj.mjOBJ_GEOM,
                    geom_name,
                )
            except Exception:
                gid = -1
            if gid >= 0:
                mat_id = int(model.geom_matid[gid])
                if mat_id >= 0:
                    return model.mat_rgba[mat_id]
        raise RuntimeError(
            "Could not find the ground-plane material; background replacement "
            "cannot make the floor transparent."
        )

    def _set_ground_alpha(self, alpha):
        self._ground_rgba()[3] = alpha

    def _ensure_painter(self):
        if self._painter is None:
            self._painter = _SkyboxPainter(self.physics)
            buf = _tex_buffer(self.physics.model)
            a, s = self._painter.adr, self._painter.size
            self._default_skybox = np.array(buf[a : a + s], copy=True)
            self._default_ground_alpha = float(self._ground_rgba()[3])
        return self._painter

    def _restore_clean_background(self):
        if self._painter is None or self._default_skybox is None:
            return  # never painted => still the original skybox
        self._painter.restore(self._default_skybox)
        self._set_ground_alpha(self._default_ground_alpha)

    def _apply_background(self, frame_idx):
        if not self.bg_enabled:
            return
        painter = self._ensure_painter()
        srcs = self._sources(self.bg_split)
        if not srcs:
            return
        src = srcs[self.bg_video_idx % len(srcs)]
        painter.paint(src.get(self.bg_frame_offset + frame_idx, painter.face_hw))
        # make the floor see-through so the video skybox is visible from the
        # overhead camera (this is what distracting_control's ground_plane_alpha does)
        self._set_ground_alpha(self.ground_plane_alpha)

    # ---------------- state <-> physics ----------------

    def get_state(self):
        qpos = np.asarray(self.physics.data.qpos[:2], dtype=np.float64)
        qvel = np.asarray(self.physics.data.qvel[:2], dtype=np.float64)
        finger = np.asarray(
            self.physics.named.data.geom_xpos["finger", :2], dtype=np.float64
        )
        target = np.asarray(
            self.physics.named.model.geom_pos["target", :2], dtype=np.float64
        )
        return np.concatenate([qpos, qvel, finger, target]).astype(np.float32)

    def set_state(self, state):
        """state: at least [qpos(2), qvel(2)]; dims 6:8 set the target when present."""
        state = np.asarray(state, dtype=np.float64).ravel()
        if state.shape[0] >= STATE_DIM:
            self.physics.named.model.geom_pos["target", "x"] = state[6]
            self.physics.named.model.geom_pos["target", "y"] = state[7]
        self.physics.named.model.geom_size["target", 0] = self.target_size
        with self.physics.reset_context():
            self.physics.data.qpos[:2] = state[:2]
            self.physics.data.qvel[:2] = state[2:4]
        self.physics.forward()

    def _state_from_pose(self, qpos, target):
        st = np.zeros(STATE_DIM, dtype=np.float32)
        st[:2] = qpos
        st[2:4] = 0.0
        st[6:8] = target
        self.set_state(st)
        return self.get_state()  # fills in the resulting finger position

    def _finger_dist(self, qpos, target):
        return float(np.linalg.norm(self._state_from_pose(qpos, target)[4:6] - target))

    def solve_pose_for_target(self, target, rs, n_coarse=200, n_refine=120):
        """Numeric IK: a pose whose fingertip sits on `target`.

        Done by forward kinematics search rather than closed-form IK so it stays
        correct if the reacher model's link lengths ever change. Random coarse
        sampling followed by a shrinking local search; converges well inside the
        0.05 success radius.
        """
        best_q, best_d = None, np.inf
        for _ in range(n_coarse):
            q = np.array(
                [rs.uniform(-np.pi, np.pi), rs.uniform(-WRIST_LIMIT, WRIST_LIMIT)]
            )
            d = self._finger_dist(q, target)
            if d < best_d:
                best_q, best_d = q, d
        scale = 0.4
        for _ in range(n_refine):
            q = best_q + rs.normal(scale=scale, size=2)
            q[1] = np.clip(q[1], -WRIST_LIMIT, WRIST_LIMIT)
            d = self._finger_dist(q, target)
            if d < best_d:
                best_q, best_d = q, d
            else:
                scale *= 0.97
        return best_q, best_d

    def sample_random_init_goal_states(self, seed):
        """Returns (init_state, goal_state) sharing one target.

        The goal is a pose whose *fingertip is on the target*, so "reach the goal
        image" and "solve the task" are the same thing -- otherwise success
        (fingertip-on-target) and the planner's objective (match the goal image)
        would measure different things and the experiment would be unreadable.
        The target marker is identical in both images; only the arm pose differs.
        """
        rs = np.random.RandomState(seed)
        angle = rs.uniform(0, 2 * np.pi)
        radius = rs.uniform(*TARGET_RADIUS_RANGE)
        target = np.array([radius * np.sin(angle), radius * np.cos(angle)])

        goal_q, goal_err = self.solve_pose_for_target(target, rs)
        goal_state = self._state_from_pose(goal_q, target)

        # init pose: anywhere, as long as the task is not already solved
        init_state = None
        for _ in range(100):
            q = np.array(
                [rs.uniform(-np.pi, np.pi), rs.uniform(-WRIST_LIMIT, WRIST_LIMIT)]
            )
            st = self._state_from_pose(q, target)
            if np.linalg.norm(st[4:6] - target) > 3.0 * self.target_size:
                init_state = st
                break
        if init_state is None:  # pathological target; fall back to the last draw
            init_state = st

        self._last_goal_ik_err = goal_err
        return init_state, goal_state

    # ---------------- DINO-WM env interface ----------------

    def update_env(self, env_info):
        """Nothing to carry over from the dataset: the target rides in the state."""
        return None

    def eval_state(self, goal_state, cur_state):
        goal_state = np.asarray(goal_state).ravel()
        cur_state = np.asarray(cur_state).ravel()
        # the true task metric: is the fingertip on the target?
        finger_target_dist = float(np.linalg.norm(cur_state[4:6] - cur_state[6:8]))
        return {
            "success": bool(finger_target_dist <= self.target_size),
            "state_dist": float(np.linalg.norm(goal_state[:4] - cur_state[:4])),
            "finger_target_dist": finger_target_dist,
            "finger_goal_dist": float(
                np.linalg.norm(goal_state[4:6] - cur_state[4:6])
            ),
        }

    def _get_obs(self):
        visual = self.physics.render(
            height=self.img_size, width=self.img_size, camera_id=self.camera_id
        )
        state = self.get_state()
        return {"visual": visual, "proprio": state[:PROPRIO_DIM].copy()}

    def seed(self, seed=None):
        if seed is not None:
            self.random_state = np.random.RandomState(seed)
        return [seed]

    def set_init_state(self, init_state):
        self.reset_to_state = init_state

    def reset(self):
        self._dm_env.reset()  # resets dm_control's time/step counters
        self._step_count = 0
        self._frame_counter = 0
        if self.bg_enabled:
            self._apply_background(0)
        else:
            self._restore_clean_background()
        state = self.reset_to_state
        if state is None:
            state, _ = self.sample_random_init_goal_states(
                self.random_state.randint(0, 2**31 - 1)
            )
        self.set_state(state)
        return self._get_obs(), self.get_state()

    def prepare(self, seed, init_state):
        """Reset to a controlled state. obs['visual']: (H, W, C) uint8."""
        self.seed(seed)
        self.set_init_state(init_state)
        return self.reset()

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64).ravel(), -1.0, 1.0)
        for _ in range(self.action_repeat):
            self.physics.set_control(action)
            self.physics.step()
        self._step_count += 1
        if self.bg_enabled and self.bg_dynamic:
            self._frame_counter += 1
            self._apply_background(self._frame_counter)
        obs = self._get_obs()
        state = self.get_state()
        finger_target_dist = float(np.linalg.norm(state[4:6] - state[6:8]))
        info = {
            "state": state,
            "finger_target_dist": finger_target_dist,
            "pos_finger": state[4:6],
        }
        return obs, float(finger_target_dist <= self.target_size), False, info

    def step_multiple(self, actions):
        from utils import aggregate_dct

        obses, rewards, dones, infos = [], [], [], []
        for action in actions:
            o, r, d, info = self.step(action)
            obses.append(o)
            rewards.append(r)
            dones.append(d)
            infos.append(info)
        return (
            aggregate_dct(obses),
            np.stack(rewards),
            np.stack(dones),
            aggregate_dct(infos),
        )

    def rollout(self, seed, init_state, actions):
        """obses['visual']: (T+1, H, W, C); states: (T+1, state_dim)."""
        obs, state = self.prepare(seed, init_state)
        obses, rewards, dones, infos = self.step_multiple(actions)
        for k in obses.keys():
            obses[k] = np.vstack([np.expand_dims(obs[k], 0), obses[k]])
        states = np.vstack([np.expand_dims(state, 0), infos["state"]])
        return obses, np.stack(states)

    def close(self):
        try:
            self._dm_env.close()
        except Exception:
            pass


def _register():
    """Importing this module is enough for gym.make('distracting_reacher')."""
    try:
        from gym.envs.registration import register, registry
    except Exception:
        return
    try:
        known = getattr(registry, "env_specs", registry)
        if "distracting_reacher" in known:
            return
    except TypeError:
        pass
    try:
        register(
            id="distracting_reacher",
            entry_point="env.distracting_reacher:DistractingReacherWrapper",
            max_episode_steps=1000,
            reward_threshold=1.0,
        )
    except Exception:
        pass  # already registered elsewhere (e.g. env/__init__.py)


_register()
