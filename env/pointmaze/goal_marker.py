"""[repro] Marker for the PointMaze goal position, drawn on saved plots/videos only.

The env renders only the agent (green dot); the target site is parked off-screen (with_target=False),
so the planning goal (state_g[:2]) is visible only in the separate goal image. This module maps a
state's (x, y) to pixels of the env render and draws a red ring of the success radius around it, with
four short ticks pointing inward at the goal.
Observations are left untouched.
"""
import numpy as np
import cv2
import torch

# Camera of MazeEnv renders (U_MAZE), fixed after MazeEnv.prepare_for_render:
# mujoco-py's default free camera (MjRenderContext._init_camera) looks at the median geom position
# when the render context is created. The first render happens in prepare_for_render, after it moves
# the particle to world (2.29, 3.17); the median is then (3, 3, 0) = maze centre. (A context created
# with the particle at world y < 3 would look elsewhere; nothing in the repo renders before that.)
# The distance is model.stat.extent and prepare_for_render points the camera straight down
# (elevation -90, azimuth 90). fovy is MuJoCo's default 45 degrees. Looking straight down, the z = 0
# plane maps to the image by a scale + shift. mujoco-py returns the GL framebuffer bottom row first and
# MazeEnv._render_frame does not flip it, so image x grows with world x and image y with world y.
RENDER_SIZE = 224  # MazeEnv._render_frame: sim.render(224, 224)
LOOKAT_XY = (3.0, 3.0)
CAM_DISTANCE = 9.734846922834954  # model.stat.extent of the U_MAZE model
FOVY_DEG = 45.0
PARTICLE_ORIGIN = (1.2, 1.2)  # particle body pos: world = state[:2] + origin (slide joints, qpos0 = 0)
SUCCESS_RADIUS = 0.5  # PointMazeWrapper.eval_state: success if |state_g[:2] - state[:2]| < 0.5
PX_PER_UNIT = (RENDER_SIZE / 2) / np.tan(np.deg2rad(FOVY_DEG) / 2) / CAM_DISTANCE  # 27.78 at z = 0


def state_to_px(xy, img_size=RENDER_SIZE):
    """Pixel coordinates (x, y; pixel centres at integers) of state (x, y) in an img_size render.

    A render resized from RENDER_SIZE (cv2/torchvision bilinear) maps pixel u to (u + 0.5) * s - 0.5.
    Checked against the agent's centroid in static renders (set_state) over the open maze cells:
    max error 0.26 px at 224. Frames from env.step lag the returned state by one 0.01 s physics
    substep (MazeEnv renders right after mj_step), i.e. up to ~2 px (0.07 units) at full speed.
    """
    s = img_size / RENDER_SIZE
    c = (RENDER_SIZE - 1) / 2
    px = c + PX_PER_UNIT * (float(xy[0]) + PARTICLE_ORIGIN[0] - LOOKAT_XY[0])
    py = c + PX_PER_UNIT * (float(xy[1]) + PARTICLE_ORIGIN[1] - LOOKAT_XY[1])
    return (px + 0.5) * s - 0.5, (py + 0.5) * s - 0.5


def goal_marker_alpha(img_size, goal_xy, thickness=1, upsample=1):
    """Anti-aliased coverage mask (H, W) in [0, 1] of the goal marker for an img_size render
    (upsampled by `upsample`): a ring of the success radius (success = the agent's state ends inside it)
    and four short ticks pointing inward from it, which leave the agent visible when it reaches the goal.
    """
    size = img_size * upsample
    mask = np.zeros((size, size), np.uint8)  # cv2 anti-aliases only on 8-bit images
    gx, gy = state_to_px(goal_xy, img_size)
    gx, gy = (gx + 0.5) * upsample - 0.5, (gy + 0.5) * upsample - 0.5
    unit = PX_PER_UNIT * size / RENDER_SIZE  # pixels per world unit
    shift = 4  # sub-pixel precision for cv2
    fx = lambda v: int(round(v * (1 << shift)))
    r_ring, r_tick = SUCCESS_RADIUS * unit, 0.32 * unit  # ticks end outside the agent (site radius 0.2)
    cv2.circle(mask, (fx(gx), fx(gy)), fx(r_ring), 255, thickness, cv2.LINE_AA, shift)
    for dx, dy in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
        cv2.line(mask, (fx(gx + dx * r_ring), fx(gy + dy * r_ring)), (fx(gx + dx * r_tick), fx(gy + dy * r_tick)),
                 255, thickness, cv2.LINE_AA, shift)
    return mask.astype(np.float32) / 255


def blend_marker(img, alpha, color):
    """Paint color over img (H, W, C) with coverage alpha (H, W), in place; uint8 or float img."""
    a = alpha[..., None]
    out = img.astype(np.float32) * (1 - a) + np.asarray(color, np.float32) * a
    img[...] = np.clip(np.round(out), 0, 255) if img.dtype == np.uint8 else out
    return img


def draw_goal_marker(img, goal_xy, color, thickness=1, upsample=1):
    """Draw the goal marker on img (H, W, C) in place and return it.

    img may be uint8 or float (color must then be in the same value range). If img is an
    upsampled render (H = upsample * render size), pass upsample so the marker is scaled to match.
    """
    alpha = goal_marker_alpha(img.shape[0] // upsample, goal_xy, thickness, upsample)
    return blend_marker(img, alpha, color)


def draw_goal_markers(visuals, goal_states, color=(1.0, -1.0, -1.0), thickness=1):
    """Mark each sample's goal position on its frames (for PlanEvaluator plots).

    visuals: (B, T, C, H, W) float tensor in the evaluator's [-1, 1] image space (C = RGB, H = W).
    goal_states: (B, state_dim) PointMaze states; the position is state[:2].
    Frames that PlanEvaluator._mask_traj zeroed (after an MPC success) are left blank.
    Returns a new CPU tensor; the input is not modified.
    """
    imgs = visuals.detach().cpu().permute(0, 1, 3, 4, 2).numpy().copy()  # B T H W C
    assert imgs.shape[2] == imgs.shape[3], f"square frames expected, got {imgs.shape[2:4]}"
    color = np.asarray(color, np.float32)
    for b in range(imgs.shape[0]):
        alpha = goal_marker_alpha(imgs.shape[2], goal_states[b][:2], thickness)
        ys, xs = np.nonzero(alpha)
        if len(ys) == 0:  # goal outside the image
            continue
        win = (slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1))
        a = alpha[win][None, ..., None]
        shown = imgs[b].reshape(imgs.shape[1], -1).any(1)  # not zeroed by the mask
        patch = imgs[b, :, win[0], win[1]][shown]  # copies only the window
        imgs[b, shown, win[0], win[1]] = patch * (1 - a) + color * a
    return torch.from_numpy(imgs).permute(0, 1, 4, 2, 3)
