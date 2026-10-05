"""[repro] Outline of the PushT goal block pose, drawn on saved plots/videos only.

The env always renders the fixed default target (LightGreen T at (256, 256, pi/4)) and the world
model is trained on images that contain it, so observations are left untouched. Planning goals
(state_g) are arbitrary block poses; this module draws them as an outline for human inspection.
"""
import numpy as np
import cv2
import torch
import shapely.geometry as sg
import shapely.ops

from env.pusht.pusht_env import PushTEnv

WINDOW_SIZE = 512  # PushTEnv.window_size: physics/canvas coordinates
_local_polys = {}


def block_local_polygons(shape="T"):
    """Body-local outer outline(s) of the block, taken from the env so they match the rendered block.

    The block is made of several convex pymunk polygons; their union is returned so the outline has
    no internal edges.
    """
    if shape not in _local_polys:
        env = PushTEnv(shape=shape)
        env._setup()
        union = shapely.ops.unary_union(
            [sg.Polygon([tuple(v) for v in s.get_vertices()]) for s in env.block.shapes]
        )
        geoms = union.geoms if hasattr(union, "geoms") else [union]
        _local_polys[shape] = [
            np.array(g.exterior.coords[:-1], dtype=np.float64) for g in geoms
        ]
    return _local_polys[shape]


def block_outline_px(block_pose, img_size, shape="T"):
    """Block polygons in pixel coordinates of an img_size x img_size render.

    block_pose: (x, y, angle) in canvas coordinates, i.e. state[2:5].
    The block is drawn at position + R(angle) @ v (body origin, as in PushTEnv._render_frame),
    then the 512 canvas is resized with cv2.INTER_LINEAR, which maps pixel x to (x + 0.5) * s - 0.5.
    """
    x, y, angle = (float(v) for v in block_pose[:3])
    c, s = np.cos(angle), np.sin(angle)
    rot = np.array([[c, -s], [s, c]])
    scale = img_size / WINDOW_SIZE
    return [
        (verts @ rot.T + (x, y) + 0.5) * scale - 0.5
        for verts in block_local_polygons(shape)
    ]


def draw_block_outline(img, block_pose, color, thickness=1, shape="T", upsample=1):
    """Draw the outline of block_pose on img (H, W, C) in place and return it.

    img may be uint8 or float (color must then be in the same value range). If img is an
    upsampled render (H = upsample * render size), pass upsample so the outline is scaled to match.
    """
    size = img.shape[0] // upsample
    shift = 4  # sub-pixel vertex precision for cv2
    for poly in block_outline_px(block_pose, size, shape):
        pts = (poly + 0.5) * upsample - 0.5
        pts = np.round(pts * (1 << shift)).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [pts], True, color, thickness, cv2.LINE_AA, shift)
    return img


def draw_goal_outlines(visuals, goal_states, shapes=None, color=(1.0, -1.0, -1.0)):
    """Outline each sample's goal block pose on all of its frames (for PlanEvaluator plots).

    visuals: (B, T, C, H, W) float tensor in the evaluator's [-1, 1] image space (C = RGB).
    goal_states: (B, state_dim) PushT states; the block pose is state[2:5].
    shapes: per-sample block shape names (PushTEnv.shape); "T" if None.
    Frames that PlanEvaluator._mask_traj zeroed (after an MPC success) are left blank.
    Returns a new CPU tensor; the input is not modified.
    """
    imgs = visuals.detach().cpu().permute(0, 1, 3, 4, 2).numpy().copy()  # B T H W C
    for b in range(imgs.shape[0]):
        shape = shapes[b] if shapes is not None else "T"
        for img in imgs[b]:
            if img.any():
                draw_block_outline(img, goal_states[b][2:5], color, thickness=1, shape=shape)
    return torch.from_numpy(imgs).permute(0, 1, 4, 2, 3)
