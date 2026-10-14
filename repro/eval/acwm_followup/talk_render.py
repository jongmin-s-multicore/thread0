"""InD/OoD 발표 자료: 4번 실험(디코더 확인)과, 같은 에피소드로 만든 1번·3번 영상.

4번 실험: 실제 프레임을 DINOv2 인코더 -> 디코더에 바로 통과시킨다 (predictor 를 거치지 않는다).
디코더는 InD 데이터로만 학습한 원래 모델의 것이다. 이 재구성에서 천·모래·물의 양과 쌓임 판정이 실제와 같으면
디코더는 OoD 의 크기와 양을 그릴 수 있다.

1단계: 원래 모델로 InD test, OoD test 의 모든 에피소드를 계산하고 양을 잰다.
  열 4개: Real / Encoder->Decoder / Model 25-step (25 프레임 전의 실제 프레임에서 5 모델 스텝) / Model rollout (첫 프레임에서 끝까지)
  -> <out>/<env>_summary.json, <out>/<env>_measures.npz
2단계: 그룹마다 대표 에피소드 3개로 영상과 정지 그림을 만든다. 영상에는 25-step 칸을 넣지 않는다 (2026-10-14 에 뺐다. 수치는 1단계에 그대로 있다).
  slide1  Real | Model rollout
  slide2  Real | InD only rollout | InD+OoD rollout   (Cloth Move 만)
  slide3  Real | 1x train rollout | 8x train rollout   (Pour Water 제외)
  slide4  Real | Encoder->Decoder | Model rollout
  --videos_only 는 1단계를 건너뛰고 <env>_summary.json 의 대표 에피소드로 영상만 다시 만든다. --slides slide2 처럼 슬라이드를 고를 수 있다.
  -> <out>/videos/<slide>_<env>_<set>_<group>_3eps.{mp4,gif} (에피소드 3개를 세로로), <slide>_<env>_<set>_<group>.{mp4,gif} (가운데 에피소드 하나)
     <out>/stills/<slide>_<env>_<set>_<group>.png (가운데 에피소드, 행 = 열 종류, 열 = 프레임 0 과 프레임 25 이후 4개)
Cloth Move 의 대표 에피소드는 2번(OoD 섞어 학습) 영상 results/acwm_phys/figs/cloth_h1f5_oodmix 와 같은 held-out 에피소드다.
Push Sand, Stack Cube 는 3번(8배 학습) 영상 results/acwm_phys/long/figs25 와 같은 에피소드다. Pour Water 는 rollout 의 상자 물 양 오차로
10 / 50 / 90 % 분위 에피소드를 고른다.

usage (cwd = $DINO_WM, env sourced):
  python <this> {cloth,sand,stack,water} <out_dir> [--max_eps N] [--no_video]
"""
import os, sys, json, time, argparse, subprocess
from pathlib import Path
import numpy as np
import torch
import hydra
import imageio
from omegaconf import OmegaConf

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "repro", "eval"))
from plan import load_model
from utils import seed
from datasets.acwm_phys_dset import load_acwm_phys_split
from datasets.acwm_pour_water_dset import load_acwm_pour_water_split
from datasets.img_transforms import default_transform
import eval_pour_water as EPW
from eval_acwm_phys import episode_groups
from eval_acwm_phys_quant import measure, MIN_PX
from render_acwm_phys import ASPECT, write_gif
import cv2
from trend_acwm_phys import is_stacked

WORK = Path(os.environ["DINO_WORK"])
CKPT = Path(os.environ["DINO_TRAIN"]) / "outputs"
AHEAD = 5  # model steps of the fixed-horizon prediction: 5 x frameskip 5 = 25 frames, the DINO-WM planning horizon
SPEC = {
    "cloth": dict(env="acwm_cloth_move", base=("train_cloth_h1f5", 15), more=("train_cloth_h1f5_long", 120), mix=("train_cloth_h1f5_oodmix", 15),
                  held_out=WORK / "runs/colab_acwm_long/ood/train_cloth_h1f5_oodmix_split.json",
                  hero={"ind_test/all": ["episode_00055.mp4", "episode_00074.mp4", "episode_00004.mp4"],
                        "ood_test/smaller": ["smaller_episode_00025.mp4", "smaller_episode_00011.mp4", "smaller_episode_00019.mp4"],
                        "ood_test/larger": ["larger_episode_00023.mp4", "larger_episode_00019.mp4", "larger_episode_00045.mp4"]}),
    "sand": dict(env="acwm_push_sand", base=("train_sand_h1f5", 18), more=("train_sand_h1f5_long", 144),
                 hero={"ind_test/all": ["episode_9.mp4", "episode_80.mp4", "episode_2.mp4"],
                       "ood_test/all": ["episode_66.mp4", "episode_11.mp4", "episode_37.mp4"]}),
    "stack": dict(env="acwm_stack_cube", base=("train_stack_h1f5", 25), more=("train_stack_h1f5_long", 200),
                  # the middle episode (single-episode videos) is a failed stack that the model draws as stacked
                  hero={"ind_test/fail": ["episode_52.mp4", "episode_53.mp4", "episode_57.mp4"],
                        "ood_test/fail": ["episode_61.mp4", "episode_63.mp4", "episode_119.mp4"]}),
    "water": dict(env="acwm_pour_water", base=("train_pw_h1f5", 6), hero=None),
}
KINDS = ("real", "recon", "pred", "rollout")
VIDEO_KINDS = ("real", "recon", "rollout")  # columns of the slide 4 and per-episode videos (no 25-step column)
LABEL = {"real": "Real", "recon": "Encoder->Decoder", "pred": "Model 25-step", "rollout": "Model rollout"}
FILL_FROM = 50  # Pour Water: frame index (x frameskip 5 = frame 250) from which the box fill is averaged, as in eval_pour_water.py


def show(frame, env, label=None, scale=1):
    """frame -> (h*scale, w*scale) with a label in the top-left corner; the font shrinks when the label is wider than the panel"""
    h = frame.shape[0] * scale
    w = int(round(h * ASPECT.get(env, 1.0) / 2)) * 2
    im = cv2.resize(np.ascontiguousarray(frame), (w, h), interpolation=cv2.INTER_CUBIC)
    if label:
        fs = 0.45 if scale == 1 else 0.6
        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
        if tw + 10 > w:
            fs *= (w - 10) / tw
            (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
        cv2.rectangle(im, (0, 0), (tw + 8, th + base + 6), (0, 0, 0), -1)
        cv2.putText(im, label, (4, th + 3), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 1, cv2.LINE_AA)
    return im


def hcat(cells, gap=4):
    """panels side by side with a white gap"""
    sep = np.full((cells[0].shape[0], gap, 3), 255, np.uint8)
    out = [cells[0]]
    for c in cells[1:]:
        out += [sep, c]
    return np.concatenate(out, 1)


def load(run, epoch, cfg, dev):
    path = CKPT / run / "checkpoints" / f"model_{epoch}.pth"
    assert path.exists(), path  # load_model builds an untrained model when the file is missing
    seed(cfg.training.seed)
    model = load_model(path, cfg, cfg.num_action_repeat, dev)
    model.eval()
    return model


def load_sets(key, cfg):
    """{set: (dataset, {group: member indices}, transform)}"""
    ds_cfg = cfg.env.dataset
    out = {}
    if key == "water":
        tf = default_transform(cfg.img_size)
        for sp in ("ind_test", "ood_test"):
            d = load_acwm_pour_water_split(tf, ds_cfg.data_path, sp, n_rollout=ds_cfg.n_rollout, exclude=ds_cfg.exclude, normalize_action=cfg.normalize_action)
            idx = np.arange(len(d))
            if sp == "ind_test":
                groups = {"all": idx}
            else:
                w0 = np.array([EPW.water_px_frame0(d, i) for i in idx])
                groups = {"low": idx[w0 < EPW.W0_SPLIT], "high": idx[w0 >= EPW.W0_SPLIT]}
            out[sp] = (d, groups, tf)
        return out
    ds_kw = dict(n_rollout=ds_cfg.n_rollout, exclude=list(ds_cfg.exclude), normalize_action=cfg.normalize_action,
                 action_dims=None if ds_cfg.action_dims is None else list(ds_cfg.action_dims), action_format=ds_cfg.action_format)
    tf = hydra.utils.instantiate(ds_cfg.transform)
    for sp in ("ind_test", "ood_test"):
        d = load_acwm_phys_split(tf, ds_cfg.data_path, sp, **ds_kw)
        groups, _ = episode_groups(cfg.env.name, sp, d, list(range(len(d))))
        out[sp] = (d, {g: np.asarray(m, int) for g, m in groups.items()}, tf)
    return out


def episode(key, d, tf, e, f):
    """real frames every f frames (k, 3, 224, 224) in [-1, 1] and the macro actions (1, k-1, a)"""
    if key == "water":
        ep = EPW.Episode(d, e, tf)
        k = 300 // f
        return ep.vis[::f][:k], ep.act[: f * (k - 1)].reshape(1, k - 1, -1)
    k = min(len(d.actions[e]) // f, (d.n_obs[e] - 1) // f) + 1
    vis = d.get_frames(e, range(0, f * (k - 1) + 1, f))[0]["visual"]
    return vis, d.actions[e][: f * (k - 1)].reshape(1, k - 1, -1)


@torch.no_grad()
def frames_of(model, vis, fa, H, dev, recon=True):
    """{real, recon, pred, rollout}: (k, h, w, 3) uint8. pred = frame t predicted from the real frame AHEAD steps earlier;
    the first H + AHEAD - 1 frames have no such start and are taken from the rollout (same rule as render_compare_acwm_phys.py)."""
    k = len(vis)
    zeros = lambda b, n: torch.zeros(b, n, 1, device=dev)
    z, _ = model.rollout({"visual": vis[None, :H].to(dev), "proprio": zeros(1, H)}, fa.to(dev))
    roll = model.decode_obs(z)[0]["visual"][0]
    nw = k - H - AHEAD + 1
    io = torch.arange(H)[None] + torch.arange(nw)[:, None]
    ia = torch.arange(H + AHEAD - 1)[None] + torch.arange(nw)[:, None]
    zk, _ = model.rollout({"visual": vis[io].to(dev), "proprio": zeros(nw, H)}, fa[0][ia].to(dev))
    pred = model.decode_obs({n: v[:, -1:] for n, v in zk.items()})[0]["visual"][:, 0]
    out = {"real": EPW.to_u8(vis), "pred": EPW.to_u8(torch.cat([roll[: H + AHEAD - 1], pred])), "rollout": EPW.to_u8(roll)}
    if recon:  # encoder -> decoder on the real frames, no predictor
        zr = model.encode_obs({"visual": vis[None].to(dev), "proprio": zeros(1, k)})
        out["recon"] = EPW.to_u8(model.decode_obs(zr)[0]["visual"][0])
    return out


def iou(a, b):
    u = (a | b).sum()
    return float((a & b).sum() / u) if u >= MIN_PX else np.nan


def measures(key, fr):
    """{quantity: {kind: (k,) array}}. cloth/sand: object pixels and mask IoU against Real; stack: stacked rule; water: box fill and all water pixels"""
    env = SPEC[key]["env"]
    out = {}
    if key in ("cloth", "sand"):
        masks = {}
        for kd, x in fr.items():
            res = [measure(env, im) for im in x]
            out.setdefault("px", {})[kd] = np.array([r[0][key][0] for r in res], float)
            masks[kd] = [r[1][key] for r in res]
        for kd in fr:
            if kd != "real":
                out.setdefault("iou", {})[kd] = np.array([iou(a, b) for a, b in zip(masks[kd], masks["real"])])
    elif key == "stack":
        for kd, x in fr.items():
            out.setdefault("stacked", {})[kd] = np.array([is_stacked(measure(env, im)[0]) for im in x], float)
    else:
        for kd, x in fr.items():
            out.setdefault("box", {})[kd] = EPW.band_count(x, x.shape[1])
            out.setdefault("px", {})[kd] = np.array([EPW.water_mask(im).sum() for im in x], float)
    return out


def amount_text(key, m, kd, k):
    if key in ("cloth", "sand"):
        return f"  {int(m['px'][kd][k]):,} px"
    if key == "water":
        return f"  box {int(m['box'][kd][k]):,} px"
    return "  stacked" if m["stacked"][kd][k] else ""


def summarize(key, M, sets, H, held_out):
    """group means over all episodes of the group (and over the held-out ones for Cloth Move OoD)"""
    t25 = H + AHEAD - 1  # first frame index of the 25-step column that is really 25 frames ahead
    res = {}
    for sp, (d, groups, _) in sets.items():
        for g, mem in groups.items():
            mem = [int(e) for e in mem if (sp, int(e)) in M]
            if not mem:
                continue
            subsets = {"all": mem}
            if held_out and sp == held_out["extra_split"] and g in held_out["held_out"]:
                ok = set(held_out["held_out"][g])
                subsets["held_out"] = [e for e in mem if d.video_names[e] in ok]
            r = {}
            for sub, es in subsets.items():
                s = {"n": len(es)}
                for q in M[(sp, es[0])]:
                    kinds = list(M[(sp, es[0])][q])
                    kmax = max(len(M[(sp, e)][q]["real" if "real" in kinds else kinds[0]]) for e in es)
                    pad = lambda v: np.concatenate([v, np.full(kmax - len(v), np.nan)])
                    s[q] = {kd: {"last": float(np.mean([M[(sp, e)][q][kd][-1] for e in es])),
                                 "from25": float(np.mean([np.nanmean(M[(sp, e)][q][kd][t25:]) for e in es])),
                                 "curve": np.nanmean(np.stack([pad(M[(sp, e)][q][kd]) for e in es]), 0).round(4).tolist()} for kd in kinds}
                    if key == "water" and q == "box":
                        for kd in kinds:
                            s[q][kd]["fill"] = float(np.mean([M[(sp, e)][q][kd][FILL_FROM:].mean() for e in es]))
                    if key == "stack":  # last frame stacked: rate, and agreement with the real last frame
                        for kd in kinds:
                            s[q][kd]["agree_last"] = float(np.mean([M[(sp, e)][q][kd][-1] == M[(sp, e)][q]["real"][-1] for e in es]))
                r[sub] = s
            res[f"{sp}/{g}"] = r
    return res


def pick_heroes(key, M, sets):
    if SPEC[key]["hero"] is not None:
        out = {}
        for grp, names in SPEC[key]["hero"].items():
            sp = grp.split("/")[0]
            d = sets[sp][0]
            idx = {d.video_names[i]: i for i in range(len(d))}
            out[grp] = [int(idx[n]) for n in names]
        return out
    out = {}  # Pour Water: 10 / 50 / 90 % quantiles of the rollout box-fill error
    for sp, (d, groups, _) in sets.items():
        for g, mem in groups.items():
            mem = [int(e) for e in mem if (sp, int(e)) in M]
            if len(mem) < 3:
                continue
            err = [abs(M[(sp, e)]["box"]["rollout"][FILL_FROM:].mean() - M[(sp, e)]["box"]["real"][FILL_FROM:].mean()) for e in mem]
            order = [mem[i] for i in np.argsort(err, kind="stable")]
            out[f"{sp}/{g}"] = [order[int(round(q * (len(order) - 1)))] for q in (0.1, 0.5, 0.9)]
    return out


def render(key, spec, cfg, sets, heroes, model, out, H, f, env, fps, dev, slides=("slide1", "slide2", "slide3", "slide4")):
    """videos and stills of the chosen episodes (step 2)"""
    more = load(*spec["more"], cfg, dev) if "more" in spec and "slide3" in slides else None
    mix = load(*spec["mix"], cfg, dev) if "mix" in spec and "slide2" in slides else None
    for grp, es in heroes.items():
        sp, g = grp.split("/")
        d, _, tf = sets[sp]
        rows = []
        for e in es:
            vis, fa = episode(key, d, tf, e, f)
            fr = frames_of(model, vis, fa, H, dev)
            m = measures(key, fr)
            cell = {"slide1": [(LABEL[k], fr[k], m, k) for k in ("real", "rollout")],
                    "slide4": [(LABEL[k], fr[k], m, k) for k in VIDEO_KINDS]}
            for slide, other, (a, b) in (("slide2", mix, ("InD only", "InD+OoD")), ("slide3", more, ("1x train", "8x train"))):
                if other is None:
                    continue
                fo = frames_of(other, vis, fa, H, dev, recon=False)
                mo = measures(key, fo)
                cell[slide] = [("Real", fr["real"], m, "real"), (f"{a} rollout", fr["rollout"], m, "rollout"), (f"{b} rollout", fo["rollout"], mo, "rollout")]
            rows.append((d.video_names[e], cell))
        for slide in slides:
            if slide not in rows[0][1]:
                continue
            stem = f"{slide}_{key}_{sp}_{g}"

            def tile(cell, k, scale):
                return hcat([show(x[k], env, (f"{lab}  frame {k * f}" if kd == "real" and lab == "Real" else lab) + amount_text(key, m, kd, k), scale)
                             for lab, x, m, kd in cell])

            def clip(frames):
                return [frames[0]] * 2 + frames + [frames[-1]] * (2 * fps)  # hold the first and the last frame

            n_fr = min(len(c[slide][0][1]) for _, c in rows)
            vcat = lambda ims: np.concatenate(sum([[im, np.full((4, im.shape[1], 3), 255, np.uint8)] for im in ims], [])[:-1], 0)
            three = [vcat([tile(c[slide], k, 2) for _, c in rows]) for k in range(n_fr)]
            imageio.mimsave(out / "videos" / f"{stem}_3eps.mp4", clip(three), fps=fps, macro_block_size=2)
            write_gif(clip([vcat([tile(c[slide], k, 1) for _, c in rows]) for k in range(n_fr)]), out / "videos" / f"{stem}_3eps.gif", fps)
            mid = rows[1][1][slide]
            k_mid = len(mid[0][1])
            imageio.mimsave(out / "videos" / f"{stem}.mp4", clip([tile(mid, k, 2) for k in range(k_mid)]), fps=fps, macro_block_size=2)
            write_gif(clip([tile(mid, k, 1) for k in range(k_mid)]), out / "videos" / f"{stem}.gif", fps)
            ks = sorted({0, *np.linspace(H + AHEAD - 1, k_mid - 1, 4).round().astype(int).tolist()})  # frame 0 and four frames from frame 25 on
            grid = np.concatenate([hcat([show(x[k], env, (f"{lab}  frame {k * f}" if c == 0 else f"frame {k * f}") + amount_text(key, m, kd, k), 2)
                                         for c, k in enumerate(ks)]) for lab, x, m, kd in mid], 0)
            imageio.imwrite(out / "stills" / f"{stem}.png", grid)
        print(f"[{key}] rendered {grp}: {[n for n, _ in rows]}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("key", choices=list(SPEC)); ap.add_argument("out_dir")
    ap.add_argument("--max_eps", type=int, default=0, help="per set, for a quick test")
    ap.add_argument("--no_video", action="store_true")
    ap.add_argument("--videos_only", action="store_true", help="skip step 1, take the episodes from <out>/<env>_summary.json")
    ap.add_argument("--slides", default="slide1,slide2,slide3,slide4", help="which slides to render")
    ap.add_argument("--fps", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    t_start = time.time()
    key, spec, dev = args.key, SPEC[args.key], args.device
    out = Path(args.out_dir); (out / "videos").mkdir(parents=True, exist_ok=True); (out / "stills").mkdir(exist_ok=True)
    run, epoch = spec["base"]
    cfg = OmegaConf.load(CKPT / run / "hydra.yaml")
    H, f = cfg.num_hist, cfg.frameskip
    env = cfg.env.name
    fps = args.fps or (6 if key == "water" else 3)
    sets = load_sets(key, cfg)
    held_out = json.load(open(spec["held_out"])) if "held_out" in spec else None

    model = load(run, epoch, cfg, dev)
    if args.videos_only:
        info = json.load(open(out / f"{key}_summary.json"))
        heroes = {}
        for grp, ns in info["heroes"].items():
            d = sets[grp.split("/")[0]][0]
            heroes[grp] = [int(list(d.video_names).index(n)) for n in ns]
        render(key, spec, cfg, sets, heroes, model, out, H, f, env, fps, dev, tuple(args.slides.split(",")))
        return

    # ---- 1: every episode with the original (InD) model
    M, names = {}, {}
    for sp, (d, groups, tf) in sets.items():
        eps = sorted({int(i) for m in groups.values() for i in m})
        if args.max_eps:
            eps = eps[: args.max_eps]
        for n_, e in enumerate(eps):
            vis, fa = episode(key, d, tf, e, f)
            M[(sp, e)] = measures(key, frames_of(model, vis, fa, H, dev))
            names[(sp, e)] = d.video_names[e]
            if n_ % 25 == 0:
                print(f"[{key} {sp}] {n_}/{len(eps)} {time.time() - t_start:.0f}s", flush=True)
    summary = summarize(key, M, sets, H, held_out)
    heroes = pick_heroes(key, M, sets)
    arrays = {}
    for (sp, e), m in M.items():
        for q, v in m.items():
            for kd, a in v.items():
                arrays.setdefault(f"{sp}/{q}/{kd}", []).append(a)
    kmax = max(len(a) for v in arrays.values() for a in v)
    np.savez_compressed(out / f"{key}_measures.npz", **{k: np.stack([np.concatenate([a, np.full(kmax - len(a), np.nan)]) for a in v]) for k, v in arrays.items()},
                        **{f"{sp}/names": np.array([names[(s, e)] for (s, e) in M if s == sp]) for sp in sets})
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True).stdout.strip())
    info = {"env": env, "model": f"{run} epoch {epoch}", "more_model": f"{spec['more'][0]} epoch {spec['more'][1]}" if "more" in spec else None,
            "frameskip": f, "num_hist": H, "ahead_model_steps": AHEAD, "columns": {k: LABEL[k] for k in KINDS},
            "code": f"thread0 {commit}{' (uncommitted changes in tracked files)' if dirty else ''}",
            "max_eps": args.max_eps, "heroes": {g: [sets[g.split('/')[0]][0].video_names[e] for e in es] for g, es in heroes.items()},
            "groups": summary}
    json.dump(info, open(out / f"{key}_summary.json", "w"), indent=1)
    print(f"[{key}] measured {len(M)} episodes in {time.time() - t_start:.0f}s", flush=True)
    if args.no_video:
        return

    render(key, spec, cfg, sets, heroes, model, out, H, f, env, fps, dev)
    info["elapsed_s"] = round(time.time() - t_start)
    json.dump(info, open(out / f"{key}_summary.json", "w"), indent=1)
    print(f"[{key}] done in {info['elapsed_s']}s", flush=True)


if __name__ == "__main__":
    main()
