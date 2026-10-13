"""루트 README 의 ACWM-Phys 결과 표 두 개를 결과 JSON 에서 만든다 (Pour Water 는 num_hist 1 모델).
읽는 파일: <results>/pour_water/h1f5_ep6.json, <results>/acwm_phys/{cloth,sand,stack}_h1f5.json 과 _quant.json.

usage: python repro/eval/tables_acwm_phys.py $DINO_WORK/results"""
import sys, json
R = sys.argv[1]
ci = lambda v: f"{v[0]:.2f} [{v[1]:.2f}, {v[2]:.2f}]"
rows_a, rows_b = [], []
# ---- Pour Water (num_hist 1): tables aggregate window frames >= 3
pw = json.load(open(f"{R}/pour_water/h1f5_ep6.json"))
for st, g, label in (("ind_test", "all", "InD test"), ("ood_test", "low", "OoD low volume"), ("ood_test", "high", "OoD high volume")):
    s = pw["sets"][st]; m1, m2, m5 = s["m1"][g], s["m2"][g], s["m5"][g]
    rows_a.append(("Pour Water", label, s["group_sizes"][g], m1["latent_mse"]["model"][0] / m1["latent_mse"]["copy_latent"][0],
                   m2["all"]["model"]["lpips"][0], m2["all"]["copy_last"]["lpips"][0], m2["all"]["recon_ceiling"]["lpips"][0], m2["skill_vs_recon_copy"]["lpips"],
                   m5["all"]["shuffled"]["lpips"][0] / m5["all"]["true"]["lpips"][0], m5["late"]["shuffled"]["lpips"][0] / m5["late"]["true"]["lpips"][0]))
    f = s["m3_fill"][g]
    rows_b.append(("Pour Water", label, "water pixels in the box, frames ≥ 250", f["gt_px"][0], f["model_px"][0], f["recon_px"][0]))
for tag, env, obj, groups in (("cloth_h1f5", "Cloth Move", "cloth", (("ind_test", "all", "InD test"), ("ood_test", "smaller", "OoD smaller cloth"), ("ood_test", "larger", "OoD larger cloth"))),
                              ("sand_h1f5", "Push Sand", "sand", (("ind_test", "all", "InD test"), ("ood_test", "all", "OoD 2× particles"))),
                              ("stack_h1f5", "Stack Cube", None, (("ind_test", "all", "InD test"), ("ood_test", "all", "OoD start position")))):
    r = json.load(open(f"{R}/acwm_phys/{tag}.json")); q = json.load(open(f"{R}/acwm_phys/{tag}_quant.json"))
    for st, g, label in groups:
        s = r["sets"][st]; m1, m2, m5 = s["m1"][g], s["m2"][g], s["m5"][g]
        rows_a.append((env, label, s["group_sizes"][g], m1["latent_mse"]["model"][0] / m1["latent_mse"]["copy_latent"][0],
                       m2["model"]["lpips"][0], m2["copy_last"]["lpips"][0], m2["recon_ceiling"]["lpips"][0], m2["skill_vs_recon_copy"]["lpips"],
                       m5["shuffled"]["lpips"][0] / m5["true"]["lpips"][0], None))
        if obj:
            d = q["sets"][st]["groups"][g][obj]
            rows_b.append((env, label, f"{obj} pixels, last frame", d["gt_px_last"][0], d["model"]["px_last"][0], d["recon"]["px_last"][0]))
    if obj is None:
        for st, lab in (("ind_test", "InD test"), ("ood_test", "OoD")):
            for g in ("success", "fail"):
                d = q["sets"][st]["groups"][g]["stacked_last_frame"]; n = q["sets"][st]["group_sizes"][g]
                rows_b.append((env, f"{lab}, {'successful' if g == 'success' else 'failed'} stacks ({n})", "last frame shows the cube stacked", d["gt"]["stacked_rate"], d["model"]["stacked_rate"], d["recon"]["stacked_rate"]))
print("| Environment | Test set | Episodes | One-step latent error, model ÷ copy | Rollout LPIPS: model / copy-last / decoder ceiling | Skill [95 % CI] | LPIPS with another episode's actions ÷ true actions |")
print("|---|---|---|---|---|---|---|")
for env, label, n, lr, a, b, c, sk, sh, sh_late in rows_a:
    print(f"| {env} | {label} | {n} | {lr:.2f} | {a:.3f} / {b:.3f} / {c:.3f} | {ci(sk)} | {sh:.1f}×" + (f" ({sh_late:.1f}× for windows starting at frame ≥ 200)" if sh_late else "") + " |")
print()
print("| Environment | Test set | Quantity | Real | Model | Decoder on real frames |")
print("|---|---|---|---|---|---|")
for env, label, what, g, m, d in rows_b:
    f = (lambda x: f"{100 * x:.0f} %") if "stacked" in what else (lambda x: f"{x:,.0f}")
    print(f"| {env} | {label} | {what} | {f(g)} | {f(m)} | {f(d)} |")
