"""Recompute every interval, p-value and split-count claim in the v5 text."""
import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent
DRAFT = ROOT / (sys.argv[1] if len(sys.argv) > 1 else "95_English_draft_sci_v5.md")
TEXT = DRAFT.read_text(encoding="utf-8")


def load(name):
    return json.loads((ROOT / "suite_output" / name).read_text(encoding="utf-8"))


SRC = {"EuroSAT_RGB": load("resnet18_main.json"), "DTD": load("resnet18_main.json"),
       "Mini": load("resnet18_mini.json")}
MULTI = {bb: load(f"{bb}_mini.json") for bb in ("resnet18", "resnet50", "vit_b_16")}
fails, n = [], 0


def series(rows, dom, method, ratio, ood=0.2):
    sel = [x for x in rows if x["domain"] == dom and x["method"] == method
           and x["ratio"] == ratio and x["ood"] == ood]
    sel = sorted(sel, key=lambda z: z["split"])
    return (np.array([x["accuracy"] for x in sel]),
            np.array([x["auroc"] if x["auroc"] is not None else np.nan for x in sel]))


def claim(label, rows, dom, a, b, ratio, want_diff, want_lo, want_hi, want_pos, ood=0.2):
    global n
    n += 1
    A, _ = series(rows, dom, a, ratio, ood)
    B, _ = series(rows, dom, b, ratio, ood)
    d = A - B
    lo = d.mean() - 1.96 * d.std(ddof=1) / np.sqrt(len(d))
    hi = d.mean() + 1.96 * d.std(ddof=1) / np.sqrt(len(d))
    pos = int((d > 0).sum())
    for got, want, what in ((d.mean(), want_diff, "diff"), (lo, want_lo, "lo"),
                            (hi, want_hi, "hi")):
        if abs(got - want) > 6e-4:
            fails.append(f"{label} {what}: text {want} vs recomputed {got:.4f}")
    if pos != want_pos:
        fails.append(f"{label} pos: text {want_pos} vs recomputed {pos}")


# --- Section 4.2, miniImageNet paragraph ---
claim("mini 0.3 clean-run", SRC["Mini"], "Mini", "bal_soft", "proto_deep", 0.3,
      0.1059, 0.1017, 0.1101, 10)
claim("mini 0.5", SRC["Mini"], "Mini", "bal_soft", "proto_deep", 0.5,
      0.1152, 0.1049, 0.1255, 10)
claim("mini clean", SRC["Mini"], "Mini", "bal_soft", "proto_deep", 0.0,
      0.0173, 0.0153, 0.0193, 10)

# --- AUROC endpoints on mini ---
n += 1
_, pu = series(SRC["Mini"], "Mini", "proto_deep", 0.3)
_, ou = series(SRC["Mini"], "Mini", "bal_soft", 0.3)
if abs(np.nanmean(pu) - 0.753) > 6e-4 or abs(np.nanmean(ou) - 0.837) > 6e-4:
    fails.append(f"mini AUROC endpoints: {np.nanmean(pu):.3f} -> {np.nanmean(ou):.3f}")

# --- all 18 cells: positive on 10/10 and p < 1e-9 ---
n += 1
worst = 0.0
for bb, rows in MULTI.items():
    for dom in ("Mini",):
        p, _ = series(rows, dom, "proto_deep", 0.3)
        for m in ("mstcr_ood", "bal_soft"):
            a, _ = series(rows, dom, m, 0.3)
            pv = stats.ttest_rel(a, p).pvalue
            worst = max(worst, pv)
            if (a - p <= 0).any():
                fails.append(f"18-cell {bb} {dom} {m}: not all positive")
for bb, f in (("resnet18", "resnet18_main.json"), ("resnet50", "resnet50_main.json"),
              ("vit_b_16", "vit_b_16_main.json")):
    rows = load(f)
    for dom in ("EuroSAT_RGB", "DTD"):
        p, _ = series(rows, dom, "proto_deep", 0.3)
        for m in ("mstcr_ood", "bal_soft"):
            a, _ = series(rows, dom, m, 0.3)
            pv = stats.ttest_rel(a, p).pvalue
            worst = max(worst, pv)
            if (a - p <= 0).any():
                fails.append(f"18-cell {bb} {dom} {m}: not all positive")
if worst >= 1e-9:
    fails.append(f"worst p over 18 cells is {worst:.2e}, text claims <1e-9")

# --- 4.3: the 84x84 re-extraction ---
n += 1
r84 = load("resnet18_mini84.json")
t84, _ = series(r84, "Mini", "tcr_deep", 0.3)
m84, _ = series(r84, "Mini", "mstcr", 0.3)
step84 = m84.mean() - t84.mean()
if abs(step84 - (-0.023)) > 1.1e-3:
    fails.append(f"84x84 multi-scale step: text -0.023 vs {step84:+.4f}")

# --- 4.6 spread claims ---
n += 1
tp = json.loads((ROOT / "tp_baseline_output" / "tp_mini.json").read_text(encoding="utf-8"))
lap = json.loads((ROOT / "lap_shot_output" / "lap_mini.json").read_text(encoding="utf-8"))


def tpv(dom, meth, ratio):
    s = [x["accuracy"] for x in tp if x["domain"] == dom and x["method"] == meth
         and x["ratio"] == ratio and x["ood"] == 0.2]
    return float(np.mean(s))


def lapv(ratio, tag="ls_k5_lam10.0"):
    s = [x["accuracy"] for x in lap if x["method"] == tag and x["ratio"] == ratio]
    return float(np.mean(s))


mini_lp30, mini_ls30 = tpv("Mini", "lp_cos_k5", 0.3), lapv(0.3)
mini_ours30 = series(SRC["Mini"], "Mini", "bal_soft", 0.3)[0].mean()
if abs((mini_ours30 - mini_lp30) - 0.1669) > 1.1e-3:
    fails.append(f"mini ours-lp at 0.3: {mini_ours30 - mini_lp30:+.4f} vs text 0.1669")
if abs((mini_ours30 - mini_ls30) - 0.1710) > 1.1e-3:
    fails.append(f"mini ours-ls at 0.3: {mini_ours30 - mini_ls30:+.4f} vs text 0.1710")

mini_lp50, mini_ls50 = tpv("Mini", "lp_cos_k5", 0.5), lapv(0.5)
mini_lp0, mini_ls0 = tpv("Mini", "lp_cos_k5", 0.0), lapv(0.0)
mini_o50 = series(SRC["Mini"], "Mini", "bal_soft", 0.5)[0].mean()
mini_o0 = series(SRC["Mini"], "Mini", "bal_soft", 0.0)[0].mean()
loss_graph = (mini_lp0 - mini_lp50, mini_ls0 - mini_ls50)
loss_ours = mini_o0 - mini_o50
if not (abs(loss_graph[0] - 0.3455) < 1.1e-3 and abs(loss_graph[1] - 0.3493) < 1.1e-3):
    fails.append(f"mini graph loss 0->0.5: {loss_graph} vs text 0.345/0.349")
if abs(loss_ours - 0.1910) > 1.1e-3:
    fails.append(f"mini ours loss: {loss_ours:.4f} vs text 0.191")

# --- 4.8 numbers ---
n += 1
imb = json.loads((ROOT / "imbalance_output" / "imbalance_resnet18_mini.json").read_text(encoding="utf-8"))


def imbv(meth, alpha):
    if alpha == float("inf"):
        s = [x["accuracy"] for x in imb if x["method"] == meth and np.isinf(x["alpha"])]
    else:
        s = [x["accuracy"] for x in imb if x["method"] == meth and abs(x["alpha"] - alpha) < 1e-9]
    return float(np.mean(s))


checks_48 = [
    ("fixed quota balanced vs proto", imbv("bal_fixed", float("inf")) - imbv("proto", float("inf")), 0.142),
    ("fixed quota severe cost", imbv("proto", 0.3) - imbv("bal_fixed", 0.3), 0.250),
    ("hard3 moderate gain", imbv("hard3", 1.0) - imbv("proto", 1.0), 0.059),
    ("hard3 balanced loss", imbv("bal_fixed", float("inf")) - imbv("hard3", float("inf")), 0.050),
    ("switch balanced loss", imbv("bal_fixed", float("inf")) - imbv("adapt", float("inf")), 0.036),
]
for lab, got, want in checks_48:
    if abs(got - want) > 1.1e-3:
        fails.append(f"4.8 {lab}: {got:+.4f} vs text {want}")

print(f"claims checked: {n + len(checks_48)}")
if fails:
    print(f"FAILURES {len(fails)}")
    for f in fails:
        print("  -", f)
    sys.exit(1)
print("ALL TEXT CLAIMS RECONCILE")
