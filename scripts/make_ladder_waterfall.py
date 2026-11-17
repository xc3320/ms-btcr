"""Alternative rendering of the component ladder as a waterfall.

Reads the same JSON as make_figures_v2.f4_ladder and draws the same six levels,
but the bar height encodes the increment each component contributes instead of
the absolute accuracy, which Table 4 already gives.

Every increment is a *paired* per-split difference: for split s,
d_s = level_i(s) - level_{i-1}(s).  The error bar is the 95% CI of that paired
difference (t, df = 9), not the CI of a level, so a step whose CI crosses zero
is visibly inconclusive.

Usage:  python make_ladder_waterfall.py   -> figures/figE2_ladder_wf.pdf/.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fig_style as S  # noqa: E402

ROOT = HERE.parent
SUITE = ROOT / "suite_output"
DOMAINS = ("EuroSAT_RGB", "DTD")
LADDER = ("proto_deep", "tcr_deep", "mstcr", "mstcr_ood", "bal_iter", "bal_soft")
SHORT = ("single\nscale", "+trans-\nductive", "+multi-\nscale",
         "+OOD\nweight", "+balance", "full\n(MS-BTCR)")
COL = {
    "proto_deep": "#7f7f7f",
    "tcr_deep": "#4c8cbf",
    "mstcr": "#1f4e79",
    "mstcr_ood": "#a61b29",
    "bal_iter": "#2e7d6f",
    "bal_soft": "#c58f2a",
}
T975_DF9 = 2.2622          # two-sided 95% Student t, 9 degrees of freedom

S.apply()
ROWS = json.loads((SUITE / "resnet18_main.json").read_text(encoding="utf-8"))


def per_split(dom, method):
    """Accuracy of one rung, one value per split, same episodes for every rung."""
    out = {}
    for r in ROWS:
        if (r["domain"] == dom and r["ood"] == 0.2 and r["ratio"] == 0.3
                and r["method"] == method and r["accuracy"] is not None):
            out[r["split"]] = r["accuracy"]
    return np.array([out[s] for s in sorted(out)])


def ci95(v):
    n = len(v)
    half = T975_DF9 * v.std(ddof=1) / np.sqrt(n)
    return v.mean() - half, v.mean() + half


LVL = {d: [per_split(d, m) for m in LADDER] for d in DOMAINS}
assert all(len(x) == 10 for d in DOMAINS for x in LVL[d]), "expected 10 splits"

fig, axes = plt.subplots(1, 2, figsize=(S.DOUBLE, 2.8), sharey=True)
MEANS = {d: [x.mean() for x in LVL[d]] for d in DOMAINS}
# one shared vertical scale: the argument compares step sizes across domains
FLOOR = min(min(ci95(x)[0] for x in LVL[d]) for d in DOMAINS) - 0.02
CEIL = max(max(ci95(x)[1] for x in LVL[d]) for d in DOMAINS) + 0.04

for ax, dom in zip(axes, DOMAINS):
    vals = MEANS[dom]
    x = np.arange(len(LADDER))

    # rung 1: the base level, drawn from the axis floor
    lo0, hi0 = ci95(LVL[dom][0])
    ax.bar(x[0], vals[0] - FLOOR, bottom=FLOOR, width=0.62,
           color=COL[LADDER[0]], edgecolor="#333333", lw=0.4)
    ax.vlines(x[0], lo0, hi0, color="#333333", lw=0.7)

    # rungs 2..n: floating bars whose height is the paired increment
    for i in range(1, len(LADDER)):
        d = LVL[dom][i] - LVL[dom][i - 1]
        step = d.mean()
        dlo, dhi = ci95(d)
        crosses_zero = dlo * dhi < 0

        if abs(step) < 5e-4:
            ax.hlines(vals[i], x[i] - 0.31, x[i] + 0.31,
                      color=COL[LADDER[i]], lw=2.2)
        else:
            ax.bar(x[i], step, bottom=vals[i - 1], width=0.62,
                   color=COL[LADDER[i]], edgecolor="#333333", lw=0.4,
                   alpha=0.92 if step >= 0 else 0.55)
        ax.hlines(vals[i - 1], x[i - 1] + 0.31, x[i] - 0.31,
                  color="#888888", lw=0.6, ls=(0, (2, 2)))

        # paired-difference CI, drawn on the step it belongs to
        top = max(vals[i], vals[i - 1])
        ax.vlines(x[i], vals[i - 1] + dlo, vals[i - 1] + dhi,
                  color="#333333", lw=0.7,
                  linestyles="dotted" if crosses_zero else "solid")

        sign = "+" if step >= 0 else "\u2212"
        ax.text(x[i], top + 0.004, f"{sign}{abs(step):.3f}",
                ha="center", va="bottom", fontsize=6.2,
                color="#999999" if crosses_zero else "#222222")

    ax.set_xticks(x)
    ax.set_xticklabels(SHORT, fontsize=6.4)
    ax.set_title(S.DOMAIN_LABEL[dom], color=S.DOMAIN_COLOR[dom])
    ax.grid(axis="y", lw=0.5)
    ax.set_axisbelow(True)
    ax.set_ylim(FLOOR, CEIL)
    ax.set_xlim(-0.6, len(LADDER) - 0.35)

axes[0].set_ylabel("Accuracy (noise 0.3, 20% OOD queries)")
fig.tight_layout()
S.save(fig, "figE2_ladder_wf")

# print the numbers so they can be checked against Table 4 without opening the png
for dom in DOMAINS:
    print(dom)
    for i, m in enumerate(LADDER):
        if i == 0:
            print(f"  {m:<11} level {MEANS[dom][0]:.4f}")
        else:
            d = LVL[dom][i] - LVL[dom][i - 1]
            lo, hi = ci95(d)
            print(f"  {m:<11} step {d.mean():+.4f}  CI [{lo:+.4f}, {hi:+.4f}]"
                  f"{'  crosses 0' if lo * hi < 0 else ''}")
