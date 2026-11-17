"""Figures for the balanced-transduction manuscript.

Every panel reads a JSON written by an experiment script. No number is typed by
hand, nothing is smoothed, clipped or back-filled. Where a curve is genuinely
absent at some x it is left out and the caption says so.
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
RATIOS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)
DOMAINS = ("EuroSAT_RGB", "DTD")
RATIO_TICKS = RATIOS

S.apply()

STYLE = {
    "proto_deep": dict(color="#7f7f7f", ls="-", marker="o", label="Single-scale prototype"),
    "tcr_deep": dict(color="#4c8cbf", ls="--", marker="s", label="Single-scale transductive"),
    "mstcr": dict(color="#1f4e79", ls="-.", marker="^", label="Multi-scale transductive"),
    "mstcr_ood": dict(color="#a61b29", ls=":", marker="D", label="Multi-scale + OOD weight"),
    "bal_iter": dict(color="#2e7d6f", ls="--", marker="v", label="Balanced (no OOD term)"),
    "bal_soft": dict(color="#c58f2a", ls="-", marker="P", label="MS-BTCR (full)"),
    "obtcr": dict(color="#6a4c93", ls=":", marker="*", label="Hard gate, then balance"),
}
LADDER = ("proto_deep", "tcr_deep", "mstcr", "mstcr_ood", "bal_iter", "bal_soft")


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


SUITE_ROWS = []
for _bk in ("resnet18", "resnet50", "vit_b_16"):
    SUITE_ROWS += load(SUITE / f"{_bk}_main.json")
STATS = load(SUITE / "stats_suite.json")


def series(rows, dom, method, metric, bk="resnet18", ood=0.2, ratios=RATIOS):
    out = []
    for r in ratios:
        v = [x[metric] for x in rows if x["backbone"] == bk and x["domain"] == dom
             and x["ood"] == ood and x["ratio"] == r and x["method"] == method
             and x.get(metric) is not None]
        out.append(np.mean(v) if v else np.nan)
    return np.array(out)


def band(dom, method, metric, bk="resnet18", ood=0.2, ratios=RATIOS):
    lo, hi = [], []
    for r in ratios:
        c = STATS["cells"].get(f"{bk}|{dom}|ood{ood}|r{r}|{method}|{metric}")
        lo.append(c["ci95_low"] if c else np.nan)
        hi.append(c["ci95_high"] if c else np.nan)
    return np.array(lo), np.array(hi)


# ------------------------------------------------------------ figure 1
def f1_architecture():
    fig, ax = plt.subplots(figsize=(S.DOUBLE, 2.7))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 42)
    ax.axis("off")

    def box(x, y, w, h, text, fc="#eef2f7", fs=6.4, bold=False):
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=fc,
                                   edgecolor="#333333", lw=0.7))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, fontweight="bold" if bold else "normal",
                color="#222222", linespacing=1.35)

    def arrow(x1, y1, x2, y2, dashed=False, color="#555555"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", lw=0.7, color=color,
                                    ls=(0, (3, 2)) if dashed else "-",
                                    shrinkA=0, shrinkB=0))

    # main chain, left to right, one row
    yc = 26.0
    box(1, yc - 6, 13, 12, "frozen\nbackbone\n(one pass)", "#f2f2f2", bold=True)
    for dy, t in ((-9.5, "view 1"), (0.0, "view 2"), (9.5, "view 3")):
        box(18, yc + dy - 3.2, 9.5, 6.4, t, "#dbe5f1")
        arrow(14.2, yc, 17.8, yc + dy)
    box(31, yc - 3.4, 11, 6.8, "row unit\nnorm", "#fdf0e6", bold=True)
    for dy in (-9.5, 0.0, 9.5):
        arrow(27.7, yc + dy, 30.8, yc)
    box(45.5, yc - 3.4, 11, 6.8, "concat", "#fdf0e6", bold=True)
    arrow(42.2, yc, 45.3, yc)
    box(60, yc - 4, 14, 8, "balanced\nassignment", "#dbe5f1", bold=True)
    arrow(56.7, yc, 59.8, yc)
    box(77.5, yc - 4, 12, 8, "centre\nupdate", "#dbe5f1", bold=True)
    arrow(74.2, yc, 77.3, yc)

    # inputs and outputs
    box(60, yc + 9, 14, 6.4, "support + query\nrows", "#e8f0e8", 6.2)
    arrow(67, yc + 8.8, 67, yc + 4.2)
    box(92.5, yc + 1.5, 7, 6, "class\npred.", "#f2f2f2", 6.2)
    box(92.5, yc - 9.5, 7, 6, "OOD\nscore", "#f2f2f2", 6.2)
    arrow(89.7, yc + 1.5, 92.3, yc + 4.2)
    arrow(89.7, yc - 1.5, 92.3, yc - 6.3)

    # weighting branch: distances leave the centre update, weights come back in
    box(70.5, yc - 17.5, 13, 7, "continuous\nOOD weight", "#fdf0e6", 6.2, True)
    arrow(83.0, yc - 4.2, 83.0, yc - 10.3, dashed=True)
    arrow(75.5, yc - 10.3, 75.5, yc - 4.2)
    ax.text(83.8, yc - 7.2, "distances", fontsize=6.0, color="#555555",
            va="center", ha="left")
    ax.text(74.7, yc - 7.2, "weights", fontsize=6.0, color="#555555",
            va="center", ha="right")
    ax.text(77.0, yc - 20.0, "threshold median + k\u00b7MAD", fontsize=6.0,
            color="#555555", ha="center")

    ax.text(46, 2.0, "5 re-estimation rounds; no support row is discarded "
                     "and no label is corrected",
            ha="center", fontsize=6.8, style="italic", color="#a61b29")
    S.save(fig, "figE1_architecture")


# ------------------------------------------------------------ figure 2/3
def f2_main(metric, fname, ylab, arms=("proto_deep", "tcr_deep", "mstcr",
                                       "mstcr_ood", "bal_soft")):
    fig, axes = plt.subplots(1, 2, figsize=(S.DOUBLE, 2.55), sharey=True)
    for ax, dom in zip(axes, DOMAINS):
        for m in arms:
            y = series(SUITE_ROWS, dom, m, metric)
            lo, hi = band(dom, m, metric)
            st = STYLE[m]
            ax.plot(RATIO_TICKS, y, color=st["color"], ls=st["ls"],
                    marker=st["marker"], ms=3.2, lw=1.1, label=st["label"])
            ax.fill_between(RATIO_TICKS, lo, hi, color=st["color"], alpha=0.12, lw=0)
        ax.set_title(S.DOMAIN_LABEL[dom], color=S.DOMAIN_COLOR[dom])
        ax.set_xlabel("Support-set label-noise rate")
        ax.set_xticks(RATIO_TICKS)
        ax.grid(axis="y", lw=0.5)
        ax.set_axisbelow(True)
    axes[0].set_ylabel(ylab)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(arms), bbox_to_anchor=(0.5, -0.07))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    S.save(fig, fname)


# ------------------------------------------------------------ figure 4
def f4_ladder():
    ratios = (0.3,)
    fig, axes = plt.subplots(1, 2, figsize=(S.DOUBLE, 2.5))
    w = 0.14
    for ax, dom in zip(axes, DOMAINS):
        x = np.arange(len(LADDER))
        for i, r in enumerate(ratios):
            vals = [series(SUITE_ROWS, dom, m, "accuracy", ratios=(r,))[0]
                    for m in LADDER]
            lo = [STATS["cells"][f"resnet18|{dom}|ood0.2|r{r}|{m}|accuracy"]["ci95_low"]
                  for m in LADDER]
            hi = [STATS["cells"][f"resnet18|{dom}|ood0.2|r{r}|{m}|accuracy"]["ci95_high"]
                  for m in LADDER]
            err = [v - l for v, l in zip(vals, lo)], [h - v for v, h in zip(vals, hi)]
            ax.bar(x + (i - 0.5) * w, vals, width=w,
                   color=[STYLE[m]["color"] for m in LADDER],
                   edgecolor="#333333", lw=0.4,
                   yerr=err, capsize=1.6,
                   error_kw=dict(lw=0.6, ecolor="#444444"))
        ax.set_xticks(x)
        ax.set_xticklabels(["single\nscale", "+trans-\nductive", "+multi-\nscale",
                            "+OOD\nweight", "+balance", "full\n(MS-BTCR)"], fontsize=6.6)
        ax.set_title(S.DOMAIN_LABEL[dom], color=S.DOMAIN_COLOR[dom])
        ax.grid(axis="y", lw=0.5)
        ax.set_axisbelow(True)
        ax.set_ylim(0.55, 0.86)
    axes[0].set_ylabel("Accuracy (noise 0.3, 20% OOD queries)")
    fig.tight_layout()
    S.save(fig, "figE2_ladder")


# ------------------------------------------------------------ figure 5
def f5_backbone():
    bks = ("resnet18", "resnet50", "vit_b_16")
    names = ("ResNet-18", "ResNet-50", "ViT-B/16")
    fig, axes = plt.subplots(1, 2, figsize=(S.DOUBLE, 2.4), sharey=True)
    w = 0.26
    for ax, dom in zip(axes, DOMAINS):
        x = np.arange(len(bks))
        for i, m in enumerate(("mstcr_ood", "bal_soft")):
            vals, los, his = [], [], []
            for bk in bks:
                c = STATS["paired"][f"{bk}|{dom}|ood0.2|r0.3|{m}-proto|accuracy"]
                vals.append(c["diff_mean"])
                los.append(c["ci95_low"])
                his.append(c["ci95_high"])
            err = [v - l for v, l in zip(vals, los)], [h - v for v, h in zip(vals, his)]
            ax.bar(x + (i - 0.5) * w, vals, width=w, color=STYLE[m]["color"],
                   edgecolor="#333333", lw=0.4, yerr=err, capsize=1.8,
                   error_kw=dict(lw=0.6, ecolor="#444444"),
                   label=STYLE[m]["label"])
        ax.axhline(0, color="#333333", lw=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(names)
        ax.set_title(S.DOMAIN_LABEL[dom], color=S.DOMAIN_COLOR[dom])
        ax.grid(axis="y", lw=0.5)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Accuracy gain over single-scale\nprototype (noise 0.3)")
    axes[0].legend(loc="upper left")
    fig.tight_layout()
    S.save(fig, "figE3_backbone")


# ------------------------------------------------------------ figure 6
def f6_sensitivity():
    rows = load(SUITE / "sensitivity_main.json")
    knobs = {r["knob"] for r in rows}
    kx = sorted([k for k in knobs if k.startswith("k=")],
                key=lambda s: float(s.split("=")[1]))
    rx = sorted([k for k in knobs if k.startswith("rounds=")],
                key=lambda s: int(s.split("=")[1]))
    # "unit" is not stored separately: it is the default configuration, so the
    # k=2 / rounds=5 cell is exactly that space.
    sx = ["k=2", "space=raw", "space=zscore"]

    def val(knob, dom, ratio, metric="accuracy"):
        v = [r[metric] for r in rows if r["knob"] == knob and r["domain"] == dom
             and r["ratio"] == ratio]
        return float(np.mean(v)) if v else np.nan

    fig, axes = plt.subplots(1, 3, figsize=(S.DOUBLE, 2.3))
    for ax, (xs, title, xtick) in zip(axes, (
            (kx, "MAD multiplier k", [float(k.split("=")[1]) for k in kx]),
            # rounds are unevenly spaced (1,2,3,5,8,12); plotting them on their
            # numeric values would bunch the first three points together, so the
            # axis is categorical and the labels carry the value
            (rx, "Re-estimation rounds", list(range(len(rx)))),
            (sx, "View normalisation", [0, 1, 2]))):
        for dom, ls, mk in (("EuroSAT_RGB", "-", "o"), ("DTD", "--", "s")):
            ys = [val(k, dom, 0.3) for k in xs]
            ax.plot(xtick, ys, color=S.DOMAIN_COLOR[dom], ls=ls, marker=mk, ms=3.2,
                    lw=1.1, label=S.DOMAIN_LABEL[dom])
        ax.set_title(title)
        ax.grid(axis="y", lw=0.5)
        ax.set_axisbelow(True)
        if xs is sx:
            ax.set_xticks([0, 1, 2])
            ax.set_xticklabels(["unit", "raw", "z-score"])
        if xs is kx:
            ax.set_xlabel("k in median + k\u00b7MAD")
        if xs is rx:
            ax.set_xticks(list(range(len(rx))))
            ax.set_xticklabels([k.split("=")[1] for k in rx])
            ax.set_xlabel("rounds")
    axes[0].set_ylabel("Accuracy (noise 0.3, 20% OOD)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    S.save(fig, "figE4_sensitivity")


# ------------------------------------------------------------ figure 7
def f7_oracle():
    rows = load(ROOT / "oracle_probe_output" / "oracle_probe.json")
    fig, axes = plt.subplots(1, 2, figsize=(S.DOUBLE, 2.45), sharey=False)
    labels = {"clean": ("clean support", "#2e7d6f", "-", "o"),
              "noisy": ("noisy support", "#a61b29", "-", "s"),
              "oracle": ("noise removed (oracle)", "#c58f2a", "--", "^"),
              "random": ("equal rows removed (random)", "#7f7f7f", ":", "v")}
    for ax, dom in zip(axes, DOMAINS):
        for meth, (lab, col, ls, mk) in labels.items():
            ys = []
            for r in RATIO_TICKS:
                v = [x["accuracy"] for x in rows if x["domain"] == dom
                     and x["ratio"] == r and x["method"] == meth and x["rule"] == "proto"]
                ys.append(np.mean(v) if v else np.nan)
            ax.plot(RATIO_TICKS, ys, color=col, ls=ls, marker=mk, ms=3.0, lw=1.0,
                    label=lab)
        ys = series(SUITE_ROWS, dom, "bal_soft", "accuracy", ood=0.2)
        ax.plot(RATIO_TICKS, ys, color="#1f4e79", ls="-", marker="P", ms=3.4, lw=1.4,
                label="MS-BTCR (20% OOD queries)")
        ax.set_title(S.DOMAIN_LABEL[dom], color=S.DOMAIN_COLOR[dom])
        ax.set_xlabel("Support-set label-noise rate")
        ax.set_xticks(RATIO_TICKS)
        ax.grid(axis="y", lw=0.5)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Accuracy")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.10))
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    S.save(fig, "figE5_oracle")


if __name__ == "__main__":
    f1_architecture()
    f2_main("accuracy", "figE6_accuracy", "Accuracy")
    f2_main("auroc", "figE7_auroc", "OOD detection AUROC")
    f4_ladder()
    f5_backbone()
    f6_sensitivity()
    f7_oracle()
    print("done")
