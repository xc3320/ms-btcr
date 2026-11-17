"""Figures for the MS-TCR manuscript.

Every panel reads the JSON written by an experiment script; no number is typed by
hand and nothing is smoothed, clipped or back-filled. Where a curve is genuinely
absent at some x (detector F1 at zero noise, since there is nothing to detect) the
point is left out and said so in the caption rather than drawn at zero.
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
MS = ROOT / "multiscale_output"
RATIOS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)
DOMAINS = ("EuroSAT_RGB", "DTD")

S.apply()

# fixed look for each method across panels
STYLE = {
    "proto_l4":   dict(color="#7f7f7f", ls="-",  marker="o", label="Single-scale prototype"),
    "proto":      dict(color="#7f7f7f", ls="-",  marker="o", label="Prototype (baseline)"),
    "tcr_l4":     dict(color="#4c8cbf", ls="--", marker="s", label="Single-scale transductive"),
    "mstcr":      dict(color="#1f4e79", ls="-",  marker="^", label="MS-TCR"),
    "mstcr_ood":  dict(color="#a61b29", ls="-",  marker="D", label="MS-TCR + OOD gating"),
    "proto_cat":  dict(color="#7f7f7f", ls="-",  marker="o", label="Prototype on concat"),
    "lp_cos_k5":  dict(color="#2e7d6f", ls="--", marker="v", label="Label prop. (k=5)"),
    "lp_cos_k10": dict(color="#2e7d6f", ls=":",  marker="v", label="Label prop. (k=10)"),
    "lp_rbf_k10": dict(color="#7aa87a", ls="-.", marker="v", label="Label prop. (RBF)"),
    "drop_any":   dict(color="#a61b29", ls="--", marker="x", label="Prune (any view)"),
    "drop_all":   dict(color="#c0554f", ls="--", marker="x", label="Prune (all views)"),
    "weight":     dict(color="#c58f2a", ls="-.", marker="P", label="LOO re-weighting"),
    "correct3":   dict(color="#6a4c93", ls=":",  marker="*", label="Strict correction"),
}


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def series(rows, dom, method, metric, ood=0.2, key_ood="ood"):
    out = []
    for r in RATIOS:
        v = [x[metric] for x in rows if x["domain"] == dom and x["method"] == method
             and x["ratio"] == r and x.get(key_ood) == ood and x.get(metric) is not None]
        out.append(float(np.mean(v)) if v else np.nan)
    return np.array(out)


def ci(rows, dom, method, metric, ood=0.2):
    lo, hi = [], []
    for r in RATIOS:
        v = [x[metric] for x in rows if x["domain"] == dom and x["method"] == method
             and x["ratio"] == r and x.get("ood") == ood and x.get(metric) is not None]
        if len(v) > 1:
            a = np.array(v, dtype=float)
            se = a.std(ddof=1) / np.sqrt(len(a))
            lo.append(a.mean() - 1.96 * se)
            hi.append(a.mean() + 1.96 * se)
        else:
            lo.append(np.nan)
            hi.append(np.nan)
    return np.array(lo), np.array(hi)


# ---------------------------------------------------------------- figure 1: design
def f1_architecture():
    fig, ax = plt.subplots(figsize=(S.DOUBLE, 2.45))
    ax.set_xlim(0, 100); ax.set_ylim(0, 34); ax.axis("off")

    def box(x, y, w, h, text, fc="#eef2f7", ec="#333333", fs=7.2, bold=False):
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, lw=0.7))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, fontweight="bold" if bold else "normal", color="#222222")

    def arrow(x1, y1, x2, y2, txt=None):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", lw=0.7, color="#555555",
                                    shrinkA=0, shrinkB=0))
        if txt:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 1.2, txt, ha="center", va="bottom",
                    fontsize=6.6, color="#555555")

    box(1, 11, 17, 12, "Frozen ResNet-18\n(single forward)", "#f2f2f2", fs=6.8, bold=True)
    ys = [25, 16, 7]
    names = ["layer2\n128-d", "layer3\n256-d", "layer4\n512-d"]
    for y, n in zip(ys, names):
        box(22, y, 12, 6.4, n, "#dbe5f1", fs=6.8)
        arrow(18.2, 17, 21.8, y + 3.2)

    box(37, 14, 15, 8, "unit-norm\neach view", "#fdf0e6", bold=True)
    for y in ys:
        arrow(34.2, y + 3.2, 36.8, 18)
    box(55, 14, 15, 8, "concatenate\n896-d", "#fdf0e6", bold=True)
    arrow(52.2, 18, 54.8, 18)

    box(55, 26, 15, 6.5, "support + query\n= 100 rows", "#e8f0e8", fs=6.8)
    box(73, 14, 14, 8, "transductive\ncentre update", "#dbe5f1", bold=True)
    arrow(70.2, 18, 72.8, 18)
    arrow(62.5, 25.7, 79, 22.4, "pooled")

    box(90, 20, 9.5, 6.2, "class\nprediction", "#f2f2f2", fs=6.6)
    box(90, 10, 9.5, 6.2, "OOD\nscore", "#f2f2f2", fs=6.6)
    arrow(87.2, 20, 89.8, 22.8)
    arrow(87.2, 16, 89.8, 13.4)

    box(73, 2.5, 14, 6, "median + k·MAD\ngating", "#fdf0e6", fs=6.6)
    arrow(80, 8.7, 80, 13.8)

    ax.text(50, 0.6, "no labels are corrected and no support row is discarded",
            ha="center", fontsize=7, style="italic", color="#a61b29")
    S.save(fig, "figD1_architecture")


# ---------------------------------------------------------------- figure 2/3
def f2_main(metric, fname, ylab):
    rows = load(ROOT / "mstcr_output" / "mstcr_main.json")
    fig, axes = plt.subplots(1, 2, figsize=(S.DOUBLE, 2.5), sharey=True)
    for ax, dom in zip(axes, DOMAINS):
        for m in ("proto_l4", "tcr_l4", "mstcr", "mstcr_ood"):
            y = series(rows, dom, m, metric)
            lo, hi = ci(rows, dom, m, metric)
            st = STYLE[m]
            ax.plot(RATIOS, y, color=st["color"], ls=st["ls"], marker=st["marker"],
                    ms=3.2, lw=1.1, label=st["label"])
            ax.fill_between(RATIOS, lo, hi, color=st["color"], alpha=0.12, lw=0)
        ax.set_title(S.DOMAIN_LABEL[dom], color=S.DOMAIN_COLOR[dom])
        ax.set_xlabel("Support-set label-noise rate")
        ax.set_xticks(RATIOS)
        ax.grid(axis="y", lw=0.5)
        ax.set_axisbelow(True)
    axes[0].set_ylabel(ylab)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.055))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    S.save(fig, fname)


# ---------------------------------------------------------------- figure 4: strategies
def f4_strategies():
    d = load(ROOT / "diagnostic_output" / "diag_main.json")
    rows = d["acc"]
    fig, axes = plt.subplots(1, 2, figsize=(S.DOUBLE, 2.5), sharey=True)
    order = ["proto", "drop_any", "drop_all", "weight", "correct3", "mstcr"]
    for ax, dom in zip(axes, DOMAINS):
        for m in order:
            st = STYLE[m]
            y = series(rows, dom, m, "accuracy", ood=None, key_ood="__none__")
            ax.plot(RATIOS, y, color=st["color"], ls=st["ls"], marker=st["marker"],
                    ms=3.2, lw=1.1, label=st["label"])
        ax.set_title(S.DOMAIN_LABEL[dom], color=S.DOMAIN_COLOR[dom])
        ax.set_xlabel("Support-set label-noise rate")
        ax.set_xticks(RATIOS)
        ax.grid(axis="y", lw=0.5)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Accuracy (clean query set)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.10))
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    S.save(fig, "figD3_strategies")


# ---------------------------------------------------------------- figure 5: norms
def f5_norms():
    fig, axes = plt.subplots(1, 2, figsize=(S.DOUBLE, 2.3),
                             gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    labels, means, colors = [], [], []
    for dom in DOMAINS:
        ms = np.load(MS / ("eurosat_multiscale.npz" if dom == "EuroSAT_RGB" else "dtd_multiscale.npz"))
        for k, name in (("x2", "L2"), ("x3", "L3"), ("x4", "L4")):
            labels.append(f"{name}\n{S.DOMAIN_LABEL[dom]}")
            means.append(float(np.linalg.norm(ms[k].astype(np.float64), axis=1).mean()))
            colors.append(S.DOMAIN_COLOR[dom])
    x = np.arange(len(labels))
    ax.bar(x, means, color=colors, width=0.62)
    for xi, v in zip(x, means):
        ax.text(xi, v + 0.4, f"{v:.1f}", ha="center", fontsize=6.6)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=6.6)
    ax.set_ylabel("Mean feature L2 norm")
    ax.set_title("Raw view magnitudes")
    ax.grid(axis="y", lw=0.5); ax.set_axisbelow(True)

    # second panel: normalised vs raw concatenation, from the ablation run
    ax = axes[1]
    abl = load(ROOT / "ms_ablation_output" / "ablation_smoke20.json")["acc"]
    vals_norm, vals_raw = [], []
    for dom in DOMAINS:
        for m, bucket in (("tk_norm_concat", vals_norm), ("tk_raw_concat", vals_raw)):
            v = [x["accuracy"] for x in abl if x["domain"] == dom and x["method"] == m
                 and x["ratio"] == 0.3]
            bucket.append(float(np.mean(v)) if v else np.nan)
    x = np.arange(len(DOMAINS))
    w = 0.36
    ax.bar(x - w / 2, vals_raw, w, label="raw concat", color="#c58f2a")
    ax.bar(x + w / 2, vals_norm, w, label="unit-norm concat", color="#1f4e79")
    for xi, a, b in zip(x, vals_raw, vals_norm):
        ax.text(xi - w / 2, a + 0.006, f"{a:.3f}", ha="center", fontsize=6.4)
        ax.text(xi + w / 2, b + 0.006, f"{b:.3f}", ha="center", fontsize=6.4)
    ax.set_xticks(x); ax.set_xticklabels([S.DOMAIN_LABEL[d] for d in DOMAINS])
    ax.set_ylabel("Accuracy at 30% noise")
    ax.set_ylim(0.60, 0.845)
    ax.set_title("Effect of normalising before concat")
    ax.legend(fontsize=6.2, loc="upper center", ncol=2,
              bbox_to_anchor=(0.5, -0.13), handlelength=1.2)
    ax.grid(axis="y", lw=0.5); ax.set_axisbelow(True)
    fig.tight_layout()
    S.save(fig, "figD4_norms")


# ---------------------------------------------------------------- figure 6: ablation
def f6_ablation():
    rows = load(ROOT / "mstcr_output" / "mstcr_main.json")
    fig, axes = plt.subplots(1, 2, figsize=(S.DOUBLE, 2.5), sharey=True)
    for ax, dom in zip(axes, DOMAINS):
        base = series(rows, dom, "proto_l4", "accuracy")
        single = series(rows, dom, "tcr_l4", "accuracy")
        full = series(rows, dom, "mstcr", "accuracy")
        ax.fill_between(RATIOS, base, single, color="#4c8cbf", alpha=0.38, lw=0,
                        label="gain from transductive update")
        ax.fill_between(RATIOS, single, full, color="#c58f2a", alpha=0.38, lw=0,
                        label="gain from multi-scale")
        st = STYLE["proto_l4"]
        ax.plot(RATIOS, base, color=st["color"], ls="-", marker="o", ms=3, lw=1.1,
                label="single-scale prototype")
        st = STYLE["mstcr"]
        ax.plot(RATIOS, full, color=st["color"], ls="-", marker="^", ms=3.2, lw=1.2,
                label="MS-TCR")
        ax.set_title(S.DOMAIN_LABEL[dom], color=S.DOMAIN_COLOR[dom])
        ax.set_xlabel("Support-set label-noise rate")
        ax.set_xticks(RATIOS)
        ax.grid(axis="y", lw=0.5); ax.set_axisbelow(True)
    axes[0].set_ylabel("Accuracy (20% OOD queries)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.10), fontsize=6.6)
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    S.save(fig, "figD5_ablation")


# ---------------------------------------------------------------- figure 7: label prop
def f7_labelprop():
    ms = load(ROOT / "mstcr_output" / "mstcr_main.json")
    tp = load(ROOT / "tp_baseline_output" / "tp_main.json")
    fig, axes = plt.subplots(1, 2, figsize=(S.DOUBLE, 2.5), sharey=True)
    for ax, dom in zip(axes, DOMAINS):
        for m, src in (("proto_cat", tp), ("lp_cos_k5", tp), ("lp_cos_k10", tp),
                       ("lp_rbf_k10", tp), ("mstcr_ood", ms)):
            st = STYLE[m]
            y = series(src, dom, m, "accuracy")
            ax.plot(RATIOS, y, color=st["color"], ls=st["ls"], marker=st["marker"],
                    ms=3.2, lw=1.1, label=st["label"])
        ax.set_title(S.DOMAIN_LABEL[dom], color=S.DOMAIN_COLOR[dom])
        ax.set_xlabel("Support-set label-noise rate")
        ax.set_xticks(RATIOS)
        ax.grid(axis="y", lw=0.5); ax.set_axisbelow(True)
    axes[0].set_ylabel("Accuracy (20% OOD queries)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.10), fontsize=6.6)
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    S.save(fig, "figD6_labelprop")


# ---------------------------------------------------------------- figure 8: sensitivity
def f8_sensitivity():
    rows = load(ROOT / "mstcr_sensitivity_output" / "sens_main.json")
    fig, axes = plt.subplots(1, 3, figsize=(S.DOUBLE, 2.2))

    ax = axes[0]
    ks = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
    for dom in DOMAINS:
        y = [float(np.mean([r["accuracy"] for r in rows if r["domain"] == dom
                            and r["ratio"] == 0.3 and r["knob"] == f"k={k:g}"])) for k in ks]
        ax.plot(ks, y, color=S.DOMAIN_COLOR[dom], marker="o", ms=3, lw=1.1,
                label=S.DOMAIN_LABEL[dom])
    ax.set_xlabel("MAD multiplier k"); ax.set_ylabel("Accuracy at 30% noise")
    ax.set_title("OOD gate threshold"); ax.legend(fontsize=6.4)
    ax.grid(axis="y", lw=0.5); ax.set_axisbelow(True)

    ax = axes[1]
    rs = [1, 2, 3, 5, 8, 12]
    for dom in DOMAINS:
        y = [float(np.mean([r["accuracy"] for r in rows if r["domain"] == dom
                            and r["ratio"] == 0.3 and r["knob"] == f"rounds={k}"])) for k in rs]
        ax.plot(rs, y, color=S.DOMAIN_COLOR[dom], marker="s", ms=3, lw=1.1,
                label=S.DOMAIN_LABEL[dom])
    ax.set_xlabel("Centre-update rounds"); ax.set_ylabel("Accuracy at 30% noise")
    ax.set_title("Iteration count"); ax.legend(fontsize=6.4)
    ax.grid(axis="y", lw=0.5); ax.set_axisbelow(True)

    ax = axes[2]
    for i, dom in enumerate(DOMAINS):
        l2 = float(np.mean([r["accuracy"] for r in rows if r["domain"] == dom
                            and r["ratio"] == 0.3 and r["knob"] == "k=2"]))
        zs = float(np.mean([r["accuracy"] for r in rows if r["domain"] == dom
                            and r["ratio"] == 0.3 and r["knob"] == "norm=zscore"]))
        ax.bar([i * 1.5, i * 1.5 + 0.55], [l2, zs], width=0.5,
               color=[S.DOMAIN_COLOR[dom], "#c8c8c8"], edgecolor="#333333", lw=0.4)
        ax.text(i * 1.5, l2 + 0.004, f"{l2:.3f}", ha="center", fontsize=6.2)
        ax.text(i * 1.5 + 0.55, zs + 0.004, f"{zs:.3f}", ha="center", fontsize=6.2)
    ax.set_xticks([0.27, 1.77])
    ax.set_xticklabels(["EuroSAT", "DTD"], fontsize=6.8)
    ax.set_ylim(0.60, 0.83)
    ax.set_ylabel("Accuracy at 30% noise")
    ax.set_title("Normalisation choice")
    ax.grid(axis="y", lw=0.5); ax.set_axisbelow(True)

    fig.tight_layout()
    S.save(fig, "figD7_sensitivity")


if __name__ == "__main__":
    import sys as _s
    todo = _s.argv[1:] or ["1", "2", "4", "5", "6", "7", "8"]
    if "1" in todo: f1_architecture(); print("figD1 ok")
    if "2" in todo:
        f2_main("accuracy", "figD2_main_accuracy", "Accuracy (20% OOD queries)")
        print("figD2 ok")
        f2_main("auroc", "figD2b_main_auroc", "AUROC (OOD detection)")
        print("figD2b ok")
    if "4" in todo: f4_strategies(); print("figD3 ok")
    if "5" in todo: f5_norms(); print("figD4 ok")
    if "6" in todo: f6_ablation(); print("figD5 ok")
    if "7" in todo: f7_labelprop(); print("figD6 ok")
    if "8" in todo: f8_sensitivity(); print("figD7 ok")
