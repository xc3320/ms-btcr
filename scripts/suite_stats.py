"""Statistics for the unified suite, pooled over backbones.

Inference unit is the split, not the episode: 100 episodes inside a split share
the same image pool, so treating them as independent samples would shrink the
interval by roughly sqrt(100) for free. Every number the paper reports is a mean
over 10 split means, with a paired t interval on the per-split differences.

Three comparison families are emitted, because the paper makes three kinds of
claim:
  vs proto_deep   the headline gain of the final rule over the single-scale rule
  vs mstcr_ood    what balancing adds once the rest of the method is in place
  vs bal_iter     what out-of-distribution weighting adds on top of balancing
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats as sps

ROOT = Path(__file__).resolve().parent.parent
SUITE = ROOT / "suite_output"
BACKBONES = ("resnet18", "resnet50", "vit_b_16")
ARMS = ("proto_deep", "tcr_deep", "tcr_raw", "mstcr", "mstcr_ood",
        "bal_iter", "bal_ood", "bal_soft", "obtcr")
FINAL = "bal_soft"


def t_ci(vals):
    a = np.asarray(vals, dtype=float)
    a = a[np.isfinite(a)]
    n = len(a)
    m = float(a.mean())
    sd = float(a.std(ddof=1)) if n > 1 else float("nan")
    half = float(sps.t.ppf(0.975, n - 1) * sd / np.sqrt(n)) if n > 1 and sd > 0 else 0.0
    return m, sd, m - half, m + half, n


def load():
    rows = []
    for bk in BACKBONES:
        f = SUITE / f"{bk}_main.json"
        if not f.exists():
            raise SystemExit(f"missing {f}")
        rows.extend(json.loads(f.read_text(encoding="utf-8")))
    return rows


def main():
    rows = load()
    cells, paired = {}, {}

    for bk in BACKBONES:
        sub = [r for r in rows if r["backbone"] == bk]
        for dom in ("EuroSAT_RGB", "DTD"):
            for ood in (0.0, 0.2):
                for ratio in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5):
                    for metric in ("accuracy", "auroc"):
                        for arm in ARMS:
                            v = [r[metric] for r in sub if r["domain"] == dom
                                 and r["ood"] == ood and r["ratio"] == ratio
                                 and r["method"] == arm and r.get(metric) is not None]
                            if len(v) < 2:
                                continue
                            m, sd, lo, hi, n = t_ci(v)
                            cells[f"{bk}|{dom}|ood{ood}|r{ratio}|{arm}|{metric}"] = {
                                "mean": round(m, 6), "sd": round(sd, 6),
                                "ci95_low": round(lo, 6), "ci95_high": round(hi, 6),
                                "n_splits": n}

                        for arm in ("tcr_deep", "tcr_raw", "mstcr", "mstcr_ood",
                                    "bal_iter", "bal_ood", "bal_soft", "obtcr"):
                            base = {r["split"]: r[metric] for r in sub
                                    if r["domain"] == dom and r["ood"] == ood
                                    and r["ratio"] == ratio
                                    and r["method"] == "proto_deep"
                                    and r.get(metric) is not None}
                            cur = {r["split"]: r[metric] for r in sub
                                   if r["domain"] == dom and r["ood"] == ood
                                   and r["ratio"] == ratio and r["method"] == arm
                                   and r.get(metric) is not None}
                            shared = sorted(set(base) & set(cur))
                            if len(shared) < 3:
                                continue
                            d = np.array([cur[s] - base[s] for s in shared])
                            m, sd, lo, hi, n = t_ci(d)
                            t, p = sps.ttest_1samp(d, 0.0)
                            paired[f"{bk}|{dom}|ood{ood}|r{ratio}|{arm}-proto|{metric}"] = {
                                "diff_mean": round(m, 6), "ci95_low": round(lo, 6),
                                "ci95_high": round(hi, 6), "t": round(float(t), 4),
                                "p": round(float(p), 8), "n_splits": n,
                                "n_splits_won": int((d > 0).sum())}

                        # head-to-head between the two most informative arms
                        for arm, ref in ((FINAL, "mstcr_ood"), (FINAL, "bal_iter")):
                            base = {r["split"]: r[metric] for r in sub
                                    if r["domain"] == dom and r["ood"] == ood
                                    and r["ratio"] == ratio and r["method"] == ref
                                    and r.get(metric) is not None}
                            cur = {r["split"]: r[metric] for r in sub
                                   if r["domain"] == dom and r["ood"] == ood
                                   and r["ratio"] == ratio and r["method"] == arm
                                   and r.get(metric) is not None}
                            shared = sorted(set(base) & set(cur))
                            if len(shared) < 3:
                                continue
                            d = np.array([cur[s] - base[s] for s in shared])
                            m, sd, lo, hi, n = t_ci(d)
                            t, p = sps.ttest_1samp(d, 0.0)
                            paired[f"{bk}|{dom}|ood{ood}|r{ratio}|{arm}-{ref}|{metric}"] = {
                                "diff_mean": round(m, 6), "ci95_low": round(lo, 6),
                                "ci95_high": round(hi, 6), "t": round(float(t), 4),
                                "p": round(float(p), 8), "n_splits": n,
                                "n_splits_won": int((d > 0).sum())}

    (SUITE / "stats_suite.json").write_text(
        json.dumps({"cells": cells, "paired": paired}, indent=1), encoding="utf-8")

    print("=== accuracy, ood=0.2: gain of the final rule over single-scale prototype ===")
    for dom in ("EuroSAT_RGB", "DTD"):
        print(f"\n{dom}")
        print(f"{'backbone':10s} {'noise':>6s} {'proto':>7s} {'final':>7s} "
              f"{'diff':>8s} {'95% CI':>20s} {'p':>10s} {'won':>6s}")
        for bk in BACKBONES:
            for ratio in (0.0, 0.2, 0.3, 0.5):
                kb = f"{bk}|{dom}|ood0.2|r{ratio}|proto_deep|accuracy"
                kf = f"{bk}|{dom}|ood0.2|r{ratio}|{FINAL}|accuracy"
                kp = f"{bk}|{dom}|ood0.2|r{ratio}|{FINAL}-proto|accuracy"
                if kb not in cells or kf not in cells or kp not in paired:
                    continue
                v = paired[kp]
                print(f"{bk:10s} {ratio:6.1f} {cells[kb]['mean']:7.4f} "
                      f"{cells[kf]['mean']:7.4f} {v['diff_mean']:+8.4f} "
                      f"[{v['ci95_low']:+.4f},{v['ci95_high']:+.4f}] "
                      f"{v['p']:10.2e} {v['n_splits_won']:>3d}/10")

    print("\n=== what each block adds, accuracy at noise 0.3, ood=0.2 ===")
    print(f"{'backbone':10s} {'domain':12s} " + "".join(f"{a:>10s}" for a in
          ("proto", "tcr", "mstcr", "mstcr_ood", "bal_iter", "bal_soft")))
    for bk in BACKBONES:
        for dom in ("EuroSAT_RGB", "DTD"):
            line = f"{bk:10s} {dom:12s} "
            for arm in ("proto_deep", "tcr_deep", "mstcr", "mstcr_ood",
                        "bal_iter", "bal_soft"):
                k = f"{bk}|{dom}|ood0.2|r0.3|{arm}|accuracy"
                line += f"{cells[k]['mean']:10.4f}" if k in cells else f"{'-':>10s}"
            print(line)

    print("\n=== final vs the two strongest alternatives (accuracy, noise 0.3, ood=0.2) ===")
    for bk in BACKBONES:
        for dom in ("EuroSAT_RGB", "DTD"):
            for ref in ("mstcr_ood", "bal_iter"):
                k = f"{bk}|{dom}|ood0.2|r0.3|{FINAL}-{ref}|accuracy"
                if k not in paired:
                    continue
                v = paired[k]
                print(f"{bk:10s} {dom:12s} vs {ref:10s} {v['diff_mean']:+8.4f} "
                      f"[{v['ci95_low']:+.4f},{v['ci95_high']:+.4f}] "
                      f"p={v['p']:.2e} won {v['n_splits_won']}/10")


if __name__ == "__main__":
    main()
