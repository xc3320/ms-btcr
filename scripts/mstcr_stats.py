"""Split-level statistics for the MS-TCR comparison.

Each domain x condition x method cell holds 10 independent split means (each from
100 episodes). Inference is across splits, not across episodes, because episodes
within a split share the same image pool and are not independent - the same
choice as the earlier round, so numbers stay comparable with the previous drafts.

Reported per cell: mean, sd, 95% t interval across the 10 splits.
Reported per comparison: paired difference per split, its mean and interval, and
a two-sided paired t test, so a gain is not claimed from overlapping intervals.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats as sps

ROOT = Path(__file__).resolve().parent.parent
MSTCR = ROOT / "mstcr_output" / "mstcr_main.json"


def t_ci(vals):
    a = np.asarray(vals, dtype=float)
    a = a[np.isfinite(a)]
    n = len(a)
    m = float(a.mean())
    sd = float(a.std(ddof=1)) if n > 1 else float("nan")
    half = float(sps.t.ppf(0.975, n - 1) * sd / np.sqrt(n)) if n > 1 and sd > 0 else 0.0
    return m, sd, m - half, m + half, n


def main():
    rows = json.loads(MSTCR.read_text(encoding="utf-8"))
    out = {}

    for dom in ("EuroSAT_RGB", "DTD"):
        for ood in (0.0, 0.2):
            for ratio in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5):
                for metric in ("accuracy", "auroc"):
                    for meth in ("proto_l4", "tcr_l4", "mstcr", "mstcr_ood"):
                        vals = [r[metric] for r in rows
                                if r["domain"] == dom and r["ood"] == ood
                                and r["ratio"] == ratio and r["method"] == meth
                                and r.get(metric) is not None]
                        if len(vals) < 2:
                            continue
                        m, sd, lo, hi, n = t_ci(vals)
                        out[f"{dom}|ood{ood}|r{ratio}|{meth}|{metric}"] = {
                            "mean": round(m, 6), "sd": round(sd, 6),
                            "ci95_low": round(lo, 6), "ci95_high": round(hi, 6), "n_splits": n}

    # paired differences against the single-scale prototypical baseline
    pair = {}
    for dom in ("EuroSAT_RGB", "DTD"):
        for ood in (0.0, 0.2):
            for ratio in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5):
                for metric in ("accuracy", "auroc"):
                    base = {r["split"]: r[metric] for r in rows
                            if r["domain"] == dom and r["ood"] == ood and r["ratio"] == ratio
                            and r["method"] == "proto_l4" and r.get(metric) is not None}
                    for meth in ("tcr_l4", "mstcr", "mstcr_ood"):
                        cur = {r["split"]: r[metric] for r in rows
                               if r["domain"] == dom and r["ood"] == ood and r["ratio"] == ratio
                               and r["method"] == meth and r.get(metric) is not None}
                        shared = sorted(set(base) & set(cur))
                        if len(shared) < 3:
                            continue
                        d = np.array([cur[s] - base[s] for s in shared])
                        m, sd, lo, hi, n = t_ci(d)
                        t, p = sps.ttest_1samp(d, 0.0)
                        pair[f"{dom}|ood{ood}|r{ratio}|{meth}-proto|{metric}"] = {
                            "diff_mean": round(m, 6), "ci95_low": round(lo, 6),
                            "ci95_high": round(hi, 6), "t": round(float(t), 4),
                            "p": round(float(p), 6), "n_splits": n,
                            "n_splits_won": int((d > 0).sum())}

    (ROOT / "mstcr_output" / "stats_main.json").write_text(
        json.dumps({"cells": out, "paired": pair}, indent=1), encoding="utf-8")

    print("=== paired difference vs proto_l4 (accuracy, ood=0.2) ===")
    print(f"{'domain':12s} {'ratio':>5s} {'method':>10s} {'diff':>9s} {'95% CI':>20s} {'p':>9s} {'won':>6s}")
    for dom in ("EuroSAT_RGB", "DTD"):
        for ratio in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5):
            for meth in ("tcr_l4", "mstcr", "mstcr_ood"):
                k = f"{dom}|ood0.2|r{ratio}|{meth}-proto|accuracy"
                if k not in pair:
                    continue
                v = pair[k]
                print(f"{dom:12s} {ratio:5.1f} {meth:>10s} {v['diff_mean']:+9.4f} "
                      f"[{v['ci95_low']:+.4f},{v['ci95_high']:+.4f}] {v['p']:9.2e} {v['n_splits_won']:>3d}/10")

    print("\n=== paired difference vs proto_l4 (AUROC, ood=0.2) ===")
    for dom in ("EuroSAT_RGB", "DTD"):
        for ratio in (0.0, 0.2, 0.3, 0.5):
            for meth in ("mstcr", "mstcr_ood"):
                k = f"{dom}|ood0.2|r{ratio}|{meth}-proto|auroc"
                if k not in pair:
                    continue
                v = pair[k]
                print(f"{dom:12s} {ratio:5.1f} {meth:>10s} {v['diff_mean']:+9.4f} "
                      f"[{v['ci95_low']:+.4f},{v['ci95_high']:+.4f}] {v['p']:9.2e} {v['n_splits_won']:>3d}/10")


if __name__ == "__main__":
    main()
