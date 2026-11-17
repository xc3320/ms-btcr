"""Split-level statistics pooled across backbones.

Same inference unit as mstcr_stats.py: 10 split means per cell, paired t test
across splits. The ResNet-18 run and the two new runs use the same grid, the
same episode streams (seed 111000 + fold) and the same five arms, so the only
difference between rows is the feature source.

Method names differ between files (proto_l4 vs proto_deep); they are mapped to
one vocabulary here before pooling, otherwise a typo would silently drop a
backbone from the comparison instead of raising.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats as sps

ROOT = Path(__file__).resolve().parent.parent
RAW = {
    "resnet18": ROOT / "mstcr_output" / "mstcr_main.json",
    "resnet50": ROOT / "backbone_output" / "resnet50_main.json",
    "vit_b_16": ROOT / "backbone_output" / "vit_b_16_main.json",
}
ALIAS = {"proto_l4": "proto_deep", "tcr_l4": "tcr_deep",
         "tcr_raw": "tcr_raw", "mstcr": "mstcr", "mstcr_ood": "mstcr_ood"}
METHS = ["proto_deep", "tcr_deep", "tcr_raw", "mstcr", "mstcr_ood"]


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
    for bk, f in RAW.items():
        if not f.exists():
            raise SystemExit(f"missing: {f}")
        for r in json.loads(f.read_text(encoding="utf-8")):
            m = ALIAS.get(r["method"])
            if m is None:
                raise SystemExit(f"unmapped method {r['method']} in {f.name}")
            rows.append({**r, "backbone": bk, "method": m})
    return rows


def main():
    rows = load()
    out, pair = {}, {}

    for bk in RAW:
        bk_rows = [r for r in rows if r["backbone"] == bk]
        for dom in ("EuroSAT_RGB", "DTD"):
            for ood in (0.0, 0.2):
                for ratio in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5):
                    for metric in ("accuracy", "auroc"):
                        for meth in METHS:
                            v = [r[metric] for r in bk_rows if r["domain"] == dom
                                 and r["ood"] == ood and r["ratio"] == ratio
                                 and r["method"] == meth and r.get(metric) is not None]
                            if len(v) < 2:
                                continue
                            m, sd, lo, hi, n = t_ci(v)
                            out[f"{bk}|{dom}|ood{ood}|r{ratio}|{meth}|{metric}"] = {
                                "mean": round(m, 6), "sd": round(sd, 6),
                                "ci95_low": round(lo, 6), "ci95_high": round(hi, 6),
                                "n_splits": n}

        for dom in ("EuroSAT_RGB", "DTD"):
            for ood in (0.0, 0.2):
                for ratio in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5):
                    for metric in ("accuracy", "auroc"):
                        base = {r["split"]: r[metric] for r in bk_rows
                                if r["domain"] == dom and r["ood"] == ood
                                and r["ratio"] == ratio and r["method"] == "proto_deep"
                                and r.get(metric) is not None}
                        for meth in ("tcr_deep", "tcr_raw", "mstcr", "mstcr_ood"):
                            cur = {r["split"]: r[metric] for r in bk_rows
                                   if r["domain"] == dom and r["ood"] == ood
                                   and r["ratio"] == ratio and r["method"] == meth
                                   and r.get(metric) is not None}
                            shared = sorted(set(base) & set(cur))
                            if len(shared) < 3:
                                continue
                            d = np.array([cur[s] - base[s] for s in shared])
                            m, sd, lo, hi, n = t_ci(d)
                            t, p = sps.ttest_1samp(d, 0.0)
                            pair[f"{bk}|{dom}|ood{ood}|r{ratio}|{meth}-proto|{metric}"] = {
                                "diff_mean": round(m, 6), "ci95_low": round(lo, 6),
                                "ci95_high": round(hi, 6), "t": round(float(t), 4),
                                "p": round(float(p), 6), "n_splits": n,
                                "n_splits_won": int((d > 0).sum())}

    (ROOT / "backbone_output" / "stats_backbone.json").write_text(
        json.dumps({"cells": out, "paired": pair}, indent=1), encoding="utf-8")

    print("=== accuracy, ood=0.2, gain over single-scale prototype per backbone ===")
    for dom in ("EuroSAT_RGB", "DTD"):
        for ratio in (0.0, 0.2, 0.3, 0.5):
            print(f"\n{dom}  noise={ratio}")
            print(f"{'backbone':10s} {'arm':>10s} {'base':>7s} {'full':>7s} "
                  f"{'diff':>8s} {'95% CI':>19s} {'p':>9s} {'won':>6s}")
            for bk in RAW:
                kb = f"{bk}|{dom}|ood0.2|r{ratio}|proto_deep|accuracy"
                kf = f"{bk}|{dom}|ood0.2|r{ratio}|mstcr_ood|accuracy"
                kp = f"{bk}|{dom}|ood0.2|r{ratio}|mstcr_ood-proto|accuracy"
                if kb not in out or kf not in out or kp not in pair:
                    print(f"{bk:10s}  -- missing --")
                    continue
                v = pair[kp]
                print(f"{bk:10s} {'mstcr_ood':>10s} {out[kb]['mean']:7.4f} "
                      f"{out[kf]['mean']:7.4f} {v['diff_mean']:+8.4f} "
                      f"[{v['ci95_low']:+.4f},{v['ci95_high']:+.4f}] "
                      f"{v['p']:9.2e} {v['n_splits_won']:>3d}/10")

    print("\n=== AUROC, ood=0.2, noise=0.3 ===")
    print(f"{'backbone':10s} {'domain':12s} {'base':>7s} {'full':>7s} {'diff':>8s} {'won':>6s}")
    for bk in RAW:
        for dom in ("EuroSAT_RGB", "DTD"):
            kb = f"{bk}|{dom}|ood0.2|r0.3|proto_deep|auroc"
            kf = f"{bk}|{dom}|ood0.2|r0.3|mstcr_ood|auroc"
            kp = f"{bk}|{dom}|ood0.2|r0.3|mstcr_ood-proto|auroc"
            if kb in out and kf in out and kp in pair:
                v = pair[kp]
                print(f"{bk:10s} {dom:12s} {out[kb]['mean']:7.4f} {out[kf]['mean']:7.4f} "
                      f"{v['diff_mean']:+8.4f} {v['n_splits_won']:>3d}/10")


if __name__ == "__main__":
    main()
