"""Sensitivity of MS-TCR to its three free knobs.

A training-free method that only works at one setting has not been shown to work.
The knobs are:
  k      - MAD multiplier in the OOD down-weighting threshold (median + k*MAD)
  rounds - Lloyd iterations for centre re-estimation
  norm   - how views are put on a common scale before concatenation

Swept at 0% and 30% support noise with a 20% out-of-distribution query set,
which is the regime the method is meant for. Reported as accuracy and AUROC.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats as sps

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from oracle_probe import N_WAY, K_SHOT, N_QUERY, split_indices  # noqa: E402
from mstcr_experiment import corrupt, unit, draw_episode  # noqa: E402

ROOT = HERE.parent
MS = ROOT / "multiscale_output"
VIEWS = ("l2", "l3", "l4")
RATIOS = (0.0, 0.3)
OOD_R = 0.2
K_GRID = (1.0, 1.5, 2.0, 2.5, 3.0, 4.0)
R_GRID = (1, 2, 3, 5, 8, 12)


def tcr_core(sx, sy, qx, classes, rounds, ood_aware, mad_k):
    m = np.stack([sx[sy == c].mean(0) for c in classes])
    allx = np.vstack([sx, qx])
    n_sup = len(sx)
    for _ in range(rounds):
        d = ((allx[:, None, :] - m[None, :, :]) ** 2).sum(2)
        lab = d.argmin(1)
        w = np.ones(len(allx))
        if ood_aware:
            s = d.min(1)
            med = np.median(s)
            mad = np.median(np.abs(s - med)) + 1e-12
            w[s > med + mad_k * mad] = 0.1
            w[:n_sup] = 1.0
        for k in range(len(classes)):
            sel = lab == k
            if sel.any():
                m[k] = np.average(allx[sel], axis=0, weights=w[sel])
    d = ((qx[:, None, :] - m[None, :, :]) ** 2).sum(2)
    return classes[d.argmin(1)], d.min(1)


def zscore_all(x):
    mu = x.mean(0, keepdims=True)
    sd = x.std(0, keepdims=True) + 1e-12
    return (x - mu) / sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--tag", default="main")
    args = ap.parse_args()
    t0 = time.time()
    out_dir = ROOT / "mstcr_sensitivity_output"
    out_dir.mkdir(exist_ok=True)

    rows = []
    for name, msfile in (("EuroSAT_RGB", MS / "eurosat_multiscale.npz"),
                         ("DTD", MS / "dtd_multiscale.npz")):
        ms = np.load(msfile, allow_pickle=False)
        views = {k: ms[f"x{k[1]}"].astype(np.float64) for k in VIEWS}
        key2i = {str(k): i for i, k in enumerate(ms["keys"])}
        y = ms["y"]

        for fi in range(1, 11):
            idx = split_indices(name, fi, y, key2i)
            tv = {k: views[k][idx] for k in VIEWS}
            tv["cat"] = np.hstack([unit(tv[k]) for k in VIEWS])
            # z-score variant uses per-split statistics, i.e. transductive and label-free
            tv["z"] = np.hstack([zscore_all(tv[k]) for k in VIEWS])
            ty = y[idx]
            rng = np.random.default_rng(115_000 + fi)

            for ratio in RATIOS:
                bucket: dict[str, list[tuple[float, float]]] = {}
                for _ in range(args.episodes):
                    s, q, sy = draw_episode(ty, rng)
                    qy = ty[q]
                    sy_n, _ = corrupt(sy, ratio, rng)
                    classes = np.unique(sy)
                    outside = np.flatnonzero(~np.isin(ty, classes))
                    n_ood = max(1, int(round(len(q) * OOD_R)))
                    pick = rng.choice(outside, size=min(n_ood, len(outside)), replace=False)
                    qx_idx = np.concatenate([q, pick])
                    qy_use = np.concatenate([qy, np.full(len(pick), -1)])
                    is_ood = np.concatenate([np.zeros(len(q), dtype=bool),
                                             np.ones(len(pick), dtype=bool)])
                    clean = ~is_ood

                    def rec(tag, pred, score):
                        a = float((pred[clean] == qy_use[clean]).mean())
                        lb = np.concatenate([np.zeros(clean.sum()), np.ones(is_ood.sum())])
                        sc = np.concatenate([score[clean], score[is_ood]])
                        r = sps.rankdata(sc)[lb == 1].mean()
                        au = float((r - (is_ood.sum() + 1) / 2) / clean.sum())
                        bucket.setdefault(tag, []).append((a, au))

                    sp_ = {k: tv[k][s] for k in tv}
                    qp_ = {k: tv[k][qx_idx] for k in tv}

                    for k in K_GRID:
                        p, sc = tcr_core(sp_["cat"], sy_n, qp_["cat"], classes, 5, True, k)
                        rec(f"k={k:g}", p, sc)
                    for r in R_GRID:
                        p, sc = tcr_core(sp_["cat"], sy_n, qp_["cat"], classes, r, True, 2.0)
                        rec(f"rounds={r}", p, sc)
                    p, sc = tcr_core(sp_["z"], sy_n, qp_["z"], classes, 5, True, 2.0)
                    rec("norm=zscore", p, sc)
                for tag, vals in bucket.items():
                    rows.append({"domain": name, "split": fi, "ratio": ratio, "knob": tag,
                                 "accuracy": float(np.mean([v[0] for v in vals])),
                                 "auroc": float(np.mean([v[1] for v in vals]))})
            print(f"  {name} split {fi} done", flush=True)

    (out_dir / f"sens_{args.tag}.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")

    print(f"\n=== sensitivity ({time.time()-t0:.0f}s, episodes={args.episodes}) ===")
    for dom in ("EuroSAT_RGB", "DTD"):
        for ratio in RATIOS:
            print(f"\n{dom} noise={ratio}")
            print(f"{'knob':14s}{'ACC':>9s}{'AUROC':>9s}")
            knobs = sorted({r["knob"] for r in rows if r["domain"] == dom and r["ratio"] == ratio})
            for kb in knobs:
                sel = [r for r in rows if r["domain"] == dom and r["ratio"] == ratio and r["knob"] == kb]
                print(f"{kb:14s}{np.mean([x['accuracy'] for x in sel]):9.4f}"
                      f"{np.mean([x['auroc'] for x in sel]):9.4f}")


if __name__ == "__main__":
    main()
