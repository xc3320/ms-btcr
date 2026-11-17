"""Sensitivity of the final rule to its three knobs, ablated one at a time.

k      : MAD multiplier in the decay threshold, median + k*MAD
rounds : re-estimation iterations
norm   : how the three views are put on a common scale before concatenation
         (unit = divide by row norm, raw = plain concatenation, zscore = per
         dimension standardisation)

Only the ResNet-18 cache is used. Sensitivity is a property of the inference
rule, not of the backbone, and the backbone question is answered separately by
the three-backbone suite.

30 episodes per cell: enough to see a knob actually doing something (the grid
spacing between settings is much larger than the standard error at 30 episodes),
while the headline numbers keep the full 100.
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

from mstcr_experiment import (MAD_K, ROUNDS, corrupt, draw_episode,  # noqa: E402
                              sqdist, unit)
from oracle_probe import split_indices  # noqa: E402
from suite_main import balanced_assign  # noqa: E402

ROOT = HERE.parent
MS = ROOT / "multiscale_output"
KEYS = ("x2", "x3", "x4")
RATIOS = (0.0, 0.3)
OOD_R = 0.2
K_GRID = (1.0, 1.5, 2.0, 2.5, 3.0, 4.0)
R_GRID = (1, 2, 3, 5, 8, 12)


def bal_soft(sx, sy, qx, classes, rounds=ROUNDS, mad_k=MAD_K):
    m = np.stack([sx[sy == c].mean(0) for c in classes])
    allx = np.vstack([sx, qx])
    n_sup = len(sx)
    for _ in range(rounds):
        dq = sqdist(qx, m)
        lab = balanced_assign(dq)
        s = dq.min(1)
        med = np.median(s)
        mad = np.median(np.abs(s - med)) + 1e-12
        wq = np.exp(-np.maximum(s - (med + mad_k * mad), 0.0) / mad)
        w = np.concatenate([np.ones(n_sup), wq])
        for c in range(len(classes)):
            sel = np.concatenate([sy == classes[c], lab == c])
            if sel.any():
                m[c] = np.average(allx[sel], axis=0, weights=w[sel])
    d = sqdist(qx, m)
    return classes[d.argmin(1)], d.min(1)


def zscore(x):
    return (x - x.mean(0, keepdims=True)) / (x.std(0, keepdims=True) + 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--tag", default="main")
    a = ap.parse_args()
    t0 = time.time()
    rows = []

    for name, short in (("EuroSAT_RGB", "eurosat"), ("DTD", "dtd")):
        z = np.load(MS / f"{short}_multiscale.npz", allow_pickle=False)
        views = [z[k].astype(np.float64) for k in KEYS]
        key2i = {str(k): i for i, k in enumerate(z["keys"])}
        y = z["y"]

        for fi in range(1, 11):
            idx = split_indices(name, fi, y, key2i)
            v = [q[idx] for q in views]
            space = {
                "unit": np.hstack([unit(q) for q in v]),
                "raw": np.hstack(v),
                "zscore": np.hstack([zscore(q) for q in v]),
            }
            ty = y[idx]
            rng = np.random.default_rng(111_000 + fi)

            for ratio in RATIOS:
                for space_name in ("unit", "raw", "zscore"):
                    bucket: dict[str, list[tuple[float, float]]] = {}
                    for _ in range(a.episodes):
                        s, q, sy = draw_episode(ty, rng)
                        qy = ty[q]
                        sy_n, _ = corrupt(sy, ratio, rng)
                        classes = np.unique(sy)
                        qx_idx, qy_use = q, qy
                        is_ood = np.zeros(len(q), dtype=bool)
                        outside = np.flatnonzero(~np.isin(ty, classes))
                        n_ood = max(1, int(round(len(q) * OOD_R)))
                        pick = rng.choice(outside, size=min(n_ood, len(outside)),
                                          replace=False)
                        qx_idx = np.concatenate([q, pick])
                        qy_use = np.concatenate([qy, np.full(len(pick), -1)])
                        is_ood = np.concatenate([is_ood, np.ones(len(pick), dtype=bool)])

                        sx, qx = space[space_name][s], space[space_name][qx_idx]

                        def rec(tag, pred, score):
                            clean = ~is_ood
                            acc = float((pred[clean] == qy_use[clean]).mean())
                            lb = np.concatenate([np.zeros(clean.sum()),
                                                 np.ones(is_ood.sum())])
                            sc = np.concatenate([score[clean], score[is_ood]])
                            r = sps.rankdata(sc)[lb == 1].mean()
                            au = float((r - (is_ood.sum() + 1) / 2) / clean.sum())
                            bucket.setdefault(tag, []).append((acc, au))

                        if space_name == "unit":     # only meaningful there
                            for k in K_GRID:
                                p, sc = bal_soft(sx, sy_n, qx, classes, mad_k=k)
                                rec(f"k={k:g}", p, sc)
                            for r_ in R_GRID:
                                p, sc = bal_soft(sx, sy_n, qx, classes, rounds=r_)
                                rec(f"rounds={r_}", p, sc)
                        else:
                            p, sc = bal_soft(sx, sy_n, qx, classes)
                            rec(f"space={space_name}", p, sc)

                    for k, vals in bucket.items():
                        rows.append({"domain": name, "split": fi, "ratio": ratio,
                                     "knob": k,
                                     "accuracy": float(np.mean([x[0] for x in vals])),
                                     "auroc": float(np.mean([x[1] for x in vals])),
                                     "n": len(vals)})
            print(f"  {name} split {fi} done {time.time()-t0:.0f}s", flush=True)

    out = ROOT / "suite_output" / f"sensitivity_{a.tag}.json"
    out.write_text(json.dumps(rows, indent=1), encoding="utf-8")

    print(f"\n=== sensitivity ({a.episodes} episodes, ood={OOD_R}) ===")
    for dom in ("EuroSAT_RGB", "DTD"):
        for ratio in RATIOS:
            print(f"\n{dom}  noise={ratio}")
            print(f"{'setting':14s} {'acc':>8s} {'auroc':>8s}")
            for k in [f"k={x:g}" for x in K_GRID] + [f"rounds={x}" for x in R_GRID] \
                    + ["space=unit", "space=raw", "space=zscore"]:
                v = [x for x in rows if x["domain"] == dom and x["ratio"] == ratio
                     and x["knob"] == k]
                if not v:
                    continue
                print(f"{k:14s} {np.mean([x['accuracy'] for x in v]):8.4f} "
                      f"{np.mean([x['auroc'] for x in v]):8.3f}")


if __name__ == "__main__":
    main()
