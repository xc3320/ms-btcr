"""Balanced-assignment transductive baseline (exact transportation solve).

Why this baseline and not a re-implementation of TIM or PT-MAP: both are
optimisation procedures with several free choices (temperature, initialisation,
stopping rule), and a loose re-implementation would either flatter or unfairly
damage this paper's method. Balanced assignment has no free parameter and a
unique optimum, so the comparison is unambiguous and repeatable.

The protocol makes this baseline dangerous on purpose: the query set is
balanced by construction (15 per class), which is exactly the structure a
capacity-constrained assignment exploits. If enforcing balance already buys the
whole gain, MS-TCR adds nothing.

Formulation. With d_ik the squared distance from query i to prototype k, solve

    min  sum_ik c_ik y_ik
    s.t. sum_k y_ik = 1            (every query takes one label)
         sum_i y_ik = n_q / n_cls   (every class receives its quota)

which is a transportation problem; the constraint matrix is totally unimodular,
so the LP optimum is integral and no rounding is needed. When out-of-distribution
queries are present the quotas are scaled by (clean queries / all queries), so
each row ships a fraction of its unit mass and the solution stays feasible
without being told which rows are out of distribution.

Two arms:
  bal_assign  one exact balanced assignment to the support-only prototypes
  bal_iter    three rounds of (assign -> re-estimate prototypes from the
              assignment), i.e. the same iteration budget family as MS-TCR but
              with balance enforced instead of OOD down-weighting
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats as sps
from scipy.optimize import linprog

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from mstcr_experiment import (OOD_GRID, corrupt, draw_episode, proto,  # noqa: E402
                              tcr, unit)
from oracle_probe import RATIOS, split_indices  # noqa: E402

ROOT = HERE.parent
# (EuroSAT cache, DTD cache, view keys). The ResNet-18 cache predates the
# backbone runs and names its views after the ResNet stages; the backbone
# caches use positional names. Both are mapped to a common order here.
SRC = {
    "resnet18": (ROOT / "multiscale_output" / "eurosat_multiscale.npz",
                 ROOT / "multiscale_output" / "dtd_multiscale.npz",
                 ("x2", "x3", "x4")),
    "resnet50": (ROOT / "backbone_output" / "eurosat_resnet50.npz",
                 ROOT / "backbone_output" / "dtd_resnet50.npz",
                 ("x1", "x2", "x3")),
    "vit_b_16": (ROOT / "backbone_output" / "eurosat_vit_b_16.npz",
                 ROOT / "backbone_output" / "dtd_vit_b_16.npz",
                 ("x1", "x2", "x3")),
}


def balanced_assign(cost: np.ndarray, quota: float, supply: float = 1.0) -> np.ndarray:
    """Exact capacity-constrained assignment of rows to columns.

    cost   : (n_rows, n_cols) non-negative distances
    quota  : mass each column must receive
    supply : mass each row ships; supply * n_rows must equal quota * n_cols.
             Values below 1 are used when some rows may be out of distribution:
             every row still ships the same fraction, so the solve never needs
             to know which rows those are.
    """
    n, k = cost.shape
    c = cost.reshape(-1)
    A_eq, b_eq = [], []
    for i in range(n):                      # row constraints
        row = np.zeros(n * k)
        row[i * k:(i + 1) * k] = 1.0
        A_eq.append(row)
        b_eq.append(supply)
    for j in range(k):                      # column quotas
        col = np.zeros(n * k)
        col[j::k] = 1.0
        A_eq.append(col)
        b_eq.append(quota)
    res = linprog(c, A_eq=np.array(A_eq), b_eq=np.array(b_eq),
                  bounds=[(0, 1)] * (n * k), method="highs")
    if not res.success:
        raise RuntimeError(f"transportation solve failed: {res.message}")
    return res.x.reshape(n, k)


def run(features: str, episodes: int, tag: str) -> None:
    t0 = time.time()
    rows = []
    for name, f in (("EuroSAT_RGB", SRC[features][0]), ("DTD", SRC[features][1])):
        keys = SRC[features][2]
        z = np.load(f, allow_pickle=False)
        views = [z[k].astype(np.float64) for k in keys]
        key2i = {str(k): i for i, k in enumerate(z["keys"])}
        y = z["y"]

        for fi in range(1, 11):
            idx = split_indices(name, fi, y, key2i)
            v = [a[idx] for a in views]
            deep = v[-1]
            cat = np.hstack([unit(a) for a in v])
            ty = y[idx]
            rng = np.random.default_rng(111_000 + fi)

            for ratio in RATIOS:
                for ood_r in OOD_GRID:
                    bucket: dict[str, list[tuple[float, float]]] = {}
                    for _ in range(episodes):
                        s, q, sy = draw_episode(ty, rng)
                        qy = ty[q]
                        sy_n, _ = corrupt(sy, ratio, rng)
                        classes = np.unique(sy)
                        qx_idx, qy_use = q, qy
                        is_ood = np.zeros(len(q), dtype=bool)
                        if ood_r > 0:
                            outside = np.flatnonzero(~np.isin(ty, classes))
                            n_ood = max(1, int(round(len(q) * ood_r)))
                            pick = rng.choice(outside, size=min(n_ood, len(outside)),
                                              replace=False)
                            qx_idx = np.concatenate([q, pick])
                            qy_use = np.concatenate([qy, np.full(len(pick), -1)])
                            is_ood = np.concatenate([is_ood,
                                                     np.ones(len(pick), dtype=bool)])

                        def rec(tag_, pred, score):
                            clean = ~is_ood
                            a = float((pred[clean] == qy_use[clean]).mean())
                            au = np.nan
                            if is_ood.any():
                                lb = np.concatenate([np.zeros(clean.sum()),
                                                     np.ones(is_ood.sum())])
                                sc = np.concatenate([score[clean], score[is_ood]])
                                r = sps.rankdata(sc)[lb == 1].mean()
                                au = float((r - (is_ood.sum() + 1) / 2) / clean.sum())
                            bucket.setdefault(tag_, []).append((a, au))

                        # column quota in "clean-query units"; supply is the
                        # same fraction for every row so total mass balances
                        quota = len(q) / len(classes)
                        supply = len(q) / len(qx_idx)

                        for tag_name, space in (("bal_assign", cat),
                                                ("bal_deep", deep)):
                            sxp, qxp = space[s], space[qx_idx]
                            m = np.stack([sxp[sy_n == c].mean(0) for c in classes])
                            d = ((qxp[:, None, :] - m[None, :, :]) ** 2).sum(2)
                            assign = balanced_assign(d, quota, supply)
                            pred = classes[assign.argmax(1)]
                            score = (assign * d).sum(1)      # cost of chosen column
                            rec(tag_name, pred, score)

                        # same loop, but prototypes re-estimated each round
                        sxp, qxp = cat[s], cat[qx_idx]
                        m = np.stack([sxp[sy_n == c].mean(0) for c in classes])
                        allx = np.vstack([sxp, qxp])
                        for _r in range(3):
                            d = ((allx[:, None, :] - m[None, :, :]) ** 2).sum(2)
                            dq = d[len(sxp):]
                            assign = balanced_assign(dq, quota, supply)
                            lab = assign.argmax(1)
                            for k in range(len(classes)):
                                sel = np.concatenate([sy_n == classes[k], lab == k])
                                if sel.any():
                                    m[k] = allx[sel].mean(0)
                        d = ((qxp[:, None, :] - m[None, :, :]) ** 2).sum(2)
                        rec("bal_iter", classes[d.argmin(1)], d.min(1))

                        # reference arms, identical stream, for a paired comparison
                        p, sc = proto(deep[s], sy_n, deep[qx_idx])
                        rec("proto_deep", p, sc)
                        p, sc = tcr(cat[s], sy_n, cat[qx_idx], classes, ood_aware=True)
                        rec("mstcr_ood", p, sc)

                    for k, vals in bucket.items():
                        au = [x[1] for x in vals if np.isfinite(x[1])]
                        rows.append({"backbone": features, "domain": name,
                                     "split": fi, "ratio": ratio, "ood": ood_r,
                                     "method": k,
                                     "accuracy": float(np.mean([x[0] for x in vals])),
                                     "auroc": float(np.mean(au)) if au else None,
                                     "n": len(vals)})
            print(f"  {name} split {fi} done {time.time()-t0:.0f}s", flush=True)

    out = ROOT / "backbone_output" / f"balassign_{features}_{tag}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(f"saved {out.name}")

    print(f"\n=== {features}  ACC/AUROC ===")
    for dom in ("EuroSAT_RGB", "DTD"):
        for ood_r in OOD_GRID:
            print(f"\n{dom}  ood={ood_r}")
            print(f"{'method':12s}" + "".join(f"{r:>13.1f}" for r in RATIOS))
            for meth in ("proto_deep", "bal_deep", "bal_assign", "bal_iter",
                         "mstcr_ood"):
                line = f"{meth:12s}"
                for ratio in RATIOS:
                    sel = [x for x in rows if x["domain"] == dom
                           and x["method"] == meth and x["ratio"] == ratio
                           and x["ood"] == ood_r]
                    a = np.mean([x["accuracy"] for x in sel])
                    au = [x["auroc"] for x in sel if x["auroc"] is not None]
                    line += f"{a:7.4f}" + (f"/{np.mean(au):.3f}" if au else "      ")
                print(line)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="resnet18", choices=sorted(SRC))
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--tag", default="main")
    a = ap.parse_args()
    run(a.features, a.episodes, a.tag)
