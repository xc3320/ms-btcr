"""Single main suite: every arm the paper reports, one random stream, one grid.

Why one script instead of the earlier separate runs: the balanced-assignment
baseline turned out to beat the multi-scale transductive method on accuracy in
almost every cell, so the final method has to be redefined as the combination
and every arm has to be measured side by side under identical episodes. Running
the arms in separate processes made that comparison needlessly fragile.

Arms
  proto_deep  single-scale prototypical rule on the deepest view
  tcr_deep    transductive centre re-estimation, deepest view only
  tcr_raw     multi-scale, unit-normalisation removed (ablation)
  mstcr       multi-scale, unit-normalised concatenation, free re-estimation
  mstcr_ood   mstcr plus median + k*MAD down-weighting of far rows
  bal_iter    balanced assignment over all query rows, re-estimated (baseline)
  obtcr       proposed: gate out-of-distribution rows, then exact balanced
              assignment over the gated inliers, re-estimated

Balanced assignment is solved exactly as an integer transportation problem via
linear_sum_assignment on a column-expanded cost matrix (each class expanded to
its quota of slots). At equal quotas this is identical to the fractional-supply
linear programme used before - dividing every constraint by the row supply
recovers the same polytope - but it is two orders of magnitude faster, which is
what makes a seven-arm suite affordable.

Out: suite_output/{backbone}_{tag}.json, one row per
(backbone, domain, split, ratio, ood, method).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats as sps
from scipy.optimize import linear_sum_assignment

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from mstcr_experiment import (MAD_K, OOD_GRID, ROUNDS, corrupt, draw_episode,  # noqa: E402
                              proto, sqdist, tcr, unit)
from oracle_probe import RATIOS, split_indices  # noqa: E402

ROOT = HERE.parent
OUT = ROOT / "suite_output"
ARMS = ("proto_deep", "tcr_deep", "tcr_raw", "mstcr", "mstcr_ood",
        "bal_iter", "bal_ood", "bal_soft", "obtcr")
FILES = {
    "resnet18": (ROOT / "multiscale_output" / "eurosat_multiscale.npz",
                 ROOT / "multiscale_output" / "dtd_multiscale.npz",
                 ROOT / "backbone_output" / "miniimagenet_resnet18.npz",
                 ("x2", "x3", "x4")),
    "resnet50": (ROOT / "backbone_output" / "eurosat_resnet50.npz",
                 ROOT / "backbone_output" / "dtd_resnet50.npz",
                 ROOT / "backbone_output" / "miniimagenet_resnet50.npz",
                 ("x1", "x2", "x3")),
    "vit_b_16": (ROOT / "backbone_output" / "eurosat_vit_b_16.npz",
                 ROOT / "backbone_output" / "dtd_vit_b_16.npz",
                 ROOT / "backbone_output" / "miniimagenet_vit_b_16.npz",
                 ("x1", "x2", "x3")),
}


def quotas(n: int, k: int) -> np.ndarray:
    """Split n rows over k classes as evenly as possible, remainder to the first."""
    base, rem = divmod(n, k)
    q = np.full(k, base, dtype=int)
    q[:rem] += 1
    return q


def balanced_assign(cost: np.ndarray) -> np.ndarray:
    """Exact min-cost assignment of rows to classes with equal per-class quotas.

    cost : (n_rows, n_classes). Returns an integer label per row.
    """
    n, k = cost.shape
    q = quotas(n, k)
    cols = np.repeat(np.arange(k), q)          # one column per available slot
    big = cost[:, cols]                        # (n, n) square
    r, c = linear_sum_assignment(big)
    lab = np.empty(n, dtype=int)
    lab[r] = cols[c]
    return lab


def bal_iter(sx, sy, qx, classes, rounds=ROUNDS):
    """Balanced assignment over every query row, prototypes re-estimated."""
    m = np.stack([sx[sy == c].mean(0) for c in classes])
    allx = np.vstack([sx, qx])
    for _ in range(rounds):
        dq = sqdist(qx, m)
        lab = balanced_assign(dq)
        for k in range(len(classes)):
            sel = np.concatenate([sy == classes[k], lab == k])
            rows = allx[sel]
            if len(rows):
                m[k] = rows.mean(0)
    d = sqdist(qx, m)
    return classes[d.argmin(1)], d.min(1)


def bal_ood(sx, sy, qx, classes, rounds=ROUNDS, mad_k=MAD_K):
    """Balanced assignment over every query row, with far rows down-weighted.

    Sits between bal_iter and obtcr: the assignment still covers all rows, so
    the balanced structure is never broken, but the rows the MAD gate rejects
    contribute a tenth of a vote to the class mean. Excluding them outright, as
    obtcr does, changes which rows exist; down-weighting them only changes how
    much they count.
    """
    m = np.stack([sx[sy == c].mean(0) for c in classes])
    allx = np.vstack([sx, qx])
    n_sup = len(sx)
    for _ in range(rounds):
        dq = sqdist(qx, m)
        lab = balanced_assign(dq)
        s = dq.min(1)
        med = np.median(s)
        mad = np.median(np.abs(s - med)) + 1e-12
        wq = np.where(s > med + mad_k * mad, 0.1, 1.0)
        w = np.concatenate([np.ones(n_sup), wq])
        for k in range(len(classes)):
            sel = np.concatenate([sy == classes[k], lab == k])
            if sel.any():
                m[k] = np.average(allx[sel], axis=0, weights=w[sel])
    d = sqdist(qx, m)
    return classes[d.argmin(1)], d.min(1)


def bal_soft(sx, sy, qx, classes, rounds=ROUNDS, mad_k=MAD_K):
    """Balanced assignment with a continuous, hinge-decayed OOD weight.

    bal_ood drops every row beyond median + k*MAD to a tenth of a vote, which
    is a cliff: in a closed query set the gate still fires on the tail of the
    clean distance distribution and those rows are penalised for nothing.
    Decaying the weight from 1 at the threshold instead of stepping it down
    keeps near-threshold rows almost intact while rows far beyond it still
    lose their influence, so the closed-set condition should not pay for the
    open-set protection.
    """
    m = np.stack([sx[sy == c].mean(0) for c in classes])
    allx = np.vstack([sx, qx])
    n_sup = len(sx)
    for _ in range(rounds):
        dq = sqdist(qx, m)
        lab = balanced_assign(dq)
        s = dq.min(1)
        med = np.median(s)
        mad = np.median(np.abs(s - med)) + 1e-12
        excess = np.maximum(s - (med + mad_k * mad), 0.0)
        wq = np.exp(-excess / mad)
        w = np.concatenate([np.ones(n_sup), wq])
        for k in range(len(classes)):
            sel = np.concatenate([sy == classes[k], lab == k])
            if sel.any():
                m[k] = np.average(allx[sel], axis=0, weights=w[sel])
    d = sqdist(qx, m)
    return classes[d.argmin(1)], d.min(1)


def obtcr(sx, sy, qx, classes, rounds=ROUNDS, mad_k=MAD_K):
    """Gate far rows, then balance only over the gated inliers.

    The gate is the same median + k*MAD rule as mstcr_ood, so the two arms share
    one mechanism; what changes is what happens to the rows it keeps. Balancing
    the inliers replaces the free per-row assignment, which removes the class
    collapse that free re-estimation allows, while rows the gate rejects are
    never forced into a class the way a plain balanced rule would force them.
    """
    m = np.stack([sx[sy == c].mean(0) for c in classes])
    for _ in range(rounds):
        dq = sqdist(qx, m)
        s = dq.min(1)
        med = np.median(s)
        mad = np.median(np.abs(s - med)) + 1e-12
        keep = s <= med + mad_k * mad
        if keep.sum() < 2 * len(classes):     # gate too aggressive, keep all
            keep = np.ones(len(qx), dtype=bool)
        qi = qx[keep]
        lab = balanced_assign(sqdist(qi, m))
        for k in range(len(classes)):
            sup = sx[sy == classes[k]]
            sel = qi[lab == k]
            rows = np.vstack([sup, sel]) if len(sel) else sup
            if len(rows):
                m[k] = rows.mean(0)
    d = sqdist(qx, m)
    return classes[d.argmin(1)], d.min(1)


def run(backbone: str, episodes: int, tag: str, only: str | None = None,
        mini_file: str | None = None) -> None:
    t0 = time.time()
    rows = []
    keys = FILES[backbone][3]

    doms = [("EuroSAT_RGB", FILES[backbone][0]), ("DTD", FILES[backbone][1]),
            ("Mini", Path(mini_file) if mini_file else FILES[backbone][2])]
    if only:
        doms = [d for d in doms if d[0].startswith(only)]

    for name, f in doms:
        z = np.load(f, allow_pickle=False)
        views = [z[k].astype(np.float64) for k in keys]
        key2i = {str(k): i for i, k in enumerate(z["keys"])}
        y = z["y"]

        for fi in range(1, 11):
            idx = split_indices(name, fi, y, key2i)
            v = [a[idx] for a in views]
            space = {"deep": v[-1],
                     "raw": np.hstack(v),
                     "cat": np.hstack([unit(a) for a in v])}
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

                        sp_ = {k: space[k][s] for k in space}
                        qp_ = {k: space[k][qx_idx] for k in space}

                        p, sc = proto(sp_["deep"], sy_n, qp_["deep"])
                        rec("proto_deep", p, sc)
                        p, sc = tcr(sp_["deep"], sy_n, qp_["deep"], classes)
                        rec("tcr_deep", p, sc)
                        p, sc = tcr(sp_["raw"], sy_n, qp_["raw"], classes)
                        rec("tcr_raw", p, sc)
                        p, sc = tcr(sp_["cat"], sy_n, qp_["cat"], classes)
                        rec("mstcr", p, sc)
                        p, sc = tcr(sp_["cat"], sy_n, qp_["cat"], classes, ood_aware=True)
                        rec("mstcr_ood", p, sc)
                        p, sc = bal_iter(sp_["cat"], sy_n, qp_["cat"], classes)
                        rec("bal_iter", p, sc)
                        p, sc = bal_ood(sp_["cat"], sy_n, qp_["cat"], classes)
                        rec("bal_ood", p, sc)
                        p, sc = bal_soft(sp_["cat"], sy_n, qp_["cat"], classes)
                        rec("bal_soft", p, sc)
                        p, sc = obtcr(sp_["cat"], sy_n, qp_["cat"], classes)
                        rec("obtcr", p, sc)

                    for k, vals in bucket.items():
                        au = [x[1] for x in vals if np.isfinite(x[1])]
                        rows.append({"backbone": backbone, "domain": name,
                                     "split": fi, "ratio": ratio, "ood": ood_r,
                                     "method": k,
                                     "accuracy": float(np.mean([x[0] for x in vals])),
                                     "auroc": float(np.mean(au)) if au else None,
                                     "n": len(vals)})
            print(f"  {name} split {fi} done {time.time()-t0:.0f}s", flush=True)

    OUT.mkdir(exist_ok=True)
    out = OUT / f"{backbone}_{tag}.json"
    out.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(f"saved {out}")

    for dom in (d[0] for d in doms):
        for ood_r in OOD_GRID:
            print(f"\n{backbone}  {dom}  ood={ood_r}")
            print(f"{'arm':12s}" + "".join(f"{r:>13.1f}" for r in RATIOS))
            for a in ARMS:
                line = f"{a:12s}"
                for ratio in RATIOS:
                    sel = [x for x in rows if x["domain"] == dom and x["method"] == a
                           and x["ratio"] == ratio and x["ood"] == ood_r]
                    acc = np.mean([x["accuracy"] for x in sel])
                    au = [x["auroc"] for x in sel if x["auroc"] is not None]
                    line += f"{acc:7.4f}" + (f"/{np.mean(au):.3f}" if au else "      ")
                print(line)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="resnet18", choices=sorted(FILES))
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--tag", default="main")
    ap.add_argument("--only", default=None,
                    help="restrict to one domain, e.g. Mini; writes <backbone>_<tag>.json")
    ap.add_argument("--mini-file", default=None,
                    help="override the Mini cache, e.g. the 84x84 features")
    a = ap.parse_args()
    run(a.backbone, a.episodes, a.tag, a.only, a.mini_file)
