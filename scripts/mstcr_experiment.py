"""MS-TCR: multi-scale transductive centroid re-estimation.

Final method, assembled from what the probes actually supported and nothing else.

Components, each with the measurement that justifies it:
  1. unit-normalised concatenation of layer2/3/4
     layer4 has ~8x the norm of layer2, so plain concatenation is dominated by
     the deepest view (raw concat: 0.7692 vs normalised 0.8061 at 30% noise).
  2. transductive centre re-estimation over support+query
     the query set is 3x the support set, so centres estimated from all 100 rows
     carry far less label noise than centres from 25 (probe 1: +0.065).
  3. OOD down-weighting during re-estimation, thresholded without knowing the
     OOD rate. Detected by distance to nearest centre, cut at median + k * MAD,
     so the method is never told that 20% of queries are out-of-distribution.
  4. OOD score read in the concatenated space (AUROC 0.92 vs 0.74 in layer4).

Explicitly NOT included: label cleaning, sample pruning, label correction.
Probe 3 and 4 showed all three are harmful at this shot count, because the
support set cannot identify its own corrupted rows (precision ceiling 0.5-0.6).
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

from oracle_probe import N_WAY, K_SHOT, N_QUERY, RATIOS, split_indices  # noqa: E402

ROOT = HERE.parent
MS = ROOT / "multiscale_output"
VIEWS = ("l2", "l3", "l4")
OOD_GRID = (0.0, 0.2)
ROUNDS = 5
MAD_K = 2.0


def corrupt(sy, ratio, rng, max_tries=200):
    """Flip a fraction of support labels, never emptying a class.

    Retrying matters for correctness, not cosmetics: with 5 shots and a flip
    rate of 0.5 an entire class can lose all five rows. Its prototype then has
    no members, and np.argmin treats the resulting NaN column as the smallest
    value, so every query in that episode silently receives the same label.
    Measured incidence before the fix was 0.2%-1.0% of episodes at ratios
    0.3-0.5 and 0 below that. Redrawing only the flip set (the episode itself is
    unchanged) keeps the corruption distribution as it was while ensuring the
    task stays a well-defined 5-way task.
    """
    sy2 = sy.copy()
    flipped = np.zeros(len(sy), dtype=bool)
    if ratio <= 0:
        return sy2, flipped
    n = max(1, int(round(len(sy) * ratio)))
    classes = np.unique(sy)
    for _ in range(max_tries):
        sy2 = sy.copy()
        flipped = np.zeros(len(sy), dtype=bool)
        for i in rng.choice(len(sy), size=n, replace=False):
            sy2[i] = rng.choice(classes[classes != sy[i]])
            flipped[i] = True
        if all((sy2 == c).any() for c in classes):
            return sy2, flipped
    raise RuntimeError(f"could not place {n} flips without emptying a class")


def unit(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def sqdist(x, m):
    """Squared Euclidean distances, expanded so the work goes through dgemm.

    The broadcast form ((x[:,None,:] - m[None,:,:])**2).sum(2) allocates an
    (n, k, d) temporary; at d = 896 and 100 rows that dominates the run time of
    every arm. The expansion is algebraically identical and clips the small
    negative values that cancellation can produce.
    """
    d = (x * x).sum(1)[:, None] + (m * m).sum(1)[None, :] - 2.0 * (x @ m.T)
    return np.maximum(d, 0.0)


def draw_episode(ty, rng):
    classes = np.unique(ty)
    chosen = rng.choice(classes, size=N_WAY, replace=False)
    s_idx, q_idx = [], []
    for c in chosen:
        pool = np.flatnonzero(ty == c)
        perm = rng.permutation(pool)
        s_idx.append(perm[:K_SHOT])
        q_idx.append(perm[K_SHOT:K_SHOT + N_QUERY])
    return np.concatenate(s_idx), np.concatenate(q_idx), ty[np.concatenate(s_idx)]


def proto(sx, sy, qx):
    classes = np.unique(sy)
    m = np.stack([sx[sy == c].mean(0) for c in classes])
    d = sqdist(qx, m)
    return classes[d.argmin(1)], d.min(1)


def tcr(sx, sy, qx, classes, rounds=ROUNDS, ood_aware=False, mad_k=MAD_K):
    """Transductive centroid re-estimation with optional OOD down-weighting."""
    m = np.stack([sx[sy == c].mean(0) for c in classes])
    allx = np.vstack([sx, qx])
    n_sup = len(sx)
    for _ in range(rounds):
        d = sqdist(allx, m)
        lab = d.argmin(1)
        w = np.ones(len(allx))
        if ood_aware:
            s = d.min(1)
            med = np.median(s)
            mad = np.median(np.abs(s - med)) + 1e-12
            far = s > med + mad_k * mad
            w[far] = 0.1
            w[:n_sup] = 1.0     # never down-weight the labelled rows
        for k in range(len(classes)):
            sel = lab == k
            if sel.any():
                m[k] = np.average(allx[sel], axis=0, weights=w[sel])
    d = sqdist(qx, m)
    return classes[d.argmin(1)], d.min(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--tag", default="smoke30")
    args = ap.parse_args()
    t0 = time.time()
    out = ROOT / "mstcr_output"
    out.mkdir(exist_ok=True)

    acc_rows, per_split = [], []
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
            tv["raw"] = np.hstack([tv[k] for k in VIEWS])
            ty = y[idx]
            rng = np.random.default_rng(111_000 + fi)

            for ratio in RATIOS:
                for ood_r in OOD_GRID:
                    bucket: dict[str, list[tuple[float, float]]] = {}
                    for _ in range(args.episodes):
                        s, q, sy = draw_episode(ty, rng)
                        qy = ty[q]
                        sy_n, _ = corrupt(sy, ratio, rng)
                        classes = np.unique(sy)
                        qx_idx, qy_use = q, qy
                        is_ood = np.zeros(len(q), dtype=bool)
                        if ood_r > 0:
                            outside = np.flatnonzero(~np.isin(ty, classes))
                            n_ood = max(1, int(round(len(q) * ood_r)))
                            pick = rng.choice(outside, size=min(n_ood, len(outside)), replace=False)
                            qx_idx = np.concatenate([q, pick])
                            qy_use = np.concatenate([qy, np.full(len(pick), -1)])
                            is_ood = np.concatenate([is_ood, np.ones(len(pick), dtype=bool)])

                        def rec(tag, pred, score):
                            clean = ~is_ood
                            a = float((pred[clean] == qy_use[clean]).mean())
                            au = np.nan
                            if is_ood.any():
                                lb = np.concatenate([np.zeros(clean.sum()), np.ones(is_ood.sum())])
                                sc = np.concatenate([score[clean], score[is_ood]])
                                try:
                                    r = sps.rankdata(sc)[lb == 1].mean()
                                    au = float((r - (is_ood.sum() + 1) / 2) / clean.sum())
                                except Exception:
                                    au = np.nan
                            bucket.setdefault(tag, []).append((a, au))

                        sp_ = {k: tv[k][s] for k in tv}
                        qp_ = {k: tv[k][qx_idx] for k in tv}

                        p, sc = proto(sp_["l4"], sy_n, qp_["l4"])
                        rec("proto_l4", p, sc)
                        p, sc = tcr(sp_["l4"], sy_n, qp_["l4"], classes)
                        rec("tcr_l4", p, sc)
                        p, sc = tcr(sp_["raw"], sy_n, qp_["raw"], classes)
                        rec("tcr_raw", p, sc)
                        p, sc = tcr(sp_["cat"], sy_n, qp_["cat"], classes)
                        rec("mstcr", p, sc)
                        p, sc = tcr(sp_["cat"], sy_n, qp_["cat"], classes, ood_aware=True)
                        rec("mstcr_ood", p, sc)

                    for tag, vals in bucket.items():
                        a = float(np.mean([v[0] for v in vals]))
                        au = [v[1] for v in vals if np.isfinite(v[1])]
                        acc_rows.append({"domain": name, "split": fi, "ratio": ratio,
                                         "ood": ood_r, "method": tag, "accuracy": a,
                                         "auroc": float(np.mean(au)) if au else None,
                                         "n": len(vals)})
            print(f"  {name} split {fi} done", flush=True)

    (out / f"mstcr_{args.tag}.json").write_text(json.dumps(acc_rows, indent=1), encoding="utf-8")

    methods = ["proto_l4", "tcr_l4", "tcr_raw", "mstcr", "mstcr_ood"]
    print(f"\n=== ACC / AUROC  ({time.time()-t0:.0f}s, episodes={args.episodes}) ===")
    for dom in ("EuroSAT_RGB", "DTD"):
        for ood_r in OOD_GRID:
            print(f"\n{dom}  ood={ood_r}")
            print(f"{'method':11s}" + "".join(f"{r:>12.1f}" for r in RATIOS))
            for m in methods:
                line = f"{m:11s}"
                for ratio in RATIOS:
                    sel = [x for x in acc_rows if x["domain"] == dom and x["method"] == m
                           and x["ratio"] == ratio and x["ood"] == ood_r]
                    if not sel:
                        line += f"{'-':>12s}"
                        continue
                    a = np.mean([x["accuracy"] for x in sel])
                    au = [x["auroc"] for x in sel if x["auroc"] is not None]
                    line += f"{a:7.4f}" + (f"/{np.mean(au):.3f}" if au else "      ")
                print(line)


if __name__ == "__main__":
    main()
