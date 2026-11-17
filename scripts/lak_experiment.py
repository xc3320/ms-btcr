"""LAK: Label-Anchored K-means for noisy transductive few-shot classification.

Why this exists
---------------
Probes established three negative results and one positive one:
  - support-set leave-one-out weighting recovers ~10% of the oracle headroom  (probe 3)
  - hard support-set pruning is worse than doing nothing                      (probe 3)
  - iterated hard label correction drifts and loses accuracy                  (probe 4)
  - one round of query-set clustering with support prototypes as initial
    centres recovers most of the oracle headroom (EuroSAT +0.061 at 30% noise)

The last result works, but only for exactly one round. With more rounds the
support labels stop influencing the centres and the procedure degenerates into
plain unsupervised k-means on the query set, which misaligns classes. So the
useful ingredient is not "correction" but "keeping the label information alive
while the centres move toward the query distribution".

LAK makes that explicit. Each round, a centre is a convex combination of
  (a) the mean of all embeddings currently assigned to that group  (data term)
  (b) the mean of the support rows carrying that class label        (anchor term)
with weight alpha. alpha = 1 recovers the prototypical rule (never moves),
alpha = 0 recovers k-means (labels discarded after initialisation). The claim
to test is that an intermediate alpha lets many rounds run without drift, and
beats both endpoints under label noise.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import protocol as P  # noqa: E402
from oracle_probe import SCALE, N_WAY, K_SHOT, N_QUERY, RATIOS, split_indices  # noqa: E402

ROOT = HERE.parent
ALPHAS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
ROUNDS = (1, 3, 5)


def corrupt(sy, ratio, rng):
    sy2 = sy.copy()
    flipped = np.zeros(len(sy), dtype=bool)
    if ratio <= 0:
        return sy2, flipped
    n = max(1, int(round(len(sy) * ratio)))
    classes = np.unique(sy)
    for i in rng.choice(len(sy), size=n, replace=False):
        sy2[i] = rng.choice(classes[classes != sy[i]])
        flipped[i] = True
    return sy2, flipped


def proto_predict(sx, sy, qx):
    classes = np.unique(sy)
    protos = np.stack([sx[sy == c].mean(0) for c in classes])
    d = ((qx[:, None, :] - protos[None, :, :]) ** 2).sum(2)
    return classes[d.argmin(1)]


def snn_predict(sx, sy, qx, scale=SCALE):
    classes = np.unique(sy)
    sx_n = sx / (np.linalg.norm(sx, axis=1, keepdims=True) + 1e-12)
    qx_n = qx / (np.linalg.norm(qx, axis=1, keepdims=True) + 1e-12)
    att = np.exp(scale * (qx_n @ sx_n.T))
    att /= att.sum(1, keepdims=True) + 1e-12
    cls_att = np.stack([att[:, sy == c].sum(1) for c in classes])
    return classes[cls_att.argmax(0)]


def lak_centres(sx, sy, qx, classes, alpha: float, rounds: int):
    """Label-anchored centre updates. Returns centres in `classes` order."""
    anchors = np.stack([sx[sy == c].mean(0) if (sy == c).any() else sx.mean(0) for c in classes])
    m = anchors.copy()
    if alpha >= 1.0 or rounds <= 0:
        return m
    allx = np.vstack([sx, qx])
    for _ in range(rounds):
        lab = ((allx[:, None, :] - m[None, :, :]) ** 2).sum(2).argmin(1)
        for k in range(len(classes)):
            sel = lab == k
            if not sel.any():
                m[k] = anchors[k]
                continue
            m[k] = (1.0 - alpha) * allx[sel].mean(0) + alpha * anchors[k]
    return m


def alignment_cost(centres, sx, sy):
    """Class order in `centres` is already fixed, so no matching is needed."""
    return centres


def run_domain(name, x, y, key2i, n_episodes: int):
    rows = []
    for fi in range(1, 11):
        idx = split_indices(name, fi, y, key2i)
        tx, ty = x[idx], y[idx]
        rng = np.random.default_rng(99_000 + fi)
        for ratio in RATIOS:
            bucket: dict[str, list[float]] = {}
            for _ in range(n_episodes):
                ep = P.make_episode(tx, ty, N_WAY, K_SHOT, N_QUERY, rng)
                sx, qx, qy = ep.support_x, ep.query_x, ep.query_y
                classes = np.unique(ep.support_y)
                sy_n, _ = corrupt(ep.support_y, ratio, rng)

                def rec(tag, pred):
                    bucket.setdefault(tag, []).append(float((pred == qy).mean()))

                rec("proto", proto_predict(sx, sy_n, qx))
                rec("snn", snn_predict(sx, sy_n, qx))

                for alpha in ALPHAS:
                    for r in ROUNDS:
                        m = lak_centres(sx, sy_n, qx, classes, alpha, r)
                        d = ((qx[:, None, :] - m[None, :, :]) ** 2).sum(2)
                        rec(f"lak_a{alpha:g}_r{r}", classes[d.argmin(1)])
            for tag, vals in bucket.items():
                rows.append({"domain": name, "split": fi, "ratio": ratio,
                             "method": tag, "accuracy": float(np.mean(vals)), "n": len(vals)})
        print(f"  {name} split {fi} done", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--tag", default="main")
    args = ap.parse_args()

    t0 = time.time()
    out = ROOT / "lak_output"
    out.mkdir(exist_ok=True)
    rows = []
    for name, path in (
        ("EuroSAT_RGB", ROOT / "full_output" / "eurosat_all_features.npz"),
        ("DTD", ROOT / "full_output" / "dtd_all_features.npz"),
    ):
        d = np.load(path, allow_pickle=True)
        key2i = {str(k): i for i, k in enumerate(d["keys"])}
        print(f"{name}: {d['x'].shape}", flush=True)
        rows += run_domain(name, d["x"].astype(np.float64), d["y"], key2i, args.episodes)

    (out / f"lak_{args.tag}.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")

    methods = sorted({r["method"] for r in rows})
    for dom in ("EuroSAT_RGB", "DTD"):
        print(f"\n{dom}  (episodes={args.episodes}, {time.time()-t0:.0f}s)")
        print(f"{'method':16s}" + "".join(f"{r:>8.1f}" for r in RATIOS))
        for m in methods:
            line = f"{m:16s}"
            for ratio in RATIOS:
                sel = [r["accuracy"] for r in rows if r["domain"] == dom and r["method"] == m and r["ratio"] == ratio]
                line += f"{np.mean(sel):8.4f}" if sel else f"{'-':>8s}"
            print(line)


if __name__ == "__main__":
    main()
