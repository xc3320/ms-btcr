"""QGSC: Query-Guided Support Calibration.

Idea
----
Existing noisy few-shot methods work on the support set alone, or train a model
to be robust. Both are limited here: the support set has 25 rows, too few for
reliable self-validation (probe 3: LOO weighting recovers ~10% of the oracle
headroom). The query set is three times larger and is available at inference
time in the transductive setting.

QGSC uses the query set as an independent view of the task's class structure:
cluster the pooled support+query embeddings into N_WAY groups with the support
prototypes as initial centres, align groups to classes by majority support vote,
then treat a support row as mislabelled when its own class group is not the
nearest group. Corrections are applied to labels, not by dropping rows, so no
shot is lost.

Variants reported
-----------------
  proto / snn      - the two inference rules, uncorrected (baselines)
  qgsc_c           - classify with the query-derived group centres only
  qgsc_p           - correct labels, rebuild prototypes, classify with prototypes
  qgsc_h           - average of the two decision functions
  qgsc_soft        - soft label correction weighted by the margin, no hard flip
  qgsc_c2          - qgsc_c with a second refinement round
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
TAU = 0.0          # margin above which a label is flipped
KM_ITER = 10


# ---------------------------------------------------------------- primitives


def corrupt(sy: np.ndarray, ratio: float, rng: np.random.Generator):
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


def kmeans(x: np.ndarray, init: np.ndarray, n_iter: int = KM_ITER):
    c = init.copy()
    for _ in range(n_iter):
        lab = ((x[:, None, :] - c[None, :, :]) ** 2).sum(2).argmin(1)
        for k in range(len(c)):
            m = lab == k
            if m.any():
                c[k] = x[m].mean(0)
    return c


def align(centres: np.ndarray, sx: np.ndarray, sy: np.ndarray) -> np.ndarray:
    """Return cls2clu: class value -> centre row index."""
    classes = np.unique(sy)
    sl = ((sx[:, None, :] - centres[None, :, :]) ** 2).sum(2).argmin(1)
    cost = np.zeros((len(classes), len(centres)))
    for i, c in enumerate(classes):
        for k in range(len(centres)):
            cost[i, k] = -np.sum((sy == c) & (sl == k))
    rows, cols = linear_sum_assignment(cost)
    return {classes[r]: int(cols[j]) for j, r in enumerate(rows)}


def qgsc_core(sx, sy, qx, classes: np.ndarray, rounds: int = 1, tau: float = TAU):
    """Cluster, align, then flip labels whose own class group is not nearest.

    `classes` is the task's class set, fixed by the episode, so that a class all
    of whose support rows were corrupted still occupies a cluster.
    Returns corrected labels, group centres, and cls2clu alignment.
    """
    sy_cur = sy.copy()
    centres = None
    cls2clu = None
    for _ in range(rounds):
        protos = []
        for c in classes:
            sel = sy_cur == c
            if sel.sum() == 0:
                sel = sy == c          # fall back to the pre-correction rows
            protos.append(sx[sel].mean(0) if sel.sum() else sx.mean(0))
        protos = np.stack(protos)
        allx = np.vstack([sx, qx])
        centres = kmeans(allx, protos)
        cls2clu = align(centres, sx, sy_cur)
        if len(cls2clu) < len(classes):
            # keep alignment total: unmatched classes fall back to nearest centre
            for c in classes:
                if c not in cls2clu:
                    cls2clu[c] = int(((sx[sy == c].mean(0)[None, :] - centres) ** 2).sum(1).argmin())

        d = ((sx[:, None, :] - centres[None, :, :]) ** 2).sum(2)
        new = sy_cur.copy()
        for i in range(len(sx)):
            own = cls2clu.get(sy_cur[i], -1)
            if own < 0:
                continue
            nearest = int(d[i].argmin())
            if nearest == own:
                continue
            d_own, d_new = d[i, own], d[i, nearest]
            margin = (d_own - d_new) / (d_own + d_new + 1e-12)
            if margin > tau:
                inv = {v: k for k, v in cls2clu.items()}
                if nearest in inv:
                    new[i] = inv[nearest]
        sy_cur = new
    return sy_cur, centres, cls2clu


def classify_from_centres(centres, cls2clu, qx):
    inv = {v: k for k, v in cls2clu.items()}
    ks = np.array([v for v in cls2clu.values()])
    cs = np.array([k for k in cls2clu.keys()])
    d = ((qx[:, None, :] - centres[ks][None, :, :]) ** 2).sum(2)
    return cs[d.argmin(1)]


# ---------------------------------------------------------------- evaluation


def run_domain(name, x, y, key2i, n_episodes: int, rounds: int = 1):
    rows = []
    for fi in range(1, 11):
        idx = split_indices(name, fi, y, key2i)
        tx, ty = x[idx], y[idx]
        rng = np.random.default_rng(97_000 + fi)
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

                sy_c, centres, cls2clu = qgsc_core(sx, sy_n, qx, classes, rounds=rounds)
                rec("qgsc_c", classify_from_centres(centres, cls2clu, qx))

                if len(np.unique(sy_c)) >= 2:
                    rec("qgsc_p", proto_predict(sx, sy_c, qx))

                    # hybrid: sum of squared distance to the rebuilt prototype and
                    # to the query-derived group centre, both indexed by class
                    ks = np.array([cls2clu[c] for c in classes])
                    pcs = np.stack([sx[sy_c == c].mean(0) if (sy_c == c).any()
                                    else sx[sy_n == c].mean(0) for c in classes])
                    d1 = ((qx[:, None, :] - pcs[None, :, :]) ** 2).sum(2)
                    d2 = ((qx[:, None, :] - centres[ks][None, :, :]) ** 2).sum(2)
                    rec("qgsc_h", classes[(d1 + d2).argmin(1)])
            for tag, vals in bucket.items():
                rows.append({"domain": name, "split": fi, "ratio": ratio,
                             "method": tag, "accuracy": float(np.mean(vals)), "n": len(vals)})
        print(f"  {name} split {fi} done", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--tag", default="main")
    args = ap.parse_args()

    t0 = time.time()
    out = ROOT / "qgsc_output"
    out.mkdir(exist_ok=True)
    rows = []
    for name, path in (
        ("EuroSAT_RGB", ROOT / "full_output" / "eurosat_all_features.npz"),
        ("DTD", ROOT / "full_output" / "dtd_all_features.npz"),
    ):
        d = np.load(path, allow_pickle=True)
        key2i = {str(k): i for i, k in enumerate(d["keys"])}
        print(f"{name}: {d['x'].shape}", flush=True)
        rows += run_domain(name, d["x"].astype(np.float64), d["y"], key2i,
                           args.episodes, args.rounds)

    (out / f"qgsc_{args.tag}.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")

    methods = sorted({r["method"] for r in rows})
    print(f"\n=== accuracy (episodes={args.episodes}, rounds={args.rounds}, {time.time()-t0:.0f}s) ===")
    for dom in ("EuroSAT_RGB", "DTD"):
        print(f"\n{dom}")
        print(f"{'method':10s}" + "".join(f"{r:>8.1f}" for r in RATIOS))
        for m in methods:
            line = f"{m:10s}"
            for ratio in RATIOS:
                sel = [r["accuracy"] for r in rows if r["domain"] == dom and r["method"] == m and r["ratio"] == ratio]
                line += f"{np.mean(sel):8.4f}" if sel else f"{'-':>8s}"
            print(line)


if __name__ == "__main__":
    main()
