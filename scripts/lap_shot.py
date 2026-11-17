"""LaplacianShot baseline: an optimisation-based transductive rule, implemented.

The manuscript's weakest comparison is that every competing arm is a rule we
wrote. LaplacianShot (Ziko et al., ECCV 2020) is the published representative of
the optimisation family: it minimises a Laplacian-regularised objective over the
query labels with a simplex constraint, solved by projected gradient. It is
implemented here in the form the paper specifies, on the same concatenated
features as the proposed method, so the difference measured is the inference
rule and not the representation.

Two hyperparameters are swept and the best setting is reported, which is
generous to the baseline: k in {3, 5} neighbours and lambda in {1, 10}. The
step size is 1/(1+lambda), the value that makes the projected-gradient iteration
stable for this objective.
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

from oracle_probe import K_SHOT, N_QUERY, N_WAY, RATIOS, corrupt, split_indices  # noqa: E402

ROOT = HERE.parent
MS = ROOT / "multiscale_output"
VIEWS = ("l2", "l3", "l4")
OOD_R = 0.2
GRID = ((3, 1.0), (3, 10.0), (5, 1.0), (5, 10.0))


def unit(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def draw_episode(ty, rng):
    classes = np.unique(ty)
    chosen = rng.choice(classes, size=N_WAY, replace=False)
    s_idx, q_idx = [], []
    for c in chosen:
        perm = rng.permutation(np.flatnonzero(ty == c))
        s_idx.append(perm[:K_SHOT])
        q_idx.append(perm[K_SHOT:K_SHOT + N_QUERY])
    return np.concatenate(s_idx), np.concatenate(q_idx)


def proj_simplex_rows(v: np.ndarray) -> np.ndarray:
    """Euclidean projection of every row of v onto the probability simplex."""
    u = np.sort(v, axis=1)[:, ::-1]
    css = np.cumsum(u, axis=1) - 1.0
    ind = np.arange(1, v.shape[1] + 1)
    cond = u - css / ind > 0
    rho = ind[cond.sum(1) - 1]
    theta = css[np.arange(len(v)), rho - 1] / rho
    return np.maximum(v - theta[:, None], 0.0)


def laplacian_shot(sx, sy, qx, classes, *, k=3, lam=0.1, iters=100):
    """Ziko et al. objective, solved by projected gradient on the simplex."""
    allx = unit(np.vstack([sx, qx]))
    n = len(allx)
    S = allx @ allx.T
    np.fill_diagonal(S, -np.inf)
    kk = min(k, n - 1)
    idx = np.argpartition(-S, kk, axis=1)[:, :kk]
    A = np.zeros((n, n))
    rows = np.repeat(np.arange(n), kk)
    A[rows, idx.ravel()] = 1.0
    A = np.maximum(A, A.T)
    deg = A.sum(1)
    deg[deg == 0] = 1.0
    Wn = A / np.sqrt(np.outer(deg, deg))
    L = np.eye(n) - Wn

    C = len(classes)
    Y0 = np.zeros((n, C))
    pos = {c: j for j, c in enumerate(classes)}
    for i, c in enumerate(sy):
        Y0[i, pos[c]] = 1.0

    Y = Y0.copy()
    alpha = 1.0 / (1.0 + lam)
    for _ in range(iters):
        grad = (Y - Y0) + lam * (L @ Y)
        Y = proj_simplex_rows(Y - alpha * grad)
    out = classes[Y[len(sy):].argmax(1)]
    score = 1.0 - Y[len(sy):].max(1)
    return out, score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--tag", default="main")
    ap.add_argument("--only", default=None,
                    help="restrict to one domain, e.g. Mini")
    args = ap.parse_args()
    t0 = time.time()
    out_dir = ROOT / "lap_shot_output"
    out_dir.mkdir(exist_ok=True)

    rows = []
    doms = [("EuroSAT_RGB", MS / "eurosat_multiscale.npz"),
            ("DTD", MS / "dtd_multiscale.npz"),
            ("Mini", ROOT / "backbone_output" / "miniimagenet_resnet18.npz")]
    if args.only:
        doms = [d for d in doms if d[0].startswith(args.only)]

    for name, msfile in doms:
        ms = np.load(msfile, allow_pickle=False)
        views = {key: ms[f"x{key[1]}"].astype(np.float64) for key in VIEWS}
        key2i = {str(key): i for i, key in enumerate(ms["keys"])}
        y = ms["y"]

        for fi in range(1, 11):
            idx = split_indices(name, fi, y, key2i)
            cat = np.hstack([unit(views[key][idx]) for key in VIEWS])
            ty = y[idx]
            rng = np.random.default_rng(113_000 + fi)

            for ratio in RATIOS:
                bucket: dict[str, list[float]] = {}
                for _ in range(args.episodes):
                    s, q = draw_episode(ty, rng)
                    qy = ty[q]
                    sy_n, _ = corrupt(sy=ty[s], ratio=ratio, rng=rng)
                    classes = np.unique(sy_n)

                    is_ood = np.zeros(len(q), dtype=bool)
                    outside = np.flatnonzero(~np.isin(ty, classes))
                    n_ood = max(1, int(round(len(q) * OOD_R)))
                    pick = rng.choice(outside, size=min(n_ood, len(outside)), replace=False)
                    qi = np.concatenate([q, pick])
                    qy_use = np.concatenate([qy, np.full(len(pick), -1)])
                    is_ood = np.concatenate([is_ood, np.ones(len(pick), dtype=bool)])

                    sx, qx = cat[s], cat[qi]
                    for k, lam in GRID:
                        pred, score = laplacian_shot(sx, sy_n, qx, classes, k=k, lam=lam)
                        clean = ~is_ood
                        acc = float((pred[clean] == qy_use[clean]).mean())
                        lb = np.concatenate([np.zeros(clean.sum()), np.ones(is_ood.sum())])
                        sc = np.concatenate([score[clean], score[is_ood]])
                        r = sps.rankdata(sc)[lb == 1].mean()
                        auroc = float((r - (is_ood.sum() + 1) / 2) / clean.sum())
                        bucket.setdefault(f"ls_k{k}_lam{lam}", []).append((acc, auroc))

                for tag, vals in bucket.items():
                    rows.append({"domain": name, "split": fi, "ratio": ratio,
                                 "method": tag,
                                 "accuracy": float(np.mean([v[0] for v in vals])),
                                 "auroc": float(np.mean([v[1] for v in vals]))})
            print(f"  {name} split {fi} done ({time.time()-t0:.0f}s)", flush=True)

    (out_dir / f"lap_{args.tag}.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")

    print(f"\n=== LaplacianShot, ood={OOD_R}, {args.episodes} episodes ({time.time()-t0:.0f}s) ===")
    methods = sorted({r["method"] for r in rows})
    for dom in (d[0] for d in doms):
        print(f"\n{dom}")
        print(f"{'method':16s}" + "".join(f"{r:>14.1f}" for r in RATIOS))
        for m in methods:
            line = f"{m:16s}"
            for ratio in RATIOS:
                sel = [x for x in rows if x["domain"] == dom and x["method"] == m
                       and x["ratio"] == ratio]
                acc = np.mean([x["accuracy"] for x in sel])
                au = np.mean([x["auroc"] for x in sel])
                line += f"{acc:9.4f}/{au:.3f}"
            print(line)


if __name__ == "__main__":
    main()
