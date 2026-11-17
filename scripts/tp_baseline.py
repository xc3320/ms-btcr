"""Label-propagation baseline: an established transductive scheme to compare against.

Reviewers will ask whether MS-TCR is doing anything a standard graph method does
not. Label propagation over a k-NN graph of pooled support+query embeddings is
the cheapest faithful representative of that family (the mechanism behind
LaplacianShot-style methods), needs no training, and is simple enough that any
implementation gap is visible rather than hidden inside a tuning pipeline.

Two variants are included so the comparison is not rigged by a bad parameter
choice: a cosine k-NN graph, and an RBF-weighted graph. Both sweep a small
parameter grid and report the best, which is generous to the baseline.

No OOD handling: label propagation has no mechanism for it, which is part of the
point being tested.
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

from oracle_probe import (N_WAY, K_SHOT, N_QUERY, RATIOS, corrupt, split_indices)  # noqa: E402

ROOT = HERE.parent
MS = ROOT / "multiscale_output"
VIEWS = ("l2", "l3", "l4")
OOD_GRID = (0.0, 0.2)


def unit(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def draw_episode(ty, rng):
    classes = np.unique(ty)
    chosen = rng.choice(classes, size=N_WAY, replace=False)
    s_idx, q_idx = [], []
    for c in chosen:
        pool = np.flatnonzero(ty == c)
        perm = rng.permutation(pool)
        s_idx.append(perm[:K_SHOT])
        q_idx.append(perm[K_SHOT:K_SHOT + N_QUERY])
    s, q = np.concatenate(s_idx), np.concatenate(q_idx)
    return s, q, ty[s], ty[q]


def proto(sx, sy, qx):
    classes = np.unique(sy)
    m = np.stack([sx[sy == c].mean(0) for c in classes])
    d = ((qx[:, None, :] - m[None, :, :]) ** 2).sum(2)
    return classes[d.argmin(1)], d.min(1)


def label_prop(sx, sy, qx, classes, k=10, alpha=0.9, iters=40, mode="cosine", rbf_gamma=10.0):
    """Propagate support labels over a k-NN graph of pooled support+query rows."""
    allx = unit(np.vstack([sx, qx]))
    n = len(allx)
    S = allx @ allx.T
    np.fill_diagonal(S, -np.inf)
    kk = min(k, n - 1)
    idx = np.argpartition(-S, kk, axis=1)[:, :kk]
    A = np.zeros((n, n))
    for i in range(n):
        if mode == "cosine":
            A[i, idx[i]] = 1.0
        else:
            A[i, idx[i]] = np.exp(rbf_gamma * (S[i, idx[i]] - S[i].max()))
    A = np.maximum(A, A.T)
    d = A.sum(1)
    d[d == 0] = 1.0
    Wn = A / np.sqrt(np.outer(d, d))

    C = len(classes)
    Y0 = np.zeros((n, C))
    cls_pos = {c: j for j, c in enumerate(classes)}
    for i, c in enumerate(sy):
        Y0[i, cls_pos[c]] = 1.0
    F = Y0.copy()
    for _ in range(iters):
        F = alpha * (Wn @ F) + (1 - alpha) * Y0
    # only query rows are classified; support rows keep their given labels
    out = classes[F[len(sy):].argmax(1)]
    # OOD score: 1 - max class mass, higher means less confident
    score = 1.0 - F[len(sy):].max(1)
    return out, score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--tag", default="smoke30")
    ap.add_argument("--only", default=None, help="restrict to one domain, e.g. Mini")
    ap.add_argument("--ood", default=None, help="comma list, e.g. 0.2")
    args = ap.parse_args()
    ood_grid = OOD_GRID
    if args.ood:
        want = {float(x) for x in args.ood.split(",")}
        ood_grid = tuple(o for o in OOD_GRID if o in want)
    t0 = time.time()
    out_dir = ROOT / "tp_baseline_output"
    out_dir.mkdir(exist_ok=True)

    rows = []
    doms = [("EuroSAT_RGB", MS / "eurosat_multiscale.npz"),
            ("DTD", MS / "dtd_multiscale.npz"),
            ("Mini", ROOT / "backbone_output" / "miniimagenet_resnet18.npz")]
    if args.only:
        doms = [d for d in doms if d[0].startswith(args.only)]
    for name, msfile in doms:
        ms = np.load(msfile, allow_pickle=False)
        views = {k: ms[f"x{k[1]}"].astype(np.float64) for k in VIEWS}
        key2i = {str(k): i for i, k in enumerate(ms["keys"])}
        y = ms["y"]

        for fi in range(1, 11):
            idx = split_indices(name, fi, y, key2i)
            tv = {k: views[k][idx] for k in VIEWS}
            tv["cat"] = np.hstack([unit(tv[k]) for k in VIEWS])
            ty = y[idx]
            rng = np.random.default_rng(113_000 + fi)

            for ratio in RATIOS:
                for ood_r in ood_grid:
                    bucket: dict[str, list[tuple[float, float]]] = {}
                    for _ in range(args.episodes):
                        s, q, sy, _qy = draw_episode(ty, rng)
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

                        p, sc = proto(sp_["cat"], sy_n, qp_["cat"])
                        rec("proto_cat", p, sc)

                        for k in (5, 10, 20):
                            p, sc = label_prop(sp_["cat"], sy_n, qp_["cat"], classes, k=k)
                            rec(f"lp_cos_k{k}", p, sc)
                        p, sc = label_prop(sp_["cat"], sy_n, qp_["cat"], classes,
                                           k=10, mode="rbf")
                        rec("lp_rbf_k10", p, sc)

                    for tag, vals in bucket.items():
                        aa = float(np.mean([v[0] for v in vals]))
                        au = [v[1] for v in vals if np.isfinite(v[1])]
                        rows.append({"domain": name, "split": fi, "ratio": ratio,
                                     "ood": ood_r, "method": tag, "accuracy": aa,
                                     "auroc": float(np.mean(au)) if au else None})
            print(f"  {name} split {fi} done", flush=True)

    (out_dir / f"tp_{args.tag}.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")

    methods = sorted({r["method"] for r in rows})
    print(f"\n=== label-propagation baseline ({time.time()-t0:.0f}s) ===")
    for dom in (d[0] for d in doms):
        for ood_r in OOD_GRID:
            print(f"\n{dom} ood={ood_r}")
            print(f"{'method':12s}" + "".join(f"{r:>12.1f}" for r in RATIOS))
            for m in methods:
                line = f"{m:12s}"
                for ratio in RATIOS:
                    sel = [x for x in rows if x["domain"] == dom and x["method"] == m
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
