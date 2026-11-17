"""Does estimating the query composition beat assuming a uniform one?

The published protocol fixes 15 queries per class, so the balanced assignment can
take that quota as given. A real query set need not be balanced. This script makes
the composition a controlled factor: class counts are drawn from a Dirichlet
distribution and two rules are compared on identical episodes, one that still
assumes M/N per class and one that re-estimates the proportions from the
assignment itself in EM fashion.

Questions, in order:
  1. how much accuracy the uniform assumption loses as imbalance grows,
  2. how many EM passes are needed before the estimate stops improving,
  3. whether the estimated proportions approach the true ones.

alpha is a property of the test condition, not of the method: alpha = infinity is
the balanced protocol, small alpha is a severely skewed query set. The realised
total-variation distance between the drawn and the uniform composition is recorded
for every episode so the severity of each level can be stated in the paper rather
than asserted.

DTD holds 40 images per class in total, five of which are support, so no class can
receive more than 35 queries; targets are clipped at that cap and the overflow is
redistributed.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from mstcr_experiment import MAD_K, N_QUERY, ROUNDS, corrupt, proto, sqdist, unit  # noqa: E402,F401
from oracle_probe import N_WAY, K_SHOT, split_indices  # noqa: E402
from suite_main import FILES  # noqa: E402

ROOT = HERE.parent
OUT = ROOT / "imbalance_output"
EM_GRID = (1, 2, 3, 5, 10)


def quotas_lr(p: np.ndarray, n: int, cap: int | None = None) -> np.ndarray:
    """Largest-remainder quotas summing exactly to n, at least one per class."""
    k = len(p)
    if cap is not None and cap * k < n:
        cap = None
    q = np.maximum(np.floor(p * n).astype(int), 1)
    while q.sum() > n:
        j = int(np.argmax(np.where(q > 1, q, -1)))
        if q[j] <= 1:
            raise RuntimeError("cannot reduce quotas to n")
        q[j] -= 1
    frac = p * n - np.floor(p * n)
    for j in np.argsort(-frac):
        if q.sum() >= n:
            break
        room = n - q.sum() if cap is None else min(n - q.sum(), cap - q[j])
        q[j] += max(0, room)
    if q.sum() < n:
        for j in np.argsort(q):
            if q.sum() >= n:
                break
            room = n - q.sum() if cap is None else min(n - q.sum(), cap - q[j])
            q[j] += max(0, room)
    if q.sum() != n:
        raise RuntimeError(f"quota mismatch {q.sum()} != {n}")
    return q


def assign(cost: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Exact min-cost assignment of rows to classes with the given quotas."""
    from scipy.optimize import linear_sum_assignment
    cols = np.repeat(np.arange(len(q)), q)
    if len(cols) != cost.shape[0]:
        raise ValueError("quota total does not match rows")
    r, c = linear_sum_assignment(cost[:, cols])
    lab = np.empty(cost.shape[0], dtype=int)
    lab[r] = cols[c]
    return lab


def _step(sx, sy, qx, classes, m, q, mad_k):
    """One block: assign under q, re-estimate centres with continuous weights."""
    n_sup = len(sx)
    allx = np.vstack([sx, qx])
    dq = sqdist(qx, m)
    lab = assign(dq, q)
    s = dq.min(1)
    med = np.median(s)
    mad = np.median(np.abs(s - med)) + 1e-12
    wq = np.exp(-np.maximum(s - (med + mad_k * mad), 0.0) / mad)
    w = np.concatenate([np.ones(n_sup), wq])
    for k in range(len(classes)):
        sel = np.concatenate([sy == classes[k], lab == k])
        if sel.any():
            m[k] = np.average(allx[sel], axis=0, weights=w[sel])
    return lab, m


def proportions(d: np.ndarray, mode: str) -> np.ndarray:
    """Class proportions from distances to the current centres.

    Counting the labels produced by a quota-constrained assignment cannot work:
    those counts equal the quota by construction, so the update would return the
    quota it started from. Both estimators below therefore ignore the quota and
    read the data directly. `hard` takes the nearest centre; `soft` keeps the
    full responsibility vector with the median squared distance as the scale, so
    no temperature has to be chosen.
    """
    if mode == "hard":
        return np.bincount(d.argmin(1), minlength=d.shape[1]) / d.shape[0]
    scale = np.median(d) + 1e-12
    logits = -d / scale
    logits -= logits.max(1, keepdims=True)
    r = np.exp(logits)
    r /= r.sum(1, keepdims=True)
    return r.mean(0)


NULL_TV: dict[tuple[int, int, float], float] = {}


def null_tv_quantile(n_rows: int, k: int, level: float = 0.95,
                     draws: int = 100_000) -> float:
    """Quantile of the TV distance to uniform under a balanced multinomial.

    The gate compares an estimate of the query composition with uniform, but the
    estimate is noisy even when the truth is uniform, so the reference point has
    to be the sampling distribution of that statistic rather than zero.
    """
    key = (n_rows, k, level)
    if key not in NULL_TV:
        rng = np.random.default_rng(11)
        c = rng.multinomial(n_rows, np.full(k, 1 / k), size=draws) / n_rows
        NULL_TV[key] = float(np.quantile(0.5 * np.abs(c - 1 / k).sum(1), level))
    return NULL_TV[key]


def run_adapt(sx, sy, qx, classes, *, passes=3, rounds=ROUNDS, mad_k=MAD_K,
              cap=None, level=0.95):
    """Choose the quota with a calibrated test instead of assuming one.

    Estimated proportions only pay off when the query set is in fact skewed; on a
    balanced one the same estimate costs accuracy. The choice is therefore made
    by a test with a known null distribution: if the estimated composition is
    statistically indistinguishable from uniform, the protocol quota is kept.
    """
    k_cls = len(classes)
    m0 = np.stack([sx[sy == c].mean(0) for c in classes])
    p_hat = proportions(sqdist(qx, m0), "hard")
    tv_hat = float(np.abs(p_hat - 1 / k_cls).sum() / 2)
    thr = null_tv_quantile(len(qx), k_cls, level)
    use_est = bool(tv_hat > thr)
    pred, s, p_used = run_rule(sx, sy, qx, classes,
                               em_passes=passes if use_est else 0,
                               rounds=rounds, mad_k=mad_k, cap=cap, mode="hard")
    return pred, s, p_used, tv_hat, use_est


def run_rule(sx, sy, qx, classes, *, em_passes=0, rounds=ROUNDS, mad_k=MAD_K,
             cap=None, mode="soft", shrink=1.0):
    """Balanced assignment with a fixed quota (em_passes=0) or an estimated one."""
    k_cls = len(classes)
    m = np.stack([sx[sy == c].mean(0) for c in classes])
    q = quotas_lr(np.full(k_cls, 1 / k_cls), len(qx), cap=cap)
    for it in range(rounds):
        if it < em_passes:
            p = proportions(sqdist(qx, m), mode)
            p = (1.0 - shrink) / k_cls + shrink * p     # shrink towards uniform
            q = quotas_lr(p, len(qx), cap=cap)
        lab, m = _step(sx, sy, qx, classes, m, q, mad_k)
    d = sqdist(qx, m)
    return classes[d.argmin(1)], d.min(1), q / q.sum()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="resnet18")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--noise", type=float, default=0.3)
    ap.add_argument("--alpha", default="0.3,1.0,inf")
    ap.add_argument("--tag", default="main")
    ap.add_argument("--only", default=None,
                    help="restrict to one domain, e.g. Mini")
    args = ap.parse_args()

    alphas = [float("inf") if a.strip().lower().startswith("inf") else float(a)
              for a in args.alpha.split(",")]
    OUT.mkdir(exist_ok=True)
    t0 = time.time()
    rows: list[dict] = []
    f_eu, f_dtd, f_mini, keys = FILES[args.backbone]
    M = N_QUERY * N_WAY

    doms = [("EuroSAT_RGB", f_eu), ("DTD", f_dtd), ("Mini", f_mini)]
    if args.only:
        doms = [d for d in doms if d[0].startswith(args.only)]

    for name, f in doms:
        z = np.load(f, allow_pickle=False)
        views = [z[k].astype(np.float64) for k in keys]
        key2i = {str(k): i for i, k in enumerate(z["keys"])}
        y = z["y"]

        for fi in range(1, 11):
            idx = split_indices(name, fi, y, key2i)
            cat = np.hstack([unit(v[idx]) for v in views])
            ty = y[idx]
            rng = np.random.default_rng(220_000 + fi)

            for alpha in alphas:
                for ep in range(args.episodes):
                    classes = rng.choice(np.unique(ty), size=N_WAY, replace=False)
                    pools = {c: rng.permutation(np.flatnonzero(ty == c))
                             for c in classes}
                    cap = min(len(v) for v in pools.values()) - K_SHOT
                    cap = max(1, cap)
                    if np.isinf(alpha):
                        p_true = np.full(N_WAY, 1 / N_WAY)
                    else:
                        p_true = rng.dirichlet(np.full(N_WAY, alpha))
                    q = quotas_lr(p_true, M, cap=cap)

                    s_idx, q_idx = [], []
                    for c, nq in zip(classes, q):
                        perm = pools[c]
                        s_idx.append(perm[:K_SHOT])
                        q_idx.append(perm[K_SHOT:K_SHOT + nq])
                    s = np.concatenate(s_idx)
                    qi = np.concatenate(q_idx)
                    sy = ty[s]
                    qy = ty[qi]
                    sy_n, _ = corrupt(sy, args.noise, rng)
                    sx, qx = cat[s], cat[qi]
                    p_obs = q / q.sum()
                    tv = float(np.abs(p_obs - 1 / N_WAY).sum() / 2)

                    base = {"domain": name, "split": fi, "alpha": alpha, "ep": ep,
                            "tv_uniform": tv, "q_obs": p_obs.tolist(),
                            "q_drawn": p_true.tolist()}

                    pred, _ = proto(sx, sy_n, qx)[:2]
                    rows.append({**base, "method": "proto", "em_passes": 0,
                                 "accuracy": float((pred == qy).mean()),
                                 "quota_err": None})

                    for tag, passes, mode, lam in (
                            ("bal_fixed", 0, "soft", 1.0),
                            *[(f"hard{p}", p, "hard", 1.0) for p in EM_GRID],
                            *[(f"soft{p}", p, "soft", 1.0) for p in EM_GRID[:3]],
                            *[(f"sh{int(l * 100)}", 3, "hard", l)
                              for l in (0.25, 0.5, 0.75)],
                    ):
                        pred, _, p_hat = run_rule(sx, sy_n, qx, classes,
                                                  em_passes=passes, cap=cap,
                                                  mode=mode, shrink=lam)
                        rows.append({**base, "method": tag, "em_passes": passes,
                                     "shrink": lam,
                                     "accuracy": float((pred == qy).mean()),
                                     "quota_err": float(np.abs(p_hat - p_obs).sum() / 2)})

                    pred, _, p_hat, tv_hat, use_est = run_adapt(
                        sx, sy_n, qx, classes, cap=cap)
                    rows.append({**base, "method": "adapt", "em_passes": 3,
                                 "accuracy": float((pred == qy).mean()),
                                 "quota_err": float(np.abs(p_hat - p_obs).sum() / 2),
                                 "tv_hat": tv_hat, "gate": int(use_est)})
            print(f"  {name} split {fi} done ({time.time() - t0:.0f}s)", flush=True)

    path = OUT / f"imbalance_{args.backbone}_{args.tag}.json"
    path.write_text(json.dumps(rows, indent=1), encoding="utf-8")

    methods = (["proto", "bal_fixed"] + [f"hard{p}" for p in EM_GRID]
               + [f"soft{p}" for p in EM_GRID[:3]] + ["sh25", "sh50", "sh75", "adapt"])
    print(f"\n=== {args.backbone}, noise {args.noise}, {args.episodes} episodes, "
          f"{time.time() - t0:.0f}s ===")
    for dom in (d[0] for d in doms):
        tvs = [r["tv_uniform"] for r in rows if r["domain"] == dom
               and r["method"] == "proto" and not np.isinf(r["alpha"])
               and r["alpha"] == 0.3]
        print(f"\n{dom}: TV distance to uniform at alpha=0.3 has mean "
              f"{np.mean(tvs):.3f}" if tvs else f"\n{dom}")
        head = "".join(f"{('alpha=' + str(a)):>17s}" for a in alphas)
        print(f"{'method':10s}{head}")
        for m in methods:
            line = f"{m:10s}"
            for a in alphas:
                sel = [r for r in rows if r["domain"] == dom and r["method"] == m
                       and r["alpha"] == a]
                if not sel:
                    line += f"{'-':>17s}"
                    continue
                acc = float(np.mean([r["accuracy"] for r in sel]))
                qe = [r["quota_err"] for r in sel if r["quota_err"] is not None]
                line += f"{acc:10.4f}" + (f"/{np.mean(qe):.3f}" if qe else "      ")
            print(line)
        print("  gate fires on: " + "  ".join(
            f"alpha={a}:{np.mean([r['gate'] for r in rows if r['domain'] == dom and r['method'] == 'adapt' and r['alpha'] == a]):.3f}"
            for a in alphas))
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
