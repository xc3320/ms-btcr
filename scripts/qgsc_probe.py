"""Probe 4: is the query set a stronger label-noise detector than the support set?

Probe 3 established that leave-one-out on 25 support samples recovers only about
a tenth of the oracle headroom, because 25 points cannot support reliable
self-validation. The query set is three times larger and is observed at
inference time in the standard transductive few-shot setting.

This probe compares two detectors on identical episodes:
  loo    - support-only leave-one-out accusation (probe 3 baseline)
  qgsc   - cluster the 25 support + 75 query embeddings into N_WAY groups,
           match clusters to classes by majority support vote, then accuse a
           support row when its own class cluster is not the nearest one.

Reported as precision / recall / F1 against the known flipped mask.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import protocol as P  # noqa: E402
from oracle_probe import SCALE, N_WAY, K_SHOT, N_QUERY, RATIOS, split_indices, corrupt  # noqa: E402

ROOT = HERE.parent


def kmeans_fixed(x: np.ndarray, init: np.ndarray, n_iter: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """Lloyd iterations with fixed initial centres. Returns centres and hard labels."""
    c = init.copy()
    lab = np.zeros(len(x), dtype=np.int64)
    for _ in range(n_iter):
        d = ((x[:, None, :] - c[None, :, :]) ** 2).sum(2)
        lab = d.argmin(1)
        for k in range(len(c)):
            m = lab == k
            if m.any():
                c[k] = x[m].mean(0)
    return c, lab


def qgsc_detect(sx: np.ndarray, sy: np.ndarray, qx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Accuse support rows whose class cluster is not their nearest cluster."""
    classes = np.unique(sy)
    protos = np.stack([sx[sy == c].mean(0) for c in classes])
    allx = np.vstack([sx, qx])
    centres, lab = kmeans_fixed(allx, protos)

    sup_lab = lab[: len(sx)]
    # match clusters to classes by how many support rows of each class land in them
    cost = np.zeros((len(classes), len(centres)))
    for i, c in enumerate(classes):
        for k in range(len(centres)):
            cost[i, k] = -np.sum((sy == c) & (sup_lab == k))
    rows, cols = linear_sum_assignment(cost)
    cls2clu = {classes[r]: cols[j] for j, r in enumerate(rows)}

    d = ((sx[:, None, :] - centres[None, :, :]) ** 2).sum(2)
    nearest = d.argmin(1)
    accused = np.array([nearest[i] != cls2clu.get(sy[i], -1) for i in range(len(sy))])
    return accused, cls2clu_predict(centres, cls2clu, classes)


def cls2clu_predict(centres, cls2clu, classes):
    return np.array([cls2clu.get(c, -1) for c in classes])


def loo_detect(sx: np.ndarray, sy: np.ndarray) -> np.ndarray:
    n = len(sy)
    accused = np.zeros(n, dtype=bool)
    for i in range(n):
        keep = np.ones(n, dtype=bool)
        keep[i] = False
        present = np.unique(sy[keep])
        if len(present) < 2:
            continue
        protos = np.stack([sx[keep][sy[keep] == c].mean(0) for c in present])
        d = ((sx[i][None, :] - protos) ** 2).sum(1)
        accused[i] = bool(present[d.argmin()] != sy[i])
    return accused


def run(name, x, y, key2i, n_episodes: int = 100):
    rows = []
    for fi in range(1, 11):
        idx = split_indices(name, fi, y, key2i)
        tx, ty = x[idx], y[idx]
        rng = np.random.default_rng(95_000 + fi)
        for ratio in RATIOS:
            acc = {k: 0 for k in ("loo_tp", "loo_fp", "loo_fn", "qg_tp", "qg_fp", "qg_fn", "n_bad", "n_ok")}
            for _ in range(n_episodes):
                ep = P.make_episode(tx, ty, N_WAY, K_SHOT, N_QUERY, rng)
                sy_n, flipped = corrupt(ep.support_y, ratio, rng)
                la = loo_detect(ep.support_x, sy_n)
                qa, _ = qgsc_detect(ep.support_x, sy_n, ep.query_x)
                acc["loo_tp"] += int((la & flipped).sum())
                acc["loo_fp"] += int((la & ~flipped).sum())
                acc["loo_fn"] += int((~la & flipped).sum())
                acc["qg_tp"] += int((qa & flipped).sum())
                acc["qg_fp"] += int((qa & ~flipped).sum())
                acc["qg_fn"] += int((~qa & flipped).sum())
                acc["n_bad"] += int(flipped.sum())
                acc["n_ok"] += int((~flipped).sum())

            def prf(tp, fp, fn):
                p = tp / (tp + fp) if tp + fp else float("nan")
                r = tp / (tp + fn) if tp + fn else float("nan")
                f = 2 * p * r / (p + r) if p and r and np.isfinite(p + r) and (p + r) > 0 else float("nan")
                return p, r, f

            lp, lr, lf = prf(acc["loo_tp"], acc["loo_fp"], acc["loo_fn"])
            qp, qr, qf = prf(acc["qg_tp"], acc["qg_fp"], acc["qg_fn"])
            rows.append({"domain": name, "split": fi, "ratio": ratio,
                         "loo_precision": lp, "loo_recall": lr, "loo_f1": lf,
                         "qg_precision": qp, "qg_recall": qr, "qg_f1": qf})
        print(f"  {name} split {fi} done", flush=True)
    return rows


def main():
    out = ROOT / "qgsc_probe_output"
    out.mkdir(exist_ok=True)
    rows = []
    for name, path in (
        ("EuroSAT_RGB", ROOT / "full_output" / "eurosat_all_features.npz"),
        ("DTD", ROOT / "full_output" / "dtd_all_features.npz"),
    ):
        d = np.load(path, allow_pickle=True)
        key2i = {str(k): i for i, k in enumerate(d["keys"])}
        print(f"{name}: {d['x'].shape}", flush=True)
        rows += run(name, d["x"].astype(np.float64), d["y"], key2i)
    (out / "qgsc_probe.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")

    print("\n=== detector comparison ===")
    print(f"{'domain':13s} {'ratio':>5s} | {'loo_P':>6s} {'loo_R':>6s} {'loo_F1':>7s} | {'qg_P':>6s} {'qg_R':>6s} {'qg_F1':>7s}")
    for dom in ("EuroSAT_RGB", "DTD"):
        for ratio in RATIOS:
            sel = [r for r in rows if r["domain"] == dom and r["ratio"] == ratio]
            if not sel:
                continue
            m = lambda k: float(np.nanmean([r[k] for r in sel]))  # noqa: E731
            print(f"{dom:13s} {ratio:5.1f} | {m('loo_precision'):6.3f} {m('loo_recall'):6.3f} {m('loo_f1'):7.3f} "
                  f"| {m('qg_precision'):6.3f} {m('qg_recall'):6.3f} {m('qg_f1'):7.3f}")


if __name__ == "__main__":
    main()
