"""Detector quality in the concatenated space versus per-view voting.

Earlier attempts ran leave-one-out on each view and then voted (F1 0.69-0.73 at
30% noise). But the concatenated space is measurably more separable than any
single view: OOD AUROC goes from 0.74 to 0.92 there. If separability is what the
detector needs, running leave-one-out directly in that space should beat voting
over weaker views, and voting is the wrong way to combine them.

Compared here on identical episodes, against the known flipped mask:
  l4      - single deepest view
  vote    - majority of the three per-view votes
  cat     - leave-one-out in the unit-normalised concatenated space
  cat_norm_loo - same, but distance normalised by within-class spread
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from oracle_probe import N_WAY, K_SHOT, N_QUERY, RATIOS, split_indices  # noqa: E402

ROOT = HERE.parent
MS = ROOT / "multiscale_output"
VIEWS = ("l2", "l3", "l4")


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


def loo_accuse(sx, sy, norm: bool = False):
    n = len(sy)
    acc = np.zeros(n, dtype=bool)
    for i in range(n):
        keep = np.ones(n, dtype=bool)
        keep[i] = False
        present = np.unique(sy[keep])
        if len(present) < 2:
            continue
        protos = np.stack([sx[keep][sy[keep] == c].mean(0) for c in present])
        d = ((sx[i][None, :] - protos) ** 2).sum(1)
        if norm:
            spread = np.array([sx[keep][sy[keep] == c].std(0).mean() + 1e-9 for c in present])
            d = d / (spread ** 2)
        acc[i] = bool(present[d.argmin()] != sy[i])
    return acc


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else float("nan")
    r = tp / (tp + fn) if tp + fn else float("nan")
    f = 2 * p * r / (p + r) if np.isfinite(p) and np.isfinite(r) and (p + r) > 0 else float("nan")
    return p, r, f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=40)
    args = ap.parse_args()

    records = []
    for name, msfile in (("EuroSAT_RGB", MS / "eurosat_multiscale.npz"),
                         ("DTD", MS / "dtd_multiscale.npz")):
        ms = np.load(msfile, allow_pickle=False)
        views = {k: ms[f"x{k[1]}"].astype(np.float64) for k in VIEWS}
        key2i = {str(k): i for i, k in enumerate(ms["keys"])}
        y = ms["y"]
        print(f"\n{'='*100}\n{name}")

        tally = {}
        for fi in range(1, 11):
            idx = split_indices(name, fi, y, key2i)
            tv = {k: views[k][idx] for k in VIEWS}
            tv["cat"] = np.hstack([unit(tv[k]) for k in VIEWS])
            ty = y[idx]
            rng = np.random.default_rng(109_000 + fi)
            for ratio in RATIOS:
                for _ in range(args.episodes):
                    s, q, sy, qy = draw_episode(ty, rng)
                    sy_n, flipped = corrupt(sy, ratio, rng)
                    sp = {k: tv[k][s] for k in tv}
                    a = {k: loo_accuse(sp[k], sy_n) for k in ("l2", "l3", "l4")}
                    accs = {
                        "l4": a["l4"],
                        "vote": (a["l2"].astype(int) + a["l3"].astype(int) + a["l4"].astype(int)) >= 2,
                        "cat": loo_accuse(sp["cat"], sy_n),
                        "cat_norm": loo_accuse(sp["cat"], sy_n, norm=True),
                    }
                    for t, ac in accs.items():
                        k = (fi, ratio, t)
                        c = tally.setdefault(k, [0, 0, 0])
                        c[0] += int((ac & flipped).sum())
                        c[1] += int((ac & ~flipped).sum())
                        c[2] += int((~ac & flipped).sum())
            print(f"  split {fi} done", flush=True)

        print(f"\n{'ratio':>5s} | {'l4':>20s} | {'vote':>20s} | {'cat':>20s} | {'cat_norm':>20s}")
        print(f"{'':>5s} | " + " | ".join(f"{'P    R    F1':>20s}" for _ in range(4)))
        for ratio in RATIOS:
            cells = []
            for t in ("l4", "vote", "cat", "cat_norm"):
                tp = sum(tally[(fi, ratio, t)][0] for fi in range(1, 11))
                fp = sum(tally[(fi, ratio, t)][1] for fi in range(1, 11))
                fn = sum(tally[(fi, ratio, t)][2] for fi in range(1, 11))
                p, r, f = prf(tp, fp, fn)
                records.append({"domain": name, "ratio": ratio, "view": t,
                                "precision": float(p), "recall": float(r),
                                "f1": float(f), "episodes_per_split": args.episodes,
                                "n_splits": 10})
                cells.append(f"{p:5.3f}{r:6.3f}{f:6.3f}" if np.isfinite(f) else "   --    --    --")
            print(f"{ratio:5.1f} | " + " | ".join(f"{c:>20s}" for c in cells))

    out = ROOT / "detector_output" / "detector_main.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(records, indent=1), encoding="utf-8")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
