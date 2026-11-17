"""Probe 2: can a support set estimate its own label-noise level without labels?

The whole method rests on one claim: leave-one-out accuracy computed with the
support set's OWN (possibly corrupted) labels is an unbiased-ish estimator of
1 - noise_rate. The mechanism is that a corrupted sample carries a label that no
subset of the remaining samples supports, so LOO marks it wrong almost surely.

If this holds, a training-free method can read off the noise level at inference
time and switch inference strategy accordingly. If it does not hold, the design
has no foundation and must be abandoned before any method code is written.

Outputs per (domain, split, ratio):
  loo_acc_proto / loo_acc_snn   - LOO accuracy per rule
  precision / recall            - treating "LOO-voted wrong" as a noise detector
  gap                           - observed LOO error rate vs true noise rate
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import protocol as P  # noqa: E402
from oracle_probe import SCALE, N_WAY, K_SHOT, N_QUERY, RATIOS, split_indices, corrupt, DATA  # noqa: E402

ROOT = HERE.parent


def loo_vote_proto(sx: np.ndarray, sy: np.ndarray) -> np.ndarray:
    """For each support row, predict it from the remaining rows. Returns bool correct."""
    n = len(sy)
    correct = np.zeros(n, dtype=bool)
    classes = np.unique(sy)
    for i in range(n):
        keep = np.ones(n, dtype=bool)
        keep[i] = False
        others_x, others_y = sx[keep], sy[keep]
        present = np.unique(others_y)
        if len(present) < 2:
            correct[i] = True  # no evidence either way; do not accuse
            continue
        protos = np.stack([others_x[others_y == c].mean(0) for c in present])
        d = ((sx[i][None, :] - protos) ** 2).sum(1)
        correct[i] = bool(present[d.argmin()] == sy[i])
    return correct


def loo_vote_snn(sx: np.ndarray, sy: np.ndarray, scale: float = SCALE) -> np.ndarray:
    n = len(sy)
    correct = np.zeros(n, dtype=bool)
    xn = sx / (np.linalg.norm(sx, axis=1, keepdims=True) + 1e-12)
    for i in range(n):
        keep = np.ones(n, dtype=bool)
        keep[i] = False
        present = np.unique(sy[keep])
        if len(present) < 2:
            correct[i] = True
            continue
        sim = xn[i] @ xn[keep].T
        att = np.exp(scale * sim)
        att /= att.sum()
        cls_att = {c: att[sy[keep] == c].sum() for c in present}
        best = max(cls_att, key=cls_att.get)
        correct[i] = bool(best == sy[i])
    return correct


def run(name: str, x: np.ndarray, y: np.ndarray, key2i: dict[str, int], n_episodes: int = 100):
    rows = []
    for fi in range(1, 11):
        idx = split_indices(name, fi, y, key2i)
        tx, ty = x[idx], y[idx]
        rng = np.random.default_rng(91_000 + fi)
        for ratio in RATIOS:
            agg = {k: [] for k in ("loo_proto", "loo_snn", "tp", "fp", "fn", "tn", "n")}
            for _ in range(n_episodes):
                ep = P.make_episode(tx, ty, N_WAY, K_SHOT, N_QUERY, rng)
                sy_n, flipped = corrupt(ep.support_y, ratio, rng)
                cp = loo_vote_proto(ep.support_x, sy_n)
                cs = loo_vote_snn(ep.support_x, sy_n)
                agg["loo_proto"].append(cp.mean())
                agg["loo_snn"].append(cs.mean())
                # detector = union of the two rules accusing a row
                accused = ~(cp & cs)
                agg["tp"].append(int((accused & flipped).sum()))
                agg["fp"].append(int((accused & ~flipped).sum()))
                agg["fn"].append(int((~accused & flipped).sum()))
                agg["tn"].append(int((~accused & ~flipped).sum()))
                agg["n"].append(len(sy_n))
            tp, fp, fn, tn = (sum(agg[k]) for k in ("tp", "fp", "fn", "tn"))
            prec = tp / (tp + fp) if tp + fp else float("nan")
            rec = tp / (tp + fn) if tp + fn else float("nan")
            rows.append(
                {
                    "domain": name,
                    "split": fi,
                    "ratio": ratio,
                    "loo_proto": float(np.mean(agg["loo_proto"])),
                    "loo_snn": float(np.mean(agg["loo_snn"])),
                    "precision": prec,
                    "recall": rec,
                    "flag_rate": (tp + fp) / max(1, sum(agg["n"])),
                }
            )
        print(f"  {name} split {fi} done", flush=True)
    return rows


def main() -> None:
    out = ROOT / "loo_probe_output"
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

    (out / "loo_probe.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")

    print("\n=== LOO as a noise estimator (mean over splits) ===")
    print(f"{'domain':13s} {'ratio':>5s} {'1-r':>6s} {'loo_p':>7s} {'loo_s':>7s} {'prec':>6s} {'rec':>6s} {'flag':>6s}")
    for dom in ("EuroSAT_RGB", "DTD"):
        for ratio in RATIOS:
            sel = [r for r in rows if r["domain"] == dom and r["ratio"] == ratio]
            if not sel:
                continue
            m = lambda k: float(np.nanmean([r[k] for r in sel]))  # noqa: E731
            print(f"{dom:13s} {ratio:5.1f} {1-ratio:6.3f} {m('loo_proto'):7.4f} {m('loo_snn'):7.4f} "
                  f"{m('precision'):6.3f} {m('recall'):6.3f} {m('flag_rate'):6.3f}")

    # correlation between estimated noise (1 - loo) and true noise
    for dom in ("EuroSAT_RGB", "DTD"):
        for rule in ("loo_proto", "loo_snn"):
            sel = [r for r in rows if r["domain"] == dom]
            est = np.array([1 - r[rule] for r in sel])
            true = np.array([r["ratio"] for r in sel])
            print(f"{dom:13s} {rule}: pearson={np.corrcoef(est, true)[0,1]:.4f} "
                  f"mae={np.mean(np.abs(est-true)):.4f}")


if __name__ == "__main__":
    main()
