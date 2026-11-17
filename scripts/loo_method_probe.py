"""Probe 3: how much of the oracle headroom does a training-free LOO signal recover?

oracle_probe showed that knowing WHICH labels are corrupted is worth +0.18 to
+0.21 accuracy at 30% noise. This probe tests whether a support set can produce
that knowledge by itself, with no training and no external data.

Signals tried, all computed from the support set alone:
  loo_hard_any   - drop a row if ANY rule misclassifies it under leave-one-out
  loo_hard_all   - drop a row only if BOTH rules misclassify it
  loo_soft       - keep every row, weight it by a logistic function of the
                   leave-one-out margin (distance advantage of its own class)
  random_drop    - control: same number of rows dropped at random
  oracle         - upper reference from probe 1
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import protocol as P  # noqa: E402
from oracle_probe import SCALE, N_WAY, K_SHOT, N_QUERY, RATIOS, split_indices, corrupt  # noqa: E402

ROOT = HERE.parent
TEMPS = (2.0, 5.0, 10.0)


def loo_margin_proto(sx: np.ndarray, sy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Leave-one-out margin and correctness under the prototype rule."""
    n = len(sy)
    margin = np.full(n, np.nan)
    correct = np.ones(n, dtype=bool)
    for i in range(n):
        keep = np.ones(n, dtype=bool)
        keep[i] = False
        present = np.unique(sy[keep])
        if len(present) < 2:
            continue
        protos = np.stack([sx[keep][sy[keep] == c].mean(0) for c in present])
        d = ((sx[i][None, :] - protos) ** 2).sum(1)
        order = np.argsort(d)
        own = np.flatnonzero(present == sy[i])
        d_own = d[own[0]] if len(own) else np.inf
        d_oth = d[order[0]] if present[order[0]] != sy[i] else (d[order[1]] if len(order) > 1 else np.inf)
        margin[i] = (d_oth - d_own) / (d_oth + d_own + 1e-12) if np.isfinite(d_oth + d_own) else -1.0
        correct[i] = bool(d_own <= d_oth)
    return margin, correct


def loo_margin_snn(sx: np.ndarray, sy: np.ndarray, scale: float = SCALE) -> tuple[np.ndarray, np.ndarray]:
    n = len(sy)
    margin = np.full(n, np.nan)
    correct = np.ones(n, dtype=bool)
    xn = sx / (np.linalg.norm(sx, axis=1, keepdims=True) + 1e-12)
    for i in range(n):
        keep = np.ones(n, dtype=bool)
        keep[i] = False
        present = np.unique(sy[keep])
        if len(present) < 2:
            continue
        sim = xn[i] @ xn[keep].T
        att = np.exp(scale * sim)
        att /= att.sum()
        cls = {c: att[sy[keep] == c].sum() for c in present}
        own = cls.get(sy[i], 0.0)
        other = max((v for k, v in cls.items() if k != sy[i]), default=0.0)
        margin[i] = (own - other) / (own + other + 1e-12)
        correct[i] = bool(own >= other)
    return margin, correct


def proto_predict(sx, sy, qx, w=None):
    classes = np.unique(sy)
    if w is None:
        w = np.ones(len(sy))
    protos = np.stack([np.average(sx[sy == c], axis=0, weights=w[sy == c] + 1e-9) for c in classes])
    d = ((qx[:, None, :] - protos[None, :, :]) ** 2).sum(2)
    return classes[d.argmin(1)]


def snn_predict(sx, sy, qx, w=None, scale: float = SCALE):
    classes = np.unique(sy)
    if w is None:
        w = np.ones(len(sy))
    sx_n = sx / (np.linalg.norm(sx, axis=1, keepdims=True) + 1e-12)
    qx_n = qx / (np.linalg.norm(qx, axis=1, keepdims=True) + 1e-12)
    att = np.exp(scale * (qx_n @ sx_n.T)) * w[None, :]
    att /= att.sum(1, keepdims=True) + 1e-12
    cls_att = np.stack([att[:, sy == c].sum(1) for c in classes])
    return classes[cls_att.argmax(0)]


def run(name, x, y, key2i, n_episodes: int = 100):
    rows = []
    for fi in range(1, 11):
        idx = split_indices(name, fi, y, key2i)
        tx, ty = x[idx], y[idx]
        rng = np.random.default_rng(93_000 + fi)
        for ratio in RATIOS:
            bucket: dict[str, list[float]] = {}
            for _ in range(n_episodes):
                ep = P.make_episode(tx, ty, N_WAY, K_SHOT, N_QUERY, rng)
                sx, qx, qy = ep.support_x, ep.query_x, ep.query_y
                sy_n, flipped = corrupt(ep.support_y, ratio, rng)
                mp, cp = loo_margin_proto(sx, sy_n)
                ms, cs = loo_margin_snn(sx, sy_n)

                def rec(tag, pred):
                    bucket.setdefault(tag, []).append(float((pred == qy).mean()))

                rec("base_proto", proto_predict(sx, sy_n, qx))
                rec("base_snn", snn_predict(sx, sy_n, qx))

                # hard drop: any rule accuses
                ka = cp & cs
                if len(np.unique(sy_n[ka])) >= 2:
                    rec("hard_any", proto_predict(sx[ka], sy_n[ka], qx))
                # hard drop: both rules accuse
                kb = cp | cs
                if len(np.unique(sy_n[kb])) >= 2:
                    rec("hard_all", proto_predict(sx[kb], sy_n[kb], qx))

                # soft weights from the mean margin of the two rules
                m = np.nanmean(np.vstack([mp, ms]), axis=0)
                m = np.nan_to_num(m, nan=0.0)
                for t in TEMPS:
                    w = 1.0 / (1.0 + np.exp(-t * m))
                    rec(f"soft{t:g}_proto", proto_predict(sx, sy_n, qx, w))
                    rec(f"soft{t:g}_snn", snn_predict(sx, sy_n, qx, w))

                # control: drop as many rows as hard_any did, picked at random
                n_drop = int((~ka).sum())
                if n_drop and len(sy_n) - n_drop >= 2 * N_WAY:
                    kr = np.ones(len(sy_n), dtype=bool)
                    kr[rng.choice(len(sy_n), size=n_drop, replace=False)] = False
                    if len(np.unique(sy_n[kr])) >= 2:
                        rec("random_drop", proto_predict(sx[kr], sy_n[kr], qx))

            for tag, vals in bucket.items():
                rows.append({"domain": name, "split": fi, "ratio": ratio, "method": tag,
                             "accuracy": float(np.mean(vals)), "n": len(vals)})
        print(f"  {name} split {fi} done", flush=True)
    return rows


def main():
    out = ROOT / "loo_method_output"
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
    (out / "loo_method_probe.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")

    methods = sorted({r["method"] for r in rows})
    print("\n=== accuracy by method ===")
    for dom in ("EuroSAT_RGB", "DTD"):
        hdr = f"{'method':16s}" + "".join(f"{r:>8.1f}" for r in RATIOS)
        print(f"\n{dom}\n{hdr}")
        for m in methods:
            line = f"{m:16s}"
            for ratio in RATIOS:
                sel = [r["accuracy"] for r in rows if r["domain"] == dom and r["method"] == m and r["ratio"] == ratio]
                line += f"{np.mean(sel):8.4f}" if sel else f"{'-':>8s}"
            print(line)


if __name__ == "__main__":
    main()
