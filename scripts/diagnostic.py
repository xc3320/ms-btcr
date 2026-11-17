"""One matched comparison: every strategy on the same episodes.

The probes ran separately and at different episode counts, which is fine for
deciding direction but not for a figure or a paper table. Here every strategy
sees identical episodes, so the only thing that differs is what the strategy does
with the support set.

Strategies:
  proto        - prototypical rule, no treatment (baseline)
  drop_any     - remove a support row if any view misclassifies it under LOO
  drop_all     - remove only when all three views misclassify it
  weight       - keep all rows, weight by logistic of the LOO margin
  correct3     - three rounds of hard label correction against cluster centres
  mstcr        - multi-scale transductive centroid re-estimation (this paper)

Also logs detector quality (precision / recall / F1) against the known flipped
mask, so the failure can be attributed to the detector rather than to the policy.

Query set is clean here on purpose: this isolates the support-noise mechanism.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from oracle_probe import N_WAY, K_SHOT, N_QUERY, RATIOS, split_indices  # noqa: E402
from mstcr_experiment import corrupt, unit, draw_episode  # noqa: E402

ROOT = HERE.parent
MS = ROOT / "multiscale_output"
VIEWS = ("l2", "l3", "l4")


def proto_predict(sx, sy, qx, w=None):
    classes = np.unique(sy)
    if w is None:
        w = np.ones(len(sy))
    m = np.stack([np.average(sx[sy == c], axis=0, weights=w[sy == c] + 1e-9) for c in classes])
    d = ((qx[:, None, :] - m[None, :, :]) ** 2).sum(2)
    return classes[d.argmin(1)]


def loo_margin(sx, sy):
    n = len(sy)
    out = np.zeros(n)
    ok = np.ones(n, dtype=bool)
    for i in range(n):
        keep = np.ones(n, dtype=bool)
        keep[i] = False
        present = np.unique(sy[keep])
        if len(present) < 2:
            continue
        m = np.stack([sx[keep][sy[keep] == c].mean(0) for c in present])
        d = ((sx[i][None, :] - m) ** 2).sum(1)
        own = np.flatnonzero(present == sy[i])
        if len(own) == 0:
            out[i] = -1.0
            ok[i] = False
            continue
        d_own = d[own[0]]
        others = np.delete(d, own[0])
        d_oth = others.min() if len(others) else np.inf
        out[i] = (d_oth - d_own) / (d_oth + d_own + 1e-12)
        ok[i] = d_own <= d_oth
    return out, ok


def tcr(sx, sy, qx, classes, rounds=5):
    m = np.stack([sx[sy == c].mean(0) for c in classes])
    allx = np.vstack([sx, qx])
    for _ in range(rounds):
        lab = ((allx[:, None, :] - m[None, :, :]) ** 2).sum(2).argmin(1)
        for k in range(len(classes)):
            sel = lab == k
            if sel.any():
                m[k] = allx[sel].mean(0)
    d = ((qx[:, None, :] - m[None, :, :]) ** 2).sum(2)
    return classes[d.argmin(1)]


def correct_iter(sx, sy, qx, classes, rounds=3):
    sy_c = sy.copy()
    for _ in range(rounds):
        protos = np.stack([sx[sy_c == c].mean(0) if (sy_c == c).any() else sx[sy == c].mean(0)
                           for c in classes])
        allx = np.vstack([sx, qx])
        m = protos.copy()
        for _ in range(5):
            lab = ((allx[:, None, :] - m[None, :, :]) ** 2).sum(2).argmin(1)
            for k in range(len(classes)):
                sel = lab == k
                if sel.any():
                    m[k] = allx[sel].mean(0)
        d = ((sx[:, None, :] - m[None, :, :]) ** 2).sum(2)
        nearest = classes[d.argmin(1)]
        own_idx = np.array([list(classes).index(c) for c in sy_c])
        own_d = d[np.arange(len(sx)), own_idx]
        new_d = d.min(1)
        flip = (new_d < own_d * 0.5) & (nearest != sy_c)
        sy_c = np.where(flip, nearest, sy_c)
    if len(np.unique(sy_c)) < 2:
        return proto_predict(sx, sy, qx)
    return proto_predict(sx, sy_c, qx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--tag", default="main")
    args = ap.parse_args()
    t0 = time.time()
    out_dir = ROOT / "diagnostic_output"
    out_dir.mkdir(exist_ok=True)

    rows, det = [], []
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
            ty = y[idx]
            rng = np.random.default_rng(117_000 + fi)

            for ratio in RATIOS:
                bucket: dict[str, list[float]] = {}
                cnt = {f"{t}_{s}": 0 for t in ("one", "cat") for s in ("tp", "fp", "fn")}
                for _ in range(args.episodes):
                    s, q, sy = draw_episode(ty, rng)
                    qy = ty[q]
                    sy_n, flipped = corrupt(sy, ratio, rng)
                    classes = np.unique(sy)
                    sp = {k: tv[k][s] for k in tv}
                    qp = {k: tv[k][q] for k in tv}

                    def rec(tag, pred):
                        bucket.setdefault(tag, []).append(float((pred == qy).mean()))

                    rec("proto", proto_predict(sp["cat"], sy_n, qp["cat"]))

                    marg = {}
                    ok = {}
                    for k in VIEWS:
                        marg[k], ok[k] = loo_margin(sp[k], sy_n)
                    any_bad = ~(ok["l2"] & ok["l3"] & ok["l4"])
                    all_bad = ~(ok["l2"] | ok["l3"] | ok["l4"])

                    if (~any_bad).sum() >= 2 * N_WAY and len(np.unique(sy_n[~any_bad])) >= 2:
                        rec("drop_any", proto_predict(sp["cat"][~any_bad], sy_n[~any_bad], qp["cat"]))
                    if (~all_bad).sum() >= 2 * N_WAY and len(np.unique(sy_n[~all_bad])) >= 2:
                        rec("drop_all", proto_predict(sp["cat"][~all_bad], sy_n[~all_bad], qp["cat"]))

                    m_mean = np.mean([marg[k] for k in VIEWS], axis=0)
                    rec("weight", proto_predict(sp["cat"], sy_n, qp["cat"],
                                                1.0 / (1.0 + np.exp(-2.0 * m_mean))))
                    rec("correct3", correct_iter(sp["cat"], sy_n, qp["cat"], classes))
                    rec("mstcr", tcr(sp["cat"], sy_n, qp["cat"], classes))

                    # detectors
                    m_cat, ok_cat = loo_margin(sp["cat"], sy_n)
                    acc_one = ~ok["l4"]
                    acc_cat = ~ok_cat
                    for tag, acc in (("one", acc_one), ("cat", acc_cat)):
                        cnt[f"{tag}_tp"] += int((acc & flipped).sum())
                        cnt[f"{tag}_fp"] += int((acc & ~flipped).sum())
                        cnt[f"{tag}_fn"] += int((~acc & flipped).sum())

                for tag, vals in bucket.items():
                    rows.append({"domain": name, "split": fi, "ratio": ratio,
                                 "method": tag, "accuracy": float(np.mean(vals)),
                                 "n_episodes": len(vals)})

                def prf(t):
                    tp, fp, fn = cnt[f"{t}_tp"], cnt[f"{t}_fp"], cnt[f"{t}_fn"]
                    p = tp / (tp + fp) if tp + fp else float("nan")
                    r = tp / (tp + fn) if tp + fn else float("nan")
                    f = 2 * p * r / (p + r) if np.isfinite(p) and np.isfinite(r) and (p + r) > 0 else float("nan")
                    return p, r, f

                det.append({"domain": name, "split": fi, "ratio": ratio,
                            "one_p": prf("one")[0], "one_r": prf("one")[1], "one_f": prf("one")[2],
                            "cat_p": prf("cat")[0], "cat_r": prf("cat")[1], "cat_f": prf("cat")[2]})
            print(f"  {name} split {fi} done", flush=True)

    (out_dir / f"diag_{args.tag}.json").write_text(
        json.dumps({"acc": rows, "det": det}, indent=1), encoding="utf-8")

    print(f"\n=== strategy comparison ({time.time()-t0:.0f}s) ===")
    for dom in ("EuroSAT_RGB", "DTD"):
        print(f"\n{dom}")
        meths = sorted({r["method"] for r in rows if r["domain"] == dom})
        print(f"{'method':10s}" + "".join(f"{r:>8.1f}" for r in RATIOS))
        for m in meths:
            line = f"{m:10s}"
            for ratio in RATIOS:
                s = [r["accuracy"] for r in rows if r["domain"] == dom and r["method"] == m and r["ratio"] == ratio]
                line += f"{np.mean(s):8.4f}" if s else f"{'-':>8s}"
            print(line)
        print(f"\n{'detector':10s}" + "".join(f"{r:>8.1f}" for r in RATIOS) + "   (P/R/F1 collapsed to F1)")
        for t in ("one", "cat"):
            line = f"{t:10s}"
            for ratio in RATIOS:
                s = [d[f"{t}_f"] for d in det if d["domain"] == dom and d["ratio"] == ratio]
                line += f"{np.nanmean(s):8.3f}" if s else f"{'-':>8s}"
            print(line)


if __name__ == "__main__":
    main()
