"""Does a mixed-out query set break transductive clustering?

Every transductive few-shot method assumes the query set is closed-set: the
unlabelled samples all belong to the N classes under test. Under the degradation
profile this project has already measured, that assumption fails - the query set
can carry out-of-distribution samples. If clustering collapses in that regime,
then transductive gains reported on clean query sets do not transfer, and an
OOD-aware variant is needed.

Conditions crossed here:
  noise in {0, 0.3} x ood in {0, 0.2}

Methods:
  proto      - prototypical rule, support only
  snn        - soft nearest neighbour, support only
  tk_r1      - one round of query k-means from support prototypes
  tk_r5      - five rounds, labels used only for initialisation
  oatc       - five rounds with distance-based down-weighting of likely OOD rows

Accuracy is read on clean queries only, exactly as in the main protocol, so an
OOD-heavy query set cannot inflate or deflate the number by construction.
AUROC is reported for the OOD conditions as a secondary check.
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

import protocol as P  # noqa: E402
from oracle_probe import SCALE, N_WAY, K_SHOT, N_QUERY, split_indices  # noqa: E402

ROOT = HERE.parent
GRID = ((0.0, 0.0), (0.3, 0.0), (0.0, 0.2), (0.3, 0.2))
OOD_RATIO_KEEP = 0.75   # fraction of pooled embeddings trusted when re-estimating centres


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
    return classes[d.argmin(1)], d.min(1)


def snn_predict(sx, sy, qx, scale=SCALE):
    classes = np.unique(sy)
    sx_n = sx / (np.linalg.norm(sx, axis=1, keepdims=True) + 1e-12)
    qx_n = qx / (np.linalg.norm(qx, axis=1, keepdims=True) + 1e-12)
    att = np.exp(scale * (qx_n @ sx_n.T))
    att /= att.sum(1, keepdims=True) + 1e-12
    cls_att = np.stack([att[:, sy == c].sum(1) for c in classes])
    return classes[cls_att.argmax(0)], 1.0 - cls_att.max(0)


def tkmeans(sx, sy, qx, classes, rounds: int, ood_aware: bool = False):
    m = np.stack([sx[sy == c].mean(0) for c in classes])
    allx = np.vstack([sx, qx])
    if rounds <= 0:
        d = ((qx[:, None, :] - m[None, :, :]) ** 2).sum(2)
        return classes[d.argmin(1)], d.min(1)
    for _ in range(rounds):
        d = ((allx[:, None, :] - m[None, :, :]) ** 2).sum(2)
        lab = d.argmin(1)
        w = np.ones(len(allx))
        if ood_aware:
            s = d.min(1)
            thr = np.quantile(s, OOD_RATIO_KEEP)
            w = np.where(s > thr, 0.1, 1.0)
        for k in range(len(classes)):
            sel = lab == k
            if not sel.any():
                continue
            m[k] = np.average(allx[sel], axis=0, weights=w[sel])
    d = ((qx[:, None, :] - m[None, :, :]) ** 2).sum(2)
    return classes[d.argmin(1)], d.min(1)


def run_domain(name, x, y, key2i, n_episodes):
    rows = []
    for fi in range(1, 11):
        idx = split_indices(name, fi, y, key2i)
        tx, ty = x[idx], y[idx]
        rng = np.random.default_rng(101_000 + fi)
        for noise, ood in GRID:
            bucket: dict[str, list[tuple[float, float | None]]] = {}
            for _ in range(n_episodes):
                ep = P.make_episode(tx, ty, N_WAY, K_SHOT, N_QUERY, rng)
                sx, qx, qy = ep.support_x, ep.query_x, ep.query_y
                classes = np.unique(ep.support_y)
                sy_n, _ = corrupt(ep.support_y, noise, rng)
                q_is_ood = np.zeros(len(qx), dtype=bool)
                if ood > 0:
                    n_ood = max(1, int(round(len(qx) * ood)))
                    pick = rng.choice(len(ep.ood_pool_x), size=n_ood, replace=False)
                    qx = np.vstack([qx, ep.ood_pool_x[pick]])
                    qy = np.concatenate([qy, np.full(n_ood, -1)])
                    q_is_ood = np.concatenate([q_is_ood, np.ones(n_ood, dtype=bool)])

                clean = ~q_is_ood

                def rec(tag, pred, score=None):
                    acc = float((pred[clean] == qy[clean]).mean())
                    au = None
                    if q_is_ood.any():
                        s = score if score is not None else np.zeros(len(qx))
                        try:
                            lb = np.concatenate([np.zeros(clean.sum()), np.ones(q_is_ood.sum())])
                            sc = np.concatenate([s[clean], s[q_is_ood]])
                            r = sps.rankdata(sc)[lb == 1].mean()
                            au = float((r - (q_is_ood.sum() + 1) / 2) / clean.sum())
                        except Exception:
                            au = None
                    bucket.setdefault(tag, []).append((acc, au))

                p, s = proto_predict(sx, sy_n, qx); rec("proto", p, s)
                p, s = snn_predict(sx, sy_n, qx); rec("snn", p, s)
                p, s = tkmeans(sx, sy_n, qx, classes, 1); rec("tk_r1", p, s)
                p, s = tkmeans(sx, sy_n, qx, classes, 5); rec("tk_r5", p, s)
                p, s = tkmeans(sx, sy_n, qx, classes, 5, ood_aware=True); rec("oatc", p, s)

            for tag, vals in bucket.items():
                accs = [v[0] for v in vals]
                aus = [v[1] for v in vals if v[1] is not None]
                rows.append({"domain": name, "split": fi, "noise": noise, "ood": ood,
                             "method": tag, "accuracy": float(np.mean(accs)),
                             "auroc": float(np.mean(aus)) if aus else None, "n": len(vals)})
        print(f"  {name} split {fi} done", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--tag", default="main")
    args = ap.parse_args()
    t0 = time.time()
    out = ROOT / "ood_transductive_output"
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

    (out / f"ood_t_{args.tag}.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")

    methods = ["proto", "snn", "tk_r1", "tk_r5", "oatc"]
    print(f"\n=== ACC / AUROC  (episodes={args.episodes}, {time.time()-t0:.0f}s) ===")
    for dom in ("EuroSAT_RGB", "DTD"):
        print(f"\n{dom}")
        print(f"{'method':8s}" + "".join(f"{f'n{n} o{o}':>16s}" for n, o in GRID))
        for m in methods:
            line = f"{m:8s}"
            for noise, ood in GRID:
                sel = [r for r in rows if r["domain"] == dom and r["method"] == m
                       and r["noise"] == noise and r["ood"] == ood]
                a = np.mean([r["accuracy"] for r in sel]) if sel else float("nan")
                au = [r["auroc"] for r in sel if r["auroc"] is not None]
                aus = f"/{np.mean(au):.3f}" if au else "     "
                line += f"{a:8.4f}{aus:>8s}"
            print(line)


if __name__ == "__main__":
    main()
