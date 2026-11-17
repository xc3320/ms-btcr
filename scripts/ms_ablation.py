"""Where does the multi-scale transductive gain come from, and does it help OOD?

The accuracy table showed concatenated multi-scale features with transductive
clustering beating the prototypical rule by +0.113 at 30% noise. That number
mixes three ingredients, and they must be separated before any of it is claimed
as a contribution:
  (a) richer features (multi-scale instead of layer4 only)
  (b) transductive re-estimation of centres from the query set
  (c) label noise being diluted by (b)

Ablations:
  norm_concat   - unit-normalised concatenation (the version that scored 0.8804)
  raw_concat    - plain concatenation, layer4 dominates by norm
  l4_only       - single view, transductive
  rounds 1/5/15 - how far the centres travel
  weighted      - multi-scale agreement used as support weights before clustering

OOD side: if views disagree about whether a query belongs to any class, that
disagreement may be a better out-of-distribution score than a single distance.
Three scores are compared by AUROC on the same episodes.
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


def proto_predict(sx, sy, qx):
    classes = np.unique(sy)
    protos = np.stack([sx[sy == c].mean(0) for c in classes])
    d = ((qx[:, None, :] - protos[None, :, :]) ** 2).sum(2)
    return classes[d.argmin(1)], d.min(1)


def tk(sx, sy, qx, classes, rounds=5, w=None):
    m = np.stack([sx[sy == c].mean(0) for c in classes])
    allx = np.vstack([sx, qx])
    ww = None if w is None else np.concatenate([w, np.ones(len(qx))])
    for _ in range(rounds):
        lab = ((allx[:, None, :] - m[None, :, :]) ** 2).sum(2).argmin(1)
        for k in range(len(classes)):
            sel = lab == k
            if not sel.any():
                continue
            m[k] = (np.average(allx[sel], axis=0, weights=ww[sel]) if ww is not None
                    else allx[sel].mean(0))
    d = ((qx[:, None, :] - m[None, :, :]) ** 2).sum(2)
    return classes[d.argmin(1)], d.min(1), m


def loo_accuse(sx, sy):
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
        acc[i] = bool(present[d.argmin()] != sy[i])
    return acc


def auroc(clean_scores, ood_scores):
    if len(ood_scores) == 0 or len(clean_scores) == 0:
        return None
    lb = np.concatenate([np.zeros(len(clean_scores)), np.ones(len(ood_scores))])
    sc = np.concatenate([clean_scores, ood_scores])
    r = sps.rankdata(sc)[lb == 1].mean()
    return float((r - (len(ood_scores) + 1) / 2) / len(clean_scores))


def run(name, views, key2i, y_all, n_episodes, ood_ratio=0.2):
    acc_rows, ood_rows = [], []
    for fi in range(1, 11):
        idx = split_indices(name, fi, y_all, key2i)
        tv = {k: views[k][idx] for k in VIEWS}
        ty = y_all[idx]
        tv["cat"] = np.hstack([unit(tv[k]) for k in VIEWS])
        tv["raw"] = np.hstack([tv[k] for k in VIEWS])
        # OOD pool: rows whose class is not among the 5 drawn; use whole split
        rng = np.random.default_rng(107_000 + fi)
        for ratio in RATIOS:
            bucket: dict[str, list[float]] = {}
            ood: dict[str, list[float]] = {}
            for _ in range(n_episodes):
                s, q, sy, qy = draw_episode(ty, rng)
                sy_n, flipped = corrupt(sy, ratio, rng)
                classes = np.unique(sy)

                # OOD rows: same split, classes outside the drawn five
                outside = np.flatnonzero(~np.isin(ty, classes))
                n_ood = max(1, int(round(len(q) * ood_ratio)))
                o_idx = rng.choice(outside, size=min(n_ood, len(outside)), replace=False)

                sp = {k: tv[k][s] for k in tv}
                up = {k: tv[k][q] for k in tv}
                op = {k: tv[k][o_idx] for k in tv}

                def rec(tag, pred):
                    bucket.setdefault(tag, []).append(float((pred == qy).mean()))

                p, s4 = proto_predict(sp["l4"], sy_n, up["l4"])
                rec("proto_l4", p)

                rec("tk_norm_concat", tk(sp["cat"], sy_n, up["cat"], classes, 5)[0])
                rec("tk_raw_concat", tk(sp["raw"], sy_n, up["raw"], classes, 5)[0])
                for r in (1, 5, 15):
                    rec(f"tk_l4_r{r}", tk(sp["l4"], sy_n, up["l4"], classes, r)[0])
                for r in (1, 5, 15):
                    rec(f"tk_cat_r{r}", tk(sp["cat"], sy_n, up["cat"], classes, r)[0])

                # multi-scale agreement weights on support rows
                a = {k: loo_accuse(sp[k], sy_n) for k in VIEWS}
                agree = (a["l2"].astype(int) + a["l3"].astype(int) + a["l4"].astype(int))
                w = np.where(agree == 3, 1.0, (np.where(agree == 2, 0.6, 0.2)))
                rec("tk_cat_weighted", tk(sp["cat"], sy_n, up["cat"], classes, 5, w)[0])

                # --- OOD scores, computed on a clean-support episode branch ---
                # use the same noisy support as the classifier would see
                pcat, scat, mcat = tk(sp["cat"], sy_n, up["cat"], classes, 5)
                _, s4t, _ = tk(sp["l4"], sy_n, up["l4"], classes, 5)
                _, _, m4 = tk(sp["l4"], sy_n, up["l4"], classes, 5)

                def dist(x, m):
                    return ((x[:, None, :] - m[None, :, :]) ** 2).sum(2).min(1)

                sc_c4 = dist(up["l4"], m4)
                sc_o4 = dist(op["l4"], m4)
                sc_cc = dist(up["cat"], mcat)
                sc_oc = dist(op["cat"], mcat)
                # agreement score: how many views put this row near any centre
                multi_c, multi_o = [], []
                ms = {}
                for k in VIEWS:
                    _, _, m = tk(sp[k], sy_n, up[k], classes, 5)
                    ms[k] = m
                for tag, pts in (("c", up), ("o", op)):
                    z = np.zeros(len(pts["l4"]))
                    for k in VIEWS:
                        dk = dist(pts[k], ms[k])
                        # rank within the pooled set of this episode's queries
                        z += dk / (dk.mean() + 1e-12)
                    (multi_c if tag == "c" else multi_o).append(z)
                ood.setdefault("auroc_l4", []).append(auroc(sc_c4, sc_o4) or np.nan)
                ood.setdefault("auroc_cat", []).append(auroc(sc_cc, sc_oc) or np.nan)
                ood.setdefault("auroc_multi", []).append(
                    auroc(multi_c[0], multi_o[0]) or np.nan)
            for tag, vals in bucket.items():
                acc_rows.append({"domain": name, "split": fi, "ratio": ratio,
                                 "method": tag, "accuracy": float(np.mean(vals))})
            for tag, vals in ood.items():
                ood_rows.append({"domain": name, "split": fi, "ratio": ratio,
                                 "score": tag, "auroc": float(np.nanmean(vals))})
        print(f"  {name} split {fi} done", flush=True)
    return acc_rows, ood_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--tag", default="smoke20")
    args = ap.parse_args()
    t0 = time.time()
    out = ROOT / "ms_ablation_output"
    out.mkdir(exist_ok=True)

    A, O = [], []
    for name, msfile in (("EuroSAT_RGB", MS / "eurosat_multiscale.npz"),
                         ("DTD", MS / "dtd_multiscale.npz")):
        ms = np.load(msfile, allow_pickle=False)
        views = {k: ms[f"x{k[1]}"].astype(np.float64) for k in VIEWS}
        key2i = {str(k): i for i, k in enumerate(ms["keys"])}
        a, o = run(name, views, key2i, ms["y"], args.episodes)
        A += a
        O += o

    (out / f"ablation_{args.tag}.json").write_text(
        json.dumps({"acc": A, "ood": O}, indent=1), encoding="utf-8")

    print(f"\n=== accuracy ({time.time()-t0:.0f}s) ===")
    for dom in ("EuroSAT_RGB", "DTD"):
        print(f"\n{dom}")
        ms_ = sorted({x["method"] for x in A if x["domain"] == dom})
        print(f"{'method':18s}" + "".join(f"{r:>8.1f}" for r in RATIOS))
        for m in ms_:
            line = f"{m:18s}"
            for ratio in RATIOS:
                s = [x["accuracy"] for x in A if x["domain"] == dom and x["method"] == m and x["ratio"] == ratio]
                line += f"{np.mean(s):8.4f}" if s else f"{'-':>8s}"
            print(line)

    print("\n=== OOD AUROC ===")
    for dom in ("EuroSAT_RGB", "DTD"):
        print(f"\n{dom}")
        print(f"{'score':14s}" + "".join(f"{r:>8.1f}" for r in RATIOS))
        for sc in sorted({x["score"] for x in O if x["domain"] == dom}):
            line = f"{sc:14s}"
            for ratio in RATIOS:
                s = [x["auroc"] for x in O if x["domain"] == dom and x["score"] == sc and x["ratio"] == ratio]
                line += f"{np.nanmean(s):8.4f}" if s else f"{'-':>8s}"
            print(line)


if __name__ == "__main__":
    main()
