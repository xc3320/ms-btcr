"""Multi-scale label-noise detection and transductive clustering.

Part 1 asks whether three views vote more strictly than one. Probe 4 measured a
single-view leave-one-out detector at precision 0.55 / F1 0.69 on EuroSAT at 30%
noise: it accuses far more rows than are actually corrupted. If 2-of-3 or 3-of-3
agreement raises precision without collapsing recall, the signal becomes usable.

Part 2 asks whether that signal, or the extra views themselves, change accuracy.
Views are unit-normalised before concatenation because layer4 carries roughly 8x
the norm of layer2 and would otherwise dominate the distance.

Episodes are drawn here rather than through protocol.make_episode, because every
view must slice the SAME image rows; draw once, index all views.
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
    """5-way 5-shot 15-query draw, returning row indices so all views align."""
    classes = np.unique(ty)
    chosen = rng.choice(classes, size=N_WAY, replace=False)
    s_idx, q_idx = [], []
    for c in chosen:
        pool = np.flatnonzero(ty == c)
        perm = rng.permutation(pool)
        s_idx.append(perm[:K_SHOT])
        q_idx.append(perm[K_SHOT:K_SHOT + N_QUERY])
    s = np.concatenate(s_idx)
    q = np.concatenate(q_idx)
    return s, q, ty[s], ty[q]


def proto_predict(sx, sy, qx):
    classes = np.unique(sy)
    protos = np.stack([sx[sy == c].mean(0) for c in classes])
    d = ((qx[:, None, :] - protos[None, :, :]) ** 2).sum(2)
    return classes[d.argmin(1)]


def tk(sx, sy, qx, classes, rounds: int = 5):
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


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else float("nan")
    r = tp / (tp + fn) if tp + fn else float("nan")
    f = 2 * p * r / (p + r) if np.isfinite(p) and np.isfinite(r) and (p + r) > 0 else float("nan")
    return p, r, f


def run(name, views, key2i, y_all, n_episodes):
    rows, det = [], []
    for fi in range(1, 11):
        idx = split_indices(name, fi, y_all, key2i)
        tv = {k: views[k][idx] for k in VIEWS}
        ty = y_all[idx]
        tv["cat"] = np.hstack([unit(tv[k]) for k in VIEWS])
        rng = np.random.default_rng(105_000 + fi)
        for ratio in RATIOS:
            bucket: dict[str, list[float]] = {}
            cnt = {f"{t}_{s}": 0 for t in ("one", "any", "maj") for s in ("tp", "fp", "fn")}
            for _ in range(n_episodes):
                s, q, sy, qy = draw_episode(ty, rng)
                sy_n, flipped = corrupt(sy, ratio, rng)
                classes = np.unique(sy)

                def rec(tag, pred):
                    bucket.setdefault(tag, []).append(float((pred == qy).mean()))

                sp = {k: tv[k][s] for k in tv}
                qp = {k: tv[k][q] for k in tv}
                rec("proto_l4", proto_predict(sp["l4"], sy_n, qp["l4"]))
                for k in VIEWS + ("cat",):
                    rec(f"tk_{k}", tk(sp[k], sy_n, qp[k], classes))

                stack = np.vstack([tk(sp[k], sy_n, qp[k], classes) for k in VIEWS])
                vote = [np.bincount(stack[:, j]).argmax() for j in range(stack.shape[1])]
                vote = np.asarray(vote)
                rec("tk_vote", np.where(np.isin(vote, classes), vote, classes[0]))

                a = {k: loo_accuse(sp[k], sy_n) for k in VIEWS}
                cand = {
                    "one": a["l4"],
                    "any": a["l2"] | a["l3"] | a["l4"],
                    "maj": (a["l2"].astype(int) + a["l3"].astype(int) + a["l4"].astype(int)) >= 2,
                }
                for t, acc in cand.items():
                    cnt[f"{t}_tp"] += int((acc & flipped).sum())
                    cnt[f"{t}_fp"] += int((acc & ~flipped).sum())
                    cnt[f"{t}_fn"] += int((~acc & flipped).sum())

            for tag, vals in bucket.items():
                rows.append({"domain": name, "split": fi, "ratio": ratio,
                             "method": tag, "accuracy": float(np.mean(vals))})
            det.append({"domain": name, "split": fi, "ratio": ratio,
                        **{t: prf(cnt[f"{t}_tp"], cnt[f"{t}_fp"], cnt[f"{t}_fn"]) for t in ("one", "any", "maj")}})
        print(f"  {name} split {fi} done", flush=True)
    return rows, det


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--tag", default="smoke20")
    args = ap.parse_args()
    t0 = time.time()
    out = ROOT / "ms_output"
    out.mkdir(exist_ok=True)

    all_rows, all_det = [], []
    for name, msfile in (("EuroSAT_RGB", MS / "eurosat_multiscale.npz"),
                         ("DTD", MS / "dtd_multiscale.npz")):
        ms = np.load(msfile, allow_pickle=False)
        views = {k: ms[f"x{k[1]}"].astype(np.float64) for k in VIEWS}
        key2i = {str(k): i for i, k in enumerate(ms["keys"])}
        print(f"{name}: loaded", flush=True)
        r, d = run(name, views, key2i, ms["y"], args.episodes)
        all_rows += r
        all_det += d

    (out / f"ms_{args.tag}.json").write_text(
        json.dumps({"acc": all_rows, "det": all_det}, indent=1), encoding="utf-8")

    print(f"\n=== detector P / R / F1  ({time.time()-t0:.0f}s) ===")
    print(f"{'domain':13s} {'ratio':>5s} | {'1-view':>22s} | {'any-of-3':>22s} | {'majority':>22s}")
    for dom in ("EuroSAT_RGB", "DTD"):
        for ratio in RATIOS:
            sel = [x for x in all_det if x["domain"] == dom and x["ratio"] == ratio]
            if not sel:
                continue
            cell = lambda k: " ".join(f"{np.nanmean([s[k][i] for s in sel]):5.3f}" for i in range(3))  # noqa: E731
            print(f"{dom:13s} {ratio:5.1f} | {cell('one'):>22s} | {cell('any'):>22s} | {cell('maj'):>22s}")

    print("\n=== accuracy ===")
    for dom in ("EuroSAT_RGB", "DTD"):
        print(f"\n{dom}")
        methods = sorted({x["method"] for x in all_rows if x["domain"] == dom})
        print(f"{'method':10s}" + "".join(f"{r:>8.1f}" for r in RATIOS))
        for m in methods:
            line = f"{m:10s}"
            for ratio in RATIOS:
                s = [x["accuracy"] for x in all_rows
                     if x["domain"] == dom and x["method"] == m and x["ratio"] == ratio]
                line += f"{np.mean(s):8.4f}" if s else f"{'-':>8s}"
            print(line)


if __name__ == "__main__":
    main()
