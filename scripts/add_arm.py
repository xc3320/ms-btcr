"""Compute one or more arms on the existing grid and merge them into the suite file.

Same episode stream as suite_main (the arms never draw from the generator, so
computing a subset reproduces the identical episodes). Rows already present for
the requested arms are replaced, everything else is preserved untouched.

Use: python add_arm.py --backbone resnet18 --arms bal_soft --episodes 100
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

import suite_main as S  # noqa: E402
from mstcr_experiment import OOD_GRID, corrupt, draw_episode, unit  # noqa: E402
from oracle_probe import RATIOS, split_indices  # noqa: E402

OUT = S.OUT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True, choices=sorted(S.FILES))
    ap.add_argument("--arms", required=True, help="comma separated, as in ARMS")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--tag", default="main")
    a = ap.parse_args()
    arms = [x.strip() for x in a.arms.split(",") if x.strip()]
    for x in arms:
        if x not in S.ARMS:
            raise SystemExit(f"unknown arm {x}; known: {S.ARMS}")

    f = OUT / f"{a.backbone}_{a.tag}.json"
    old = json.loads(f.read_text(encoding="utf-8"))
    kept = [r for r in old if r["method"] not in arms]
    print(f"replacing {len(old)-len(kept)} rows, keeping {len(kept)}", flush=True)

    t0 = time.time()
    new = []
    keys = S.FILES[a.backbone][3]
    for name, path in (("EuroSAT_RGB", S.FILES[a.backbone][0]),
                       ("DTD", S.FILES[a.backbone][1])):
        z = np.load(path, allow_pickle=False)
        views = [z[k].astype(np.float64) for k in keys]
        key2i = {str(k): i for i, k in enumerate(z["keys"])}
        y = z["y"]

        for fi in range(1, 11):
            idx = split_indices(name, fi, y, key2i)
            v = [q[idx] for q in views]
            cat = np.hstack([unit(q) for q in v])
            ty = y[idx]
            rng = np.random.default_rng(111_000 + fi)

            for ratio in RATIOS:
                for ood_r in OOD_GRID:
                    bucket: dict[str, list[tuple[float, float]]] = {}
                    for _ in range(a.episodes):
                        s, q, sy = draw_episode(ty, rng)
                        qy = ty[q]
                        sy_n, _ = corrupt(sy, ratio, rng)
                        classes = np.unique(sy)
                        qx_idx, qy_use = q, qy
                        is_ood = np.zeros(len(q), dtype=bool)
                        if ood_r > 0:
                            outside = np.flatnonzero(~np.isin(ty, classes))
                            n_ood = max(1, int(round(len(q) * ood_r)))
                            pick = rng.choice(outside, size=min(n_ood, len(outside)),
                                              replace=False)
                            qx_idx = np.concatenate([q, pick])
                            qy_use = np.concatenate([qy, np.full(len(pick), -1)])
                            is_ood = np.concatenate([is_ood,
                                                     np.ones(len(pick), dtype=bool)])

                        def rec(tag_, pred, score):
                            clean = ~is_ood
                            acc = float((pred[clean] == qy_use[clean]).mean())
                            au = np.nan
                            if is_ood.any():
                                lb = np.concatenate([np.zeros(clean.sum()),
                                                     np.ones(is_ood.sum())])
                                sc = np.concatenate([score[clean], score[is_ood]])
                                r = sps.rankdata(sc)[lb == 1].mean()
                                au = float((r - (is_ood.sum() + 1) / 2) / clean.sum())
                            bucket.setdefault(tag_, []).append((acc, au))

                        sxp, qxp = cat[s], cat[qx_idx]
                        for arm in arms:
                            fn = getattr(S, arm)
                            p, sc = fn(sxp, sy_n, qxp, classes)
                            rec(arm, p, sc)

                    for k, vals in bucket.items():
                        au = [x[1] for x in vals if np.isfinite(x[1])]
                        new.append({"backbone": a.backbone, "domain": name,
                                    "split": fi, "ratio": ratio, "ood": ood_r,
                                    "method": k,
                                    "accuracy": float(np.mean([x[0] for x in vals])),
                                    "auroc": float(np.mean(au)) if au else None,
                                    "n": len(vals)})
            print(f"  {name} split {fi} done {time.time()-t0:.0f}s", flush=True)

    merged = kept + new
    merged.sort(key=lambda r: (r["backbone"], r["domain"], r["split"],
                               r["ratio"], r["ood"], r["method"]))
    f.write_text(json.dumps(merged, indent=1), encoding="utf-8")
    print(f"merged {len(new)} rows -> {f.name} ({len(merged)} total)")


if __name__ == "__main__":
    main()
