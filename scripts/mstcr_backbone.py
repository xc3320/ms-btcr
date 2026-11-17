"""Run the full MS-TCR protocol on a second and third backbone.

The protocol is imported, not reimplemented: corrupt / unit / draw_episode /
proto / tcr all come from mstcr_experiment, so the only thing that changes
between backbones is where the three views come from. That keeps the
ResNet-18 numbers and these numbers comparable line by line.

Same grid as the main run: 10 splits x 3 noise ratios x 2 OOD rates x N
episodes, and the same five arms. The single-scale arm is the deepest view,
which is the view a layer4-only prototype baseline would use.

Output: backbone_output/{backbone}_main.json, one row per
(domain, split, ratio, ood, method).
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

from mstcr_experiment import (MAD_K, OOD_GRID, ROUNDS, corrupt, draw_episode,  # noqa: E402
                              proto, tcr, unit)
from oracle_probe import RATIOS, split_indices  # noqa: E402

ROOT = HERE.parent
BK = ROOT / "backbone_output"
METHODS = ["proto_deep", "tcr_deep", "tcr_raw", "mstcr", "mstcr_ood"]


def load(backbone: str, domain: str):
    f = BK / f"{'eurosat' if domain == 'EuroSAT_RGB' else 'dtd'}_{backbone}.npz"
    z = np.load(f, allow_pickle=False)
    views = {f"x{i+1}": z[f"x{i+1}"].astype(np.float64) for i in range(3)}
    key2i = {str(k): i for i, k in enumerate(z["keys"])}
    return views, z["y"], key2i


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True, choices=("resnet50", "vit_b_16"))
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--tag", default="main")
    args = ap.parse_args()
    t0 = time.time()
    rows = []

    for name in ("EuroSAT_RGB", "DTD"):
        views, y, key2i = load(args.backbone, name)
        for fi in range(1, 11):
            idx = split_indices(name, fi, y, key2i)
            tv = {k: views[k][idx] for k in views}
            tv["cat"] = np.hstack([unit(tv[k]) for k in ("x1", "x2", "x3")])
            tv["raw"] = np.hstack([tv[k] for k in ("x1", "x2", "x3")])
            ty = y[idx]
            rng = np.random.default_rng(111_000 + fi)   # same streams as main run

            for ratio in RATIOS:
                for ood_r in OOD_GRID:
                    bucket: dict[str, list[tuple[float, float]]] = {}
                    for _ in range(args.episodes):
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

                        def rec(tag, pred, score, is_ood=is_ood, qy_use=qy_use):
                            clean = ~is_ood
                            a = float((pred[clean] == qy_use[clean]).mean())
                            au = np.nan
                            if is_ood.any():
                                lb = np.concatenate([np.zeros(clean.sum()),
                                                     np.ones(is_ood.sum())])
                                sc = np.concatenate([score[clean], score[is_ood]])
                                r = sps.rankdata(sc)[lb == 1].mean()
                                au = float((r - (is_ood.sum() + 1) / 2) / clean.sum())
                            bucket.setdefault(tag, []).append((a, au))

                        sp_ = {k: tv[k][s] for k in tv}
                        qp_ = {k: tv[k][qx_idx] for k in tv}

                        p, sc = proto(sp_["x3"], sy_n, qp_["x3"])
                        rec("proto_deep", p, sc)
                        p, sc = tcr(sp_["x3"], sy_n, qp_["x3"], classes)
                        rec("tcr_deep", p, sc)
                        p, sc = tcr(sp_["raw"], sy_n, qp_["raw"], classes)
                        rec("tcr_raw", p, sc)
                        p, sc = tcr(sp_["cat"], sy_n, qp_["cat"], classes)
                        rec("mstcr", p, sc)
                        p, sc = tcr(sp_["cat"], sy_n, qp_["cat"], classes,
                                    ood_aware=True)
                        rec("mstcr_ood", p, sc)

                    for tag, vals in bucket.items():
                        au = [v[1] for v in vals if np.isfinite(v[1])]
                        rows.append({"backbone": args.backbone, "domain": name,
                                     "split": fi, "ratio": ratio, "ood": ood_r,
                                     "method": tag,
                                     "accuracy": float(np.mean([v[0] for v in vals])),
                                     "auroc": float(np.mean(au)) if au else None,
                                     "n": len(vals)})
            print(f"  {name} split {fi} done  {time.time()-t0:.0f}s", flush=True)

    BK.mkdir(exist_ok=True)
    out = BK / f"{args.backbone}_{args.tag}.json"
    out.write_text(json.dumps(rows, indent=1), encoding="utf-8")

    print(f"\n=== {args.backbone}  ACC/AUROC  ({time.time()-t0:.0f}s, "
          f"episodes={args.episodes}) ===")
    for dom in ("EuroSAT_RGB", "DTD"):
        for ood_r in OOD_GRID:
            print(f"\n{dom}  ood={ood_r}")
            print(f"{'method':12s}" + "".join(f"{r:>13.1f}" for r in RATIOS))
            for m in METHODS:
                line = f"{m:12s}"
                for ratio in RATIOS:
                    sel = [x for x in rows if x["domain"] == dom and x["method"] == m
                           and x["ratio"] == ratio and x["ood"] == ood_r]
                    a = np.mean([x["accuracy"] for x in sel])
                    au = [x["auroc"] for x in sel if x["auroc"] is not None]
                    line += f"{a:7.4f}" + (f"/{np.mean(au):.3f}" if au else "      ")
                print(line)


if __name__ == "__main__":
    main()
