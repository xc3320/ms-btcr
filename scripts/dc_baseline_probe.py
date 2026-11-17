# -*- coding: utf-8 -*-
"""DC 改编基线：小规模可行性与耗时探针。

依据 26_...md 第四节。官方 base_features.plk 来自 Google Drive，本机取不到，
故用「当前划分内未抽进本 episode 的目标域类别」估计 base_means / base_cov。
这是按 Distribution Calibration 算法改编，不是复现 Yang et al. 的官方数值。

输出只用于判断能否扩规模，不写进论文。
用法：python dc_baseline_probe.py [每组合重复数，默认 1]
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from protocol import apply_factors, make_episode, prototypical_predict
import full_experiment as F

ROOT = Path(__file__).resolve().parent
OUT = ROOT.parent / "dc_probe"
OUT.mkdir(parents=True, exist_ok=True)

REPEATS = int(sys.argv[1]) if len(sys.argv) > 1 else 1
PARTITION = 1
BASE_SEED = 20261115 + PARTITION * 100000
CFG = dict(F.CFG, n_repeats=REPEATS, base_seed=BASE_SEED)
CONDITIONS = F.conditions_8()
FEATURES = {
    "EuroSAT_RGB": ROOT.parent / "full_output" / "eurosat_all_features.npz",
    "DTD": ROOT.parent / "full_output" / "dtd_all_features.npz",
}

np.random.seed(20261115)  # 保留仅为兼容旧探针；采样不再依赖全局状态


def dc_calibrate(q: np.ndarray, means, covs, k: int = 2, alpha: float = 0.21):
    """照抄官方 evaluate_DC.py 的 distribution_calibration。"""
    dist = np.asarray([np.linalg.norm(q - m) for m in means])
    index = np.argpartition(dist, k)[:k]
    mu = np.mean(np.concatenate([np.asarray(means)[index], q[np.newaxis, :]]), axis=0)
    cov = np.mean(np.asarray(covs)[index], axis=0) + alpha
    return mu, cov


def load_domain(name: str, npz: Path):
    z = np.load(npz, allow_pickle=False)
    x_all, y_all = z["x"], z["y"]
    keys = [str(k) for k in z["keys"]]
    key2i = {k: i for i, k in enumerate(keys)}
    idx = F.split_indices(name, PARTITION, y_all, keys, key2i)
    x, y = x_all[idx], y_all[idx]
    stats = {}
    for c in np.unique(y):
        f = x[y == c].astype(np.float64)
        stats[int(c)] = (f.mean(axis=0), np.cov(f.T))
    return x, y, stats, int(len(idx)), int(len(np.unique(y)))


def run_one(x, y, stats, cond, seed, diagnose: bool):
    rng = np.random.default_rng(BASE_SEED + seed)
    ep = make_episode(x, y, n_way=int(CFG["n_way"]), k_shot=int(CFG["k_shot"]),
                      n_query=int(CFG["n_query"]), rng=rng, ood_pool_x=None)
    chosen = set(np.unique(ep.support_y).tolist())
    pool = [stats[c] for c in sorted(stats) if c not in chosen]
    means = [m for m, _ in pool]
    covs = [v for _, v in pool]

    ep = apply_factors(ep, cond, CFG, rng)
    mask = ep.clean_query_mask()

    proto_pred, _ = prototypical_predict(ep.support_x, ep.support_y, ep.query_x)
    proto_acc = float((proto_pred[mask] == ep.query_y[mask]).mean())

    t0 = time.perf_counter()
    if (ep.support_x < 0).any() or (ep.query_x < 0).any():
        raise RuntimeError("negative feature found; Tukey power(0.5) undefined")
    sx = np.power(ep.support_x.astype(np.float64), 0.5)
    qx = np.power(ep.query_x.astype(np.float64), 0.5)

    num = int(750 / int(CFG["k_shot"]))
    sampled, labels, min_eig = [], [], None
    for i in range(len(sx)):
        mu, cov = dc_calibrate(sx[i], means, covs)
        if diagnose and i == 0:
            min_eig = float(np.linalg.eigvalsh(cov)[0])
        # 每个 episode 用自己那条随机流，结果不随 episode 执行顺序改变。
        sampled.append(rng.multivariate_normal(mu, cov, size=num))
        labels.extend([int(ep.support_y[i])] * num)
    X = np.concatenate([sx] + sampled)
    Y = np.concatenate([ep.support_y, np.asarray(labels)])
    clf = LogisticRegression(max_iter=1000).fit(X, Y)
    pred = clf.predict(qx[mask])
    dc_acc = float((pred == ep.query_y[mask]).mean())
    elapsed = time.perf_counter() - t0

    return {
        "proto_acc": round(proto_acc, 6),
        "dc_acc": round(dc_acc, 6),
        "seconds": round(elapsed, 3),
        "n_base_classes": len(means),
        "n_support": int(len(sx)),
        "n_augmented": int(len(X)),
        "min_eig_of_calibrated_cov": None if min_eig is None else round(min_eig, 6),
        "sampled_absmax": round(float(np.abs(np.concatenate(sampled)).max()), 6),
        "all_finite": bool(np.isfinite(X).all()),
    }


def main() -> None:
    started = time.perf_counter()
    rows = []
    for domain, npz in FEATURES.items():
        x, y, stats, n_images, n_classes = load_domain(domain, npz)
        print(f"{domain}: split {PARTITION}, {n_images} images, "
              f"{n_classes} classes, base pool >= {n_classes - 5}", flush=True)
        for cond in CONDITIONS:
            for seed in range(REPEATS):
                r = run_one(x, y, stats, cond, seed,
                            diagnose=(cond is CONDITIONS[0] and seed == 0))
                rows.append({"domain": domain, "condition": cond.label(),
                             "seed": seed, **r})
                print(f"  {cond.label():58s} seed {seed}: "
                      f"proto {r['proto_acc']:.4f} dc {r['dc_acc']:.4f} "
                      f"{r['seconds']:.2f}s", flush=True)

    with (OUT / f"dc_probe_r{REPEATS}.csv").open("w", encoding="utf-8",
                                                 newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    secs = [r["seconds"] for r in rows]
    by_cell = {}
    for r in rows:
        k = (r["domain"], r["condition"])
        by_cell.setdefault(k, []).append(r)
    cells = [
        {"domain": d, "condition": c,
         "n": len(rs),
         "proto_acc_mean": round(float(np.mean([q["proto_acc"] for q in rs])), 6),
         "dc_acc_mean": round(float(np.mean([q["dc_acc"] for q in rs])), 6),
         "seconds_mean": round(float(np.mean([q["seconds"] for q in rs])), 3)}
        for (d, c), rs in sorted(by_cell.items())
    ]
    mean_s = float(np.mean(secs))
    summary = {
        "status": "FEASIBILITY PROBE - adapted baseline, NOT a reproduction",
        "repeats_per_cell": REPEATS,
        "n_domains": 2,
        "n_conditions": len(CONDITIONS),
        "n_episodes": len(rows),
        "partition": PARTITION,
        "total_seconds": round(time.perf_counter() - started, 1),
        "seconds_per_episode_mean": round(mean_s, 3),
        "seconds_per_episode_min": round(float(np.min(secs)), 3),
        "seconds_per_episode_max": round(float(np.max(secs)), 3),
        "projected_seconds_20_repeats_320_episodes": round(mean_s * 320, 1),
        "projected_seconds_full_16000_episodes": round(mean_s * 16000, 1),
        "all_finite": all(r["all_finite"] for r in rows),
        "min_eig_samples": [r["min_eig_of_calibrated_cov"] for r in rows
                            if r["min_eig_of_calibrated_cov"] is not None],
        "cells": cells,
        "algorithm_note": "Tukey beta=0.5; k=2 nearest base means; alpha=0.21; "
                          "750/n_shot samples per support feature; "
                          "LogisticRegression(max_iter=1000). Copied from "
                          "baseline_dc/evaluate_DC.py.",
        "adaptation_note": "base_means/base_cov estimated from target-domain "
                           "classes NOT sampled into the current episode, "
                           "inside the same split. This is transductive and "
                           "differs from the official miniImageNet base statistics.",
        "known_blocks": [
            "official base_features.plk unavailable (Google Drive HTTP 000)",
            "upstream repository has no declared license (GitHub API license: None)",
            "no AUROC reported for this baseline; accuracy on clean queries only",
        ],
    }
    (OUT / f"dc_probe_r{REPEATS}_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n" + json.dumps({k: v for k, v in summary.items() if k != "cells"},
                            indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
