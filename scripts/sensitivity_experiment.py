# -*- coding: utf-8 -*-
"""退化条件比例的敏感性补充实验。

回应 34_ 的 B2：v5 把三种退化条件的注入比例固定为 0.30/0.20/0.20，
审稿人会问结论是否只是这个比例下的特例。本脚本只改支持集标签噪声比例，
其余配置与 full_experiment.py 一致，复用已缓存的 ResNet-18 特征。

输出只用于检查效应方向是否随比例单调、结论是否稳健，不替代主实验。
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np

from protocol import Record, estimate_effects, run_grid
import full_experiment as F

ROOT = Path(__file__).resolve().parent
OUT = ROOT.parent / "sensitivity_output"
OUT.mkdir(parents=True, exist_ok=True)

FEATURES = {
    "EuroSAT_RGB": ROOT.parent / "full_output" / "eurosat_all_features.npz",
    "DTD": ROOT.parent / "full_output" / "dtd_all_features.npz",
}
RATIOS = [0.10, 0.30, 0.50]
SPLITS = [1, 2]
N_REPEATS = 50
CONDS = F.conditions_8()


def load(npz: Path):
    z = np.load(npz, allow_pickle=False)
    keys = [str(k) for k in z["keys"]]
    return z["x"], z["y"], keys, {k: i for i, k in enumerate(keys)}


def rec_factory(domain: str, p: int, ratio: float, base_seed: int):
    def make(cond, seed) -> Record:
        return Record(
            dataset_split=f"{domain};split={p};support_noise_ratio={ratio}",
            noise_type_ratio=(f"support_replace={ratio};"
                              f"query_ood={F.CFG['query_ood_ratio']};"
                              f"support_missing={F.CFG['incomplete_ratio']}"),
            source_model_weight="torchvision/resnet18;IMAGENET1K_V1;sha256:f37072fd47e89c5e",
            method_code_version="protocol.py:sha256:48ebda4f868ff3df",
            cited_paper_local_version="Snell et al., NeurIPS 2017; arXiv:1703.05175",
            seed_repeats=f"seed={base_seed + seed};repeats={N_REPEATS}",
        )
    return make


def cell_mean(rows: list[dict], label: str, metric: str):
    vals = [r[metric] for r in rows
            if r["condition"] == label and r[metric] is not None]
    return float(np.mean(vals)) if vals else None


def main() -> None:
    started = time.time()
    rows = []
    for domain, npz in FEATURES.items():
        x_all, y_all, keys, key2i = load(npz)
        for p in SPLITS:
            idx = F.split_indices(domain, p, y_all, keys, key2i)
            x, y = x_all[idx], y_all[idx]
            for ratio in RATIOS:
                base_seed = 20261115 + p * 100000 + int(ratio * 1000)
                cfg = dict(F.CFG, support_noise_ratio=ratio,
                           n_repeats=N_REPEATS, base_seed=base_seed)
                grid = run_grid(x, y, cfg,
                                rec_factory(domain, p, ratio, base_seed),
                                conditions=CONDS)
                eff = estimate_effects(grid, "accuracy")
                a_only = cell_mean(grid, "query_ood+source_free", "auroc")
                a_noisy = cell_mean(grid, "support_label_noise+query_ood+source_free", "auroc")
                rows.append({
                    "domain": domain,
                    "split": p,
                    "support_noise_ratio": ratio,
                    "n_episodes": len(grid),
                    "clean_accuracy": round(cell_mean(grid, "source_free", "accuracy"), 6),
                    "noise_main_effect": round(eff["main_effects"]["support_label_noise"], 6),
                    "auroc_ood_only": round(a_only, 6),
                    "auroc_ood_plus_noise": round(a_noisy, 6),
                    "auroc_diff": round(a_noisy - a_only, 6),
                })
                print(f"{domain:12s} split{p} ratio={ratio:.2f} "
                      f"noise_effect={rows[-1]['noise_main_effect']:+.4f} "
                      f"auroc_diff={rows[-1]['auroc_diff']:+.4f}", flush=True)

    with (OUT / "sensitivity_results.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    by_ratio = {}
    for ratio in RATIOS:
        sub = [r for r in rows if r["support_noise_ratio"] == ratio]
        by_ratio[str(ratio)] = {
            "n_groups": len(sub),
            "noise_main_effect_mean": round(float(np.mean(
                [r["noise_main_effect"] for r in sub])), 6),
            "noise_main_effect_range": [min(r["noise_main_effect"] for r in sub),
                                        max(r["noise_main_effect"] for r in sub)],
            "auroc_diff_mean": round(float(np.mean([r["auroc_diff"] for r in sub])), 6),
            "auroc_diff_range": [min(r["auroc_diff"] for r in sub),
                                 max(r["auroc_diff"] for r in sub)],
            "all_noise_effects_negative": all(r["noise_main_effect"] < 0 for r in sub),
            "all_auroc_diffs_negative": all(r["auroc_diff"] < 0 for r in sub),
        }
    summary = {
        "status": "SUPPLEMENTARY SENSITIVITY CHECK - not a replacement for the main experiment",
        "ratios": RATIOS,
        "splits": SPLITS,
        "repeats_per_cell": N_REPEATS,
        "n_conditions": len(CONDS),
        "n_rows_total": sum(r["n_episodes"] for r in rows),
        "elapsed_seconds": round(time.time() - started, 1),
        "by_ratio": by_ratio,
        "monotone_noise_effect": all(
            by_ratio[str(RATIOS[i])]["noise_main_effect_mean"] >
            by_ratio[str(RATIOS[i + 1])]["noise_main_effect_mean"]
            for i in range(len(RATIOS) - 1)),
        "note": "Only support_label_noise_ratio varies. query_ood_ratio and "
                "incomplete_ratio stay at the main-experiment values.",
    }
    (OUT / "sensitivity_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n" + json.dumps({k: v for k, v in summary.items()
                             if k != "by_ratio"}, indent=2, ensure_ascii=False), flush=True)
    print(json.dumps(by_ratio, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
