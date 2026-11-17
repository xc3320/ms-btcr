# -*- coding: utf-8 -*-
"""Attention-scale sensitivity for the second (Matching Networks) baseline.

SCALE is an implementation choice, not a constant published by Vinyals et al.
This script checks whether the direction of the three main conclusions depends
on it. Same cached features, same splits, same episode seeds as the main run.

2 domains x 2 splits x 3 scales x 8 conditions x 50 repeats = 4,800 rows.
Output is a direction check only, not a replacement for the main numbers.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from protocol import Condition, Record, estimate_effects
import full_experiment as F
import secondbaseline_experiment as S

ROOT = Path(__file__).resolve().parent
OUT = ROOT.parent / "second_baseline_output"
OUT.mkdir(parents=True, exist_ok=True)

SCALES = [5.0, 10.0, 20.0]
SPLITS = [1, 2]
N_REPEATS = 50
CONDS = F.conditions_8()


def rec_factory(domain: str, p: int, scale: float, base_seed: int):
    def make(cond: Condition, seed: int) -> Record:
        return Record(
            dataset_split=f"{domain};split={p};scale={scale}",
            noise_type_ratio=("support_replace=0.30;query_ood=0.20;"
                              "support_missing=0.20"),
            source_model_weight="torchvision/resnet18;IMAGENET1K_V1;sha256:f37072fd47e89c5e",
            method_code_version=f"secondbaseline_experiment.py;scale={scale}",
            cited_paper_local_version="Vinyals et al., NIPS 2016; arXiv:1606.04080",
            seed_repeats=f"seed={base_seed + seed};repeats={N_REPEATS}",
        )
    return make


def main() -> None:
    rows = []
    for domain, npz in S.FEATURES.items():
        x_all, y_all, keys, key2i = S.load(npz)
        for p in SPLITS:
            idx = F.split_indices(domain, p, y_all, keys, key2i)
            x, y = x_all[idx], y_all[idx]
            for scale in SCALES:
                S.SCALE = scale
                base_seed = 20261115 + p * 100000
                cfg = dict(F.CFG, n_repeats=N_REPEATS, base_seed=base_seed)
                grid = S.run_grid_matching(x, y, cfg,
                                          rec_factory(domain, p, scale, base_seed),
                                          CONDS)
                eff = estimate_effects(grid, "accuracy")
                a_only = np.mean([r["auroc"] for r in grid
                                  if r["condition"] == "query_ood+source_free"
                                  and r["auroc"] is not None])
                a_noisy = np.mean([r["auroc"] for r in grid
                                   if r["condition"] == "support_label_noise+query_ood+source_free"
                                   and r["auroc"] is not None])
                clean = np.mean([r["accuracy"] for r in grid
                                 if r["condition"] == "source_free"])
                rows.append({
                    "domain": domain, "split": p, "scale": scale,
                    "clean_accuracy": round(float(clean), 6),
                    "noise_main_effect": round(float(eff["main_effects"]["support_label_noise"]), 6),
                    "incomplete_main_effect": round(float(eff["main_effects"]["incomplete_new_class"]), 6),
                    "noise_x_incomplete": round(float(
                        eff["two_way_interactions"]["support_label_noisexincomplete_new_class"]), 6),
                    "auroc_ood_only": round(float(a_only), 6),
                    "auroc_ood_plus_noise": round(float(a_noisy), 6),
                    "h2_auroc_delta": round(float(a_noisy - a_only), 6),
                    "n_episodes": len(grid),
                })
                print(f"{domain:12s} split{p} scale={scale:4.0f} "
                      f"clean={clean:.4f} noise_eff={rows[-1]['noise_main_effect']:+.4f} "
                      f"h2_delta={rows[-1]['h2_auroc_delta']:+.4f}", flush=True)

    with (OUT / "scale_sensitivity.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    by_scale = {}
    for s in SCALES:
        sub = [r for r in rows if r["scale"] == s]
        by_scale[str(s)] = {
            "n_groups": len(sub),
            "clean_accuracy_mean": round(float(np.mean([r["clean_accuracy"] for r in sub])), 6),
            "noise_effect_mean": round(float(np.mean([r["noise_main_effect"] for r in sub])), 6),
            "incomplete_effect_mean": round(float(np.mean([r["incomplete_main_effect"] for r in sub])), 6),
            "h2_delta_mean": round(float(np.mean([r["h2_auroc_delta"] for r in sub])), 6),
            "all_noise_effects_negative": all(r["noise_main_effect"] < 0 for r in sub),
            "all_incomplete_effects_negative": all(r["incomplete_main_effect"] < 0 for r in sub),
            "all_h2_deltas_negative": all(r["h2_auroc_delta"] < 0 for r in sub),
        }

    summary = {
        "status": "SCALE SENSITIVITY CHECK - direction only, not main numbers",
        "scales": SCALES, "splits": SPLITS, "repeats_per_cell": N_REPEATS,
        "n_rows_total": sum(r["n_episodes"] for r in rows),
        "by_scale": by_scale,
        "all_scales_direction_consistent": all(
            v["all_noise_effects_negative"] and v["all_incomplete_effects_negative"]
            and v["all_h2_deltas_negative"] for v in by_scale.values()),
        "note": ("SCALE is an implementation choice made on frozen ImageNet features. "
                 "SCALE=1 was rejected: unscaled softmax degenerates into class-size "
                 "voting and drops clean accuracy from 0.84 to 0.39 when 20 percent "
                 "of support samples are removed. Absolute numbers depend on SCALE; "
                 "this file only supports direction stability across 5/10/20."),
    }
    (OUT / "scale_sensitivity_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
