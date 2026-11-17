"""Second independent inference rule: Matching-Networks-style soft nearest neighbour.

Reuses the SAME cached features, splits, seeds and episode sampling as
full_experiment.py. Only the prediction rule changes. protocol.py is imported
read-only and is NOT modified: its fingerprint is recorded in the six mandatory
fields of the main experiment and must stay stable.

Rule (Vinyals et al., NIPS 2016, arXiv:1606.04080, attention over support set):
  L2-normalise support and query features; cosine similarity matrix;
  softmax over support samples; sum attention by class; predict argmax class.
  OOD score = 1 - max class attention (larger means more likely OOD).
A fixed scale SCALE multiplies cosine before softmax. SCALE=1 reproduces the
unscaled paper formula and degenerates into class-size voting on frozen
ImageNet features, so the main run uses SCALE=10 and a scale sensitivity check
is reported separately.

No official DeepMind code repository was located; this is an implementation of
the published inference rule, not a reproduction of the paper's training or
numbers. All six record fields are still attached to every row.
"""
from __future__ import annotations

import csv
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats

from protocol import (
    Condition, Episode, Record, SIX_FIELDS, apply_factors,
    estimate_effects, file_fingerprint, make_episode, write_csv,
)
import full_experiment as F

ROOT = Path(__file__).resolve().parent
OUT = ROOT.parent / "second_baseline_output"
FULL = ROOT.parent / "full_output"
OUT.mkdir(parents=True, exist_ok=True)

FEATURES = {
    "EuroSAT_RGB": FULL / "eurosat_all_features.npz",
    "DTD": FULL / "dtd_all_features.npz",
}

SMOKE = len(sys.argv) > 1 and sys.argv[1].strip().lower() == "smoke"
SCALE = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0
N_REPEATS = 5 if SMOKE else F.N_REPEATS
N_SPLITS = 1 if SMOKE else F.N_SPLITS
DOMAINS = ["EuroSAT_RGB"] if SMOKE else list(FEATURES)
CONDS = F.conditions_8()
B = 20_000
BOOT_SEED = 20261115
CODE_ID = file_fingerprint(ROOT / "secondbaseline_experiment.py")


def matching_predict(support_x: np.ndarray, support_y: np.ndarray,
                     query_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Attention-weighted nearest-neighbour rule. Returns predictions and OOD scores."""
    eps = 1e-12
    sx = support_x / (np.linalg.norm(support_x, axis=1, keepdims=True) + eps)
    qx = query_x / (np.linalg.norm(query_x, axis=1, keepdims=True) + eps)
    sim = qx @ sx.T
    w = np.exp(SCALE * (sim - sim.max(axis=1, keepdims=True)))
    w = w / w.sum(axis=1, keepdims=True)
    classes = np.unique(support_y)
    probs = np.stack([w[:, support_y == c].sum(axis=1) for c in classes], axis=1)
    pred = classes[probs.argmax(axis=1)]
    ood_score = 1.0 - probs.max(axis=1)
    return pred, ood_score


def accuracy_and_auroc(ep: Episode) -> tuple[float, float | None]:
    """Identical metric definition to protocol.accuracy_and_auroc, different rule."""
    pred, score = matching_predict(ep.support_x, ep.support_y, ep.query_x)
    mask = ep.clean_query_mask()
    acc = float((pred[mask] == ep.query_y[mask]).mean()) if mask.any() else float("nan")
    if ep.query_is_ood.any() and mask.any():
        labels = np.concatenate([np.zeros(mask.sum()),
                                 np.ones(int(ep.query_is_ood.sum()))])
        scores = np.concatenate([score[mask], score[ep.query_is_ood]])
        try:
            auroc = float(stats.rankdata(scores)[labels == 1].mean())
            auroc = (auroc - (ep.query_is_ood.sum() + 1) / 2) / mask.sum()
        except Exception:
            auroc = None
    else:
        auroc = None
    return acc, auroc


def run_grid_matching(x: np.ndarray, y: np.ndarray, cfg: dict,
                      rec_factory, conditions) -> list[dict]:
    """Byte-for-byte the same episode sampling as protocol.run_grid."""
    rows: list[dict] = []
    for cond in conditions:
        for seed in range(int(cfg["n_repeats"])):
            rng = np.random.default_rng(int(cfg["base_seed"]) + seed)
            ep = make_episode(x, y, n_way=int(cfg["n_way"]),
                              k_shot=int(cfg["k_shot"]),
                              n_query=int(cfg["n_query"]), rng=rng)
            ep = apply_factors(ep, cond, cfg, rng)
            acc, auroc = accuracy_and_auroc(ep)
            rec = rec_factory(cond, seed)
            rows.append({
                "condition": cond.label(),
                **{f: int(getattr(cond, f)) for f in
                   ("support_label_noise", "query_ood", "source_free",
                    "incomplete_new_class")},
                "seed": seed,
                "accuracy": round(acc, 6),
                "auroc": None if auroc is None else round(auroc, 6),
                "n_support": int(len(ep.support_y)),
                "n_query": int(len(ep.query_y)),
                "n_ood": int(ep.query_is_ood.sum()),
                **rec.as_row(),
            })
    return rows


def rec_factory_for(domain: str, p: int, base_seed: int):
    split_note = F.SPLIT_NOTE[domain].format(p=p)

    def make(cond: Condition, seed: int) -> Record:
        on = [f for f in ("support_label_noise", "query_ood",
                          "source_free", "incomplete_new_class")
              if getattr(cond, f)]
        ratios = []
        if cond.support_label_noise:
            ratios.append(f"support_replace={F.CFG['support_noise_ratio']}")
        if cond.query_ood:
            ratios.append(f"query_ood={F.CFG['query_ood_ratio']}")
        if cond.incomplete_new_class:
            ratios.append(f"support_missing={F.CFG['incomplete_ratio']}")
        return Record(
            dataset_split=(f"{domain};{split_note};N={F.CFG['n_way']},"
                           f"K={F.CFG['k_shot']},Q={F.CFG['n_query']};"
                           f"episode_seed={base_seed+seed}"),
            noise_type_ratio=";".join(on) + "|" + ";".join(ratios),
            source_model_weight="torchvision/resnet18;IMAGENET1K_V1;sha256:f37072fd47e89c5e",
            method_code_version=f"secondbaseline_experiment.py:{CODE_ID};scale={SCALE}",
            cited_paper_local_version=(
                "Vinyals et al., NIPS 2016; arXiv:1606.04080; "
                "attention-over-support inference rule; no official code repo located"),
            seed_repeats=f"seed={base_seed+seed};repeats={N_REPEATS}",
        )
    return make


def load(npz: Path):
    z = np.load(npz, allow_pickle=False)
    keys = [str(k) for k in z["keys"]]
    return z["x"], z["y"], keys, {k: i for i, k in enumerate(keys)}


def t_ci(vals: np.ndarray) -> dict:
    a = np.asarray(vals, dtype=float)
    n = len(a)
    mean = float(a.mean())
    sd = float(a.std(ddof=1)) if n > 1 else float("nan")
    if n > 1 and sd > 0:
        half = float(stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n))
    else:
        half = 0.0
    return {"mean": round(mean, 6), "ci95_low": round(mean - half, 6),
            "ci95_high": round(mean + half, 6), "sd": round(sd, 6), "n": n}


def cluster_boot(vals: np.ndarray, seed: int) -> tuple[float, float]:
    a = np.asarray(vals, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(B, len(a)))
    means = a[idx].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def cell_mean(rows: list[dict], label: str, metric: str) -> float | None:
    vals = [r[metric] for r in rows if r["condition"] == label and r[metric] is not None]
    return float(np.mean(vals)) if vals else None


def main() -> None:
    started = time.time()
    all_rows: list[dict] = []
    effect_rows: list[dict] = []

    for domain in DOMAINS:
        x_all, y_all, keys, key2i = load(FEATURES[domain])
        for p in range(1, N_SPLITS + 1):
            idx = F.split_indices(domain, p, y_all, keys, key2i)
            x, y = x_all[idx], y_all[idx]
            base_seed = 20261115 + p * 100000
            cfg = dict(F.CFG, n_repeats=N_REPEATS, base_seed=base_seed)
            rows = run_grid_matching(x, y, cfg, rec_factory_for(domain, p, base_seed), CONDS)
            rows = [{"domain": domain, "split": p, **r} for r in rows]
            all_rows.extend(rows)
            eff = estimate_effects(rows, "accuracy")
            er = {"domain": domain, "split": p, "n_images": int(len(idx)),
                  "design": eff["design"]}
            er.update({f"main::{k}": v for k, v in eff["main_effects"].items()})
            er.update({f"inter::{k}": v for k, v in eff["two_way_interactions"].items()})
            a_only = cell_mean(rows, "query_ood+source_free", "auroc")
            a_noisy = cell_mean(rows, "support_label_noise+query_ood+source_free", "auroc")
            er["h2_auroc_delta"] = round(a_noisy - a_only, 6)
            effect_rows.append(er)
            print(f"  {domain:12s} split {p:02d}: {len(rows)} rows "
                  f"acc_mean={np.mean([r['accuracy'] for r in rows]):.4f} "
                  f"noise_effect={eff['main_effects']['support_label_noise']:+.4f} "
                  f"h2_delta={er['h2_auroc_delta']:+.4f}", flush=True)

    suffix = "_smoke" if SMOKE else ""
    write_csv(all_rows, OUT / f"results_knn{suffix}.csv")
    write_csv(effect_rows, OUT / f"effects_by_split_knn{suffix}.csv")

    six_ok = all(bool(r[k]) for r in all_rows for k in SIX_FIELDS)
    print("six fields complete:", six_ok, "| rows:", len(all_rows), flush=True)

    if SMOKE:
        print("SMOKE ONLY - no comparison or summary produced.", flush=True)
        return

    # ---- paired comparison against the prototypical-rule main experiment ----
    with (FULL / "results_full.csv").open(encoding="utf-8-sig", newline="") as f:
        proto = {}
        for r in csv.DictReader(f):
            k = (r["domain"], int(r["split"]), r["condition"], int(r["seed"]))
            proto[k] = (float(r["accuracy"]),
                        None if r["auroc"] in ("", "None") else float(r["auroc"]))

    pair_rows = []
    missing = 0
    for r in all_rows:
        k = (r["domain"], r["split"], r["condition"], r["seed"])
        if k not in proto:
            missing += 1
            continue
        pa, pu = proto[k]
        ka, ku = r["accuracy"], r["auroc"]
        pair_rows.append({
            "domain": r["domain"], "split": r["split"], "condition": r["condition"],
            "seed": r["seed"], "proto_acc": pa, "knn_acc": ka,
            "acc_diff": ka - pa,
            "proto_auroc": "" if pu is None else pu,
            "knn_auroc": "" if ku is None else ku,
            "auroc_diff": "" if (pu is None or ku is None) else ku - pu,
        })
    assert missing == 0, f"unmatched episodes: {missing}"
    with (OUT / "pairwise_diff.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(pair_rows[0]))
        w.writeheader()
        w.writerows(pair_rows)

    # ---- per-domain summary --------------------------------------------------
    domains = sorted(DOMAINS)
    summary = {
        "status": "second inference rule on identical episodes; not a training reproduction",
        "second_rule": "Matching Networks attention over support set (Vinyals et al., NIPS 2016)",
        "attention_scale": SCALE,
        "official_code_repo": "not located; only published inference rule implemented",
        "n_rows": len(all_rows), "n_splits": N_SPLITS,
        "n_repeats": N_REPEATS, "n_conditions": len(CONDS),
        "six_fields_complete": six_ok,
        "paired_episodes": len(pair_rows),
        "elapsed_seconds": round(time.time() - started, 1),
        "bootstrap_B": B,
        "domains": {},
    }

    for d in domains:
        sub = [r for r in effect_rows if r["domain"] == d]
        clean = [p for p in pair_rows if p["domain"] == d and p["condition"] == "source_free"]
        noise = [p for p in pair_rows if p["domain"] == d
                 and p["condition"] == "support_label_noise+source_free"]
        auroc_pairs = [p for p in pair_rows if p["domain"] == d and p["auroc_diff"] != ""]

        clean_diff = np.array([p["acc_diff"] for p in clean])
        noise_diff = np.array([p["acc_diff"] for p in noise])
        h2 = np.array([r["h2_auroc_delta"] for r in sub])
        boot_lo, boot_hi = cluster_boot(h2, BOOT_SEED + (0 if d == "DTD" else 1))

        eff_names = ["main::support_label_noise", "main::incomplete_new_class",
                     "inter::support_label_noisexincomplete_new_class"]
        eff_ci = {name: t_ci(np.array([float(r[name]) for r in sub]))
                  for name in eff_names}

        summary["domains"][d] = {
            "knn_clean_accuracy": round(float(np.mean(
                [r["accuracy"] for r in all_rows if r["domain"] == d
                 and r["condition"] == "source_free"])), 6),
            "proto_clean_accuracy": round(float(np.mean([p["proto_acc"] for p in clean])), 6),
            "paired_acc_diff_clean": t_ci(clean_diff),
            "paired_acc_diff_noise_cell": t_ci(noise_diff),
            "paired_auroc_diff_all_ood_cells": t_ci(np.array(
                [p["auroc_diff"] for p in auroc_pairs], dtype=float)),
            "knn_effects_split_t": eff_ci,
            "knn_h2_auroc_delta": {
                "mean": round(float(h2.mean()), 6),
                "split_t": t_ci(h2),
                "cluster_boot": [round(boot_lo, 6), round(boot_hi, 6)],
                "all_splits_negative": bool(np.all(h2 < 0)),
            },
        }

    direction = {
        "noise_effect_negative_both_domains": all(
            summary["domains"][d]["knn_effects_split_t"]["main::support_label_noise"]["mean"] < 0
            for d in domains),
        "incomplete_effect_negative_both_domains": all(
            summary["domains"][d]["knn_effects_split_t"]["main::incomplete_new_class"]["mean"] < 0
            for d in domains),
        "interaction_negative_both_domains": all(
            summary["domains"][d]["knn_effects_split_t"][
                "inter::support_label_noisexincomplete_new_class"]["mean"] < 0
            for d in domains),
        "h2_delta_negative_both_domains": all(
            summary["domains"][d]["knn_h2_auroc_delta"]["all_splits_negative"]
            for d in domains),
    }
    summary["direction_agreement"] = direction

    (OUT / "comparison_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
