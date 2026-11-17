"""Cluster-robust variance re-estimation for the 16,000-row main experiment.

Reads full_output/results_full.csv only. It does NOT re-run any experiment and
does NOT generate new performance numbers. All estimates are re-aggregations
of the recorded rows.

Three interval definitions are compared for every quantity:
  nominal    : treat all 1,000 episodes of a cell as independent (t interval);
               for effects, delta method on the pooled cell variances.
  split_t    : treat the 10 splits as the independent units (t, df = 9).
  cluster_bt : percentile bootstrap that resamples the 10 split-level values
               with replacement (B = 20,000).

Also reports one-way random-effects ANOVA ICC and the design effect, which
quantify how much the nominal interval understates uncertainty.
"""
from __future__ import annotations

import csv
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "full_output" / "results_full.csv"
EFFECTS_CSV = ROOT / "full_output" / "effects_by_split.csv"
OUT = ROOT / "robust_output"
OUT.mkdir(parents=True, exist_ok=True)

FACTORS = ("support_label_noise", "query_ood", "source_free", "incomplete_new_class")
ACTIVE = ("support_label_noise", "query_ood", "incomplete_new_class")
B = 20_000
BOOT_SEED = 20261115


def make_label(bits: tuple[int, ...]) -> str:
    state = {"source_free": 1}
    state.update(dict(zip(ACTIVE, bits)))
    return "+".join(f for f in FACTORS if state.get(f, 0)) or "clean"


def read_rows() -> list[dict]:
    rows: list[dict] = []
    with SRC.open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            row = dict(r)
            row["split"] = int(row["split"])
            for k in FACTORS:
                row[k] = int(row[k])
            row["accuracy"] = float(row["accuracy"])
            a = row.get("auroc", "")
            row["auroc"] = None if a in ("", "None", None) else float(a)
            rows.append(row)
    return rows


def effect_estimates(cell_means: dict[str, float]) -> dict[str, float]:
    """Same definition as protocol.estimate_effects with source_free fixed at 1."""
    combos = list(itertools.product((0, 1), repeat=len(ACTIVE)))

    def cell(bits: tuple[int, ...]) -> float:
        return cell_means[make_label(bits)]

    out: dict[str, float] = {}
    for i, f in enumerate(ACTIVE):
        hi = [cell(b) for b in combos if b[i] == 1]
        lo = [cell(b) for b in combos if b[i] == 0]
        out[f"main::{f}"] = float(np.mean(hi) - np.mean(lo))
    for i, j in itertools.combinations(range(len(ACTIVE)), 2):
        same = [cell(b) for b in combos if (b[i] + b[j]) % 2 == 0]
        diff = [cell(b) for b in combos if (b[i] + b[j]) % 2 == 1]
        out[f"inter::{ACTIVE[i]}x{ACTIVE[j]}"] = float(np.mean(same) - np.mean(diff))
    return out


def t_interval(vals: np.ndarray) -> tuple[float, float, float, float]:
    a = np.asarray(vals, dtype=float)
    n = len(a)
    mean = float(a.mean())
    se = float(a.std(ddof=1) / np.sqrt(n))
    t = float(stats.t.ppf(0.975, n - 1))
    return mean, mean - t * se, mean + t * se, se


def cluster_bootstrap(vals: np.ndarray, seed: int) -> tuple[float, float]:
    a = np.asarray(vals, dtype=float)
    n = len(a)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(B, n))
    means = a[idx].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def delta_nominal_effect(var_cells: dict[str, float]) -> float:
    """SE of a 2^3 main effect or interaction if all episodes were independent.

    Each effect is mean(4 cells) - mean(4 cells), so every one of the 8 cell
    means carries coefficient +/- 1/4.
    """
    return float(np.sqrt(sum(v / 16.0 for v in var_cells.values())))


def icc_one_way(groups: list[np.ndarray]) -> tuple[float, float, float]:
    g = len(groups)
    n = len(groups[0])
    grand = float(np.concatenate(groups).mean())
    gmeans = np.array([x.mean() for x in groups])
    msb = float(n * np.sum((gmeans - grand) ** 2) / (g - 1))
    msw = float(sum(np.sum((x - x.mean()) ** 2) for x in groups) / (g * (n - 1)))
    icc = (msb - msw) / (msb + (n - 1) * msw)
    deff = 1.0 + (n - 1) * icc
    return icc, deff, msb / msw if msw > 0 else float("nan")


def main() -> None:
    rows = read_rows()
    assert len(rows) == 16_000, f"expected 16,000 rows, got {len(rows)}"

    # (domain, split, condition) -> episode arrays
    cells: dict[tuple, dict[str, list]] = defaultdict(lambda: {"acc": [], "auroc": []})
    for r in rows:
        key = (r["domain"], r["split"], r["condition"])
        cells[key]["acc"].append(r["accuracy"])
        if r["auroc"] is not None:
            cells[key]["auroc"].append(r["auroc"])

    assert all(len(v["acc"]) == 100 for v in cells.values())
    assert len(cells) == 2 * 10 * 8

    # (domain, split) -> condition -> arrays
    by_ds: dict[tuple, dict[str, dict[str, np.ndarray]]] = defaultdict(dict)
    for (d, s, c), v in cells.items():
        by_ds[(d, s)][c] = {
            "acc": np.asarray(v["acc"], dtype=float),
            "auroc": np.asarray(v["auroc"], dtype=float) if v["auroc"] else None,
        }

    domains = sorted({d for d, s in by_ds})
    splits_per_domain = {d: sorted({s for dd, s in by_ds if dd == d}) for d in domains}
    assert all(len(v) == 10 for v in splits_per_domain.values())

    # ---- verification against the recorded per-split effect table -----------
    recorded = {}
    with EFFECTS_CSV.open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            recorded[(r["domain"], int(r["split"]))] = r

    recomputed_effects = {}
    max_abs_diff = 0.0
    for (d, s), condmap in by_ds.items():
        cm = {c: float(v["acc"].mean()) for c, v in condmap.items()}
        eff = effect_estimates(cm)
        recomputed_effects[(d, s)] = eff
        rec = recorded[(d, s)]
        for k, val in eff.items():
            max_abs_diff = max(max_abs_diff, abs(val - float(rec[k])))

    # ---- assemble all quantities -------------------------------------------
    # A quantity is identified by (domain, kind, name, metric).
    # kinds: cell_mean, effect, auroc_delta
    out_rows: list[dict] = []
    icc_rows: list[dict] = []

    all_labels = [make_label(b) for b in itertools.product((0, 1), repeat=3)]

    for d in domains:
        splits = splits_per_domain[d]

        # --- cell means (accuracy, 8 cells) ----------------------------------
        for lab in all_labels:
            pooled = np.concatenate([by_ds[(d, s)][lab]["acc"] for s in splits])
            split_vals = np.array([by_ds[(d, s)][lab]["acc"].mean() for s in splits])
            m_nom, lo_nom, hi_nom, se_nom = t_interval(pooled)
            m_st, lo_st, hi_st, se_st = t_interval(split_vals)
            lo_bt, hi_bt = cluster_bootstrap(split_vals, BOOT_SEED + len(out_rows))
            icc, deff, ratio = icc_one_way([by_ds[(d, s)][lab]["acc"] for s in splits])
            icc_rows.append({"domain": d, "metric": "accuracy", "cell": lab,
                             "icc": icc, "design_effect": deff, "msb_msw": ratio,
                             "n_clusters": 10, "n_per_cluster": 100})
            out_rows.append(dict(domain=d, kind="cell_mean", metric="accuracy", name=lab,
                                 estimate=m_st, nominal_lo=lo_nom, nominal_hi=hi_nom,
                                 split_t_lo=lo_st, split_t_hi=hi_st,
                                 cluster_boot_lo=lo_bt, cluster_boot_hi=hi_bt,
                                 nominal_se=se_nom, split_se=se_st,
                                 width_ratio_split_over_nominal=(hi_st - lo_st) / (hi_nom - lo_nom)))

        # --- effects (accuracy, 6 columns) -----------------------------------
        eff_names = list(recomputed_effects[(d, splits[0])].keys())
        pooled_cell_means = {}
        pooled_cell_vars = {}
        for lab in all_labels:
            arr = np.concatenate([by_ds[(d, s)][lab]["acc"] for s in splits])
            pooled_cell_means[lab] = float(arr.mean())
            pooled_cell_vars[lab] = float(arr.var(ddof=1) / len(arr))
        pooled_eff = effect_estimates(pooled_cell_means)
        se_eff_nom = delta_nominal_effect(pooled_cell_vars)
        for name in eff_names:
            split_vals = np.array([recomputed_effects[(d, s)][name] for s in splits])
            m_st, lo_st, hi_st, se_st = t_interval(split_vals)
            lo_bt, hi_bt = cluster_bootstrap(split_vals, BOOT_SEED + 1000 + len(out_rows))
            lo_nom = pooled_eff[name] - 1.96 * se_eff_nom
            hi_nom = pooled_eff[name] + 1.96 * se_eff_nom
            out_rows.append(dict(domain=d, kind="effect", metric="accuracy", name=name,
                                 estimate=m_st, nominal_lo=lo_nom, nominal_hi=hi_nom,
                                 split_t_lo=lo_st, split_t_hi=hi_st,
                                 cluster_boot_lo=lo_bt, cluster_boot_hi=hi_bt,
                                 nominal_se=se_eff_nom, split_se=se_st,
                                 width_ratio_split_over_nominal=(hi_st - lo_st) / (hi_nom - lo_nom)))

        # --- AUROC cell means (4 cells with query_ood on) ---------------------
        ood_labels = [lab for lab in all_labels if "query_ood" in lab]
        for lab in ood_labels:
            pooled = np.concatenate([by_ds[(d, s)][lab]["auroc"] for s in splits])
            split_vals = np.array([by_ds[(d, s)][lab]["auroc"].mean() for s in splits])
            m_nom, lo_nom, hi_nom, se_nom = t_interval(pooled)
            m_st, lo_st, hi_st, se_st = t_interval(split_vals)
            lo_bt, hi_bt = cluster_bootstrap(split_vals, BOOT_SEED + 2000 + len(out_rows))
            icc, deff, ratio = icc_one_way([by_ds[(d, s)][lab]["auroc"] for s in splits])
            icc_rows.append({"domain": d, "metric": "auroc", "cell": lab,
                             "icc": icc, "design_effect": deff, "msb_msw": ratio,
                             "n_clusters": 10, "n_per_cluster": 100})
            out_rows.append(dict(domain=d, kind="cell_mean", metric="auroc", name=lab,
                                 estimate=m_st, nominal_lo=lo_nom, nominal_hi=hi_nom,
                                 split_t_lo=lo_st, split_t_hi=hi_st,
                                 cluster_boot_lo=lo_bt, cluster_boot_hi=hi_bt,
                                 nominal_se=se_nom, split_se=se_st,
                                 width_ratio_split_over_nominal=(hi_st - lo_st) / (hi_nom - lo_nom)))

        # --- H2 quantity: AUROC(noise on) - AUROC(noise off), both with OOD --
        lab_off = make_label((0, 1, 0))
        lab_on = make_label((1, 1, 0))
        split_deltas = np.array([
            by_ds[(d, s)][lab_on]["auroc"].mean() - by_ds[(d, s)][lab_off]["auroc"].mean()
            for s in splits
        ])
        arr_off = np.concatenate([by_ds[(d, s)][lab_off]["auroc"] for s in splits])
        arr_on = np.concatenate([by_ds[(d, s)][lab_on]["auroc"] for s in splits])
        m_nom = float(arr_on.mean() - arr_off.mean())
        se_nom = float(np.sqrt(arr_on.var(ddof=1) / len(arr_on) + arr_off.var(ddof=1) / len(arr_off)))
        m_st, lo_st, hi_st, se_st = t_interval(split_deltas)
        lo_bt, hi_bt = cluster_bootstrap(split_deltas, BOOT_SEED + 3000)
        out_rows.append(dict(domain=d, kind="auroc_delta", metric="auroc",
                             name="H2: AUROC(OOD+noise) - AUROC(OOD only)",
                             estimate=m_st,
                             nominal_lo=m_nom - 1.96 * se_nom, nominal_hi=m_nom + 1.96 * se_nom,
                             split_t_lo=lo_st, split_t_hi=hi_st,
                             cluster_boot_lo=lo_bt, cluster_boot_hi=hi_bt,
                             nominal_se=se_nom, split_se=se_st,
                             width_ratio_split_over_nominal=(hi_st - lo_st) / (2 * 1.96 * se_nom)))

    # ---- post-process: mark degenerate (structurally zero) quantities -------
    for r in out_rows:
        r["degenerate"] = bool((r["split_t_hi"] - r["split_t_lo"]) < 1e-9)

    # ---- write CSVs ----------------------------------------------------------
    fields = list(out_rows[0].keys())
    with (OUT / "robust_intervals.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in out_rows:
            w.writerow({k: (round(v, 8) if isinstance(v, float) else v) for k, v in r.items()})

    icc_fields = list(icc_rows[0].keys())
    with (OUT / "icc_design_effect.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=icc_fields)
        w.writeheader()
        for r in icc_rows:
            w.writerow({k: (round(v, 8) if isinstance(v, float) else v) for k, v in r.items()})

    icc_vals = [r["icc"] for r in icc_rows]
    deff_vals = [r["design_effect"] for r in icc_rows]
    non_deg = [r for r in out_rows if not r["degenerate"]]
    widths_non_deg = [r["width_ratio_split_over_nominal"] for r in non_deg]

    def excludes_zero(r: dict, lo: str, hi: str) -> bool:
        return (r[lo] > 0) or (r[hi] < 0)

    # Only the three non-zero accuracy effects and the two H2 deltas are
    # meaningful for a sign-stability check. Degenerate effects are excluded.
    sign_rows = [r for r in non_deg
                 if (r["kind"] == "effect" and abs(r["estimate"]) > 1e-9)
                 or r["kind"] == "auroc_delta"]
    sign_stability = {
        "n_checked": len(sign_rows),
        "nominal_all_exclude_zero": all(excludes_zero(r, "nominal_lo", "nominal_hi") for r in sign_rows),
        "split_t_all_exclude_zero": all(excludes_zero(r, "split_t_lo", "split_t_hi") for r in sign_rows),
        "cluster_boot_all_exclude_zero": all(excludes_zero(r, "cluster_boot_lo", "cluster_boot_hi") for r in sign_rows),
    }

    summary = {
        "source": "full_output/results_full.csv",
        "n_rows": len(rows), "n_cells": len(cells),
        "n_clusters_per_domain": 10, "n_episodes_per_cluster_per_cell": 100,
        "bootstrap_B": B, "bootstrap_seed_base": BOOT_SEED,
        "verification_max_abs_diff_vs_effects_by_split_csv": max_abs_diff,
        "icc_min": min(icc_vals), "icc_max": max(icc_vals),
        "icc_median": float(np.median(icc_vals)),
        "icc_negative_cells": sum(1 for v in icc_vals if v < 0),
        "icc_cells_total": len(icc_vals),
        "design_effect_min": min(deff_vals), "design_effect_max": max(deff_vals),
        "n_quantities": len(out_rows),
        "n_degenerate": len(out_rows) - len(non_deg),
        "width_ratio_non_degenerate_min": min(widths_non_deg),
        "width_ratio_non_degenerate_max": max(widths_non_deg),
        "n_width_ratio_gt_1": sum(1 for v in widths_non_deg if v > 1),
        "n_width_ratio_lt_1": sum(1 for v in widths_non_deg if v < 1),
        "sign_stability": sign_stability,
        "limits": [
            "only 10 clusters per domain; both t and bootstrap intervals rest on few independent units",
            "DTD official partitions overlap (split1 and split10 share 470 images), so clusters are not strictly independent",
            "EuroSAT folds are disjoint but drawn from the same 27,000-image pool",
            "cluster bootstrap resamples split-level values and does not resample episodes within a split",
            "negative ANOVA ICC estimates are reported as estimated, not truncated at zero",
            "all numbers are re-aggregations of recorded rows; no experiment was re-run",
        ],
    }
    (OUT / "robust_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("rows:", len(rows), "cells:", len(cells))
    print("verification max abs diff vs effects_by_split.csv:", max_abs_diff)
    print("ICC range:", round(min(icc_vals), 6), "..", round(max(icc_vals), 6),
          "| median", round(float(np.median(icc_vals)), 6),
          "| negative cells", sum(1 for v in icc_vals if v < 0), "/", len(icc_vals))
    print("design effect range:", round(min(deff_vals), 4), "..", round(max(deff_vals), 4))
    print("quantities:", len(out_rows), "| degenerate:", len(out_rows) - len(non_deg))
    print("width ratio (non-degenerate):", round(min(widths_non_deg), 3), "..",
          round(max(widths_non_deg), 3), "| gt1:", sum(1 for v in widths_non_deg if v > 1),
          "| lt1:", sum(1 for v in widths_non_deg if v < 1))
    print("sign stability:", sign_stability)
    print("--- per-cell ICC (accuracy, clean baseline and noise cell) ---")
    for r in icc_rows:
        if r["metric"] == "accuracy" and r["cell"] in ("source_free", "support_label_noise+source_free"):
            print(f"  {r['domain']:12s} {r['cell']:45s} ICC={r['icc']:+.6f} deff={r['design_effect']:.4f}")
    print("--- three non-zero accuracy effects (EuroSAT) ---")
    for r in out_rows:
        if r["domain"] == "EuroSAT_RGB" and r["kind"] == "effect" and abs(r["estimate"]) > 1e-9:
            print(f"  {r['name']:52s} est={r['estimate']:.6f} "
                  f"nom=[{r['nominal_lo']:.6f},{r['nominal_hi']:.6f}] "
                  f"split=[{r['split_t_lo']:.6f},{r['split_t_hi']:.6f}] "
                  f"boot=[{r['cluster_boot_lo']:.6f},{r['cluster_boot_hi']:.6f}]")
    print("--- H2 AUROC delta with cluster intervals ---")
    for r in out_rows:
        if r["kind"] == "auroc_delta":
            print(f"  {r['domain']:12s} est={r['estimate']:.6f} "
                  f"nominal=[{r['nominal_lo']:.6f},{r['nominal_hi']:.6f}] "
                  f"split_t=[{r['split_t_lo']:.6f},{r['split_t_hi']:.6f}] "
                  f"cluster_boot=[{r['cluster_boot_lo']:.6f},{r['cluster_boot_hi']:.6f}]")
    print("--- clean baseline cell means ---")
    for r in out_rows:
        if r["kind"] == "cell_mean" and r["metric"] == "accuracy" and r["name"] == "source_free":
            print(f"  {r['domain']:12s} est={r['estimate']:.6f} "
                  f"nominal=[{r['nominal_lo']:.6f},{r['nominal_hi']:.6f}] "
                  f"split_t=[{r['split_t_lo']:.6f},{r['split_t_hi']:.6f}] "
                  f"cluster_boot=[{r['cluster_boot_lo']:.6f},{r['cluster_boot_hi']:.6f}] "
                  f"width_ratio={r['width_ratio_split_over_nominal']:.3f}")


if __name__ == "__main__":
    main()
