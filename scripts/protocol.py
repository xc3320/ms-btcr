"""Factorial evaluation protocol for the paper draft (v2, section 3).

Scope of this module:
  - build N-way K-shot episodes from an embedding matrix,
  - switch four degradation factors on/off independently,
  - attach the six mandatory record fields to every reported number,
  - compute accuracy, OOD AUROC, main effects and two-way interactions.

This module does NOT extract image features. The caller supplies an embedding
matrix. That keeps the protocol testable on CPU before the GPU stack is ready.

All numbers produced by smoke_test.py come from SYNTHETIC random features.
They are a code self-check, not research results, and must never be copied
into the manuscript.
"""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
from scipy import stats

FACTORS = (
    "support_label_noise",
    "query_ood",
    "source_free",
    "incomplete_new_class",
)

SIX_FIELDS = (
    "dataset_split",
    "noise_type_ratio",
    "source_model_weight",
    "method_code_version",
    "cited_paper_local_version",
    "seed_repeats",
)


@dataclass
class Condition:
    """One cell of the factorial grid. Every field is a 0/1 switch."""

    support_label_noise: int = 0
    query_ood: int = 0
    source_free: int = 0
    incomplete_new_class: int = 0

    def key(self) -> tuple[int, int, int, int]:
        return (
            self.support_label_noise,
            self.query_ood,
            self.source_free,
            self.incomplete_new_class,
        )

    def label(self) -> str:
        return "+".join(f for f, v in zip(FACTORS, self.key()) if v) or "clean"


@dataclass
class Record:
    """The six fields that must travel with every reported number."""

    dataset_split: str
    noise_type_ratio: str
    source_model_weight: str
    method_code_version: str
    cited_paper_local_version: str
    seed_repeats: str

    def as_row(self) -> dict:
        return asdict(self)


@dataclass
class Episode:
    support_x: np.ndarray
    support_y: np.ndarray
    query_x: np.ndarray
    query_y: np.ndarray
    query_is_ood: np.ndarray
    ood_pool_x: np.ndarray

    def clean_query_mask(self) -> np.ndarray:
        return ~self.query_is_ood


def make_episode(
    x: np.ndarray,
    y: np.ndarray,
    n_way: int,
    k_shot: int,
    n_query: int,
    rng: np.random.Generator,
    ood_pool_x: np.ndarray | None = None,
) -> Episode:
    """Sample an N-way K-shot episode. OOD samples are drawn from held-out classes."""
    classes = np.unique(y)
    if len(classes) < n_way + 1:
        raise ValueError("need at least n_way+1 classes to reserve an OOD pool")
    chosen = rng.choice(classes, size=n_way, replace=False)
    others = np.setdiff1d(classes, chosen)

    sx, sy, qx, qy = [], [], [], []
    for c in chosen:
        idx = np.flatnonzero(y == c)
        rng.shuffle(idx)
        picked = idx[: k_shot + n_query]
        sx.append(x[picked[:k_shot]])
        sy.append(np.full(k_shot, c, dtype=np.int64))
        qx.append(x[picked[k_shot : k_shot + n_query]])
        qy.append(np.full(n_query, c, dtype=np.int64))

    if ood_pool_x is None:
        ood_idx = np.flatnonzero(np.isin(y, others))
        ood_pool_x = x[ood_idx]

    return Episode(
        support_x=np.concatenate(sx),
        support_y=np.concatenate(sy),
        query_x=np.concatenate(qx),
        query_y=np.concatenate(qy),
        query_is_ood=np.zeros(n_way * n_query, dtype=bool),
        ood_pool_x=ood_pool_x,
    )


def apply_factors(
    ep: Episode,
    cond: Condition,
    cfg: dict,
    rng: np.random.Generator,
) -> Episode:
    """Apply the four independent switches. Each one changes one thing only."""
    sx, sy = ep.support_x.copy(), ep.support_y.copy()
    qx, qy, qood = ep.query_x.copy(), ep.query_y.copy(), ep.query_is_ood.copy()

    applied = []
    if cond.support_label_noise:
        ratio = float(cfg["support_noise_ratio"])
        chosen = np.unique(sy)
        n_flip = max(1, int(round(len(sy) * ratio)))
        for i in rng.choice(len(sy), size=n_flip, replace=False):
            wrong = chosen[chosen != sy[i]]
            sy[i] = rng.choice(wrong)
        applied.append(f"support_label_replace@{ratio:.2f}")

    if cond.incomplete_new_class:
        ratio = float(cfg["incomplete_ratio"])
        n_drop = max(1, int(round(len(sy) * ratio)))
        keep = np.setdiff1d(np.arange(len(sy)), rng.choice(len(sy), size=n_drop, replace=False))
        sx, sy = sx[keep], sy[keep]
        applied.append(f"support_label_missing@{ratio:.2f}")

    if cond.query_ood:
        ratio = float(cfg["query_ood_ratio"])
        n_ood = max(1, int(round(len(qx) * ratio)))
        pool_idx = rng.choice(len(ep.ood_pool_x), size=n_ood, replace=False)
        ood_x = ep.ood_pool_x[pool_idx]
        qx = np.concatenate([qx, ood_x])
        qy = np.concatenate([qy, np.full(n_ood, -1, dtype=np.int64)])
        qood = np.concatenate([qood, np.ones(n_ood, dtype=bool)])
        applied.append(f"query_ood_mix@{ratio:.2f}")

    if cond.source_free:
        applied.append("source_data_withheld")

    return Episode(sx, sy, qx, qy, qood, ep.ood_pool_x)


def prototypical_predict(support_x: np.ndarray, support_y: np.ndarray, query_x: np.ndarray):
    """Distance-to-prototype classifier. Returns predicted labels and OOD scores."""
    classes = np.unique(support_y)
    protos = np.stack([support_x[support_y == c].mean(axis=0) for c in classes])
    dist = ((query_x[:, None, :] - protos[None, :, :]) ** 2).sum(axis=2)
    pred = classes[dist.argmin(axis=1)]
    ood_score = dist.min(axis=1)
    return pred, ood_score


def accuracy_and_auroc(ep: Episode) -> tuple[float, float | None]:
    """ACC on in-distribution queries; AUROC for separating mixed-in OOD samples."""
    pred, score = prototypical_predict(ep.support_x, ep.support_y, ep.query_x)
    mask = ep.clean_query_mask()
    acc = float((pred[mask] == ep.query_y[mask]).mean()) if mask.any() else float("nan")
    if ep.query_is_ood.any() and mask.any():
        labels = np.concatenate([np.zeros(mask.sum()), np.ones(ep.query_is_ood.sum())])
        scores = np.concatenate([score[mask], score[ep.query_is_ood]])
        try:
            auroc = float(stats.rankdata(scores)[labels == 1].mean())
            auroc = (auroc - (ep.query_is_ood.sum() + 1) / 2) / mask.sum()
        except Exception:
            auroc = None
    else:
        auroc = None
    return acc, auroc


def all_conditions() -> list[Condition]:
    return [Condition(*bits) for bits in itertools.product((0, 1), repeat=len(FACTORS))]


def run_grid(
    x: np.ndarray,
    y: np.ndarray,
    cfg: dict,
    rec_factory: Callable[[Condition, int], Record],
    ood_pool_x: np.ndarray | None = None,
    conditions: Iterable[Condition] | None = None,
) -> list[dict]:
    """Run the grid. Pass conditions to restrict to a feasible subset.

    The minimum pilot can only fix source_free=1 (no accessible source-domain
    training set), so it passes 8 conditions instead of the full 16.
    """
    rows: list[dict] = []
    for cond in (all_conditions() if conditions is None else conditions):
        for seed in range(int(cfg["n_repeats"])):
            rng = np.random.default_rng(int(cfg["base_seed"]) + seed)
            ep = make_episode(
                x, y,
                n_way=int(cfg["n_way"]),
                k_shot=int(cfg["k_shot"]),
                n_query=int(cfg["n_query"]),
                rng=rng,
                ood_pool_x=ood_pool_x,
            )
            ep = apply_factors(ep, cond, cfg, rng)
            acc, auroc = accuracy_and_auroc(ep)
            rec = rec_factory(cond, seed)
            row = {
                "condition": cond.label(),
                **{f: int(getattr(cond, f)) for f in FACTORS},
                "seed": seed,
                "accuracy": round(acc, 6),
                "auroc": None if auroc is None else round(auroc, 6),
                "n_support": int(len(ep.support_y)),
                "n_query": int(len(ep.query_y)),
                "n_ood": int(ep.query_is_ood.sum()),
                **rec.as_row(),
            }
            rows.append(row)
    return rows


def estimate_effects(rows: list[dict], metric: str = "accuracy") -> dict:
    """Main effects and two-way interactions over the ACTUAL design.

    A partial grid (e.g. the pilot fixes source_free=1 and runs only 8 cells)
    has no observations for the missing cells. Averaging over a hard-coded
    2^4 grid would propagate NaN through every estimate. Instead, detect
    which factors actually vary, hold the rest at their observed value, and
    compute effects over the resulting 2^k design.
    """
    active = [f for f in FACTORS if len({r[f] for r in rows}) > 1]
    fixed = {f: int(rows[0][f]) for f in FACTORS if f not in active}

    means = {
        c: float(np.mean([r[metric] for r in rows
                         if r["condition"] == c and r[metric] is not None]))
        for c in {r["condition"] for r in rows}
    }

    def cell(bits: tuple[int, ...]) -> float:
        state = dict(fixed)
        state.update(dict(zip(active, bits)))
        return means.get(Condition(**state).label(), float("nan"))

    combos = list(itertools.product((0, 1), repeat=len(active)))
    main = {}
    for i, f in enumerate(active):
        hi = [cell(b) for b in combos if b[i] == 1]
        lo = [cell(b) for b in combos if b[i] == 0]
        main[f] = round(float(np.mean(hi) - np.mean(lo)), 6)

    inter = {}
    for i, j in itertools.combinations(range(len(active)), 2):
        same = [cell(b) for b in combos if (b[i] + b[j]) % 2 == 0]
        diff = [cell(b) for b in combos if (b[i] + b[j]) % 2 == 1]
        inter[f"{active[i]}x{active[j]}"] = round(float(np.mean(same) - np.mean(diff)), 6)

    return {
        "design": f"2^{len(active)} partial grid, {len(combos)} cells",
        "active_factors": active,
        "fixed_factors": fixed,
        "cell_means": {k: round(v, 6) for k, v in means.items()},
        "main_effects": main,
        "two_way_interactions": inter,
    }


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def file_fingerprint(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return "sha256:" + h.hexdigest()[:16]
