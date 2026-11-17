"""Probe: how much of the label-noise damage is recoverable in principle?

Before designing a cleaning method we need the ceiling. If perfect knowledge of
which labels were corrupted buys almost nothing, then the damage comes from
losing shots, not from corrupted prototypes, and no cleaning method can help.
Conversely a large oracle gain means the signal is there to be found.

Reference points per (domain, split, ratio, rule):
  clean        - no corruption
  noisy        - corrupted, nothing done
  oracle_drop  - corrupted rows removed using ground truth
  random_drop  - same number of rows removed at random

oracle_drop minus random_drop is the recoverable headroom: it isolates the value
of knowing WHICH rows are bad, as opposed to merely having fewer rows.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import protocol as P  # noqa: E402

ROOT = HERE.parent
SCALE = 10.0
N_WAY, K_SHOT, N_QUERY = 5, 5, 15
RATIOS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)


DATA = ROOT / "data"
SPLIT_SEED = 20261115


def split_indices(domain: str, p: int, y: np.ndarray, key2i: dict[str, int]) -> np.ndarray:
    """Same construction as full_experiment.split_indices, so folds match the main run."""
    if domain == "DTD":
        meta = DATA / "dtd" / "labels" / f"train{p}.txt"
        lines = [ln.strip() for ln in meta.read_text(encoding="utf-8").splitlines() if ln.strip()]
        idx = np.asarray([key2i[ln] for ln in lines], dtype=np.int64)
        if len(idx) != 1880:
            raise RuntimeError(f"DTD partition {p}: expected 1880 rows, got {len(idx)}")
        return idx
    # EuroSAT: ten disjoint per-class stratified folds, shared seed with the main run.
    rng = np.random.default_rng(SPLIT_SEED)
    by_class: dict[int, list[int]] = {}
    for i, c in enumerate(y):
        by_class.setdefault(int(c), []).append(i)
    fold: list[int] = []
    for c in sorted(by_class):
        order = np.asarray(by_class[c], dtype=np.int64)
        rng.shuffle(order)
        fold.extend(np.array_split(order, 10)[p - 1].tolist())
    return np.asarray(sorted(fold), dtype=np.int64)


def proto_predict(sx, sy, qx):
    classes = np.unique(sy)
    protos = np.stack([sx[sy == c].mean(0) for c in classes])
    d = ((qx[:, None, :] - protos[None, :, :]) ** 2).sum(2)
    return classes[d.argmin(1)]


def snn_predict(sx, sy, qx, scale: float = SCALE):
    classes = np.unique(sy)
    sx_n = sx / (np.linalg.norm(sx, axis=1, keepdims=True) + 1e-12)
    qx_n = qx / (np.linalg.norm(qx, axis=1, keepdims=True) + 1e-12)
    sim = qx_n @ sx_n.T
    att = np.exp(scale * sim)
    att /= att.sum(1, keepdims=True)
    cls_att = np.stack([att[:, sy == c].sum(1) for c in classes])
    return classes[cls_att.argmax(0)]


RULES = {"proto": proto_predict, "snn": snn_predict}


def corrupt(sy: np.ndarray, ratio: float, rng: np.random.Generator,
            max_tries: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """Replace a fraction of support labels with a wrong class. Returns labels and flipped mask.

    The flip set is redrawn until every class keeps at least one row. Without
    that retry an entire class can lose all five rows at high flip rates, its
    prototype becomes NaN, and np.argmin then maps every query to that NaN
    column - a silently invalid episode rather than a hard error.
    """
    sy2 = sy.copy()
    flipped = np.zeros(len(sy), dtype=bool)
    if ratio <= 0:
        return sy2, flipped
    n = max(1, int(round(len(sy) * ratio)))
    classes = np.unique(sy)
    for _ in range(max_tries):
        sy2 = sy.copy()
        flipped = np.zeros(len(sy), dtype=bool)
        for i in rng.choice(len(sy), size=n, replace=False):
            sy2[i] = rng.choice(classes[classes != sy[i]])
            flipped[i] = True
        if all((sy2 == c).any() for c in classes):
            return sy2, flipped
    raise RuntimeError(f"could not place {n} flips without emptying a class")


def run_domain(name: str, x: np.ndarray, y: np.ndarray, key2i: dict[str, int],
               n_episodes: int = 100) -> list[dict]:
    rows: list[dict] = []
    for fi in range(1, 11):
        test_idx = split_indices(name, fi, y, key2i)
        tx, ty = x[test_idx], y[test_idx]
        rng = np.random.default_rng(71_000 + fi)
        for ratio in RATIOS:
            accs = {f"{m}|{r}": [] for m in ("clean", "noisy", "oracle", "random") for r in RULES}
            for _ in range(n_episodes):
                ep = P.make_episode(tx, ty, N_WAY, K_SHOT, N_QUERY, rng)
                sx, sy, qx, qy = ep.support_x, ep.support_y, ep.query_x, ep.query_y
                sy_n, flipped = corrupt(sy, ratio, rng)
                n_bad = int(flipped.sum())

                keep_oracle = ~flipped
                keep_random = np.ones(len(sy), dtype=bool)
                if n_bad > 0:
                    drop = rng.choice(len(sy), size=n_bad, replace=False)
                    keep_random[drop] = False

                variants = {
                    "clean": (sx, sy),
                    "noisy": (sx, sy_n),
                    "oracle": (sx[keep_oracle], sy_n[keep_oracle]),
                    "random": (sx[keep_random], sy_n[keep_random]),
                }
                for mname, (vx, vy) in variants.items():
                    if len(np.unique(vy)) < 2:
                        for rname in RULES:
                            accs[f"{mname}|{rname}"].append(np.nan)
                        continue
                    for rname, fn in RULES.items():
                        pred = fn(vx, vy, qx)
                        accs[f"{mname}|{rname}"].append(float((pred == qy).mean()))

            for key, vals in accs.items():
                mname, rname = key.split("|")
                rows.append(
                    {
                        "domain": name,
                        "split": fi,
                        "ratio": ratio,
                        "method": mname,
                        "rule": rname,
                        "accuracy": float(np.nanmean(vals)),
                        "n_episodes": n_episodes,
                        "n_nan": int(np.isnan(vals).sum()),
                    }
                )
        print(f"  {name} split {fi} done", flush=True)
    return rows


def main() -> None:
    out = ROOT / "oracle_probe_output"
    out.mkdir(exist_ok=True)
    all_rows: list[dict] = []
    for name, path in (
        ("EuroSAT_RGB", ROOT / "full_output" / "eurosat_all_features.npz"),
        ("DTD", ROOT / "full_output" / "dtd_all_features.npz"),
    ):
        d = np.load(path, allow_pickle=True)
        keys = [str(k) for k in d["keys"]]
        key2i = {k: i for i, k in enumerate(keys)}
        print(f"{name}: {d['x'].shape}", flush=True)
        all_rows += run_domain(name, d["x"].astype(np.float64), d["y"], key2i)

    (out / "oracle_probe.json").write_text(
        json.dumps(all_rows, indent=1, ensure_ascii=False), encoding="utf-8"
    )

    print("\n=== mean accuracy over splits ===")
    print(f"{'domain':13s} {'ratio':>5s} {'rule':>5s} {'clean':>7s} {'noisy':>7s} {'oracle':>7s} {'random':>7s} {'headroom':>8s}")
    for dom in ("EuroSAT_RGB", "DTD"):
        for rule in ("proto", "snn"):
            for ratio in RATIOS:
                sel = [r for r in all_rows if r["domain"] == dom and r["rule"] == rule and r["ratio"] == ratio]
                if not sel:
                    continue
                g = lambda m: float(np.mean([r["accuracy"] for r in sel if r["method"] == m]))  # noqa: E731
                print(
                    f"{dom:13s} {ratio:5.1f} {rule:>5s} {g('clean'):7.4f} {g('noisy'):7.4f} "
                    f"{g('oracle'):7.4f} {g('random'):7.4f} {g('oracle') - g('random'):+8.4f}"
                )


if __name__ == "__main__":
    main()
