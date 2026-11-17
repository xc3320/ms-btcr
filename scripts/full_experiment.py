"""Full repeated experiment: 2 domains x 10 splits x 8 conditions x 100 repeats.

Motivation (see 25_...md section 5): the two-domain pilot used 5 repeats per
cell and produced 95% CI half-widths averaging 0.1123, far too wide to say
anything about cross-domain similarity of the noise effect. This run raises
repeats to 100 and, more importantly, repeats the whole 2^3 design across 10
data splits per domain, so that split-to-split variability is measured.

Split definitions (NOT comparable across domains, recorded explicitly):
  * DTD          - the 10 OFFICIAL partitions, train split of each
                   (labels/train1..10.txt, 1880 images each)
  * EuroSAT_RGB  - 10 disjoint, per-class-stratified seeded folds built from
                   all 27,000 images (~2700 per fold), NOT an official split
                   (the 10 classes are imbalanced: 2000-3000 images each)

Both domains use ImageNet-pretrained ResNet-18 (fc removed) 512-d embeddings
and the prototypical-network rule in protocol.py. Every row carries the six
mandatory record fields. source_free is fixed at 1 (v3 section 3).

Limits that still block publication: no published baseline method reproduced,
split definitions differ across domains, only one feature extractor. See
26_...md.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image
from scipy import stats
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18

from protocol import (
    Condition, Record, SIX_FIELDS, estimate_effects, file_fingerprint,
    run_grid, write_csv,
)

ROOT = Path(__file__).resolve().parent
OUT = ROOT.parent / "full_output"
DATA = ROOT.parent / "data"

EUROSAT_ZIP_MD5 = "f46e308c4d50d4bf32fedad2d3d62f3b"
DTD_TAR_MD5 = "fff73e5086ae6bdbea199a49dfb8a4c1"

N_REPEATS = 100
N_SPLITS = 10
SPLIT_SEED = 20261115

CFG = {
    "n_way": 5,
    "k_shot": 5,
    "n_query": 15,
    "n_repeats": N_REPEATS,
    "base_seed": 20261115,
    "support_noise_ratio": 0.30,
    "incomplete_ratio": 0.20,
    "query_ood_ratio": 0.20,
}

SPLIT_NOTE = {
    "EuroSAT_RGB": "seeded-10-fold-disjoint-per-class-stratified;~2700-images-per-fold;NOT-official-split",
    "DTD": "official-partition-{p}-train;1880-images;40-per-class",
}


def md5_full(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_short(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()[:16]


def weight_cache_path() -> Path | None:
    d = Path(torch.hub.get_dir()) / "checkpoints"
    hits = sorted(d.glob("resnet18-*.pth")) if d.exists() else []
    return hits[0] if hits else None


class ImageList(Dataset):
    def __init__(self, paths: list[Path], targets: list[int], transform):
        self.paths, self.targets, self.transform = paths, targets, transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        with Image.open(self.paths[i]) as im:
            img = im.convert("RGB")
        return self.transform(img), self.targets[i]


def scan_images(folder: Path) -> tuple[list[Path], list[str], list[int], dict[str, int]]:
    paths = sorted(folder.rglob("*.jpg"))
    classes = sorted({p.parent.name for p in paths})
    c2i = {c: i for i, c in enumerate(classes)}
    keys = [f"{p.parent.name}/{p.name}" for p in paths]
    targets = [c2i[p.parent.name] for p in paths]
    key2i = {k: i for i, k in enumerate(keys)}
    return paths, keys, targets, key2i


def embed_all(paths: list[Path], targets: list[int], device: str,
              cache: Path, keys: list[str], label: str):
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        print(f"  {label}: cached {z['x'].shape}", flush=True)
        return z["x"], z["y"], [str(k) for k in z["keys"]]
    weights = ResNet18_Weights.IMAGENET1K_V1
    model = resnet18(weights=weights).eval().to(device)
    backbone = torch.nn.Sequential(*list(model.children())[:-1])
    ds = ImageList(paths, targets, weights.transforms())
    loader = DataLoader(ds, batch_size=128, shuffle=False, num_workers=0)
    chunks = []
    t0 = time.time()
    with torch.no_grad():
        for i, (images, _) in enumerate(loader):
            chunks.append(backbone(images.to(device)).flatten(1).cpu().numpy())
            if (i + 1) % 20 == 0:
                print(f"    {label} {min((i+1)*128, len(ds))}/{len(ds)}", flush=True)
    x = np.concatenate(chunks).astype(np.float32)
    y = np.asarray(targets, dtype=np.int64)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, x=x, y=y, keys=np.asarray(keys))
    print(f"  {label}: extracted {x.shape} in {time.time()-t0:.1f}s", flush=True)
    return x, y, keys


def split_indices(domain: str, p: int, y: np.ndarray, keys: list[str],
                  key2i: dict[str, int]) -> np.ndarray:
    if domain == "DTD":
        meta = DATA / "dtd" / "labels" / f"train{p}.txt"
        lines = [ln.strip() for ln in meta.read_text(encoding="utf-8").splitlines()
                 if ln.strip()]
        idx = np.asarray([key2i[ln] for ln in lines], dtype=np.int64)
        if len(idx) != 1880:
            raise RuntimeError(f"DTD partition {p}: expected 1880 rows, got {len(idx)}")
        return idx
    # EuroSAT: 10 disjoint per-class-stratified folds. Classes are imbalanced
    # (2000-3000 images each), so array_split keeps every fold near 2700 images.
    rng = np.random.default_rng(SPLIT_SEED)
    by_class: dict[int, list[int]] = defaultdict(list)
    for i, c in enumerate(y):
        by_class[int(c)].append(i)
    fold: list[int] = []
    for c in sorted(by_class):
        order = np.asarray(by_class[c], dtype=np.int64)
        rng.shuffle(order)
        fold.extend(np.array_split(order, N_SPLITS)[p - 1].tolist())
    return np.asarray(sorted(fold), dtype=np.int64)


def conditions_8() -> list[Condition]:
    conds = [Condition(a, b, 1, d)
             for a in (0, 1) for b in (0, 1) for d in (0, 1)]
    conds.sort(key=lambda c: c.key())
    return conds


def make_rec_factory(domain: str, p: int, weight_id: str, proto_id: str,
                     base_seed: int):
    split_note = SPLIT_NOTE[domain].format(p=p)

    def rec_factory(cond: Condition, seed: int) -> Record:
        on = [f for f in ("support_label_noise", "query_ood",
                          "source_free", "incomplete_new_class")
              if getattr(cond, f)]
        ratios = []
        if cond.support_label_noise:
            ratios.append(f"support_replace={CFG['support_noise_ratio']}")
        if cond.query_ood:
            ratios.append(f"query_ood={CFG['query_ood_ratio']}")
        if cond.incomplete_new_class:
            ratios.append(f"support_missing={CFG['incomplete_ratio']}")
        return Record(
            dataset_split=(f"{domain};{split_note};N={CFG['n_way']},K={CFG['k_shot']},"
                           f"Q={CFG['n_query']};episode_seed={base_seed+seed}"),
            noise_type_ratio=";".join(on) + "|" + ";".join(ratios),
            source_model_weight=f"torchvision/resnet18;IMAGENET1K_V1;{weight_id}",
            method_code_version=f"protocol.py:{proto_id}",
            cited_paper_local_version="none (no published baseline reproduced)",
            seed_repeats=f"seed={base_seed+seed};repeats={N_REPEATS}",
        )
    return rec_factory


def t_ci(vals: list[float]) -> dict:
    a = np.asarray(vals, dtype=float)
    n = len(a)
    mean, sd = float(a.mean()), float(a.std(ddof=1)) if n > 1 else float("nan")
    if n > 1 and sd > 0:
        half = float(stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n))
        lo, hi = mean - half, mean + half
    else:
        lo = hi = mean
    return {"n": n, "mean": round(mean, 6), "sd": round(sd, 6),
            "ci95_low": round(lo, 6), "ci95_high": round(hi, 6),
            "min": round(float(a.min()), 6), "max": round(float(a.max()), 6)}


def main() -> None:
    if not torch.cuda.is_available():
        print("ABORT: CUDA not available. Refusing CPU fallback.", file=sys.stderr)
        sys.exit(2)
    device = "cuda"
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)

    md5s = {
        "EuroSAT_RGB.zip": md5_full(DATA / "EuroSAT_RGB.zip"),
        "dtd-r1.0.1.tar.gz": md5_full(DATA / "dtd-r1.0.1.tar.gz"),
    }
    print("archive md5:", json.dumps(md5s), flush=True)
    if md5s["EuroSAT_RGB.zip"] != EUROSAT_ZIP_MD5 or md5s["dtd-r1.0.1.tar.gz"] != DTD_TAR_MD5:
        print("ABORT: archive md5 mismatch.", file=sys.stderr)
        sys.exit(3)

    print("torch", torch.__version__, "| cuda", torch.version.cuda,
          "| gpu", torch.cuda.get_device_name(0), flush=True)

    eurosat_paths, eurosat_keys, eurosat_y, eurosat_k2i = scan_images(DATA / "EuroSAT_RGB")
    dtd_paths, dtd_keys, dtd_y, dtd_k2i = scan_images(DATA / "dtd" / "images")
    print(f"scanned: EuroSAT {len(eurosat_paths)} images / "
          f"DTD {len(dtd_paths)} images", flush=True)

    feats = {
        "EuroSAT_RGB": (*embed_all(eurosat_paths, eurosat_y, device,
                                   OUT / "eurosat_all_features.npz",
                                   eurosat_keys, "EuroSAT_RGB"), eurosat_k2i),
        "DTD": (*embed_all(dtd_paths, dtd_y, device,
                           OUT / "dtd_all_features.npz",
                           dtd_keys, "DTD"), dtd_k2i),
    }

    wp = weight_cache_path()
    weight_id = sha256_short(wp) if wp else "cache-path-not-resolved"
    proto_id = file_fingerprint(ROOT / "protocol.py")
    conds = conditions_8()

    all_rows: list[dict] = []
    effect_rows: list[dict] = []
    for domain, (x_all, y_all, keys, k2i) in feats.items():
        key2i = {k: i for i, k in enumerate(keys)}
        for p in range(1, N_SPLITS + 1):
            idx = split_indices(domain, p, y_all, keys, key2i)
            x, y = x_all[idx], y_all[idx]
            base_seed = 20261115 + p * 100000
            cfg = dict(CFG, base_seed=base_seed)
            rows = run_grid(x, y, cfg,
                            make_rec_factory(domain, p, weight_id, proto_id, base_seed),
                            conditions=conds)
            rows = [{"domain": domain, "split": p, **r} for r in rows]
            all_rows.extend(rows)
            eff = estimate_effects(rows, "accuracy")
            er = {"domain": domain, "split": p, "n_images": int(len(idx)),
                  "design": eff["design"]}
            er.update({f"main::{k}": v for k, v in eff["main_effects"].items()})
            er.update({f"inter::{k}": v for k, v in eff["two_way_interactions"].items()})
            effect_rows.append(er)
            print(f"  {domain} split {p:02d}: {len(rows)} rows, "
                  f"acc mean {np.mean([r['accuracy'] for r in rows]):.4f}, "
                  f"noise effect {eff['main_effects']['support_label_noise']:+.4f}",
                  flush=True)

    write_csv(all_rows, OUT / "results_full.csv")
    write_csv(effect_rows, OUT / "effects_by_split.csv")

    # Aggregate cell statistics (n = N_SPLITS * N_REPEATS per cell).
    stats_rows = []
    for metric in ("accuracy", "auroc"):
        groups: dict[tuple[str, str], list[float]] = defaultdict(list)
        for r in all_rows:
            v = r.get(metric)
            if v is not None and np.isfinite(v):
                groups[(r["domain"], r["condition"])].append(float(v))
        for (domain, cond), vals in sorted(groups.items()):
            s = t_ci(vals)
            stats_rows.append({"domain": domain, "condition": cond,
                               "metric": metric, **s})
    write_csv(stats_rows, OUT / "cell_stats_full.csv")

    # Effect stability across splits, per domain.
    effect_names = ["main::support_label_noise", "main::query_ood",
                    "main::incomplete_new_class",
                    "inter::support_label_noisexquery_ood",
                    "inter::support_label_noisexincomplete_new_class",
                    "inter::query_oodxincomplete_new_class"]
    stability: dict[str, dict] = {}
    for domain in feats:
        sub = [r for r in effect_rows if r["domain"] == domain]
        stability[domain] = {
            name: t_ci([float(r[name]) for r in sub]) for name in effect_names
        }

    # Descriptive cross-domain contrast of the noise effect (split types differ).
    e = [float(r["main::support_label_noise"]) for r in effect_rows
         if r["domain"] == "EuroSAT_RGB"]
    d = [float(r["main::support_label_noise"]) for r in effect_rows
         if r["domain"] == "DTD"]
    ttest = stats.ttest_ind(d, e, equal_var=False)
    pooled_sd = float(np.sqrt(np.var(d, ddof=1) / len(d) + np.var(e, ddof=1) / len(e)))
    diff = float(np.mean(d) - np.mean(e))
    cross = {
        "warning": ("descriptive only: EuroSAT uses seeded folds, DTD uses official "
                    "partitions; the two split definitions are not comparable"),
        "EuroSAT_mean": round(float(np.mean(e)), 6),
        "DTD_mean": round(float(np.mean(d)), 6),
        "DTD_minus_EuroSAT": round(diff, 6),
        "approx_ci95_of_difference": [round(diff - 1.96 * pooled_sd, 6),
                                      round(diff + 1.96 * pooled_sd, 6)],
        "welch_t_p_two_sided": float(ttest.pvalue),
    }

    six_ok = all(bool(r[k]) for r in all_rows for k in SIX_FIELDS)
    log = {
        "status": "REPEATED PROTOCOL VALIDATION - still not a publishable result",
        "improvement_over_previous": (
            "repeats 5 -> 100; 1 DTD partition -> all 10; "
            "EuroSAT seeded single subset -> 10 disjoint folds; "
            "split-to-split effect variability now measured"),
        "remaining_limits": [
            "no published baseline method reproduced (prototypical rule only)",
            "EuroSAT folds are seeded, not official; DTD partitions are official",
            "split definitions differ across domains",
            "single feature extractor (ImageNet ResNet-18)",
            "source_free fixed at 1 (no accessible source-domain training set)",
            "DTD license terms still to be confirmed by the author",
        ],
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda_build": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "weight_id": weight_id,
        "protocol_py": proto_id,
        "archive_md5": md5s,
        "n_domains": 2, "n_splits_per_domain": N_SPLITS,
        "n_conditions": len(conds), "n_repeats_per_cell": N_REPEATS,
        "n_rows_total": len(all_rows),
        "six_fields_complete": six_ok,
        "accuracy_range": [min(r["accuracy"] for r in all_rows),
                           max(r["accuracy"] for r in all_rows)],
        "elapsed_seconds": round(time.time() - started, 1),
        "config": CFG,
    }
    summary = {"run_log": log, "effect_stability_across_splits": stability,
               "cross_domain_noise_effect": cross}
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(log, indent=2, ensure_ascii=False), flush=True)
    print("\nEffect stability (mean [95% CI] across 10 splits):", flush=True)
    for domain in stability:
        for name in ("main::support_label_noise", "main::incomplete_new_class",
                     "inter::support_label_noisexincomplete_new_class"):
            s = stability[domain][name]
            print(f"  {domain:12s} {name:52s} {s['mean']:+.4f} "
                  f"[{s['ci95_low']:+.4f}, {s['ci95_high']:+.4f}]", flush=True)
    print("\nCross-domain noise-effect contrast:", json.dumps(cross, indent=2), flush=True)


if __name__ == "__main__":
    main()
