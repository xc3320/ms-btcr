"""Two-domain factorial experiment (stage D4/E).

Runs the SAME 2^3 design (source_free fixed at 1, per v3 section 3) on two
target domains:

  * EuroSAT_RGB  - Zenodo 7711810, MIT, seeded 300-image-per-class subset
                   (NO official train/test split)
  * DTD          - Oxford VGG r1.0.1, official partition 1 TRAIN split
                   (1880 images, 47 classes, 40 per class)

Both domains use ImageNet-pretrained ResNet-18 (fc removed) 512-d embeddings
and a prototypical-network classifier. Every reported number carries the six
mandatory record fields. Cell means and 95% t-intervals are computed from the
actual repeats.

IMPORTANT LIMITS (see 25_...md): the two domains do NOT share a comparable
split definition; no published baseline method is reproduced here; only one
partition of DTD is used. These numbers validate the protocol across domains,
they are not yet the manuscript's formal result.
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
    Condition, Record, estimate_effects, file_fingerprint, run_grid, write_csv,
    FACTORS, SIX_FIELDS,
)

ROOT = Path(__file__).resolve().parent
OUT = ROOT.parent / "formal_output"
DATA = ROOT.parent / "data"
PILOT = ROOT.parent / "pilot_output"

EUROSAT_CACHE = PILOT / "eurosat_resnet18_features.npz"
DTD_CACHE = OUT / "dtd_resnet18_partition1_train.npz"

EUROSAT_ZIP_MD5 = "f46e308c4d50d4bf32fedad2d3d62f3b"
DTD_TAR_MD5 = "fff73e5086ae6bdbea199a49dfb8a4c1"

CFG = {
    "n_way": 5,
    "k_shot": 5,
    "n_query": 15,
    "n_repeats": 5,
    "base_seed": 20261115,
    "support_noise_ratio": 0.30,
    "incomplete_ratio": 0.20,
    "query_ood_ratio": 0.20,
}

SPLITS = {
    "EuroSAT_RGB": (
        "EuroSAT_RGB/Zenodo-7711810;seeded-subset-300-per-class;"
        "subset_seed=20261115;NO-official-split"
    ),
    "DTD": (
        "DTD-r1.0.1/Oxford-VGG;official-partition-1-train-split;"
        "1880-images;47-classes;40-per-class"
    ),
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


def embed(dataset, device: str) -> tuple[np.ndarray, np.ndarray]:
    weights = ResNet18_Weights.IMAGENET1K_V1
    model = resnet18(weights=weights).eval().to(device)
    backbone = torch.nn.Sequential(*list(model.children())[:-1])
    loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=0)
    chunks, ys = [], []
    t0 = time.time()
    with torch.no_grad():
        for i, (images, targets) in enumerate(loader):
            f = backbone(images.to(device)).flatten(1).cpu().numpy()
            chunks.append(f)
            ys.append(targets.numpy())
            if (i + 1) % 10 == 0:
                print(f"    embedded {min((i+1)*loader.batch_size, len(dataset))}/{len(dataset)}",
                      flush=True)
    x = np.concatenate(chunks).astype(np.float32)
    y = np.concatenate(ys).astype(np.int64)
    print(f"    extraction took {time.time()-t0:.1f}s, shape={x.shape}", flush=True)
    return x, y


class DTDOfficialSplit(Dataset):
    """Read DTD's official split directly from labels/train{partition}.txt.

    torchvision.datasets.DTD expects root/dtd/dtd/images, while the official
    archive extracts to root/dtd/images. Rather than fabricate an extra nesting
    level, use the format documented in torchvision 0.24.1's dtd.py: each line
    is 'class/filename', the image lives at images/class/filename, and the
    target is the index of the class in the sorted class list.
    """

    def __init__(self, root: Path, split: str = "train", partition: int = 1,
                 transform=None):
        base = Path(root) / "dtd"
        meta = base / "labels" / f"{split}{partition}.txt"
        entries = [ln.strip() for ln in meta.read_text(encoding="utf-8").splitlines()
                   if ln.strip()]
        self.classes = sorted({e.split("/")[0] for e in entries})
        c2i = {c: i for i, c in enumerate(self.classes)}
        self.samples = []
        for e in entries:
            cls, name = e.split("/")
            path = base / "images" / cls / name
            if not path.is_file():
                raise RuntimeError(f"official split references missing image: {path}")
            self.samples.append((path, c2i[cls]))
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        path, target = self.samples[i]
        with Image.open(path) as im:
            img = im.convert("RGB")
        return self.transform(img), target


def load_dtd_features(device: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    if DTD_CACHE.exists():
        z = np.load(DTD_CACHE, allow_pickle=True)
        print(f"  DTD: loaded cached features {z['x'].shape}", flush=True)
        return z["x"], z["y"], list(z["classes"])
    weights = ResNet18_Weights.IMAGENET1K_V1
    ds = DTDOfficialSplit(DATA, split="train", partition=1,
                          transform=weights.transforms())
    print(f"  DTD: official partition-1 train split, classes={len(ds.classes)}, "
          f"images={len(ds)}", flush=True)
    x, y = embed(ds, device)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(DTD_CACHE, x=x, y=y, classes=np.asarray(ds.classes))
    return x, y, list(ds.classes)


def conditions_8() -> list[Condition]:
    conds = [Condition(a, b, 1, d)
             for a in (0, 1) for b in (0, 1) for d in (0, 1)]
    conds.sort(key=lambda c: c.key())
    return conds


def make_rec_factory(domain: str, weight_id: str, proto_id: str):
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
            dataset_split=(f"{SPLITS[domain]};N={CFG['n_way']},K={CFG['k_shot']},"
                           f"Q={CFG['n_query']};episode_seed={CFG['base_seed']+seed}"),
            noise_type_ratio=";".join(on) + "|" + ";".join(ratios),
            source_model_weight=f"torchvision/resnet18;IMAGENET1K_V1;{weight_id}",
            method_code_version=f"protocol.py:{proto_id}",
            cited_paper_local_version="none (no published baseline reproduced)",
            seed_repeats=f"seed={CFG['base_seed']+seed};repeats={CFG['n_repeats']}",
        )
    return rec_factory


def cell_stats(rows: list[dict], metric: str) -> list[dict]:
    groups: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in rows:
        v = r.get(metric)
        if v is not None and np.isfinite(v):
            groups[(r["domain"], r["condition"])].append(float(v))
    out = []
    for (domain, cond), vals in sorted(groups.items()):
        a = np.asarray(vals, dtype=float)
        mean, n = float(a.mean()), len(a)
        sd = float(a.std(ddof=1)) if n > 1 else float("nan")
        if n > 1 and sd > 0:
            half = float(stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n))
            lo, hi = mean - half, mean + half
        else:
            lo = hi = mean
        out.append({
            "domain": domain, "condition": cond, "metric": metric, "n": n,
            "mean": round(mean, 6), "sd": round(sd, 6),
            "ci95_low": round(lo, 6), "ci95_high": round(hi, 6),
        })
    return out


def main() -> None:
    if not torch.cuda.is_available():
        print("ABORT: CUDA not available. Refusing CPU fallback.", file=sys.stderr)
        sys.exit(2)
    device = "cuda"
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    print("torch", torch.__version__, "| cuda_build", torch.version.cuda,
          "| gpu", torch.cuda.get_device_name(0), flush=True)

    md5s = {
        "EuroSAT_RGB.zip": md5_full(DATA / "EuroSAT_RGB.zip") if (DATA / "EuroSAT_RGB.zip").exists() else "missing",
        "dtd-r1.0.1.tar.gz": md5_full(DATA / "dtd-r1.0.1.tar.gz") if (DATA / "dtd-r1.0.1.tar.gz").exists() else "missing",
    }
    print("archive md5:", json.dumps(md5s), flush=True)
    if md5s["EuroSAT_RGB.zip"] != EUROSAT_ZIP_MD5 or md5s["dtd-r1.0.1.tar.gz"] != DTD_TAR_MD5:
        print("ABORT: archive md5 mismatch. Do not produce numbers from unverified data.",
              file=sys.stderr)
        sys.exit(3)

    z = np.load(EUROSAT_CACHE, allow_pickle=True)
    domains: dict[str, tuple[np.ndarray, np.ndarray, list[str]]] = {
        "EuroSAT_RGB": (z["x"], z["y"], list(z["classes"])),
    }
    print(f"  EuroSAT_RGB: cached features {z['x'].shape}", flush=True)
    domains["DTD"] = load_dtd_features(device)

    wp = weight_cache_path()
    weight_id = sha256_short(wp) if wp else "cache-path-not-resolved"
    proto_id = file_fingerprint(ROOT / "protocol.py")
    conds = conditions_8()

    all_rows: list[dict] = []
    per_domain_effects: dict[str, dict] = {}
    for name, (x, y, classes) in domains.items():
        print(f"=== running grid on {name} ({len(conds)} cells x {CFG['n_repeats']} repeats) ===",
              flush=True)
        rows = run_grid(x, y, CFG, make_rec_factory(name, weight_id, proto_id),
                        conditions=conds)
        rows = [{"domain": name, **r} for r in rows]
        all_rows.extend(rows)
        eff = estimate_effects(rows, "accuracy")
        per_domain_effects[name] = {
            "design": eff["design"],
            "active_factors": eff["active_factors"],
            "fixed_factors": eff["fixed_factors"],
            "main_effects": eff["main_effects"],
            "two_way_interactions": eff["two_way_interactions"],
        }
        print(f"  {name} accuracy mean "
              f"{np.mean([r['accuracy'] for r in rows]):.4f} | "
              f"main effects {json.dumps(eff['main_effects'])}", flush=True)

    write_csv(all_rows, OUT / "two_domain_results.csv")
    acc_stats = cell_stats(all_rows, "accuracy")
    auroc_stats = cell_stats(all_rows, "auroc")
    write_csv(acc_stats + auroc_stats, OUT / "two_domain_cell_stats.csv")
    (OUT / "two_domain_effects_by_domain.json").write_text(
        json.dumps(per_domain_effects, indent=2, ensure_ascii=False), encoding="utf-8")

    # Cross-domain contrast: difference of the support-noise main effect.
    delta = None
    try:
        delta = round(float(
            per_domain_effects["DTD"]["main_effects"]["support_label_noise"]
            - per_domain_effects["EuroSAT_RGB"]["main_effects"]["support_label_noise"]), 6)
    except (KeyError, TypeError):
        pass

    six_ok = all(bool(r[k]) for r in all_rows for k in SIX_FIELDS)
    log = {
        "status": "PROTOCOL VALIDATION ACROSS TWO DOMAINS - not a publishable result",
        "limits": [
            "EuroSAT uses a seeded subset with no official split; DTD uses official partition-1 train split",
            "split definitions differ across domains, so cross-domain numbers are not strictly comparable",
            "only one DTD partition (1 of 10) is used",
            "no published baseline method reproduced; prototypical network only",
            "5 repeats per cell; CIs are t-intervals over 5 episodes",
            "source_free fixed at 1 (no accessible source-domain training set)",
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
        "archive_md5_expected": {
            "EuroSAT_RGB.zip": EUROSAT_ZIP_MD5,
            "dtd-r1.0.1.tar.gz": DTD_TAR_MD5,
        },
        "domain_details": {
            "EuroSAT_RGB": {"classes": len(domains["EuroSAT_RGB"][2]),
                            "images": int(len(domains["EuroSAT_RGB"][1])),
                            "split": SPLITS["EuroSAT_RGB"]},
            "DTD": {"classes": len(domains["DTD"][2]),
                    "images": int(len(domains["DTD"][1])),
                    "split": SPLITS["DTD"]},
        },
        "n_conditions_per_domain": len(conds),
        "n_rows_total": len(all_rows),
        "six_fields_complete": six_ok,
        "accuracy_range": [min(r["accuracy"] for r in all_rows),
                           max(r["accuracy"] for r in all_rows)],
        "support_noise_effect_difference_DTD_minus_EuroSAT": delta,
        "elapsed_seconds": round(time.time() - started, 1),
        "config": CFG,
    }
    (OUT / "two_domain_run_log.json").write_text(
        json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(log, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
