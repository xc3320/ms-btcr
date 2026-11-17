"""Minimum pilot experiment (stage D3): real images, real features, real numbers.

Pipeline:
  EuroSAT_RGB (Zenodo 7711810) -> ImageNet-pretrained ResNet-18 (fc removed)
  -> cached 512-d embeddings -> factorial protocol -> result table + run log.

DATASET CHANGE (2026-11-15): the original plan used CIFAR-100. Two findings
changed that: (1) its parent corpus, 80 Million Tiny Images, was formally
withdrawn by MIT; (2) the official CIFAR host delivered only ~12 KB/s.
See 16_数据伦理与许可核验.md. EuroSAT has an explicit MIT license, comes from
open Sentinel-2 imagery, and is a standard cross-domain few-shot target.

IMPORTANT: this is a pipeline pilot, not a publishable result. Only 8
conditions are run because source_free is fixed at 1 (no accessible
source-domain training set). EuroSAT has no official train/test split, so
this pilot uses a seeded 300-image-per-class subset, not an official split.

Precondition: cu128 PyTorch installed and torch.cuda.is_available() is True.
This script refuses to run on CPU so that a silent fallback cannot produce
numbers that look like GPU results.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18

from protocol import (
    Condition, Record, estimate_effects, file_fingerprint, run_grid, write_csv,
)

ROOT = Path(__file__).resolve().parent
OUT = ROOT.parent / "pilot_output"
DATA = ROOT.parent / "data"
CACHE = OUT / "eurosat_resnet18_features.npz"

ZENODO_MD5 = "f46e308c4d50d4bf32fedad2d3d62f3b"
MAX_PER_CLASS = 300
SUBSET_SEED = 20261115

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


class EuroSATSubset(Dataset):
    """Load a seeded per-class subset from the extracted EuroSAT_RGB tree.

    Class directories are discovered by looking for directories that contain
    .jpg files, so the exact nesting depth of the zip does not matter.
    """

    def __init__(self, root: Path, transform, max_per_class: int, seed: int):
        paths = sorted(Path(root).rglob("*.jpg"))
        if not paths:
            raise RuntimeError(
                f"No .jpg files under {root}. Download and extract EuroSAT_RGB.zip first."
            )
        by_class: dict[str, list[Path]] = {}
        for p in paths:
            by_class.setdefault(p.parent.name, []).append(p)
        self.classes = sorted(by_class)
        rng = np.random.default_rng(seed)
        self.samples: list[tuple[Path, int]] = []
        for ci, name in enumerate(self.classes):
            files = sorted(by_class[name])
            if len(files) > max_per_class:
                pick = np.sort(rng.choice(len(files), size=max_per_class, replace=False))
                files = [files[i] for i in pick]
            self.samples.extend((f, ci) for f in files)
        self.targets = [t for _, t in self.samples]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        path, target = self.samples[i]
        with Image.open(path) as im:
            img = im.convert("RGB")
        return self.transform(img), target


def sha256_short(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()[:16]


def md5_full(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def weight_cache_path() -> Path | None:
    d = Path(torch.hub.get_dir()) / "checkpoints"
    hits = sorted(d.glob("resnet18-*.pth")) if d.exists() else []
    return hits[0] if hits else None


def extract_features(device: str) -> tuple[np.ndarray, np.ndarray, EuroSATSubset]:
    weights = ResNet18_Weights.IMAGENET1K_V1
    model = resnet18(weights=weights).eval().to(device)
    backbone = torch.nn.Sequential(*list(model.children())[:-1])

    ds = EuroSATSubset(DATA, weights.transforms(), MAX_PER_CLASS, SUBSET_SEED)
    print(f"  classes={len(ds.classes)} images={len(ds)}", flush=True)
    loader = DataLoader(ds, batch_size=128, shuffle=False, num_workers=0)

    chunks = []
    t0 = time.time()
    with torch.no_grad():
        for i, (images, _) in enumerate(loader):
            f = backbone(images.to(device)).flatten(1).cpu().numpy()
            chunks.append(f)
            if (i + 1) % 10 == 0:
                print(f"  embedded {min((i+1)*loader.batch_size, len(ds))}/{len(ds)}", flush=True)
    x = np.concatenate(chunks).astype(np.float32)
    y = np.asarray(ds.targets, dtype=np.int64)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, x=x, y=y, classes=np.asarray(ds.classes))
    print(f"  feature extraction took {time.time() - t0:.1f}s, shape={x.shape}", flush=True)
    return x, y, ds


def main() -> None:
    if not torch.cuda.is_available():
        print("ABORT: CUDA not available. Refusing to run a CPU fallback.", file=sys.stderr)
        sys.exit(2)

    device = "cuda"
    started = time.time()
    print("torch", torch.__version__, "| cuda_build", torch.version.cuda,
          "| device", torch.cuda.get_device_name(0), flush=True)

    zip_path = DATA / "EuroSAT_RGB.zip"
    archive_md5 = md5_full(zip_path) if zip_path.exists() else "archive-not-on-disk"
    if archive_md5 != ZENODO_MD5:
        print(f"WARNING: archive md5={archive_md5}, expected {ZENODO_MD5}", flush=True)

    if CACHE.exists():
        z = np.load(CACHE, allow_pickle=True)
        x, y = z["x"], z["y"]
        classes = list(z["classes"])
        print(f"loaded cached features {x.shape} from {CACHE.name}", flush=True)
    else:
        print("extracting EuroSAT features...", flush=True)
        x, y, ds = extract_features(device)
        classes = ds.classes

    wp = weight_cache_path()
    weight_id = sha256_short(wp) if wp else "cache-path-not-resolved"
    proto_id = file_fingerprint(ROOT / "protocol.py")

    conditions = [
        Condition(a, b, 1, d)
        for a in (0, 1) for b in (0, 1) for d in (0, 1)
    ]
    conditions.sort(key=lambda c: c.key())

    split_tag = (f"EuroSAT_RGB/Zenodo-7711810;seeded-subset-{MAX_PER_CLASS}per-class"
                 f";subset_seed={SUBSET_SEED};N={CFG['n_way']},K={CFG['k_shot']},Q={CFG['n_query']}")

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
            dataset_split=f"{split_tag};episode_seed={CFG['base_seed']+seed}",
            noise_type_ratio=";".join(on) + "|" + ";".join(ratios),
            source_model_weight=f"torchvision/resnet18;IMAGENET1K_V1;{weight_id}",
            method_code_version=f"protocol.py:{proto_id}",
            cited_paper_local_version="pilot/none (no cited method evaluated)",
            seed_repeats=f"seed={CFG['base_seed']+seed};repeats={CFG['n_repeats']}",
        )

    rows = run_grid(x, y, CFG, rec_factory, conditions=conditions)
    write_csv(rows, OUT / "pilot_results.csv")
    effects = estimate_effects(rows, "accuracy")
    (OUT / "pilot_effects.json").write_text(
        json.dumps(effects, indent=2, ensure_ascii=False), encoding="utf-8")

    log = {
        "status": "PILOT ONLY - not a publishable result",
        "reason": ("seeded EuroSAT subset is not an official split; source_free fixed at 1; "
                   "only a prototypical-network baseline; no published method reproduced"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda_build": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "weight_id": weight_id,
        "protocol_py": proto_id,
        "dataset": "EuroSAT_RGB, Zenodo record 7711810, MIT license",
        "archive_md5": archive_md5,
        "archive_md5_expected": ZENODO_MD5,
        "archive_md5_match": archive_md5 == ZENODO_MD5,
        "classes": classes,
        "max_per_class": MAX_PER_CLASS,
        "subset_seed": SUBSET_SEED,
        "feature_shape": list(x.shape),
        "n_conditions": len(conditions),
        "n_rows": len(rows),
        "accuracy_range": [min(r["accuracy"] for r in rows), max(r["accuracy"] for r in rows)],
        "elapsed_seconds": round(time.time() - started, 1),
        "config": CFG,
    }
    (OUT / "pilot_run_log.json").write_text(
        json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(log, indent=2, ensure_ascii=False), flush=True)
    print("\nMain effects (PILOT, not manuscript data):", flush=True)
    print(json.dumps(effects["main_effects"], indent=2), flush=True)


if __name__ == "__main__":
    main()
