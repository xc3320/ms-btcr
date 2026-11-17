"""Extract multi-scale ResNet-18 embeddings (layer2 / layer3 / layer4).

Why: probes showed a single-scale support set cannot identify its own corrupted
labels well enough to help. A second and third view of the same images costs
almost nothing (the network already computes them) and gives the label test
three independent-ish chances to agree. That agreement, not any single view, is
the candidate noise signal.

Outputs one npz per domain with x2 (128-d), x3 (256-d), x4 (512-d) plus the
same y / keys arrays as the single-scale cache, so folds stay bit-identical.
x4 should match the existing 512-d cache exactly; that equality is asserted
after extraction rather than assumed.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import full_experiment as F  # noqa: E402

ROOT = HERE.parent
OUT = ROOT / "multiscale_output"


class ImageList(Dataset):
    def __init__(self, paths, targets, transform):
        self.paths, self.targets, self.transform = paths, targets, transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            img = im.convert("RGB")
        return self.transform(img), self.targets[i]


def extract(domain: str, folder: Path, cache: Path, device: str) -> None:
    paths, keys, targets, _ = F.scan_images(folder)
    print(f"{domain}: {len(paths)} images, {len(set(targets))} classes", flush=True)
    weights = ResNet18_Weights.IMAGENET1K_V1
    model = resnet18(weights=weights).eval().to(device)

    buf: dict[str, torch.Tensor] = {}
    for name, mod in (("l2", model.layer2), ("l3", model.layer3), ("l4", model.layer4)):
        def mk(key):
            def hook(_m, _i, out):
                buf[key] = out.mean(dim=(2, 3))
            return hook
        mod.register_forward_hook(mk(name))

    ds = ImageList(paths, targets, weights.transforms())
    loader = DataLoader(ds, batch_size=128, shuffle=False, num_workers=0)
    chunks = {"l2": [], "l3": [], "l4": []}
    t0 = time.time()
    with torch.no_grad():
        for i, (images, _) in enumerate(loader):
            model(images.to(device))
            for k in chunks:
                chunks[k].append(buf[k].cpu().numpy())
            if (i + 1) % 40 == 0:
                print(f"  {min((i+1)*128, len(ds))}/{len(ds)}", flush=True)

    x2 = np.concatenate(chunks["l2"]).astype(np.float32)
    x3 = np.concatenate(chunks["l3"]).astype(np.float32)
    x4 = np.concatenate(chunks["l4"]).astype(np.float32)
    y = np.asarray(targets, dtype=np.int64)
    print(f"  shapes l2={x2.shape} l3={x3.shape} l4={x4.shape}  {time.time()-t0:.1f}s", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, x2=x2, x3=x3, x4=x4, y=y, keys=np.asarray(keys))
    print(f"  saved {cache}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="both", choices=("EuroSAT_RGB", "DTD", "both"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    print(f"device={args.device}", flush=True)

    if args.domain in ("EuroSAT_RGB", "both"):
        extract("EuroSAT_RGB", ROOT / "data" / "EuroSAT_RGB",
                OUT / "eurosat_multiscale.npz", args.device)
    if args.domain in ("DTD", "both"):
        extract("DTD", ROOT / "data" / "dtd" / "images",
                OUT / "dtd_multiscale.npz", args.device)


if __name__ == "__main__":
    main()
