"""Extract three depth views for a backbone other than ResNet-18.

Purpose: the paper's gain was measured on one backbone only. A reviewer will
ask whether multi-scale transductive centroid re-estimation (MS-TCR) is a
property of the method or an accident of ResNet-18. This script produces the
same kind of cache as multiscale_features.py for two further backbones:

  resnet50  -- a deeper CNN of the same family; views are layer2/3/4,
               mirroring the ResNet-18 recipe exactly.
  vit_b_16  -- a different architecture family; there are no residual stages,
               so the analogue is depth in the encoder stack. Views are the
               token-mean output of blocks 5, 8 and 11 (0-indexed), CLS token
               excluded, so the three views differ in the same way layer
               depth differs in a CNN.

Nothing about the method changes between backbones: the same unit-norm
concatenation, the same five re-estimation rounds, the same median + k*MAD
OOD threshold. Only the source of the three views changes.

Image order comes from full_experiment.scan_images, so keys and labels stay
identical across every cache and every fold slice keeps the same membership.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import (ResNet18_Weights, ResNet50_Weights,
                                ViT_B_16_Weights, resnet18, resnet50, vit_b_16)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import full_experiment as F  # noqa: E402

ROOT = HERE.parent
OUT = ROOT / "backbone_output"

# (module attribute path, human name) for each backbone
SPECS = {
    "resnet18": {
        # Same weights and same three stages as multiscale_features.py, so the
        # cache written here is interchangeable with eurosat_multiscale.npz.
        # key_offset keeps the payload keys at x2/x3/x4, the layout that
        # suite_main expects for resnet18.
        "views": ("layer2", "layer3", "layer4"),
        "dims": (128, 256, 512),
        "build": lambda: resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).eval(),
        "transform": lambda: ResNet18_Weights.IMAGENET1K_V1.transforms(),
        "key_offset": 1,
    },
    "resnet50": {
        "views": ("layer2", "layer3", "layer4"),
        "dims": (512, 1024, 2048),
        "build": lambda: resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).eval(),
        "transform": lambda: ResNet50_Weights.IMAGENET1K_V2.transforms(),
    },
    "vit_b_16": {
        # block indices inside model.encoder.layers
        "views": (5, 8, 11),
        "dims": (768, 768, 768),
        "build": lambda: vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1).eval(),
        "transform": lambda: ViT_B_16_Weights.IMAGENET1K_V1.transforms(),
    },
}


class ImageList(Dataset):
    def __init__(self, paths, transform):
        self.paths, self.transform = paths, transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            img = im.convert("RGB")
        return self.transform(img)


def _hooks_cnn(model, names, buf):
    for name in names:
        mod = model.get_submodule(name)

        def mk(key):
            def hook(_m, _i, out):
                buf[key] = out.mean(dim=(2, 3))
            return hook

        mod.register_forward_hook(mk(name))


def _hooks_vit(model, blocks, buf):
    for b in blocks:
        mod = model.encoder.layers[b]

        def mk(key):
            def hook(_m, _i, out):
                t = out[0] if isinstance(out, tuple) else out
                buf[key] = t[:, 1:, :].mean(dim=1)   # patch tokens, CLS dropped
            return hook

        mod.register_forward_hook(mk(b))


def extract(backbone: str, domain: str, folder: Path, cache: Path,
            device: str) -> dict:
    spec = SPECS[backbone]
    paths, keys, targets, _ = F.scan_images(folder)
    print(f"[{backbone}] {domain}: {len(paths)} images, "
          f"{len(set(targets))} classes", flush=True)

    model = spec["build"]().to(device)
    buf: dict = {}
    if backbone.startswith("resnet"):
        _hooks_cnn(model, spec["views"], buf)
    else:
        _hooks_vit(model, spec["views"], buf)

    ds = ImageList(paths, spec["transform"]())
    loader = DataLoader(ds, batch_size=128, shuffle=False, num_workers=0)
    chunks = {v: [] for v in spec["views"]}
    t0 = time.time()
    with torch.no_grad():
        for i, images in enumerate(loader):
            model(images.to(device))
            for v in chunks:
                chunks[v].append(buf[v].float().cpu().numpy())
            if (i + 1) % 40 == 0:
                print(f"  {min((i+1)*128, len(ds))}/{len(ds)}  "
                      f"{time.time()-t0:.0f}s", flush=True)

    xs = {v: np.concatenate(chunks[v]).astype(np.float32) for v in spec["views"]}
    y = np.asarray(targets, dtype=np.int64)
    for v, d in zip(spec["views"], spec["dims"]):
        assert xs[v].shape == (len(paths), d), (v, xs[v].shape, d)

    OUT.mkdir(parents=True, exist_ok=True)
    off = spec.get("key_offset", 0)
    payload = {f"x{i+1+off}": xs[v] for i, v in enumerate(spec["views"])}
    payload.update(y=y, keys=np.asarray(keys))
    np.savez_compressed(cache, **payload)
    dt = time.time() - t0
    print(f"  saved {cache.name}  dims="
          f"{[int(xs[v].shape[1]) for v in spec['views']]}  {dt:.1f}s", flush=True)
    return {"backbone": backbone, "domain": domain, "images": len(paths),
            "views": [str(v) for v in spec["views"]],
            "dims": [int(xs[v].shape[1]) for v in spec["views"]],
            "seconds": round(dt, 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True, choices=sorted(SPECS))
    ap.add_argument("--domain", default="both",
                    choices=("EuroSAT_RGB", "DTD", "MINI", "both"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    print(f"device={args.device} backbone={args.backbone}", flush=True)

    info = []
    if args.domain in ("EuroSAT_RGB", "both"):
        info.append(extract(args.backbone, "EuroSAT_RGB",
                            ROOT / "data" / "EuroSAT_RGB",
                            OUT / f"eurosat_{args.backbone}.npz", args.device))
    if args.domain in ("DTD", "both"):
        info.append(extract(args.backbone, "DTD",
                            ROOT / "data" / "dtd" / "images",
                            OUT / f"dtd_{args.backbone}.npz", args.device))
    if args.domain in ("MINI", "both"):
        # 20 novel classes of the miniImageNet test split, 600 images each.
        # Built by mini_prepare.py from the public parquet shards.
        info.append(extract(args.backbone, "MINI",
                            ROOT / "data" / "miniimagenet" / "test",
                            OUT / f"miniimagenet_{args.backbone}.npz", args.device))

    log = OUT / "extract_log.json"
    old = json.loads(log.read_text(encoding="utf-8")) if log.exists() else []
    old = [r for r in old if not (r["backbone"] == args.backbone
                                  and r["domain"] in {i["domain"] for i in info})]
    log.write_text(json.dumps(old + info, indent=1), encoding="utf-8")
    print("logged:", [f'{i["domain"]}/{i["backbone"]}' for i in info])


if __name__ == "__main__":
    main()
