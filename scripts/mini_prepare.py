"""Materialise the miniImageNet test split as an ImageFolder tree.

Why only the test split: the method is training-free, so no base class is ever
used for fitting. The 20 test classes of the Ravi & Larochelle split are the
novel classes that a 5-way episode draws from, and 12000 images at 600 per
class are what the protocol needs. Downloading the 64 training classes would
add 2.5 GB that no experiment reads.

Input : the two parquet shards of the `test` config
        (GATE-engine/mini_imagenet on the Hub), 12000 rows, columns image/label.
Output: data/miniimagenet/test/<cls>/*.jpg plus a manifest with the image
        sizes and the per-class counts, so the paper can state them.

The class directories are named cls00..cls19 in ascending label order. Labels
in this release are integer codes, not WordNet ids, and no experiment depends
on the names; the manifest records the mapping so it stays traceable.
"""
from __future__ import annotations

import argparse
import io
import json
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SRC = ROOT.parent / "_downloads" / "mini"
DEFAULT_DST = ROOT / "data" / "miniimagenet" / "test"


def row_bytes(cell) -> bytes:
    """Pull the JPEG payload out of whatever shape the column has."""
    if isinstance(cell, (bytes, bytearray, memoryview)):
        return bytes(cell)
    if isinstance(cell, dict):
        for k in ("bytes", "data", "content"):
            if cell.get(k) is not None:
                return row_bytes(cell[k])
        raise KeyError(f"no byte field in {sorted(cell)}")
    if hasattr(cell, "as_py"):
        return row_bytes(cell.as_py())
    raise TypeError(f"unsupported cell type {type(cell)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC)
    ap.add_argument("--dst", type=Path, default=DEFAULT_DST)
    ap.add_argument("--resize", type=int, default=None,
                    help="downscale every image to NxN; 84 reproduces the "
                         "classic miniImageNet resolution from this 224x224 release")
    args = ap.parse_args()

    shards = sorted(args.src.glob("test-*.parquet"))
    if not shards:
        raise SystemExit(f"no test parquet under {args.src}")

    args.dst.mkdir(parents=True, exist_ok=True)
    counts: Counter = Counter()
    sizes: Counter = Counter()
    n = 0
    for shard in shards:
        table = pq.read_table(shard, columns=["image", "label"])
        images = table.column("image").to_pylist()
        labels = table.column("label").to_pylist()
        print(f"{shard.name}: {len(labels)} rows", flush=True)
        for cell, lab in zip(images, labels):
            cls = f"cls{int(lab):02d}"
            out_dir = args.dst / cls
            out_dir.mkdir(exist_ok=True)
            path = out_dir / f"{n:05d}.jpg"
            with Image.open(io.BytesIO(row_bytes(cell))) as im:
                out = im.convert("RGB")
                if args.resize:
                    out = out.resize((args.resize, args.resize), Image.LANCZOS)
                sizes[out.size] += 1
                out.save(path, format="JPEG", quality=95)
            counts[cls] += 1
            n += 1
            if n % 2000 == 0:
                print(f"  {n}", flush=True)

    dst_abs = args.dst.resolve()
    try:
        root_rel = str(dst_abs.relative_to(ROOT))
    except ValueError:
        root_rel = str(dst_abs)
    manifest = {
        "source": "GATE-engine/mini_imagenet (Hub), test split",
        "split": "test (20 novel classes of the Ravi & Larochelle split)",
        "images": n,
        "classes": len(counts),
        "per_class": sorted(set(counts.values())),
        "class_counts": dict(sorted(counts.items())),
        "image_sizes": {f"{w}x{h}": c for (w, h), c in sizes.most_common()},
        "root": root_rel,
    }
    mpath = args.dst.parent / (f"manifest{args.resize}.json" if args.resize
                              else "manifest.json")
    mpath.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in manifest.items()
                      if k != "class_counts"}, indent=1))


if __name__ == "__main__":
    main()
