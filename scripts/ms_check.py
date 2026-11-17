"""Sanity check on the multi-scale caches before any experiment uses them.

Two things must hold:
  1. layer4 reproduces the existing 512-d single-scale cache, otherwise folds and
     previously reported numbers would describe different features.
  2. the three views are on comparable footing after unit normalisation, so a
     concatenation cannot be dominated by whichever layer happens to have the
     largest norm.
"""
from __future__ import annotations

import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MS = ROOT / "multiscale_output"

for name, msfile, oldfile in (
    ("EuroSAT_RGB", MS / "eurosat_multiscale.npz", ROOT / "full_output" / "eurosat_all_features.npz"),
    ("DTD", MS / "dtd_multiscale.npz", ROOT / "full_output" / "dtd_all_features.npz"),
):
    ms = np.load(msfile, allow_pickle=False)
    old = np.load(oldfile, allow_pickle=True)
    same_y = np.array_equal(ms["y"], old["y"])
    same_k = list(ms["keys"]) == [str(k) for k in old["keys"]]
    same_x = np.allclose(ms["x4"], old["x"], atol=1e-5)
    print(f"{name}: keys_match={same_k} y_match={same_y} layer4==single={same_x} "
          f"max|diff|={np.abs(ms['x4'] - old['x']).max():.2e}")
    for k in ("x2", "x3", "x4"):
        v = ms[k].astype(np.float64)
        n = np.linalg.norm(v, axis=1)
        print(f"  {k}: {v.shape} norm mean={n.mean():8.2f} sd={n.std():7.2f} "
              f"|x|mean={np.abs(v).mean():.4f} |x|max={np.abs(v).max():7.2f}")
