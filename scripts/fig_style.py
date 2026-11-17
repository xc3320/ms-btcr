"""Shared figure style for the manuscript figures.

All figures are built from the CSV/JSON files produced by the experiment
scripts. Nothing in this module changes a number; it only sets typography,
colour and canvas size so the set looks like one paper.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
FIGDIR = ROOT / "figures"

# Journal-typical widths (mm -> inch)
SINGLE = 85 / 25.4
ONEHALF = 120 / 25.4
DOUBLE = 170 / 25.4

# Domains and inference rules use fixed hues across every figure.
C_EURO = "#1f4e79"   # EuroSAT_RGB  (deep blue)
C_DTD = "#c55a11"    # DTD           (burnt orange)
C_PROTO = "#1f4e79"  # prototypical
C_TCR = "#4c8cbf"
C_MS = "#1f4e79"
C_OOD = "#a61b29"
C_BAL = "#2e7d6f"
C_FULL = "#c58f2a"
C_ORACLE = "#c58f2a"
C_RANDOM = "#7f7f7f"
C_CLEAN = "#2e7d6f"
C_KNN = "#2e7d6f"    # matching-network style
C_GREY = "#7f7f7f"
C_GRID = "#d9d9d9"
C_NEG = "#a61b29"    # negative sign marker
C_POS = "#1a7f4b"    # positive sign marker
DOMAIN_COLOR = {"EuroSAT_RGB": C_EURO, "DTD": C_DTD}
DOMAIN_LABEL = {"EuroSAT_RGB": "EuroSAT", "DTD": "DTD"}

# Canonical order of the eight cells: 2^3 grid, source_free is the fixed premise.
CELL_ORDER = [
    "source_free",
    "support_label_noise+source_free",
    "query_ood+source_free",
    "support_label_noise+query_ood+source_free",
    "source_free+incomplete_new_class",
    "support_label_noise+source_free+incomplete_new_class",
    "query_ood+source_free+incomplete_new_class",
    "support_label_noise+query_ood+source_free+incomplete_new_class",
]
CELL_LABEL = {
    "source_free": "—",
    "support_label_noise+source_free": "N",
    "query_ood+source_free": "O",
    "support_label_noise+query_ood+source_free": "N+O",
    "source_free+incomplete_new_class": "I",
    "support_label_noise+source_free+incomplete_new_class": "N+I",
    "query_ood+source_free+incomplete_new_class": "O+I",
    "support_label_noise+query_ood+source_free+incomplete_new_class": "N+O+I",
}
CELL_TITLE = {
    "source_free": "clean",
    "support_label_noise+source_free": "N",
    "query_ood+source_free": "O",
    "support_label_noise+query_ood+source_free": "N+O",
    "source_free+incomplete_new_class": "I",
    "support_label_noise+source_free+incomplete_new_class": "N+I",
    "query_ood+source_free+incomplete_new_class": "O+I",
    "support_label_noise+query_ood+source_free+incomplete_new_class": "N+O+I",
}


def apply() -> None:
    mpl.rcParams.update({
        "font.family": "Arial",
        "font.size": 8,
        "font.weight": "normal",
        "axes.titlesize": 9,
        "axes.titleweight": "bold",
        "axes.labelsize": 8,
        "axes.linewidth": 0.7,
        "axes.edgecolor": "#333333",
        "axes.labelcolor": "#222222",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.6,
        "ytick.major.size": 2.6,
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        "legend.fontsize": 7,
        "legend.frameon": False,
        "legend.handlelength": 1.6,
        "legend.columnspacing": 1.2,
        "grid.color": C_GRID,
        "grid.linewidth": 0.5,
        "grid.alpha": 1.0,
        "figure.dpi": 120,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,   # embed as TrueType so the journal can edit text
        "ps.fonttype": 42,
        "axes.unicode_minus": True,
    })


def save(fig, name: str) -> list[str]:
    """Write PDF (vector, for submission) and PNG (600 dpi, for reading)."""
    FIGDIR.mkdir(parents=True, exist_ok=True)
    outs = []
    for ext in ("pdf", "png"):
        p = FIGDIR / f"{name}.{ext}"
        fig.savefig(p, format=ext)
        outs.append(str(p.relative_to(ROOT)))
    plt.close(fig)
    return outs


def panel_tag(ax, tag: str, dx: float = -0.13, dy: float = 1.06) -> None:
    ax.text(dx, dy, tag, transform=ax.transAxes, fontsize=9,
            fontweight="bold", va="top", ha="left", color="#222222")
