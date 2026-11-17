"""Exact cell-by-cell check of the tables that changed when miniImageNet was added.

Every value is recomputed from the run files and compared with the string in the
draft. A failure prints the table, the row label, the column and both values.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DRAFT = ROOT / (sys.argv[1] if len(sys.argv) > 1 else "89_English_draft_sci_v4.md")

SUITE = {bb: json.loads((ROOT / "suite_output" / f"{bb}_main.json").read_text(encoding="utf-8"))
         for bb in ("resnet18", "resnet50", "vit_b_16")}
MINI = {bb: json.loads((ROOT / "suite_output" / f"{bb}_mini.json").read_text(encoding="utf-8"))
        for bb in ("resnet18", "resnet50", "vit_b_16")}
IMB = (json.loads((ROOT / "imbalance_output" / "imbalance_resnet18_main.json").read_text(encoding="utf-8"))
       + [r for r in json.loads((ROOT / "imbalance_output" / "imbalance_resnet18_adapt.json").read_text(encoding="utf-8"))
          if r["method"] == "adapt"]
       + json.loads((ROOT / "imbalance_output" / "imbalance_resnet18_mini.json").read_text(encoding="utf-8")))
TP = json.loads((ROOT / "tp_baseline_output" / "tp_mini.json").read_text(encoding="utf-8"))
LAP = json.loads((ROOT / "lap_shot_output" / "lap_mini.json").read_text(encoding="utf-8"))

RATIOS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)
fails: list[str] = []
checks = 0


def cells(rows, domain, method, ratio=0.3, ood=0.2):
    sel = [x for x in rows if x.get("domain") == domain and x["method"] == method
           and x["ratio"] == ratio and x["ood"] == ood]
    if not sel:
        return None, None
    return (float(np.mean([x["accuracy"] for x in sel])),
            float(np.mean([x["auroc"] for x in sel])) if sel[0].get("auroc") is not None else None)


def close(a: float, b: float, tol: float = 6e-4) -> bool:
    return abs(a - b) <= tol


def get_tables(text: str) -> dict[int, list[list[str]]]:
    out: dict[int, list[list[str]]] = {}
    for m in re.finditer(r"\*\*Table (\d+)\.\*\*", text):
        num = int(m.group(1))
        tail = text[m.end():]
        lines = []
        for line in tail.split("\n"):
            if line.startswith("|"):
                lines.append(line)
            elif line.strip() == "" and lines:
                break
        rows = []
        for line in lines:
            parts = [p.strip() for p in line.strip().strip("|").split("|")]
            if all(set(p) <= set("-: ") for p in parts):
                continue
            rows.append(parts)
        out[num] = rows
    return out


def num(s: str):
    m = re.search(r"-?\d+\.?\d*", s.replace("\u2212", "-"))
    return float(m.group(0)) if m else None


def chk(label, got, want, tol=6e-4):
    global checks
    checks += 1
    if got is None:
        fails.append(f"{label}: expected {want}, source empty")
    elif not close(got, want, tol):
        fails.append(f"{label}: draft {want} vs recomputed {got:.4f}")


def fmt_pairs(s: str):
    """'0.7018/0.638' or '0.7523 (+0.0505)' or '0.638→0.810'"""
    return s


text = DRAFT.read_text(encoding="utf-8")
T = get_tables(text)

# ---------------- Table 3: miniImageNet main ----------------
ARM = {"Single-scale prototype": "proto_deep",
       "Single-scale transductive": "tcr_deep",
       "Multi-scale transductive": "mstcr",
       "Multi-scale + OOD weight": "mstcr_ood",
       "Balanced assignment (no OOD term)": "bal_iter",
       "Ours": "bal_soft"}
for row in T[3][1:]:
    lab = row[0]
    if lab not in ARM:
        continue
    for j, ratio in enumerate(RATIOS):
        acc, au = cells(MINI["resnet18"], "Mini", ARM[lab], ratio)
        a_s, u_s = row[1 + j].split("/")
        chk(f"T3 {lab} r={ratio} acc", acc, float(a_s))
        chk(f"T3 {lab} r={ratio} auroc", au, float(u_s), 6e-4)

# ---------------- Table 4: ladder, three domains ----------------
LADDER = [("Single-scale prototype", "proto_deep"),
          ("+ transductive re-estimation (single scale)", "tcr_deep"),
          ("+ multi-scale normalised concatenation", "mstcr"),
          ("+ out-of-distribution weighting", "mstcr_ood"),
          ("+ balanced quota (no OOD term)", "bal_iter"),
          ("Ours (balanced + continuous weighting)", "bal_soft")]
prev = {}
for row in T[4][1:]:
    lab = row[0]
    if lab not in dict(LADDER):
        continue
    arm = dict(LADDER)[lab]
    for col, (dom, src) in enumerate((("EuroSAT_RGB", SUITE["resnet18"]),
                                      ("DTD", SUITE["resnet18"]),
                                      ("Mini", MINI["resnet18"]))):
        acc, au = cells(src, dom, arm)
        acell = row[1 + col * 2]
        ucell = row[2 + col * 2]
        chk(f"T4 {lab} {dom} acc", acc, float(acell.split("(")[0]))
        chk(f"T4 {lab} {dom} auroc", au, float(ucell.split("(")[0]), 6e-4)
        m = re.search(r"\(([+-]?[\d.]+)\)", acell)
        if m and dom in prev:
            chk(f"T4 {lab} {dom} acc-step", acc - prev[dom][0], float(m.group(1).replace("\u2212", "-")))
        m2 = re.search(r"\(([+-−]?[\d.]+)\)", ucell)
        if m2 and dom in prev and prev[dom][1] is not None and au is not None:
            chk(f"T4 {lab} {dom} auroc-step", au - prev[dom][1], float(m2.group(1).replace("\u2212", "-")), 1.1e-3)
        prev.setdefault(dom, (acc, au)) if dom not in prev else None
        prev[dom] = (acc, au)

# ---------------- Table 5: backbones x domains ----------------
BB = {"ResNet-18": "resnet18", "ResNet-50": "resnet50", "ViT-B/16": "vit_b_16"}
DOM = {"EuroSAT": "EuroSAT_RGB", "DTD": "DTD", "miniImageNet": "Mini"}
for row in T[5][1:]:
    bb, dom = row[0], row[1]
    if bb not in BB or dom not in DOM:
        continue
    src = SUITE[BB[bb]] if dom != "miniImageNet" else MINI[BB[bb]]
    p, _ = cells(src, DOM[dom], "proto_deep")
    o, _ = cells(src, DOM[dom], "bal_soft")
    m, _ = cells(src, DOM[dom], "mstcr_ood")
    chk(f"T5 {bb} {dom} proto", p, float(row[2]))
    chk(f"T5 {bb} {dom} mstcr+ood", m, float(row[3].split("(")[0]))
    chk(f"T5 {bb} {dom} mstcr step", m - p, float(re.search(r"\(([+-][\d.]+)\)", row[3]).group(1)))
    chk(f"T5 {bb} {dom} ours", o, float(row[4].split("(")[0]))
    chk(f"T5 {bb} {dom} ours step", o - p, float(re.search(r"\(([+-][\d.]+)\)", row[4]).group(1)))
    lo, hi = row[5].split("→")
    chk(f"T5 {bb} {dom} auroc-lo", cells(src, DOM[dom], "proto_deep")[1], float(lo))
    chk(f"T5 {bb} {dom} auroc-hi", cells(src, DOM[dom], "bal_soft")[1], float(hi))

# ---------------- Table 8: propagation / LaplacianShot ----------------
def tp_cell(domain, method, ratio):
    sel = [x for x in TP if x["domain"] == domain and x["method"] == method
           and x["ratio"] == ratio and x["ood"] == 0.2]
    return float(np.mean([x["accuracy"] for x in sel])) if sel else None


def lap_cell(ratio, tag="ls_k5_lam10.0"):
    sel = [x for x in LAP if x["method"] == tag and x["ratio"] == ratio]
    return float(np.mean([x["accuracy"] for x in sel])) if sel else None


def suite_cell(rows, domain, method, ratio):
    return cells(rows, domain, method, ratio)[0]


for row in T[8][1:]:
    lab = row[0]
    for j, ratio in enumerate(RATIOS):
        want = float(row[1 + j])
        if lab == "miniImageNet prototype on concat":
            chk(f"T8 {lab} r={ratio}", tp_cell("Mini", "proto_cat", ratio), want)
        elif lab == "miniImageNet label propagation ($k$=5)":
            chk(f"T8 {lab} r={ratio}", tp_cell("Mini", "lp_cos_k5", ratio), want)
        elif lab.startswith("miniImageNet LaplacianShot"):
            chk(f"T8 {lab} r={ratio}", lap_cell(ratio), want)
        elif lab == "miniImageNet ours":
            chk(f"T8 {lab} r={ratio}", suite_cell(MINI["resnet18"], "Mini", "bal_soft", ratio), want)
    # 其余行是既有表，略过

# ---------------- Table 9: skewed composition ----------------
ALPHA = {"α=0.3": 0.3, "α=1.0": 1.0, "balanced": float("inf")}
METH = {"single-scale prototype": "proto", "fixed quota": "bal_fixed",
        "estimated quota, 3 passes": "hard3",
        "estimated quota, soft responsibilities": "soft3",
        "calibrated switch (ours)": "adapt"}
for row in T[9][1:]:
    lab = row[0]
    dom = None
    for k, v in (("miniImageNet", "Mini"), ("EuroSAT", "EuroSAT_RGB"), ("DTD", "DTD")):
        if lab.startswith(k):
            dom = v
            rest = lab[len(k):].strip()
            break
    if dom is None:
        continue
    meth = METH.get(rest)
    if meth is None:
        continue
    for j, (colname, alpha) in enumerate(ALPHA.items()):
        if alpha == float("inf"):
            sel = [x["accuracy"] for x in IMB if x["domain"] == dom
                   and x["method"] == meth and x["alpha"] is not None
                   and np.isinf(x["alpha"])]
        else:
            sel = [x["accuracy"] for x in IMB if x["domain"] == dom
                   and x["method"] == meth and x["alpha"] is not None
                   and abs(x["alpha"] - alpha) < 1e-9]
        chk(f"T9 {lab} {colname}", float(np.mean(sel)) if sel else None, float(row[1 + j]))

print(f"table rows parsed: { {k: len(v) for k, v in sorted(T.items())} }")
print(f"checks run: {checks}")
if fails:
    print(f"FAILURES: {len(fails)}")
    for f in fails[:40]:
        print("  -", f)
    sys.exit(1)
print("ALL CELLS MATCH")
