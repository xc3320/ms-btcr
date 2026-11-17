"""Compare the numeric content of every table between the English and Chinese drafts."""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def tables(path: Path):
    text = path.read_text(encoding="utf-8")
    out = {}
    for m in re.finditer(r"\*\*(?:Table|表) (\d+)[\.\*\u3000 ]", text):
        num = int(m.group(1))
        rows = []
        for line in text[m.end():].split("\n"):
            if line.startswith("|"):
                rows.append(line)
            elif line.strip() == "" and rows:
                break
        out[num] = rows
    return out


def nums(rows):
    vals = []
    for line in rows[1:]:
        if all(set(p) <= set("-: ") for p in line.strip().strip("|").split("|")):
            continue
        for cell in line.strip().strip("|").split("|"):
            for v in re.findall(r"-?\d+\.\d+", cell.replace("\u2212", "-")):
                vals.append(round(float(v), 4))
    return vals


en = tables(ROOT / sys.argv[1])
zh = tables(ROOT / sys.argv[2])
bad = 0
for k in sorted(en):
    a, b = nums(en[k]), nums(zh.get(k, []))
    if a == b:
        print(f"table {k}: {len(a):3d} numbers, identical")
    else:
        bad += 1
        print(f"table {k}: EN {len(a)} vs ZH {len(b)} numbers")
        for i, (x, y) in enumerate(zip(a, b)):
            if x != y:
                print(f"   #{i}: EN {x} ZH {y}")
        if len(a) != len(b):
            print("   length differs:", a[len(b):] if len(a) > len(b) else b[len(a):])
print("MISMATCH" if bad else "ALL TABLES CONSISTENT")
