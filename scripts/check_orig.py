# -*- coding: utf-8 -*-
"""投稿前自检：稿件内部重复度 + AI 写作痕迹分布。
不做查重（无 iThenticate 权限），只做能确定性算出来的两件事：
  1) 稿内段落/长句重复（编辑一眼能看出的注水与自我复制）
  2) 句长分布、句首模板、套话密度（AI 味的可量化部分）
"""
import re, sys, json, difflib
from collections import Counter, defaultdict

PATH = sys.argv[1] if len(sys.argv) > 1 else "95_English_draft_sci_v5.md"

raw = open(PATH, encoding="utf-8").read()
lines = raw.split("\n")

# 切掉参考文献
body_lines, in_refs = [], False
for ln in lines:
    if re.match(r"^##\s*References", ln):
        in_refs = True
    if not in_refs:
        body_lines.append(ln)
body = "\n".join(body_lines)

# 段落化，带小节标签
paras, sec = [], "Front"
for ln in body.split("\n"):
    s = ln.strip()
    if not s:
        continue
    if s.startswith("#"):
        sec = s.lstrip("#").strip()
        continue
    if s.startswith("|") or re.match(r"^\*\*(Table|Figure|Fig\.)", s):
        continue
    paras.append({"sec": sec, "text": s})

print(f"文件: {PATH}")
print(f"正文段落数: {len(paras)}")

WORD = re.compile(r"[A-Za-z][A-Za-z\-']+|\d+(?:\.\d+)?")
def toks(t):
    return WORD.findall(t.lower())

# ---------- 1. 段内长重复（>=10 词连续相同） ----------
SM_MIN = 10
hits = []
for i in range(len(paras)):
    for j in range(i + 1, len(paras)):
        a, b = toks(paras[i]["text"]), toks(paras[j]["text"])
        if len(a) < SM_MIN or len(b) < SM_MIN:
            continue
        sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
        if sm.quick_ratio() < 0.35:
            continue
        best = max(sm.get_matching_blocks(), key=lambda m: m.size)
        if best.size >= SM_MIN:
            hits.append((best.size, i, j, " ".join(a[best.a:best.a + best.size])))
hits.sort(reverse=True)
print(f"\n[1] 跨段落连续重复 >= {SM_MIN} 词：{len(hits)} 处")
for size, i, j, frag in hits[:12]:
    print(f"  {size:>3} 词 | {paras[i]['sec'][:28]} #{i}  vs  {paras[j]['sec'][:28]} #{j}")
    print(f"        «{frag[:110]}»")

# ---------- 2. 全稿 n-gram 重复（模板化写作） ----------
for N in (8, 12):
    grams = Counter()
    where = defaultdict(set)
    for k, p in enumerate(paras):
        t = toks(p["text"])
        for x in range(len(t) - N + 1):
            g = " ".join(t[x:x + N])
            grams[g] += 1
            where[g].add(k)
    dup = [(g, c) for g, c in grams.items() if len(where[g]) > 1]
    dup.sort(key=lambda z: -z[1])
    print(f"\n[2] 出现在不同段落的 {N}-gram：{len(dup)} 条")
    for g, c in dup[:8]:
        print(f"  x{c}  «{g[:100]}»")

# ---------- 3. 句级统计 ----------
text = " ".join(p["text"] for p in paras)
text = re.sub(r"\$[^$]*\$", " MATH ", text)
sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.split()) >= 4]
lens = [len(s.split()) for s in sents]
n = len(lens)
mean = sum(lens) / n
sd = (sum((x - mean) ** 2 for x in lens) / n) ** 0.5
print(f"\n[3] 句子统计：{n} 句，平均 {mean:.1f} 词，标准差 {sd:.1f}，"
      f"变异系数 {sd/mean:.2f}")
print(f"    最长 {max(lens)} 词 / 最短 {min(lens)} 词；"
      f">40 词的长句 {sum(1 for x in lens if x > 40)} 句 "
      f"({sum(1 for x in lens if x > 40)/n*100:.0f}%)")

starts = Counter(s.split()[0].lower() for s in sents)
print("    句首词 Top8：" + ", ".join(f"{w}({c})" for w, c in starts.most_common(8)))

# ---------- 4. 套话密度 ----------
low = text.lower()
flags = {
    "moreover": r"\bmoreover\b",
    "furthermore": r"\bfurthermore\b",
    "notably": r"\bnotably\b",
    "crucially": r"\bcrucially\b",
    "importantly": r"\bimportantly\b",
    "in summary": r"\bin summary\b",
    "it is worth noting": r"\bit is worth noting\b",
    "plays a crucial role": r"\bplays? a (crucial|key|vital|significant) role\b",
    "delve": r"\bdelve\b",
    "leverage": r"\bleverag\w*",
    "comprehensive": r"\bcomprehensive\b",
    "novel": r"\bnovel\b",
    "in this paper, we": r"\bin this paper,? we\b",
    "to the best of our knowledge": r"\bto the best of our knowledge\b",
    "shed light": r"\bshed light\b",
    "paves the way": r"\bpaves? the way\b",
    "em dash": r"—",
}
print("\n[4] 套话/特征词计数：")
rows = [(k, len(re.findall(v, low))) for k, v in flags.items()]
for k, c in sorted(rows, key=lambda z: -z[1]):
    mark = "  <-- 偏高" if c >= 3 and k not in ("comprehensive", "novel", "em dash") else ""
    print(f"    {k:<28} {c}{mark}")

total_words = len(toks(text))
print(f"\n[5] 正文词数（去表格去参考文献）：{total_words}")
print(f"    套话合计 {sum(c for k,c in rows if k!='em dash')} 次，"
      f"密度 {sum(c for k,c in rows if k!='em dash')/total_words*1000:.1f} 次/千词")
