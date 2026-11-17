# -*- coding: utf-8 -*-
"""对照引文摘要做逐字借用检测。
思路：iThenticate 的高相似片段绝大多数来自本文最相关的已发表文献；
把自己稿子与 37 篇引文的摘要逐词比对，取最长连续相同词串，
>= 8 词即可视为需要改写的借用风险。
"""
import re, json, time, urllib.request, urllib.parse, difflib, os, sys

PATH = sys.argv[1] if len(sys.argv) > 1 else "95_English_draft_sci_v5.md"
CACHE = "_abstract_cache.json"

raw = open(PATH, encoding="utf-8").read()
body, refs = raw.split("## References", 1)

dois = []
for m in re.finditer(r"doi:\s*(10\.[^\s,;]+)", refs):
    d = m.group(1).rstrip(".")
    dois.append(d)
print(f"参考文献中带 DOI 的条目: {len(dois)}")

cache = json.load(open(CACHE, encoding="utf-8")) if os.path.exists(CACHE) else {}
def get(d):
    if d in cache:
        return cache[d]
    url = "https://api.openalex.org/works/doi:" + urllib.parse.quote(d) + "?mailto=workbuddy@example.com"
    try:
        data = json.loads(urllib.request.urlopen(url, timeout=25).read())
        inv = data.get("abstract_inverted_index")
        txt = None
        if inv:
            pos = {}
            for w, idxs in inv.items():
                for i in idxs:
                    pos[i] = w
            txt = " ".join(pos[i] for i in sorted(pos))
        cache[d] = {"title": data.get("title"), "abstract": txt}
    except Exception as e:
        cache[d] = {"title": None, "abstract": None, "err": str(e)}
    time.sleep(0.4)
    return cache[d]

abstracts = {}
for d in dois:
    r = get(d)
    if r.get("abstract"):
        abstracts[d] = r
json.dump(cache, open(CACHE, "w", encoding="utf-8"), ensure_ascii=False)
print(f"取到摘要: {len(abstracts)} / {len(dois)}")

WORD = re.compile(r"[A-Za-z][A-Za-z\-']+|\d+(?:\.\d+)?")
def toks(t):
    return WORD.findall(re.sub(r"<[^>]+>", " ", t).lower())

body_t = toks(body)
MIN = 8
rows = []
for d, r in abstracts.items():
    a = toks(r["abstract"])
    if len(a) < 20:
        continue
    sm = difflib.SequenceMatcher(None, a, body_t, autojunk=False)
    blocks = [b for b in sm.get_matching_blocks() if b.size >= MIN]
    total = sum(b.size for b in blocks)
    top = max(blocks, key=lambda b: b.size) if blocks else None
    rows.append({
        "doi": d,
        "title": (r["title"] or "")[:60],
        "abs_words": len(a),
        "max_run": top.size if top else 0,
        "frag": " ".join(a[top.a:top.a + top.size]) if top else "",
        "matched_words": total,
        "cover": total / len(a),
    })

rows.sort(key=lambda z: (-z["max_run"], -z["matched_words"]))
print(f"\n比对 {len(rows)} 篇摘要，最长连续相同词串排序（阈值 {MIN} 词）：\n")
for r in rows[:10]:
    flag = "  !! 需改写" if r["max_run"] >= MIN else ""
    print(f"  run={r['max_run']:>2}  cover={r['cover']*100:>4.1f}%  {r['title']}{flag}")
    if r["frag"]:
        print(f"        «{r['frag'][:120]}»")

bad = [r for r in rows if r["max_run"] >= MIN]
print(f"\n结论：{len(bad)} / {len(rows)} 篇存在 >= {MIN} 词连续相同片段")
if bad:
    print("命中合计字数占比最高:", f"{max(r['cover'] for r in bad)*100:.1f}%")
