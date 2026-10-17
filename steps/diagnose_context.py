"""วัดผลขั้น contextual retrieval ของ authenticRAG.py ด้วย LLM จริง (ต้องมี key ใน .env)

    python steps/diagnose_context.py          # analyzer standard แบบครู
    python steps/diagnose_context.py thai     # เปลี่ยน analyzer ของ BM25 เป็น thai

เทียบ 3 แบบ บนชุด chunk เดียวกับโค้ดครู (MarkdownNodeParser(chunk_size=256)):
  none   = ไม่มี context
  bug    = แบบโค้ดครู: <document> คือ chunk ตัวเอง
  fixed  = <document> คือไฟล์เต็มตามที่ Anthropic ออกแบบไว้
context ที่สร้างแล้วเก็บไว้ที่ results/diagnosis_contexts.json (รันซ้ำไม่เรียก LLM ใหม่)
"""
import json
import re
import statistics
import sys
import time
from pathlib import Path

import numpy as np
from llama_index.core import Document
from llama_index.core.node_parser import MarkdownNodeParser

from common import CORPUS, ROOT, embedder, llm, opensearch

K = 5
ANALYZER = sys.argv[1] if sys.argv[1:] else "standard"
QUESTIONS = [q for q in json.loads(Path(__file__).with_name("questions.json").read_text(encoding="utf-8")) if q["evidence"]]
FILES = ["1.md", "2.md", "44.md", "5555.md"]
CACHE = ROOT / "results" / "diagnosis_contexts.json"
VEC, BM25 = "diag-ctx-vector", "diag-ctx-bm25"
SUFFIX = "" if ANALYZER == "standard" else f"_{ANALYZER}"
THAI = re.compile(r"[฀-๿]")
PROMPT = """<document>
{doc}
</document>

Here is the chunk we want to situate within the whole document:
<chunk>
{chunk}
</chunk>

Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval. Answer only with the succinct context and nothing else.
"""

full = {f: (CORPUS / f).read_text(encoding="utf-8") for f in FILES}
nodes = MarkdownNodeParser(chunk_size=256).get_nodes_from_documents(
    [Document(text=full[f], metadata={"source": f}) for f in FILES])
chunks = [n.text for n in nodes]
srcs = [n.metadata["source"] for n in nodes]

cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {"bug": {}, "fixed": {}}
for mode in ("bug", "fixed"):
    for i, (c, s) in enumerate(zip(chunks, srcs)):
        if str(i) in cache[mode]:
            continue
        t = time.time()
        # max_tokens=100 แบบครู ตัดคำตอบของโมเดลที่มีส่วน "คิด" ทิ้งหมด จึงให้ 2000 แล้ววัดความยาวแยก
        cache[mode][str(i)] = llm(PROMPT.format(doc=c if mode == "bug" else full[s], chunk=c), max_tokens=2000, temperature=0.1)
        CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"{mode} {i + 1}/{len(chunks)} ({time.time() - t:.0f}s): {cache[mode][str(i)][:60]!r}")

client, model = opensearch(), embedder()
text = {f"doc_{i}": c for i, c in enumerate(chunks)}


def build(ctx):
    for name in (VEC, BM25):
        client.indices.delete(index=name, ignore=[404])
    client.indices.create(index=BM25, body={  # settings เดียวกับครู
        "settings": {"analysis": {"analyzer": {"default": {"type": ANALYZER}}}, "similarity": {"default": {"type": "BM25"}}},
        "mappings": {"properties": {"content": {"type": "text", "analyzer": ANALYZER},
                                    "contextualized_content": {"type": "text", "analyzer": ANALYZER}}}})
    client.indices.create(index=VEC, body={"settings": {"index.knn": True, "index.knn.space_type": "cosinesimil"},
                                           "mappings": {"properties": {"embedding": {"type": "knn_vector", "dimension": 1024}}}})
    emb = model.encode([f"{c}\n\n{x}" for c, x in zip(chunks, ctx)], batch_size=8, show_progress_bar=False)
    bulk = []
    for i, (c, x, e) in enumerate(zip(chunks, ctx, emb)):
        bulk += [{"index": {"_index": VEC, "_id": f"doc_{i}"}}, {"embedding": e.tolist(), "content": c},
                 {"index": {"_index": BM25, "_id": f"doc_{i}"}}, {"content": c, "contextualized_content": x}]
    for s in range(0, len(bulk), 200):
        assert not client.bulk(body=bulk[s:s + 200])["errors"]
    client.indices.refresh(index=f"{VEC},{BM25}")


def bm25(q, k=10):
    return [h["_id"] for h in client.search(index=BM25, body={"size": k, "query": {"multi_match": {
        "query": q, "fields": ["content", "contextualized_content"], "type": "best_fields"}}})["hits"]["hits"]]


def dense(q, k=10):
    v = model.encode(q).tolist()
    return [h["_id"] for h in client.search(index=VEC, body={"size": k, "query": {"knn": {"embedding": {"vector": v, "k": k}}}})["hits"]["hits"]]


def hybrid(q):
    sc = {}
    for r in (bm25(q), dense(q)):
        for rank, d in enumerate(r):
            sc[d] = sc.get(d, 0) + 1 / (60 + rank)
    return sorted(sc, key=sc.get, reverse=True)


def metrics(fn):
    ranks = []
    for q in QUESTIONS:
        got = fn(q["question"])[:K]
        ranks.append(next((i for i, d in enumerate(got, 1) if any(e in text[d] for e in q["evidence"])), None))
    n = len(ranks)
    return {"Hit@1": sum(r == 1 for r in ranks) / n, "Hit@5": sum(r is not None for r in ranks) / n,
            "MRR@5": sum(1 / r for r in ranks if r) / n}


out, report = [], {}
out.append("## ลักษณะของ context ที่ LLM สร้าง\n")
out.append("| แบบ | ภาษาไทยเป็นหลัก | ความยาวเฉลี่ย (ตัวอักษร) | ระบุชื่อโรคของไฟล์ |\n|---|---|---|---|")
disease = {"1.md": "หัดเยอรมัน", "2.md": "อหิวาต", "44.md": "ต้อกระจก", "5555.md": "กรดไหลย้อน"}
for mode in ("bug", "fixed"):
    ctx = [cache[mode][str(i)] for i in range(len(chunks))]
    thai = statistics.mean(len(THAI.findall(x)) / max(1, len(x)) > 0.5 for x in ctx)
    named = statistics.mean(disease[s] in x for x, s in zip(ctx, srcs))
    report[f"ctx_{mode}"] = {"thai_rate": thai, "mean_chars": statistics.mean(map(len, ctx)), "names_disease_rate": named}
    out.append(f"| {mode} | {thai:.0%} | {statistics.mean(map(len, ctx)):.0f} | {named:.0%} |")

out.append("\n## Retrieval\n\n| context | ระบบ | Hit@1 | Hit@5 | MRR@5 |\n|---|---|---|---|---|")
for mode in ("none", "bug", "fixed"):
    build([""] * len(chunks) if mode == "none" else [cache[mode][str(i)] for i in range(len(chunks))])
    for name, fn in (("BM25", bm25), ("dense", dense), ("hybrid", hybrid)):
        m = metrics(fn)
        report[f"{mode}/{name}"] = m
        out.append(f"| {mode} | {name} | {m['Hit@1']:.2f} | {m['Hit@5']:.2f} | {m['MRR@5']:.2f} |")
for name in (VEC, BM25):
    client.indices.delete(index=name, ignore=[404])

print("\n".join(out))
(ROOT / "results" / f"diagnosis_context{SUFFIX}.md").write_text(
    f"# ผลของ context จาก LLM ({len(QUESTIONS)} คำถาม, top-{K}, BM25 analyzer: {ANALYZER})\n\n" + "\n".join(out) + "\n", encoding="utf-8")
(ROOT / "results" / f"diagnosis_context{SUFFIX}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
