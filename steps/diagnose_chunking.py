"""แก้ chunk size ของ authenticRAG.py แล้ววัดผล (ไม่เรียก LLM)

    python steps/diagnose_chunking.py

ปัญหา: MarkdownNodeParser(chunk_size=256) ตัดตาม heading อย่างเดียว chunk_size ไม่มีผล
วิธีแก้: ตัดตาม heading ก่อน แล้วตัด section ที่ยาวเกินซ้ำด้วย SentenceSplitter
        โดยนับ token ด้วย tokenizer ของ bge-m3 (ค่าเริ่มต้นของ llama-index คือ tiktoken ของ OpenAI ซึ่งนับไทยไม่ตรง)
BM25 ใช้ analyzer thai ตามผลรอบก่อน และไม่ใส่ context จาก LLM (รอบก่อนวัดแล้วว่าแทบไม่มีผล)
ผลลัพธ์: results/diagnosis_chunking.md และ .json
"""
import json
import statistics
from pathlib import Path

import numpy as np
from llama_index.core import Document
from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter

from common import CORPUS, ROOT, embedder, opensearch

K = 5
QUESTIONS = [q for q in json.loads(Path(__file__).with_name("questions.json").read_text(encoding="utf-8")) if q["evidence"]]
FILES = ["1.md", "2.md", "44.md", "5555.md"]
VEC, BM25 = "diag-chunk-vector", "diag-chunk-bm25"
client, model = opensearch(), embedder()
tok = model.tokenizer
ntoks = lambda t: len(tok.encode(t, add_special_tokens=False))  # noqa: E731
full = {f: (CORPUS / f).read_text(encoding="utf-8") for f in FILES}
docs = [Document(text=full[f], metadata={"source": f}) for f in FILES]


def teacher():
    return MarkdownNodeParser(chunk_size=256).get_nodes_from_documents(docs)


def fixed(size):
    def run():
        sections = MarkdownNodeParser().get_nodes_from_documents(docs)
        splitter = SentenceSplitter(chunk_size=size, chunk_overlap=size // 8, tokenizer=lambda t: tok.encode(t, add_special_tokens=False))
        return splitter.get_nodes_from_documents([Document(text=n.text, metadata={"source": n.metadata["source"]}) for n in sections])
    return run


def build(chunks):
    for name in (VEC, BM25):
        client.indices.delete(index=name, ignore=[404])
    client.indices.create(index=BM25, body={"settings": {"analysis": {"analyzer": {"default": {"type": "thai"}}}},
                                            "mappings": {"properties": {"content": {"type": "text", "analyzer": "thai"}}}})
    client.indices.create(index=VEC, body={"settings": {"index.knn": True, "index.knn.space_type": "cosinesimil"},
                                           "mappings": {"properties": {"embedding": {"type": "knn_vector", "dimension": 1024}}}})
    emb = model.encode(chunks, batch_size=8, show_progress_bar=False)
    bulk = []
    for i, (c, e) in enumerate(zip(chunks, emb)):
        bulk += [{"index": {"_index": VEC, "_id": f"doc_{i}"}}, {"embedding": e.tolist()},
                 {"index": {"_index": BM25, "_id": f"doc_{i}"}}, {"content": c}]
    for s in range(0, len(bulk), 200):
        assert not client.bulk(body=bulk[s:s + 200])["errors"]
    client.indices.refresh(index=f"{VEC},{BM25}")


def bm25(q, k=10):
    return [h["_id"] for h in client.search(index=BM25, body={"size": k, "query": {"match": {"content": q}}})["hits"]["hits"]]


def dense(q, k=10):
    v = model.encode(q).tolist()
    return [h["_id"] for h in client.search(index=VEC, body={"size": k, "query": {"knn": {"embedding": {"vector": v, "k": k}}}})["hits"]["hits"]]


def hybrid(q):
    sc = {}
    for r in (bm25(q), dense(q)):
        for rank, d in enumerate(r):
            sc[d] = sc.get(d, 0) + 1 / (60 + rank)
    return sorted(sc, key=sc.get, reverse=True)


report, rows = {}, []
for label, chunker, limit in [("ครู: Markdown อย่างเดียว (chunk_size=256 ไม่มีผล)", teacher, 256),
                              ("แก้: Markdown + split 128", fixed(128), 128),
                              ("แก้: Markdown + split 256", fixed(256), 256),
                              ("แก้: Markdown + split 512", fixed(512), 512),
                              ("แก้: Markdown + split 1024", fixed(1024), 1024)]:
    nodes = chunker()
    chunks, srcs = [n.text for n in nodes], [n.metadata["source"] for n in nodes]
    t = [ntoks(c) for c in chunks]
    ev = [(q["file"], e) for q in QUESTIONS for e in q["evidence"]]
    split_ev = sum(not any(e in c for c, s in zip(chunks, srcs) if s == f) for f, e in ev)
    build(chunks)
    text = {f"doc_{i}": c for i, c in enumerate(chunks)}
    m = {"chunks": len(chunks), "tokens_median": statistics.median(t), "tokens_max": max(t),
         "over_limit_rate": sum(x > limit for x in t) / len(t), "evidence_split": split_ev, "evidence_total": len(ev)}
    for name, fn in (("BM25", bm25), ("dense", dense), ("hybrid", hybrid)):
        ranks, chars = [], []
        for q in QUESTIONS:
            got = fn(q["question"])[:K]
            ranks.append(next((i for i, d in enumerate(got, 1) if any(e in text[d] for e in q["evidence"])), None))
            chars.append(sum(len(text[d]) for d in got))
        n = len(ranks)
        m[name] = {"Hit@1": sum(r == 1 for r in ranks) / n, "Hit@5": sum(r is not None for r in ranks) / n,
                   "MRR@5": sum(1 / r for r in ranks if r) / n, "chars_to_llm": statistics.mean(chars)}
    report[label] = m
    print(label, json.dumps(m, ensure_ascii=False))
for name in (VEC, BM25):
    client.indices.delete(index=name, ignore=[404])

md = [f"# Chunking ({len(QUESTIONS)} คำถาม, top-{K}, BM25 analyzer thai, ไม่มี context)\n",
      "## ขนาด chunk (token ของ bge-m3)\n",
      "| วิธี | chunk | median | max | เกินขนาดที่ตั้ง | evidence ขาดข้าม chunk |", "|---|---|---|---|---|---|"]
md += [f"| {k} | {m['chunks']} | {m['tokens_median']:.0f} | {m['tokens_max']} | {m['over_limit_rate']:.0%} | {m['evidence_split']}/{m['evidence_total']} |"
       for k, m in report.items()]
md += ["\n## Retrieval (Hit@1 / Hit@5 / MRR@5)\n", "| วิธี | BM25 | dense | hybrid | ตัวอักษรส่ง LLM (hybrid) |", "|---|---|---|---|---|"]
fmt = lambda x: f"{x['Hit@1']:.2f} / {x['Hit@5']:.2f} / {x['MRR@5']:.2f}"  # noqa: E731
md += [f"| {k} | {fmt(m['BM25'])} | {fmt(m['dense'])} | {fmt(m['hybrid'])} | {m['hybrid']['chars_to_llm']:,.0f} |" for k, m in report.items()]
print("\n".join(md))
(ROOT / "results" / "diagnosis_chunking.md").write_text("\n".join(md) + "\n", encoding="utf-8")
(ROOT / "results" / "diagnosis_chunking.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
