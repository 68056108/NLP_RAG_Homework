"""ตรวจระบบ authenticRAG.py ของครูทีละขั้น แล้ววัดเป็นตัวเลข (ไม่ต้องใช้ LLM API key)

    python steps/diagnose_authenticRAG.py

ใช้ค่าตั้งเดียวกับโค้ดครูทุกอย่าง (MarkdownNodeParser(chunk_size=256), analyzer standard, knn mapping,
ค้นฝั่งละ 10, RRF rank เริ่ม 0) แต่สร้าง index ชื่อ diag-* แยกไว้ ไม่ไปทับ index ของครู
ขั้นสร้าง context ด้วย LLM วัดเฉพาะ "ข้อมูลที่ส่งเข้าไป" (ไม่เรียก LLM จริง)
ผลลัพธ์: results/diagnosis.md และ results/diagnosis.json
"""
import json
import re
import statistics
from pathlib import Path

import numpy as np
from llama_index.core import Document
from llama_index.core.node_parser import MarkdownNodeParser

from common import CORPUS, ROOT, embedder, opensearch

K = 5
QUESTIONS = [q for q in json.loads(Path(__file__).with_name("questions.json").read_text(encoding="utf-8")) if q["evidence"]]
FILES = ["1.md", "2.md", "44.md", "5555.md"]
VEC, BM25 = "diag-teacher-vector", "diag-teacher-bm25"
THAI = re.compile(r"[฀-๿]")
client = opensearch()
model = embedder()
tok = model.tokenizer
report, out = {}, []


def say(line=""):
    print(line)
    out.append(line)


def pct(x):
    return f"{100 * x:.1f}%"


# ---------------------------------------------------------------- 1) โหลดไฟล์
say("## 1) โหลดเอกสาร (TextLoader)")
from langchain_community.document_loaders import TextLoader  # noqa: E402

full = {f: (CORPUS / f).read_text(encoding="utf-8") for f in FILES}
pages = [p for f in FILES for p in TextLoader(str(CORPUS / f)).load()]
lost = sum(len(full[f]) for f in FILES) - sum(len(p.page_content) for p in pages)
say(f"- ไฟล์ {len(FILES)} ไฟล์, ตัวอักษรที่หายระหว่างโหลด: {lost}")
report["load"] = {"files": len(FILES), "chars_lost": lost}

# ---------------------------------------------------------------- 2) chunking
say("\n## 2) Chunking — MarkdownNodeParser(chunk_size=256)")
nodes = MarkdownNodeParser(chunk_size=256).get_nodes_from_documents(
    [Document(text=p.page_content, metadata=p.metadata) for p in pages])
chunks = [n.text for n in nodes]
srcs = [Path(n.metadata.get("source", "")).name for n in nodes]
ntok = [len(tok.encode(c, add_special_tokens=False)) for c in chunks]
nchr = [len(c) for c in chunks]
over = sum(t > 256 for t in ntok)
tiny = sum(c < 50 for c in nchr)
max_seq = model.max_seq_length
cut = sum(max(0, t - max_seq) for t in ntok) / sum(ntok)
# evidence ที่ถูกตัดขาด: มีในไฟล์เต็ม แต่ไม่มี chunk ไหนเก็บไว้ครบ
ev = [(q["file"], e) for q in QUESTIONS for e in q["evidence"] if e in full[q["file"]]]
split_ev = sum(not any(e in c for c, s in zip(chunks, srcs) if s == f) for f, e in ev)
say(f"- {len(chunks)} chunks; token (tokenizer ของ bge-m3) min={min(ntok)} median={statistics.median(ntok):.0f} "
    f"p90={np.percentile(ntok, 90):.0f} max={max(ntok)}")
say(f"- chunk ที่เกิน 256 token (ค่าที่ตั้งไว้): {over}/{len(chunks)} = {pct(over / len(chunks))}"
    f"  -> MarkdownNodeParser ตัดตาม heading เท่านั้น ไม่สน chunk_size")
say(f"- chunk เล็กมาก (< 50 ตัวอักษร เช่นมีแต่หัวข้อ): {tiny} = {pct(tiny / len(chunks))}")
say(f"- เนื้อหาที่เกิน max_seq_length={max_seq} ของ embedding (ถูกตัดทิ้งตอน embed): {pct(cut)}")
say(f"- evidence ที่ถูกตัดขาดข้าม chunk: {split_ev}/{len(ev)}")
report["chunking"] = {"chunks": len(chunks), "tokens_median": statistics.median(ntok), "tokens_max": max(ntok),
                      "over_256_tokens": over, "over_256_rate": over / len(chunks), "tiny_chunks": tiny,
                      "embed_truncated_rate": cut, "evidence_split": split_ev, "evidence_total": len(ev)}

# ---------------------------------------------------------------- 3) contextual retrieval: input ที่ส่งให้ LLM
say("\n## 3) Contextual retrieval — เอกสารที่ส่งให้ LLM สร้าง context")
# โค้ดครู: full_doc_content = doc.page_content  (ส่ง chunk ตัวเองไปเป็น <document>)
cover = [len(c) / len(full[s]) for c, s in zip(chunks, srcs)]
same = sum(1 for _ in chunks)  # ทุก chunk: <document> == <chunk>
say(f"- chunk ที่ <document> เป็นข้อความเดียวกับ <chunk>: {same}/{len(chunks)} = {pct(same / len(chunks))}")
say(f"- LLM เห็นเอกสารต้นฉบับเฉลี่ยเพียง {pct(statistics.mean(cover))} ของไฟล์ (ควรเป็น 100%)")
say(f"- เรียก LLM {len(chunks)} ครั้ง/การ index หนึ่งรอบ (ไม่มี cache)")
report["contextual"] = {"document_equals_chunk_rate": same / len(chunks), "mean_doc_coverage": statistics.mean(cover),
                        "llm_calls": len(chunks)}

# ---------------------------------------------------------------- 4) analyzer
say("\n## 4) Analyzer ของ BM25 index — standard vs thai")
for name in (VEC, BM25):
    client.indices.delete(index=name, ignore=[404])
client.indices.create(index=BM25, body={  # settings เดียวกับครู
    "settings": {"analysis": {"analyzer": {"default": {"type": "standard"}}},
                 "similarity": {"default": {"type": "BM25"}}, "index.queries.cache.enabled": False},
    "mappings": {"properties": {"content": {"type": "text", "analyzer": "standard"},
                                "contextualized_content": {"type": "text", "analyzer": "standard"},
                                "doc_id": {"type": "keyword"}}}})


def analyze(analyzer, text):
    return [t["token"] for t in client.indices.analyze(body={"analyzer": analyzer, "text": text})["tokens"]]


an = {}
for a in ("standard", "thai"):
    vocab, toks = set(), []
    for c in chunks:
        t = analyze(a, c)
        toks += t
        vocab |= set(t)
    th = [t for t in toks if THAI.search(t)]
    qhit = [np.mean([t in vocab for t in analyze(a, q["question"])] or [0]) for q in QUESTIONS]
    an[a] = {"tokens": len(toks), "vocab": len(vocab), "thai_token_mean_len": statistics.mean(map(len, th)),
             "thai_tokens_over_20_chars": sum(len(t) > 20 for t in th) / len(th),
             "query_term_found_in_corpus": float(np.mean(qhit))}
    say(f"- {a:8}: token ทั้งหมด {len(toks):,}, ความยาว token ไทยเฉลี่ย {an[a]['thai_token_mean_len']:.1f} ตัวอักษร, "
        f"token ไทยยาว > 20 ตัวอักษร {pct(an[a]['thai_tokens_over_20_chars'])}, "
        f"คำในคำถามที่พบใน corpus {pct(an[a]['query_term_found_in_corpus'])}")
q0 = QUESTIONS[1]["question"]
say(f"- ตัวอย่าง {q0!r}\n  - standard -> {analyze('standard', q0)}\n  - thai     -> {analyze('thai', q0)}")
report["analyzer"] = an

# ---------------------------------------------------------------- 5) index + retrieval
say("\n## 5) Retrieval (ใช้ settings ของครู, ไม่มี context จาก LLM)")
client.indices.create(index=VEC, body={  # settings เดียวกับครู
    "settings": {"index.knn": True, "index.knn.space_type": "cosinesimil"},
    "mappings": {"properties": {"embedding": {"type": "knn_vector", "dimension": model.get_sentence_embedding_dimension()},
                                "doc_id": {"type": "keyword"}, "content": {"type": "text"}}}})
emb = model.encode(chunks, batch_size=8, show_progress_bar=False)
bulk = []
for i, (c, e) in enumerate(zip(chunks, emb)):
    bulk += [{"index": {"_index": VEC, "_id": f"doc_{i}"}}, {"embedding": e.tolist(), "doc_id": f"doc_{i}", "content": c},
             {"index": {"_index": BM25, "_id": f"doc_{i}"}}, {"content": c, "contextualized_content": "", "doc_id": f"doc_{i}"}]
for s in range(0, len(bulk), 200):
    assert not client.bulk(body=bulk[s:s + 200])["errors"]
client.indices.refresh(index=f"{VEC},{BM25}")
space = client.indices.get_mapping(index=VEC)[VEC]["mappings"]["properties"]["embedding"]
say(f"- knn mapping ที่ได้จริง: {json.dumps(space)}  (setting index.knn.space_type เป็นแบบเก่า)")


def bm25(q, k=10):
    return [h["_id"] for h in client.search(index=BM25, body={"size": k, "query": {"multi_match": {
        "query": q, "fields": ["content", "contextualized_content"], "type": "best_fields"}}})["hits"]["hits"]]


def dense(q, k=10):
    v = model.encode(q).tolist()
    return [h["_id"] for h in client.search(index=VEC, body={"size": k, "query": {"knn": {"embedding": {"vector": v, "k": k}}}})["hits"]["hits"]]


def exact(q, k=10):
    v = model.encode(q)
    sims = emb @ v / (np.linalg.norm(emb, axis=1) * np.linalg.norm(v))
    return [f"doc_{i}" for i in np.argsort(-sims)[:k]]


def rrf(rankings, start):
    sc = {}
    for r in rankings:
        for rank, d in enumerate(r, start):
            sc[d] = sc.get(d, 0) + 1 / (60 + rank)
    return sorted(sc, key=sc.get, reverse=True)


text = {f"doc_{i}": c for i, c in enumerate(chunks)}


def metrics(fn):
    ranks, chars = [], []
    for q in QUESTIONS:
        got = fn(q["question"])[:K]
        r = next((i for i, d in enumerate(got, 1) if any(e in text[d] for e in q["evidence"])), None)
        ranks.append(r)
        chars.append(sum(len(text[d]) for d in got))
    n = len(ranks)
    return {"Hit@1": sum(r == 1 for r in ranks) / n, "Hit@5": sum(r is not None for r in ranks) / n,
            "MRR@5": sum(1 / r for r in ranks if r) / n, "zero_results": sum(not fn(q["question"]) for q in QUESTIONS),
            "chars_to_llm": statistics.mean(chars)}


systems = {"BM25 (standard)": bm25, "dense (knn)": dense, "dense (exact cosine)": exact,
           "hybrid RRF (ครู)": lambda q: rrf([bm25(q), dense(q)], 0)}
say("\n| ระบบ | Hit@1 | Hit@5 | MRR@5 | คำถามที่ไม่ได้ผลเลย | ตัวอักษรที่ส่ง LLM (top-5) |\n|---|---|---|---|---|---|")
report["retrieval"] = {}
for name, fn in systems.items():
    m = metrics(fn)
    report["retrieval"][name] = m
    say(f"| {name} | {m['Hit@1']:.2f} | {m['Hit@5']:.2f} | {m['MRR@5']:.2f} | {m['zero_results']}/{len(QUESTIONS)} | {m['chars_to_llm']:,.0f} |")

ann = statistics.mean(len(set(dense(q["question"])) & set(exact(q["question"]))) / 10 for q in QUESTIONS)
rrf_diff = sum(rrf([bm25(q["question"]), dense(q["question"])], 0)[:K] != rrf([bm25(q["question"]), dense(q["question"])], 1)[:K]
               for q in QUESTIONS)
say(f"\n- knn เทียบ exact cosine: recall@10 = {ann:.2f}")
say(f"- RRF rank เริ่ม 0 vs 1: top-5 ต่างกัน {rrf_diff}/{len(QUESTIONS)} คำถาม")
report["retrieval_checks"] = {"knn_recall_at_10_vs_exact": ann, "rrf_rank0_vs_rank1_top5_diff": rrf_diff}

# ---------------------------------------------------------------- 6) truncate_context
say("\n## 6) truncate_context — ประมาณ token ด้วย len(text.split()) * 1.3")
est, real = [], []
for q in QUESTIONS:
    ctx = "".join(f"Content: {text[d]}\nContext: \n\n" for d in rrf([bm25(q["question"]), dense(q["question"])], 0)[:K])
    est.append(len(ctx.split()) * 1.3)
    real.append(len(tok.encode(ctx, add_special_tokens=False)))
ratio = statistics.mean(r / e for r, e in zip(real, est))
miss = sum(e <= 4000 < r for e, r in zip(est, real))
say(f"- token จริง (tokenizer bge-m3) / ค่าประมาณ = {ratio:.1f} เท่า (ประมาณต่ำกว่าจริง)")
say(f"- context ที่เกิน 4,000 token จริงแต่ไม่ถูกตัด: {miss}/{len(QUESTIONS)}; token จริง median={statistics.median(real):,.0f} max={max(real):,}")
report["truncate"] = {"real_over_estimate": ratio, "missed_truncations": miss, "real_tokens_max": max(real)}

for name in (VEC, BM25):
    client.indices.delete(index=name, ignore=[404])
(ROOT / "results").mkdir(exist_ok=True)
(ROOT / "results" / "diagnosis.md").write_text(
    f"# ผลตรวจ authenticRAG ทีละขั้น ({len(QUESTIONS)} คำถาม, top-{K})\n\n" + "\n".join(out) + "\n", encoding="utf-8")
(ROOT / "results" / "diagnosis.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
say("\nบันทึกที่ results/diagnosis.md และ results/diagnosis.json")
