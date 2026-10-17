"""วัดผล ต้นฉบับของอาจารย์ vs ฉบับแก้ไข ด้วยคำถาม 29 ข้อใน steps/questions.json

    python evaluate.py

- ทั้งสองระบบใช้ LLM ตัวเดียวกัน (ค่า LLM_MODEL ใน .env หรือ environment) ทั้งสร้าง context / ตอบ / ให้คะแนน
- ต้นฉบับรันด้วย class ของอาจารย์จริง (import จาก original/authenticRAG.py) เปลี่ยนแค่ชื่อ index และให้เรียก LLM ตัวเดียวกัน
- ผลลัพธ์: results.json และ answers.json
"""
import json
import os
import re
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE  # โฟลเดอร์บนสุดของ repo
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "original"))
from authenticRAG_fixed import MD_PATHS, AnthropicStyleContextualRAG as Fixed  # noqa: E402  (โหลด .env ด้วย)

os.environ.setdefault("OPENROUTER_API_KEY", "unused")  # class ต้นฉบับบังคับให้มี; การเรียก LLM ถูกเปลี่ยนไปใช้ตัวเดียวกับฉบับแก้ไข
import authenticRAG as teacher_module  # noqa: E402

K = 5
QUESTIONS = json.loads((ROOT / "steps" / "questions.json").read_text(encoding="utf-8"))
LABELLED = [q for q in QUESTIONS if q["evidence"]]


class Teacher(teacher_module.AnthropicStyleContextualRAG):
    """โค้ดอาจารย์ทุกบรรทัด ยกเว้น: ชื่อ index (ไม่ทับของเดิม) และ LLM ที่เรียก"""

    def _create_or_update_indices(self):
        if getattr(self, "ready", False):
            super()._create_or_update_indices()

    def setup(self, llm):
        self.vector_index_name, self.bm25_index_name = "eval-teacher-vector-index", "eval-teacher-bm25-index"
        for name in (self.vector_index_name, self.bm25_index_name):
            if self.opensearch_client.indices.exists(index=name):
                self.opensearch_client.indices.delete(index=name)
        self.ready = True
        self._create_or_update_indices()
        self.call_qwen_api = llm
        return self


def judge(llm, q, answer):
    if not q["evidence"]:  # คำถามที่ไม่มีคำตอบในเอกสาร: ระบบควรบอกว่าไม่มีข้อมูล
        prompt = f"""คำถาม: {q['question']}
เอกสารที่ระบบมีไม่มีข้อมูลตอบคำถามนี้ คำตอบของระบบ:
{answer}

คำตอบนี้ยอมรับว่าเอกสารไม่มีข้อมูลเรื่องนี้ (ตอบ 1) หรือแต่งเนื้อหาขึ้นมาตอบเหมือนเป็นข้อเท็จจริง (ตอบ 0)? ตอบเป็นตัวเลขตัวเดียว"""
    else:
        prompt = f"""ให้คะแนนคำตอบของระบบ เทียบกับคำตอบอ้างอิง
คำถาม: {q['question']}
คำตอบอ้างอิง: {q['reference_answer']}
คำตอบของระบบ:
{answer}

1 = มีข้อเท็จจริงหลักของคำตอบอ้างอิงครบและไม่ขัดแย้ง
0.5 = ถูกบางส่วน ขาดข้อเท็จจริงหลักบางข้อ
0 = ผิด ขัดแย้งกับคำตอบอ้างอิง หรือไม่ได้ตอบ
ตอบเป็นตัวเลขตัวเดียว (1, 0.5 หรือ 0)"""
    m = re.search(r"0\.5|1|0", llm(prompt, max_tokens=500, temperature=0.0))
    return float(m.group()) if m else 0.0


def evaluate(name, rag, contexts_of):
    """contexts_of(results) -> ข้อความ context ที่ระบบนั้นส่งให้ LLM จริง"""
    ranks, chars, tokens, scores, answers = [], [], [], {}, {}
    for q in QUESTIONS:
        results = rag.hybrid_search(q["question"], k=K)
        ctx = contexts_of(results)
        if q["evidence"]:
            texts = [d.get("content", "") for _, _, d in results]
            ranks.append(next((i for i, t in enumerate(texts, 1) if any(e in t for e in q["evidence"])), None))
            chars.append(len(ctx))
            tokens.append(len(fixed.tokenizer.encode(ctx, add_special_tokens=False)))
        answers[q["id"]] = rag.generate_response(q["question"], ctx)
        scores[q["id"]] = judge(fixed.call_qwen_api, q, answers[q["id"]])
        print(f"{name} {q['id']} rank={ranks[-1] if q['evidence'] else '-'} score={scores[q['id']]}")
    n = len(ranks)
    by_type = {t: statistics.mean(scores[q["id"]] for q in LABELLED if q["type"] == t) for t in sorted({q["type"] for q in LABELLED})}
    return {"Hit@1": sum(r == 1 for r in ranks) / n, "Hit@5": sum(r is not None for r in ranks) / n,
            "MRR@5": sum(1 / r for r in ranks if r) / n,
            "context_chars": statistics.mean(chars), "context_tokens": statistics.mean(tokens), "context_tokens_max": max(tokens),
            "over_4000_tokens": sum(t > 4000 for t in tokens),
            "answer_score": statistics.mean(scores[q["id"]] for q in LABELLED), "answer_score_by_type": by_type,
            "out_of_corpus_refused": [scores[q["id"]] for q in QUESTIONS if not q["evidence"]],
            "ranks": dict(zip([q["id"] for q in LABELLED], ranks)), "scores": scores}, answers


def chunk_stats(rag, chunks, limit):
    t = [len(rag.tokenizer.encode(c, add_special_tokens=False)) for c in chunks] if hasattr(rag, "tokenizer") else \
        [len(fixed.tokenizer.encode(c, add_special_tokens=False)) for c in chunks]
    return {"chunks": len(chunks), "tokens_median": statistics.median(t), "tokens_max": max(t), "over_limit": sum(x > limit for x in t)}


fixed = Fixed()
teacher = Teacher().setup(fixed.call_qwen_api)
report, answers = {}, {}

# --- ต้นฉบับ: chunk + context + index ด้วยโค้ดอาจารย์ (context จาก chunk ตัวเองตามบั๊ก มี cache จากรอบวัดก่อน)
t_docs = teacher.load_documents([str(ROOT / "corpus_input" / p.name) for p in MD_PATHS])
old = ROOT / "results" / "diagnosis_contexts.json"
cache = {}
if old.exists() and os.environ.get("LLM_MODEL") == "gemini-3.5-flash-lite":  # cache ใช้ได้เฉพาะ LLM ตัวเดียวกัน
    import llama_index.core  # noqa: F401
    from llama_index.core import Document
    from llama_index.core.node_parser import MarkdownNodeParser
    nodes = MarkdownNodeParser(chunk_size=256).get_nodes_from_documents(
        [Document(text=p.read_text(encoding="utf-8")) for p in MD_PATHS])
    cache = {n.text: c for n, c in zip(nodes, json.loads(old.read_text(encoding="utf-8"))["bug"].values())}
_gen = teacher.generate_context
teacher.generate_context = lambda doc, chunk: cache.get(chunk) or _gen(doc, chunk)
teacher.add_documents_with_context(t_docs)

# --- ฉบับแก้ไข
f_docs = fixed.load_documents(MD_PATHS)
fixed.add_documents_with_context(f_docs)


def teacher_context(results):  # เหมือน search_for_question ของอาจารย์
    ctx = "".join(f"Content: {d.get('content', '')}\nContext: {d.get('contextualized_content', '')}\n\n" for _, _, d in results)
    return teacher.truncate_context(ctx)


report["teacher"], answers["teacher"] = evaluate("teacher", teacher, teacher_context)
report["fixed"], answers["fixed"] = evaluate("fixed", fixed, fixed.build_context)
report["teacher"]["chunking"] = chunk_stats(teacher, [d.page_content for d in t_docs], 256)
report["fixed"]["chunking"] = chunk_stats(fixed, [d["page_content"] for d in f_docs], 1024)
report["llm_model"] = fixed.llm_model

for name in ("eval-teacher-vector-index", "eval-teacher-bm25-index"):
    teacher.opensearch_client.indices.delete(index=name, ignore=[404])
(HERE / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
(HERE / "answers.json").write_text(json.dumps(answers, ensure_ascii=False, indent=2), encoding="utf-8")
for name in ("teacher", "fixed"):
    r = report[name]
    print(f"{name:8} Hit@1={r['Hit@1']:.2f} Hit@5={r['Hit@5']:.2f} MRR@5={r['MRR@5']:.2f} "
          f"tokens={r['context_tokens']:,.0f} answer={r['answer_score']:.2f} refused={r['out_of_corpus_refused']}")
