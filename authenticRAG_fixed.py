"""authenticRAG.py ของอาจารย์ (2025-authenticRAG) ฉบับแก้ไข

โครงสร้าง class / ชื่อ method เหมือนต้นฉบับ จุดที่แก้มีป้าย "FIX n" ตรงกับหัวข้อใน CHANGES.md

    python authenticRAG_fixed/authenticRAG_fixed.py              # สร้าง index ใหม่ แล้วถาม 2 คำถามตัวอย่าง
    python authenticRAG_fixed/authenticRAG_fixed.py --search     # ไม่ index ใหม่ ถามอย่างเดียว (แทน onlysearchAuthenticRAG.py)

ตั้งค่า LLM ผ่าน environment หรือไฟล์ .env ที่โฟลเดอร์ kdai-rag-workshop:
    OPENROUTER_API_KEY=...                       (ค่าเดิมของอาจารย์: Qwen ผ่าน OpenRouter)
  หรือ LLM_API_KEY / LLM_BASE_URL / LLM_MODEL   (เช่น Gemini ที่รองรับ OpenAI SDK)
"""
import json
import os
import re
import sys
import time
from pathlib import Path

from openai import APIConnectionError, InternalServerError, OpenAI, RateLimitError
from opensearchpy import OpenSearch
from sentence_transformers import SentenceTransformer
from llama_index.core import Document
from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

# FIX 9: อ่าน .env และรองรับ LLM เจ้าอื่นที่ใช้ OpenAI SDK ได้ (ต้นฉบับผูกกับ OPENROUTER_API_KEY อย่างเดียว)
_env = ROOT / ".env"
if _env.exists():
    for _line in _env.read_text(encoding="utf-8").splitlines():
        if "=" in _line and not _line.lstrip().startswith("#"):
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())


class AnthropicStyleContextualRAG:
    def __init__(self, opensearch_host="localhost", opensearch_port=9200,
                 vector_index_name="authentic-fixed-vector-index", bm25_index_name="authentic-fixed-bm25-index",
                 chunk_size=1024, context_cache=HERE / "contexts_cache.json"):
        self.api_key = os.environ.get("LLM_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("ยังไม่ได้ตั้ง OPENROUTER_API_KEY (หรือ LLM_API_KEY)")
        self.llm_model = os.environ.get("LLM_MODEL", "qwen/qwen-2.5-72b-instruct")
        self.client = OpenAI(api_key=self.api_key, base_url=os.environ.get("LLM_BASE_URL", "https://openrouter.ai/api/v1"))

        self.embed_model = SentenceTransformer('BAAI/bge-m3')
        self.embedding_dim = self.embed_model.get_sentence_embedding_dimension()
        self.tokenizer = self.embed_model.tokenizer

        class EmbeddingsWrapper:
            def __init__(self, embed_model):
                self.embed_model = embed_model

            def embed_query(self, text):
                return self.embed_model.encode(text).tolist()

            def embed_documents(self, documents):
                return self.embed_model.encode(documents, batch_size=8).tolist()

        self.encoder = EmbeddingsWrapper(self.embed_model)

        self.opensearch_client = OpenSearch(hosts=[{'host': opensearch_host, 'port': opensearch_port}],
                                            use_ssl=False, timeout=60)

        # FIX 6: ชื่อ index ใหม่ ไม่ปนกับ index เดิมของอาจารย์ (ซึ่งเป็น analyzer standard)
        self.vector_index_name = vector_index_name
        self.bm25_index_name = bm25_index_name
        self.rrf_k = 60

        # FIX 2: MarkdownNodeParser ไม่สน chunk_size (ตัดตาม heading อย่างเดียว ได้ chunk ยาวถึง 2,431 token)
        # -> ตัดตาม heading ก่อน แล้วตัด section ที่ยาวเกินด้วย SentenceSplitter
        #    นับ token ด้วย tokenizer ของ bge-m3 (ค่าเริ่มต้นของ llama-index คือ tiktoken ของ OpenAI ซึ่งนับไทยไม่ตรง)
        #    ขนาด 1024 มาจากการวัด 128/256/512/1024 (ดู CHANGES.md)
        self.md_parser = MarkdownNodeParser()
        self.splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_size // 8,
                                         tokenizer=lambda t: self.tokenizer.encode(t, add_special_tokens=False))

        # FIX 8: cache context ไว้ในไฟล์ รัน index ซ้ำไม่ต้องเรียก LLM ใหม่ทุก chunk
        self.context_cache_path = Path(context_cache)
        self.context_cache = json.loads(self.context_cache_path.read_text(encoding="utf-8")) if self.context_cache_path.exists() else {}

    def _create_or_update_indices(self, recreate=False):
        """สร้าง indices ที่ใช้เก็บ embeddings และ BM25"""
        # FIX 6: ต้นฉบับสร้างเฉพาะตอน "ยังไม่มี" -> ถ้า index เก่ามีอยู่แล้ว settings ใหม่ (เช่น analyzer) จะไม่ถูกใช้เลย
        # และ chunk เก่าที่ id เกินจำนวนใหม่ค้างอยู่ใน index -> ตอน index ใหม่ให้ลบแล้วสร้างใหม่
        if recreate:
            for name in (self.vector_index_name, self.bm25_index_name):
                if self.opensearch_client.indices.exists(index=name):
                    self.opensearch_client.indices.delete(index=name)

        if not self.opensearch_client.indices.exists(index=self.vector_index_name):
            vector_settings = {
                "settings": {"index.knn": True},
                "mappings": {
                    "properties": {
                        # FIX 7: ระบุ space_type ใน method ของ field (setting index.knn.space_type ระดับ index เป็นแบบเก่า)
                        "embedding": {"type": "knn_vector", "dimension": self.embedding_dim,
                                      "method": {"name": "hnsw", "engine": "lucene", "space_type": "cosinesimil"}},
                        "doc_id": {"type": "keyword"},
                        "source": {"type": "keyword"},
                        "content": {"type": "text"},
                        "chunk_id": {"type": "keyword"}
                    }
                }
            }
            self.opensearch_client.indices.create(index=self.vector_index_name, body=vector_settings)
            print(f"สร้าง vector index {self.vector_index_name} สำเร็จ")

        if not self.opensearch_client.indices.exists(index=self.bm25_index_name):
            # FIX 1: analyzer standard ตัดคำไทยไม่ได้ (ทั้งประโยคเป็น token เดียว คำในคำถามเจอใน corpus แค่ 5.7%)
            # -> ใช้ analyzer thai ของ OpenSearch (เจอ 91.3%)
            bm25_settings = {
                "settings": {
                    "analysis": {"analyzer": {"default": {"type": "thai"}}},
                    "similarity": {"default": {"type": "BM25"}},
                    "index.queries.cache.enabled": False
                },
                "mappings": {
                    "properties": {
                        "content": {"type": "text", "analyzer": "thai"},
                        "contextualized_content": {"type": "text", "analyzer": "thai"},
                        "doc_id": {"type": "keyword"},
                        "source": {"type": "keyword"},
                        "chunk_id": {"type": "keyword"},
                        "original_index": {"type": "integer", "index": False}
                    }
                }
            }
            self.opensearch_client.indices.create(index=self.bm25_index_name, body=bm25_settings)
            print(f"สร้าง BM25 index {self.bm25_index_name} สำเร็จ")

    def call_qwen_api(self, prompt, max_tokens=500, temperature=0.2):
        """เรียก LLM เพื่อสร้างข้อความ"""
        # FIX 5: ต้นฉบับ return ข้อความ "Qwen API error: ..." เมื่อเรียกไม่สำเร็จ แล้วข้อความนี้ถูกเก็บเป็น context ใน index
        # -> retry เมื่อชน rate limit / server ล่มชั่วคราว ถ้ายังไม่ได้ให้ raise (ไม่เก็บขยะลง index)
        for attempt in range(6):
            try:
                completion = self.client.chat.completions.create(
                    model=self.llm_model,
                    messages=[{"role": "system", "content": "You are a helpful assistant."},
                              {"role": "user", "content": prompt}],
                    max_tokens=max_tokens, temperature=temperature)
                text = completion.choices[0].message.content or ""
                return re.sub(r"(?s)<thought>.*?(</thought>|$)", "", text).strip()
            except (RateLimitError, InternalServerError, APIConnectionError) as e:
                m = re.search(r"retry in ([\d.]+)s", str(e))
                wait = float(m.group(1)) + 2 if m else 20
                if attempt == 5 or wait > 300:
                    raise
                print(f"  ({type(e).__name__}, รอ {wait:.0f} วินาที)")
                time.sleep(wait)

    def load_documents(self, md_paths):
        """โหลดเอกสาร Markdown และแบ่งเป็น chunks"""
        print(f"Loading documents from {len(md_paths)} files...")
        # FIX 3: เก็บเอกสารเต็มไว้ เพื่อส่งให้ LLM ตอนสร้าง context (ต้นฉบับไม่เก็บ เลยส่ง chunk แทน)
        # ตัด TextLoader/LangChain ออก: อ่านไฟล์ตรง ๆ ได้ผลเท่ากัน (วัดแล้ว ตัวอักษรหาย 0) และไม่ต้องพึ่ง langchain รุ่นเก่า
        self.full_docs = {}
        llama_docs = []
        for path in md_paths:
            text = Path(path).read_text(encoding="utf-8")
            self.full_docs[str(path)] = text
            llama_docs.append(Document(text=text, metadata={"source": str(path)}))

        sections = self.md_parser.get_nodes_from_documents(llama_docs)
        chunks = self.splitter.get_nodes_from_documents(
            [Document(text=s.text, metadata={"source": s.metadata["source"]}) for s in sections])
        print(f"Split documents into {len(chunks)} chunks (จาก {len(sections)} sections)")
        return [{"page_content": c.text, "metadata": {"source": c.metadata["source"]}} for c in chunks]

    def get_context_prompt(self, document_content, chunk_content):
        """prompt สร้างบริบท (เหมือนตัวอย่าง Anthropic และต้นฉบับ)"""
        return f"""<document>
{document_content}
</document>

Here is the chunk we want to situate within the whole document:
<chunk>
{chunk_content}
</chunk>

Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval. Answer only with the succinct context and nothing else.
"""

    def generate_context(self, document_content, chunk_content):
        """สร้างบริบทด้วย LLM (มี cache)"""
        key = f"{len(document_content)}:{chunk_content}"
        if key not in self.context_cache:
            # max_tokens 100 ของต้นฉบับพอสำหรับ Qwen แต่โมเดลที่มีส่วน "คิด" ใช้ token หมดก่อนตอบ ได้คำตอบว่าง -> 1000
            # (prompt สั่งให้ตอบสั้นอยู่แล้ว วัดได้เฉลี่ย ~146 ตัวอักษร)
            self.context_cache[key] = self.call_qwen_api(self.get_context_prompt(document_content, chunk_content),
                                                         max_tokens=1000, temperature=0.1)
            self.context_cache_path.write_text(json.dumps(self.context_cache, ensure_ascii=False, indent=1), encoding="utf-8")
        return self.context_cache[key]

    def add_documents_with_context(self, docs):
        """เพิ่มเอกสารเข้า OpenSearch พร้อมบริบทเสริม"""
        print(f"\nAdding {len(docs)} documents to OpenSearch with contextual information...")
        self._create_or_update_indices(recreate=True)

        contexts, texts = [], []
        for i, doc in enumerate(docs):
            # FIX 3: ต้นฉบับใช้ full_doc_content = doc.page_content -> LLM เห็นแค่ chunk ตัวเอง (เฉลี่ย 7.5% ของไฟล์)
            source = doc["metadata"]["source"]
            context = self.generate_context(self.full_docs[source], doc["page_content"])
            print(f"Generated context for chunk {i + 1}/{len(docs)}: {context[:50]}...")
            contexts.append(context)
            texts.append(f"{doc['page_content']}\n\n{context}")
        embeddings = self.encoder.embed_documents(texts)

        bulk = []
        for i, (doc, context, embedding) in enumerate(zip(docs, contexts, embeddings)):
            source = Path(doc["metadata"]["source"]).name
            bulk += [{"index": {"_index": self.vector_index_name, "_id": f"doc_{i}"}},
                     {"embedding": embedding, "doc_id": f"doc_{i}", "source": source, "content": doc["page_content"], "chunk_id": i},
                     {"index": {"_index": self.bm25_index_name, "_id": f"doc_{i}"}},
                     {"content": doc["page_content"], "contextualized_content": context, "doc_id": f"doc_{i}",
                      "source": source, "chunk_id": i, "original_index": i}]
        # ส่งเป็นชุด ๆ (ต้นฉบับส่งทีเดียวทั้งหมด ถ้า chunk เยอะ request ใหญ่เกิน)
        for start in range(0, len(bulk), 200):
            response = self.opensearch_client.bulk(body=bulk[start:start + 200])
            if response.get("errors"):
                raise RuntimeError(f"bulk upload ผิดพลาด: {response['items'][:2]}")
        self.opensearch_client.indices.refresh(index=f"{self.vector_index_name},{self.bm25_index_name}")
        print(f"Successfully added {len(docs)} documents to vector index and BM25 index")
        return len(docs)

    def sparse_search(self, query, k=10):
        """ค้นหาด้วย BM25"""
        response = self.opensearch_client.search(index=self.bm25_index_name, body={
            "query": {"multi_match": {"query": query, "fields": ["content", "contextualized_content"], "type": "best_fields"}},
            "size": k})
        return [(hit["_id"], hit["_score"], hit["_source"]) for hit in response["hits"]["hits"]]

    def dense_search(self, query, k=10):
        """ค้นหาด้วย vector similarity"""
        query_embedding = self.encoder.embed_query(query)
        response = self.opensearch_client.search(index=self.vector_index_name, body={
            "query": {"knn": {"embedding": {"vector": query_embedding, "k": k}}}, "size": k})
        return [(hit["_id"], hit["_score"], hit["_source"]) for hit in response["hits"]["hits"]]

    def rrf_fusion(self, rankings, k=60):
        """ผสมผลลัพธ์ด้วย Reciprocal Rank Fusion"""
        fused_scores, doc_sources = {}, {}
        for results in rankings:
            # FIX 7: สูตร RRF ใช้ rank เริ่มที่ 1 (ต้นฉบับเริ่ม 0; วัดแล้วลำดับไม่เปลี่ยน แต่ให้ตรงสูตร)
            for rank, (doc_id, score, source) in enumerate(results, 1):
                fused_scores[doc_id] = fused_scores.get(doc_id, 0) + 1.0 / (k + rank)
                # vector index ไม่มี contextualized_content -> เก็บ source ฝั่ง BM25 ไว้ถ้ามี
                if doc_id not in doc_sources or "contextualized_content" in source:
                    doc_sources[doc_id] = source
        sorted_results = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        return [(doc_id, score, doc_sources[doc_id]) for doc_id, score in sorted_results]

    def hybrid_search(self, query, k=5, rrf_k=60, verbose=True):
        """ค้นหาแบบ hybrid (BM25 + dense) ด้วย RRF"""
        sparse_results = self.sparse_search(query, k=k * 2)
        dense_results = self.dense_search(query, k=k * 2)
        fused_results = self.rrf_fusion([sparse_results, dense_results], k=rrf_k)
        if verbose:
            print(f"\nSparse {len(sparse_results)} / Dense {len(dense_results)} / RRF {len(fused_results)} results")
        return fused_results[:k]

    def truncate_context(self, context, max_tokens=4000):
        """ย่อ context ให้ไม่เกิน max_tokens"""
        # FIX 4: ต้นฉบับประมาณ token = จำนวนคำ (split ช่องว่าง) x 1.3 -> ภาษาไทยไม่เว้นวรรค ประมาณต่ำกว่าจริง 4.3 เท่า
        # -> นับ token จริงด้วย tokenizer ของ bge-m3 (ใกล้เคียง tokenizer ของ LLM หลายภาษามากกว่า) แล้วตัดตาม token
        ids = self.tokenizer.encode(context, add_special_tokens=False)
        if len(ids) <= max_tokens:
            return context
        return self.tokenizer.decode(ids[:max_tokens]) + " ...(truncated)"

    def build_context(self, results):
        """รวม chunk ที่ค้นเจอเป็น context ส่งให้ LLM"""
        context = ""
        for doc_id, score, doc in results:
            context += f"Content: {doc.get('content', '')}\nContext: {doc.get('contextualized_content', '')}\n\n"
        return self.truncate_context(context)

    def search_for_question(self, question, k=5, verbose=True):
        """ค้นหาเอกสารที่เกี่ยวข้องกับคำถามแล้วสร้างคำตอบ"""
        results = self.hybrid_search(question, k=k, verbose=verbose)
        if not results:
            return self.generate_response(question)
        if verbose:
            print(f"\nFound {len(results)} relevant documents for question: '{question}'")
            for i, (doc_id, score, doc) in enumerate(results, 1):
                print(f"  {i}. {doc_id} ({doc.get('source', '')}) score={score:.4f} context={doc.get('contextualized_content', '')[:80]}")
        return self.generate_response(question, self.build_context(results))

    def generate_response(self, question, context="", max_tokens=1000):
        """สร้างคำตอบจากคำถามและ context"""
        if not context:
            prompt = f"""You are a question answering assistant. Answer the question as truthful and helpful as possible.

Question: {question}"""
        else:
            prompt = f"""You are a question answering assistant. Answer the question as truthful and helpful as possible.

Context information:
{context}

Question: {question}"""
        return self.call_qwen_api(prompt, max_tokens=max_tokens, temperature=0.6)


MD_PATHS = [ROOT / "corpus_input" / f for f in ("1.md", "2.md", "44.md", "5555.md")]


def main(argv):
    rag = AnthropicStyleContextualRAG()
    if "--search" not in argv:
        docs = rag.load_documents(MD_PATHS)
        rag.add_documents_with_context(docs)
    for question in ("ผมจะรู้ได้อย่างไรว่าผมเป็นโรคหัดแบบไหน", "โรคต้อกระจก หรือโรคอะไรที่เกี่ยวข้องกันบ้าง"):
        print(f"\nQuestion: {question}")
        print(f"\nFinal Answer: {rag.search_for_question(question)}")


if __name__ == "__main__":
    main(sys.argv[1:])
