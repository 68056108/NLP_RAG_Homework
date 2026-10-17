# authenticRAG ฉบับแก้ไข: แก้อะไร ทำไม และได้ผลอย่างไร

งาน KDAI NLP Assignment: ระบบ RAG ภาษาไทยบนเอกสารโรค 4 ไฟล์ ([`corpus_input/`](corpus_input))

| โจทย์ | อยู่ที่ไหนใน repo |
|---|---|
| **(1) ระบบ RAG ตามสไลด์ของอาจารย์** (สไลด์หน้า 22, 25) | [`original/authenticRAG.py`](original/authenticRAG.py) (สร้าง index + ถามตอบ) และ [`original/onlysearchAuthenticRAG.py`](original/onlysearchAuthenticRAG.py) (ถามตอบอย่างเดียว) คัดลอกจาก [aekanun2020/2025-authenticRAG](https://github.com/aekanun2020/2025-authenticRAG) โดยไม่แก้ |
| **(2) ปรับปรุงโดยไม่เปลี่ยน LLM** | [`authenticRAG_fixed.py`](authenticRAG_fixed.py) แก้ chunking (FIX 2), search/scoring ของ BM25 ด้วย analyzer ภาษาไทย (FIX 1), ข้อความที่ใช้ทำ embedding และ BM25 ผ่าน context จากเอกสารเต็ม (FIX 3) และการตัด context ก่อนส่ง LLM (FIX 4) ทุกจุดมีป้าย `FIX n` ในโค้ด อธิบายไว้ใน [ข้อ 2](#2-สิ่งที่แก้) ไม่ได้เพิ่ม reranker และใช้ embedding model เดิม (bge-m3) |
| **(3) ต่างกันตรงไหน ดีขึ้นอย่างไร เพราะอะไร** | [ข้อ 2](#2-สิ่งที่แก้) (ปัญหา, ตัวเลขที่วัดได้, วิธีแก้) และ [ข้อ 3](#3-ผลลัพธ์-ต้นฉบับ-vs-ฉบับแก้ไข) (ผลเทียบกัน) วัดด้วย [`evaluate.py`](evaluate.py) และ [`steps/diagnose_*.py`](steps) |

ฐานโค้ด: [`original/authenticRAG.py`](original/authenticRAG.py) ของอาจารย์ (commit [`4567d3d`](https://github.com/aekanun2020/2025-authenticRAG/tree/4567d3d5d377f6e38ac18afd62133476425f98bf))
ฉบับแก้ไข: [`authenticRAG_fixed.py`](authenticRAG_fixed.py) ใช้ชื่อ class และ method เหมือนต้นฉบับ ทุกจุดที่แก้มีป้าย `FIX n` ในโค้ด

### โครงสร้าง repo

```
original/                       โค้ดต้นฉบับของอาจารย์ (โจทย์ข้อ 1)
authenticRAG_fixed.py           ฉบับแก้ไข (โจทย์ข้อ 2)
evaluate.py                     วัดผล ต้นฉบับ vs ฉบับแก้ไข -> results.json, answers.json
corpus_input/                   เอกสาร 4 ไฟล์ (1.md หัดเยอรมัน, 2.md อหิวาตกโรค, 44.md ต้อกระจก, 5555.md กรดไหลย้อน)
steps/questions.json            ชุดคำถาม 29 ข้อ
steps/diagnose_*.py             สคริปต์ตรวจทีละขั้น (chunking, analyzer, context) -> results/diagnosis*.md
steps/common.py                 ฟังก์ชันที่สคริปต์ใน steps/ ใช้ร่วมกัน
results/                        ผลการตรวจทีละขั้น
contexts_cache.json             context ที่ LLM สร้างให้ฉบับแก้ไข (รันซ้ำไม่ต้องเรียก LLM ใหม่)
```

### ติดตั้งและรัน

ทดสอบกับ Python 3.11 และ OpenSearch 2.19.1

**1. ติดตั้ง library**

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

embedding model `BAAI/bge-m3` (~2 GB) จะโหลดอัตโนมัติตอนรันครั้งแรก

**2. เปิด OpenSearch** (ต้องมี Docker)

```bash
docker run -d --name opensearch -p 127.0.0.1:9200:9200 -e discovery.type=single-node -e DISABLE_SECURITY_PLUGIN=true -e DISABLE_INSTALL_DEMO_CONFIG=true -e "OPENSEARCH_JAVA_OPTS=-Xms1g -Xmx1g" opensearchproject/opensearch:2.19.1
```

**3. ตั้งค่า LLM** สร้างไฟล์ `.env` ไว้ที่โฟลเดอร์บนสุดของ repo (ที่เดียวกับ `README.md`) ไฟล์นี้อยู่ใน `.gitignore` แล้ว จึงไม่ถูก commit

ใช้ Qwen ผ่าน OpenRouter (ค่าเดิมของอาจารย์):
```
OPENROUTER_API_KEY=ใส่ key จาก https://openrouter.ai/keys
```

หรือใช้ LLM เจ้าอื่นที่รองรับ OpenAI SDK เช่น Gemini (ผลในข้อ 3 รันด้วยค่านี้):
```
LLM_API_KEY=ใส่ key จาก https://aistudio.google.com/apikey
LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
LLM_MODEL=gemini-3.5-flash-lite
```

ถ้ามีทั้งสองชุด โค้ดจะใช้ `LLM_API_KEY` ก่อน ส่วน `LLM_MODEL` ถ้าไม่ได้ตั้งไว้จะใช้ `qwen/qwen-2.5-72b-instruct`

**4. รัน**

```bash
python authenticRAG_fixed.py              # index ใหม่ + ถามตัวอย่าง 2 ข้อ
python authenticRAG_fixed.py --search     # ถามอย่างเดียว
python evaluate.py                        # วัดผล ต้นฉบับ vs ฉบับแก้ไข
python steps/diagnose_authenticRAG.py     # ตรวจต้นฉบับทีละขั้น (ไม่เรียก LLM)
python steps/diagnose_context.py thai     # วัดผลของ context (ใช้ context ที่เก็บไว้ใน results/)
python steps/diagnose_chunking.py         # เทียบขนาด chunk (ไม่เรียก LLM)
```

ต้นฉบับของอาจารย์รันจากโฟลเดอร์บนสุดของ repo ด้วย `python original/authenticRAG.py` (โค้ดอ่านเอกสารจาก `./corpus_input/` และต้องตั้ง `OPENROUTER_API_KEY` เป็น environment variable เพราะต้นฉบับไม่อ่านไฟล์ `.env`)

---

## 1. ตัวชี้วัด

ชุดทดสอบ: [`steps/questions.json`](steps/questions.json) มี 29 คำถาม
ที่มา: คัดลอกจากชุดคำถามตัวอย่าง `steps/questions.json` ของเอกสารเวิร์กช็อป [kdai-rag-workshop](https://github.com/Patchanon04/kdai-rag-workshop) โดย 5 ข้อแบบ `official` (q25–q29) มาจากคำถามตัวอย่างของอาจารย์ใน `onlysearchAuthenticRAG.py`
- 28 ข้อมี **evidence** คือข้อความที่คัดลอกตรงจากเอกสาร ถ้า chunk ไหนมีข้อความนี้อยู่ ถือว่า chunk นั้น "ถูก"
  วิธีนี้ไม่ผูกกับวิธีตัด chunk จึงเทียบระบบที่ตัด chunk ต่างกันได้
- คำถามแบ่งเป็น 4 แบบ: `keyword` (ใช้คำตรงกับเอกสาร), `specific` (ถามตัวเลขหรือชื่อเฉพาะ), `paraphrase` (ถามด้วยคำของชาวบ้าน ไม่ตรงกับคำในเอกสาร), `official` (คำถามตัวอย่างของอาจารย์)
- 1 ข้อ (q29) ไม่มีคำตอบในเอกสาร ใช้ทดสอบว่าระบบแต่งคำตอบขึ้นมาเองหรือไม่

| ตัวชี้วัด | ความหมาย | ค่าที่ดี |
|---|---|---|
| **Hit@1** | สัดส่วนคำถามที่ chunk อันดับ 1 เป็น chunk ที่ถูก | สูง |
| **Hit@5** | สัดส่วนคำถามที่มี chunk ถูกอยู่ใน 5 อันดับแรก (คือข้อมูลที่ LLM ได้เห็น) | สูง |
| **MRR@5** | ค่าเฉลี่ยของ 1/อันดับที่เจอ chunk ถูกครั้งแรก (อันดับ 1 = 1, อันดับ 2 = 0.5, ... ไม่เจอ = 0) | สูง |
| **context tokens** | จำนวน token ที่ส่งให้ LLM ต่อคำถาม (นับด้วย tokenizer ของ bge-m3) | ต่ำ ถ้า Hit เท่ากัน |
| **answer score** | LLM ให้คะแนนคำตอบเทียบกับคำตอบอ้างอิง: 1 = ครบ, 0.5 = บางส่วน, 0 = ผิดหรือไม่ตอบ | สูง |
| **out-of-corpus** | q29: 1 = ยอมรับว่าไม่มีข้อมูล, 0 = แต่งคำตอบขึ้นมา | 1 |

ตัวชี้วัดรายขั้น (ดู [`steps/diagnose_*.py`](steps)):
- **chunk ที่เกินขนาด**: สัดส่วน chunk ที่ยาวเกิน `chunk_size` ที่ตั้งไว้
- **คำในคำถามที่พบใน corpus**: สัดส่วน token ของคำถามที่ analyzer ตัดออกมาแล้วตรงกับ token ใน index ถ้าต่ำ แปลว่า BM25 ไม่มีทางจับคู่ได้
- **ส่วนของเอกสารที่ LLM เห็น**: ความยาวของ `<document>` ที่ส่งไปตอนสร้าง context หารด้วยความยาวไฟล์เต็ม

---

## 2. สิ่งที่แก้

เรียงจากผลกระทบมากไปน้อย ตัวเลข "วัดได้" มาจากการตรวจทีละขั้น ([`results/diagnosis*.md`](results))

| # | ปัญหาในต้นฉบับ | วัดได้ | วิธีแก้ |
|---|---|---|---|
| **FIX 1** | BM25 ใช้ analyzer `standard` ซึ่งตัดคำไทยไม่ได้ ทั้งประโยคเป็น token เดียว เช่น `ระยะฟักตัวของโรคหัดเยอรมันนานกี่วัน` ได้ 1 token | คำในคำถามที่พบใน corpus **5.7%**, BM25 Hit@5 **0.07**, ค้นแล้วไม่เจออะไรเลย 23/28 ข้อ | ใช้ analyzer `thai` ของ OpenSearch (ตัดได้ `ระยะ / ฟักตัว / โรค / หัดเยอรมัน / นาน / กี่`) คำที่พบเพิ่มเป็น 91.3% และ BM25 Hit@5 เป็น 1.00 |
| **FIX 2** | `MarkdownNodeParser(chunk_size=256)` ไม่ได้ใช้ `chunk_size` ตัดตามหัวข้ออย่างเดียว | **66%** ของ chunk ยาวเกิน 256 token, ใหญ่สุด 2,431 token | ตัดตามหัวข้อก่อน แล้วตัด section ที่ยาวเกินด้วย `SentenceSplitter` นับ token ด้วย tokenizer ของ bge-m3 ใช้ขนาด **1024** (ทดลอง 128/256/512/1024 แล้ว 1024 ได้ผลดีที่สุด ดู [`results/diagnosis_chunking.md`](results/diagnosis_chunking.md)) |
| **FIX 3** | ตอนสร้าง context ใช้ `full_doc_content = doc.page_content` จึงส่ง chunk ตัวเองไปแทนเอกสารเต็ม | **100%** ของ chunk, LLM เห็นเฉลี่ยแค่ **7.5%** ของไฟล์ | เก็บเอกสารเต็มไว้ตอนโหลด แล้วส่งไฟล์เต็มเป็น `<document>` ตามที่ Anthropic ออกแบบไว้ |
| **FIX 4** | `truncate_context` ประมาณ token จากจำนวนคำ (แยกด้วยช่องว่าง) × 1.3 ซึ่งใช้กับภาษาไทยที่ไม่เว้นวรรคไม่ได้ | token จริงมากกว่าที่ประมาณ **4.3 เท่า**, ส่ง context เกิน 4,000 token ให้ LLM **6/28 ข้อ** | นับ token จริงด้วย tokenizer แล้วตัดตาม token ได้ผลเหลือเกิน 1/28 ข้อ (4,006 token เพราะต่อท้าย `...(truncated)`) |
| **FIX 5** | เมื่อเรียก API ไม่สำเร็จ `call_qwen_api` คืนข้อความ `"Qwen API error: ..."` แล้วข้อความนี้ถูกเก็บลง index เป็น context | free tier ชน rate limit บ่อย (เห็นหลายสิบครั้งระหว่างรันวัดผล) | ลองใหม่เมื่อชน rate limit หรือ server ล่มชั่วคราว ถ้ายังไม่ได้ให้ error ออกมาเลย ไม่เก็บข้อความ error ลง index |
| **FIX 6** | สร้าง index เฉพาะตอน "ยังไม่มี" ถ้ามี index เก่าอยู่แล้ว ค่าตั้งใหม่ (เช่น analyzer) จะไม่ถูกใช้ และ chunk เก่าที่ id เกินจำนวนใหม่ค้างอยู่ | ถ้าไม่แก้ข้อนี้ FIX 1 จะไม่มีผลกับเครื่องที่เคยรันต้นฉบับมาแล้ว | ลบแล้วสร้าง index ใหม่ทุกครั้งที่ index และใช้ชื่อใหม่ `authentic-fixed-*` ไม่ทับ index ของอาจารย์ |
| **FIX 7** | knn ตั้ง `space_type` ผ่าน setting ระดับ index แบบเก่า และ RRF นับอันดับเริ่มที่ 0 | ไม่มีผลกับลำดับ (knn recall@10 เทียบ exact = 1.00, RRF top-5 ต่าง 0/28) | ระบุ `method.space_type` ใน mapping และนับอันดับเริ่มที่ 1 ตามสูตร แก้ให้ถูกต้องเฉยๆ ไม่ได้เปลี่ยนผล |
| **FIX 8** | เรียก LLM ใหม่ทุก chunk ทุกครั้งที่ index | 53 ครั้งต่อรอบ | cache context ไว้ใน `contexts_cache.json` |
| **FIX 9** | ผูกกับ `OPENROUTER_API_KEY` อย่างเดียว และ `max_tokens=100` ทำให้โมเดลที่มีส่วน "คิด" ตอบออกมาว่าง | - | อ่าน `.env` รองรับ `LLM_API_KEY / LLM_BASE_URL / LLM_MODEL` และใช้ `max_tokens=1000` |

ส่วนที่**ไม่ได้แก้** เพื่อให้ยังเทียบกับต้นฉบับได้: prompt สร้าง context, prompt ตอบคำถาม, embedding model (bge-m3), ค้นแบบ hybrid ฝั่งละ 10 แล้วรวมด้วย RRF เหลือ 5 อันดับ, `temperature` ทุกจุด
ตัด TextLoader และ LangChain ออก เพราะอ่านไฟล์ตรงๆ ได้ข้อความเท่ากัน (วัดแล้ว ตัวอักษรหาย 0) และไม่ต้องพึ่ง LangChain รุ่นเก่า

---

## 3. ผลลัพธ์: ต้นฉบับ vs ฉบับแก้ไข

รันด้วย [`evaluate.py`](evaluate.py) เมื่อ 14 ต.ค. 2026
- LLM: `gemini-3.5-flash-lite` ตัวเดียวกันทั้งสองระบบ ใช้ทั้งสร้าง context, ตอบคำถาม และให้คะแนน
- Embedding: `BAAI/bge-m3`
- ต้นฉบับรันด้วย class ของอาจารย์จริง เปลี่ยนแค่ชื่อ index และให้เรียก LLM ตัวเดียวกัน
- คำตอบทุกข้ออยู่ใน [`answers.json`](answers.json) ตัวเลขทั้งหมดอยู่ใน [`results.json`](results.json)

| ตัวชี้วัด | ต้นฉบับ | ฉบับแก้ไข |
|---|---|---|
| **Hit@1** | 0.68 | **0.86** |
| **Hit@5** | 0.89 | **1.00** |
| **MRR@5** | 0.75 | **0.92** |
| **answer score** (28 ข้อ) | 0.93 | **1.00** |
| answer score เฉพาะข้อ paraphrase (8 ข้อ) | 0.75 | **1.00** |
| out-of-corpus (q29) | 1 | 1 |
| context ที่ส่ง LLM (token เฉลี่ย / สูงสุด) | 3,178 / 4,661 | 3,037 / 4,006 |
| ส่ง context เกิน 4,000 token | 6/28 | 1/28 |
| จำนวน chunk / ใหญ่สุด (token) | 53 / 2,431 | 58 / 985 |
| chunk ที่เกินขนาดที่ตั้งไว้ | 35 (66%) | 0 |

### คำถามที่ผลเปลี่ยน

| ข้อ | แบบ | อันดับ chunk ที่ถูก | คะแนนคำตอบ |
|---|---|---|---|
| q03 แม่เป็นไข้ออกผื่นตอนท้อง ลูกหูไม่ได้ยิน | paraphrase | ไม่อยู่ใน 5 อันดับ → **1** | 0.5 → **1** |
| q10 ถ่ายเป็นน้ำขาวขุ่นเหมือนน้ำล้างข้าว | paraphrase | ไม่อยู่ใน 5 อันดับ → **1** | 0.5 → **1** |
| q21 ลูกแหวะนมบ่อย น้ำหนักไม่ขึ้น | paraphrase | ไม่อยู่ใน 5 อันดับ → **1** | 0 → **1** |
| q05, q08, q14, q20 | keyword / specific / paraphrase | 2–4 → **1** | 1 → 1 |
| q09, q28 | paraphrase / official | 5 → 3 | 1 → 1 |
| q06, q25 | paraphrase / official | 1 → 2 (แย่ลง) | 1 → 1 |

ทุกข้อที่คำตอบดีขึ้นเป็นแบบ **paraphrase** คือคำถามที่ใช้คำต่างจากเอกสาร ในต้นฉบับ chunk ที่ถูกหลุดจาก 5 อันดับแรก LLM จึงไม่ได้เห็นข้อมูล ในฉบับแก้ไข chunk ที่ถูกขึ้นมาเป็นอันดับ 1 ทั้งสามข้อ

### ข้อควรระวังในการอ่านผล
- **ชุดทดสอบเล็ก:** มี 28 ข้อ ต่างกัน 1 ข้อเท่ากับค่าเปลี่ยน 0.036 ผลชัดในทิศทาง (ดีขึ้น 9 ข้อ แย่ลง 2 ข้อ แต่อันดับยังอยู่ที่ 2) แต่ขนาดของผลยังมีความไม่แน่นอน
- **ใช้ LLM ตัวเดียวกันตอบและให้คะแนน:** คะแนนคำตอบควรใช้ประกอบกับตัวชี้วัด retrieval ซึ่งตรวจด้วยการเทียบข้อความตรงๆ ไม่ได้ใช้ LLM ตัดสิน
- **คำตอบสุ่มได้:** ตอบที่ `temperature=0.6` ตามต้นฉบับ รันซ้ำแล้วคะแนนคำตอบอาจขยับเล็กน้อย
- **ไม่ได้ใช้โมเดลของอาจารย์:** ต้นฉบับออกแบบไว้กับ Qwen 2.5 72B ผ่าน OpenRouter แต่รอบนี้ใช้ Gemini เพราะ key ที่มีเป็นของ Gemini ส่วน `gemma-4-26b` บน free tier รับได้ 16,000 token ต่อนาที ซึ่งน้อยกว่าขนาดไฟล์เต็ม 1 ไฟล์ จึงใช้ทำ FIX 3 ไม่ได้
- **จำนวน token ที่ส่งให้ LLM แทบไม่ลด:** chunk ขนาด 1024 ยังใหญ่ ถ้าต้องการประหยัด ขนาด 128 ส่งข้อความน้อยกว่าเดิมประมาณ 6 เท่า แลกกับ Hit@5 ที่ลดลงเหลือ 0.93 (ดู [`results/diagnosis_chunking.md`](results/diagnosis_chunking.md))

### แต่ละ fix ช่วยได้เท่าไร (จากการวัดแยกทีละขั้น, hybrid Hit@5 / MRR@5)

| สถานะ | Hit@5 | MRR@5 | ที่มา |
|---|---|---|---|
| ต้นฉบับ (analyzer standard + context จาก chunk) | 0.89 | 0.75 | [`results/diagnosis_context.md`](results/diagnosis_context.md) |
| + FIX 3 (context จากไฟล์เต็ม) | 0.93 | 0.77 | [`results/diagnosis_context.md`](results/diagnosis_context.md) |
| + FIX 1 (analyzer thai) | 1.00 | 0.86 | [`results/diagnosis_context_thai.md`](results/diagnosis_context_thai.md) |
| + FIX 2 (chunk 1024) = ฉบับแก้ไข | 1.00 | 0.92 | [`evaluate.py`](evaluate.py) |

FIX 1 (analyzer) ช่วยมากที่สุด เมื่อแก้ analyzer แล้ว FIX 3 (context) ไม่มีผลกับ BM25 เลย เพราะคำค้นไปตรงกับเนื้อหาของ chunk เองอยู่แล้ว ส่วน FIX 2 ทำให้ chunk ที่ถูกขึ้นมาอยู่อันดับ 1 บ่อยขึ้น
