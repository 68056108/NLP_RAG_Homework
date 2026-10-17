# ผลตรวจ authenticRAG ทีละขั้น (28 คำถาม, top-5)

## 1) โหลดเอกสาร (TextLoader)
- ไฟล์ 4 ไฟล์, ตัวอักษรที่หายระหว่างโหลด: 0

## 2) Chunking — MarkdownNodeParser(chunk_size=256)
- 53 chunks; token (tokenizer ของ bge-m3) min=31 median=330 p90=840 max=2431
- chunk ที่เกิน 256 token (ค่าที่ตั้งไว้): 35/53 = 66.0%  -> MarkdownNodeParser ตัดตาม heading เท่านั้น ไม่สน chunk_size
- chunk เล็กมาก (< 50 ตัวอักษร เช่นมีแต่หัวข้อ): 0 = 0.0%
- เนื้อหาที่เกิน max_seq_length=8192 ของ embedding (ถูกตัดทิ้งตอน embed): 0.0%
- evidence ที่ถูกตัดขาดข้าม chunk: 0/49

## 3) Contextual retrieval — เอกสารที่ส่งให้ LLM สร้าง context
- chunk ที่ <document> เป็นข้อความเดียวกับ <chunk>: 53/53 = 100.0%
- LLM เห็นเอกสารต้นฉบับเฉลี่ยเพียง 7.5% ของไฟล์ (ควรเป็น 100%)
- เรียก LLM 53 ครั้ง/การ index หนึ่งรอบ (ไม่มี cache)

## 4) Analyzer ของ BM25 index — standard vs thai
- standard: token ทั้งหมด 4,381, ความยาว token ไทยเฉลี่ย 22.9 ตัวอักษร, token ไทยยาว > 20 ตัวอักษร 44.1%, คำในคำถามที่พบใน corpus 5.7%
- thai    : token ทั้งหมด 12,227, ความยาว token ไทยเฉลี่ย 4.5 ตัวอักษร, token ไทยยาว > 20 ตัวอักษร 0.0%, คำในคำถามที่พบใน corpus 91.3%
- ตัวอย่าง 'ระยะฟักตัวของโรคหัดเยอรมันนานกี่วัน'
  - standard -> ['ระยะฟักตัวของโรคหัดเยอรมันนานกี่วัน']
  - thai     -> ['ระยะ', 'ฟักตัว', 'โรค', 'หัดเยอรมัน', 'นาน', 'กี่']

## 5) Retrieval (ใช้ settings ของครู, ไม่มี context จาก LLM)
- knn mapping ที่ได้จริง: {"type": "knn_vector", "dimension": 1024}  (setting index.knn.space_type เป็นแบบเก่า)

| ระบบ | Hit@1 | Hit@5 | MRR@5 | คำถามที่ไม่ได้ผลเลย | ตัวอักษรที่ส่ง LLM (top-5) |
|---|---|---|---|---|---|
| BM25 (standard) | 0.00 | 0.07 | 0.04 | 23/28 | 1,982 |
| dense (knn) | 0.71 | 0.96 | 0.81 | 0/28 | 10,661 |
| dense (exact cosine) | 0.71 | 0.96 | 0.81 | 0/28 | 10,661 |
| hybrid RRF (ครู) | 0.68 | 0.96 | 0.80 | 0/28 | 10,949 |

- knn เทียบ exact cosine: recall@10 = 1.00
- RRF rank เริ่ม 0 vs 1: top-5 ต่างกัน 0/28 คำถาม

## 6) truncate_context — ประมาณ token ด้วย len(text.split()) * 1.3
- token จริง (tokenizer bge-m3) / ค่าประมาณ = 4.3 เท่า (ประมาณต่ำกว่าจริง)
- context ที่เกิน 4,000 token จริงแต่ไม่ถูกตัด: 4/28; token จริง median=2,958 max=4,661
