"""ของที่ทุก step ใช้ร่วมกัน: อ่าน .env, เรียก LLM, ต่อ OpenSearch, โหลด embedding model

ตั้งค่าในไฟล์ .env ที่โฟลเดอร์ rag-workshop (ไม่ต้องติดตั้ง python-dotenv):
    OPENROUTER_API_KEY=sk-or-...                 # ตามสไลด์ (Qwen ผ่าน OpenRouter)
  หรือใช้ LLM เจ้าอื่นที่รองรับ OpenAI SDK:
    LLM_API_KEY=...
    LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
    LLM_MODEL=gemini-2.5-flash
"""
import os
import re
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "corpus_input"

_env = ROOT / ".env"
if _env.exists():
    for line in _env.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

LLM_MODEL = os.environ.get("LLM_MODEL", "qwen/qwen-2.5-72b-instruct")
_client = None


def llm(prompt, max_tokens=4000, temperature=0.2, system="You are a helpful assistant."):
    """ส่ง prompt ไปที่ LLM แล้วคืนข้อความคำตอบ (ใช้ OpenAI SDK เพราะเป็น 'มาตรฐาน' ที่เจ้าอื่นเลียนแบบ)"""
    global _client
    if _client is None:
        from openai import OpenAI
        key = os.environ.get("LLM_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise SystemExit("ยังไม่ได้ตั้ง OPENROUTER_API_KEY (หรือ LLM_API_KEY) ในไฟล์ .env")
        _client = OpenAI(api_key=key, base_url=os.environ.get("LLM_BASE_URL", "https://openrouter.ai/api/v1"))
    from openai import APIConnectionError, InternalServerError, RateLimitError
    for attempt in range(8):
        try:
            r = _client.chat.completions.create(
                model=LLM_MODEL, max_tokens=max_tokens, temperature=temperature,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}])
            break
        except (RateLimitError, InternalServerError, APIConnectionError) as e:
            # ponytail: free tier ชน limit ต่อนาที หรือ server ล่มชั่วคราว -> รอแล้วลองใหม่; ถ้าบอกให้รอนาน (เช่นโควตารายวันหมด) ก็หยุดเลย
            m = re.search(r"retry in ([\d.]+)s", str(e))
            wait = float(m.group(1)) + 2 if m else 30
            if wait > 300 or attempt == 7:
                raise
            print(f"  ({type(e).__name__}, รอ {wait:.0f} วินาที)")
            time.sleep(wait)
    text = r.choices[0].message.content or ""
    return re.sub(r"(?s)<thought>.*?(</thought>|$)", "", text).strip()  # บางโมเดลพ่นส่วน "คิด" มาด้วย


def opensearch():
    from opensearchpy import OpenSearch
    return OpenSearch(hosts=[{"host": "localhost", "port": 9200}], use_ssl=False, timeout=60)


_embedder = None


def embedder():
    """BAAI/bge-m3: embedding model หลายภาษา (รวมไทย) ได้ vector 1024 มิติ — โหลดครั้งแรก ~2 GB"""
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer("BAAI/bge-m3")
    return _embedder


def notebook_setup():
    """เรียกใน cell แรกของ notebook: ปิดข้อความรก (progress bar, warning) และเลือก font ไทยที่เครื่องมีจริงให้กราฟ"""
    import logging
    import warnings
    os.environ["TQDM_DISABLE"] = "1"                    # แถบ "Loading weights" / "Batches"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    warnings.filterwarnings("ignore")
    for name in ("huggingface_hub", "transformers", "sentence_transformers", "matplotlib.font_manager"):
        logging.getLogger(name).setLevel(logging.ERROR)
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    have = {f.name for f in font_manager.fontManager.ttflist}
    thai = [f for f in ("Tahoma", "Leelawadee UI", "Thonburi", "Noto Sans Thai", "Loma", "Garuda") if f in have]
    plt.rcParams["font.family"] = thai + ["DejaVu Sans"]   # ไม่มี font ไทยเลย = ตัวหนังสือในกราฟเป็นสี่เหลี่ยม แต่โค้ดยังทำงาน


def read_corpus():
    """คืน {ชื่อไฟล์: ข้อความทั้งไฟล์} ของเอกสาร 4 ไฟล์"""
    return {p.name: p.read_text(encoding="utf-8") for p in sorted(CORPUS.glob("*.md"))}
