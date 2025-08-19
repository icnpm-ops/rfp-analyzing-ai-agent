# backend/upload_guide.py

import json
import os
import pickle
import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from datetime import datetime
from config import EMBED_MODEL

# 🔧 경로 설정
GUIDE_DIR = os.path.join(os.path.dirname(__file__), "guide")
os.makedirs(GUIDE_DIR, exist_ok=True)
INDEX_PATH = os.path.join(GUIDE_DIR, "guide.index")
TEXTS_PATH = os.path.join(GUIDE_DIR, "guide_texts.pkl")
METADATA_PATH = os.path.join(GUIDE_DIR, "guide_metadata.json")

# 🔐 API 키 설정
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ✅ 문장 임베딩
def get_embedding(text: str) -> list:
    res = client.embeddings.create(input=[text], model=EMBED_MODEL)
    return res.data[0].embedding

# ✅ 인덱스 구축 및 저장
def build_guide_index(texts: list[str]):
    index = faiss.IndexFlatL2(1536)
    matrix = np.array([get_embedding(t) for t in texts]).astype("float32")
    index.add(matrix)
    faiss.write_index(index, INDEX_PATH)
    with open(TEXTS_PATH, "wb") as f:
        pickle.dump(texts, f)
    print(f"✅ Guide 인덱스 저장 완료: {INDEX_PATH}")

# ✅ 메타데이터 저장
def save_guide_metadata(title: str, source: str):
    upload_time = datetime.utcnow().isoformat()
    entry = {
        "title": title,
        "source": source,
        "docType": "Guide",
        "uploadAt": upload_time
    }

    # 누적 저장
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = []

    data.append(entry)

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"✅ Guide 메타데이터 저장 완료: {entry}")

# ✅ 테스트 실행 (직접 실행 시)
if __name__ == "__main__":
    sample_texts = [
        "제안서에는 구체적인 기술 구현 계획이 포함되어야 한다.",
        "문제 정의는 도입부에 명확하게 기술되어야 한다.",
        "성과 목표는 정량적 수치와 기한으로 구체화해야 한다."
    ]

    build_guide_index(sample_texts)
    save_guide_metadata(
        title="내장 제안서 작성 가이드",
        source="internal-guide"
    )
