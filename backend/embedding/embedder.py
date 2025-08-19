import os
import pickle
import numpy as np
import faiss
from dotenv import load_dotenv
from openai import OpenAI
from config import INDEX_PATH, TEXTS_PATH, EMBED_MODEL
from typing import List, Dict, Union

# 환경변수로 API 키 불러오고 상수값 정의(임베딩 모델, 저장 경로)

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

EMBED_DIM = 1536 # text-embedding-3-small 기준

# 단일 문장을 벡터화해 결과를 1536차원 리스트로
def get_embedding(text: str) -> List[float]:
    res = client.embeddings.create(
        input = [text],
        model=EMBED_MODEL
    )
    return res.data[0].embedding

# 벡터화된 문장들을 FAISS 인덱스에 추가하고 원문 텍스트는 .pkl로 따로 저장(검색 결과 매핑용)
# FAISS 기본은 IndexFlatL2(L2 거리 기반) - GPU도 지원하긴 함
# .pkl로 매핑 저장하는 이유는 "n번째 벡터가 어떤 문장이었는지" 확인 가능하게 하기 위함

def _load_text_store() -> List[Union[str, Dict]]:
    if not os.path.exists(TEXTS_PATH):
        return []
    with open(TEXTS_PATH, "rb") as f:
        return pickle.load(f)
    
def _save_text_store(items: List[Union[str, Dict]]):
    with open(TEXTS_PATH, "wb") as f:
        pickle.dump(items, f)

def _ensure_index() -> faiss.IndexFlatL2:
    if os.path.exists(INDEX_PATH):
        return faiss.read_index(INDEX_PATH)
    index = faiss.IndexFlatL2(EMBED_DIM)
    faiss.write_index(index, INDEX_PATH)
    return index

def _migrate_items(items: List[Union[str, Dict]]) -> List[Dict]:
    """ 기존 str 포맷을 dict 포맷으로 """
    migrated = []
    for it in items:
        if isinstance(it, str):
            migrated.append({"text": it, "docId": "legacy"})
        else:
            migrated.append(it)
    return migrated

def append_texts(chunks: List[str], doc_id: str):
    """ 새 청크들을 임베딩하여 기존 인덱스/텍스트 스토어에 append"""
    # 1) 임베딩
    vectors = [get_embedding(t) for t in chunks]
    matrix = np.array(vectors, dtype="float32")

    # 2) 인덱스 로드/추가/저장
    index = _ensure_index()
    index.add(matrix)
    faiss.write_index(index, INDEX_PATH)

    # 3) 텍스트 저장소 업데이트
    items = _load_text_store()
    items = _migrate_items(items)
    items.extend({"text": t, "docId": doc_id} for t in chunks)
    _save_text_store(items)

def build_index(texts: list[str]):
    index = faiss.IndexFlatL2(EMBED_DIM)
    vectors = []

    for i, text in enumerate(texts):
        print(f"[{i+1}/{len(texts)}] Embedding 중: {text[:30]}...")
        emb = get_embedding(text)
        vectors.append(emb)

    matrix = np.array(vectors).astype("float32")
    index.add(matrix)

    # 인덱스 저장 전에 디렉토리 확인
    os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
    faiss.write_index(index, INDEX_PATH)
    with open(TEXTS_PATH, "wb") as f:
        pickle.dump(texts, f)

    print("저장 완료:")
    print(f"- FAISS index: {INDEX_PATH}")
    print(f"- 원문 목록: {TEXTS_PATH}")

if __name__ == "__main__":
    sample_texts = [
        "본 제안서는 인공지능 기반 분석 도구 개발을 목표로 합니다.",
        "RFP는 연구개발 과제를 정의하는 문서입니다.",
        "OCR과 VLM 기술을 접목해 복합 문서 분석이 가능합니다.",
        "프로젝트는 2025년 8월까지 베타 버전 완성을 목표로 합니다.",
        "도표나 그래프도 정량적으로 해석할 수 있도록 설계됩니다."
    ]
    build_index(sample_texts)