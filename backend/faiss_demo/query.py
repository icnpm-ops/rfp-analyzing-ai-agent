import os
import faiss
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

# 목적
# 사용자 질문을 임베딩 후 FAISS 인덱스에서 검색
# 가장 유사한 문장 상위 3개 출력

# 출력 
# cd backend
# python faiss_demo/query.py

# Embedding : 문장의 의미를 숫자 벡터로 바꿔 유사도 비교가 가능해짐
# FAISS : 벡터 간 유사도 검색이 매우 빠르고 메모리 효율적
# 추후 제안서 문장과 RFP 문장을 의미 기반으로 매칭하기 위해 먼저 테스트
# 언어 수준의 의미 벡터를 만들기 위한 고성능 Embedding API

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_embedding(text: str) -> list:
    res = client.embeddings.create(
        input=[text],
        model="text-embedding-3-small"
    )
    return res.data[0].embedding

def search_index(index_path: str, query_text: str, texts_path: str, top_k=3):
    index = faiss.read_index(index_path)
    emb = get_embedding(query_text)
    xq = np.array([emb]).astype("float32")

    D, I = index.search(xq, top_k)

    with open(texts_path, "r", encoding="utf-8") as f:
        all_texts = [line.strip() for line in f if line.strip()]

    print(f"\n Query : {query_text}\n")
    for i, idx in enumerate(I[0]):
        print(f"[Top {i+1}] {all_texts[idx]} (거리: {D[0][i]:.4f})")

if __name__ == "__main__":
    search_index(
        index_path="faiss_demo/rfp.index",
        query_text="AI 분석 도구 개발 제안",
        texts_path="faiss_demo/data.txt"
    )