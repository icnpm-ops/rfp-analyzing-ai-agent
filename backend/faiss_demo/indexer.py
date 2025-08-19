import os
import faiss
from openai import OpenAI
import numpy as np
from dotenv import load_dotenv
load_dotenv()

# 목적
# data.txt의 문장들을 OpenAI API로 임베딩
# FAISS 인덱스로 저장(rfp.index)
# 다음 단계에서 검색용으로 사용

# 실행
# cd backend
# python faiss_demo/indexer.py
# -> faiss_demo/rfp.index 생성됨

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_embedding(text: str) -> list:
    res = client.embeddings.create(
        input=[text],
        model="text-embedding-3-small" # 최신 모델
    )
    return res.data[0].embedding

def build_index(texts: list, save_path:str):
    dim = 1536 # embedding dimension for text-embedding-3-small
    index = faiss.IndexFlatL2(dim)

    vectors = [get_embedding(t) for t in texts]
    vectors_np = np.array(vectors).astype('float32')
    index.add(vectors_np)

    faiss.write_index(index, save_path)
    print(f"Saved FAISS index to {save_path}")

    return index, vectors_np

if __name__=="__main__":
    with open("faiss_demo/data.txt", "r", encoding="utf-8") as f:
        texts = [line.strip() for line in f if line.strip()]

    build_index(texts, "faiss_demo/rfp.index")
