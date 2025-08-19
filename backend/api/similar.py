import os
import pickle
import numpy as np
import faiss
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
from config import INDEX_PATH, TEXTS_PATH, EMBED_MODEL



load_dotenv()

# .env 기반 API 키 관리 -> 보안성 향상
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# API 라웉로 모듈화 쉽도록. FastAPI 모듈을 분리 관리할 수 있어 유지 보수에 유리

router = APIRouter()


# 요청값은 검색질의와 반환할 갯수
# 응답값은 문장 + 유사도 점수
# pydantic.BaseModel 은 자동 타입 검증 + 문서 자동화 (/docs)
class SimilarityRequest(BaseModel):
    query: str
    top_k: int = 3

class SimilarResult(BaseModel):
    text: str
    score: float

def get_embedding(text: str) -> list:
    res = client.embeddings.create(
        input=[text],
        model=EMBED_MODEL
    )
    return res.data[0].embedding

# OPENAI로 쿼리 임베딩 생성
# FAISS 인덱스로 유사 벡터 검색
# 상위 k개 결과 반환
# index.search()는 score, index 쌍 반환(거리값 작을수록 유사함)
# float32로 맞춰야 FAISS 오류 없음

def search_similar(query: str, k: int = 3) -> list[tuple[str,float]]:
    print("📂 현재 작업 디렉토리:", os.getcwd())
    print("📂 INDEX_PATH:", os.path.abspath(INDEX_PATH))
    print("📂 TEXTS_PATH:", os.path.abspath(TEXTS_PATH))


    if not os.path.exists(INDEX_PATH) or not os.path.exists(TEXTS_PATH):
        raise FileNotFoundError("FAISS index or texts.pkl not found")
    
    index = faiss.read_index(INDEX_PATH)
    with open(TEXTS_PATH, "rb") as f:
        texts = pickle.load(f)

    query_emb = np.array([get_embedding(query)], dtype="float32")
    scores, indices = index.search(query_emb, k)

    results = []
    for i, score in zip(indices[0], scores[0]):
        results.append((texts[i], float(score)))
    return results

@router.post("/search-similar", response_model=list[SimilarResult])
async def search_endpoint(req: SimilarityRequest):
    try:
        results = search_similar(req.query, req.top_k)
        return [{"text": text, "score": round(score, 4)} for text, score in results]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    