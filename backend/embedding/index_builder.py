# backend/embedding/index_builder.py
import os
import sys
import time
import math
import logging
import traceback
import sqlite3
from typing import List, Dict, Optional, Tuple

import numpy as np
import faiss
from dotenv import load_dotenv
from openai import OpenAI

from backend.config import INDEX_PATH, TEXTS_PATH, EMBED_MODEL

# -----------------------------
# 로깅
# -----------------------------
LOG_LEVEL = os.getenv("RFP_LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("index_builder")

# -----------------------------
# OpenAI
# -----------------------------
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# -----------------------------
# 설정
# -----------------------------
EMBED_DIM = 1536  # text-embedding-3-small
DEFAULT_BATCH_SIZE = int(os.getenv("RFP_EMBED_BATCH", "128"))
DEFAULT_INDEX_FACTORY = os.getenv("RFP_FAISS_FACTORY", "IVF4096,PQ64")
FAISS_METRIC = faiss.METRIC_INNER_PRODUCT  # 코사인 유사도처럼 쓰려면 내적+정규화 권장

# SQLite 경로: TEXTS_PATH와 같은 폴더에 chunks.db 생성
META_DIR = os.path.dirname(TEXTS_PATH)
META_DB_PATH = os.path.join(META_DIR, "chunks.db")

# -----------------------------
# 유틸
# -----------------------------
def _ensure_dir(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)

# -----------------------------
# SQLite (메타 저장소)
# -----------------------------
def _open_meta_db() -> sqlite3.Connection:
    _ensure_dir(META_DB_PATH)
    conn = sqlite3.connect(META_DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id      INTEGER PRIMARY KEY,   -- 벡터 ID와 동일
            docId   TEXT,
            text    TEXT
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_docId ON chunks(docId);")
    return conn

def _meta_max_id(conn: sqlite3.Connection) -> int:
    cur = conn.execute("SELECT COALESCE(MAX(id), -1) FROM chunks")
    (mx,) = cur.fetchone()
    return int(mx)

def _meta_insert_many(conn: sqlite3.Connection, rows: List[Tuple[int,str,str]]):
    with conn:
        conn.executemany(
            "INSERT OR REPLACE INTO chunks(id, docId, text) VALUES (?,?,?)",
            rows
        )

def _meta_get_by_ids(conn: sqlite3.Connection, ids: List[int]) -> List[Dict]:
    if not ids:
        return []
    qmarks = ",".join("?" for _ in ids)
    cur = conn.execute(
        f"SELECT id, docId, text FROM chunks WHERE id IN ({qmarks})",
        [int(i) for i in ids]
    )
    return [{"id": r[0], "docId": r[1], "text": r[2]} for r in cur.fetchall()]

def migrate_pickle_to_sqlite():
    """기존 texts.pkl을 SQLite(chunks.db)로 1회 이관."""
    if not os.path.exists(TEXTS_PATH):
        log.info("No pickle to migrate (texts.pkl not found).")
        return
    conn = _open_meta_db()
    try:
        import pickle
        with open(TEXTS_PATH, "rb") as f:
            items = pickle.load(f)
        migrated: List[Dict] = []
        for it in items:
            if isinstance(it, str):
                migrated.append({"text": it, "docId": "legacy"})
            else:
                migrated.append({"text": it.get("text",""), "docId": it.get("docId","legacy")})

        base = _meta_max_id(conn) + 1
        rows = [(base + i, it["docId"], it["text"]) for i, it in enumerate(migrated)]
        _meta_insert_many(conn, rows)
        log.info(f"Migrated {len(rows)} items from pickle to SQLite at {META_DB_PATH}")
        log.info("Pickle kept for backup. You can delete it after verifying.")
    except Exception:
        log.error("Migration failed.")
        log.error("".join(traceback.format_exception(*sys.exc_info())))
        raise
    finally:
        conn.close()

# -----------------------------
# OpenAI 임베딩 (배치)
# -----------------------------
def get_embeddings_batch(texts: List[str], model: str, max_retry: int = 3, retry_delay: float = 2.0) -> np.ndarray:
    for attempt in range(1, max_retry + 1):
        try:
            t0 = time.time()
            res = client.embeddings.create(input=texts, model=model)
            vecs = np.array([d.embedding for d in res.data], dtype="float32")

            # (선택) 코사인 유사도 정합 위해 정규화
            if FAISS_METRIC == faiss.METRIC_INNER_PRODUCT:
                faiss.normalize_L2(vecs)

            dt = time.time() - t0
            log.info(f"Embeddings OK | batch={len(texts)} | dim={vecs.shape[1]} | {dt:.2f}s")
            return vecs
        except Exception as e:
            log.warning(f"[attempt {attempt}/{max_retry}] Embedding API error: {e}")
            if attempt == max_retry:
                log.error("Embedding failed after max retries. Raising exception.")
                raise
            time.sleep(retry_delay * attempt)

# -----------------------------
# 인덱스
# -----------------------------
def _new_index(dim: int, index_factory: Optional[str] = None) -> faiss.Index:
    if index_factory is None or index_factory.lower() == "flat":
        log.info("Creating IndexFlat")
        return faiss.IndexFlatIP(dim) if FAISS_METRIC == faiss.METRIC_INNER_PRODUCT else faiss.IndexFlatL2(dim)
    log.info(f"Creating index via factory: {index_factory}")
    return faiss.index_factory(dim, index_factory, FAISS_METRIC)

def _load_or_create_index(dim: int, index_path: str, index_factory: Optional[str]) -> faiss.Index:
    if os.path.exists(index_path):
        log.info(f"Reading existing FAISS index: {index_path}")
        return faiss.read_index(index_path)
    idx = _new_index(dim, index_factory)
    _ensure_dir(index_path)
    faiss.write_index(idx, index_path)
    log.info(f"Created new FAISS index: {index_path}")
    return idx

def _wrap_idmap(index: faiss.Index) -> faiss.Index:
    if isinstance(index, (faiss.IndexIDMap, faiss.IndexIDMap2)):
        return index
    if getattr(index, "ntotal", 0) == 0:
        log.info("Wrapping empty index with IndexIDMap2 (stable integer IDs).")
        return faiss.IndexIDMap2(index)
    log.warning("Index already contains vectors(ntotal>0); skip IDMap wrapping. "
                "Implicit IDs (0..ntotal-1) will be used.")
    return index

def _train_if_needed(index: faiss.Index, sample_vectors: np.ndarray):
    if hasattr(index, "is_trained") and not index.is_trained:
        log.info(f"Training index... | samples={len(sample_vectors)} dim={sample_vectors.shape[1]}")
        index.train(sample_vectors)
        log.info("Index trained.")

# -----------------------------
# 공개 API: append_texts
# -----------------------------
def append_texts(chunks: List[str], doc_id: str, batch_size: int = DEFAULT_BATCH_SIZE):
    if not chunks:
        log.warning("append_texts called with empty chunks.")
        return

    conn = _open_meta_db()
    try:
        index = _load_or_create_index(EMBED_DIM, INDEX_PATH, DEFAULT_INDEX_FACTORY)
        index = _wrap_idmap(index)

        total = len(chunks)
        num_batches = math.ceil(total / batch_size)
        if isinstance(index, (faiss.IndexIDMap, faiss.IndexIDMap2)):
            base_id = _meta_max_id(conn) + 1
        else:
            base_id = int(getattr(index, "ntotal", 0))
        log.info(f"Append texts | doc_id={doc_id} | total_chunks={total} | batch_size={batch_size} | start_id={base_id}")

        # 훈련 (IVF/PQ 등)
        if hasattr(index, "is_trained") and not index.is_trained:
            warm = chunks[: min(total, 100_000)]
            warm_vecs = []
            for i in range(0, len(warm), batch_size):
                batch = warm[i:i+batch_size]
                v = get_embeddings_batch(batch, EMBED_MODEL)
                warm_vecs.append(v)
            warm_mat = np.vstack(warm_vecs)
            _train_if_needed(index, warm_mat)
            del warm_vecs, warm_mat

        cur_id = base_id
        for b in range(num_batches):
            s, e = b * batch_size, min((b + 1) * batch_size, total)
            batch_texts = chunks[s:e]
            log.info(f"[append batch {b+1}/{num_batches}] embedding... range={s}:{e}")

            vecs = get_embeddings_batch(batch_texts, EMBED_MODEL)
            ids = np.arange(cur_id, cur_id + len(batch_texts), dtype=np.int64)

            if isinstance(index, (faiss.IndexIDMap, faiss.IndexIDMap2)):
                index.add_with_ids(vecs, ids)
            else:
                index.add(vecs)

            rows = [(int(ids[i]), doc_id, batch_texts[i]) for i in range(len(batch_texts))]
            _meta_insert_many(conn, rows)

            cur_id += len(batch_texts)
            log.info(f"[append batch {b+1}/{num_batches}] added={len(batch_texts)} | index.ntotal={index.ntotal}")

        _ensure_dir(INDEX_PATH)
        faiss.write_index(index, INDEX_PATH)
        log.info(f"Append done | index={INDEX_PATH} | meta_db={META_DB_PATH} | ntotal={index.ntotal}")

    except Exception:
        log.error("append_texts failed.")
        log.error("".join(traceback.format_exception(*sys.exc_info())))
        raise
    finally:
        conn.close()

# -----------------------------
# 공개 API: build_index
# -----------------------------
def build_index(texts: List[str], batch_size: int = DEFAULT_BATCH_SIZE):
    if not texts:
        log.warning("build_index called with empty texts.")
        return

    conn = _open_meta_db()
    try:
        total = len(texts)
        num_batches = math.ceil(total / batch_size)
        log.info(f"Build index | total_texts={total} | batch_size={batch_size} | index_factory={DEFAULT_INDEX_FACTORY}")

        idx = _new_index(EMBED_DIM, DEFAULT_INDEX_FACTORY)
        idx = _wrap_idmap(idx)

        with conn:
            conn.execute("DELETE FROM chunks;")
        log.info("Meta DB cleared for fresh build.")

        if hasattr(idx, "is_trained") and not idx.is_trained:
            warm = texts[: min(total, 100_000)]
            warm_vecs = []
            for b in range(0, len(warm), batch_size):
                batch = warm[b:b+batch_size]
                log.info(f"[train batch] embedding... range={b}:{b+len(batch)}")
                v = get_embeddings_batch(batch, EMBED_MODEL)
                warm_vecs.append(v)
            warm_mat = np.vstack(warm_vecs)
            _train_if_needed(idx, warm_mat)
            del warm_vecs, warm_mat

        next_id = 0
        for bi in range(num_batches):
            s, e = bi * batch_size, min((bi + 1) * batch_size, total)
            batch_texts = texts[s:e]
            log.info(f"[build batch {bi+1}/{num_batches}] embedding... range={s}:{e}")

            vecs = get_embeddings_batch(batch_texts, EMBED_MODEL)
            ids = np.arange(next_id, next_id + len(batch_texts), dtype=np.int64)
            idx.add_with_ids(vecs, ids)

            rows = [(int(ids[i]), "build", batch_texts[i]) for i in range(len(batch_texts))]
            _meta_insert_many(conn, rows)

            next_id += len(batch_texts)
            log.info(f"[build batch {bi+1}/{num_batches}] added={len(batch_texts)} | ntotal={idx.ntotal}")

        _ensure_dir(INDEX_PATH)
        faiss.write_index(idx, INDEX_PATH)

        log.info("저장 완료:")
        log.info(f"- FAISS index: {INDEX_PATH}")
        log.info(f"- Meta DB:     {META_DB_PATH}")
        log.info(f"- 총 벡터 수:   {idx.ntotal}")

    except Exception:
        log.error("build_index failed.")
        log.error("".join(traceback.format_exception(*sys.exc_info())))
        raise
    finally:
        conn.close()

# -----------------------------
# 공개 API: search
# -----------------------------
def search(query: str, topk: int = 10) -> List[Dict]:
    """쿼리 임베딩 → 인덱스 검색 → SQLite 메타 조인."""
    idx = faiss.read_index(INDEX_PATH)
    if hasattr(idx, "nprobe"):
        idx.nprobe = int(os.getenv("RFP_NPROBE", "32"))

    qv = get_embeddings_batch([query], EMBED_MODEL)[0].reshape(1, -1)
    scores, ids = idx.search(qv, topk)
    ids = ids[0].tolist()
    scores = scores[0].tolist()

    conn = _open_meta_db()
    try:
        rows = _meta_get_by_ids(conn, ids)
        row_map = {r["id"]: r for r in rows}
        result = []
        for i, vid in enumerate(ids):
            if vid in row_map:
                r = row_map[vid].copy()
                r["score"] = float(scores[i])
                result.append(r)
        return result
    finally:
        conn.close()

# -----------------------------
# 모듈 단독 실행: 1회 마이그레이션 + 미니 데모
# -----------------------------
if __name__ == "__main__":
    migrate_pickle_to_sqlite()
    sample = [
        "본 제안서는 인공지능 기반 분석 도구 개발을 목표로 합니다.",
        "RFP는 연구개발 과제를 정의하는 문서입니다.",
        "OCR과 VLM 기술을 접목해 복합 문서 분석이 가능합니다.",
        "프로젝트는 2025년 8월까지 베타 버전 완성을 목표로 합니다.",
        "도표나 그래프도 정량적으로 해석할 수 있도록 설계됩니다."
    ]
    build_index(sample)
    log.info("Search demo: " + str(search("인공지능 분석 도구", topk=3)))
