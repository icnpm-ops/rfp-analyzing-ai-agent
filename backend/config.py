# backend/config.py

import os
from pathlib import Path
# 이 config.py가 있는 폴더 기준

# backend/config.py
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 영속 디렉토리(볼륨 마운트 대상). 없으면 /data, 더 없으면 BASE_DIR/persist
PERSIST_DIR = os.getenv("PERSIST_DIR", "/data")
if not os.path.isdir(PERSIST_DIR) or not os.access(PERSIST_DIR, os.W_OK):
    PERSIST_DIR = os.path.join(BASE_DIR, "persist")
os.makedirs(PERSIST_DIR, exist_ok=True)

# 임베딩/인덱스 파일은 영속 디렉토리에
INDEX_PATH = os.path.join(PERSIST_DIR, "faiss.index")
TEXTS_PATH = os.path.join(PERSIST_DIR, "texts.pkl")

# 업로드 디렉토리도 영속화
UPLOAD_DIR = os.path.join(PERSIST_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# 기타 필요 경로…
GUIDE_REFERENCE_PATH = os.path.join(BASE_DIR, "guide", "guide_reference.txt")  # 읽기 전용이면 코드 디렉토리 유지
METADATA_PATH = os.path.join(PERSIST_DIR, "metadata.json")

# 모델명 등 상수
EMBED_MODEL = "text-embedding-3-small"

BACKEND_DIR = Path(__file__).resolve().parent

# OpenAI 모델명 (고정)
EMBED_MODEL = "text-embedding-3-small"


# 보고서 출력 루트 디렉터리 (환경변수로도 변경 가능)
REPORT_DIR = os.getenv("REPORT_DIR", os.path.join(os.getcwd(), "storage", "reports"))

# GPT 모델 (원하면 .env로 바꾸세요)
EVAL_MODEL = os.getenv("EVAL_MODEL", "gpt-4o")


ALLOWED_EXTENSIONS = (".pdf", ".docx") # 필요시 이미지 확장자 추가 가능
MAX_UPLOAD_MB = 500 # 필요시 조정

# CORS (프론트 개발 서버 도메인/포트)
FRONTEND_ORIGINS = [
    "http://localhost:5173",  # Vite 기본 포트
    "http://localhost:3000",  # CRA 기본 포트
]

# backend/config.py
PDF_DPI = 120                 # 300 → 120 (메모리/디스크 사용 크게 감소)
OCR_MAX_PAGES = 10            # OCR 최대 페이지 수
OCR_AUTO_SKIP_TEXT_LEN = 800  # 텍스트가 충분하면 OCR 생략
OCR_THREAD_WORKERS = 2        # OCR 병렬도(윈도우 2~3 권장)


# --- 텍스트 캐시/정책 ---
TEXT_CACHE_DIR = os.path.join(PERSIST_DIR, "text_cache")
os.makedirs(TEXT_CACHE_DIR, exist_ok=True)

# 원본 삭제 여부(기본: False)
DELETE_ORIGINAL_AFTER_CACHE = False  # True로 켜면 캐시 저장 후 원본 PDF/DOCX 삭제

# 보관 일수(옵션): 0 이면 자동 삭제 안 함
RETENTION_DAYS = 0
