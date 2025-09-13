
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict
from pathlib import Path
import uuid
import os
import re
import math
import logging

from pydantic import BaseModel

from backend.extract.extractor import extract_all, extract_to_txt
from backend.evaluate_instant import router as eval_instant_router

from backend.config import (
    FRONTEND_ORIGINS, UPLOAD_DIR, ALLOWED_EXTENSIONS, MAX_UPLOAD_MB  # MAX_UPLOAD_MB는 필요 시 사용
)

from backend.embedding.index_builder import append_texts  # SQLite 메타 + 배치 임베딩
from backend.embedding.metadata_store import save_metadata, get_metadata_by_id, load_all_metadata

logger = logging.getLogger(__name__)

# ---------------- App / CORS ----------------
app = FastAPI()
app.include_router(eval_instant_router, prefix="/evaluate", tags=["evaluate"])

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://rfp-analyzer.vercel.app",
        "https://rfp-analyzing-ai-agent.vercel.app",
        "https://rfp-analyzing-ai-agent-ctk0fanul-icnpm-ops-projects.vercel.app",
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- Upload helpers ----------------
def _validate_file(file: UploadFile):
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"허용되지 않은 확장자: {ext}")

def _save_upload_stream(file: UploadFile, dest: Path):
    with dest.open("wb") as buffer:
        while True:
            chunk = file.file.read(1024 * 1024)  # 1MB
            if not chunk:
                break
            buffer.write(chunk)

def _streaming_chunk_from_txt(path: str, chunk_size=1200, overlap=200):
    buf = []
    total = 0
    with open(path, "r", encoding="utf-8", errors="ignore") as rf:
        for line in rf:
            buf.append(line)
            total += len(line)
            if total >= chunk_size:
                big = "".join(buf)
                head = big[:chunk_size]
                if head.strip():
                    yield head.strip()
                tail = big[chunk_size - overlap:]
                buf = [tail] if tail else []
                total = len(tail) if tail else 0
    if buf:
        big = "".join(buf)
        step = max(1, chunk_size - overlap)
        for start in range(0, len(big), step):
            end = min(len(big), start + chunk_size)
            piece = big[start:end]
            if piece.strip():
                yield piece.strip()

def _index_after_upload(txt_path: str, doc_id: str):
    logger.info(f"[upload-bg] index start | doc_id={doc_id} | txt={txt_path}")
    try:
        batch: List[str] = []
        for ch in _streaming_chunk_from_txt(txt_path, chunk_size=1200, overlap=200):
            batch.append(ch)
            if len(batch) >= 200:
                append_texts(batch, doc_id=doc_id)
                batch.clear()
        if batch:
            append_texts(batch, doc_id=doc_id)
        logger.info(f"[upload-bg] append_texts ok | doc_id={doc_id}")
    except Exception:
        logger.exception("[upload-bg] append_texts failed")
    finally:
        logger.info(f"[upload-bg] done | doc_id={doc_id}")

# ---------------- Analyze (기존 데모용) ----------------
class AnalyzeRequest(BaseModel):
    rfpId: str
    proposalId: str

class RadarItem(BaseModel):
    axis: str
    value: int  # 0~100

class AnalyzeResponse(BaseModel):
    rfpId: str
    proposalId: str
    rfpSummary: str
    matchRate: int  # 0~100
    ivi: Dict[str, int]  # overall,planning,feasibility,evidence,risk,clarity
    radar: List[RadarItem]
    feedback: List[str]
    decision: str  # SUBMIT | HOLD | REWRITE

try:
    from extract.extractor import extract_text
except ImportError:
    def extract_text(path: str) -> str:
        ext = Path(path).suffix.lower()
        if ext == ".pdf":
            import pdfplumber
            out = []
            with pdfplumber.open(path) as pdf:
                for page in pdf.pages:
                    out.append(page.extract_text() or "")
            return "\n".join(out)
        elif ext == ".docx":
            import docx
            doc = docx.Document(path)
            return "\n".join(p.text for p in doc.paragraphs)
        else:
            return ""

def _read_text(path: str) -> str:
    try:
        res = extract_all(path, use_vlm=False)
        txt = (res.get("text") or "").strip()
    except Exception:
        txt = (extract_text(path) or "").strip()
    return txt

def _summarize_head(text: str, max_chars: int = 700) -> str:
    return text[:max_chars] + ("..." if len(text) > max_chars else "")

_word_re = re.compile(r"[A-Za-z가-힣0-9_]+")
_KO_STOP = set(["그리고","그러나","하지만","또는","및","또","또한","따라서","이것","저것","여기","저기","위해","대한","하는","했다","합니다","수","등","것","때문","보다","에서","으로","이다","있는","없는"])
_EN_STOP = set(["the","a","an","and","or","but","of","to","in","for","on","with","as","is","are","was","were","be","been","by","at","from","that","this","it","its","we","you","they"])

def _tokens(text: str) -> List[str]:
    toks = [t.lower() for t in _word_re.findall(text)]
    return [t for t in toks if t not in _KO_STOP and t not in _EN_STOP and len(t) > 1]

def _keyword_set(text: str, top_n: int = 120) -> set:
    freq: Dict[str, int] = {}
    for t in _tokens(text):
        freq[t] = freq.get(t, 0) + 1
    keys = sorted(freq.items(), key=lambda x: (-x[1], x[0]))[:top_n]
    return set([k for k,_ in keys])

def _match_rate(rfp_text: str, prop_text: str) -> int:
    rfp_kw = _keyword_set(rfp_text)
    prop_kw = _keyword_set(prop_text)
    if not rfp_kw:
        return 0
    covered = len(rfp_kw & prop_kw)
    rate = int(round(covered * 100 / len(rfp_kw)))
    return max(0, min(100, rate))

def _sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))

def _ivi_and_radar(rfp_text: str, prop_text: str, match_rate: int) -> Dict[str, any]:
    len_score = int(_sigmoid(len(prop_text) / 3000.0) * 100)
    structure_bonus = 10 if ("목차" in prop_text or "개요" in prop_text) else 0
    planning    = max(0, min(100, int(0.5*match_rate + 0.5*len_score)))
    feasibility = max(0, min(100, int(0.6*match_rate + 0.4*(len_score+structure_bonus))))
    evidence    = max(0, min(100, int(0.4*match_rate + 0.6*len_score)))
    risk        = max(0, min(100, int(0.7*match_rate + 0.3*len_score)))
    clarity     = max(0, min(100, int(0.5*match_rate + 0.5*(100 - abs(len(prop_text)-len(rfp_text))/max(1,len(rfp_text))*100))))
    overall = int(round((planning + feasibility + evidence + risk + clarity) / 5))
    radar = [
        {"axis": "Planning", "value": planning},
        {"axis": "Feasibility", "value": feasibility},
        {"axis": "Evidence", "value": evidence},
        {"axis": "Risk", "value": risk},
        {"axis": "Clarity", "value": clarity},
    ]
    return {
        "overall": overall,
        "planning": planning,
        "feasibility": feasibility,
        "evidence": evidence,
        "risk": risk,
        "clarity": clarity,
        "radar": radar,
    }

def _feedback_and_decision(match_rate: int, ivi_overall: int):
    tips: List[str] = []
    if match_rate < 60:
        tips.append("RFP 키워드 커버리지가 낮습니다. 요구사항 목록을 추출해 항목별 대응을 명시하세요.")
    if ivi_overall < 70:
        tips.append("실행 가능성과 근거 제시가 부족해 보입니다. 일정/예산/리스크 테이블과 참고 자료를 보강하세요.")
    if not tips:
        tips.append("핵심 요구사항 대응이 양호합니다. 제출 전 표/지표의 단위와 출처만 재확인하세요.")
    decision = "SUBMIT" if (ivi_overall >= 75 and match_rate >= 70) else ("HOLD" if ivi_overall >= 55 else "REWRITE")
    return tips, decision

# ---------------- 업로드 ----------------
@app.post("/upload")
async def upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    title: str = Form(...),
    docType: str = Form(...),  # "RFP" / "Proposal"
):
    if docType.lower() == "":
        raise HTTPException(status_code=403, detail="Guide 업로드는 관리자 전용 API를 사용하세요.")

    _validate_file(file)
    doc_id = str(uuid.uuid4())
    ext = Path(file.filename).suffix.lower()
    dest = Path(UPLOAD_DIR) / f"{doc_id}{ext}"

    try:
        _save_upload_stream(file, dest)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"파일 저장 오류: {e}")

    # 1) 메타데이터 저장
    try:
        save_metadata(title=title, source=str(dest), doc_type=docType, doc_id=doc_id)
    except Exception:
        import logging; logging.exception("[upload] save_metadata failed")

    # 2) 텍스트 변환을 동기 수행 (extracted.txt 보장)
    try:
        info = extract_to_txt(
            str(dest),
            use_ocr=False,          # 요청: OCR 대신 VLM
            use_vlm=True,
            doc_type_for_name=docType,  # 파일명에 RFP/Proposal 표시
            title_for_name=title,       # 파일명에 업로드 제목 표시
            doc_id_for_name=doc_id,     # 파일명에 docId 표시
        )
        txt_path = info["txt_path"]
        logger.info(f"[upload] to-txt ok | txt={txt_path} | images={info.get('n_images')} vlm_ok={info.get('n_vlm_ok')}")
    except Exception:
        logger.exception("[upload] extract_to_txt failed")
        raise HTTPException(status_code=500, detail="텍스트 변환 실패")

    # 3) 인덱싱만 백그라운드
    try:
        background_tasks.add_task(_index_after_upload, txt_path, doc_id)
    except Exception:
        logger.exception("[upload] schedule _index_after_upload failed")

    # 4) 프론트로 변환된 txt 경로 반환
    return {
        "ok": True,
        "ready": True,
        "docID": doc_id,
        "storeAs": str(dest),
        "txtPath": txt_path,  # ★ 프론트는 이 경로를 그대로 /evaluate/instant에 넘겨주세요
        "message": "업로드 및 텍스트 변환 완료. 인덱싱은 백그라운드에서 처리됩니다.",
    }

# ---------------- Analyze (데모 엔드포인트 유지) ----------------
@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest):
    rfp_meta = get_metadata_by_id(req.rfpId)
    prop_meta = get_metadata_by_id(req.proposalId)
    if not rfp_meta or not prop_meta:
        raise HTTPException(status_code=404, detail="문서 메타데이터를 찾을 수 없습니다.")
    if rfp_meta.get("docType","").lower() != "rfp" or prop_meta.get("docType","").lower() != "proposal":
        raise HTTPException(status_code=400, detail="docType(RFP/Proposal) 매칭이 올바르지 않습니다.")

    rfp_text = _read_text(rfp_meta["source"])
    prop_text = _read_text(prop_meta["source"])
    if not rfp_text or not prop_text:
        raise HTTPException(status_code=400, detail="텍스트 추출 실패(빈 문서)")

    rfp_summary = _summarize_head(rfp_text)
    mrate = _match_rate(rfp_text, prop_text)
    ivi = _ivi_and_radar(rfp_text, prop_text, mrate)
    feedback, decision = _feedback_and_decision(mrate, ivi["overall"])

    return {
        "rfpId": req.rfpId,
        "proposalId": req.proposalId,
        "rfpSummary": rfp_summary,
        "matchRate": mrate,
        "ivi": {
            "overall": ivi["overall"],
            "planning": ivi["planning"],
            "feasibility": ivi["feasibility"],
            "evidence": ivi["evidence"],
            "risk": ivi["risk"],
            "clarity": ivi["clarity"],
        },
        "radar": ivi["radar"],
        "feedback": feedback,
        "decision": decision,
    }

# ---------------- Debug ----------------
@app.get("/_debug/metadata")
async def debug_metadata():
    return load_all_metadata()
