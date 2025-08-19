# backend/evaluate_instant.py
import os, io, gc, json, re
from typing import Dict, Optional
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from openai import OpenAI

# ---- 설정 / 클라이언트 ----
EVAL_MODEL = os.getenv("EVAL_MODEL", "gpt-4o")
EVAL_MAX_CHARS = int(os.getenv("EVAL_MAX_CHARS", "150000"))
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

router = APIRouter()

# ---- 경로 보정 ----
BACKEND_DIR = Path(__file__).resolve().parent
try:
    # config에 정의돼 있으면 사용
    from config import GUIDE_DEFAULT_PATH as _GUIDE_DEFAULT
    GUIDE_DEFAULT_ABS = str((BACKEND_DIR / _GUIDE_DEFAULT).resolve()) if not Path(_GUIDE_DEFAULT).is_absolute() else _GUIDE_DEFAULT
except Exception:
    # 없으면 backend/guide/guide_reference.txt 기본값
    GUIDE_DEFAULT_ABS = str((BACKEND_DIR / "guide" / "guide_reference.txt").resolve())

def _resolve_path(p: str) -> str:
    q = Path(p)
    if not q.is_absolute():
        q = BACKEND_DIR / q
    return str(q.resolve())

# ---- 문자열/JSON 유틸 ----
def _strip_code_fences(s: str) -> str:
    """```json ... ``` 같은 코드펜스 제거"""
    if not s:
        return s
    s = s.strip()
    m = re.match(r"^```[a-zA-Z0-9]*\s*([\s\S]*?)\s*```$", s)
    return m.group(1).strip() if m else s

def _normalize_quotes(s: str) -> str:
    if not s:
        return s
    return (s.replace("\u201c", '"').replace("\u201d", '"')
              .replace("\u2018", "'").replace("\u2019", "'"))

def _parse_json_loose(raw: str) -> dict:
    """펜스/스마트쿼트가 섞여도 최대한 JSON으로 파싱"""
    if not raw:
        return {}
    s = _normalize_quotes(_strip_code_fences(raw)).strip()
    if "{" in s and "}" in s:
        i, j = s.find("{"), s.rfind("}")
        cand = s[i:j+1]
    else:
        cand = s
    try:
        return json.loads(cand)
    except Exception:
        # 마지막 구제: 주요 필드만 정규식으로 긁어서 반환
        out = {}
        m_metric = re.search(r'"?metric"?\s*:\s*"?(CP|RI|FP|ETS|IO|RM)"?', cand, re.I)
        if m_metric:
            out["metric"] = m_metric.group(1).upper()
        m_score = re.search(r'"?score_10"?\s*:\s*"?(?P<n>\d+(\.\d+)?)"?', cand)
        if m_score:
            out["score_10"] = float(m_score.group("n"))
        m_fb = re.search(r'"?feedback"?\s*:\s*"(?P<fb>[\s\S]*?)"', cand)
        if m_fb:
            out["feedback"] = m_fb.group("fb")
        if not out:
            out["raw"] = raw
        return out

def _to_int_score(v, lo=0, hi=10) -> int:
    try:
        if isinstance(v, (int, float)):
            x = int(round(float(v)))
        elif isinstance(v, str):
            m = re.search(r"\d+(\.\d+)?", v)
            x = int(round(float(m.group(0)))) if m else 0
        else:
            x = 0
    except Exception:
        x = 0
    return max(lo, min(hi, x))

def _to_percent(v) -> int:
    return _to_int_score(v, lo=0, hi=100)

# ---- 텍스트 로딩 ----
def _read_text_stream(path: str, limit: int = EVAL_MAX_CHARS) -> str:
    real = _resolve_path(path)
    if not os.path.exists(real):
        raise HTTPException(status_code=404, detail=f"file not found: {real}")
    buf, total = io.StringIO(), 0
    with open(real, "r", encoding="utf-8", errors="ignore") as rf:
        for line in rf:
            if total >= limit:
                buf.write("\n...[TRUNCATED]...\n")
                break
            buf.write(line)
            total += len(line)
    s = buf.getvalue()
    buf.close()
    del buf; gc.collect()
    return s

# ---- 프롬프트 구성 ----
METRIC_INFO = {
    "CP": ("Clarity of Purpose (CP)",
           "프로젝트의 목표와 목적이 명확하고 구체적인지 여부",
           "목적 진술의 명확성, 목표의 구체성, 프로젝트의 기대 성과",
           "CP = (명확한 목표 + 구체적인 기대 성과) / 2"),
    "RI": ("Relevance and Impact (RI)",
           "프로젝트가 해당 분야나 사회에 미치는 영향과 관련성",
           "문제의 중요성, 기대되는 사회적/학문적 영향",
           "RI = (문제의 중요성 + 사회적/학문적 영향) / 2"),
    "FP": ("Feasibility and Planning (FP)",
           "프로젝트 계획의 실현 가능성과 세부 계획의 완성도",
           "실행 계획의 세부 사항, 타임라인, 예산 계획",
           "FP = (실행 계획의 세부 사항 + 타임라인 + 예산 계획) / 3"),
    "ETS":("Expertise and Team Strength (ETS)",
           "팀 구성원의 전문성과 경험",
           "팀 구성원의 경력, 과거 성과, 관련 분야의 전문성",
           "ETS = (팀 구성원의 경력 + 과거 성과 + 전문성) / 3"),
    "IO": ("Innovation and Originality (IO)",
           "프로젝트의 혁신성과 독창성",
           "새로운 접근법, 기존 연구와의 차별성, 창의적인 해결책",
           "IO = (새로운 접근법 + 차별성 + 창의적 해결책) / 3"),
    "RM": ("Risk Management (RM)",
           "프로젝트의 리스크 관리 계획",
           "리스크 식별, 완화 전략, 비상 계획",
           "RM = (리스크 식별 + 완화 전략 + 비상 계획) / 3"),
}

def _metric_msgs(key: str, proposal_text: str):
    title, definition, measure, formula = METRIC_INFO[key]
    user = f"""
아래 제안서 텍스트를 {title} 관점에서 평가하세요.

[평가 기준]
- 정의: {definition}
- 측정방법: {measure}
- 공식: {formula}

[출력 형식(JSON)]
{{
  "metric": "{key}",
  "score_10": <0~10 정수>,
  "feedback": "<해당 영역에 대한 충분히 긴 평가와 구체적 피드백를 한국어로>"
}}

[제안서]
{proposal_text}
""".strip()
    return [
        {"role":"system","content":"You are a strict, detail-oriented evaluator. Always return valid JSON only."},
        {"role":"user","content": user}
    ]

def _similarity_msgs(rfp_text: str, prop_text: str):
    user = f"""
다음은 RFP와 Proposal입니다. RFP 요구와 Proposal 간 유사성을 평가하세요.

[출력 형식(JSON)]
{{
  "similarity_percent": <0~100 정수>,
  "feedback": "<근거를 포함한 충분히 긴 피드백을 한국어로>"
}}

[RFP]
{rfp_text}

[Proposal]
{prop_text}
""".strip()
    return [
        {"role":"system","content":"You are a careful comparator. Return valid JSON only."},
        {"role":"user","content": user}
    ]

def _guide_msgs(guide_text: str, prop_text: str):
    user = f"""
다음은 평가 가이드와 Proposal입니다. 가이드를 기준으로 Proposal을 전반적으로 평가하세요.

[출력 형식(JSON)]
{{
  "overall_score_10": <0~10 정수>,
  "feedback": "<가이드 기준에 비추어 강점/보완점/권고를 충분히 길게 한국어로>"
}}

[Guide]
{guide_text}

[Proposal]
{prop_text}
""".strip()
    return [
        {"role":"system","content":"You are a rigorous reviewer. Return valid JSON only."},
        {"role":"user","content": user}
    ]

# ---- 입력/출력 모델 ----
class InstantEvalRequest(BaseModel):
    proposalPath: str
    rfpPath: Optional[str] = None
    guidePath: Optional[str] = None

class InstantEvalResponse(BaseModel):
    metrics: Dict[str, dict]           # 6개 영역 결과(JSON)
    metricsScores: Dict[str, int]      # {CP,RI,FP,ETS,IO,RM}: 0~10
    metricsTotal10: float              # 6개 평균(10점 만점)
    similarity: Optional[dict] = None  # {similarity_percent, feedback}
    guideReview: Optional[dict] = None # {overall_score_10, feedback}

# ---- 메인 엔드포인트 ----
@router.post("/instant", response_model=InstantEvalResponse)
async def run_instant(req: InstantEvalRequest):
    # 1) 제안서 텍스트
    p_text = _read_text_stream(req.proposalPath)[:EVAL_MAX_CHARS]

    # 2) 6개 메트릭
    metrics: Dict[str, dict] = {}
    scores: Dict[str, int] = {}
    for key in ["CP","RI","FP","ETS","IO","RM"]:
        msgs = _metric_msgs(key, p_text)
        res  = client.chat.completions.create(
            model=EVAL_MODEL,
            messages=msgs,
            max_tokens=900,
            temperature=0,
            response_format={"type": "json_object"}  # ★ JSON 강제
        )
        raw  = (res.choices[0].message.content or "").strip()
        data = _parse_json_loose(raw)

        metric_name = (data.get("metric") or key).upper()
        score = _to_int_score(data.get("score_10"))
        feedback = data.get("feedback") or data.get("raw") or ""

        metrics[metric_name] = {"metric": metric_name, "score_10": score, "feedback": feedback}
        scores[metric_name] = score

        del msgs, res, raw, data; gc.collect()

    vals = list(scores.values())
    total10 = round(sum(vals)/len(vals), 2) if vals else 0.0

    # 3) RFP 유사성(선택)
    similarity = None
    if req.rfpPath:
        r_text = _read_text_stream(req.rfpPath)[:EVAL_MAX_CHARS]
        msgs = _similarity_msgs(r_text, p_text)
        res  = client.chat.completions.create(
            model=EVAL_MODEL,
            messages=msgs,
            max_tokens=1200,
            temperature=0,
            response_format={"type": "json_object"}
        )
        raw  = (res.choices[0].message.content or "").strip()
        simd = _parse_json_loose(raw)
        similarity = {
            "similarity_percent": _to_percent(simd.get("similarity_percent")),
            "feedback": simd.get("feedback") or simd.get("raw") or ""
        }
        del r_text, msgs, res, raw, simd; gc.collect()

    # 4) 가이드 종합(선택; guidePath 없으면 기본경로 사용)
    guide_review = None
    g_path = req.guidePath or GUIDE_DEFAULT_ABS
    if g_path:
        g_text = _read_text_stream(g_path)[:EVAL_MAX_CHARS]
        msgs = _guide_msgs(g_text, p_text)
        res  = client.chat.completions.create(
            model=EVAL_MODEL,
            messages=msgs,
            max_tokens=1500,
            temperature=0,
            response_format={"type": "json_object"}
        )
        raw  = (res.choices[0].message.content or "").strip()
        gd = _parse_json_loose(raw)
        guide_review = {
            "overall_score_10": _to_int_score(gd.get("overall_score_10")),
            "feedback": gd.get("feedback") or gd.get("raw") or ""
        }
        del g_text, msgs, res, raw, gd; gc.collect()

    # 5) 메모리 최소화
    del p_text; gc.collect()

    return {
        "metrics": metrics,
        "metricsScores": scores,
        "metricsTotal10": total10,
        "similarity": similarity,
        "guideReview": guide_review,
    }
