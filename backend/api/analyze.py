# backend/api/analyze.py

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from typing import List, Optional
import asyncio
import os
import tempfile
from datetime import datetime

# 우리가 만든 모듈들 import
from extract.extractor import extract_text_from_file
from embedding.embedder import EmbeddingManager
from guide.Loader import load_guide_reference
from config import UPLOAD_DIR, EMBEDDING_DIR
import json

router = APIRouter()

# 전역 매니저들 (서버 시작시 한번만 초기화)
embedding_manager = None
guide_loader = None

def init_managers():
    """서버 시작시 매니저들을 초기화하는 함수"""
    global embedding_manager, guide_loader
    
    if embedding_manager is None:
        embedding_manager = EmbeddingManager(EMBEDDING_DIR)
        
    if guide_loader is None:
        guide_loader = load_guide_reference()

@router.post("/analyze")
async def analyze_documents(
    rfp_file: UploadFile = File(..., description="RFP 문서"),
    proposal_file: UploadFile = File(..., description="제안서 문서")
):
    """
    RFP와 제안서를 받아서 종합 분석하는 API
    
    Args:
        rfp_file: RFP PDF/DOCX 파일
        proposal_file: 제안서 PDF/DOCX 파일
    
    Returns:
        분석 결과 JSON
    """
    
    # 매니저 초기화
    init_managers()
    
    try:
        # 1단계: 임시 파일로 저장
        print("📁 파일 저장 중...")
        
        # 임시 디렉터리 생성
        with tempfile.TemporaryDirectory() as temp_dir:
            # RFP 파일 저장
            rfp_path = os.path.join(temp_dir, f"rfp_{rfp_file.filename}")
            with open(rfp_path, "wb") as f:
                content = await rfp_file.read()
                f.write(content)
            
            # 제안서 파일 저장    
            proposal_path = os.path.join(temp_dir, f"proposal_{proposal_file.filename}")
            with open(proposal_path, "wb") as f:
                content = await proposal_file.read()
                f.write(content)
            
            # 2단계: 텍스트 추출
            print("📄 텍스트 추출 중...")
            rfp_text = extract_text_from_file(rfp_path)
            proposal_text = extract_text_from_file(proposal_path)
            
            if not rfp_text or not proposal_text:
                raise HTTPException(
                    status_code=400, 
                    detail="파일에서 텍스트를 추출할 수 없습니다. 파일이 손상되었거나 지원하지 않는 형식일 수 있습니다."
                )
            
            # 3단계: RFP 핵심 요건 추출
            print("🔍 RFP 핵심 요건 분석 중...")
            rfp_summary = analyze_rfp_requirements(rfp_text)
            
            # 4단계: 제안서 충족률 계산
            print("📊 요구사항 충족률 계산 중...")
            compliance_rate = calculate_compliance_rate(rfp_text, proposal_text)
            
            # 5단계: IVI 점수 계산
            print("📈 IVI 점수 계산 중...")
            ivi_scores = calculate_ivi_scores(rfp_text, proposal_text)
            
            # 6단계: GPT 피드백 생성
            print("🤖 GPT 피드백 생성 중...")
            gpt_feedback = await generate_gpt_feedback(rfp_text, proposal_text)
            
            # 7단계: 제출 권장도 판단
            print("✅ 제출 권장도 판단 중...")
            recommendation = determine_submission_recommendation(
                compliance_rate, ivi_scores, gpt_feedback
            )
            
            # 최종 결과 조합
            result = {
                "analysis_id": f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "timestamp": datetime.now().isoformat(),
                "files": {
                    "rfp_filename": rfp_file.filename,
                    "proposal_filename": proposal_file.filename
                },
                "rfp_summary": rfp_summary,
                "compliance_rate": compliance_rate,
                "ivi_scores": ivi_scores,
                "gpt_feedback": gpt_feedback,
                "recommendation": recommendation,
                "status": "completed"
            }
            
            print("✨ 분석 완료!")
            return result
            
    except Exception as e:
        print(f"❌ 분석 중 오류: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"분석 처리 중 오류가 발생했습니다: {str(e)}"
        )

def analyze_rfp_requirements(rfp_text: str) -> dict:
    """RFP에서 핵심 요건들을 추출하는 함수"""
    
    # 간단한 키워드 기반 분석 (실제로는 더 정교한 NLP 필요)
    keywords = {
        "기술요구사항": ["기술", "시스템", "플랫폼", "아키텍처", "개발"],
        "기능요구사항": ["기능", "서비스", "처리", "관리", "제공"],
        "성능요구사항": ["성능", "속도", "처리량", "응답시간", "동시접속"],
        "보안요구사항": ["보안", "암호화", "인증", "권한", "보호"],
        "기타요구사항": ["유지보수", "지원", "교육", "문서화", "납기"]
    }
    
    requirements = {}
    text_lower = rfp_text.lower()
    
    for category, words in keywords.items():
        found_items = []
        for word in words:
            if word in text_lower:
                # 해당 키워드 주변 텍스트 찾기 (실제로는 더 정교한 로직 필요)
                found_items.append(f"{word} 관련 요구사항 발견")
        
        requirements[category] = found_items[:3]  # 최대 3개까지만
    
    return {
        "requirements": requirements,
        "total_requirements": sum(len(items) for items in requirements.values()),
        "summary": f"총 {sum(len(items) for items in requirements.values())}개의 요구사항이 식별되었습니다."
    }

def calculate_compliance_rate(rfp_text: str, proposal_text: str) -> dict:
    """제안서의 RFP 요구사항 충족률을 계산"""
    
    # 단순한 키워드 매칭 방식 (실제로는 임베딩 유사도 활용)
    rfp_keywords = extract_keywords(rfp_text)
    proposal_keywords = extract_keywords(proposal_text)
    
    matched_keywords = set(rfp_keywords) & set(proposal_keywords)
    
    if len(rfp_keywords) == 0:
        compliance_rate = 0
    else:
        compliance_rate = len(matched_keywords) / len(rfp_keywords) * 100
    
    return {
        "overall_rate": round(compliance_rate, 1),
        "matched_keywords": len(matched_keywords),
        "total_rfp_keywords": len(rfp_keywords),
        "details": {
            "기술적합성": round(compliance_rate * 0.9, 1),  # 예시
            "기능충족도": round(compliance_rate * 1.1, 1),
            "경험적합성": round(compliance_rate * 0.8, 1)
        }
    }

def extract_keywords(text: str) -> list:
    """텍스트에서 핵심 키워드 추출"""
    # 간단한 구현 (실제로는 더 정교한 NLP 필요)
    important_words = []
    words = text.lower().split()
    
    # 중요한 기술/비즈니스 키워드들
    tech_keywords = [
        "시스템", "개발", "구축", "플랫폼", "서비스", "관리", "처리",
        "데이터", "보안", "성능", "기능", "요구사항", "제안", "솔루션"
    ]
    
    for word in words:
        if word in tech_keywords:
            important_words.append(word)
    
    return list(set(important_words))  # 중복 제거

def calculate_ivi_scores(rfp_text: str, proposal_text: str) -> dict:
    """IVI 5Layer 점수 계산"""
    
    # 각 레이어별 점수 계산 (실제로는 더 복잡한 로직 필요)
    scores = {
        "혁신성": calculate_layer_score(proposal_text, ["혁신", "새로운", "창의", "독창"]),
        "기술력": calculate_layer_score(proposal_text, ["기술", "전문", "노하우", "경험"]),
        "실현가능성": calculate_layer_score(proposal_text, ["구체적", "계획", "일정", "방법"]),
        "경제성": calculate_layer_score(proposal_text, ["비용", "효율", "절약", "투자"]),
        "지속가능성": calculate_layer_score(proposal_text, ["유지", "지원", "확장", "발전"])
    }
    
    # 전체 평균 점수
    total_score = sum(scores.values()) / len(scores)
    
    return {
        "layers": scores,
        "total_score": round(total_score, 1),
        "grade": get_ivi_grade(total_score),
        "radar_data": [
            {"layer": k, "score": v} for k, v in scores.items()
        ]
    }

def calculate_layer_score(text: str, keywords: list) -> float:
    """개별 레이어 점수 계산"""
    text_lower = text.lower()
    found_count = sum(1 for keyword in keywords if keyword in text_lower)
    
    # 키워드 발견 개수에 따른 점수 (최대 100점)
    base_score = min(found_count * 25, 100)
    
    # 텍스트 길이에 따른 보정 (더 자세할수록 높은 점수)
    length_bonus = min(len(text) / 1000 * 10, 20)
    
    return min(base_score + length_bonus, 100)

def get_ivi_grade(score: float) -> str:
    """IVI 점수에 따른 등급 반환"""
    if score >= 90:
        return "A+"
    elif score >= 80:
        return "A"
    elif score >= 70:
        return "B+"
    elif score >= 60:
        return "B"
    elif score >= 50:
        return "C+"
    else:
        return "C"

async def generate_gpt_feedback(rfp_text: str, proposal_text: str) -> dict:
    """GPT를 활용한 피드백 생성"""
    
    # 실제 구현에서는 OpenAI API 호출
    # 여기서는 시뮬레이션
    
    await asyncio.sleep(1)  # API 호출 시뮬레이션
    
    return {
        "overall_feedback": "제안서가 RFP의 주요 요구사항을 대부분 충족하고 있으나, 보안 관련 세부사항과 성능 최적화 방안에 대한 추가 설명이 필요합니다.",
        "strengths": [
            "기술 아키텍처에 대한 명확한 설명",
            "프로젝트 일정 및 마일스톤이 구체적으로 제시됨",
            "팀 구성원의 경험과 역량이 잘 드러남"
        ],
        "improvements": [
            "보안 요구사항에 대한 구체적인 대응방안 추가 필요",
            "성능 테스트 계획 및 기준 명시 필요",
            "위험 관리 계획의 세부사항 보완 필요"
        ],
        "recommendations": [
            "보안 컨설턴트와의 협업 계획 추가",
            "성능 벤치마킹 결과 첨부",
            "위험 요소별 대응 시나리오 작성"
        ]
    }

def determine_submission_recommendation(
    compliance_rate: dict, 
    ivi_scores: dict, 
    gpt_feedback: dict
) -> dict:
    """제출 권장도 판단"""
    
    overall_compliance = compliance_rate["overall_rate"]
    total_ivi = ivi_scores["total_score"]
    
    # 점수 기반 판단
    if overall_compliance >= 80 and total_ivi >= 75:
        recommendation = "제출 권장"
        confidence = "높음"
        reason = "요구사항 충족률과 IVI 점수가 모두 우수합니다."
        color = "green"
    elif overall_compliance >= 60 and total_ivi >= 60:
        recommendation = "조건부 제출"
        confidence = "보통"
        reason = "기본 요구사항은 만족하나 일부 개선이 필요합니다."
        color = "yellow"
    else:
        recommendation = "재작성 필요"
        confidence = "높음"
        reason = "요구사항 충족률이 낮아 대폭 개선이 필요합니다."
        color = "red"
    
    return {
        "decision": recommendation,
        "confidence": confidence,
        "reason": reason,
        "color": color,
        "scores": {
            "compliance": overall_compliance,
            "ivi": total_ivi
        }
    }