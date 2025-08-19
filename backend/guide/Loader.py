import os
from backend.config import GUIDE_REFERENCE_PATH

_guide_cache = None # 캐시용

def load_guide_reference() -> str:
    global _guide_cache
    if _guide_cache:
        return _guide_cache
    
    if not os.path.exists(GUIDE_REFERENCE_PATH):
        raise FileNotFoundError(f"Guide 기준 문서가 존재하지 않습니다 : {GUIDE_REFERENCE_PATH}")
    
    with open(GUIDE_REFERENCE_PATH, "r", encoding="utf-8") as f:
        _guide_cache = f.read().strip()

    print("Guide 기준 문서 로딩 완료")
    return _guide_cache

if __name__ == "__main__":
    print("📥 Guide 기준 문서 로딩 테스트 시작...")
    
    try:
        guide_text = load_guide_reference()
        print("✅ 로드된 내용 (앞 500자):\n")
        print(guide_text[:500])  # 너무 길 경우 일부만 출력
        print("\n📦 전체 길이:", len(guide_text), "자")
    except FileNotFoundError as e:
        print(e)
