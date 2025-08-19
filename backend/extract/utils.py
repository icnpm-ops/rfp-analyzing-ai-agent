import os

# 확장성 확보, 파일형 유추, 공통 유틸 관리
# .lower()를 통해 대소문자 구분 없이 확장자 인식

def get_file_type(filepath: str) -> str:
    ext = os.path.splitext(filepath)[-1].lower()
    if ext in ['.pdf']:
        return 'pdf'
    elif ext in ['.docx']:
        return 'docx'
    elif ext in ['.png', '.jpg', '.jpeg']:
        return 'image'
    else:
        return 'unknown'