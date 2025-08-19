
# 가장 빠르고 정확하게 텍스트를 추출하는 1차 처리
# pdfplumber : 레이아웃 유지가 좋고 신뢰도 높음. 텍스트 레이어가 있는 경우에만 잘 작동
# DOCX 파싱: pythons-docx가 가장 안정적
# docx.Document : 문단 단위 접근 가능

from docx import Document
import pdfplumber

def extract_text_from_docx(filepath: str) -> str:
    doc = Document(filepath)
    return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])

def extract_text_from_pdf(filepath: str) -> str:
    text = ""
    with pdfplumber.open(filepath) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text.strip()
