

# lang = "eng+kor"로 영어, 한글 병용 문서 대응
# pdf2image는 빠르고 오류 적음

import pytesseract
from pdf2image import convert_from_path
from PIL import Image

def extract_text_from_scanned_pdf(path: str) -> str:
    images = convert_from_path(path, dpi=300)
    texts = []
    for i, img in enumerate(images):
        print(f"OCR processing page {i + 1}/{len(images)}...")
        try:
            text = pytesseract.image_to_string(img, lang="eng+kor")
            texts.append(text)
        except pytesseract.TesseractNotFoundError:
            raise RuntimeError("Tesseract is not installed or not configured properly.")
    return "\n".join(texts).strip()

def extract_text_from_image(image_path: str) -> str:
    img = Image.open(image_path)
    try:
        text = pytesseract.image_to_string(img, lang="eng+kor")
        return text.strip()
    except pytesseract.TesseractNotFoundError:
        raise RuntimeError("Tesseract is not installed or not configured properly.")
