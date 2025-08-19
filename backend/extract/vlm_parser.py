import os
from openai import OpenAI
from PIL import Image
from dotenv import load_dotenv
from base64 import b64encode


# OCR로 추출 불가능한 복잡한 다이어그램/표/구조 분석
# base64 이미지 전송 + 유연한 프롬프트 구성

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def vlm_extract_caption_bytes(img_bytes: bytes, prompt: str) -> str:
    """이미지 바이트를 바로 VLM에 전달."""
    encoded = b64encode(img_bytes).decode("utf-8")
    res = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}}
            ],
        }],
        max_tokens=2048,
    )
    return (res.choices[0].message.content or "").strip()

def vlm_extract_caption(image_path: str, prompt: str = "이 이미지는 어떤 내용을 전달하고 있으며, 구조적 특징, 도식의 목적, 핵심 정보, 시사점 또는 전략적 함의를 요약 및 분석해 주세요. 단순 묘사가 아니라 해석을 중심으로 설명해 주세요.") -> str:
    with open(image_path, "rb") as image_file:
        encoded = b64encode(image_file.read()).decode("utf-8")
    res = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user", 
                    "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}}
                ]}
            ],
            max_tokens=2048
        )
    return res.choices[0].message.content.strip()
