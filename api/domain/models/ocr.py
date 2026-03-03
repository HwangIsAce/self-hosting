from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from datetime import datetime
import uuid


class OCRRequest(BaseModel):
    """OCR 요청 모델"""
    image_url: Optional[str] = Field(None, description="이미지 URL")
    image_base64: Optional[str] = Field(None, description="Base64 인코딩된 이미지")
    prompt_type: Optional[str] = Field("ocr_layout", description="프롬프트 타입 (ocr_layout 등)")

    model_config = {
        "json_schema_extra": {
            "example": {
                "image_base64": "base64_encoded_image_string",
                "prompt_type": "ocr_layout"
            }
        }
    }


class OCRResponse(BaseModel):
    """OCR 응답 모델 - 항상 markdown, html, json 세 가지 형식을 모두 반환"""
    model_config = {"populate_by_name": True}
    id: str = Field(..., description="요청 ID")
    created: int = Field(..., description="생성 시간 (Unix timestamp)")
    markdown: str = Field(..., description="Markdown 형식 출력")
    html: str = Field(..., description="HTML 형식 출력")
    json_output: Dict[str, Any] = Field(..., alias="json", description="JSON 형식 구조화 출력")
    metadata: Optional[Dict[str, Any]] = Field(None, description="메타데이터")
