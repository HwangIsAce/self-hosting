from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from datetime import datetime
import uuid


class OCRRequest(BaseModel):
    """OCR 요청 모델"""
    image_url: Optional[str] = Field(None, description="이미지 URL")
    image_base64: Optional[str] = Field(None, description="Base64 인코딩된 이미지")
    prompt_type: Optional[str] = Field("ocr_layout", description="프롬프트 타입 (ocr_layout 등)")
    output_format: Optional[str] = Field("markdown", description="출력 형식: markdown, html, json")
    
    class Config:
        json_schema_extra = {
            "example": {
                "image_base64": "base64_encoded_image_string",
                "prompt_type": "ocr_layout",
                "output_format": "markdown"
            }
        }


class OCRResponse(BaseModel):
    """OCR 응답 모델"""
    id: str = Field(..., description="요청 ID")
    created: int = Field(..., description="생성 시간 (Unix timestamp)")
    text: Optional[str] = Field(None, description="추출된 텍스트")
    markdown: Optional[str] = Field(None, description="Markdown 형식 출력")
    html: Optional[str] = Field(None, description="HTML 형식 출력")
    json_output: Optional[Dict[str, Any]] = Field(None, alias="json", description="JSON 형식 출력")
    metadata: Optional[Dict[str, Any]] = Field(None, description="메타데이터")
