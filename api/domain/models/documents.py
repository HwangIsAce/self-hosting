from pydantic import BaseModel, Field, model_validator
from typing import Optional, Dict, Any
from datetime import datetime
import uuid


class DocumentProcessRequest(BaseModel):
    """문서 처리 요청"""
    file_url: Optional[str] = Field(None, description="파일 URL")
    file_base64: Optional[str] = Field(None, description="Base64 인코딩된 파일")
    file_type: Optional[str] = Field(
        None, 
        description="파일 타입: pdf, docx, txt, etc."
    )
    options: Optional[Dict[str, Any]] = Field(
        None, 
        description="처리 옵션"
    )
    
    @model_validator(mode='after')
    def validate_file_source(self):
        """파일 소스 중 하나는 필수"""
        if not self.file_url and not self.file_base64:
            raise ValueError("Either file_url or file_base64 must be provided")
        return self


class DocumentProcessResponse(BaseModel):
    """문서 처리 응답 - 항상 markdown, html, json 세 가지 형식을 모두 반환"""
    id: str = Field(
        default_factory=lambda: f"doc-{uuid.uuid4().hex[:8]}",
        description="처리 ID"
    )
    object: str = Field("document.process", description="객체 타입")
    created: int = Field(
        default_factory=lambda: int(datetime.now().timestamp()),
        description="생성 시간 (Unix timestamp)"
    )
    text: str = Field(..., description="추출된 텍스트 (markdown과 동일)")
    markdown: str = Field(..., description="Markdown 형식 출력")
    html: str = Field(..., description="HTML 형식 출력")
    json_output: Dict[str, Any] = Field(..., alias="json", description="JSON 형식 구조화 출력")
    metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="문서 메타데이터 (페이지 수, 파일명 등)"
    )
    structure: Optional[Dict[str, Any]] = Field(
        None,
        description="문서 구조 (표/이미지 개수 등)"
    )

    model_config = {
        "populate_by_name": True,
        "json_schema_extra": {
            "example": {
                "id": "doc-123",
                "object": "document.process",
                "created": 1677652288,
                "text": "추출된 텍스트 내용...",
                "markdown": "# 제목\n\n본문...",
                "html": "<h1>제목</h1><p>본문...</p>",
                "json": {"texts": [], "tables": []},
                "metadata": {"page_count": 10, "file_name": "document.pdf"},
                "structure": {"sections": ["Introduction", "Main Content", "Conclusion"]}
            }
        }
    }
