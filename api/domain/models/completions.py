from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
import uuid


class CompletionRequest(BaseModel):
    """Text Completion 요청 (레거시 OpenAI API)"""
    model: str = Field(..., description="모델 이름")
    prompt: str | List[str] = Field(..., description="프롬프트")
    suffix: Optional[str] = Field(None, description="접미사")
    max_tokens: Optional[int] = Field(16, ge=1, description="최대 토큰 수")
    temperature: Optional[float] = Field(1.0, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(1.0, ge=0.0, le=1.0)
    n: Optional[int] = Field(1, ge=1, le=10, description="생성할 응답 수")
    stream: Optional[bool] = Field(False, description="스트리밍 응답")
    logprobs: Optional[int] = Field(None, ge=0, le=5, description="로그 확률 수")
    echo: Optional[bool] = Field(False, description="프롬프트 반환 여부")
    stop: Optional[List[str]] = Field(None, description="중지 시퀀스")
    presence_penalty: Optional[float] = Field(0.0, ge=-2.0, le=2.0)
    frequency_penalty: Optional[float] = Field(0.0, ge=-2.0, le=2.0)
    best_of: Optional[int] = Field(1, ge=1, description="최선의 응답 수")
    user: Optional[str] = Field(None, description="사용자 ID")


class CompletionChoice(BaseModel):
    """Completion 선택지"""
    text: str = Field(..., description="생성된 텍스트")
    index: int = Field(..., description="선택지 인덱스")
    logprobs: Optional[Dict[str, Any]] = Field(None, description="로그 확률")
    finish_reason: Optional[str] = Field(
        None,
        description="완료 이유: 'stop', 'length', 'content_filter'"
    )


class CompletionUsage(BaseModel):
    """Completion 토큰 사용량"""
    prompt_tokens: int = Field(..., description="프롬프트 토큰 수")
    completion_tokens: int = Field(..., description="완성 토큰 수")
    total_tokens: int = Field(..., description="전체 토큰 수")


class CompletionResponse(BaseModel):
    """Text Completion 응답 (레거시 OpenAI API)"""
    id: str = Field(
        default_factory=lambda: f"cmpl-{uuid.uuid4().hex[:8]}",
        description="응답 ID"
    )
    object: str = Field("text_completion", description="객체 타입")
    created: int = Field(
        default_factory=lambda: int(datetime.now().timestamp()),
        description="생성 시간 (Unix timestamp)"
    )
    model: str = Field(..., description="사용된 모델 이름")
    choices: List[CompletionChoice] = Field(..., description="생성된 선택지")
    usage: CompletionUsage = Field(..., description="토큰 사용량")
