from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Optional, List, Literal
from datetime import datetime
import uuid


class ChatMessage(BaseModel):
    """채팅 메시지 (OpenAI 스타일)"""
    role: Literal["system", "user", "assistant"] = Field(
        ..., description="메시지 역할"
    )
    content: str = Field(..., description="메시지 내용")
    
    # VLM을 위한 이미지 지원
    image_url: Optional[str] = Field(None, description="이미지 URL (VLM용)")
    image_base64: Optional[str] = Field(None, description="Base64 이미지 (VLM용)")
    
    @field_validator('content')
    @classmethod
    def content_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Content cannot be empty')
        return v.strip()
    
    @model_validator(mode='after')
    def validate_image_fields(self):
        """이미지는 user 메시지에만 허용"""
        if (self.image_url or self.image_base64) and self.role != 'user':
            raise ValueError('Image can only be in user messages')
        return self
    
    class Config:
        json_schema_extra = {
            "example": {
                "role": "user",
                "content": "안녕하세요!"
            }
        }


class ChatCompletionRequest(BaseModel):
    """Chat Completions 요청 (OpenAI 스타일)"""
    model: str = Field(
        ..., 
        description="모델 이름: 'qwen-llm', 'qwen-vlm', 'qwen-llm-7b' 등"
    )
    messages: List[ChatMessage] = Field(..., description="대화 메시지 리스트")
    
    # 생성 파라미터
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0, description="온도 파라미터")
    top_p: Optional[float] = Field(1.0, ge=0.0, le=1.0, description="Top-p 샘플링")
    top_k: Optional[int] = Field(None, ge=1, description="Top-k 샘플링")
    max_tokens: Optional[int] = Field(None, ge=1, description="최대 토큰 수")
    stop: Optional[List[str]] = Field(None, description="중지 시퀀스")
    stream: Optional[bool] = Field(False, description="스트리밍 응답")
    
    # 기타
    n: Optional[int] = Field(1, ge=1, le=10, description="생성할 응답 수")
    presence_penalty: Optional[float] = Field(0.0, ge=-2.0, le=2.0)
    frequency_penalty: Optional[float] = Field(0.0, ge=-2.0, le=2.0)
    user: Optional[str] = Field(None, description="사용자 ID")
    
    @field_validator('messages')
    @classmethod
    def validate_messages(cls, v):
        if not v:
            raise ValueError('Messages cannot be empty')
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "model": "qwen-llm-7b",
                "messages": [
                    {"role": "user", "content": "안녕하세요!"}
                ],
                "temperature": 0.7,
                "max_tokens": 1000
            }
        }


class ChatCompletionChoice(BaseModel):
    """Chat Completion 선택지"""
    index: int = Field(..., description="선택지 인덱스")
    message: ChatMessage = Field(..., description="응답 메시지")
    finish_reason: Optional[str] = Field(
        None, 
        description="완료 이유: 'stop', 'length', 'content_filter'"
    )


class Usage(BaseModel):
    """토큰 사용량"""
    prompt_tokens: int = Field(..., description="프롬프트 토큰 수")
    completion_tokens: int = Field(..., description="완성 토큰 수")
    total_tokens: int = Field(..., description="전체 토큰 수")


class ChatCompletionResponse(BaseModel):
    """Chat Completions 응답 (OpenAI 스타일)"""
    id: str = Field(
        default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:8]}",
        description="응답 ID"
    )
    object: str = Field("chat.completion", description="객체 타입")
    created: int = Field(
        default_factory=lambda: int(datetime.now().timestamp()),
        description="생성 시간 (Unix timestamp)"
    )
    model: str = Field(..., description="사용된 모델 이름")
    choices: List[ChatCompletionChoice] = Field(..., description="생성된 선택지")
    usage: Usage = Field(..., description="토큰 사용량")
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "chatcmpl-123",
                "object": "chat.completion",
                "created": 1677652288,
                "model": "qwen-llm-7b",
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "안녕하세요! 무엇을 도와드릴까요?"
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": 9,
                    "completion_tokens": 12,
                    "total_tokens": 21
                }
            }
        }
