from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime


class ModelPermission(BaseModel):
    """모델 권한"""
    id: str = Field(..., description="권한 ID")
    object: str = Field("model_permission", description="객체 타입")
    created: int = Field(..., description="생성 시간")
    allow_create_engine: bool = Field(False, description="엔진 생성 허용")
    allow_sampling: bool = Field(True, description="샘플링 허용")
    allow_logprobs: bool = Field(True, description="로그 확률 허용")
    allow_search_indices: bool = Field(False, description="검색 인덱스 허용")
    allow_view: bool = Field(True, description="조회 허용")
    allow_fine_tuning: bool = Field(False, description="파인튜닝 허용")
    organization: str = Field("*", description="조직")
    group: Optional[str] = Field(None, description="그룹")
    is_blocking: bool = Field(False, description="차단 여부")


class ModelInfo(BaseModel):
    """모델 정보"""
    id: str = Field(..., description="모델 ID")
    object: str = Field("model", description="객체 타입")
    created: int = Field(
        default_factory=lambda: int(datetime.now().timestamp()),
        description="생성 시간 (Unix timestamp)"
    )
    owned_by: str = Field("self-hosting", description="소유자")
    permission: List[ModelPermission] = Field(
        default_factory=list,
        description="권한 리스트"
    )
    root: Optional[str] = Field(None, description="루트 모델")
    parent: Optional[str] = Field(None, description="부모 모델")
    description: Optional[str] = Field(None, description="모델 설명")
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "qwen-llm-7b",
                "object": "model",
                "created": 1677652288,
                "owned_by": "self-hosting",
                "permission": [],
                "description": "Qwen LLM 7B on RunPod RTX 4090"
            }
        }


class ModelsListResponse(BaseModel):
    """모델 목록 응답"""
    object: str = Field("list", description="객체 타입")
    data: List[ModelInfo] = Field(..., description="모델 리스트")
    
    class Config:
        json_schema_extra = {
            "example": {
                "object": "list",
                "data": [
                    {
                        "id": "qwen-llm-7b",
                        "object": "model",
                        "created": 1677652288,
                        "owned_by": "self-hosting"
                    }
                ]
            }
        }
