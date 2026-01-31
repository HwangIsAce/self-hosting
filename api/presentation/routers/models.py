from fastapi import APIRouter
from api.domain.models.models import ModelsListResponse, ModelInfo
from datetime import datetime

router = APIRouter()


@router.get("/models", response_model=ModelsListResponse)
async def list_models() -> ModelsListResponse:
    """
    사용 가능한 모델 목록 조회 (OpenAI API 호환)
    
    각 모델은 RunPod Pod에 매핑됩니다:
    - qwen-llm-7b: LLM Pod (RTX 4090 #1)
    - qwen-llm-14b: LLM Pod (RTX 4090 #1)
    - qwen-vlm-7b: VLM Pod (RTX 4090 #2)
    - qwen-vlm-14b: VLM Pod (RTX 4090 #2)
    """
    models = [
        ModelInfo(
            id="qwen-llm-7b",
            created=int(datetime.now().timestamp()),
            owned_by="self-hosting",
            description="Qwen LLM 7B on RunPod RTX 4090"
        ),
        ModelInfo(
            id="qwen-llm-14b",
            created=int(datetime.now().timestamp()),
            owned_by="self-hosting",
            description="Qwen LLM 14B on RunPod RTX 4090"
        ),
        ModelInfo(
            id="qwen-vlm-7b",
            created=int(datetime.now().timestamp()),
            owned_by="self-hosting",
            description="Qwen VLM 7B on RunPod RTX 4090"
        ),
        ModelInfo(
            id="qwen-vlm-14b",
            created=int(datetime.now().timestamp()),
            owned_by="self-hosting",
            description="Qwen VLM 14B on RunPod RTX 4090"
        ),
    ]
    
    return ModelsListResponse(data=models)
