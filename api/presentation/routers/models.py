from fastapi import APIRouter
from api.domain.models.models import ModelsListResponse, ModelInfo
from datetime import datetime

router = APIRouter()


@router.get("/models", response_model=ModelsListResponse)
async def list_models() -> ModelsListResponse:
    """
    사용 가능한 모델 목록 조회 (OpenAI API 호환)
    
    각 모델은 로컬 엔진 또는 RunPod Pod에 매핑됩니다:
    - qwen-llm-7b / qwen-llm-14b: LLM 엔진 (GPU #1)
    - qwen-vlm-7b / qwen-vlm-14b: VLM 엔진 (GPU #2)
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
