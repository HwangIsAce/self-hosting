from fastapi import APIRouter, HTTPException, Depends
from api.domain.models.completions import CompletionRequest, CompletionResponse
from api.application.services.model_router_service import ModelRouterService
from api.presentation.dependencies.get_services import get_model_router_service
from api.application.exceptions.service_exceptions import UnknownModelError
from api.application.exceptions.runpod_exceptions import (
    RunPodConnectionError,
    RunPodTimeoutError,
)
from api.config.logging_config import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.post("/completions", response_model=CompletionResponse)
async def create_completion(
    request: CompletionRequest,
    router: ModelRouterService = Depends(get_model_router_service)
) -> CompletionResponse:
    """
    레거시 Text Completions 엔드포인트 (OpenAI 호환).
    현재 미구현이며, /v1/chat/completions 사용을 권장합니다.
    """
    try:
        logger.info(f"Completion request: model={request.model}")
        raise HTTPException(
            status_code=501,
            detail="Text completions is not implemented. Please use POST /v1/chat/completions instead."
        )
    except UnknownModelError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except (RunPodConnectionError, RunPodTimeoutError) as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.exception(f"Unexpected error in completion: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
