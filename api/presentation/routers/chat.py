from fastapi import APIRouter, HTTPException, Depends
from api.domain.models.chat import ChatCompletionRequest, ChatCompletionResponse
from api.application.services.chat_service import ChatService
from api.presentation.dependencies.get_services import get_chat_service
from api.application.exceptions.service_exceptions import UnknownModelError
from api.application.exceptions.runpod_exceptions import (
    RunPodConnectionError,
    RunPodTimeoutError,
    RunPodServiceError,
)
from api.config.logging_config import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.post("/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(
    request: ChatCompletionRequest,
    service: ChatService = Depends(get_chat_service)
) -> ChatCompletionResponse:
    """
    Chat Completions 엔드포인트 (OpenAI API 호환)
    
    모델별로 자동으로 적절한 RunPod Pod로 라우팅됩니다:
    - qwen-llm* → LLM Pod (RTX 4090 #1)
    - qwen-vlm* → VLM Pod (RTX 4090 #2)
    """
    try:
        logger.info(f"Chat completion request: model={request.model}, messages={len(request.messages)}")
        return await service.create_completion(request)
    except UnknownModelError as e:
        logger.warning(f"Unknown model: {request.model}")
        raise HTTPException(status_code=400, detail=str(e))
    except RunPodConnectionError as e:
        logger.error(f"RunPod connection error: {str(e)}")
        raise HTTPException(status_code=503, detail=f"Service unavailable: {str(e)}")
    except RunPodTimeoutError as e:
        logger.error(f"RunPod timeout: {str(e)}")
        raise HTTPException(status_code=504, detail=f"Request timeout: {str(e)}")
    except RunPodServiceError as e:
        logger.error(f"RunPod service error: {str(e)}")
        raise HTTPException(status_code=502, detail=f"Service error: {str(e)}")
    except Exception as e:
        logger.exception(f"Unexpected error in chat completion: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")
