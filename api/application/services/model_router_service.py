from typing import Dict, Any, Optional
from api.domain.models.chat import (
    ChatCompletionRequest, 
    ChatCompletionResponse,
    ChatCompletionChoice,
    ChatMessage,
    Usage
)
from api.infrastructure.clients.runpod_client import (
    RunPodClientFactory,
    IRunPodClient
)
from api.config.settings import settings
from api.application.exceptions.service_exceptions import UnknownModelError
from api.config.logging_config import get_logger

logger = get_logger(__name__)


class ModelRouterService:
    """모델별 요청을 적절한 RunPod Pod로 라우팅"""
    
    def __init__(self):
        self.llm_client = None
        self.vlm_client = None
        self.docling_client = None
    
    def _get_pod_type(self, model: str) -> str:
        """모델 이름으로 Pod 타입 결정"""
        model_lower = model.lower()
        for model_name, pod_type in settings.MODEL_TO_POD.items():
            if model_name in model_lower:
                return pod_type
        raise UnknownModelError(f"Unknown model: {model}")
    
    def _get_client(self, pod_type: str) -> IRunPodClient:
        """Pod 타입에 따라 클라이언트 반환"""
        if pod_type == "llm":
            if self.llm_client is None:
                self.llm_client = RunPodClientFactory.get_llm_client()
            return self.llm_client
        elif pod_type == "vlm":
            if self.vlm_client is None:
                self.vlm_client = RunPodClientFactory.get_vlm_client()
            return self.vlm_client
        elif pod_type == "docling":
            if self.docling_client is None:
                self.docling_client = RunPodClientFactory.get_docling_client()
            return self.docling_client
        else:
            raise UnknownModelError(f"Unsupported pod type: {pod_type}")
    
    async def route_chat_completion(
        self, 
        request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        """Chat completion 요청을 적절한 Pod로 라우팅"""
        pod_type = self._get_pod_type(request.model)
        client = self._get_client(pod_type)
        
        # OpenAI 형식을 RunPod Pod 형식으로 변환
        payload = self._convert_to_runpod_format(request)
        
        logger.info(f"Routing chat completion to {pod_type} pod for model {request.model}")
        
        # RunPod Pod 호출
        response = await client.post("/v1/chat/completions", json=payload)
        
        # RunPod 응답을 OpenAI 형식으로 변환
        return self._convert_to_openai_format(request.model, response)
    
    def _convert_to_runpod_format(
        self, 
        request: ChatCompletionRequest
    ) -> Dict[str, Any]:
        """OpenAI 요청 형식을 RunPod Pod 형식으로 변환"""
        payload = {
            "messages": [msg.dict(exclude_none=True) for msg in request.messages],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "top_p": request.top_p,
        }
        
        if request.top_k:
            payload["top_k"] = request.top_k
        if request.stop:
            payload["stop"] = request.stop
        if request.n and request.n > 1:
            payload["n"] = request.n
        if request.presence_penalty:
            payload["presence_penalty"] = request.presence_penalty
        if request.frequency_penalty:
            payload["frequency_penalty"] = request.frequency_penalty
        
        return payload
    
    def _convert_to_openai_format(
        self, 
        model: str, 
        runpod_response: Dict[str, Any]
    ) -> ChatCompletionResponse:
        """RunPod 응답을 OpenAI 형식으로 변환"""
        choices = []
        for idx, choice in enumerate(runpod_response.get("choices", [])):
            message_data = choice.get("message", {})
            choices.append(ChatCompletionChoice(
                index=idx,
                message=ChatMessage(
                    role=message_data.get("role", "assistant"),
                    content=message_data.get("content", "")
                ),
                finish_reason=choice.get("finish_reason", "stop")
            ))
        
        usage_data = runpod_response.get("usage", {})
        usage = Usage(
            prompt_tokens=usage_data.get("prompt_tokens", 0),
            completion_tokens=usage_data.get("completion_tokens", 0),
            total_tokens=usage_data.get("total_tokens", 0)
        )
        
        return ChatCompletionResponse(
            model=model,
            choices=choices,
            usage=usage
        )
    
    async def route_document_processing(
        self,
        file_base64: str,
        file_type: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """문서 처리 요청을 Docling Pod로 라우팅"""
        client = self._get_client("docling")
        
        payload = {
            "file_base64": file_base64,
            "file_type": file_type,
        }
        
        if options:
            payload["options"] = options
        
        logger.info(f"Routing document processing to docling pod")
        
        return await client.post("/v1/process", json=payload)
