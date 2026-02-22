from typing import Dict, Any, Optional
from api.domain.models.chat import (
    ChatCompletionRequest, 
    ChatCompletionResponse,
    ChatCompletionChoice,
    ChatMessage,
    Usage
)
from api.config.settings import settings
from api.application.exceptions.service_exceptions import UnknownModelError
from api.config.logging_config import get_logger

# 로컬 모델 엔진
from api.infrastructure.models.llm_engine import LLMEngine
from api.infrastructure.models.vllm_engine import VLLMEngine  # vLLM 엔진 추가
from api.infrastructure.models.vlm_engine import VLMEngine
from api.infrastructure.models.ocr_engine import OCREngine
from api.infrastructure.models.docling_engine import DoclingEngine

logger = get_logger(__name__)


class ModelRouterService:
    """모델별 요청을 적절한 로컬 모델 엔진으로 라우팅"""
    
    def __init__(self):
        self.llm_engine = None  # HuggingFace 엔진 (백업용)
        self.vllm_engine = None  # vLLM 엔진 (새로 추가)
        self.vlm_engine = None
        self.ocr_engine = None
        self.docling_engine = None
    
    def _get_pod_type(self, model: str) -> str:
        """모델 이름으로 Pod 타입 결정"""
        model_lower = model.lower()
        for model_name, pod_type in settings.MODEL_TO_POD.items():
            if model_name in model_lower:
                return pod_type
        raise UnknownModelError(f"Unknown model: {model}")
    
    def _get_engine(self, pod_type: str):
        """Pod 타입에 따라 모델 엔진 반환"""
        if pod_type == "llm":
            # vLLM 사용 여부 확인
            if settings.USE_VLLM:
                if self.vllm_engine is None:
                    device_map = f"cuda:{settings.LLM_GPU_ID}"
                    logger.info(f"Initializing vLLM engine for LLM on {device_map}")
                    self.vllm_engine = VLLMEngine(
                        model_name=settings.LLM_MODEL_NAME,
                        device_map=device_map,
                        use_quantization=settings.USE_QUANTIZATION
                    )
                return self.vllm_engine
            else:
                # 백업: 기존 HuggingFace 엔진
                if self.llm_engine is None:
                    device_map = f"cuda:{settings.LLM_GPU_ID}"
                    logger.info(f"Initializing HuggingFace engine for LLM on {device_map}")
                    self.llm_engine = LLMEngine(
                        model_name=settings.LLM_MODEL_NAME,
                        device_map=device_map,
                        use_quantization=settings.USE_QUANTIZATION
                    )
                return self.llm_engine
        elif pod_type == "vlm":
            if self.vlm_engine is None:
                device_map = f"cuda:{settings.VLM_GPU_ID}"
                self.vlm_engine = VLMEngine(
                    model_name=settings.VLM_MODEL_NAME,
                    device_map=device_map,
                    use_quantization=settings.USE_QUANTIZATION
                )
            return self.vlm_engine
        elif pod_type == "docling":
            # Docling 엔진: 문서 처리 (PDF, DOCX 등)
            # Chandra와 같은 GPU 2에 있지만 독립적으로 호출됨
            if self.docling_engine is None:
                self.docling_engine = DoclingEngine()
            return self.docling_engine
        elif pod_type == "ocr":
            # Chandra OCR 엔진: 이미지 OCR 처리
            # Docling과 같은 GPU 2에 있지만 독립적으로 호출됨
            if self.ocr_engine is None:
                device_map = f"cuda:{settings.OCR_GPU_ID}"
                self.ocr_engine = OCREngine(
                    model_name=settings.OCR_MODEL_NAME,
                    device_map=device_map
                )
            return self.ocr_engine
        else:
            raise UnknownModelError(f"Unsupported pod type: {pod_type}")
    
    async def route_chat_completion(
        self, 
        request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        """Chat completion 요청을 적절한 로컬 모델 엔진으로 라우팅"""
        pod_type = self._get_pod_type(request.model)
        engine = self._get_engine(pod_type)
        
        logger.info(f"Routing chat completion to {pod_type} engine for model {request.model}")
        
        # 메시지 변환
        messages = [msg.model_dump(exclude_none=True) for msg in request.messages]
        
        # 모델 엔진으로 직접 생성
        if pod_type == "llm":
            # vLLM 엔진인지 확인 (동일한 인터페이스이므로 그대로 사용 가능)
            response = await engine.generate(
                messages=messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                top_p=request.top_p,
                top_k=request.top_k,
                stop=request.stop
            )
        elif pod_type == "vlm":
            response = await engine.generate(
                messages=messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                top_p=request.top_p
            )
        else:
            raise UnknownModelError(f"Unsupported pod type for chat: {pod_type}")
        
        # OpenAI 형식으로 변환
        return self._convert_to_openai_format(request.model, response)
    
    def _convert_to_runpod_format(
        self, 
        request: ChatCompletionRequest
    ) -> Dict[str, Any]:
        """OpenAI 요청 형식을 RunPod Pod 형식으로 변환 (RunPod 모드 사용 시 활용). 현재는 로컬 엔진만 사용하므로 호출되지 않음."""
        payload = {
            "messages": [msg.model_dump(exclude_none=True) for msg in request.messages],
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
        """백엔드(로컬 엔진 또는 RunPod) 응답을 OpenAI 형식으로 변환"""
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
        """
        문서 처리 요청을 로컬 Docling 엔진으로 라우팅
        
        Docling과 Chandra는 같은 서버(GPU 2)에 있지만 각각 독립적으로 호출됩니다.
        - Docling: 문서 처리 (PDF, DOCX, TXT 등)
        - Chandra: OCR 처리 (별도 엔드포인트 /v1/ocr로 호출)
        """
        engine = self._get_engine("docling")
        
        logger.info("Routing document processing to local Docling engine (independent from Chandra OCR)")
        
        return await engine.process_document(
            file_base64=file_base64,
            file_type=file_type,
            options=options
        )
    
    async def route_ocr_processing(
        self,
        image_base64: str,
        prompt_type: Optional[str] = "ocr_layout",
        output_format: Optional[str] = "markdown",
        max_tokens: int = 1024
    ) -> Dict[str, Any]:
        """
        OCR 처리 요청을 로컬 Chandra OCR 엔진으로 라우팅
        
        Docling과 Chandra는 같은 서버(GPU 2)에 있지만 각각 독립적으로 호출됩니다.
        - Docling: 문서 처리 (별도 엔드포인트 /v1/documents/process로 호출)
        - Chandra: OCR 처리 (이 엔드포인트)
        """
        engine = self._get_engine("ocr")
        
        logger.info("Routing OCR processing to local Chandra OCR engine (independent from Docling)")
        
        return await engine.process(
            image_base64=image_base64,
            prompt_type=prompt_type,
            output_format=output_format,
            max_tokens=max_tokens
        )