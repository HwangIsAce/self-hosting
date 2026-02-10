from typing import Optional, List
import asyncio
from api.domain.models.chat import (
    ChatCompletionRequest, 
    ChatCompletionResponse,
    BatchChatCompletionResponse
)
from api.application.services.model_router_service import ModelRouterService
from api.infrastructure.adapters.cache_adapter import CacheAdapter
from api.config.logging_config import get_logger
import hashlib
import json

logger = get_logger(__name__)


class ChatService:
    """Chat Completion 서비스"""
    
    def __init__(
        self,
        model_router: ModelRouterService,
        cache: Optional[CacheAdapter] = None
    ):
        self.model_router = model_router
        self.cache = cache
    
    async def create_completion(
        self, 
        request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        """Chat completion 생성"""
        # 캐싱 (스트리밍이 아닌 경우만)
        if self.cache and not request.stream:
            cache_key = self._generate_cache_key(request)
            cached = await self.cache.get(cache_key)
            if cached:
                logger.debug(f"Cache hit for key: {cache_key}")
                return ChatCompletionResponse(**cached)
        
        # 모델 라우팅 및 RunPod Pod 호출
        logger.info(f"Creating chat completion for model: {request.model}")
        response = await self.model_router.route_chat_completion(request)
        
        # 캐싱 저장 (스트리밍이 아닌 경우만)
        if self.cache and not request.stream:
            cache_key = self._generate_cache_key(request)
            await self.cache.set(cache_key, response.model_dump(), ttl=3600)
            logger.debug(f"Cached response for key: {cache_key}")
        
        return response
    
    async def create_batch_completion(
        self,
        requests: List[ChatCompletionRequest]
    ) -> BatchChatCompletionResponse:
        """배치 chat completion 생성"""
        logger.info(f"Creating batch chat completion for {len(requests)} requests")
        
        # 모든 요청을 동시에 처리
        tasks = [self.create_completion(req) for req in requests]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 응답 처리
        processed_responses = []
        successful = 0
        failed = 0
        
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                logger.error(f"Request {i} failed: {str(response)}")
                failed += 1
                # 에러 응답 생성
                from api.domain.models.chat import ChatCompletionChoice, ChatMessage, Usage
                processed_responses.append(ChatCompletionResponse(
                    id=f"chatcmpl-error-{i}",
                    model=requests[i].model,
                    choices=[ChatCompletionChoice(
                        index=0,
                        message=ChatMessage(
                            role="assistant",
                            content=f"Error: {str(response)}"
                        ),
                        finish_reason="error"
                    )],
                    usage=Usage(
                        prompt_tokens=0,
                        completion_tokens=0,
                        total_tokens=0
                    )
                ))
            else:
                successful += 1
                processed_responses.append(response)
        
        return BatchChatCompletionResponse(
            responses=processed_responses,
            total_requests=len(requests),
            successful_requests=successful,
            failed_requests=failed
        )
    
    def _generate_cache_key(self, request: ChatCompletionRequest) -> str:
        """캐시 키 생성"""
        key_data = json.dumps(
            request.model_dump(exclude={"user", "n"} if request.n == 1 else {"user"}),
            sort_keys=True,
            default=str
        )
        return f"chat:{hashlib.md5(key_data.encode()).hexdigest()}"
