from typing import Optional
from api.domain.models.chat import ChatCompletionRequest, ChatCompletionResponse
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
            await self.cache.set(cache_key, response.dict(), ttl=3600)
            logger.debug(f"Cached response for key: {cache_key}")
        
        return response
    
    def _generate_cache_key(self, request: ChatCompletionRequest) -> str:
        """캐시 키 생성"""
        key_data = json.dumps(
            request.dict(exclude={"user", "n"} if request.n == 1 else {"user"}),
            sort_keys=True,
            default=str
        )
        return f"chat:{hashlib.md5(key_data.encode()).hexdigest()}"
