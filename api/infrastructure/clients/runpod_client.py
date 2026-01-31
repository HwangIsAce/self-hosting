from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from api.config.settings import settings
from api.application.exceptions.runpod_exceptions import (
    RunPodConnectionError,
    RunPodTimeoutError,
    RunPodServiceError,
    RunPodAuthenticationError,
)
from api.config.logging_config import get_logger

logger = get_logger(__name__)


class IRunPodClient(ABC):
    """RunPod 클라이언트 인터페이스"""
    
    @abstractmethod
    async def post(self, endpoint: str, json: Dict[str, Any]) -> Dict[str, Any]:
        """POST 요청"""
        pass
    
    @abstractmethod
    async def get(self, endpoint: str) -> Dict[str, Any]:
        """GET 요청"""
        pass
    
    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """Pod 헬스체크"""
        pass


class RunPodClient(IRunPodClient):
    """RunPod Pod와 통신하는 클라이언트"""
    
    def __init__(
        self, 
        base_url: str, 
        api_key: Optional[str] = None,
        timeout: Optional[int] = None
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or settings.RUNPOD_API_KEY
        self.timeout = timeout or settings.REQUEST_TIMEOUT
        
        # HTTP 클라이언트 설정 (연결 풀 포함)
        limits = httpx.Limits(
            max_connections=settings.MAX_CONNECTIONS,
            max_keepalive_connections=settings.MAX_KEEPALIVE_CONNECTIONS
        )
        
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.timeout,
            limits=limits,
            headers=headers
        )
        
        logger.info(f"RunPodClient initialized for {self.base_url}")
    
    async def post(self, endpoint: str, json: Dict[str, Any]) -> Dict[str, Any]:
        """POST 요청 (재시도 로직 포함)"""
        # 런타임에 settings를 읽어서 retry 데코레이터를 동적으로 적용
        from api.config.settings import settings as runtime_settings
        retry_decorator = retry(
            stop=stop_after_attempt(runtime_settings.MAX_RETRIES),
            wait=wait_exponential(multiplier=1, min=runtime_settings.RETRY_DELAY, max=10)
        )
        
        @retry_decorator
        async def _post_with_retry(endpoint: str, json: Dict[str, Any]) -> Dict[str, Any]:
            try:
                logger.debug(f"POST {endpoint} to {self.base_url}")
                response = await self.client.post(endpoint, json=json)
                response.raise_for_status()
                return response.json()
            except httpx.TimeoutException as e:
                logger.error(f"Request timeout: {str(e)}")
                raise RunPodTimeoutError(f"Request timeout: {str(e)}")
            except httpx.ConnectError as e:
                logger.error(f"Connection error: {str(e)}")
                raise RunPodConnectionError(f"Connection error: {str(e)}")
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 401:
                    logger.error("Authentication failed")
                    raise RunPodAuthenticationError(f"Authentication failed: {e.response.text}")
                logger.error(f"HTTP error {e.response.status_code}: {e.response.text}")
                raise RunPodServiceError(
                    f"HTTP error {e.response.status_code}: {e.response.text}"
                )
            except Exception as e:
                logger.error(f"Unexpected error: {str(e)}")
                raise RunPodServiceError(f"Unexpected error: {str(e)}")
        
        return await _post_with_retry(endpoint, json)
    
    async def get(self, endpoint: str) -> Dict[str, Any]:
        """GET 요청"""
        try:
            logger.debug(f"GET {endpoint} from {self.base_url}")
            response = await self.client.get(endpoint)
            response.raise_for_status()
            return response.json()
        except httpx.TimeoutException as e:
            raise RunPodTimeoutError(f"Request timeout: {str(e)}")
        except httpx.ConnectError as e:
            raise RunPodConnectionError(f"Connection error: {str(e)}")
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                raise RunPodAuthenticationError(f"Authentication failed: {e.response.text}")
            raise RunPodServiceError(
                f"HTTP error {e.response.status_code}: {e.response.text}"
            )
        except Exception as e:
            raise RunPodServiceError(f"Unexpected error: {str(e)}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Pod 헬스체크"""
        try:
            # RunPod Pod의 헬스체크 엔드포인트 (실제 엔드포인트에 맞게 수정 필요)
            result = await self.get("/health")
            return {"status": "healthy", **result}
        except Exception as e:
            logger.warning(f"Health check failed: {str(e)}")
            return {"status": "unhealthy", "error": str(e)}
    
    async def close(self):
        """클라이언트 종료"""
        await self.client.aclose()
        logger.info(f"RunPodClient closed for {self.base_url}")


class RunPodClientFactory:
    """RunPod 클라이언트 팩토리 (싱글톤 패턴)"""
    
    _llm_client: Optional[RunPodClient] = None
    _vlm_client: Optional[RunPodClient] = None
    _docling_client: Optional[RunPodClient] = None
    
    @classmethod
    def get_llm_client(cls) -> RunPodClient:
        """LLM Pod 클라이언트 (싱글톤)"""
        if cls._llm_client is None:
            if not settings.RUNPOD_LLM_ENDPOINT:
                raise ValueError("RUNPOD_LLM_ENDPOINT is not set")
            cls._llm_client = RunPodClient(
                base_url=settings.RUNPOD_LLM_ENDPOINT,
                api_key=settings.RUNPOD_API_KEY
            )
        return cls._llm_client
    
    @classmethod
    def get_vlm_client(cls) -> RunPodClient:
        """VLM Pod 클라이언트 (싱글톤)"""
        if cls._vlm_client is None:
            if not settings.RUNPOD_VLM_ENDPOINT:
                raise ValueError("RUNPOD_VLM_ENDPOINT is not set")
            cls._vlm_client = RunPodClient(
                base_url=settings.RUNPOD_VLM_ENDPOINT,
                api_key=settings.RUNPOD_API_KEY
            )
        return cls._vlm_client
    
    @classmethod
    def get_docling_client(cls) -> RunPodClient:
        """Docling Pod 클라이언트 (싱글톤)"""
        if cls._docling_client is None:
            if not settings.RUNPOD_DOCLING_ENDPOINT:
                raise ValueError("RUNPOD_DOCLING_ENDPOINT is not set")
            cls._docling_client = RunPodClient(
                base_url=settings.RUNPOD_DOCLING_ENDPOINT,
                api_key=settings.RUNPOD_API_KEY
            )
        return cls._docling_client
    
    @classmethod
    async def close_all(cls):
        """모든 클라이언트 종료"""
        if cls._llm_client:
            await cls._llm_client.close()
            cls._llm_client = None
        if cls._vlm_client:
            await cls._vlm_client.close()
            cls._vlm_client = None
        if cls._docling_client:
            await cls._docling_client.close()
            cls._docling_client = None
        logger.info("All RunPod clients closed")
