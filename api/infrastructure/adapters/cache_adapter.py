from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from api.config.logging_config import get_logger

logger = get_logger(__name__)


class ICacheAdapter(ABC):
    """캐시 어댑터 인터페이스"""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Dict[str, Any]]:
        """캐시에서 값 가져오기"""
        pass
    
    @abstractmethod
    async def set(self, key: str, value: Dict[str, Any], ttl: int = 3600) -> None:
        """캐시에 값 저장"""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> None:
        """캐시에서 값 삭제"""
        pass


class CacheAdapter(ICacheAdapter):
    """캐시 어댑터 구현체 (현재는 메모리 캐시)"""
    
    def __init__(self):
        self._cache: Dict[str, tuple[Dict[str, Any], float]] = {}
        logger.info("CacheAdapter initialized (in-memory)")
    
    async def get(self, key: str) -> Optional[Dict[str, Any]]:
        """캐시에서 값 가져오기"""
        import time
        if key in self._cache:
            value, expiry = self._cache[key]
            if time.time() < expiry:
                logger.debug(f"Cache hit: {key}")
                return value
            else:
                # 만료된 항목 삭제
                del self._cache[key]
                logger.debug(f"Cache expired: {key}")
        return None
    
    async def set(self, key: str, value: Dict[str, Any], ttl: int = 3600) -> None:
        """캐시에 값 저장"""
        import time
        expiry = time.time() + ttl
        self._cache[key] = (value, expiry)
        logger.debug(f"Cache set: {key} (TTL: {ttl}s)")
    
    async def delete(self, key: str) -> None:
        """캐시에서 값 삭제"""
        if key in self._cache:
            del self._cache[key]
            logger.debug(f"Cache deleted: {key}")
