from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import asyncio
import time
from api.config.logging_config import get_logger

logger = get_logger(__name__)

MAX_CACHE_ENTRIES = 1000
CLEANUP_INTERVAL = 300  # 5분


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
    """캐시 어댑터 구현체 (인메모리 + TTL 백그라운드 정리)"""

    def __init__(self):
        # key -> (value, expiry_timestamp, last_access_timestamp)
        self._cache: Dict[str, tuple[Dict[str, Any], float, float]] = {}
        self._cleanup_task: Optional[asyncio.Task] = None
        self._start_cleanup()
        logger.info("CacheAdapter initialized (in-memory, max=%d entries)", MAX_CACHE_ENTRIES)

    def _start_cleanup(self):
        try:
            loop = asyncio.get_running_loop()
            self._cleanup_task = loop.create_task(self._cleanup_loop())
        except RuntimeError:
            pass  # 이벤트 루프 없으면 첫 요청 시 시작

    async def _cleanup_loop(self):
        """백그라운드 TTL 정리: 만료 항목 삭제."""
        while True:
            await asyncio.sleep(CLEANUP_INTERVAL)
            now = time.time()
            expired = [k for k, (_, exp, _) in self._cache.items() if now >= exp]
            for k in expired:
                del self._cache[k]
            if expired:
                logger.debug(f"Cache cleanup: removed {len(expired)} expired entries, {len(self._cache)} remaining")

    async def get(self, key: str) -> Optional[Dict[str, Any]]:
        """캐시에서 값 가져오기"""
        if key in self._cache:
            value, expiry, _ = self._cache[key]
            if time.time() < expiry:
                # LRU 갱신
                self._cache[key] = (value, expiry, time.time())
                return value
            else:
                del self._cache[key]
        return None

    async def set(self, key: str, value: Dict[str, Any], ttl: int = 3600) -> None:
        """캐시에 값 저장"""
        # 최대 크기 초과 시 가장 오래 접근되지 않은 항목 제거 (LRU)
        if len(self._cache) >= MAX_CACHE_ENTRIES and key not in self._cache:
            oldest_key = min(self._cache, key=lambda k: self._cache[k][2])
            del self._cache[oldest_key]
            logger.debug(f"Cache evicted LRU entry: {oldest_key}")

        now = time.time()
        self._cache[key] = (value, now + ttl, now)

    async def delete(self, key: str) -> None:
        """캐시에서 값 삭제"""
        self._cache.pop(key, None)
