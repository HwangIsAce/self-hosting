"""GPU 2 공용 락 — Docling, Chandra(OCR), ColPali가 GPU 2를 직렬 사용하도록 합니다.

동시에 여러 엔진이 GPU 2에 로드되지 않도록, 모든 GPU 2 작업은 이 락을 획득한 뒤 실행합니다.
한 시점에 한 모델만 로드하도록, 락 진입 시 다른 엔진을 언로드합니다.
"""

import asyncio
import logging
from typing import Optional, Callable, Dict

logger = logging.getLogger(__name__)

# GPU 2 작업 직렬화용 락. Docling / OCR / ColPali 엔진이 추론 전에 이 락을 획득합니다.
_gpu2_lock = asyncio.Lock()

# 락 획득 타임아웃 (초). 이 시간 내에 획득하지 못하면 TimeoutError 발생.
GPU2_LOCK_TIMEOUT: float = 120.0

# 현재 GPU 2에 로드된 엔진 식별자. "docling" | "ocr" | "colpali" | None
_current_gpu2_engine: Optional[str] = None

# 엔진별 언로드 콜백 (name -> callable). unload_other_gpu2_engines에서 호출.
_engine_unloaders: Dict[str, Callable[[], None]] = {}


class GPU2LockTimeout(Exception):
    """GPU 2 락 획득 타임아웃"""
    pass


def get_current_gpu2_engine() -> Optional[str]:
    """현재 GPU 2에 로드된 엔진 식별자를 반환합니다."""
    return _current_gpu2_engine


def set_current_gpu2_engine(engine: Optional[str]) -> None:
    """현재 GPU 2에 로드된 엔진 식별자를 설정합니다. "docling" | "ocr" | "colpali" | None"""
    global _current_gpu2_engine
    _current_gpu2_engine = engine


def register_gpu2_engine(name: str, unload_callable: Callable[[], None]) -> None:
    """GPU 2 엔진의 언로드 콜백을 등록합니다. 각 엔진 __init__에서 호출."""
    _engine_unloaders[name] = unload_callable


def _force_clear_gpu2():
    """GPU 2 VRAM 강제 정리. 언로드 실패 시 최후 수단."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.warning("GPU 2 VRAM force-cleared via torch.cuda.empty_cache()")
    except Exception as e:
        logger.error(f"Failed to force-clear GPU 2: {e}")


def unload_other_gpu2_engines(except_name: Optional[str] = None) -> None:
    """except_name을 제외한 모든 등록된 엔진을 언로드합니다. 락 획득 직후 호출."""
    for name, fn in list(_engine_unloaders.items()):
        if name != except_name:
            try:
                fn()
            except Exception as e:
                logger.warning(f"GPU 2 engine unload {name} failed: {e}, retrying...")
                # 1회 재시도
                try:
                    fn()
                except Exception as e2:
                    logger.error(f"GPU 2 engine unload {name} retry failed: {e2}, force-clearing VRAM")
                    _force_clear_gpu2()


class gpu2_lock:
    """GPU 2 락 컨텍스트 매니저 (타임아웃 포함).

    Usage:
        async with gpu2_lock():
            unload_other_gpu2_engines(except_name="ocr")
            set_current_gpu2_engine("ocr")
            ...
    """

    def __init__(self, timeout: float = GPU2_LOCK_TIMEOUT):
        self.timeout = timeout

    async def __aenter__(self):
        try:
            await asyncio.wait_for(_gpu2_lock.acquire(), timeout=self.timeout)
        except asyncio.TimeoutError:
            logger.error(
                f"GPU 2 lock acquisition timed out after {self.timeout}s. "
                f"Current engine: {_current_gpu2_engine}"
            )
            raise GPU2LockTimeout(
                f"GPU 2 is busy (held by {_current_gpu2_engine or 'unknown'}). "
                f"Try again later."
            )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        _gpu2_lock.release()
        return False
