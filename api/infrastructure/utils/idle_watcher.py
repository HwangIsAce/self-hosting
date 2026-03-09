"""유휴 감지 — N초간 요청 없으면 GPU 모델을 언로드하여 비용 절감."""

import asyncio
import time
from api.config.settings import settings
from api.config.logging_config import get_logger

logger = get_logger(__name__)

# 마지막 요청 타임스탬프 (LoggingMiddleware에서 갱신)
_last_request_time: float = time.time()


def touch_last_request():
    """요청이 들어올 때마다 호출하여 타임스탬프를 갱신합니다."""
    global _last_request_time
    _last_request_time = time.time()


def get_idle_seconds() -> float:
    """마지막 요청 이후 경과 시간(초)을 반환합니다."""
    return time.time() - _last_request_time


async def idle_watcher_loop():
    """백그라운드 태스크: 유휴 시간이 설정값을 초과하면 모델 언로드."""
    interval = settings.IDLE_UNLOAD_SECONDS
    if interval <= 0:
        return  # 비활성화

    logger.info(f"Idle watcher started: will unload models after {interval}s of inactivity")
    while True:
        await asyncio.sleep(60)  # 1분 간격 체크
        idle = get_idle_seconds()
        if idle >= interval:
            logger.info(f"Idle for {idle:.0f}s (threshold={interval}s), unloading GPU models")
            _unload_all_models()
            # 언로드 후 다음 요청까지 대기 (매번 체크하지 않도록)
            await asyncio.sleep(interval)


def _unload_all_models():
    """모든 GPU 모델을 언로드하여 VRAM 해제."""
    try:
        from api.presentation.dependencies.get_services import get_model_router_service
        router = get_model_router_service()

        if router.vllm_engine and getattr(router.vllm_engine, "_loaded", False):
            router.vllm_engine = None
            logger.info("vLLM engine unloaded")

        if router.vlm_engine and getattr(router.vlm_engine, "_loaded", False):
            router.vlm_engine = None
            logger.info("VLM engine unloaded")

        if router.ocr_engine:
            from api.infrastructure.models.ocr_engine import OCREngine
            OCREngine.unload_class()
            router.ocr_engine = None
            logger.info("OCR engine unloaded")

        if router.docling_engine:
            router.docling_engine._unload()
            router.docling_engine = None
            logger.info("Docling engine unloaded")

    except Exception as e:
        logger.error(f"Error unloading models: {e}")

    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
