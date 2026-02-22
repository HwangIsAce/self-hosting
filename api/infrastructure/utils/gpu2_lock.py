"""GPU 2 공용 락 — Docling, Chandra(OCR), ColPali가 GPU 2를 직렬 사용하도록 합니다.

동시에 여러 엔진이 GPU 2에 로드되지 않도록, 모든 GPU 2 작업은 이 락을 획득한 뒤 실행합니다.
한 시점에 한 모델만 로드하도록, 락 진입 시 다른 엔진을 언로드합니다.
"""

import asyncio
from typing import Optional, Callable, Dict

# GPU 2 작업 직렬화용 락. Docling / OCR / ColPali 엔진이 추론 전에 이 락을 획득합니다.
gpu2_lock = asyncio.Lock()

# 현재 GPU 2에 로드된 엔진 식별자. "docling" | "ocr" | "colpali" | None
_current_gpu2_engine: Optional[str] = None

# 엔진별 언로드 콜백 (name -> callable). unload_other_gpu2_engines에서 호출.
_engine_unloaders: Dict[str, Callable[[], None]] = {}


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


def unload_other_gpu2_engines(except_name: Optional[str] = None) -> None:
    """except_name을 제외한 모든 등록된 엔진을 언로드합니다. 락 획득 직후 호출."""
    for name, fn in list(_engine_unloaders.items()):
        if name != except_name:
            try:
                fn()
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(f"GPU 2 engine unload {name} failed: {e}")
