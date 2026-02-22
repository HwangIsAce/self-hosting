"""GPU 2 공용 락 — Docling, Chandra(OCR), ColPali가 GPU 2를 직렬 사용하도록 합니다.

동시에 여러 엔진이 GPU 2에 로드되지 않도록, 모든 GPU 2 작업은 이 락을 획득한 뒤 실행합니다.
"""

import asyncio
from typing import Optional

# GPU 2 작업 직렬화용 락. Docling / OCR / ColPali 엔진이 추론 전에 이 락을 획득합니다.
gpu2_lock = asyncio.Lock()

# 현재 GPU 2에 로드된 엔진 식별자. "docling" | "ocr" | "colpali" | None
_current_gpu2_engine: Optional[str] = None


def get_current_gpu2_engine() -> Optional[str]:
    """현재 GPU 2에 로드된 엔진 식별자를 반환합니다."""
    return _current_gpu2_engine


def set_current_gpu2_engine(engine: Optional[str]) -> None:
    """현재 GPU 2에 로드된 엔진 식별자를 설정합니다. "docling" | "ocr" | "colpali" | None"""
    global _current_gpu2_engine
    _current_gpu2_engine = engine
