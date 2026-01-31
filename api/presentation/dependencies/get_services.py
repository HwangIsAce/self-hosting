from functools import lru_cache
from typing import Optional

from api.application.services.chat_service import ChatService
from api.application.services.document_service import DocumentService
from api.application.services.ocr_service import OCRService
from api.application.services.model_router_service import ModelRouterService
from api.infrastructure.adapters.cache_adapter import CacheAdapter, ICacheAdapter
from api.config.settings import settings


@lru_cache()
def get_model_router_service() -> ModelRouterService:
    """Model Router Service 싱글톤"""
    return ModelRouterService()


def get_chat_service() -> ChatService:
    """Chat Service 생성"""
    cache: Optional[ICacheAdapter] = None
    if settings.ENABLE_CACHE:
        cache = CacheAdapter()
    
    return ChatService(
        model_router=get_model_router_service(),
        cache=cache
    )


def get_document_service() -> DocumentService:
    """Document Service 생성"""
    return DocumentService(
        model_router=get_model_router_service()
    )


def get_ocr_service() -> OCRService:
    """OCR Service 생성"""
    return OCRService(
        model_router=get_model_router_service()
    )
