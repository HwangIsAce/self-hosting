from typing import Optional, Dict, Any
from api.application.services.model_router_service import ModelRouterService
from api.config.logging_config import get_logger

logger = get_logger(__name__)


class DocumentService:
    """문서 처리 서비스"""
    
    def __init__(
        self,
        model_router: ModelRouterService
    ):
        self.model_router = model_router
    
    async def process_document(
        self,
        file_base64: Optional[str] = None,
        file_url: Optional[str] = None,
        file_type: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """문서 처리. 현재는 파일 업로드(file_base64)만 지원합니다."""
        if not file_base64 and not file_url:
            raise ValueError("Either file_base64 or file_url must be provided")
        
        logger.info(f"Processing document: type={file_type}")
        
        if file_url:
            raise NotImplementedError("file_url은 현재 미지원입니다. 파일을 업로드해 주세요.")
        
        # Docling 프레임워크 Pod로 라우팅 (필요시 Chandra OCR 사용)
        result = await self.model_router.route_document_processing(
            file_base64=file_base64,
            file_type=file_type,
            options=options
        )
        
        return result
