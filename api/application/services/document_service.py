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
        """문서 처리"""
        if not file_base64 and not file_url:
            raise ValueError("Either file_base64 or file_url must be provided")
        
        logger.info(f"Processing document: type={file_type}")
        
        # file_url이 제공된 경우, 나중에 클라이언트에서 처리하거나
        # 여기서 다운로드하여 base64로 변환할 수 있음
        # 현재는 file_base64만 지원
        if file_url:
            # TODO: URL에서 파일 다운로드 및 base64 변환
            raise NotImplementedError("file_url processing not yet implemented")
        
        # Docling 프레임워크 Pod로 라우팅 (필요시 Chandra OCR 사용)
        result = await self.model_router.route_document_processing(
            file_base64=file_base64,
            file_type=file_type,
            options=options
        )
        
        return result
