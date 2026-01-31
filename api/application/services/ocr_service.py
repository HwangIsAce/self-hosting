from typing import Dict, Any, Optional
from api.application.services.model_router_service import ModelRouterService
from api.config.logging_config import get_logger

logger = get_logger(__name__)


class OCRService:
    """OCR 서비스 (Chandra 모델 사용)"""
    
    def __init__(
        self,
        model_router: ModelRouterService
    ):
        self.model_router = model_router
    
    async def process_ocr(
        self,
        image_base64: Optional[str] = None,
        image_url: Optional[str] = None,
        prompt_type: Optional[str] = "ocr_layout",
        output_format: Optional[str] = "markdown"
    ) -> Dict[str, Any]:
        """OCR 처리 (Chandra 모델 사용)"""
        if not image_base64 and not image_url:
            raise ValueError("Either image_base64 or image_url must be provided")
        
        logger.info(f"Processing OCR: prompt_type={prompt_type}, output_format={output_format}")
        
        # file_url이 제공된 경우, 나중에 클라이언트에서 처리하거나
        # 여기서 다운로드하여 base64로 변환할 수 있음
        # 현재는 image_base64만 지원
        if image_url:
            # TODO: URL에서 이미지 다운로드 및 base64 변환
            raise NotImplementedError("image_url processing not yet implemented")
        
        # Chandra OCR 모델로 라우팅 (같은 Pod이지만 다른 엔드포인트)
        result = await self.model_router.route_ocr_processing(
            image_base64=image_base64,
            prompt_type=prompt_type,
            output_format=output_format
        )
        
        return result
