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
        output_format: Optional[str] = "markdown",
        max_tokens: int = 1024
    ) -> Dict[str, Any]:
        """OCR 처리 (Chandra 모델 사용). 현재는 이미지 파일 업로드(image_base64)만 지원합니다."""
        if not image_base64 and not image_url:
            raise ValueError("Either image_base64 or image_url must be provided")
        
        logger.info(f"Processing OCR: prompt_type={prompt_type}, output_format={output_format}")
        
        if image_url:
            raise NotImplementedError("image_url은 현재 미지원입니다. 이미지 파일을 업로드해 주세요.")
        
        # Chandra OCR 모델로 라우팅 (같은 Pod이지만 다른 엔드포인트)
        result = await self.model_router.route_ocr_processing(
            image_base64=image_base64,
            prompt_type=prompt_type,
            output_format=output_format,
            max_tokens=max_tokens
        )
        
        return result
