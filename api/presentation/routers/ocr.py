from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Form
from api.domain.models.ocr import OCRRequest, OCRResponse
from api.application.services.ocr_service import OCRService
from api.presentation.dependencies.get_services import get_ocr_service
from datetime import datetime
import uuid
import base64
from typing import Optional
from api.config.logging_config import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.post("/ocr", response_model=OCRResponse)
async def process_ocr(
    image: Optional[UploadFile] = File(None),
    image_url: Optional[str] = Form(None),
    prompt_type: Optional[str] = Form("ocr_layout"),
    output_format: Optional[str] = Form("markdown"),
    max_tokens: Optional[int] = Form(1024),  # OCR에 적합한 기본값 (정확도와 속도의 균형)
    service: OCRService = Depends(get_ocr_service)
) -> OCRResponse:
    """
    OCR 엔드포인트 (Chandra 모델 사용)
    
    Docling과 Chandra는 같은 서버(GPU 2)에 있지만 각각 독립적으로 호출됩니다.
    - Docling: /v1/documents/process 엔드포인트로 별도 호출 (문서 처리)
    - Chandra: 이 엔드포인트로 호출 (OCR 처리)
    
    - Chandra: Hugging Face OCR 모델 (https://huggingface.co/datalab-to/chandra)
    - 출력 형식: markdown, html, json
    
    지원 이미지 형식: PNG, JPG, JPEG, PDF 등
    """
    try:
        # 이미지 소스 확인
        if not image and not image_url:
            raise ValueError("Either image file or image_url must be provided")
        
        image_base64 = None
        if image:
            logger.info(f"OCR request: filename={image.filename}, content_type={image.content_type}")
            image_content = await image.read()
            image_base64 = base64.b64encode(image_content).decode()
        elif image_url:
            logger.info(f"OCR request: image_url={image_url}")
            # TODO: URL에서 이미지 다운로드 및 base64 변환
            raise NotImplementedError("image_url processing not yet implemented")
        
        # OCR 처리
        result = await service.process_ocr(
            image_base64=image_base64,
            image_url=image_url,
            prompt_type=prompt_type,
            output_format=output_format,
            max_tokens=max_tokens or 1024
        )
        
        # 응답 생성
        return OCRResponse(
            id=f"ocr-{uuid.uuid4().hex[:8]}",
            created=int(datetime.now().timestamp()),
            text=result.get("text"),
            markdown=result.get("markdown") if output_format == "markdown" else None,
            html=result.get("html") if output_format == "html" else None,
            json_output=result.get("json") if output_format == "json" else None,
            metadata=result.get("metadata")
        )
    except ValueError as e:
        logger.warning(f"Invalid request: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except NotImplementedError as e:
        logger.warning(f"Not implemented: {str(e)}")
        raise HTTPException(status_code=501, detail=str(e))
    except Exception as e:
        logger.exception(f"Unexpected error in OCR processing: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")
