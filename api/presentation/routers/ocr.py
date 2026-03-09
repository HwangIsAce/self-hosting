from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Form
from fastapi.encoders import jsonable_encoder
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
    max_tokens: Optional[int] = Form(1024),
    service: OCRService = Depends(get_ocr_service)
) -> OCRResponse:
    """
    OCR 엔드포인트 (Chandra 모델 사용)
    항상 markdown, html, json 세 가지 형식을 모두 반환합니다.
    - Chandra: Hugging Face OCR 모델 (https://huggingface.co/datalab-to/chandra)
    - 지원 이미지 형식: PNG, JPG, JPEG, PDF 등
    """
    try:
        if not image and not image_url:
            raise ValueError("Either image file or image_url must be provided")

        image_base64 = None
        if image:
            logger.info(f"OCR request: filename={image.filename}, content_type={image.content_type}")
            image_content = await image.read()
            max_upload_bytes = 500 * 1024 * 1024  # 500MB
            if len(image_content) > max_upload_bytes:
                raise HTTPException(
                    status_code=413,
                    detail=f"Image too large: {len(image_content)} bytes. Maximum allowed: {max_upload_bytes} bytes (500MB)."
                )
            image_base64 = base64.b64encode(image_content).decode()
        elif image_url:
            logger.info(f"OCR request: image_url={image_url}")
            raise NotImplementedError("image_url은 현재 미지원입니다. 이미지 파일을 업로드해 주세요.")

        result = await service.process_ocr(
            image_base64=image_base64,
            image_url=image_url,
            prompt_type=prompt_type,
            max_tokens=max_tokens or 1024
        )

        return OCRResponse(
            id=f"ocr-{uuid.uuid4().hex[:8]}",
            created=int(datetime.now().timestamp()),
            markdown=result["markdown"],
            html=result["html"],
            json_output=jsonable_encoder(result["json"]),
            metadata=jsonable_encoder(result["metadata"]) if result.get("metadata") is not None else None
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
