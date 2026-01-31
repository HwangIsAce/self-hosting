from fastapi import APIRouter, HTTPException, Depends, UploadFile, File
from api.domain.models.documents import DocumentProcessResponse
from api.application.services.document_service import DocumentService
from api.presentation.dependencies.get_services import get_document_service
from datetime import datetime
import uuid
import base64
from api.config.logging_config import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.post("/documents/process", response_model=DocumentProcessResponse)
async def process_document(
    file: UploadFile = File(...),
    service: DocumentService = Depends(get_document_service)
) -> DocumentProcessResponse:
    """
    문서 처리 엔드포인트
    
    Docling 프레임워크를 사용하는 Pod (RTX 4090 #3)로 요청을 전달합니다.
    Docling은 필요시 Chandra OCR 모델을 사용하여 문서를 처리합니다.
    
    - Docling: 문서 처리 프레임워크 (https://docling-project.github.io/docling/)
    - Chandra: Hugging Face OCR 모델 (https://huggingface.co/datalab-to/chandra)
    
    지원 파일 형식: PDF, DOCX, TXT, 이미지 등
    """
    try:
        logger.info(f"Document processing request: filename={file.filename}, content_type={file.content_type}")
        
        # 파일 읽기
        file_content = await file.read()
        file_base64 = base64.b64encode(file_content).decode()
        file_type = file.filename.split(".")[-1] if file.filename else None
        
        # 문서 처리
        result = await service.process_document(
            file_base64=file_base64,
            file_type=file_type
        )
        
        # 응답 생성
        return DocumentProcessResponse(
            id=f"doc-{uuid.uuid4().hex[:8]}",
            created=int(datetime.now().timestamp()),
            text=result.get("text", ""),
            metadata=result.get("metadata"),
            structure=result.get("structure")
        )
    except ValueError as e:
        logger.warning(f"Invalid request: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"Unexpected error in document processing: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")
