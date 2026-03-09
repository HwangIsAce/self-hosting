from fastapi import APIRouter, HTTPException, Depends, UploadFile, File
from fastapi.encoders import jsonable_encoder
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
    문서 처리 엔드포인트 (Docling 엔진)
    
    현재는 파일 업로드만 지원합니다. file_url은 미지원입니다.
    - Docling: 문서 처리 프레임워크 (https://docling-project.github.io/docling/)
    - 지원 파일 형식: PDF, DOCX, TXT, 이미지 등
    """
    try:
        logger.info(f"Document processing request: filename={file.filename}, content_type={file.content_type}")
        
        # 파일 읽기 + 크기 제한 (50MB)
        file_content = await file.read()
        max_upload_bytes = 50 * 1024 * 1024  # 50MB
        if len(file_content) > max_upload_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"File too large: {len(file_content)} bytes. Maximum allowed: {max_upload_bytes} bytes (50MB)."
            )
        file_base64 = base64.b64encode(file_content).decode()
        file_type = file.filename.split(".")[-1] if file.filename else None
        
        # 문서 처리
        result = await service.process_document(
            file_base64=file_base64,
            file_type=file_type
        )
        
        # 응답 생성 (json/metadata/structure는 JSON 직렬화 가능하도록 보정)
        return DocumentProcessResponse(
            id=f"doc-{uuid.uuid4().hex[:8]}",
            created=int(datetime.now().timestamp()),
            text=result.get("text", ""),
            markdown=result.get("markdown", ""),
            html=result.get("html", ""),
            json_output=jsonable_encoder(result.get("json", {})),
            metadata=jsonable_encoder(result.get("metadata")) if result.get("metadata") is not None else None,
            structure=jsonable_encoder(result.get("structure")) if result.get("structure") is not None else None
        )
    except ValueError as e:
        logger.warning(f"Invalid request: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"Unexpected error in document processing: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)[:200]}")
