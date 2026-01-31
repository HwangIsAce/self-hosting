from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn

from api.presentation.routers import chat, completions, documents, models, ocr
from api.presentation.middleware.error_handler import (
    validation_exception_handler,
    http_exception_handler,
    general_exception_handler
)
from api.presentation.middleware.logging import LoggingMiddleware
from api.config.settings import settings
from api.config.logging_config import setup_logging
from api.infrastructure.clients.runpod_client import RunPodClientFactory
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

# 로깅 설정
setup_logging()
from api.config.logging_config import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """애플리케이션 생명주기 관리"""
    # Startup: RunPod Pod 연결 확인
    logger.info("Starting API Gateway...")
    
    # RunPod 클라이언트 초기화 (필요시)
    # 실제 Pod 연결은 첫 요청 시 lazy initialization
    try:
        if settings.RUNPOD_LLM_ENDPOINT:
            logger.info(f"LLM Pod endpoint: {settings.RUNPOD_LLM_ENDPOINT}")
        if settings.RUNPOD_VLM_ENDPOINT:
            logger.info(f"VLM Pod endpoint: {settings.RUNPOD_VLM_ENDPOINT}")
        if settings.RUNPOD_DOCLING_ENDPOINT:
            logger.info(f"Docling/Chandra Pod endpoint: {settings.RUNPOD_DOCLING_ENDPOINT}")
    except Exception as e:
        logger.warning(f"Could not initialize RunPod clients: {str(e)}")
    
    yield
    
    # Shutdown: 클라이언트 종료
    logger.info("Shutting down API Gateway...")
    await RunPodClientFactory.close_all()
    logger.info("API Gateway stopped")


app = FastAPI(
    title="Self-Hosting API",
    description="OpenAI-compatible API for Qwen LLM (HF), Qwen VLM (HF), and Docling Framework on RunPod RTX 4090",
    version="1.0.0",
    lifespan=lifespan
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 로깅 미들웨어
app.add_middleware(LoggingMiddleware)

# 예외 핸들러 등록
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(StarletteHTTPException, http_exception_handler)
app.add_exception_handler(Exception, general_exception_handler)

# 라우터 등록
app.include_router(chat.router, prefix="/v1", tags=["Chat"])
app.include_router(completions.router, prefix="/v1", tags=["Completions"])
app.include_router(documents.router, prefix="/v1", tags=["Documents"])
app.include_router(ocr.router, prefix="/v1", tags=["OCR"])
app.include_router(models.router, prefix="/v1", tags=["Models"])


@app.get("/")
async def root():
    """루트 엔드포인트"""
    return {
        "service": "Self-Hosting API",
        "version": "1.0.0",
        "openai_compatible": True,
        "endpoints": {
            "chat": "/v1/chat/completions",
            "completions": "/v1/completions",
            "documents": "/v1/documents/process",
            "ocr": "/v1/ocr",
            "models": "/v1/models"
        }
    }


@app.get("/health")
async def health():
    """헬스체크 엔드포인트"""
    return {"status": "healthy"}


if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )
