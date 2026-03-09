from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn
import multiprocessing

# CUDA와 multiprocessing 호환성을 위해 'spawn' start method 설정
# vLLM이 multiprocessing을 사용하기 전에 반드시 설정해야 함
try:
    if multiprocessing.get_start_method(allow_none=True) != 'spawn':
        multiprocessing.set_start_method('spawn', force=True)
except (RuntimeError, ValueError):
    # 이미 설정되었거나 설정할 수 없는 경우 무시
    pass

from api.presentation.routers import chat, completions, documents, models, ocr
from api.presentation.middleware.error_handler import (
    validation_exception_handler,
    http_exception_handler,
    general_exception_handler
)
from api.presentation.middleware.logging import LoggingMiddleware
from api.presentation.middleware.auth import APIKeyAuthMiddleware
from api.presentation.middleware.rate_limiter import RateLimitMiddleware
from api.config.settings import settings
from api.config.logging_config import setup_logging
from api.infrastructure.utils.idle_watcher import idle_watcher_loop
# RunPod 클라이언트는 선택사항 (원격 Pod 사용 시)
# from api.infrastructure.clients.runpod_client import RunPodClientFactory
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

# 로깅 설정
setup_logging()
from api.config.logging_config import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """애플리케이션 생명주기 관리"""
    # Startup: 로컬 모델 초기화
    logger.info("Starting API Gateway with local models...")
    
    try:
        # 모델 설정 로그
        logger.info(f"LLM Model: {settings.LLM_MODEL_NAME} on GPU {settings.LLM_GPU_ID}")
        logger.info(f"VLM Model: {settings.VLM_MODEL_NAME} on GPU {settings.VLM_GPU_ID}")
        logger.info(f"OCR Model: {settings.OCR_MODEL_NAME} on GPU {settings.OCR_GPU_ID}")
        
        # 프리로딩 설정에 따라 모델 미리 로드
        if settings.PRELOAD_LLM or settings.PRELOAD_VLM:
            from api.presentation.dependencies.get_services import get_model_router_service
            router = get_model_router_service()
            if settings.PRELOAD_LLM:
                try:
                    logger.info("Pre-loading LLM model...")
                    engine = router._get_engine("llm")
                    engine.load_model()
                    logger.info("LLM model pre-loaded successfully")
                except Exception as e:
                    logger.error(f"LLM pre-load failed (will retry on first request): {e}")
            if settings.PRELOAD_VLM:
                try:
                    logger.info("Pre-loading VLM model...")
                    engine = router._get_engine("vlm")
                    engine.load_model()
                    logger.info("VLM model pre-loaded successfully")
                except Exception as e:
                    logger.error(f"VLM pre-load failed (will retry on first request): {e}")
        else:
            logger.info("Models will be loaded on first request")
    except Exception as e:
        logger.warning(f"Could not initialize model settings: {str(e)}")

    # 유휴 감지 백그라운드 태스크 시작
    idle_task = asyncio.create_task(idle_watcher_loop())

    yield

    idle_task.cancel()
    
    # Shutdown: 모델 정리
    logger.info("Shutting down API Gateway...")
    # 모델 언로드는 ModelLoader가 자동으로 처리
    logger.info("API Gateway stopped")


app = FastAPI(
    title="Self-Hosting API",
    description="OpenAI-compatible API for Qwen LLM (HF), Qwen VLM (HF), and Docling Framework on local GPU or RunPod RTX 4090",
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

# Rate limiting (인증 이후에 적용)
app.add_middleware(RateLimitMiddleware)

# 인증 미들웨어 (CORS 다음, rate limit 이전)
app.add_middleware(APIKeyAuthMiddleware)

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
    """헬스체크 엔드포인트 (각 엔진 및 GPU 상태 포함)"""
    result = {"status": "healthy", "engines": {}, "gpu": []}

    # 엔진 상태
    try:
        from api.infrastructure.models.ocr_engine import OCREngine
        result["engines"]["ocr"] = bool(
            OCREngine._model is not None
            and getattr(OCREngine, "_model_loaded", False)
        )
    except Exception:
        result["engines"]["ocr"] = False

    try:
        from api.presentation.dependencies.get_services import get_model_router_service
        router = get_model_router_service()
        result["engines"]["llm"] = router.vllm_engine is not None and getattr(router.vllm_engine, "_loaded", False)
        result["engines"]["vlm"] = router.vlm_engine is not None and getattr(router.vlm_engine, "_loaded", False)
    except Exception:
        pass

    try:
        from api.infrastructure.utils.gpu2_lock import get_current_gpu2_engine
        result["engines"]["gpu2_current"] = get_current_gpu2_engine()
    except Exception:
        pass

    # GPU 메모리 정보
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                free, total = torch.cuda.mem_get_info(i)
                used = total - free
                result["gpu"].append({
                    "id": i,
                    "name": torch.cuda.get_device_name(i),
                    "used_mb": round(used / 1024 / 1024),
                    "total_mb": round(total / 1024 / 1024),
                    "utilization_pct": round(used / total * 100, 1) if total > 0 else 0,
                })
    except Exception:
        pass

    return result


if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )
