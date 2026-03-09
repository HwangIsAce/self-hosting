from fastapi import Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from api.config.logging_config import get_logger
from api.infrastructure.utils.gpu2_lock import GPU2LockTimeout

logger = get_logger(__name__)


def _is_cuda_oom(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "cuda" in msg and ("out of memory" in msg or "oom" in msg)


def _is_cuda_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "cuda" in msg or "nccl" in msg or "cublas" in msg


def _try_clear_gpu_cache():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Pydantic 검증 오류 핸들러"""
    errors = []
    for error in exc.errors():
        error_dict = {
            "type": error.get("type", "unknown"),
            "loc": error.get("loc", []),
            "msg": error.get("msg", ""),
        }
        if "ctx" in error and error["ctx"]:
            ctx = error["ctx"].copy()
            if "error" in ctx and isinstance(ctx["error"], Exception):
                ctx["error"] = str(ctx["error"])
            error_dict["ctx"] = ctx
        errors.append(error_dict)

    logger.warning(f"Validation error: {errors}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": {
                "message": "Validation error",
                "type": "invalid_request_error",
                "param": None,
                "code": None,
                "details": errors
            }
        }
    )


async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """HTTP 예외 핸들러"""
    logger.warning(f"HTTP exception: {exc.status_code} - {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "message": exc.detail,
                "type": "invalid_request_error",
                "param": None,
                "code": None
            }
        }
    )


async def general_exception_handler(request: Request, exc: Exception):
    """일반 예외 핸들러 — GPU 에러를 분류하여 적절한 상태 코드 반환"""
    error_msg = str(exc)

    # GPU 2 락 타임아웃 → 503 + Retry-After
    if isinstance(exc, GPU2LockTimeout):
        logger.warning(f"GPU 2 lock timeout: {error_msg}")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            headers={"Retry-After": "30"},
            content={
                "error": {
                    "message": error_msg,
                    "type": "server_busy",
                    "param": None,
                    "code": "gpu_busy",
                }
            },
        )

    # CUDA OOM → 503 + GPU 캐시 정리 + Retry-After
    if _is_cuda_oom(exc):
        logger.error(f"CUDA OOM: {error_msg}")
        _try_clear_gpu_cache()
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            headers={"Retry-After": "30"},
            content={
                "error": {
                    "message": "GPU out of memory. Request has been cleared. Please retry after a moment.",
                    "type": "gpu_error",
                    "param": None,
                    "code": "cuda_oom",
                }
            },
        )

    # 기타 CUDA 에러 → 503 + Retry-After
    if _is_cuda_error(exc):
        logger.error(f"CUDA error: {error_msg}")
        _try_clear_gpu_cache()
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            headers={"Retry-After": "60"},
            content={
                "error": {
                    "message": "GPU error occurred. Please retry.",
                    "type": "gpu_error",
                    "param": None,
                    "code": "cuda_error",
                }
            },
        )

    # 타임아웃 → 504
    if "timeout" in error_msg.lower():
        logger.error(f"Request timeout: {error_msg}")
        return JSONResponse(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            content={
                "error": {
                    "message": "Request timed out. Try with shorter input or retry.",
                    "type": "timeout_error",
                    "param": None,
                    "code": "timeout",
                }
            },
        )

    # 기타 → 500
    logger.exception(f"Unhandled exception: {error_msg}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": {
                "message": "An unexpected error occurred",
                "type": "internal_server_error",
                "param": None,
                "code": None
            }
        }
    )
