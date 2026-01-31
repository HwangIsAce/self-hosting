from fastapi import Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from api.config.logging_config import get_logger

logger = get_logger(__name__)


async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Pydantic 검증 오류 핸들러"""
    # errors를 JSON 직렬화 가능한 형태로 변환
    errors = []
    for error in exc.errors():
        error_dict = {
            "type": error.get("type", "unknown"),
            "loc": error.get("loc", []),
            "msg": error.get("msg", ""),
        }
        # ctx에 ValueError가 있는 경우 문자열로 변환
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
    """일반 예외 핸들러"""
    logger.exception(f"Unhandled exception: {str(exc)}")
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
