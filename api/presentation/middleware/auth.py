"""API Key 인증 미들웨어"""

from fastapi import Request, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from api.config.settings import settings
from api.config.logging_config import get_logger

logger = get_logger(__name__)

# 인증 제외 경로
PUBLIC_PATHS = {"/", "/health", "/docs", "/redoc", "/openapi.json"}


class APIKeyAuthMiddleware(BaseHTTPMiddleware):
    """X-API-Key 헤더 기반 인증 미들웨어"""

    async def dispatch(self, request: Request, call_next):
        # 공개 경로는 인증 제외
        if request.url.path in PUBLIC_PATHS:
            return await call_next(request)

        # API_KEYS가 비어있으면 인증 비활성화 (개발 모드)
        if not settings.API_KEYS:
            return await call_next(request)

        api_key = request.headers.get("X-API-Key") or request.headers.get("Authorization", "").removeprefix("Bearer ").strip()

        if not api_key or api_key not in settings.API_KEYS:
            logger.warning(
                f"Unauthorized request: {request.method} {request.url.path} "
                f"from {request.client.host if request.client else 'unknown'}"
            )
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={
                    "error": {
                        "message": "Invalid or missing API key. Provide a valid key via X-API-Key header or Authorization: Bearer <key>.",
                        "type": "authentication_error",
                        "param": None,
                        "code": "invalid_api_key",
                    }
                },
            )

        # 요청 state에 인증 정보 저장 (로깅/rate limiting에서 활용)
        request.state.api_key = api_key

        return await call_next(request)
