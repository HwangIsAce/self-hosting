"""API Key별 인메모리 Rate Limiting 미들웨어 (슬라이딩 윈도우)."""

import time
from collections import defaultdict
from fastapi import Request, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from api.config.settings import settings
from api.config.logging_config import get_logger

logger = get_logger(__name__)

# 경로 프리픽스 → (분당 허용 요청 수)
_RATE_LIMITS = {
    "/v1/ocr": 10,
    "/v1/documents": 10,
    "/v1/chat/completions": 30,
    "/v1/completions": 30,
    "/v1/models": 60,
}
_DEFAULT_RATE = 30  # 분당

# API Key별 요청 타임스탬프: key -> [timestamp, ...]
_request_log: dict[str, list[float]] = defaultdict(list)

_WINDOW = 60.0  # 1분 슬라이딩 윈도우


def _get_rate_limit(path: str) -> int:
    for prefix, limit in _RATE_LIMITS.items():
        if path.startswith(prefix):
            return limit
    return _DEFAULT_RATE


def _clean_old_entries(entries: list[float], now: float) -> list[float]:
    """윈도우 밖의 오래된 항목 제거."""
    cutoff = now - _WINDOW
    # bisect 대신 간단한 필터 (요청량이 적으므로 성능 충분)
    return [t for t in entries if t > cutoff]


class RateLimitMiddleware(BaseHTTPMiddleware):
    """API Key별 슬라이딩 윈도우 Rate Limiter."""

    async def dispatch(self, request: Request, call_next):
        # Rate limiting은 /v1 경로에만 적용
        if not request.url.path.startswith("/v1"):
            return await call_next(request)

        # API Key 식별 (auth 미들웨어가 먼저 실행됨)
        key = getattr(request.state, "api_key", None) or (
            request.client.host if request.client else "unknown"
        )

        limit = _get_rate_limit(request.url.path)
        now = time.time()

        entries = _request_log[key]
        entries = _clean_old_entries(entries, now)
        _request_log[key] = entries

        if len(entries) >= limit:
            retry_after = int(_WINDOW - (now - entries[0])) + 1
            logger.warning(
                f"Rate limit exceeded: key={key[:8]}..., "
                f"path={request.url.path}, limit={limit}/min"
            )
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                headers={"Retry-After": str(retry_after)},
                content={
                    "error": {
                        "message": f"Rate limit exceeded: {limit} requests per minute. Retry after {retry_after}s.",
                        "type": "rate_limit_error",
                        "param": None,
                        "code": "rate_limit_exceeded",
                    }
                },
            )

        entries.append(now)
        return await call_next(request)
