import logging
import logging.handlers
import json
import sys
import os
from datetime import datetime, timezone
from api.config.settings import settings


class JSONFormatter(logging.Formatter):
    """구조화 JSON 로그 포맷터."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info and record.exc_info[0] is not None:
            log_entry["exception"] = self.formatException(record.exc_info)
        # 추가 필드 (request_id 등)
        for key in ("request_id", "user", "method", "path", "status", "duration_ms"):
            val = getattr(record, key, None)
            if val is not None:
                log_entry[key] = val
        return json.dumps(log_entry, ensure_ascii=False, default=str)


def setup_logging():
    """로깅 설정: JSON 포맷 + 로그 로테이션."""
    log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
    use_json = os.getenv("LOG_FORMAT", "text").lower() == "json"

    root = logging.getLogger()
    root.setLevel(log_level)
    # 기존 핸들러 제거
    root.handlers.clear()

    # stdout 핸들러
    stream_handler = logging.StreamHandler(sys.stdout)
    if use_json:
        stream_handler.setFormatter(JSONFormatter())
    else:
        stream_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
    root.addHandler(stream_handler)

    # 파일 로테이션 핸들러 (LOG_FILE 환경변수가 설정된 경우)
    log_file = os.getenv("LOG_FILE")
    if log_file:
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=100 * 1024 * 1024,  # 100MB
            backupCount=7,
            encoding="utf-8",
        )
        file_handler.setFormatter(JSONFormatter())
        root.addHandler(file_handler)

    # 외부 라이브러리 로그 레벨 조정
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("uvicorn").setLevel(logging.INFO)

    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured: level={settings.LOG_LEVEL}, format={'json' if use_json else 'text'}")


def get_logger(name: str) -> logging.Logger:
    """로거 인스턴스 반환"""
    return logging.getLogger(name)
