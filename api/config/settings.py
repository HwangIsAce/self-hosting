from pydantic_settings import BaseSettings
from typing import Optional, List


class Settings(BaseSettings):
    """RunPod 배포를 위한 설정"""
    
    # API Gateway 서버 설정
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False
    
    # RunPod Pod 엔드포인트 (각 Pod의 고유 URL)
    RUNPOD_LLM_ENDPOINT: str = ""
    RUNPOD_VLM_ENDPOINT: str = ""
    RUNPOD_DOCLING_ENDPOINT: str = ""
    
    # RunPod API 키 (선택사항, Pod 보호용)
    RUNPOD_API_KEY: Optional[str] = None
    
    # HTTP 클라이언트 설정
    REQUEST_TIMEOUT: int = 300  # 5분 (LLM 생성 시간 고려)
    MAX_RETRIES: int = 3
    RETRY_DELAY: float = 1.0
    
    # 연결 풀 설정
    MAX_CONNECTIONS: int = 100
    MAX_KEEPALIVE_CONNECTIONS: int = 20
    
    # 모델 매핑 (모델 이름 → Pod 타입)
    MODEL_TO_POD: dict = {
        "qwen-llm": "llm",
        "qwen-llm-7b": "llm",
        "qwen-llm-14b": "llm",
        "qwen-vlm": "vlm",
        "qwen-vlm-7b": "vlm",
        "qwen-vlm-14b": "vlm",
    }
    
    # CORS 설정
    CORS_ORIGINS: List[str] = ["*"]
    
    # 로깅
    LOG_LEVEL: str = "INFO"
    
    # 캐싱 (선택사항)
    ENABLE_CACHE: bool = False
    CACHE_URL: Optional[str] = None
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
