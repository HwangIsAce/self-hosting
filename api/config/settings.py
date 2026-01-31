from pydantic_settings import BaseSettings
from typing import Optional, List


class Settings(BaseSettings):
    """로컬 모델 실행을 위한 설정"""
    
    # API Gateway 서버 설정
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False
    
    # 모델 설정 (Hugging Face 모델 이름)
    LLM_MODEL_NAME: str = "Qwen/Qwen2.5-7B-Instruct"
    VLM_MODEL_NAME: str = "Qwen/Qwen2-VL-7B-Instruct"
    OCR_MODEL_NAME: str = "datalab-to/chandra"
    
    # GPU 할당
    LLM_GPU_ID: int = 0  # GPU 0: Qwen LLM
    VLM_GPU_ID: int = 1  # GPU 1: Qwen VLM
    OCR_GPU_ID: int = 2  # GPU 2: Docling + Chandra OCR
    
    # 모델 옵션
    USE_QUANTIZATION: bool = False  # 양자화 사용 여부
    LOAD_IN_8BIT: bool = False
    LOAD_IN_4BIT: bool = False
    
    # 모델 캐시 디렉토리
    MODEL_CACHE_DIR: str = "/root/.cache/huggingface"
    
    # RunPod Pod 엔드포인트 (선택사항 - 원격 Pod 사용 시)
    # 로컬 모델 사용 시 비워두면 됨
    RUNPOD_LLM_ENDPOINT: str = ""
    RUNPOD_VLM_ENDPOINT: str = ""
    RUNPOD_DOCLING_ENDPOINT: str = ""
    
    # RunPod API 키 (선택사항)
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
