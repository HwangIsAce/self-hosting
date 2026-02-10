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
    
    # vLLM 설정
    USE_VLLM: bool = True  # vLLM 사용 여부 (기본값: True)
    VLLM_GPU_MEMORY_UTILIZATION: float = 0.75  # GPU 메모리 사용률 (0.0-1.0) - 메모리 초과 방지
    VLLM_MAX_MODEL_LEN: Optional[int] = None  # 최대 시퀀스 길이 (None = 자동)
    VLLM_TENSOR_PARALLEL_SIZE: int = 1  # 텐서 병렬화 (멀티 GPU 시)
    VLLM_DTYPE: str = "float16"  # 모델 데이터 타입
    VLLM_TRUST_REMOTE_CODE: bool = True  # trust_remote_code
    
    # Speculative Decoding (선택사항 - 나중에 활성화 가능)
    VLLM_SPECULATIVE_MODEL: Optional[str] = None  # 예: "Qwen/Qwen2.5-0.5B-Instruct"
    VLLM_NUM_SPECULATIVE_TOKENS: int = 5  # Speculative Decoding 토큰 수
    
    # Continuous Batching 설정
    VLLM_MAX_NUM_BATCHED_TOKENS: Optional[int] = None  # 최대 배치 토큰 수
    VLLM_MAX_NUM_SEQS: int = 256  # 최대 동시 시퀀스 수
    
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
    
    # 배치 처리 설정
    BATCH_MAX_REQUESTS: int = 100  # 배치 최대 요청 수
    BATCH_CHUNK_SIZE: Optional[int] = None  # 대량 배치 청크 크기 (None = 자동)
    BATCH_TIMEOUT: float = 600.0  # 배치 처리 타임아웃 (초) - 10분
    BATCH_ALLOW_PARTIAL_FAILURE: bool = True  # 부분 실패 허용
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
