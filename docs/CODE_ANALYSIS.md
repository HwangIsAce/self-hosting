# Self-Hosting 코드 분석 및 설명

이 문서는 Self-Hosting API 프로젝트의 구조, 아키텍처, 주요 컴포넌트를 정리한 분석 문서입니다.

---

## 1. 프로젝트 개요

**Self-Hosting API**는 **OpenAI 호환 API 게이트웨이**로, 다음 서비스를 하나의 API로 통합합니다.

| 서비스 | 모델/프레임워크 | 용도 |
|--------|-----------------|------|
| **Qwen LLM** | Hugging Face / vLLM | 텍스트 생성 (채팅) |
| **Qwen VLM** | Hugging Face | 이미지+텍스트 멀티모달 |
| **Docling** | Docling Framework | 문서 처리 (PDF, DOCX 등) |
| **Chandra** | Hugging Face (datalab-to/chandra) | OCR (이미지 → 텍스트) |

- **실행 방식**: RunPod 원격 Pod 또는 **로컬 GPU**에서 직접 모델 실행 (설정에 따라 선택).
- **API 스타일**: OpenAI API 형식 (`/v1/chat/completions`, `/v1/models` 등)으로 호출 가능.

---

## 2. 디렉터리 구조 (레이어드 아키텍처)

```
api/
├── main.py                    # FastAPI 앱 진입점, 라우터/미들웨어 등록
├── config/                    # 설정
│   ├── settings.py            # 환경 변수 기반 설정 (Pydantic Settings)
│   └── logging_config.py      # 로깅 설정
├── domain/                    # 도메인 모델 (비즈니스 데이터 구조)
│   └── models/
│       ├── chat.py            # Chat 요청/응답 (OpenAI 스타일)
│       ├── completions.py
│       ├── documents.py
│       ├── models.py          # /v1/models 응답 모델
│       └── ocr.py
├── application/               # 애플리케이션 서비스 (유스케이스)
│   ├── services/
│   │   ├── chat_service.py         # 채팅 완성 생성, 배치, 캐시
│   │   ├── model_router_service.py # 모델별 엔진 라우팅 (핵심)
│   │   ├── document_service.py    # 문서 처리 오케스트레이션
│   │   └── ocr_service.py         # OCR 오케스트레이션
│   └── exceptions/
│       ├── service_exceptions.py  # UnknownModelError 등
│       └── runpod_exceptions.py   # RunPod 관련 예외
├── infrastructure/            # 외부 연동 및 구현체
│   ├── models/                # 모델 실행 엔진
│   │   ├── model_loader.py    # Hugging Face 모델 로드 (LLM/VLM/OCR)
│   │   ├── llm_engine.py      # HF 기반 LLM (백업용)
│   │   ├── vllm_engine.py     # vLLM 기반 LLM (기본, 고성능)
│   │   ├── vlm_engine.py      # VLM (Qwen2-VL)
│   │   ├── ocr_engine.py      # Chandra OCR
│   │   └── docling_engine.py # Docling 문서 변환
│   ├── clients/
│   │   └── runpod_client.py   # RunPod 원격 호출 (선택 사용)
│   ├── adapters/
│   │   └── cache_adapter.py   # 응답 캐시 (선택)
│   └── utils/
│       └── batch_processor.py # 배치 처리 유틸
└── presentation/              # HTTP 레이어
    ├── routers/
    │   ├── chat.py            # POST /v1/chat/completions, /v1/chat/completions/batch
    │   ├── completions.py     # POST /v1/completions
    │   ├── documents.py       # POST /v1/documents/process
    │   ├── ocr.py             # POST /v1/ocr
    │   └── models.py          # GET /v1/models
    ├── middleware/
    │   ├── logging.py         # 요청/응답 로깅
    │   └── error_handler.py   # 검증/HTTP/일반 예외 → JSON 응답
    └── dependencies/
        └── get_services.py    # ChatService, DocumentService, OCRService, ModelRouterService 주입
```

- **도메인**: 요청/응답 스키마, 비즈니스 규칙(검증 등).
- **애플리케이션**: 채팅/문서/OCR 흐름 조율, 모델 라우팅, 캐시/배치 처리.
- **인프라**: 실제 모델 로딩/실행(HF, vLLM, Docling, Chandra), RunPod 클라이언트, 캐시 어댑터.
- **프레젠테이션**: FastAPI 라우터, 미들웨어, 의존성 주입.

---

## 3. 요청 흐름 (Chat 예시)

1. **클라이언트** → `POST /v1/chat/completions` (JSON: `model`, `messages`, `temperature` 등)
2. **라우터** (`api/presentation/routers/chat.py`)  
   - `ChatCompletionRequest`로 파싱 후 `ChatService.create_completion(request)` 호출.
3. **ChatService** (`api/application/services/chat_service.py`)  
   - (선택) 캐시 조회 → 없으면 `model_router.route_chat_completion(request)` 호출.  
   - 응답 수신 후 (선택) 캐시 저장, `ChatCompletionResponse` 반환.
4. **ModelRouterService** (`api/application/services/model_router_service.py`)  
   - `request.model`으로 Pod 타입 결정: `qwen-llm*` → `llm`, `qwen-vlm*` → `vlm`.  
   - `_get_engine(pod_type)`으로 엔진 획득:  
     - **LLM**: `USE_VLLM=True`면 `VLLMEngine`, 아니면 `LLMEngine` (Hugging Face).  
     - **VLM**: `VLMEngine`.  
   - 해당 엔진의 `generate(messages, temperature, max_tokens, ...)` 호출.  
   - 엔진 응답을 `_convert_to_openai_format()`으로 OpenAI 형식으로 변환해 반환.
5. **엔진** (예: `VLLMEngine`)  
   - 첫 요청 시 `load_model()` (vLLM `AsyncLLMEngine` 초기화).  
   - 메시지를 채팅 템플릿으로 프롬프트 문자열로 변환 후 vLLM으로 생성.  
   - `{ choices, usage }` 형태로 반환.

문서 처리(`/v1/documents/process`)와 OCR(`/v1/ocr`)도 같은 패턴: 라우터 → 서비스 → `ModelRouterService` → `DoclingEngine` / `OCREngine`.

---

## 4. 설정 (api/config/settings.py)

- **서버**: `HOST`, `PORT`, `DEBUG`
- **모델 식별자**: `LLM_MODEL_NAME`, `VLM_MODEL_NAME`, `OCR_MODEL_NAME` (Hugging Face 경로)
- **GPU 할당**: `LLM_GPU_ID`, `VLM_GPU_ID`, `OCR_GPU_ID` (기본 0, 1, 2)
- **vLLM**: `USE_VLLM`, `VLLM_GPU_MEMORY_UTILIZATION`, `VLLM_MAX_MODEL_LEN`, `VLLM_MAX_NUM_SEQS`, `VLLM_SPECULATIVE_MODEL` 등
- **RunPod** (선택): `RUNPOD_LLM_ENDPOINT`, `RUNPOD_VLM_ENDPOINT`, `RUNPOD_DOCLING_ENDPOINT`, `RUNPOD_API_KEY`
- **배치/캐시**: `BATCH_*`, `ENABLE_CACHE`, `CACHE_URL`
- **모델 매핑**: `MODEL_TO_POD` (예: `qwen-llm-7b` → `llm`)

`.env` 또는 환경 변수로 오버라이드 가능.

---

## 5. 주요 컴포넌트 설명

### 5.1 ModelRouterService (핵심 라우터)

- **역할**: 모델 이름 → Pod 타입 → 해당 엔진 인스턴스 반환.
- **엔진 생성**:  
  - LLM: vLLM 사용 시 `VLLMEngine`(싱글톤), 미사용 시 `LLMEngine`.  
  - VLM: `VLMEngine`.  
  - Docling: `DoclingEngine`.  
  - OCR: `OCREngine`.
- **메서드**:  
  - `route_chat_completion(request)` → Chat 응답.  
  - `route_document_processing(file_base64, ...)` → Docling 결과.  
  - `route_ocr_processing(image_base64, ...)` → Chandra OCR 결과.

### 5.2 VLLMEngine (LLM 추론)

- **역할**: vLLM `AsyncLLMEngine`으로 Qwen LLM 텍스트 생성.
- **특징**:  
  - 첫 요청 시 lazy 로드.  
  - Continuous Batching, `SamplingParams` (temperature, top_p, max_tokens, stop).  
  - 채팅 템플릿은 토크나이저 `apply_chat_template` 사용.  
  - 배치 생성 `generate_batch()` 지원, 청크 단위 처리 가능.
- **설정**: `settings`의 vLLM 관련 값 사용 (GPU 메모리, max_model_len, speculative 등).

### 5.3 LLMEngine (Hugging Face 백업)

- **역할**: `USE_VLLM=False`일 때 사용. `ModelLoader`로 HF `AutoModelForCausalLM` + 토크나이저 로드.
- **최적화**: Flash Attention 2(선택), `torch.compile`, 8bit/4bit 양자화 옵션.

### 5.4 VLMEngine

- **역할**: Qwen2-VL (이미지+텍스트). `ModelLoader.load_vlm_model()` → `Qwen2VLForConditionalGeneration` + 프로세서.
- **입력**: 메시지 내 `image_url` / `image_base64` 처리 후 `generate()`.

### 5.5 DoclingEngine

- **역할**: Docling `DocumentConverter`로 PDF/DOCX 등 → 텍스트/구조 추출.
- **옵션**: PDF 파이프라인에서 OCR, 표 구조/셀 매칭 활성화.
- **입력**: Base64 파일 + 파일 타입. TXT는 Docling 미지원으로 별도 처리.

### 5.6 OCREngine (Chandra)

- **역할**: Chandra OCR로 이미지 → 텍스트 (markdown/html/json 등).
- **구현**: `chandra-ocr` 패키지 사용 시 `generate_hf` + `parse_markdown` 등. 없으면 ModelLoader로 HF 모델 직접 로드하는 fallback.

### 5.7 ModelLoader

- **역할**: Hugging Face 모델/토크나이저/프로세서 로드 및 재사용.
- **메서드**: `load_llm_model`, `load_vlm_model`, `load_ocr_model`, `unload_model`, `get_model` 등.
- **공통**: Flash Attention 2 시도, `device_map`, 양자화 옵션, `torch.compile`(환경 변수로 비활성화 가능).

---

## 6. API 엔드포인트 요약

| 메서드 | 경로 | 설명 |
|--------|------|------|
| GET | `/` | 서비스 정보 및 엔드포인트 목록 |
| GET | `/health` | 헬스체크 |
| GET | `/v1/models` | 사용 가능 모델 목록 (OpenAI 형식) |
| POST | `/v1/chat/completions` | 채팅 완성 (LLM/VLM) |
| POST | `/v1/chat/completions/batch` | 배치 채팅 완성 (최대 100건) |
| POST | `/v1/completions` | 레거시 completions (텍스트 완성) |
| POST | `/v1/documents/process` | 문서 처리 (Docling, 파일 업로드) |
| POST | `/v1/ocr` | OCR (Chandra, 이미지 업로드 또는 image_url) |

---

## 7. 예외 처리

- **도메인/애플리케이션**: `UnknownModelError`, RunPod 관련 `RunPodConnectionError`, `RunPodTimeoutError`, `RunPodServiceError`.
- **프레젠테이션**: `api/presentation/middleware/error_handler.py`에서 `RequestValidationError`, `HTTPException`, 일반 `Exception`을 일관된 JSON 응답으로 변환.
- **라우터**: 위 예외를 잡아 400/502/503/504/500 등 적절한 HTTP 상태 코드로 매핑.

---

## 8. 테스트 및 스크립트

- **tests/**  
  - `conftest.py`: pytest 픽스처.  
  - `unit/`: `test_runpod_client.py`, `test_domain_models.py`.  
  - `integration/`: `test_api.py`, `test_services.py`.
- **scripts/**  
  - 배포/헬스: `deploy.sh`, `health_check.py`.  
  - 성능/기능 테스트: vLLM, Flash Attention, Chandra, Docling, 배치 등 관련 스크립트 다수.

---

## 9. 배포

- **Docker**: `Dockerfile` + `docker-compose.yml` (서비스명 `api-gateway`, 포트 8000, 헬스체크 포함).
- **로컬**: `uv sync` 후 `uvicorn api.main:app --host 0.0.0.0 --port 8000`.
- 상세: `DEPLOYMENT.md`, `README.md` 참고.

---

이 문서는 코드베이스 분석 결과를 바탕으로 작성되었으며, 설정이나 엔드포인트 변경 시 해당 파일들을 함께 참고하면 됩니다.
