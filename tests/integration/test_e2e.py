"""E2E 테스트 — 이번 변경사항(인증, rate limit, 업로드 제한, 에러 분류 등) 검증"""

import pytest
import sys
from unittest.mock import MagicMock, AsyncMock, patch

# ── GPU 의존 라이브러리 mock (macOS 등 GPU 없는 환경용) ──
# 반드시 api.* import 전에 설정해야 함

# torch mock
mock_torch = MagicMock()
mock_torch.cuda.is_available.return_value = False
mock_torch.cuda.device_count.return_value = 0
mock_torch.bfloat16 = "bfloat16"
mock_torch.float16 = "float16"
mock_torch.no_grad.return_value.__enter__ = MagicMock()
mock_torch.no_grad.return_value.__exit__ = MagicMock()
sys.modules['torch'] = mock_torch
sys.modules['torch.cuda'] = mock_torch.cuda

# transformers mock
sys.modules['transformers'] = MagicMock()

# PIL mock
mock_pil = MagicMock()
sys.modules['PIL'] = mock_pil
sys.modules['PIL.Image'] = mock_pil.Image

# vllm mock
sys.modules['vllm'] = MagicMock()
sys.modules['vllm.engine'] = MagicMock()
sys.modules['vllm.engine.arg_utils'] = MagicMock()
sys.modules['vllm.engine.async_llm_engine'] = MagicMock()
sys.modules['vllm.sampling_params'] = MagicMock()

# chandra_ocr mock
sys.modules['chandra_ocr'] = MagicMock()

# docling mock
sys.modules['docling'] = MagicMock()
sys.modules['docling.document_converter'] = MagicMock()
sys.modules['docling.datamodel'] = MagicMock()
sys.modules['docling.datamodel.base_models'] = MagicMock()
sys.modules['docling.datamodel.pipeline_options'] = MagicMock()

# tenacity mock (optional; may be installable but mock for safety)
if 'tenacity' not in sys.modules:
    sys.modules['tenacity'] = MagicMock()

# accelerate mock
sys.modules['accelerate'] = MagicMock()

# einops mock
sys.modules['einops'] = MagicMock()

# protobuf mock
sys.modules['google'] = MagicMock()
sys.modules['google.protobuf'] = MagicMock()

# ── settings mock ──
mock_settings = MagicMock()
mock_settings.RUNPOD_API_KEY = None
mock_settings.REQUEST_TIMEOUT = 300
mock_settings.MAX_RETRIES = 3
mock_settings.RETRY_DELAY = 1.0
mock_settings.MAX_CONNECTIONS = 100
mock_settings.MAX_KEEPALIVE_CONNECTIONS = 20
mock_settings.RUNPOD_LLM_ENDPOINT = ""
mock_settings.RUNPOD_VLM_ENDPOINT = ""
mock_settings.RUNPOD_DOCLING_ENDPOINT = ""
mock_settings.CORS_ORIGINS = []
mock_settings.LOG_LEVEL = "INFO"
mock_settings.LOG_FORMAT = "text"
mock_settings.LOG_FILE = None
mock_settings.ENABLE_CACHE = False
mock_settings.HOST = "0.0.0.0"
mock_settings.PORT = 8000
mock_settings.DEBUG = False
mock_settings.API_KEYS = ["test-key-1", "test-key-2"]
mock_settings.IDLE_UNLOAD_SECONDS = 0
mock_settings.PRELOAD_LLM = False
mock_settings.PRELOAD_VLM = False
mock_settings.LLM_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
mock_settings.VLM_MODEL_NAME = "Qwen/Qwen2-VL-7B-Instruct"
mock_settings.OCR_MODEL_NAME = "datalab-to/chandra"
mock_settings.COLPALI_MODEL_NAME = "vidore/colpali-v1.3-hf"
mock_settings.LLM_GPU_ID = 0
mock_settings.VLM_GPU_ID = 1
mock_settings.OCR_GPU_ID = 2
mock_settings.COLPALI_GPU_ID = 2
mock_settings.USE_VLLM = True
mock_settings.USE_QUANTIZATION = False
mock_settings.MODEL_TO_POD = {
    "qwen-llm": "llm",
    "qwen-llm-7b": "llm",
    "qwen-llm-14b": "llm",
    "qwen-vlm": "vlm",
    "qwen-vlm-7b": "vlm",
    "qwen-vlm-14b": "vlm",
}
mock_settings.BATCH_MAX_REQUESTS = 100
mock_settings.BATCH_CHUNK_SIZE = None
mock_settings.BATCH_TIMEOUT = 600.0
mock_settings.BATCH_ALLOW_PARTIAL_FAILURE = True
mock_settings.VLLM_MAX_NUM_SEQS = 256

sys.modules['api.config.settings'] = MagicMock(settings=mock_settings)

from fastapi.testclient import TestClient
from api.main import app


@pytest.fixture
def client():
    """인증 없는 테스트 클라이언트"""
    return TestClient(app)


@pytest.fixture
def auth_headers():
    """유효한 API Key 헤더"""
    return {"X-API-Key": "test-key-1"}


@pytest.fixture
def bearer_headers():
    """Bearer 토큰 형식 헤더"""
    return {"Authorization": "Bearer test-key-2"}


# ──────────────────────────────────────────
#  Phase 1: 인증 테스트
# ──────────────────────────────────────────

class TestAuthentication:
    """API Key 인증 미들웨어 E2E 테스트"""

    def test_public_endpoints_no_auth(self, client):
        """공개 경로는 인증 없이 접근 가능"""
        assert client.get("/").status_code == 200
        assert client.get("/health").status_code == 200
        assert client.get("/docs").status_code == 200

    def test_protected_endpoint_no_key(self, client):
        """API Key 없이 보호 경로 접근 → 401"""
        response = client.get("/v1/models")
        assert response.status_code == 401
        data = response.json()
        assert data["error"]["code"] == "invalid_api_key"

    def test_protected_endpoint_invalid_key(self, client):
        """잘못된 API Key → 401"""
        response = client.get(
            "/v1/models",
            headers={"X-API-Key": "wrong-key"}
        )
        assert response.status_code == 401

    def test_protected_endpoint_valid_key(self, client, auth_headers):
        """유효한 X-API-Key → 200"""
        response = client.get("/v1/models", headers=auth_headers)
        assert response.status_code == 200

    def test_protected_endpoint_bearer_token(self, client, bearer_headers):
        """Bearer 토큰 형식도 인증 성공"""
        response = client.get("/v1/models", headers=bearer_headers)
        assert response.status_code == 200

    def test_chat_endpoint_requires_auth(self, client):
        """Chat 엔드포인트도 인증 필요"""
        response = client.post(
            "/v1/chat/completions",
            json={"model": "qwen-llm-7b", "messages": [{"role": "user", "content": "Hi"}]}
        )
        assert response.status_code == 401

    def test_ocr_endpoint_requires_auth(self, client):
        """OCR 엔드포인트도 인증 필요"""
        response = client.post("/v1/ocr")
        assert response.status_code == 401

    def test_documents_endpoint_requires_auth(self, client):
        """Documents 엔드포인트도 인증 필요"""
        response = client.post("/v1/documents/process")
        assert response.status_code == 401


# ──────────────────────────────────────────
#  Phase 1: CORS 테스트
# ──────────────────────────────────────────

class TestCORS:
    """CORS 설정 테스트"""

    def test_cors_no_origin_allowed(self, client, auth_headers):
        """CORS_ORIGINS=[] 일 때 Origin 헤더가 있으면 CORS 헤더 없음"""
        response = client.get(
            "/v1/models",
            headers={**auth_headers, "Origin": "https://evil.com"}
        )
        # 요청 자체는 성공하지만 CORS 허용 헤더가 없음
        assert "access-control-allow-origin" not in response.headers


# ──────────────────────────────────────────
#  Phase 2: 업로드 크기 제한 테스트
# ──────────────────────────────────────────

class TestUploadLimits:
    """파일 업로드 크기 제한 E2E 테스트"""

    def test_document_under_limit(self, client, auth_headers):
        """500MB 이하 문서 → 크기 검증 통과 (서비스 mock 필요)"""
        from api.application.services.document_service import DocumentService
        from api.presentation.dependencies.get_services import get_document_service

        mock_service = DocumentService(model_router=MagicMock())
        mock_service.process_document = AsyncMock(return_value={
            "text": "ok", "markdown": "ok", "html": "<p>ok</p>",
            "json": {}, "metadata": {}, "structure": {}
        })
        client.app.dependency_overrides[get_document_service] = lambda: mock_service
        try:
            small_file = b"x" * 1024  # 1KB
            response = client.post(
                "/v1/documents/process",
                headers=auth_headers,
                files={"file": ("test.pdf", small_file, "application/pdf")}
            )
            assert response.status_code == 200
        finally:
            client.app.dependency_overrides.pop(get_document_service, None)

    def test_document_over_limit(self, client, auth_headers):
        """500MB 초과 문서 → 413"""
        over_limit = b"x" * (500 * 1024 * 1024 + 1)  # 500MB + 1 byte
        response = client.post(
            "/v1/documents/process",
            headers=auth_headers,
            files={"file": ("big.pdf", over_limit, "application/pdf")}
        )
        assert response.status_code == 413

    def test_ocr_over_limit(self, client, auth_headers):
        """500MB 초과 이미지 → 413"""
        over_limit = b"x" * (500 * 1024 * 1024 + 1)
        response = client.post(
            "/v1/ocr",
            headers=auth_headers,
            data={"prompt_type": "ocr_layout"},
            files={"image": ("big.png", over_limit, "image/png")}
        )
        assert response.status_code == 413


# ──────────────────────────────────────────
#  Phase 2: 입력 검증 테스트
# ──────────────────────────────────────────

class TestInputValidation:
    """요청 입력 검증 E2E 테스트"""

    def test_chat_missing_messages(self, client, auth_headers):
        """messages 필드 누락 → 422"""
        response = client.post(
            "/v1/chat/completions",
            headers=auth_headers,
            json={"model": "qwen-llm-7b"}
        )
        assert response.status_code == 422

    def test_chat_empty_messages(self, client, auth_headers):
        """빈 messages 리스트 → 422"""
        response = client.post(
            "/v1/chat/completions",
            headers=auth_headers,
            json={"model": "qwen-llm-7b", "messages": []}
        )
        assert response.status_code == 422

    def test_chat_invalid_model(self, client, auth_headers):
        """알 수 없는 모델 → 400"""
        response = client.post(
            "/v1/chat/completions",
            headers=auth_headers,
            json={
                "model": "nonexistent-model",
                "messages": [{"role": "user", "content": "Hello"}]
            }
        )
        assert response.status_code == 400

    def test_chat_invalid_json(self, client, auth_headers):
        """잘못된 JSON → 400 또는 422"""
        response = client.post(
            "/v1/chat/completions",
            headers={**auth_headers, "Content-Type": "application/json"},
            data="not valid json"
        )
        assert response.status_code in [400, 422]

    def test_ocr_no_image(self, client, auth_headers):
        """이미지 없이 OCR → 400"""
        response = client.post(
            "/v1/ocr",
            headers=auth_headers,
            data={"prompt_type": "ocr_layout"}
        )
        assert response.status_code == 400


# ──────────────────────────────────────────
#  Phase 3: 헬스체크 확장 테스트
# ──────────────────────────────────────────

class TestHealthEndpoint:
    """확장된 /health 엔드포인트 E2E 테스트"""

    def test_health_response_structure(self, client):
        """/health 응답에 engines, gpu 필드 포함"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "engines" in data
        assert "gpu" in data
        assert isinstance(data["engines"], dict)
        assert isinstance(data["gpu"], list)

    def test_health_engines_keys(self, client):
        """engines에 ocr 키가 포함"""
        response = client.get("/health")
        data = response.json()
        assert "ocr" in data["engines"]


# ──────────────────────────────────────────
#  Phase 4: 에러 핸들링 분류 테스트
# ──────────────────────────────────────────

class TestErrorClassification:
    """에러 응답 분류 E2E 테스트"""

    def test_404_returns_error_format(self, client, auth_headers):
        """존재하지 않는 경로 → 404 + OpenAI 에러 형식"""
        response = client.get("/v1/nonexistent", headers=auth_headers)
        assert response.status_code == 404
        data = response.json()
        assert "error" in data
        assert "message" in data["error"]

    def test_validation_error_format(self, client, auth_headers):
        """검증 에러 → 422 + details 포함"""
        response = client.post(
            "/v1/chat/completions",
            headers=auth_headers,
            json={"model": "qwen-llm-7b"}  # messages 누락
        )
        assert response.status_code == 422
        data = response.json()
        assert data["error"]["type"] == "invalid_request_error"
        assert "details" in data["error"]


# ──────────────────────────────────────────
#  Mock 서비스를 이용한 정상 플로우 테스트
# ──────────────────────────────────────────

class TestDocumentProcessFlow:
    """문서 처리 정상 플로우 E2E"""

    def test_document_process_returns_all_formats(self, client, auth_headers):
        """문서 처리 응답에 markdown/html/json 모두 포함"""
        from api.application.services.document_service import DocumentService
        from api.presentation.dependencies.get_services import get_document_service

        mock_result = {
            "text": "Hello World",
            "markdown": "# Hello World",
            "html": "<h1>Hello World</h1>",
            "json": {"text": "Hello World"},
            "metadata": {"pages": 3, "file_type": "pdf"},
            "structure": {"tables": 1},
        }
        mock_service = DocumentService(model_router=MagicMock())
        mock_service.process_document = AsyncMock(return_value=mock_result)

        client.app.dependency_overrides[get_document_service] = lambda: mock_service
        try:
            response = client.post(
                "/v1/documents/process",
                headers=auth_headers,
                files={"file": ("test.pdf", b"%PDF-1.4 fake", "application/pdf")}
            )
            assert response.status_code == 200
            data = response.json()
            assert data["markdown"] == "# Hello World"
            assert data["html"] == "<h1>Hello World</h1>"
            assert data["json"] == {"text": "Hello World"}
            assert data["metadata"]["pages"] == 3
        finally:
            client.app.dependency_overrides.pop(get_document_service, None)


class TestOCRProcessFlow:
    """OCR 처리 정상 플로우 E2E"""

    def test_ocr_returns_all_formats(self, client, auth_headers):
        """OCR 응답에 markdown/html/json 모두 포함"""
        from api.application.services.ocr_service import OCRService
        from api.presentation.dependencies.get_services import get_ocr_service

        mock_result = {
            "markdown": "| col1 | col2 |\n|---|---|\n| a | b |",
            "html": "<table><tr><td>a</td><td>b</td></tr></table>",
            "json": {"markdown": "...", "html": "...", "metadata": {}},
            "metadata": {"parser": "chandra"},
        }
        mock_service = OCRService(model_router=MagicMock())
        mock_service.process_ocr = AsyncMock(return_value=mock_result)

        client.app.dependency_overrides[get_ocr_service] = lambda: mock_service
        try:
            minimal_png = (
                b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
                b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00"
                b"\x00\x01\x01\x00\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
            )
            response = client.post(
                "/v1/ocr",
                headers=auth_headers,
                data={"prompt_type": "ocr_layout", "max_tokens": "1024"},
                files={"image": ("test.png", minimal_png, "image/png")}
            )
            assert response.status_code == 200
            data = response.json()
            assert "markdown" in data
            assert "html" in data
            assert "json" in data
        finally:
            client.app.dependency_overrides.pop(get_ocr_service, None)


class TestModelsEndpoint:
    """Models 엔드포인트 E2E"""

    def test_list_models_with_auth(self, client, auth_headers):
        """인증 후 모델 목록 조회"""
        response = client.get("/v1/models", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"
        assert len(data["data"]) > 0
        model_ids = [m["id"] for m in data["data"]]
        assert "qwen-llm-7b" in model_ids
