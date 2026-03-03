import pytest
import sys
from unittest.mock import patch, MagicMock, AsyncMock

# settings를 mock하여 import 문제 방지
mock_settings = MagicMock()
mock_settings.RUNPOD_API_KEY = None
mock_settings.REQUEST_TIMEOUT = 300
mock_settings.MAX_CONNECTIONS = 100
mock_settings.MAX_KEEPALIVE_CONNECTIONS = 20
mock_settings.RUNPOD_LLM_ENDPOINT = "https://llm.runpod.net"
mock_settings.RUNPOD_VLM_ENDPOINT = "https://vlm.runpod.net"
mock_settings.RUNPOD_DOCLING_ENDPOINT = "https://docling.runpod.net"
mock_settings.CORS_ORIGINS = ["*"]
mock_settings.LOG_LEVEL = "INFO"
mock_settings.ENABLE_CACHE = False
mock_settings.HOST = "0.0.0.0"
mock_settings.PORT = 8000
mock_settings.DEBUG = False
mock_settings.MODEL_TO_POD = {
    "qwen-llm": "llm",
    "qwen-llm-7b": "llm",
    "qwen-llm-14b": "llm",
    "qwen-vlm": "vlm",
    "qwen-vlm-7b": "vlm",
    "qwen-vlm-14b": "vlm",
}

# settings 모듈을 mock
sys.modules['api.config.settings'] = MagicMock(settings=mock_settings)

from fastapi.testclient import TestClient
from api.main import app


@pytest.fixture
def client():
    """테스트 클라이언트"""
    return TestClient(app)


class TestRootEndpoints:
    """루트 엔드포인트 테스트"""
    
    def test_root_endpoint(self, client):
        """루트 엔드포인트 테스트"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "Self-Hosting API"
        assert data["version"] == "1.0.0"
        assert data["openai_compatible"] is True
        assert "endpoints" in data
    
    def test_health_endpoint(self, client):
        """헬스체크 엔드포인트 테스트"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"


class TestModelsEndpoint:
    """Models 엔드포인트 테스트"""
    
    def test_list_models(self, client):
        """모델 목록 조회 테스트"""
        response = client.get("/v1/models")
        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"
        assert "data" in data
        assert len(data["data"]) > 0
        
        # 모델 정보 확인
        model_ids = [model["id"] for model in data["data"]]
        assert "qwen-llm-7b" in model_ids
        assert "qwen-vlm-7b" in model_ids


class TestChatEndpoint:
    """Chat 엔드포인트 테스트"""
    
    def test_chat_completions_invalid_model(self, client):
        """잘못된 모델로 요청 시 에러"""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "invalid-model",
                "messages": [{"role": "user", "content": "Hello"}]
            }
        )
        assert response.status_code == 400
    
    def test_chat_completions_missing_fields(self, client):
        """필수 필드 누락 시 검증 에러"""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen-llm-7b"
                # messages 누락
            }
        )
        assert response.status_code == 422  # Validation error
    
    def test_chat_completions_empty_messages(self, client):
        """빈 메시지 리스트 시 검증 에러"""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen-llm-7b",
                "messages": []
            }
        )
        assert response.status_code == 422


class TestDocumentsEndpoint:
    """Documents 엔드포인트 테스트"""

    def test_documents_process_no_file(self, client):
        """파일 없이 요청 시 에러"""
        response = client.post("/v1/documents/process")
        assert response.status_code == 422  # Validation error

    def test_documents_process_response_schema(self, client):
        """문서 처리 응답에 markdown, html, json 필드가 포함되는지 검증"""
        from api.application.services.document_service import DocumentService
        from api.presentation.dependencies.get_services import get_document_service

        mock_result = {
            "text": "Extracted text",
            "markdown": "# Doc\n\nExtracted text",
            "html": "<h1>Doc</h1><p>Extracted text</p>",
            "json": {"texts": [], "tables": []},
            "metadata": {"pages": 1},
            "structure": {"tables": 0},
        }
        mock_service = DocumentService(model_router=MagicMock())
        mock_service.process_document = AsyncMock(return_value=mock_result)

        client.app.dependency_overrides[get_document_service] = lambda: mock_service
        try:
            response = client.post(
                "/v1/documents/process",
                files={"file": ("test.txt", b"Hello world", "text/plain")},
            )
            assert response.status_code == 200
            data = response.json()
            assert "markdown" in data
            assert "html" in data
            assert "json" in data
            assert data["markdown"] == mock_result["markdown"]
            assert data["html"] == mock_result["html"]
            assert data["json"] == mock_result["json"]
        finally:
            client.app.dependency_overrides.pop(get_document_service, None)


class TestOCREndpoint:
    """OCR 엔드포인트 테스트"""

    def test_ocr_no_image_returns_error(self, client):
        """이미지 없이 OCR 요청 시 400 에러"""
        response = client.post("/v1/ocr")
        assert response.status_code == 400

    def test_ocr_response_schema(self, client):
        """OCR 응답에 markdown, html, json 필드가 항상 포함되는지 검증"""
        from api.application.services.ocr_service import OCRService
        from api.presentation.dependencies.get_services import get_ocr_service

        mock_result = {
            "markdown": "# OCR result",
            "html": "<h1>OCR result</h1>",
            "json": {"markdown": "# OCR result", "html": "<h1>OCR result</h1>", "metadata": {}},
            "metadata": {"parser": "chandra"},
        }
        mock_service = OCRService(model_router=MagicMock())
        mock_service.process_ocr = AsyncMock(return_value=mock_result)

        client.app.dependency_overrides[get_ocr_service] = lambda: mock_service
        try:
            # 최소한의 1x1 PNG 바이트 (실제 OCR 호출 없이 스키마만 검증)
            minimal_png_bytes = (
                b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
                b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00"
                b"\x00\x01\x01\x00\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
            )
            response = client.post(
                "/v1/ocr",
                data={"prompt_type": "ocr_layout", "max_tokens": 1024},
                files={"image": ("test.png", minimal_png_bytes, "image/png")},
            )
            assert response.status_code == 200, response.text
            data = response.json()
            assert "markdown" in data
            assert "html" in data
            assert "json" in data
            assert data["markdown"] == mock_result["markdown"]
            assert data["html"] == mock_result["html"]
            assert data["json"] == mock_result["json"]
        finally:
            client.app.dependency_overrides.pop(get_ocr_service, None)


class TestErrorHandling:
    """에러 핸들링 테스트"""
    
    def test_404_not_found(self, client):
        """존재하지 않는 엔드포인트"""
        response = client.get("/v1/nonexistent")
        assert response.status_code == 404
    
    def test_invalid_json(self, client):
        """잘못된 JSON 요청"""
        response = client.post(
            "/v1/chat/completions",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code in [400, 422]
