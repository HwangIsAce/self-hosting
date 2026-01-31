import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from api.application.services.model_router_service import ModelRouterService
from api.application.services.chat_service import ChatService
from api.application.services.document_service import DocumentService
from api.domain.models.chat import ChatCompletionRequest, ChatMessage
from api.application.exceptions.service_exceptions import UnknownModelError


@pytest.fixture
def mock_model_router():
    """Mock ModelRouterService"""
    router = ModelRouterService()
    
    # Mock clients
    mock_llm_client = AsyncMock()
    mock_vlm_client = AsyncMock()
    mock_docling_client = AsyncMock()
    
    router.llm_client = mock_llm_client
    router.vlm_client = mock_vlm_client
    router.docling_client = mock_docling_client
    
    return router


@pytest.fixture
def mock_runpod_response():
    """Mock RunPod 응답"""
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "Hello! How can I help you?"
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 8,
            "total_tokens": 18
        }
    }


class TestModelRouterService:
    """ModelRouterService 테스트"""
    
    @pytest.mark.asyncio
    async def test_route_chat_completion_llm(self, mock_model_router, mock_runpod_response):
        """LLM 모델로 라우팅"""
        mock_model_router.llm_client.post = AsyncMock(return_value=mock_runpod_response)
        
        request = ChatCompletionRequest(
            model="qwen-llm-7b",
            messages=[ChatMessage(role="user", content="Hello")]
        )
        
        response = await mock_model_router.route_chat_completion(request)
        
        assert response.model == "qwen-llm-7b"
        assert len(response.choices) == 1
        assert response.choices[0].message.content == "Hello! How can I help you?"
        assert response.usage.total_tokens == 18
        mock_model_router.llm_client.post.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_route_chat_completion_vlm(self, mock_model_router, mock_runpod_response):
        """VLM 모델로 라우팅"""
        mock_model_router.vlm_client.post = AsyncMock(return_value=mock_runpod_response)
        
        request = ChatCompletionRequest(
            model="qwen-vlm-7b",
            messages=[ChatMessage(role="user", content="What's in this image?", image_url="https://example.com/image.jpg")]
        )
        
        response = await mock_model_router.route_chat_completion(request)
        
        assert response.model == "qwen-vlm-7b"
        mock_model_router.vlm_client.post.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_route_chat_completion_unknown_model(self, mock_model_router):
        """알 수 없는 모델 에러"""
        request = ChatCompletionRequest(
            model="unknown-model",
            messages=[ChatMessage(role="user", content="Hello")]
        )
        
        with pytest.raises(UnknownModelError):
            await mock_model_router.route_chat_completion(request)
    
    @pytest.mark.asyncio
    async def test_convert_to_runpod_format(self, mock_model_router):
        """OpenAI 형식을 RunPod 형식으로 변환"""
        request = ChatCompletionRequest(
            model="qwen-llm-7b",
            messages=[ChatMessage(role="user", content="Hello")],
            temperature=0.7,
            max_tokens=100,
            top_p=0.9
        )
        
        payload = mock_model_router._convert_to_runpod_format(request)
        
        assert "messages" in payload
        assert payload["temperature"] == 0.7
        assert payload["max_tokens"] == 100
        assert payload["top_p"] == 0.9
    
    @pytest.mark.asyncio
    async def test_route_document_processing(self, mock_model_router):
        """문서 처리 라우팅"""
        mock_response = {"text": "Extracted text", "metadata": {}}
        mock_model_router.docling_client.post = AsyncMock(return_value=mock_response)
        
        result = await mock_model_router.route_document_processing(
            file_base64="base64content",
            file_type="pdf"
        )
        
        assert result["text"] == "Extracted text"
        mock_model_router.docling_client.post.assert_called_once()


class TestChatService:
    """ChatService 테스트"""
    
    @pytest.mark.asyncio
    async def test_create_completion_without_cache(self, mock_model_router, mock_runpod_response):
        """캐시 없이 completion 생성"""
        mock_model_router.llm_client.post = AsyncMock(return_value=mock_runpod_response)
        
        service = ChatService(model_router=mock_model_router, cache=None)
        
        request = ChatCompletionRequest(
            model="qwen-llm-7b",
            messages=[ChatMessage(role="user", content="Hello")]
        )
        
        response = await service.create_completion(request)
        
        assert response.model == "qwen-llm-7b"
        assert len(response.choices) == 1
    
    @pytest.mark.asyncio
    async def test_create_completion_with_cache(self, mock_model_router, mock_runpod_response):
        """캐시와 함께 completion 생성"""
        from api.infrastructure.adapters.cache_adapter import CacheAdapter
        
        cache = CacheAdapter()
        mock_model_router.llm_client.post = AsyncMock(return_value=mock_runpod_response)
        
        service = ChatService(model_router=mock_model_router, cache=cache)
        
        request = ChatCompletionRequest(
            model="qwen-llm-7b",
            messages=[ChatMessage(role="user", content="Hello")]
        )
        
        # 첫 번째 요청 (캐시 미스)
        response1 = await service.create_completion(request)
        assert response1.model == "qwen-llm-7b"
        
        # 두 번째 요청 (캐시 히트)
        response2 = await service.create_completion(request)
        assert response2.model == "qwen-llm-7b"
        
        # 두 번째 요청에서는 실제 호출이 한 번만 발생했어야 함
        assert mock_model_router.llm_client.post.call_count == 1


class TestDocumentService:
    """DocumentService 테스트"""
    
    @pytest.mark.asyncio
    async def test_process_document(self, mock_model_router):
        """문서 처리"""
        mock_response = {
            "text": "Extracted text content",
            "metadata": {"page_count": 5}
        }
        mock_model_router.docling_client.post = AsyncMock(return_value=mock_response)
        
        service = DocumentService(model_router=mock_model_router)
        
        result = await service.process_document(
            file_base64="base64content",
            file_type="pdf"
        )
        
        assert result["text"] == "Extracted text content"
        assert result["metadata"]["page_count"] == 5
    
    @pytest.mark.asyncio
    async def test_process_document_no_source(self, mock_model_router):
        """파일 소스가 없는 경우 에러"""
        service = DocumentService(model_router=mock_model_router)
        
        with pytest.raises(ValueError):
            await service.process_document()
