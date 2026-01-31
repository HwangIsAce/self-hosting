import pytest
from pydantic import ValidationError

from api.domain.models.chat import (
    ChatMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionChoice,
    Usage
)
from api.domain.models.completions import (
    CompletionRequest,
    CompletionResponse,
    CompletionChoice,
    CompletionUsage
)
from api.domain.models.documents import (
    DocumentProcessRequest,
    DocumentProcessResponse
)
from api.domain.models.models import ModelInfo, ModelsListResponse


class TestChatModels:
    """Chat 모델 테스트"""
    
    def test_chat_message_valid(self):
        """유효한 ChatMessage 생성"""
        message = ChatMessage(role="user", content="Hello")
        assert message.role == "user"
        assert message.content == "Hello"
    
    def test_chat_message_empty_content(self):
        """빈 content 검증"""
        with pytest.raises(ValidationError):
            ChatMessage(role="user", content="")
    
    def test_chat_message_with_image(self):
        """이미지 포함 user 메시지"""
        message = ChatMessage(
            role="user",
            content="What's in this image?",
            image_url="https://example.com/image.jpg"
        )
        assert message.image_url == "https://example.com/image.jpg"
    
    def test_chat_message_image_only_in_user(self):
        """이미지는 user 메시지에만 허용"""
        with pytest.raises(ValidationError):
            ChatMessage(
                role="assistant",
                content="Response",
                image_url="https://example.com/image.jpg"
            )
    
    def test_chat_completion_request_valid(self):
        """유효한 ChatCompletionRequest 생성"""
        request = ChatCompletionRequest(
            model="qwen-llm-7b",
            messages=[ChatMessage(role="user", content="Hello")]
        )
        assert request.model == "qwen-llm-7b"
        assert len(request.messages) == 1
        assert request.temperature == 0.7  # 기본값
    
    def test_chat_completion_request_empty_messages(self):
        """빈 messages 검증"""
        with pytest.raises(ValidationError):
            ChatCompletionRequest(
                model="qwen-llm-7b",
                messages=[]
            )
    
    def test_chat_completion_request_temperature_range(self):
        """temperature 범위 검증"""
        with pytest.raises(ValidationError):
            ChatCompletionRequest(
                model="qwen-llm-7b",
                messages=[ChatMessage(role="user", content="Hello")],
                temperature=3.0  # 범위 초과
            )
    
    def test_chat_completion_response(self):
        """ChatCompletionResponse 생성"""
        response = ChatCompletionResponse(
            model="qwen-llm-7b",
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content="Hi!"),
                    finish_reason="stop"
                )
            ],
            usage=Usage(
                prompt_tokens=10,
                completion_tokens=5,
                total_tokens=15
            )
        )
        assert response.object == "chat.completion"
        assert len(response.choices) == 1
        assert response.usage.total_tokens == 15


class TestCompletionModels:
    """Completion 모델 테스트"""
    
    def test_completion_request_valid(self):
        """유효한 CompletionRequest 생성"""
        request = CompletionRequest(
            model="qwen-llm-7b",
            prompt="Hello"
        )
        assert request.model == "qwen-llm-7b"
        assert request.prompt == "Hello"
        assert request.max_tokens == 16  # 기본값
    
    def test_completion_request_list_prompt(self):
        """리스트 형태의 prompt"""
        request = CompletionRequest(
            model="qwen-llm-7b",
            prompt=["Hello", "World"]
        )
        assert isinstance(request.prompt, list)
    
    def test_completion_response(self):
        """CompletionResponse 생성"""
        response = CompletionResponse(
            model="qwen-llm-7b",
            choices=[
                CompletionChoice(
                    text="Hello world",
                    index=0,
                    finish_reason="stop"
                )
            ],
            usage=CompletionUsage(
                prompt_tokens=5,
                completion_tokens=2,
                total_tokens=7
            )
        )
        assert response.object == "text_completion"
        assert len(response.choices) == 1


class TestDocumentModels:
    """Document 모델 테스트"""
    
    def test_document_process_request_with_url(self):
        """file_url로 DocumentProcessRequest 생성"""
        request = DocumentProcessRequest(
            file_url="https://example.com/doc.pdf"
        )
        assert request.file_url == "https://example.com/doc.pdf"
    
    def test_document_process_request_with_base64(self):
        """file_base64로 DocumentProcessRequest 생성"""
        request = DocumentProcessRequest(
            file_base64="base64encodedcontent"
        )
        assert request.file_base64 == "base64encodedcontent"
    
    def test_document_process_request_no_source(self):
        """file_url과 file_base64 모두 없으면 에러"""
        with pytest.raises(ValidationError):
            DocumentProcessRequest()
    
    def test_document_process_response(self):
        """DocumentProcessResponse 생성"""
        response = DocumentProcessResponse(
            text="Extracted text",
            metadata={"page_count": 10},
            structure={"sections": ["Intro", "Main"]}
        )
        assert response.object == "document.process"
        assert response.text == "Extracted text"
        assert response.metadata["page_count"] == 10


class TestModelModels:
    """Model 모델 테스트"""
    
    def test_model_info(self):
        """ModelInfo 생성"""
        model = ModelInfo(
            id="qwen-llm-7b",
            description="Qwen LLM 7B"
        )
        assert model.id == "qwen-llm-7b"
        assert model.object == "model"
        assert model.owned_by == "self-hosting"
    
    def test_models_list_response(self):
        """ModelsListResponse 생성"""
        models = [
            ModelInfo(id="qwen-llm-7b"),
            ModelInfo(id="qwen-vlm-7b")
        ]
        response = ModelsListResponse(data=models)
        assert response.object == "list"
        assert len(response.data) == 2
