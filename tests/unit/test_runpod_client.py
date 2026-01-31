import pytest
import sys
from unittest.mock import AsyncMock, patch, MagicMock
import httpx

# settings를 mock하여 import 문제 방지
# @retry 데코레이터가 클래스 정의 시점에 settings.MAX_RETRIES를 읽기 때문에
# RunPodClient를 import하기 전에 settings를 먼저 mock해야 함
mock_settings = MagicMock()
# type()을 사용하여 실제 값으로 설정 (MagicMock이 아닌)
type(mock_settings).RUNPOD_API_KEY = None
type(mock_settings).REQUEST_TIMEOUT = 300
type(mock_settings).MAX_RETRIES = 3  # tenacity를 위해 실제 정수 값 필요
type(mock_settings).RETRY_DELAY = 1.0
type(mock_settings).MAX_CONNECTIONS = 100
type(mock_settings).MAX_KEEPALIVE_CONNECTIONS = 20
type(mock_settings).RUNPOD_LLM_ENDPOINT = "https://llm.runpod.net"
type(mock_settings).RUNPOD_VLM_ENDPOINT = "https://vlm.runpod.net"
type(mock_settings).RUNPOD_DOCLING_ENDPOINT = "https://docling.runpod.net"
type(mock_settings).CORS_ORIGINS = ["*"]
type(mock_settings).LOG_LEVEL = "INFO"
type(mock_settings).ENABLE_CACHE = False
type(mock_settings).HOST = "0.0.0.0"
type(mock_settings).PORT = 8000
type(mock_settings).DEBUG = False
type(mock_settings).MODEL_TO_POD = {
    "qwen-llm": "llm",
    "qwen-llm-7b": "llm",
    "qwen-llm-14b": "llm",
    "qwen-vlm": "vlm",
    "qwen-vlm-7b": "vlm",
    "qwen-vlm-14b": "vlm",
}

# settings 모듈을 mock (import 전에)
sys.modules['api.config.settings'] = MagicMock(settings=mock_settings)

from api.infrastructure.clients.runpod_client import RunPodClient, RunPodClientFactory
from api.application.exceptions.runpod_exceptions import (
    RunPodConnectionError,
    RunPodTimeoutError,
    RunPodServiceError,
    RunPodAuthenticationError,
)


@pytest.fixture
def mock_client():
    """Mock RunPod 클라이언트"""
    # settings를 직접 patch하여 tenacity가 올바른 값을 받도록 함
    # conftest의 mock을 덮어쓰기 위해 새로 patch
    import api.infrastructure.clients.runpod_client as runpod_client_module
    import api.config.settings as settings_module
    
    # 기존 settings를 백업
    original_runpod_settings = getattr(runpod_client_module, 'settings', None)
    original_config_settings = getattr(settings_module, 'settings', None)
    
    # 새로운 mock settings 생성 (실제 값으로 설정)
    from unittest.mock import MagicMock
    new_mock_settings = MagicMock()
    type(new_mock_settings).RUNPOD_API_KEY = None
    type(new_mock_settings).REQUEST_TIMEOUT = 300
    type(new_mock_settings).MAX_RETRIES = 3  # tenacity를 위해 실제 정수 값 필요
    type(new_mock_settings).RETRY_DELAY = 1.0
    type(new_mock_settings).MAX_CONNECTIONS = 100
    type(new_mock_settings).MAX_KEEPALIVE_CONNECTIONS = 20
    
    # settings를 교체 (runpod_client와 config 모두)
    runpod_client_module.settings = new_mock_settings
    settings_module.settings = new_mock_settings
    
    try:
        # RunPodClient를 새로 생성 (settings가 교체된 후)
        client = RunPodClient(base_url="https://test.runpod.net")
        yield client
        # Cleanup
        import asyncio
        try:
            asyncio.run(client.close())
        except:
            pass
    finally:
        # 원래 settings 복원
        if original_runpod_settings:
            runpod_client_module.settings = original_runpod_settings
        if original_config_settings:
            settings_module.settings = original_config_settings


@pytest.mark.asyncio
async def test_post_success(mock_client):
    """POST 요청 성공 테스트"""
    mock_response = MagicMock()
    mock_response.json.return_value = {"result": "success"}
    mock_response.raise_for_status = MagicMock()
    
    mock_client.client.post = AsyncMock(return_value=mock_response)
    
    result = await mock_client.post("/test", json={"key": "value"})
    
    assert result == {"result": "success"}
    mock_client.client.post.assert_called_once_with("/test", json={"key": "value"})


@pytest.mark.asyncio
async def test_post_timeout_error(mock_client):
    """POST 요청 타임아웃 테스트"""
    from tenacity import RetryError
    mock_client.client.post = AsyncMock(side_effect=httpx.TimeoutException("Timeout"))
    
    # 재시도 후 RetryError가 발생하지만, 내부 예외는 RunPodTimeoutError
    with pytest.raises(RetryError) as exc_info:
        await mock_client.post("/test", json={"key": "value"})
    # RetryError 내부의 예외가 RunPodTimeoutError인지 확인
    assert isinstance(exc_info.value.last_attempt.exception(), RunPodTimeoutError)


@pytest.mark.asyncio
async def test_post_connection_error(mock_client):
    """POST 요청 연결 오류 테스트"""
    from tenacity import RetryError
    mock_client.client.post = AsyncMock(side_effect=httpx.ConnectError("Connection failed"))
    
    # 재시도 후 RetryError가 발생하지만, 내부 예외는 RunPodConnectionError
    with pytest.raises(RetryError) as exc_info:
        await mock_client.post("/test", json={"key": "value"})
    # RetryError 내부의 예외가 RunPodConnectionError인지 확인
    assert isinstance(exc_info.value.last_attempt.exception(), RunPodConnectionError)


@pytest.mark.asyncio
async def test_post_authentication_error(mock_client):
    """POST 요청 인증 오류 테스트"""
    from tenacity import RetryError
    mock_response = MagicMock()
    mock_response.status_code = 401
    mock_response.text = "Unauthorized"
    
    error = httpx.HTTPStatusError("401", request=MagicMock(), response=mock_response)
    mock_client.client.post = AsyncMock(side_effect=error)
    
    # 재시도 후 RetryError가 발생하지만, 내부 예외는 RunPodAuthenticationError
    with pytest.raises(RetryError) as exc_info:
        await mock_client.post("/test", json={"key": "value"})
    # RetryError 내부의 예외가 RunPodAuthenticationError인지 확인
    assert isinstance(exc_info.value.last_attempt.exception(), RunPodAuthenticationError)


@pytest.mark.asyncio
async def test_get_success(mock_client):
    """GET 요청 성공 테스트"""
    mock_response = MagicMock()
    mock_response.json.return_value = {"status": "ok"}
    mock_response.raise_for_status = MagicMock()
    
    mock_client.client.get = AsyncMock(return_value=mock_response)
    
    result = await mock_client.get("/test")
    
    assert result == {"status": "ok"}
    mock_client.client.get.assert_called_once_with("/test")


@pytest.mark.asyncio
async def test_health_check_success(mock_client):
    """헬스체크 성공 테스트"""
    mock_response = MagicMock()
    mock_response.json.return_value = {"status": "healthy"}
    mock_response.raise_for_status = MagicMock()
    
    mock_client.client.get = AsyncMock(return_value=mock_response)
    
    result = await mock_client.health_check()
    
    assert result["status"] == "healthy"
    assert "status" in result


@pytest.mark.asyncio
async def test_health_check_failure(mock_client):
    """헬스체크 실패 테스트"""
    mock_client.client.get = AsyncMock(side_effect=Exception("Connection failed"))
    
    result = await mock_client.health_check()
    
    assert result["status"] == "unhealthy"
    assert "error" in result


def test_client_factory_singleton():
    """클라이언트 팩토리 싱글톤 테스트"""
    with patch("api.infrastructure.clients.runpod_client.settings") as mock_settings:
        mock_settings.RUNPOD_LLM_ENDPOINT = "https://llm.runpod.net"
        mock_settings.RUNPOD_VLM_ENDPOINT = "https://vlm.runpod.net"
        mock_settings.RUNPOD_DOCLING_ENDPOINT = "https://docling.runpod.net"
        mock_settings.RUNPOD_API_KEY = None
        mock_settings.REQUEST_TIMEOUT = 300
        mock_settings.MAX_CONNECTIONS = 100
        mock_settings.MAX_KEEPALIVE_CONNECTIONS = 20
        
        # Reset singleton instances
        RunPodClientFactory._llm_client = None
        RunPodClientFactory._vlm_client = None
        RunPodClientFactory._docling_client = None
        
        client1 = RunPodClientFactory.get_llm_client()
        client2 = RunPodClientFactory.get_llm_client()
        
        # Same instance should be returned
        assert client1 is client2
