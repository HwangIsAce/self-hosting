import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import httpx

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
    with patch("api.infrastructure.clients.runpod_client.settings") as mock_settings:
        mock_settings.RUNPOD_API_KEY = None
        mock_settings.REQUEST_TIMEOUT = 300
        mock_settings.MAX_CONNECTIONS = 100
        mock_settings.MAX_KEEPALIVE_CONNECTIONS = 20
        
        client = RunPodClient(base_url="https://test.runpod.net")
        yield client
        # Cleanup
        import asyncio
        try:
            asyncio.run(client.close())
        except:
            pass


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
    mock_client.client.post = AsyncMock(side_effect=httpx.TimeoutException("Timeout"))
    
    with pytest.raises(RunPodTimeoutError):
        await mock_client.post("/test", json={"key": "value"})


@pytest.mark.asyncio
async def test_post_connection_error(mock_client):
    """POST 요청 연결 오류 테스트"""
    mock_client.client.post = AsyncMock(side_effect=httpx.ConnectError("Connection failed"))
    
    with pytest.raises(RunPodConnectionError):
        await mock_client.post("/test", json={"key": "value"})


@pytest.mark.asyncio
async def test_post_authentication_error(mock_client):
    """POST 요청 인증 오류 테스트"""
    mock_response = MagicMock()
    mock_response.status_code = 401
    mock_response.text = "Unauthorized"
    
    error = httpx.HTTPStatusError("401", request=MagicMock(), response=mock_response)
    mock_client.client.post = AsyncMock(side_effect=error)
    
    with pytest.raises(RunPodAuthenticationError):
        await mock_client.post("/test", json={"key": "value"})


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
