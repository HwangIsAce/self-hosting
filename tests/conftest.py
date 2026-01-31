"""pytest 설정 파일"""
import pytest
from unittest.mock import patch, MagicMock

# settings 모듈을 미리 mock하여 import 문제 방지
@pytest.fixture(scope="session", autouse=True)
def mock_settings():
    """전역 settings mock"""
    with patch("api.config.settings") as mock:
        mock.RUNPOD_API_KEY = None
        mock.REQUEST_TIMEOUT = 300
        mock.MAX_CONNECTIONS = 100
        mock.MAX_KEEPALIVE_CONNECTIONS = 20
        mock.RUNPOD_LLM_ENDPOINT = "https://llm.runpod.net"
        mock.RUNPOD_VLM_ENDPOINT = "https://vlm.runpod.net"
        mock.RUNPOD_DOCLING_ENDPOINT = "https://docling.runpod.net"
        yield mock
