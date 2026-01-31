"""pytest 설정 파일"""
import pytest
from unittest.mock import patch, MagicMock
import sys

# settings 모듈을 미리 mock하여 import 문제 방지
@pytest.fixture(scope="session", autouse=True)
def mock_settings():
    """전역 settings mock"""
    mock_settings_obj = MagicMock()
    mock_settings_obj.RUNPOD_API_KEY = None
    mock_settings_obj.REQUEST_TIMEOUT = 300
    mock_settings_obj.MAX_CONNECTIONS = 100
    mock_settings_obj.MAX_KEEPALIVE_CONNECTIONS = 20
    mock_settings_obj.RUNPOD_LLM_ENDPOINT = "https://llm.runpod.net"
    mock_settings_obj.RUNPOD_VLM_ENDPOINT = "https://vlm.runpod.net"
    mock_settings_obj.RUNPOD_DOCLING_ENDPOINT = "https://docling.runpod.net"
    mock_settings_obj.CORS_ORIGINS = ["*"]
    mock_settings_obj.LOG_LEVEL = "INFO"
    mock_settings_obj.ENABLE_CACHE = False
    mock_settings_obj.HOST = "0.0.0.0"
    mock_settings_obj.PORT = 8000
    mock_settings_obj.DEBUG = False
    mock_settings_obj.MODEL_TO_POD = {
        "qwen-llm": "llm",
        "qwen-llm-7b": "llm",
        "qwen-llm-14b": "llm",
        "qwen-vlm": "vlm",
        "qwen-vlm-7b": "vlm",
        "qwen-vlm-14b": "vlm",
    }
    
    # settings 모듈을 mock
    with patch.dict('sys.modules', {'api.config.settings': MagicMock(settings=mock_settings_obj)}):
        with patch('api.config.settings.settings', mock_settings_obj):
            yield mock_settings_obj
