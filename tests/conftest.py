"""pytest 설정 파일"""
import pytest
from unittest.mock import patch, MagicMock
import sys

# settings 모듈을 미리 mock하여 import 문제 방지
# test_runpod_client.py는 자체 mock을 사용하므로 제외
@pytest.fixture(scope="session", autouse=True)
def mock_settings_for_import(request):
    """전역 settings mock (import 시에만 사용)"""
    # test_runpod_client.py가 포함된 경우 mock을 완전히 비활성화
    # RunPodClient의 @retry 데코레이터가 클래스 정의 시점에 settings를 평가하므로
    # test_runpod_client.py가 먼저 실행되거나 포함된 경우 mock하지 않음
    test_files = []
    try:
        if hasattr(request, 'session') and hasattr(request.session, 'items'):
            for item in request.session.items:
                if hasattr(item, 'nodeid'):
                    test_files.append(item.nodeid)
                    if "test_runpod_client" in item.nodeid:
                        # test_runpod_client.py가 포함된 경우 mock하지 않음
                        yield
                        return
    except:
        pass
    
    # RunPodClient가 이미 import된 경우 (test_runpod_client.py가 먼저 실행된 경우)
    # settings를 mock하지 않음
    if 'api.infrastructure.clients.runpod_client' in sys.modules:
        yield
        return
    
    mock_settings_obj = MagicMock()
    # 실제 값으로 설정 (MagicMock이 아닌)
    type(mock_settings_obj).RUNPOD_API_KEY = None
    type(mock_settings_obj).REQUEST_TIMEOUT = 300
    type(mock_settings_obj).MAX_RETRIES = 3
    type(mock_settings_obj).RETRY_DELAY = 1.0
    type(mock_settings_obj).MAX_CONNECTIONS = 100
    type(mock_settings_obj).MAX_KEEPALIVE_CONNECTIONS = 20
    type(mock_settings_obj).RUNPOD_LLM_ENDPOINT = "https://llm.runpod.net"
    type(mock_settings_obj).RUNPOD_VLM_ENDPOINT = "https://vlm.runpod.net"
    type(mock_settings_obj).RUNPOD_DOCLING_ENDPOINT = "https://docling.runpod.net"
    type(mock_settings_obj).CORS_ORIGINS = ["*"]
    type(mock_settings_obj).LOG_LEVEL = "INFO"
    type(mock_settings_obj).ENABLE_CACHE = False
    type(mock_settings_obj).HOST = "0.0.0.0"
    type(mock_settings_obj).PORT = 8000
    type(mock_settings_obj).DEBUG = False
    type(mock_settings_obj).MODEL_TO_POD = {
        "qwen-llm": "llm",
        "qwen-llm-7b": "llm",
        "qwen-llm-14b": "llm",
        "qwen-vlm": "vlm",
        "qwen-vlm-7b": "vlm",
        "qwen-vlm-14b": "vlm",
    }
    
    # settings 모듈을 mock
    with patch('api.config.settings.settings', mock_settings_obj):
        yield mock_settings_obj
