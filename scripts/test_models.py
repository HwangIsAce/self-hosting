#!/usr/bin/env python3
"""모델 로드 및 호출 테스트 스크립트"""

import asyncio
import sys
import os
import torch
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from api.config.logging_config import setup_logging, get_logger
from api.config.settings import settings
from api.infrastructure.models.llm_engine import LLMEngine
from api.infrastructure.models.vlm_engine import VLMEngine
from api.infrastructure.models.ocr_engine import OCREngine
from api.infrastructure.models.docling_engine import DoclingEngine
from api.application.services.model_router_service import ModelRouterService
from api.domain.models.chat import ChatCompletionRequest, ChatMessage

# 로깅 설정
setup_logging()
logger = get_logger(__name__)


def check_environment():
    """환경 확인 (GPU, CUDA 등)"""
    print("=" * 60)
    print("환경 확인")
    print("=" * 60)
    
    # PyTorch 버전
    print(f"PyTorch 버전: {torch.__version__}")
    
    # CUDA 사용 가능 여부
    cuda_available = torch.cuda.is_available()
    print(f"CUDA 사용 가능: {cuda_available}")
    
    if cuda_available:
        print(f"CUDA 버전: {torch.version.cuda}")
        print(f"GPU 개수: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"    메모리: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
    else:
        print("⚠️  CUDA를 사용할 수 없습니다. CPU 모드로 테스트합니다.")
    
    print()


def check_device_map(device_id: int):
    """디바이스 맵 결정 (GPU 사용 가능 여부에 따라)"""
    if torch.cuda.is_available() and torch.cuda.device_count() > device_id:
        return f"cuda:{device_id}"
    else:
        print(f"⚠️  GPU {device_id}를 사용할 수 없어 CPU 모드로 전환합니다.")
        return "cpu"


async def test_llm_engine():
    """LLM 엔진 테스트"""
    print("=" * 60)
    print("LLM 엔진 테스트")
    print("=" * 60)
    
    try:
        device_map = check_device_map(settings.LLM_GPU_ID)
        print(f"모델: {settings.LLM_MODEL_NAME}")
        print(f"디바이스: {device_map}")
        print()
        
        print("LLM 엔진 초기화 중...")
        engine = LLMEngine(
            model_name=settings.LLM_MODEL_NAME,
            device_map=device_map,
            use_quantization=settings.USE_QUANTIZATION
        )
        
        print("모델 로드 중... (시간이 걸릴 수 있습니다)")
        engine.load_model()
        print("✅ LLM 모델 로드 완료")
        print()
        
        # 간단한 생성 테스트
        print("생성 테스트 중...")
        messages = [
            {"role": "user", "content": "Hello! Say 'test successful' in Korean."}
        ]
        
        result = await engine.generate(
            messages=messages,
            temperature=0.7,
            max_tokens=50,
            top_p=0.9
        )
        
        print("✅ LLM 생성 성공!")
        print(f"응답: {result['choices'][0]['message']['content']}")
        print(f"토큰 사용량: {result['usage']}")
        print()
        
        return True
        
    except Exception as e:
        print(f"❌ LLM 엔진 테스트 실패: {str(e)}")
        import traceback
        traceback.print_exc()
        print()
        return False


async def test_vlm_engine():
    """VLM 엔진 테스트"""
    print("=" * 60)
    print("VLM 엔진 테스트")
    print("=" * 60)
    
    try:
        device_map = check_device_map(settings.VLM_GPU_ID)
        print(f"모델: {settings.VLM_MODEL_NAME}")
        print(f"디바이스: {device_map}")
        print()
        
        print("VLM 엔진 초기화 중...")
        engine = VLMEngine(
            model_name=settings.VLM_MODEL_NAME,
            device_map=device_map,
            use_quantization=settings.USE_QUANTIZATION
        )
        
        print("모델 로드 중... (시간이 걸릴 수 있습니다)")
        engine.load_model()
        print("✅ VLM 모델 로드 완료")
        print()
        
        # 텍스트만 있는 메시지로 테스트 (이미지 없이)
        print("생성 테스트 중 (텍스트만)...")
        messages = [
            {"role": "user", "content": "Describe what you can do."}
        ]
        
        result = await engine.generate(
            messages=messages,
            temperature=0.7,
            max_tokens=50,
            top_p=0.9
        )
        
        print("✅ VLM 생성 성공!")
        print(f"응답: {result['choices'][0]['message']['content']}")
        print(f"토큰 사용량: {result['usage']}")
        print()
        
        return True
        
    except Exception as e:
        print(f"❌ VLM 엔진 테스트 실패: {str(e)}")
        import traceback
        traceback.print_exc()
        print()
        return False


async def test_ocr_engine():
    """OCR 엔진 테스트"""
    print("=" * 60)
    print("OCR 엔진 테스트")
    print("=" * 60)
    
    try:
        device_map = check_device_map(settings.OCR_GPU_ID)
        print(f"모델: {settings.OCR_MODEL_NAME}")
        print(f"디바이스: {device_map}")
        print()
        
        print("OCR 엔진 초기화 중...")
        engine = OCREngine(
            model_name=settings.OCR_MODEL_NAME,
            device_map=device_map
        )
        # 모델은 초기화 시 자동으로 로드됨 (싱글톤 패턴)
        print("✅ OCR 엔진 초기화 완료 (모델 자동 로드됨)")
        print()
        
        # 간단한 테스트 이미지 생성 (1x1 픽셀)
        print("OCR 처리 테스트 중...")
        from PIL import Image
        import base64
        from io import BytesIO
        
        # 작은 테스트 이미지 생성
        test_image = Image.new('RGB', (100, 50), color='white')
        buffer = BytesIO()
        test_image.save(buffer, format='PNG')
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        result = await engine.process(
            image_base64=image_base64,
            prompt_type="ocr_layout",
            output_format="markdown"
        )
        
        print("✅ OCR 처리 성공!")
        print(f"결과: {result.get('text', 'N/A')}")
        print(f"메타데이터: {result.get('metadata', {})}")
        print()
        
        return True
        
    except Exception as e:
        print(f"❌ OCR 엔진 테스트 실패: {str(e)}")
        import traceback
        traceback.print_exc()
        print()
        return False


async def test_docling_engine():
    """Docling 엔진 테스트"""
    print("=" * 60)
    print("Docling 엔진 테스트")
    print("=" * 60)
    
    try:
        print("Docling 엔진 초기화 중...")
        engine = DoclingEngine()
        
        print("초기화 중...")
        await engine.initialize()
        print("✅ Docling 엔진 초기화 완료")
        print()
        
        # 간단한 텍스트 파일 테스트
        print("문서 처리 테스트 중...")
        test_text = "This is a test document.\nIt has multiple lines.\nFor testing purposes."
        import base64
        file_base64 = base64.b64encode(test_text.encode('utf-8')).decode()
        
        result = await engine.process_document(
            file_base64=file_base64,
            file_type="txt"
        )
        
        print("✅ Docling 처리 성공!")
        print(f"텍스트: {result.get('text', 'N/A')[:100]}...")
        print(f"메타데이터: {result.get('metadata', {})}")
        print()
        
        return True
        
    except Exception as e:
        print(f"❌ Docling 엔진 테스트 실패: {str(e)}")
        import traceback
        traceback.print_exc()
        print()
        return False


async def test_model_router():
    """모델 라우터 테스트"""
    print("=" * 60)
    print("모델 라우터 테스트")
    print("=" * 60)
    
    try:
        print("모델 라우터 초기화 중...")
        router = ModelRouterService()
        print("✅ 모델 라우터 초기화 완료")
        print()
        
        # LLM 라우팅 테스트
        print("LLM 라우팅 테스트 중...")
        request = ChatCompletionRequest(
            model="qwen-llm-7b",
            messages=[
                ChatMessage(role="user", content="Say hello")
            ],
            max_tokens=20
        )
        
        response = await router.route_chat_completion(request)
        print("✅ LLM 라우팅 성공!")
        print(f"응답: {response.choices[0].message.content}")
        print()
        
        return True
        
    except Exception as e:
        print(f"❌ 모델 라우터 테스트 실패: {str(e)}")
        import traceback
        traceback.print_exc()
        print()
        return False


async def main():
    """메인 테스트 함수"""
    print("\n" + "=" * 60)
    print("모델 로드 및 호출 테스트 시작")
    print("=" * 60 + "\n")
    
    # 환경 확인
    check_environment()
    
    results = {}
    
    # 각 엔진 테스트
    print("\n" + "=" * 60)
    print("개별 엔진 테스트")
    print("=" * 60 + "\n")
    
    # Docling은 가장 간단하므로 먼저 테스트
    results['docling'] = await test_docling_engine()
    
    # LLM 테스트
    results['llm'] = await test_llm_engine()
    
    # VLM 테스트
    results['vlm'] = await test_vlm_engine()
    
    # OCR 테스트
    results['ocr'] = await test_ocr_engine()
    
    # 모델 라우터 테스트 (LLM이 성공한 경우만)
    if results['llm']:
        results['router'] = await test_model_router()
    else:
        print("⚠️  LLM 테스트가 실패하여 라우터 테스트를 건너뜁니다.")
        results['router'] = False
    
    # 결과 요약
    print("\n" + "=" * 60)
    print("테스트 결과 요약")
    print("=" * 60)
    
    for name, success in results.items():
        status = "✅ 성공" if success else "❌ 실패"
        print(f"{name.upper():10s}: {status}")
    
    total = len(results)
    passed = sum(results.values())
    print(f"\n전체: {passed}/{total} 테스트 통과")
    
    if passed == total:
        print("\n🎉 모든 테스트가 성공했습니다!")
        return 0
    else:
        print(f"\n⚠️  {total - passed}개의 테스트가 실패했습니다.")
        return 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n테스트가 중단되었습니다.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n치명적 오류: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

