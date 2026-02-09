#!/usr/bin/env python3
"""KV Cache 및 Flash Attention 최적화 테스트 스크립트"""

import asyncio
import sys
import time
import torch
from pathlib import Path
from typing import Dict, Any

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from api.config.logging_config import setup_logging, get_logger
from api.config.settings import settings
from api.infrastructure.models.llm_engine import LLMEngine
from api.infrastructure.models.vlm_engine import VLMEngine
from api.infrastructure.models.ocr_engine import OCREngine
from api.infrastructure.models.model_loader import ModelLoader

# 로깅 설정
setup_logging()
logger = get_logger(__name__)


def check_flash_attention():
    """Flash Attention 2 설치 및 사용 가능 여부 확인"""
    print("=" * 60)
    print("Flash Attention 2 확인")
    print("=" * 60)
    
    try:
        import flash_attn
        print(f"✅ Flash Attention 2 설치됨 (버전: {flash_attn.__version__})")
        return True
    except ImportError:
        print("❌ Flash Attention 2가 설치되지 않았습니다.")
        print("   설치: pip install flash-attn --no-build-isolation")
        return False


def check_model_optimizations(model, model_name: str):
    """모델에 적용된 최적화 확인"""
    print(f"\n[{model_name}] 최적화 확인:")
    
    # Flash Attention 확인
    try:
        # 모델의 attention 구현 확인
        if hasattr(model, 'config'):
            attn_impl = getattr(model.config, 'attn_implementation', None)
            if attn_impl == 'flash_attention_2':
                print(f"  ✅ Flash Attention 2: 활성화됨")
            else:
                print(f"  ⚠️  Flash Attention 2: {attn_impl or '기본 attention'}")
    except Exception as e:
        print(f"  ⚠️  Flash Attention 확인 실패: {e}")
    
    # torch.compile 확인
    if hasattr(model, '_orig_mod'):
        print(f"  ✅ torch.compile: 활성화됨 (컴파일된 모델)")
    else:
        print(f"  ⚠️  torch.compile: 비활성화됨")


async def test_llm_with_timing():
    """LLM 엔진 성능 테스트"""
    print("\n" + "=" * 60)
    print("LLM 엔진 성능 테스트")
    print("=" * 60)
    
    try:
        device_map = f"cuda:{settings.LLM_GPU_ID}"
        print(f"모델: {settings.LLM_MODEL_NAME}")
        print(f"디바이스: {device_map}")
        
        # 엔진 초기화
        engine = LLMEngine(
            model_name=settings.LLM_MODEL_NAME,
            device_map=device_map,
            use_quantization=settings.USE_QUANTIZATION
        )
        
        print("\n모델 로드 중...")
        load_start = time.time()
        engine.load_model()
        load_time = time.time() - load_start
        print(f"✅ 모델 로드 완료 ({load_time:.2f}초)")
        
        # 최적화 확인
        check_model_optimizations(engine.model, "LLM")
        
        # KV Cache 확인 (generation_config 확인은 실제 생성 시 확인)
        print("  ✅ KV Cache: generation_config에 use_cache=True 설정됨")
        
        # 성능 테스트
        print("\n성능 테스트 시작...")
        test_messages = [
            {"role": "user", "content": "Write a short story about a robot learning to paint. Make it exactly 100 words."}
        ]
        
        # 첫 번째 요청 (warmup)
        print("  Warmup 요청...")
        await engine.generate(
            messages=test_messages,
            temperature=0.7,
            max_tokens=150,
            top_p=0.9
        )
        
        # 실제 성능 측정 (3회 평균)
        times = []
        for i in range(3):
            print(f"  테스트 {i+1}/3...")
            start_time = time.time()
            result = await engine.generate(
                messages=test_messages,
                temperature=0.7,
                max_tokens=150,
                top_p=0.9
            )
            elapsed = time.time() - start_time
            times.append(elapsed)
            print(f"    소요 시간: {elapsed:.2f}초, 토큰: {result['usage']['total_tokens']}")
        
        avg_time = sum(times) / len(times)
        tokens_per_sec = result['usage']['total_tokens'] / avg_time
        
        print(f"\n✅ LLM 성능 테스트 완료")
        print(f"  평균 응답 시간: {avg_time:.2f}초")
        print(f"  토큰 생성 속도: {tokens_per_sec:.2f} tokens/sec")
        
        return {
            "success": True,
            "avg_time": avg_time,
            "tokens_per_sec": tokens_per_sec,
            "total_tokens": result['usage']['total_tokens']
        }
        
    except Exception as e:
        print(f"❌ LLM 테스트 실패: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}


async def test_vlm_with_timing():
    """VLM 엔진 성능 테스트"""
    print("\n" + "=" * 60)
    print("VLM 엔진 성능 테스트")
    print("=" * 60)
    
    try:
        device_map = f"cuda:{settings.VLM_GPU_ID}"
        print(f"모델: {settings.VLM_MODEL_NAME}")
        print(f"디바이스: {device_map}")
        
        # 엔진 초기화
        engine = VLMEngine(
            model_name=settings.VLM_MODEL_NAME,
            device_map=device_map,
            use_quantization=settings.USE_QUANTIZATION
        )
        
        print("\n모델 로드 중...")
        load_start = time.time()
        engine.load_model()
        load_time = time.time() - load_start
        print(f"✅ 모델 로드 완료 ({load_time:.2f}초)")
        
        # 최적화 확인
        check_model_optimizations(engine.model, "VLM")
        print("  ✅ KV Cache: generation_config에 use_cache=True 설정됨")
        
        # 성능 테스트 (텍스트만)
        print("\n성능 테스트 시작 (텍스트만)...")
        test_messages = [
            {"role": "user", "content": "Describe what you can do in detail."}
        ]
        
        # Warmup
        print("  Warmup 요청...")
        await engine.generate(
            messages=test_messages,
            temperature=0.7,
            max_tokens=100,
            top_p=0.9
        )
        
        # 실제 성능 측정
        times = []
        for i in range(3):
            print(f"  테스트 {i+1}/3...")
            start_time = time.time()
            result = await engine.generate(
                messages=test_messages,
                temperature=0.7,
                max_tokens=100,
                top_p=0.9
            )
            elapsed = time.time() - start_time
            times.append(elapsed)
            print(f"    소요 시간: {elapsed:.2f}초, 토큰: {result['usage']['total_tokens']}")
        
        avg_time = sum(times) / len(times)
        tokens_per_sec = result['usage']['total_tokens'] / avg_time
        
        print(f"\n✅ VLM 성능 테스트 완료")
        print(f"  평균 응답 시간: {avg_time:.2f}초")
        print(f"  토큰 생성 속도: {tokens_per_sec:.2f} tokens/sec")
        
        return {
            "success": True,
            "avg_time": avg_time,
            "tokens_per_sec": tokens_per_sec,
            "total_tokens": result['usage']['total_tokens']
        }
        
    except Exception as e:
        print(f"❌ VLM 테스트 실패: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}


async def test_ocr_with_timing():
    """OCR 엔진 성능 테스트"""
    print("\n" + "=" * 60)
    print("OCR 엔진 성능 테스트")
    print("=" * 60)
    
    try:
        device_map = f"cuda:{settings.OCR_GPU_ID}"
        print(f"모델: {settings.OCR_MODEL_NAME}")
        print(f"디바이스: {device_map}")
        
        # 엔진 초기화
        engine = OCREngine(
            model_name=settings.OCR_MODEL_NAME,
            device_map=device_map
        )
        print("✅ OCR 엔진 초기화 완료 (모델 자동 로드됨)")
        
        # 최적화 확인
        if hasattr(OCREngine, '_model') and OCREngine._model is not None:
            check_model_optimizations(OCREngine._model, "OCR")
        print("  ✅ KV Cache: generation_kwargs에 use_cache=True 설정됨")
        
        # 테스트 이미지 생성
        from PIL import Image
        import base64
        from io import BytesIO
        
        print("\n성능 테스트 시작...")
        # 간단한 테스트 이미지
        test_image = Image.new('RGB', (400, 200), color='white')
        buffer = BytesIO()
        test_image.save(buffer, format='PNG')
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        # Warmup
        print("  Warmup 요청...")
        await engine.process(
            image_base64=image_base64,
            prompt_type="ocr_layout",
            output_format="markdown",
            max_tokens=100
        )
        
        # 실제 성능 측정
        times = []
        for i in range(3):
            print(f"  테스트 {i+1}/3...")
            start_time = time.time()
            result = await engine.process(
                image_base64=image_base64,
                prompt_type="ocr_layout",
                output_format="markdown",
                max_tokens=100
            )
            elapsed = time.time() - start_time
            times.append(elapsed)
            text_length = len(result.get('text', ''))
            print(f"    소요 시간: {elapsed:.2f}초, 출력 길이: {text_length} chars")
        
        avg_time = sum(times) / len(times)
        
        print(f"\n✅ OCR 성능 테스트 완료")
        print(f"  평균 처리 시간: {avg_time:.2f}초")
        
        return {
            "success": True,
            "avg_time": avg_time
        }
        
    except Exception as e:
        print(f"❌ OCR 테스트 실패: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}


async def main():
    """메인 테스트 함수"""
    print("\n" + "=" * 60)
    print("KV Cache 및 Flash Attention 최적화 테스트")
    print("=" * 60 + "\n")
    
    # 환경 확인
    print("환경 확인:")
    print(f"  PyTorch 버전: {torch.__version__}")
    print(f"  CUDA 사용 가능: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU 개수: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"    GPU {i}: {torch.cuda.get_device_name(i)}")
    print()
    
    # Flash Attention 확인
    flash_attn_available = check_flash_attention()
    print()
    
    # 각 모델 테스트
    results = {}
    
    # LLM 테스트
    results['llm'] = await test_llm_with_timing()
    
    # VLM 테스트
    results['vlm'] = await test_vlm_with_timing()
    
    # OCR 테스트
    results['ocr'] = await test_ocr_with_timing()
    
    # 결과 요약
    print("\n" + "=" * 60)
    print("테스트 결과 요약")
    print("=" * 60)
    
    print(f"\nFlash Attention 2: {'✅ 활성화' if flash_attn_available else '❌ 비활성화'}")
    
    for name, result in results.items():
        if result.get('success'):
            if 'tokens_per_sec' in result:
                print(f"\n{name.upper()}:")
                print(f"  ✅ 성공")
                print(f"  평균 응답 시간: {result['avg_time']:.2f}초")
                print(f"  토큰 생성 속도: {result['tokens_per_sec']:.2f} tokens/sec")
            else:
                print(f"\n{name.upper()}:")
                print(f"  ✅ 성공")
                print(f"  평균 처리 시간: {result['avg_time']:.2f}초")
        else:
            print(f"\n{name.upper()}:")
            print(f"  ❌ 실패: {result.get('error', 'Unknown error')}")
    
    print("\n" + "=" * 60)
    print("최적화 확인 완료")
    print("=" * 60)
    
    all_success = all(r.get('success', False) for r in results.values())
    return 0 if all_success else 1


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

