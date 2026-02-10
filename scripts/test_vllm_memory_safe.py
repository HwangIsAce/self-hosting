#!/usr/bin/env python3
"""vLLM 메모리 안전 테스트 (Flash Attention 없이)"""

import asyncio
import sys
import time
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from api.config.logging_config import setup_logging
from api.config.settings import settings
from api.infrastructure.models.vllm_engine import VLLMEngine

setup_logging()

async def test_vllm_load():
    """vLLM 모델 로드 테스트"""
    print("="*60)
    print("vLLM 모델 로드 테스트 (메모리 안전)")
    print("="*60)
    print(f"모델: {settings.LLM_MODEL_NAME}")
    print(f"GPU: {settings.LLM_GPU_ID}")
    print(f"GPU 메모리 사용률: {settings.VLLM_GPU_MEMORY_UTILIZATION}")
    print()
    
    try:
        engine = VLLMEngine(
            model_name=settings.LLM_MODEL_NAME,
            device_map=f"cuda:{settings.LLM_GPU_ID}",
            use_quantization=settings.USE_QUANTIZATION
        )
        
        print("모델 로드 시작...")
        start = time.time()
        engine.load_model()
        elapsed = time.time() - start
        print(f"✅ 모델 로드 완료: {elapsed:.2f}초")
        
        # 간단한 생성 테스트
        print("\n간단한 생성 테스트...")
        test_messages = [
            {"role": "user", "content": "Say hello in one sentence."}
        ]
        
        start = time.time()
        result = await engine.generate(test_messages, max_tokens=50)
        elapsed = time.time() - start
        
        print(f"✅ 생성 완료: {elapsed:.2f}초")
        print(f"✅ 생성된 텍스트: {result['choices'][0]['message']['content'][:100]}...")
        print(f"✅ 토큰 사용량: {result['usage']['total_tokens']} 토큰")
        
        return True
        
    except Exception as e:
        print(f"❌ 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_vllm_load())
    sys.exit(0 if success else 1)

