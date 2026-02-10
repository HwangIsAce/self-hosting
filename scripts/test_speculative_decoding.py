#!/usr/bin/env python3
"""Speculative Decoding 테스트 스크립트"""

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

async def test_speculative_decoding():
    """Speculative Decoding 테스트"""
    print("="*70)
    print("Speculative Decoding 테스트")
    print("="*70)
    
    print(f"\n설정 확인:")
    print(f"  - Target Model: {settings.LLM_MODEL_NAME}")
    print(f"  - Speculative Model: {settings.VLLM_SPECULATIVE_MODEL}")
    print(f"  - Num Speculative Tokens: {settings.VLLM_NUM_SPECULATIVE_TOKENS}")
    
    if not settings.VLLM_SPECULATIVE_MODEL:
        print("\n❌ Speculative Decoding이 비활성화되어 있습니다.")
        print("   .env 파일에 VLLM_SPECULATIVE_MODEL을 설정하세요.")
        return
    
    print("\n✅ Speculative Decoding이 활성화되어 있습니다!")
    print("\n모델 로드 중...")
    
    engine = VLLMEngine(
        model_name=settings.LLM_MODEL_NAME,
        device_map=f"cuda:{settings.LLM_GPU_ID}",
        use_quantization=settings.USE_QUANTIZATION
    )
    
    try:
        engine.load_model()
        print("✅ 모델 로드 완료\n")
        
        # 테스트 1: 짧은 응답
        print("1. 짧은 응답 테스트 (max_tokens=50)")
        messages = [{"role": "user", "content": "Say hello in one sentence."}]
        
        start = time.time()
        result = await engine.generate(
            messages=messages,
            max_tokens=50,
            temperature=0.7
        )
        elapsed = time.time() - start
        
        content = result["choices"][0]["message"]["content"]
        tokens = result["usage"]["total_tokens"]
        tps = tokens / elapsed if elapsed > 0 else 0
        
        print(f"✅ 완료: {elapsed:.2f}초")
        print(f"✅ 토큰: {tokens} 토큰")
        print(f"✅ 속도: {tps:.2f} tokens/sec")
        print(f"✅ 응답: {content[:100]}...")
        
        # 테스트 2: 중간 길이 응답
        print("\n2. 중간 길이 응답 테스트 (max_tokens=200)")
        messages = [{"role": "user", "content": "Write a short paragraph about artificial intelligence."}]
        
        start = time.time()
        result = await engine.generate(
            messages=messages,
            max_tokens=200,
            temperature=0.7
        )
        elapsed = time.time() - start
        
        content = result["choices"][0]["message"]["content"]
        tokens = result["usage"]["total_tokens"]
        tps = tokens / elapsed if elapsed > 0 else 0
        
        print(f"✅ 완료: {elapsed:.2f}초")
        print(f"✅ 토큰: {tokens} 토큰")
        print(f"✅ 속도: {tps:.2f} tokens/sec")
        print(f"✅ 응답: {content[:150]}...")
        
        # 테스트 3: 배치 처리
        print("\n3. 배치 처리 테스트 (5개 요청)")
        messages_list = [
            [{"role": "user", "content": f"Say hello for request {i}."}]
            for i in range(5)
        ]
        
        start = time.time()
        results = await engine.generate_batch(
            messages_list=messages_list,
            max_tokens=50,
            allow_partial_failure=True
        )
        elapsed = time.time() - start
        
        successful = sum(1 for r in results if 'error' not in r)
        total_tokens = sum(
            r.get('usage', {}).get('total_tokens', 0) 
            for r in results 
            if isinstance(r, dict) and 'usage' in r
        )
        tps = total_tokens / elapsed if elapsed > 0 else 0
        
        print(f"✅ 완료: {elapsed:.2f}초")
        print(f"✅ 성공: {successful}/5 요청")
        print(f"✅ 총 토큰: {total_tokens} 토큰")
        print(f"✅ 속도: {tps:.2f} tokens/sec")
        
        print("\n" + "="*70)
        print("✅ Speculative Decoding 테스트 완료!")
        print("="*70)
        print("\n💡 참고:")
        print("   - Speculative Decoding은 짧은 응답에서 2-3배, 긴 응답에서 3-5배 빠를 수 있습니다.")
        print("   - 실제 성능 향상은 모델 크기와 하드웨어에 따라 다릅니다.")
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()

async def main():
    await test_speculative_decoding()

if __name__ == "__main__":
    asyncio.run(main())

