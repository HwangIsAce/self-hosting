#!/usr/bin/env python3
"""torch.compile() 성능 테스트 - 개선된 버전 (긴 시퀀스)"""

import asyncio
import os
import sys
import time
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from api.config.logging_config import setup_logging
from api.config.settings import settings
from api.infrastructure.models.llm_engine import LLMEngine

setup_logging()

async def test_llm(use_compile: bool, test_name: str):
    """LLM 테스트"""
    print(f"\n{'='*60}")
    print(f"{test_name}: torch.compile() {'활성화' if use_compile else '비활성화'}")
    print(f"{'='*60}")
    
    # 환경 변수 설정
    if not use_compile:
        os.environ["DISABLE_TORCH_COMPILE"] = "1"
    else:
        os.environ.pop("DISABLE_TORCH_COMPILE", None)
    
    try:
        device_map = f"cuda:{settings.LLM_GPU_ID}"
        engine = LLMEngine(
            model_name=settings.LLM_MODEL_NAME,
            device_map=device_map,
            use_quantization=settings.USE_QUANTIZATION
        )
        
        print("모델 로드 중...")
        load_start = time.time()
        engine.load_model()
        load_time = time.time() - load_start
        print(f"모델 로드 시간: {load_time:.2f}초")
        
        # 컴파일 확인
        if hasattr(engine.model, '_orig_mod'):
            print("✅ torch.compile() 적용됨")
        else:
            print("❌ torch.compile() 적용 안 됨")
        
        # 더 긴 프롬프트로 테스트
        test_messages = [
            {"role": "user", "content": "Write a detailed technical article about artificial intelligence, machine learning, and deep learning. Explain the differences, use cases, and future prospects. Make it comprehensive and at least 500 words long."}
        ]
        
        # Warmup (5회 - 컴파일된 모델은 더 많은 warmup 필요)
        print("Warmup 실행 중 (5회)...")
        for i in range(5):
            await engine.generate(
                messages=test_messages,
                temperature=0.7,
                max_tokens=300,  # 더 긴 출력
                top_p=0.9
            )
            print(f"  Warmup {i+1}/5 완료")
        
        # 실제 벤치마크 (10회)
        print("\n벤치마크 실행 중 (10회)...")
        times = []
        token_counts = []
        for i in range(10):
            start_time = time.time()
            result = await engine.generate(
                messages=test_messages,
                temperature=0.7,
                max_tokens=300,
                top_p=0.9
            )
            elapsed = time.time() - start_time
            times.append(elapsed)
            token_counts.append(result['usage']['total_tokens'])
            tokens_per_sec = result['usage']['total_tokens'] / elapsed
            print(f"  실행 {i+1}/10: {elapsed:.2f}초, {result['usage']['total_tokens']} 토큰 ({tokens_per_sec:.2f} tokens/sec)")
        
        avg_time = sum(times) / len(times)
        avg_tokens = sum(token_counts) / len(token_counts)
        tokens_per_sec = avg_tokens / avg_time
        
        return {
            "avg_time": avg_time,
            "avg_tokens": avg_tokens,
            "tokens_per_sec": tokens_per_sec,
            "times": times,
            "token_counts": token_counts
        }
    finally:
        os.environ.pop("DISABLE_TORCH_COMPILE", None)

async def main():
    print("="*60)
    print("torch.compile() 성능 비교 테스트 (개선된 버전)")
    print("="*60)
    print("\n긴 시퀀스와 더 많은 반복으로 테스트합니다.")
    print("torch.compile()은 긴 시퀀스에서 더 효과적입니다.\n")
    
    # torch.compile() 비활성화
    result_no_compile = await test_llm(use_compile=False, test_name="LLM 테스트")
    
    print("\n모델 언로드를 위해 20초 대기 중...")
    await asyncio.sleep(20)
    
    # torch.compile() 활성화
    result_with_compile = await test_llm(use_compile=True, test_name="LLM 테스트")
    
    # 결과 비교
    print("\n" + "="*60)
    print("결과 비교")
    print("="*60)
    
    no_avg = result_no_compile["avg_time"]
    yes_avg = result_with_compile["avg_time"]
    improvement = ((no_avg - yes_avg) / no_avg) * 100
    
    no_tps = result_no_compile["tokens_per_sec"]
    yes_tps = result_with_compile["tokens_per_sec"]
    tps_improvement = ((yes_tps - no_tps) / no_tps) * 100
    
    print(f"\n[LLM 성능 비교]")
    print(f"  torch.compile() 비활성화:")
    print(f"    평균 시간: {no_avg:.2f}초")
    print(f"    평균 토큰: {result_no_compile['avg_tokens']:.1f} 토큰")
    print(f"    처리 속도: {no_tps:.2f} tokens/sec")
    print(f"\n  torch.compile() 활성화:")
    print(f"    평균 시간: {yes_avg:.2f}초")
    print(f"    평균 토큰: {result_with_compile['avg_tokens']:.1f} 토큰")
    print(f"    처리 속도: {yes_tps:.2f} tokens/sec")
    print(f"\n  성능 개선:")
    print(f"    시간 개선율: {improvement:+.1f}% ({'개선' if improvement > 0 else '저하'})")
    print(f"    처리 속도 개선율: {tps_improvement:+.1f}% ({'개선' if tps_improvement > 0 else '저하'})")
    
    if improvement > 0:
        speedup = no_avg / yes_avg
        print(f"\n✅ torch.compile()이 약 {improvement:.1f}% 성능을 개선했습니다!")
        print(f"   속도 향상: {speedup:.2f}x")
    else:
        print(f"\n⚠️  torch.compile()이 이 테스트에서는 성능 개선 효과가 없었습니다.")
        print("   (짧은 시퀀스에서는 오버헤드가 더 클 수 있습니다)")
        print("   (더 긴 시퀀스나 배치 처리에서 효과가 더 클 수 있습니다)")

if __name__ == "__main__":
    asyncio.run(main())

