#!/usr/bin/env python3
"""빠른 성능 비교 테스트"""

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

async def test_llm(use_compile: bool):
    """LLM 테스트"""
    print(f"\n{'='*60}")
    print(f"LLM 테스트: torch.compile() {'활성화' if use_compile else '비활성화'}")
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
        
        test_messages = [
            {"role": "user", "content": "Write a short story about a robot learning to paint. Make it exactly 100 words."}
        ]
        
        # Warmup (3회)
        print("Warmup 실행 중 (3회)...")
        for i in range(3):
            await engine.generate(
                messages=test_messages,
                temperature=0.7,
                max_tokens=150,
                top_p=0.9
            )
            print(f"  Warmup {i+1}/3 완료")
        
        # 실제 벤치마크 (5회)
        print("\n벤치마크 실행 중 (5회)...")
        times = []
        for i in range(5):
            start_time = time.time()
            result = await engine.generate(
                messages=test_messages,
                temperature=0.7,
                max_tokens=150,
                top_p=0.9
            )
            elapsed = time.time() - start_time
            times.append(elapsed)
            print(f"  실행 {i+1}/5: {elapsed:.2f}초, {result['usage']['total_tokens']} 토큰")
        
        avg_time = sum(times) / len(times)
        avg_tokens = result['usage']['total_tokens']
        tokens_per_sec = avg_tokens / avg_time
        
        return {
            "avg_time": avg_time,
            "tokens_per_sec": tokens_per_sec,
            "times": times
        }
    finally:
        os.environ.pop("DISABLE_TORCH_COMPILE", None)

async def main():
    print("="*60)
    print("빠른 성능 비교 테스트")
    print("="*60)
    
    # torch.compile() 비활성화
    result_no_compile = await test_llm(use_compile=False)
    
    print("\n모델 언로드를 위해 15초 대기 중...")
    await asyncio.sleep(15)
    
    # torch.compile() 활성화
    result_with_compile = await test_llm(use_compile=True)
    
    # 결과 비교
    print("\n" + "="*60)
    print("결과 비교")
    print("="*60)
    
    no_avg = result_no_compile["avg_time"]
    yes_avg = result_with_compile["avg_time"]
    improvement = ((no_avg - yes_avg) / no_avg) * 100
    
    print(f"\n[LLM]")
    print(f"  torch.compile() 비활성화: {no_avg:.2f}초 ({result_no_compile['tokens_per_sec']:.2f} tokens/sec)")
    print(f"  torch.compile() 활성화:    {yes_avg:.2f}초 ({result_with_compile['tokens_per_sec']:.2f} tokens/sec)")
    print(f"  개선율: {improvement:+.1f}% ({'개선' if improvement > 0 else '저하'})")
    
    if improvement > 0:
        print(f"\n✅ torch.compile()이 약 {improvement:.1f}% 성능을 개선했습니다!")
    else:
        print(f"\n⚠️  torch.compile()이 이 경우에는 성능 개선 효과가 없었습니다.")
        print("   (첫 실행 시 컴파일 오버헤드가 있을 수 있습니다)")

if __name__ == "__main__":
    asyncio.run(main())

