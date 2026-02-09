#!/usr/bin/env python3
"""torch.compile() 및 KV Cache 최적화 성능 벤치마크 비교"""

import asyncio
import sys
import time
import torch
from pathlib import Path
from typing import Dict, Any, List

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from api.config.logging_config import setup_logging, get_logger
from api.config.settings import settings
from api.infrastructure.models.llm_engine import LLMEngine
from api.infrastructure.models.vlm_engine import VLMEngine
from api.infrastructure.models.ocr_engine import OCREngine

# 로깅 설정 (에러만)
setup_logging()
logger = get_logger(__name__)


async def benchmark_llm(use_compile: bool = True, num_runs: int = 5):
    """LLM 성능 벤치마크"""
    print(f"\n{'='*60}")
    print(f"LLM 벤치마크: torch.compile() {'활성화' if use_compile else '비활성화'}")
    print(f"{'='*60}")
    
    # 환경 변수로 torch.compile() 제어
    import os
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
        
        test_messages = [
            {"role": "user", "content": "Write a short story about a robot learning to paint. Make it exactly 100 words."}
        ]
        
        # Warmup (컴파일된 모델의 첫 실행 오버헤드 제거)
        print("Warmup 실행 중...")
        for _ in range(2):
            await engine.generate(
                messages=test_messages,
                temperature=0.7,
                max_tokens=150,
                top_p=0.9
            )
        
        # 실제 벤치마크
        print(f"벤치마크 실행 중 ({num_runs}회)...")
        times = []
        token_counts = []
        
        for i in range(num_runs):
            start_time = time.time()
            result = await engine.generate(
                messages=test_messages,
                temperature=0.7,
                max_tokens=150,
                top_p=0.9
            )
            elapsed = time.time() - start_time
            times.append(elapsed)
            token_counts.append(result['usage']['total_tokens'])
            print(f"  실행 {i+1}/{num_runs}: {elapsed:.2f}초, {result['usage']['total_tokens']} 토큰")
        
        avg_time = sum(times) / len(times)
        avg_tokens = sum(token_counts) / len(token_counts)
        tokens_per_sec = avg_tokens / avg_time
        
        return {
            "avg_time": avg_time,
            "avg_tokens": avg_tokens,
            "tokens_per_sec": tokens_per_sec,
            "times": times
        }
    finally:
        # 환경 변수 복원
        import os
        os.environ.pop("DISABLE_TORCH_COMPILE", None)


async def benchmark_vlm(use_compile: bool = True, num_runs: int = 5):
    """VLM 성능 벤치마크"""
    print(f"\n{'='*60}")
    print(f"VLM 벤치마크: torch.compile() {'활성화' if use_compile else '비활성화'}")
    print(f"{'='*60}")
    
    # 환경 변수로 torch.compile() 제어
    import os
    if not use_compile:
        os.environ["DISABLE_TORCH_COMPILE"] = "1"
    else:
        os.environ.pop("DISABLE_TORCH_COMPILE", None)
    
    try:
        device_map = f"cuda:{settings.VLM_GPU_ID}"
        engine = VLMEngine(
            model_name=settings.VLM_MODEL_NAME,
            device_map=device_map,
            use_quantization=settings.USE_QUANTIZATION
        )
        
        print("모델 로드 중...")
        load_start = time.time()
        engine.load_model()
        load_time = time.time() - load_start
        print(f"모델 로드 시간: {load_time:.2f}초")
        
        test_messages = [
            {"role": "user", "content": "Describe what you can do in detail."}
        ]
        
        # Warmup
        print("Warmup 실행 중...")
        for _ in range(2):
            await engine.generate(
                messages=test_messages,
                temperature=0.7,
                max_tokens=100,
                top_p=0.9
            )
        
        # 벤치마크
        print(f"벤치마크 실행 중 ({num_runs}회)...")
        times = []
        token_counts = []
        
        for i in range(num_runs):
            start_time = time.time()
            result = await engine.generate(
                messages=test_messages,
                temperature=0.7,
                max_tokens=100,
                top_p=0.9
            )
            elapsed = time.time() - start_time
            times.append(elapsed)
            token_counts.append(result['usage']['total_tokens'])
            print(f"  실행 {i+1}/{num_runs}: {elapsed:.2f}초, {result['usage']['total_tokens']} 토큰")
        
        avg_time = sum(times) / len(times)
        avg_tokens = sum(token_counts) / len(token_counts)
        tokens_per_sec = avg_tokens / avg_time
        
        return {
            "avg_time": avg_time,
            "avg_tokens": avg_tokens,
            "tokens_per_sec": tokens_per_sec,
            "times": times
        }
    finally:
        # 환경 변수 복원
        import os
        os.environ.pop("DISABLE_TORCH_COMPILE", None)


async def benchmark_ocr(use_compile: bool = True, num_runs: int = 5):
    """OCR 성능 벤치마크"""
    print(f"\n{'='*60}")
    print(f"OCR 벤치마크: torch.compile() {'활성화' if use_compile else '비활성화'}")
    print(f"{'='*60}")
    
    # 환경 변수로 torch.compile() 제어
    import os
    if not use_compile:
        os.environ["DISABLE_TORCH_COMPILE"] = "1"
    else:
        os.environ.pop("DISABLE_TORCH_COMPILE", None)
    
    try:
        device_map = f"cuda:{settings.OCR_GPU_ID}"
        engine = OCREngine(model_name="datalab-to/chandra", device_map=device_map)
        
        print("모델 로드 중...")
        load_start = time.time()
        # 모델은 첫 요청 시 자동 로드됨
        await engine.process(
            image_base64="iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",  # 1x1 빈 이미지
            max_tokens=100
        )
        load_time = time.time() - load_start
        print(f"모델 로드 시간: {load_time:.2f}초")
        
        # 테스트 이미지 (1x1 빈 이미지)
        test_image = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        
        # Warmup
        print("Warmup 실행 중...")
        for _ in range(2):
            await engine.process(
                image_base64=test_image,
                max_tokens=100
            )
        
        # 벤치마크
        print(f"벤치마크 실행 중 ({num_runs}회)...")
        times = []
        
        for i in range(num_runs):
            start_time = time.time()
            result = await engine.process(
                image_base64=test_image,
                max_tokens=100
            )
            elapsed = time.time() - start_time
            times.append(elapsed)
            print(f"  실행 {i+1}/{num_runs}: {elapsed:.2f}초")
        
        avg_time = sum(times) / len(times)
        
        return {
            "avg_time": avg_time,
            "times": times
        }
    finally:
        # 환경 변수 복원
        import os
        os.environ.pop("DISABLE_TORCH_COMPILE", None)


async def main():
    """메인 벤치마크 실행"""
    print("="*60)
    print("최적화 성능 비교 벤치마크")
    print("="*60)
    print("\n이 테스트는 torch.compile() 적용 전후의 성능을 비교합니다.")
    print("각 모델에 대해 KV Cache는 항상 활성화되어 있습니다.\n")
    
    results = {}
    
    # LLM 벤치마크
    print("\n" + "="*60)
    print("1단계: LLM 벤치마크 (torch.compile() 비활성화)")
    print("="*60)
    results['llm_no_compile'] = await benchmark_llm(use_compile=False, num_runs=5)
    
    # 모델 언로드를 위해 잠시 대기
    print("\n모델 언로드를 위해 10초 대기 중...")
    await asyncio.sleep(10)
    
    print("\n" + "="*60)
    print("2단계: LLM 벤치마크 (torch.compile() 활성화)")
    print("="*60)
    results['llm_with_compile'] = await benchmark_llm(use_compile=True, num_runs=5)
    
    # VLM 벤치마크
    print("\n모델 언로드를 위해 10초 대기 중...")
    await asyncio.sleep(10)
    
    print("\n" + "="*60)
    print("3단계: VLM 벤치마크 (torch.compile() 비활성화)")
    print("="*60)
    results['vlm_no_compile'] = await benchmark_vlm(use_compile=False, num_runs=5)
    
    print("\n모델 언로드를 위해 10초 대기 중...")
    await asyncio.sleep(10)
    
    print("\n" + "="*60)
    print("4단계: VLM 벤치마크 (torch.compile() 활성화)")
    print("="*60)
    results['vlm_with_compile'] = await benchmark_vlm(use_compile=True, num_runs=5)
    
    # OCR 벤치마크
    print("\n모델 언로드를 위해 10초 대기 중...")
    await asyncio.sleep(10)
    
    print("\n" + "="*60)
    print("5단계: OCR 벤치마크 (torch.compile() 비활성화)")
    print("="*60)
    results['ocr_no_compile'] = await benchmark_ocr(use_compile=False, num_runs=5)
    
    print("\n모델 언로드를 위해 10초 대기 중...")
    await asyncio.sleep(10)
    
    print("\n" + "="*60)
    print("6단계: OCR 벤치마크 (torch.compile() 활성화)")
    print("="*60)
    results['ocr_with_compile'] = await benchmark_ocr(use_compile=True, num_runs=5)
    
    # 결과 요약
    print("\n" + "="*60)
    print("벤치마크 결과 요약")
    print("="*60)
    
    # LLM 결과
    llm_no = results['llm_no_compile']
    llm_yes = results['llm_with_compile']
    llm_improvement = ((llm_no['avg_time'] - llm_yes['avg_time']) / llm_no['avg_time']) * 100
    
    print(f"\n[LLM]")
    print(f"  torch.compile() 비활성화: {llm_no['avg_time']:.2f}초 ({llm_no['tokens_per_sec']:.2f} tokens/sec)")
    print(f"  torch.compile() 활성화:    {llm_yes['avg_time']:.2f}초 ({llm_yes['tokens_per_sec']:.2f} tokens/sec)")
    print(f"  개선율: {llm_improvement:+.1f}% ({'개선' if llm_improvement > 0 else '저하'})")
    
    # VLM 결과
    vlm_no = results['vlm_no_compile']
    vlm_yes = results['vlm_with_compile']
    vlm_improvement = ((vlm_no['avg_time'] - vlm_yes['avg_time']) / vlm_no['avg_time']) * 100
    
    print(f"\n[VLM]")
    print(f"  torch.compile() 비활성화: {vlm_no['avg_time']:.2f}초 ({vlm_no['tokens_per_sec']:.2f} tokens/sec)")
    print(f"  torch.compile() 활성화:    {vlm_yes['avg_time']:.2f}초 ({vlm_yes['tokens_per_sec']:.2f} tokens/sec)")
    print(f"  개선율: {vlm_improvement:+.1f}% ({'개선' if vlm_improvement > 0 else '저하'})")
    
    # OCR 결과
    ocr_no = results['ocr_no_compile']
    ocr_yes = results['ocr_with_compile']
    ocr_improvement = ((ocr_no['avg_time'] - ocr_yes['avg_time']) / ocr_no['avg_time']) * 100
    
    print(f"\n[OCR]")
    print(f"  torch.compile() 비활성화: {ocr_no['avg_time']:.2f}초")
    print(f"  torch.compile() 활성화:    {ocr_yes['avg_time']:.2f}초")
    print(f"  개선율: {ocr_improvement:+.1f}% ({'개선' if ocr_improvement > 0 else '저하'})")
    
    print("\n" + "="*60)
    print("벤치마크 완료")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())

