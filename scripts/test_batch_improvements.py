#!/usr/bin/env python3
"""배치 처리 개선사항 테스트"""

import asyncio
import sys
import time
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from api.config.logging_config import setup_logging
from api.config.settings import settings
from api.infrastructure.models.vllm_engine import VLLMEngine
from api.infrastructure.models.llm_engine import LLMEngine
from api.infrastructure.utils.batch_processor import enrich_chunks_batch

setup_logging()

async def test_vllm_batch_improvements():
    """vLLM 배치 처리 개선사항 테스트"""
    print("="*70)
    print("vLLM 배치 처리 개선사항 테스트")
    print("="*70)
    
    engine = VLLMEngine(
        model_name=settings.LLM_MODEL_NAME,
        device_map=f"cuda:{settings.LLM_GPU_ID}",
        use_quantization=settings.USE_QUANTIZATION
    )
    engine.load_model()
    
    # 테스트 1: 일반 배치 처리
    print("\n1. 일반 배치 처리 테스트 (10개 요청)")
    messages_list = [
        [{"role": "user", "content": f"Say hello in one sentence for request {i}."}]
        for i in range(10)
    ]
    
    start = time.time()
    results = await engine.generate_batch(
        messages_list,
        max_tokens=50,
        allow_partial_failure=True
    )
    elapsed = time.time() - start
    
    successful = sum(1 for r in results if 'error' not in r)
    failed = len(results) - successful
    
    print(f"✅ 완료: {elapsed:.2f}초")
    print(f"✅ 성공: {successful}개, 실패: {failed}개")
    print(f"✅ 평균 시간: {elapsed/len(results):.2f}초/요청")
    
    # 테스트 2: 대량 배치 처리 (청크 처리)
    print("\n2. 대량 배치 처리 테스트 (50개 요청, 청크 처리)")
    messages_list_large = [
        [{"role": "user", "content": f"Write a one-sentence summary about topic {i}."}]
        for i in range(50)
    ]
    
    start = time.time()
    results_large = await engine.generate_batch(
        messages_list_large,
        max_tokens=100,
        allow_partial_failure=True,
        chunk_size=20  # 20개씩 청크로 나누기
    )
    elapsed_large = time.time() - start
    
    successful_large = sum(1 for r in results_large if 'error' not in r)
    failed_large = len(results_large) - successful_large
    total_tokens = sum(
        r.get('usage', {}).get('total_tokens', 0) 
        for r in results_large 
        if isinstance(r, dict) and 'usage' in r
    )
    tps = total_tokens / elapsed_large if elapsed_large > 0 else 0
    
    print(f"✅ 완료: {elapsed_large:.2f}초")
    print(f"✅ 성공: {successful_large}개, 실패: {failed_large}개")
    print(f"✅ 총 토큰: {total_tokens} 토큰")
    print(f"✅ 처리 속도: {tps:.2f} tokens/sec")
    
    # 테스트 3: enrich_chunks_batch 테스트
    print("\n3. enrich_chunks_batch 테스트 (30개 chunk)")
    chunks = [f"This is chunk {i} with some content." for i in range(30)]
    
    start = time.time()
    enrich_results = await enrich_chunks_batch(
        chunks,
        engine,
        max_tokens=200,
        allow_partial_failure=True
    )
    elapsed_enrich = time.time() - start
    
    successful_enrich = sum(1 for r in enrich_results if isinstance(r, dict) and 'error' not in r)
    failed_enrich = len(enrich_results) - successful_enrich
    
    print(f"✅ 완료: {elapsed_enrich:.2f}초")
    print(f"✅ 성공: {successful_enrich}개, 실패: {failed_enrich}개")
    print(f"✅ 평균 시간: {elapsed_enrich/len(chunks):.2f}초/chunk")
    
    return {
        "batch_10": {"time": elapsed, "successful": successful, "failed": failed},
        "batch_50": {"time": elapsed_large, "successful": successful_large, "failed": failed_large, "tps": tps},
        "enrich_30": {"time": elapsed_enrich, "successful": successful_enrich, "failed": failed_enrich}
    }

async def test_hf_batch():
    """HuggingFace 배치 처리 테스트"""
    print("\n" + "="*70)
    print("HuggingFace 배치 처리 테스트")
    print("="*70)
    
    engine = LLMEngine(
        model_name=settings.LLM_MODEL_NAME,
        device_map=f"cuda:{settings.LLM_GPU_ID}",
        use_quantization=settings.USE_QUANTIZATION
    )
    engine.load_model()
    
    # generate_batch 테스트
    print("\n1. HuggingFace generate_batch 테스트 (10개 요청)")
    messages_list = [
        [{"role": "user", "content": f"Say hello in one sentence for request {i}."}]
        for i in range(10)
    ]
    
    start = time.time()
    results = await engine.generate_batch(
        messages_list,
        max_tokens=50,
        allow_partial_failure=True
    )
    elapsed = time.time() - start
    
    successful = sum(1 for r in results if 'error' not in r)
    failed = len(results) - successful
    
    print(f"✅ 완료: {elapsed:.2f}초")
    print(f"✅ 성공: {successful}개, 실패: {failed}개")
    print(f"✅ 평균 시간: {elapsed/len(results):.2f}초/요청")
    
    return {"time": elapsed, "successful": successful, "failed": failed}

async def main():
    print("="*70)
    print("배치 처리 개선사항 테스트")
    print("="*70)
    print(f"모델: {settings.LLM_MODEL_NAME}")
    print(f"GPU: {settings.LLM_GPU_ID}")
    print(f"USE_VLLM: {settings.USE_VLLM}")
    
    if settings.USE_VLLM:
        vllm_results = await test_vllm_batch_improvements()
        print("\n" + "="*70)
        print("vLLM 배치 처리 결과 요약")
        print("="*70)
        print(f"배치 10개: {vllm_results['batch_10']['time']:.2f}초, "
              f"성공: {vllm_results['batch_10']['successful']}, "
              f"실패: {vllm_results['batch_10']['failed']}")
        print(f"배치 50개: {vllm_results['batch_50']['time']:.2f}초, "
              f"성공: {vllm_results['batch_50']['successful']}, "
              f"실패: {vllm_results['batch_50']['failed']}, "
              f"TPS: {vllm_results['batch_50']['tps']:.2f}")
        print(f"Enrich 30개: {vllm_results['enrich_30']['time']:.2f}초, "
              f"성공: {vllm_results['enrich_30']['successful']}, "
              f"실패: {vllm_results['enrich_30']['failed']}")
    
    # HuggingFace 테스트는 메모리 정리 후 실행
    print("\n메모리 정리 중... (30초 대기)")
    await asyncio.sleep(30)
    
    hf_results = await test_hf_batch()
    print("\n" + "="*70)
    print("HuggingFace 배치 처리 결과 요약")
    print("="*70)
    print(f"배치 10개: {hf_results['time']:.2f}초, "
          f"성공: {hf_results['successful']}, "
          f"실패: {hf_results['failed']}")
    
    print("\n" + "="*70)
    print("테스트 완료!")
    print("="*70)

if __name__ == "__main__":
    asyncio.run(main())

