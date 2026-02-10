#!/usr/bin/env python3
"""vLLM 단독 성능 테스트 (HuggingFace와 별도 실행)"""

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

async def test_vllm_single(num_runs: int = 5):
    """vLLM 단일 요청 성능 테스트"""
    print(f"\n{'='*60}")
    print(f"vLLM 단일 요청 성능 테스트 ({num_runs}회)")
    print(f"{'='*60}")
    
    engine = VLLMEngine(
        model_name=settings.LLM_MODEL_NAME,
        device_map=f"cuda:{settings.LLM_GPU_ID}",
        use_quantization=settings.USE_QUANTIZATION
    )
    
    print("모델 로드 중...")
    engine.load_model()
    print("✅ 모델 로드 완료")
    
    test_messages = [
        {"role": "user", "content": "Write a short story about AI in exactly 100 words."}
    ]
    
    # Warmup
    print("Warmup 실행 중...")
    for _ in range(2):
        await engine.generate(test_messages, max_tokens=150)
    
    # 벤치마크
    print(f"벤치마크 실행 중 ({num_runs}회)...")
    times = []
    token_counts = []
    
    for i in range(num_runs):
        start = time.time()
        result = await engine.generate(test_messages, max_tokens=150)
        elapsed = time.time() - start
        times.append(elapsed)
        tokens = result['usage']['total_tokens']
        token_counts.append(tokens)
        tps = tokens / elapsed
        print(f"  실행 {i+1}/{num_runs}: {elapsed:.2f}초, {tokens} 토큰 ({tps:.2f} tokens/sec)")
    
    avg_time = sum(times) / len(times)
    avg_tokens = sum(token_counts) / len(token_counts)
    avg_tps = avg_tokens / avg_time
    
    print(f"\n✅ 평균: {avg_time:.2f}초, {avg_tps:.2f} tokens/sec")
    return {"avg_time": avg_time, "tokens_per_sec": avg_tps}

async def test_vllm_batch(batch_size: int = 10):
    """vLLM 배치 요청 성능 테스트"""
    print(f"\n{'='*60}")
    print(f"vLLM 배치 요청 성능 테스트 ({batch_size}개 동시 요청)")
    print(f"{'='*60}")
    
    engine = VLLMEngine(
        model_name=settings.LLM_MODEL_NAME,
        device_map=f"cuda:{settings.LLM_GPU_ID}",
        use_quantization=settings.USE_QUANTIZATION
    )
    engine.load_model()
    
    messages_list = [
        [{"role": "user", "content": f"Write a one-sentence summary about topic {i}."}]
        for i in range(batch_size)
    ]
    
    print(f"vLLM generate_batch 사용...")
    start = time.time()
    results = await engine.generate_batch(messages_list, max_tokens=100)
    elapsed = time.time() - start
    
    total_tokens = sum(r['usage']['total_tokens'] for r in results)
    avg_time_per_request = elapsed / len(results)
    tps = total_tokens / elapsed
    
    print(f"\n✅ 완료: {elapsed:.2f}초")
    print(f"✅ 처리된 요청: {len(results)}개")
    print(f"✅ 총 토큰: {total_tokens} 토큰")
    print(f"✅ 평균 시간: {avg_time_per_request:.2f}초/요청")
    print(f"✅ 처리 속도: {tps:.2f} tokens/sec")
    
    return {"total_time": elapsed, "tokens_per_sec": tps, "avg_time_per_request": avg_time_per_request}

async def test_vllm_chunk_enrich(num_chunks: int = 50):
    """vLLM Chunk enrich 성능 테스트"""
    print(f"\n{'='*60}")
    print(f"vLLM Chunk Enrich 성능 테스트 ({num_chunks}개 chunk)")
    print(f"{'='*60}")
    
    engine = VLLMEngine(
        model_name=settings.LLM_MODEL_NAME,
        device_map=f"cuda:{settings.LLM_GPU_ID}",
        use_quantization=settings.USE_QUANTIZATION
    )
    engine.load_model()
    
    chunks = [f"This is chunk {i} with some content about topic {i}." for i in range(num_chunks)]
    messages_list = [
        [{"role": "user", "content": f"Enrich this chunk with additional context: {chunk}"}]
        for chunk in chunks
    ]
    
    print(f"vLLM generate_batch 사용...")
    start = time.time()
    results = await engine.generate_batch(messages_list, max_tokens=200)
    elapsed = time.time() - start
    
    total_tokens = sum(r['usage']['total_tokens'] for r in results)
    avg_time_per_chunk = elapsed / len(results)
    tps = total_tokens / elapsed
    
    print(f"\n✅ 완료: {elapsed:.2f}초")
    print(f"✅ 처리된 chunk: {len(results)}개")
    print(f"✅ 총 토큰: {total_tokens} 토큰")
    print(f"✅ 평균 시간: {avg_time_per_chunk:.2f}초/chunk")
    print(f"✅ 처리 속도: {tps:.2f} tokens/sec")
    
    return {"total_time": elapsed, "tokens_per_sec": tps, "avg_time_per_chunk": avg_time_per_chunk}

async def main():
    print("="*60)
    print("vLLM 성능 테스트")
    print("="*60)
    
    # 단일 요청 테스트
    single_result = await test_vllm_single(num_runs=5)
    
    # 배치 요청 테스트
    batch_result = await test_vllm_batch(batch_size=10)
    
    # Chunk enrich 테스트
    chunk_result = await test_vllm_chunk_enrich(num_chunks=50)
    
    print("\n" + "="*60)
    print("테스트 완료!")
    print("="*60)
    print("\n[HuggingFace 기준 (이전 테스트)]")
    print("  단일 요청: ~2.8초, ~47 tokens/sec")
    print("  배치 10개: ~10.66초, ~70 tokens/sec")
    print("  Chunk 50개: ~295.93초, ~41 tokens/sec")
    print("\n[vLLM 결과]")
    print(f"  단일 요청: {single_result['avg_time']:.2f}초, {single_result['tokens_per_sec']:.2f} tokens/sec")
    print(f"  배치 10개: {batch_result['total_time']:.2f}초, {batch_result['tokens_per_sec']:.2f} tokens/sec")
    print(f"  Chunk 50개: {chunk_result['total_time']:.2f}초, {chunk_result['tokens_per_sec']:.2f} tokens/sec")

if __name__ == "__main__":
    asyncio.run(main())

