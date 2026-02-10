#!/usr/bin/env python3
"""vLLM 성능 테스트 및 HuggingFace와 비교"""

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

setup_logging()

async def test_single_request(engine, engine_name: str, num_runs: int = 5):
    """단일 요청 성능 테스트"""
    print(f"\n{'='*60}")
    print(f"{engine_name} 단일 요청 성능 테스트 ({num_runs}회)")
    print(f"{'='*60}")
    
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
    
    return {
        "avg_time": avg_time,
        "avg_tokens": avg_tokens,
        "tokens_per_sec": avg_tps,
        "times": times
    }

async def test_batch_requests(engine, engine_name: str, batch_size: int = 10):
    """배치 요청 성능 테스트"""
    print(f"\n{'='*60}")
    print(f"{engine_name} 배치 요청 성능 테스트 ({batch_size}개 동시 요청)")
    print(f"{'='*60}")
    
    # 여러 요청 준비
    messages_list = [
        [{"role": "user", "content": f"Write a one-sentence summary about topic {i}."}]
        for i in range(batch_size)
    ]
    
    # vLLM 엔진인 경우 generate_batch 사용
    if hasattr(engine, 'generate_batch'):
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
        
        return {
            "total_time": elapsed,
            "num_requests": len(results),
            "total_tokens": total_tokens,
            "avg_time_per_request": avg_time_per_request,
            "tokens_per_sec": tps
        }
    else:
        # HuggingFace 엔진인 경우 순차 처리
        print(f"HuggingFace 순차 처리...")
        start = time.time()
        results = []
        for i, messages in enumerate(messages_list):
            result = await engine.generate(messages, max_tokens=100)
            results.append(result)
            print(f"  요청 {i+1}/{batch_size} 완료")
        elapsed = time.time() - start
        
        total_tokens = sum(r['usage']['total_tokens'] for r in results)
        avg_time_per_request = elapsed / len(results)
        tps = total_tokens / elapsed
        
        print(f"\n✅ 완료: {elapsed:.2f}초")
        print(f"✅ 처리된 요청: {len(results)}개")
        print(f"✅ 총 토큰: {total_tokens} 토큰")
        print(f"✅ 평균 시간: {avg_time_per_request:.2f}초/요청")
        print(f"✅ 처리 속도: {tps:.2f} tokens/sec")
        
        return {
            "total_time": elapsed,
            "num_requests": len(results),
            "total_tokens": total_tokens,
            "avg_time_per_request": avg_time_per_request,
            "tokens_per_sec": tps
        }

async def test_chunk_enrich(engine, engine_name: str, num_chunks: int = 50):
    """Chunk enrich 성능 테스트"""
    print(f"\n{'='*60}")
    print(f"{engine_name} Chunk Enrich 성능 테스트 ({num_chunks}개 chunk)")
    print(f"{'='*60}")
    
    # Chunk 시뮬레이션
    chunks = [f"This is chunk {i} with some content about topic {i}." for i in range(num_chunks)]
    
    messages_list = [
        [{"role": "user", "content": f"Enrich this chunk with additional context: {chunk}"}]
        for chunk in chunks
    ]
    
    # vLLM 엔진인 경우 generate_batch 사용
    if hasattr(engine, 'generate_batch'):
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
        
        return {
            "total_time": elapsed,
            "num_chunks": len(results),
            "total_tokens": total_tokens,
            "avg_time_per_chunk": avg_time_per_chunk,
            "tokens_per_sec": tps
        }
    else:
        # HuggingFace 엔진인 경우 순차 처리
        print(f"HuggingFace 순차 처리...")
        start = time.time()
        results = []
        for i, messages in enumerate(messages_list):
            result = await engine.generate(messages, max_tokens=200)
            results.append(result)
            if (i + 1) % 10 == 0:
                print(f"  Chunk {i+1}/{num_chunks} 완료")
        elapsed = time.time() - start
        
        total_tokens = sum(r['usage']['total_tokens'] for r in results)
        avg_time_per_chunk = elapsed / len(results)
        tps = total_tokens / elapsed
        
        print(f"\n✅ 완료: {elapsed:.2f}초")
        print(f"✅ 처리된 chunk: {len(results)}개")
        print(f"✅ 총 토큰: {total_tokens} 토큰")
        print(f"✅ 평균 시간: {avg_time_per_chunk:.2f}초/chunk")
        print(f"✅ 처리 속도: {tps:.2f} tokens/sec")
        
        return {
            "total_time": elapsed,
            "num_chunks": len(results),
            "total_tokens": total_tokens,
            "avg_time_per_chunk": avg_time_per_chunk,
            "tokens_per_sec": tps
        }

async def main():
    print("="*60)
    print("vLLM vs HuggingFace 성능 비교 테스트")
    print("="*60)
    
    device_map = f"cuda:{settings.LLM_GPU_ID}"
    
    # 1. HuggingFace 엔진 테스트
    print("\n" + "="*60)
    print("1단계: HuggingFace 엔진 테스트")
    print("="*60)
    
    hf_engine = LLMEngine(
        model_name=settings.LLM_MODEL_NAME,
        device_map=device_map,
        use_quantization=settings.USE_QUANTIZATION
    )
    hf_engine.load_model()
    
    # 단일 요청 테스트
    hf_single = await test_single_request(hf_engine, "HuggingFace", num_runs=5)
    
    # 배치 요청 테스트 (10개)
    hf_batch = await test_batch_requests(hf_engine, "HuggingFace", batch_size=10)
    
    # Chunk enrich 테스트 (50개)
    hf_chunk = await test_chunk_enrich(hf_engine, "HuggingFace", num_chunks=50)
    
    print("\n모델 언로드를 위해 30초 대기 중...")
    await asyncio.sleep(30)
    
    # 2. vLLM 엔진 테스트
    print("\n" + "="*60)
    print("2단계: vLLM 엔진 테스트")
    print("="*60)
    
    vllm_engine = VLLMEngine(
        model_name=settings.LLM_MODEL_NAME,
        device_map=device_map,
        use_quantization=settings.USE_QUANTIZATION
    )
    vllm_engine.load_model()
    
    # 단일 요청 테스트
    vllm_single = await test_single_request(vllm_engine, "vLLM", num_runs=5)
    
    # 배치 요청 테스트 (10개)
    vllm_batch = await test_batch_requests(vllm_engine, "vLLM", batch_size=10)
    
    # Chunk enrich 테스트 (50개)
    vllm_chunk = await test_chunk_enrich(vllm_engine, "vLLM", num_chunks=50)
    
    # 3. 결과 비교
    print("\n" + "="*60)
    print("성능 비교 결과")
    print("="*60)
    
    # 단일 요청 비교
    print("\n[단일 요청 성능]")
    print(f"  HuggingFace: {hf_single['avg_time']:.2f}초, {hf_single['tokens_per_sec']:.2f} tokens/sec")
    print(f"  vLLM:        {vllm_single['avg_time']:.2f}초, {vllm_single['tokens_per_sec']:.2f} tokens/sec")
    time_improvement = ((hf_single['avg_time'] - vllm_single['avg_time']) / hf_single['avg_time']) * 100
    tps_improvement = ((vllm_single['tokens_per_sec'] - hf_single['tokens_per_sec']) / hf_single['tokens_per_sec']) * 100
    print(f"  시간 개선:   {time_improvement:+.1f}%")
    print(f"  속도 개선:   {tps_improvement:+.1f}%")
    
    # 배치 요청 비교
    print("\n[배치 요청 성능 (10개 동시)]")
    print(f"  HuggingFace: {hf_batch['total_time']:.2f}초, {hf_batch['avg_time_per_request']:.2f}초/요청, {hf_batch['tokens_per_sec']:.2f} tokens/sec")
    print(f"  vLLM:        {vllm_batch['total_time']:.2f}초, {vllm_batch['avg_time_per_request']:.2f}초/요청, {vllm_batch['tokens_per_sec']:.2f} tokens/sec")
    batch_time_improvement = ((hf_batch['total_time'] - vllm_batch['total_time']) / hf_batch['total_time']) * 100
    batch_tps_improvement = ((vllm_batch['tokens_per_sec'] - hf_batch['tokens_per_sec']) / hf_batch['tokens_per_sec']) * 100
    speedup = hf_batch['total_time'] / vllm_batch['total_time']
    print(f"  시간 개선:   {batch_time_improvement:+.1f}%")
    print(f"  속도 개선:   {batch_tps_improvement:+.1f}%")
    print(f"  배치 속도 향상: {speedup:.2f}x")
    
    # Chunk enrich 비교
    print("\n[Chunk Enrich 성능 (50개 chunk)]")
    print(f"  HuggingFace: {hf_chunk['total_time']:.2f}초, {hf_chunk['avg_time_per_chunk']:.2f}초/chunk, {hf_chunk['tokens_per_sec']:.2f} tokens/sec")
    print(f"  vLLM:        {vllm_chunk['total_time']:.2f}초, {vllm_chunk['avg_time_per_chunk']:.2f}초/chunk, {vllm_chunk['tokens_per_sec']:.2f} tokens/sec")
    chunk_time_improvement = ((hf_chunk['total_time'] - vllm_chunk['total_time']) / hf_chunk['total_time']) * 100
    chunk_tps_improvement = ((vllm_chunk['tokens_per_sec'] - hf_chunk['tokens_per_sec']) / hf_chunk['tokens_per_sec']) * 100
    chunk_speedup = hf_chunk['total_time'] / vllm_chunk['total_time']
    print(f"  시간 개선:   {chunk_time_improvement:+.1f}%")
    print(f"  속도 개선:   {chunk_tps_improvement:+.1f}%")
    print(f"  Chunk enrich 속도 향상: {chunk_speedup:.2f}x")
    
    print("\n" + "="*60)
    print("테스트 완료!")
    print("="*60)

if __name__ == "__main__":
    asyncio.run(main())

