#!/usr/bin/env python3
"""HuggingFace vs vLLM 성능 비교 벤치마크"""

import asyncio
import sys
import time
import gc
import torch
from pathlib import Path
from typing import Dict, Any

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from api.config.logging_config import setup_logging
from api.config.settings import settings
from api.infrastructure.models.llm_engine import LLMEngine
from api.infrastructure.models.vllm_engine import VLLMEngine

setup_logging()

# 테스트 설정
TEST_CONFIG = {
    "single_runs": 5,
    "batch_size": 10,
    "chunk_count": 50,
    "warmup_runs": 2,
}

def format_speedup(hf_time: float, vllm_time: float) -> str:
    """속도 향상 배수 포맷팅"""
    if hf_time > 0 and vllm_time > 0:
        speedup = hf_time / vllm_time
        return f"{speedup:.2f}x"
    return "N/A"

def print_comparison(title: str, hf_result: Dict[str, Any], vllm_result: Dict[str, Any]):
    """비교 결과 출력"""
    print(f"\n{'='*70}")
    print(f"{title}")
    print(f"{'='*70}")
    print(f"{'Metric':<30} {'HuggingFace':<20} {'vLLM':<20} {'Speedup':<10}")
    print(f"{'-'*70}")
    
    # 시간 비교
    hf_time = hf_result.get('avg_time') or hf_result.get('total_time', 0)
    vllm_time = vllm_result.get('avg_time') or vllm_result.get('total_time', 0)
    print(f"{'Time (seconds)':<30} {hf_time:<20.2f} {vllm_time:<20.2f} {format_speedup(hf_time, vllm_time):<10}")
    
    # TPS 비교
    hf_tps = hf_result.get('tokens_per_sec', 0)
    vllm_tps = vllm_result.get('tokens_per_sec', 0)
    print(f"{'Tokens/sec':<30} {hf_tps:<20.2f} {vllm_tps:<20.2f} {format_speedup(vllm_tps, hf_tps):<10}")
    
    # 추가 메트릭
    if 'avg_time_per_request' in hf_result:
        hf_avg = hf_result.get('avg_time_per_request', 0)
        vllm_avg = vllm_result.get('avg_time_per_request', 0)
        print(f"{'Avg time/request (sec)':<30} {hf_avg:<20.2f} {vllm_avg:<20.2f} {format_speedup(hf_avg, vllm_avg):<10}")
    
    if 'avg_time_per_chunk' in hf_result:
        hf_avg = hf_result.get('avg_time_per_chunk', 0)
        vllm_avg = vllm_result.get('avg_time_per_chunk', 0)
        print(f"{'Avg time/chunk (sec)':<30} {hf_avg:<20.2f} {vllm_avg:<20.2f} {format_speedup(hf_avg, vllm_avg):<10}")

async def test_hf_single(num_runs: int = 5) -> Dict[str, Any]:
    """HuggingFace 단일 요청 테스트"""
    print(f"\n{'='*70}")
    print(f"HuggingFace 단일 요청 테스트 ({num_runs}회)")
    print(f"{'='*70}")
    
    engine = LLMEngine(
        model_name=settings.LLM_MODEL_NAME,
        device_map=f"cuda:{settings.LLM_GPU_ID}",
        use_quantization=settings.USE_QUANTIZATION
    )
    
    print("모델 로드 중...")
    load_start = time.time()
    engine.load_model()
    load_time = time.time() - load_start
    print(f"✅ 모델 로드 완료: {load_time:.2f}초")
    
    test_messages = [
        {"role": "user", "content": "Write a short story about AI in exactly 100 words."}
    ]
    
    # Warmup
    print("Warmup 실행 중...")
    for _ in range(TEST_CONFIG["warmup_runs"]):
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
        tps = tokens / elapsed if elapsed > 0 else 0
        print(f"  실행 {i+1}/{num_runs}: {elapsed:.2f}초, {tokens} 토큰 ({tps:.2f} tokens/sec)")
    
    avg_time = sum(times) / len(times)
    avg_tokens = sum(token_counts) / len(token_counts)
    avg_tps = avg_tokens / avg_time if avg_time > 0 else 0
    
    print(f"\n✅ 평균: {avg_time:.2f}초, {avg_tps:.2f} tokens/sec")
    
    # 메모리 정리
    if hasattr(engine, 'model'):
        del engine.model
    if hasattr(engine, 'tokenizer'):
        del engine.tokenizer
    if hasattr(engine, 'model_loader'):
        # ModelLoader의 캐시도 정리
        if hasattr(engine.model_loader, 'loaded_models'):
            engine.model_loader.loaded_models.clear()
        if hasattr(engine.model_loader, 'loaded_tokenizers'):
            engine.model_loader.loaded_tokenizers.clear()
        del engine.model_loader
    del engine
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    
    return {"avg_time": avg_time, "tokens_per_sec": avg_tps, "load_time": load_time}

async def test_vllm_single(num_runs: int = 5) -> Dict[str, Any]:
    """vLLM 단일 요청 테스트"""
    print(f"\n{'='*70}")
    print(f"vLLM 단일 요청 테스트 ({num_runs}회)")
    print(f"{'='*70}")
    
    engine = VLLMEngine(
        model_name=settings.LLM_MODEL_NAME,
        device_map=f"cuda:{settings.LLM_GPU_ID}",
        use_quantization=settings.USE_QUANTIZATION
    )
    
    print("모델 로드 중...")
    load_start = time.time()
    engine.load_model()
    load_time = time.time() - load_start
    print(f"✅ 모델 로드 완료: {load_time:.2f}초")
    
    test_messages = [
        {"role": "user", "content": "Write a short story about AI in exactly 100 words."}
    ]
    
    # Warmup
    print("Warmup 실행 중...")
    for _ in range(TEST_CONFIG["warmup_runs"]):
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
        tps = tokens / elapsed if elapsed > 0 else 0
        print(f"  실행 {i+1}/{num_runs}: {elapsed:.2f}초, {tokens} 토큰 ({tps:.2f} tokens/sec)")
    
    avg_time = sum(times) / len(times)
    avg_tokens = sum(token_counts) / len(token_counts)
    avg_tps = avg_tokens / avg_time if avg_time > 0 else 0
    
    print(f"\n✅ 평균: {avg_time:.2f}초, {avg_tps:.2f} tokens/sec")
    
    # 메모리 정리
    if hasattr(engine, 'model'):
        del engine.model
    if hasattr(engine, 'tokenizer'):
        del engine.tokenizer
    if hasattr(engine, 'model_loader'):
        # ModelLoader의 캐시도 정리
        if hasattr(engine.model_loader, 'loaded_models'):
            engine.model_loader.loaded_models.clear()
        if hasattr(engine.model_loader, 'loaded_tokenizers'):
            engine.model_loader.loaded_tokenizers.clear()
        del engine.model_loader
    del engine
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    
    return {"avg_time": avg_time, "tokens_per_sec": avg_tps, "load_time": load_time}

async def test_hf_batch(batch_size: int = 10) -> Dict[str, Any]:
    """HuggingFace 배치 요청 테스트"""
    print(f"\n{'='*70}")
    print(f"HuggingFace 배치 요청 테스트 ({batch_size}개 순차 처리)")
    print(f"{'='*70}")
    
    engine = LLMEngine(
        model_name=settings.LLM_MODEL_NAME,
        device_map=f"cuda:{settings.LLM_GPU_ID}",
        use_quantization=settings.USE_QUANTIZATION
    )
    engine.load_model()
    
    messages_list = [
        [{"role": "user", "content": f"Write a one-sentence summary about topic {i}."}]
        for i in range(batch_size)
    ]
    
    print(f"HuggingFace 순차 처리...")
    start = time.time()
    results = []
    for messages in messages_list:
        result = await engine.generate(messages, max_tokens=100)
        results.append(result)
    elapsed = time.time() - start
    
    total_tokens = sum(r['usage']['total_tokens'] for r in results)
    avg_time_per_request = elapsed / len(results)
    tps = total_tokens / elapsed if elapsed > 0 else 0
    
    print(f"\n✅ 완료: {elapsed:.2f}초")
    print(f"✅ 처리된 요청: {len(results)}개")
    print(f"✅ 총 토큰: {total_tokens} 토큰")
    print(f"✅ 평균 시간: {avg_time_per_request:.2f}초/요청")
    print(f"✅ 처리 속도: {tps:.2f} tokens/sec")
    
    # 메모리 정리
    del engine
    gc.collect()
    torch.cuda.empty_cache()
    
    return {"total_time": elapsed, "tokens_per_sec": tps, "avg_time_per_request": avg_time_per_request}

async def test_vllm_batch(batch_size: int = 10) -> Dict[str, Any]:
    """vLLM 배치 요청 테스트"""
    print(f"\n{'='*70}")
    print(f"vLLM 배치 요청 테스트 ({batch_size}개 동시 처리)")
    print(f"{'='*70}")
    
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
    
    print(f"vLLM generate_batch 사용 (Continuous Batching)...")
    start = time.time()
    results = await engine.generate_batch(messages_list, max_tokens=100)
    elapsed = time.time() - start
    
    total_tokens = sum(r['usage']['total_tokens'] for r in results)
    avg_time_per_request = elapsed / len(results)
    tps = total_tokens / elapsed if elapsed > 0 else 0
    
    print(f"\n✅ 완료: {elapsed:.2f}초")
    print(f"✅ 처리된 요청: {len(results)}개")
    print(f"✅ 총 토큰: {total_tokens} 토큰")
    print(f"✅ 평균 시간: {avg_time_per_request:.2f}초/요청")
    print(f"✅ 처리 속도: {tps:.2f} tokens/sec")
    
    # 메모리 정리
    del engine
    gc.collect()
    torch.cuda.empty_cache()
    
    return {"total_time": elapsed, "tokens_per_sec": tps, "avg_time_per_request": avg_time_per_request}

async def test_hf_chunk_enrich(num_chunks: int = 50) -> Dict[str, Any]:
    """HuggingFace Chunk enrich 테스트"""
    print(f"\n{'='*70}")
    print(f"HuggingFace Chunk Enrich 테스트 ({num_chunks}개 chunk)")
    print(f"{'='*70}")
    
    engine = LLMEngine(
        model_name=settings.LLM_MODEL_NAME,
        device_map=f"cuda:{settings.LLM_GPU_ID}",
        use_quantization=settings.USE_QUANTIZATION
    )
    engine.load_model()
    
    chunks = [f"This is chunk {i} with some content about topic {i}." for i in range(num_chunks)]
    
    print(f"HuggingFace 순차 처리...")
    start = time.time()
    results = []
    for chunk in chunks:
        messages = [{"role": "user", "content": f"Enrich this chunk with additional context: {chunk}"}]
        result = await engine.generate(messages, max_tokens=200)
        results.append(result)
    elapsed = time.time() - start
    
    total_tokens = sum(r['usage']['total_tokens'] for r in results)
    avg_time_per_chunk = elapsed / len(results)
    tps = total_tokens / elapsed if elapsed > 0 else 0
    
    print(f"\n✅ 완료: {elapsed:.2f}초")
    print(f"✅ 처리된 chunk: {len(results)}개")
    print(f"✅ 총 토큰: {total_tokens} 토큰")
    print(f"✅ 평균 시간: {avg_time_per_chunk:.2f}초/chunk")
    print(f"✅ 처리 속도: {tps:.2f} tokens/sec")
    
    # 메모리 정리
    del engine
    gc.collect()
    torch.cuda.empty_cache()
    
    return {"total_time": elapsed, "tokens_per_sec": tps, "avg_time_per_chunk": avg_time_per_chunk}

async def test_vllm_chunk_enrich(num_chunks: int = 50) -> Dict[str, Any]:
    """vLLM Chunk enrich 테스트"""
    print(f"\n{'='*70}")
    print(f"vLLM Chunk Enrich 테스트 ({num_chunks}개 chunk)")
    print(f"{'='*70}")
    
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
    
    print(f"vLLM generate_batch 사용 (Continuous Batching)...")
    start = time.time()
    results = await engine.generate_batch(messages_list, max_tokens=200)
    elapsed = time.time() - start
    
    total_tokens = sum(r['usage']['total_tokens'] for r in results)
    avg_time_per_chunk = elapsed / len(results)
    tps = total_tokens / elapsed if elapsed > 0 else 0
    
    print(f"\n✅ 완료: {elapsed:.2f}초")
    print(f"✅ 처리된 chunk: {len(results)}개")
    print(f"✅ 총 토큰: {total_tokens} 토큰")
    print(f"✅ 평균 시간: {avg_time_per_chunk:.2f}초/chunk")
    print(f"✅ 처리 속도: {tps:.2f} tokens/sec")
    
    # 메모리 정리
    del engine
    gc.collect()
    torch.cuda.empty_cache()
    
    return {"total_time": elapsed, "tokens_per_sec": tps, "avg_time_per_chunk": avg_time_per_chunk}

async def main():
    print("="*70)
    print("HuggingFace vs vLLM 성능 비교 벤치마크")
    print("="*70)
    print(f"모델: {settings.LLM_MODEL_NAME}")
    print(f"GPU: {settings.LLM_GPU_ID}")
    print(f"테스트 설정: {TEST_CONFIG}")
    
    results = {}
    
    # 1. 단일 요청 테스트
    print("\n" + "="*70)
    print("1. 단일 요청 성능 비교")
    print("="*70)
    
    hf_single = await test_hf_single(TEST_CONFIG["single_runs"])
    print("\n메모리 정리 중... (60초 대기)")
    # 강력한 메모리 정리
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    await asyncio.sleep(60)  # 메모리 정리 대기
    
    # vLLM 테스트 전에 GPU 메모리 사용률을 낮춤
    import os
    original_util = settings.VLLM_GPU_MEMORY_UTILIZATION
    # 메모리가 부족하면 사용률을 낮춤
    settings.VLLM_GPU_MEMORY_UTILIZATION = 0.6
    
    vllm_single = await test_vllm_single(TEST_CONFIG["single_runs"])
    
    # 원래 설정 복원
    settings.VLLM_GPU_MEMORY_UTILIZATION = original_util
    print_comparison("단일 요청 성능 비교", hf_single, vllm_single)
    results["single"] = {"hf": hf_single, "vllm": vllm_single}
    
    # 2. 배치 요청 테스트
    print("\n" + "="*70)
    print("2. 배치 요청 성능 비교")
    print("="*70)
    
    print("\n메모리 정리 중... (60초 대기)")
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    await asyncio.sleep(60)
    
    hf_batch = await test_hf_batch(TEST_CONFIG["batch_size"])
    print("\n메모리 정리 중... (60초 대기)")
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    await asyncio.sleep(60)
    
    # vLLM 테스트 전에 GPU 메모리 사용률을 낮춤
    original_util = settings.VLLM_GPU_MEMORY_UTILIZATION
    settings.VLLM_GPU_MEMORY_UTILIZATION = 0.6
    
    vllm_batch = await test_vllm_batch(TEST_CONFIG["batch_size"])
    
    # 원래 설정 복원
    settings.VLLM_GPU_MEMORY_UTILIZATION = original_util
    print_comparison("배치 요청 성능 비교", hf_batch, vllm_batch)
    results["batch"] = {"hf": hf_batch, "vllm": vllm_batch}
    
    # 3. Chunk enrich 테스트
    print("\n" + "="*70)
    print("3. Chunk Enrich 성능 비교")
    print("="*70)
    
    print("\n메모리 정리 중... (60초 대기)")
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    await asyncio.sleep(60)
    
    hf_chunk = await test_hf_chunk_enrich(TEST_CONFIG["chunk_count"])
    print("\n메모리 정리 중... (60초 대기)")
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    await asyncio.sleep(60)
    
    # vLLM 테스트 전에 GPU 메모리 사용률을 낮춤
    original_util = settings.VLLM_GPU_MEMORY_UTILIZATION
    settings.VLLM_GPU_MEMORY_UTILIZATION = 0.6
    
    vllm_chunk = await test_vllm_chunk_enrich(TEST_CONFIG["chunk_count"])
    
    # 원래 설정 복원
    settings.VLLM_GPU_MEMORY_UTILIZATION = original_util
    print_comparison("Chunk Enrich 성능 비교", hf_chunk, vllm_chunk)
    results["chunk"] = {"hf": hf_chunk, "vllm": vllm_chunk}
    
    # 최종 요약
    print("\n" + "="*70)
    print("최종 요약")
    print("="*70)
    
    print("\n📊 성능 향상 요약:")
    print(f"  단일 요청: {format_speedup(hf_single['avg_time'], vllm_single['avg_time'])} 빠름")
    print(f"  배치 요청: {format_speedup(hf_batch['total_time'], vllm_batch['total_time'])} 빠름")
    print(f"  Chunk Enrich: {format_speedup(hf_chunk['total_time'], vllm_chunk['total_time'])} 빠름")
    
    print("\n🚀 TPS 향상 요약:")
    print(f"  단일 요청: {format_speedup(vllm_single['tokens_per_sec'], hf_single['tokens_per_sec'])} 향상")
    print(f"  배치 요청: {format_speedup(vllm_batch['tokens_per_sec'], hf_batch['tokens_per_sec'])} 향상")
    print(f"  Chunk Enrich: {format_speedup(vllm_chunk['tokens_per_sec'], hf_chunk['tokens_per_sec'])} 향상")
    
    print("\n" + "="*70)
    print("벤치마크 완료!")
    print("="*70)

if __name__ == "__main__":
    asyncio.run(main())

