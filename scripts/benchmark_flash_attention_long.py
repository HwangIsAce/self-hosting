#!/usr/bin/env python3
"""긴 시퀀스에서 Flash Attention 성능 벤치마크"""

import sys
import torch
import gc
from pathlib import Path
import time

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

print("=" * 80)
print("긴 시퀀스 Flash Attention 성능 벤치마크")
print("=" * 80)

# Flash Attention 설치 확인
try:
    import flash_attn
    print(f"✅ Flash Attention 설치됨: {flash_attn.__version__}")
except ImportError:
    print("❌ Flash Attention이 설치되지 않았습니다.")
    sys.exit(1)

print(f"\n모델: {MODEL_NAME}")
print(f"디바이스: {DEVICE}")
print(f"PyTorch 버전: {torch.__version__}")
print(f"CUDA 사용 가능: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# Tokenizer 로드
print(f"\nTokenizer 로드 중...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

# 모델 로드 (SDPA - 기본)
print("\n[모델 로드] SDPA (기본)")
torch.cuda.empty_cache()
gc.collect()
model_sdpa = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map=DEVICE,
    trust_remote_code=True,
    attn_implementation=None,  # 기본 SDPA
)
model_sdpa.eval()

# 모델 로드 (Flash Attention)
print("\n[모델 로드] Flash Attention 2")
torch.cuda.empty_cache()
gc.collect()
model_fa = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map=DEVICE,
    trust_remote_code=True,
    attn_implementation="flash_attention_2",
)
model_fa.eval()

# Attention 구현 확인
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

print("\n" + "=" * 80)
print("Attention 구현 확인")
print("=" * 80)

attn_impl_sdpa = getattr(model_sdpa.config, '_attn_implementation', None)
attn_impl_fa = getattr(model_fa.config, '_attn_implementation', None)

print(f"SDPA 모델: config._attn_implementation = {attn_impl_sdpa}")
print(f"Flash Attn 모델: config._attn_implementation = {attn_impl_fa}")

interface_sdpa = ALL_ATTENTION_FUNCTIONS.get_interface(attn_impl_sdpa, None)
interface_fa = ALL_ATTENTION_FUNCTIONS.get_interface(attn_impl_fa, None)

print(f"\nSDPA 모델 attention 함수: {interface_sdpa}")
if interface_sdpa:
    print(f"  모듈: {interface_sdpa.__module__ if hasattr(interface_sdpa, '__module__') else 'N/A'}")
print(f"\nFlash Attn 모델 attention 함수: {interface_fa}")
if interface_fa:
    print(f"  이름: {interface_fa.__name__ if hasattr(interface_fa, '__name__') else 'N/A'}")
    print(f"  모듈: {interface_fa.__module__ if hasattr(interface_fa, '__module__') else 'N/A'}")

# 벤치마크 함수
def benchmark_model(model, inputs, num_runs=5, warmup=2):
    """모델 벤치마크 실행"""
    model.eval()
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(**inputs)
    torch.cuda.synchronize()
    
    # 실제 측정
    times = []
    for _ in range(num_runs):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        with torch.no_grad():
            _ = model(**inputs)
        end.record()
        torch.cuda.synchronize()
        
        times.append(start.elapsed_time(end) / 1000.0)  # ms to seconds
    
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    return avg_time, min_time, max_time

def get_memory_usage():
    """현재 GPU 메모리 사용량 반환"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**3  # GB
    return 0

# 긴 시퀀스 테스트
print("\n" + "=" * 80)
print("긴 시퀀스 성능 벤치마크")
print("=" * 80)

# 테스트할 시퀀스 길이 (토큰 수)
test_lengths = [1024, 2048, 4096, 8192]

results = []

for seq_len in test_lengths:
    print(f"\n{'=' * 80}")
    print(f"시퀀스 길이: {seq_len} 토큰")
    print(f"{'=' * 80}")
    
    # 입력 준비 (긴 텍스트 생성)
    # 실제 토큰 수가 정확히 맞도록 조정
    test_text = "The quick brown fox jumps over the lazy dog. " * (seq_len // 10)
    inputs = tokenizer(
        test_text,
        return_tensors="pt",
        max_length=seq_len,
        truncation=True,
        padding="max_length"
    ).to(DEVICE)
    
    actual_seq_len = inputs['input_ids'].shape[1]
    print(f"실제 입력 길이: {actual_seq_len} 토큰")
    
    # 메모리 초기화
    torch.cuda.empty_cache()
    gc.collect()
    initial_mem = get_memory_usage()
    
    # SDPA 벤치마크
    print(f"\n[SDPA] 벤치마크 실행 중...")
    mem_before_sdpa = get_memory_usage()
    time_sdpa, min_sdpa, max_sdpa = benchmark_model(model_sdpa, inputs, num_runs=5, warmup=2)
    mem_after_sdpa = get_memory_usage()
    mem_sdpa = mem_after_sdpa - mem_before_sdpa
    
    # 메모리 초기화
    torch.cuda.empty_cache()
    gc.collect()
    
    # Flash Attention 벤치마크
    print(f"[Flash Attention] 벤치마크 실행 중...")
    mem_before_fa = get_memory_usage()
    time_fa, min_fa, max_fa = benchmark_model(model_fa, inputs, num_runs=5, warmup=2)
    mem_after_fa = get_memory_usage()
    mem_fa = mem_after_fa - mem_before_fa
    
    # 결과 계산
    speedup = ((time_sdpa - time_fa) / time_sdpa) * 100 if time_sdpa > 0 else 0
    mem_saving = ((mem_sdpa - mem_fa) / mem_sdpa) * 100 if mem_sdpa > 0 else 0
    
    # 결과 출력
    print(f"\n{'─' * 80}")
    print(f"결과:")
    print(f"  SDPA:")
    print(f"    평균 시간: {time_sdpa:.3f}초 (최소: {min_sdpa:.3f}초, 최대: {max_sdpa:.3f}초)")
    print(f"    메모리 사용: {mem_sdpa:.2f} GB")
    print(f"  Flash Attention:")
    print(f"    평균 시간: {time_fa:.3f}초 (최소: {min_fa:.3f}초, 최대: {max_fa:.3f}초)")
    print(f"    메모리 사용: {mem_fa:.2f} GB")
    print(f"  개선:")
    print(f"    속도: {speedup:+.1f}% ({'빠름' if speedup > 0 else '느림'})")
    print(f"    메모리: {mem_saving:+.1f}% ({'절약' if mem_saving > 0 else '더 사용'})")
    
    results.append({
        'seq_len': actual_seq_len,
        'sdpa_time': time_sdpa,
        'fa_time': time_fa,
        'sdpa_mem': mem_sdpa,
        'fa_mem': mem_fa,
        'speedup': speedup,
        'mem_saving': mem_saving,
    })
    
    # 메모리 초기화
    torch.cuda.empty_cache()
    gc.collect()

# 최종 요약
print("\n" + "=" * 80)
print("최종 요약")
print("=" * 80)

print(f"\n{'시퀀스 길이':<12} {'SDPA (초)':<12} {'Flash Attn (초)':<16} {'속도 개선':<12} {'메모리 절약':<12}")
print("-" * 80)

for r in results:
    print(f"{r['seq_len']:<12} {r['sdpa_time']:<12.3f} {r['fa_time']:<16.3f} {r['speedup']:>+10.1f}% {r['mem_saving']:>+10.1f}%")

print("\n" + "=" * 80)
print("결론")
print("=" * 80)

# 가장 긴 시퀀스에서의 결과 분석
if results:
    longest = results[-1]
    print(f"\n가장 긴 시퀀스 ({longest['seq_len']} 토큰)에서:")
    if longest['speedup'] > 0:
        print(f"  ✅ Flash Attention이 {longest['speedup']:.1f}% 더 빠릅니다")
    else:
        print(f"  ⚠️  Flash Attention이 {abs(longest['speedup']):.1f}% 더 느립니다")
    
    if longest['mem_saving'] > 0:
        print(f"  ✅ Flash Attention이 {longest['mem_saving']:.1f}% 메모리를 절약합니다")
    else:
        print(f"  ⚠️  Flash Attention이 {abs(longest['mem_saving']):.1f}% 더 많은 메모리를 사용합니다")

print("\n" + "=" * 80)

