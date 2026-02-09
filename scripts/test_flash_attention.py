#!/usr/bin/env python3
"""Flash Attention 성능 비교 테스트 스크립트"""

import asyncio
import sys
import time
import torch
from pathlib import Path
from typing import Dict, Any, Optional
import gc

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 최소한의 의존성만 사용
try:
    from api.config.settings import settings
    MODEL_NAME = settings.LLM_MODEL_NAME
    GPU_ID = settings.LLM_GPU_ID
    USE_QUANTIZATION = settings.USE_QUANTIZATION
except ImportError:
    # 기본값 사용
    MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
    GPU_ID = 0
    USE_QUANTIZATION = False
    print("⚠️  api.config.settings를 불러올 수 없어 기본값을 사용합니다.")

from transformers import AutoModelForCausalLM, AutoTokenizer


def check_flash_attention():
    """Flash Attention 2 설치 및 사용 가능 여부 확인"""
    try:
        import flash_attn
        print(f"✅ Flash Attention 2 설치됨 (버전: {flash_attn.__version__})")
        return True
    except ImportError:
        print("❌ Flash Attention 2가 설치되지 않았습니다.")
        return False


def load_model_with_attention(
    model_name: str,
    device_map: str,
    attn_implementation: Optional[str] = None,
    use_quantization: bool = False
):
    """특정 attention 구현으로 모델 로드"""
    print(f"\n모델 로드 중... (attn_implementation={attn_implementation})")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.float16,
        "device_map": device_map,
    }
    
    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation
        print(f"  → Attention 구현: {attn_implementation}")
    else:
        print(f"  → Attention 구현: 기본 (sdpa)")
    
    if use_quantization:
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16
        )
        model_kwargs["quantization_config"] = quantization_config
    
    load_start = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **model_kwargs
    )
    load_time = time.time() - load_start
    
    model.eval()
    
    # 실제 사용된 attention 구현 확인
    actual_attn = getattr(model.config, 'attn_implementation', None)
    print(f"  ✅ 모델 로드 완료 ({load_time:.2f}초)")
    print(f"  실제 사용된 attention: {actual_attn or '기본'}")
    
    return model, tokenizer, load_time


async def benchmark_model(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    test_name: str,
    num_runs: int = 5,
    warmup_runs: int = 2
):
    """모델 성능 벤치마크"""
    print(f"\n[{test_name}] 벤치마크 시작...")
    
    test_messages = [
        {"role": "user", "content": "Write a detailed explanation of how neural networks work. Include at least 200 words about backpropagation, activation functions, and gradient descent."}
    ]
    
    # 프롬프트 준비
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        prompt = tokenizer.apply_chat_template(
            test_messages,
            tokenize=False,
            add_generation_prompt=True
        )
    else:
        prompt = test_messages[0]["content"]
    
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True
    ).to(model.device)
    
    # Warmup
    print(f"  Warmup 실행 중 ({warmup_runs}회)...")
    with torch.no_grad():
        for i in range(warmup_runs):
            _ = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                use_cache=True,
            )
            print(f"    Warmup {i+1}/{warmup_runs} 완료")
    
    # 실제 벤치마크
    print(f"  벤치마크 실행 중 ({num_runs}회)...")
    times = []
    token_counts = []
    memory_usage = []
    
    for i in range(num_runs):
        # 메모리 정리
        torch.cuda.empty_cache()
        gc.collect()
        
        # 메모리 측정 (시작)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            start_memory = torch.cuda.memory_allocated()
        
        start_time = time.time()
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                use_cache=True,
            )
        
        elapsed = time.time() - start_time
        
        # 메모리 측정 (종료)
        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated()
            memory_used = (peak_memory - start_memory) / 1024**3  # GB
            memory_usage.append(memory_used)
        
        # 토큰 수 계산
        generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        num_tokens = len(generated_tokens)
        token_counts.append(num_tokens)
        
        times.append(elapsed)
        print(f"    실행 {i+1}/{num_runs}: {elapsed:.2f}초, {num_tokens} 토큰")
    
    avg_time = sum(times) / len(times)
    avg_tokens = sum(token_counts) / len(token_counts)
    tokens_per_sec = avg_tokens / avg_time
    avg_memory = sum(memory_usage) / len(memory_usage) if memory_usage else 0
    
    return {
        "avg_time": avg_time,
        "tokens_per_sec": tokens_per_sec,
        "avg_tokens": avg_tokens,
        "avg_memory_gb": avg_memory,
        "times": times,
        "token_counts": token_counts,
    }


async def test_flash_attention_comparison():
    """Flash Attention 적용 전후 성능 비교"""
    print("=" * 70)
    print("Flash Attention 성능 비교 테스트")
    print("=" * 70)
    
    # 환경 확인
    print("\n환경 확인:")
    print(f"  PyTorch 버전: {torch.__version__}")
    print(f"  CUDA 사용 가능: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA 버전: {torch.version.cuda}")
        print(f"  GPU 개수: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"    GPU {i}: {torch.cuda.get_device_name(i)}")
    
    flash_attn_available = check_flash_attention()
    
    if not flash_attn_available:
        print("\n⚠️  Flash Attention이 설치되지 않아 비교 테스트를 수행할 수 없습니다.")
        return
    
    model_name = MODEL_NAME
    device_map = f"cuda:{GPU_ID}"
    use_quantization = USE_QUANTIZATION
    
    print(f"\n테스트 모델: {model_name}")
    print(f"디바이스: {device_map}")
    print(f"양자화: {use_quantization}")
    
    results = {}
    
    # 1. 기본 Attention (SDPA) 테스트
    print("\n" + "=" * 70)
    print("테스트 1: 기본 Attention (SDPA)")
    print("=" * 70)
    
    try:
        model_sdpa, tokenizer_sdpa, load_time_sdpa = load_model_with_attention(
            model_name=model_name,
            device_map=device_map,
            attn_implementation="sdpa",  # PyTorch의 기본 SDPA
            use_quantization=use_quantization
        )
        
        result_sdpa = await benchmark_model(
            model_sdpa,
            tokenizer_sdpa,
            "기본 Attention (SDPA)",
            num_runs=5,
            warmup_runs=2
        )
        result_sdpa["load_time"] = load_time_sdpa
        results["sdpa"] = result_sdpa
        
        # 메모리 정리
        del model_sdpa, tokenizer_sdpa
        torch.cuda.empty_cache()
        gc.collect()
        
        print("\n⏳ 모델 언로드를 위해 10초 대기 중...")
        await asyncio.sleep(10)
        
    except Exception as e:
        print(f"❌ SDPA 테스트 실패: {str(e)}")
        import traceback
        traceback.print_exc()
        results["sdpa"] = None
    
    # 2. Flash Attention 2 테스트
    print("\n" + "=" * 70)
    print("테스트 2: Flash Attention 2")
    print("=" * 70)
    
    try:
        model_fa, tokenizer_fa, load_time_fa = load_model_with_attention(
            model_name=model_name,
            device_map=device_map,
            attn_implementation="flash_attention_2",
            use_quantization=use_quantization
        )
        
        result_fa = await benchmark_model(
            model_fa,
            tokenizer_fa,
            "Flash Attention 2",
            num_runs=5,
            warmup_runs=2
        )
        result_fa["load_time"] = load_time_fa
        results["flash_attention_2"] = result_fa
        
        # 메모리 정리
        del model_fa, tokenizer_fa
        torch.cuda.empty_cache()
        gc.collect()
        
    except Exception as e:
        print(f"❌ Flash Attention 2 테스트 실패: {str(e)}")
        import traceback
        traceback.print_exc()
        results["flash_attention_2"] = None
    
    # 결과 비교
    print("\n" + "=" * 70)
    print("성능 비교 결과")
    print("=" * 70)
    
    if results.get("sdpa") and results.get("flash_attention_2"):
        sdpa = results["sdpa"]
        fa = results["flash_attention_2"]
        
        # 시간 개선율
        time_improvement = ((sdpa["avg_time"] - fa["avg_time"]) / sdpa["avg_time"]) * 100
        
        # 속도 개선율
        speed_improvement = ((fa["tokens_per_sec"] - sdpa["tokens_per_sec"]) / sdpa["tokens_per_sec"]) * 100
        
        # 메모리 절약율
        memory_saving = 0
        if sdpa["avg_memory_gb"] > 0 and fa["avg_memory_gb"] > 0:
            memory_saving = ((sdpa["avg_memory_gb"] - fa["avg_memory_gb"]) / sdpa["avg_memory_gb"]) * 100
        
        print(f"\n[모델 로드 시간]")
        print(f"  SDPA:           {sdpa['load_time']:.2f}초")
        print(f"  Flash Attention: {fa['load_time']:.2f}초")
        
        print(f"\n[추론 성능]")
        print(f"  SDPA:")
        print(f"    평균 응답 시간: {sdpa['avg_time']:.2f}초")
        print(f"    토큰 생성 속도: {sdpa['tokens_per_sec']:.2f} tokens/sec")
        print(f"    평균 생성 토큰: {sdpa['avg_tokens']:.0f} 토큰")
        if sdpa['avg_memory_gb'] > 0:
            print(f"    평균 메모리 사용: {sdpa['avg_memory_gb']:.2f} GB")
        
        print(f"\n  Flash Attention 2:")
        print(f"    평균 응답 시간: {fa['avg_time']:.2f}초")
        print(f"    토큰 생성 속도: {fa['tokens_per_sec']:.2f} tokens/sec")
        print(f"    평균 생성 토큰: {fa['avg_tokens']:.0f} 토큰")
        if fa['avg_memory_gb'] > 0:
            print(f"    평균 메모리 사용: {fa['avg_memory_gb']:.2f} GB")
        
        print(f"\n[개선 효과]")
        if time_improvement > 0:
            print(f"  ✅ 응답 시간: {time_improvement:+.1f}% 개선 ({sdpa['avg_time']:.2f}초 → {fa['avg_time']:.2f}초)")
        else:
            print(f"  ⚠️  응답 시간: {time_improvement:+.1f}% ({sdpa['avg_time']:.2f}초 → {fa['avg_time']:.2f}초)")
        
        if speed_improvement > 0:
            print(f"  ✅ 생성 속도: {speed_improvement:+.1f}% 개선 ({sdpa['tokens_per_sec']:.2f} → {fa['tokens_per_sec']:.2f} tokens/sec)")
        else:
            print(f"  ⚠️  생성 속도: {speed_improvement:+.1f}% ({sdpa['tokens_per_sec']:.2f} → {fa['tokens_per_sec']:.2f} tokens/sec)")
        
        if memory_saving > 0:
            print(f"  ✅ 메모리 사용: {memory_saving:+.1f}% 절약 ({sdpa['avg_memory_gb']:.2f}GB → {fa['avg_memory_gb']:.2f}GB)")
        elif memory_saving < 0:
            print(f"  ⚠️  메모리 사용: {memory_saving:+.1f}% 증가 ({sdpa['avg_memory_gb']:.2f}GB → {fa['avg_memory_gb']:.2f}GB)")
        
        print(f"\n" + "=" * 70)
        if time_improvement > 5 or speed_improvement > 5:
            print("✅ Flash Attention 2가 성능을 크게 개선했습니다!")
        elif time_improvement > 0 or speed_improvement > 0:
            print("✅ Flash Attention 2가 성능을 개선했습니다!")
        else:
            print("⚠️  이 테스트에서는 Flash Attention 2의 성능 개선 효과가 제한적입니다.")
            print("   (긴 시퀀스 길이에서 더 큰 효과를 볼 수 있습니다)")
        print("=" * 70)
    else:
        print("\n❌ 비교 테스트를 완료할 수 없습니다.")
        if not results.get("sdpa"):
            print("  - SDPA 테스트 실패")
        if not results.get("flash_attention_2"):
            print("  - Flash Attention 2 테스트 실패")


async def main():
    """메인 함수"""
    try:
        await test_flash_attention_comparison()
    except KeyboardInterrupt:
        print("\n\n테스트가 중단되었습니다.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n치명적 오류: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

