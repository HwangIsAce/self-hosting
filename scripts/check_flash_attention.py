#!/usr/bin/env python3
"""Flash Attention 적용 여부 확인 스크립트"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 기본값 설정
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

print("=" * 70)
print("Flash Attention 적용 여부 확인")
print("=" * 70)

# Flash Attention 설치 확인
print("\n1. Flash Attention 설치 확인:")
try:
    import flash_attn
    print(f"   ✅ Flash Attention 설치됨: {flash_attn.__version__}")
    flash_available = True
except ImportError:
    print("   ❌ Flash Attention이 설치되지 않았습니다.")
    flash_available = False

if not flash_available:
    print("\n⚠️  Flash Attention이 설치되지 않아 테스트를 중단합니다.")
    sys.exit(1)

# 모델 로드 (Flash Attention 없이)
print("\n2. 모델 로드 테스트:")
print(f"   모델: {MODEL_NAME}")
print(f"   디바이스: {DEVICE}")

print("\n   [테스트 1] Flash Attention 없이 로드 (attn_implementation=None):")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    
    model_no_fa = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map=DEVICE,
        trust_remote_code=True,
        # attn_implementation을 명시하지 않음
    )
    
    attn_impl_no_fa = getattr(model_no_fa.config, 'attn_implementation', None)
    print(f"   → attn_implementation: {attn_impl_no_fa}")
    
    # 실제 attention 레이어 확인
    if hasattr(model_no_fa, 'model') and hasattr(model_no_fa.model, 'layers'):
        first_layer = model_no_fa.model.layers[0]
        if hasattr(first_layer, 'self_attn'):
            attn_class = type(first_layer.self_attn).__name__
            print(f"   → Attention 클래스: {attn_class}")
    
    del model_no_fa
    torch.cuda.empty_cache()
    
except Exception as e:
    print(f"   ❌ 오류: {e}")
    import traceback
    traceback.print_exc()

print("\n   [테스트 2] Flash Attention 2로 로드 (attn_implementation='flash_attention_2'):")
try:
    model_with_fa = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map=DEVICE,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",  # 명시적으로 설정
    )
    
    attn_impl_with_fa = getattr(model_with_fa.config, 'attn_implementation', None)
    print(f"   → attn_implementation: {attn_impl_with_fa}")
    
    # 실제 attention 레이어 확인
    if hasattr(model_with_fa, 'model') and hasattr(model_with_fa.model, 'layers'):
        first_layer = model_with_fa.model.layers[0]
        if hasattr(first_layer, 'self_attn'):
            attn_class = type(first_layer.self_attn).__name__
            print(f"   → Attention 클래스: {attn_class}")
            
            # Flash Attention 관련 속성 확인
            if hasattr(first_layer.self_attn, 'flash_attn_cuda'):
                print(f"   → flash_attn_cuda 속성: {hasattr(first_layer.self_attn, 'flash_attn_cuda')}")
            
            # 모듈 이름 확인
            print(f"   → 모듈: {first_layer.self_attn.__class__.__module__}")
    
    del model_with_fa
    torch.cuda.empty_cache()
    
except Exception as e:
    print(f"   ❌ 오류: {e}")
    import traceback
    traceback.print_exc()

# ModelLoader를 통한 로드 확인
print("\n3. ModelLoader를 통한 모델 로드 확인:")
try:
    from api.infrastructure.models.model_loader import ModelLoader
    
    loader = ModelLoader()
    print("   ModelLoader로 모델 로드 중...")
    
    model_loader, tokenizer_loader = loader.load_llm_model(
        model_name=MODEL_NAME,
        device_map=DEVICE,
        use_quantization=False
    )
    
    attn_impl_loader = getattr(model_loader.config, 'attn_implementation', None)
    print(f"   → attn_implementation: {attn_impl_loader}")
    
    # 실제 attention 레이어 확인
    if hasattr(model_loader, 'model') and hasattr(model_loader.model, 'layers'):
        first_layer = model_loader.model.layers[0]
        if hasattr(first_layer, 'self_attn'):
            attn_class = type(first_layer.self_attn).__name__
            print(f"   → Attention 클래스: {attn_class}")
            print(f"   → 모듈: {first_layer.self_attn.__class__.__module__}")
    
    del model_loader, tokenizer_loader
    torch.cuda.empty_cache()
    
except Exception as e:
    print(f"   ❌ 오류: {e}")
    import traceback
    traceback.print_exc()

# 모델 지원 여부 확인
print("\n4. 모델의 Flash Attention 지원 여부 확인:")
try:
    from transformers import AutoConfig
    
    config = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)
    
    # 지원되는 attention 구현 확인
    if hasattr(config, '_attn_implementation'):
        print(f"   → _attn_implementation: {config._attn_implementation}")
    
    # config의 attn_implementation 필드 확인
    if hasattr(config, 'attn_implementation'):
        print(f"   → attn_implementation 필드: {config.attn_implementation}")
    
    # 지원 가능한 attention 구현 목록
    print(f"   → 모델 타입: {config.model_type}")
    print(f"   → 아키텍처: {config.architectures}")
    
    # Qwen 모델의 경우
    if 'Qwen' in str(config.architectures):
        print("   → Qwen 모델은 Flash Attention을 지원합니다.")
    
except Exception as e:
    print(f"   ❌ 오류: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("확인 완료")
print("=" * 70)

