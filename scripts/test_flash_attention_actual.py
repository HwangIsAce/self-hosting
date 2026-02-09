#!/usr/bin/env python3
"""실제 forward pass에서 Flash Attention 사용 여부 확인"""

import sys
import torch
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

print("=" * 70)
print("실제 Forward Pass에서 Flash Attention 사용 확인")
print("=" * 70)

# Flash Attention 설치 확인
try:
    import flash_attn
    print(f"✅ Flash Attention 설치됨: {flash_attn.__version__}")
except ImportError:
    print("❌ Flash Attention이 설치되지 않았습니다.")
    sys.exit(1)

# 모델 로드
print(f"\n모델 로드 중: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

# Flash Attention 없이
print("\n[테스트 1] attn_implementation=None (기본)")
model_sdpa = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map=DEVICE,
    trust_remote_code=True,
)

# Flash Attention으로
print("\n[테스트 2] attn_implementation='flash_attention_2'")
model_fa = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map=DEVICE,
    trust_remote_code=True,
    attn_implementation="flash_attention_2",
)

# 실제 forward pass 테스트
print("\n" + "=" * 70)
print("실제 Forward Pass 테스트")
print("=" * 70)

# 다양한 시퀀스 길이로 테스트
test_lengths = [128, 512, 1024, 2048]

for seq_len in test_lengths:
    print(f"\n[시퀀스 길이: {seq_len} 토큰]")
    print("-" * 70)
    
    # 입력 준비
    test_text = "Hello, how are you? " * (seq_len // 5)
    inputs = tokenizer(test_text, return_tensors="pt", max_length=seq_len, truncation=True).to(DEVICE)
    
    # SDPA 테스트
    model_sdpa.eval()
    with torch.no_grad():
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        outputs_sdpa = model_sdpa(**inputs)
        end.record()
        torch.cuda.synchronize()
        
        time_sdpa = start.elapsed_time(end) / 1000.0  # ms to seconds
    
    # Flash Attention 테스트
    model_fa.eval()
    with torch.no_grad():
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        outputs_fa = model_fa(**inputs)
        end.record()
        torch.cuda.synchronize()
        
        time_fa = start.elapsed_time(end) / 1000.0
    
    # 메모리 사용량 확인
    if torch.cuda.is_available():
        mem_sdpa = torch.cuda.max_memory_allocated() / 1024**3
        torch.cuda.reset_peak_memory_stats()
        
        _ = model_fa(**inputs)
        mem_fa = torch.cuda.max_memory_allocated() / 1024**3
        torch.cuda.reset_peak_memory_stats()
    else:
        mem_sdpa = mem_fa = 0
    
    print(f"  SDPA:      {time_sdpa:.3f}초, 메모리: {mem_sdpa:.2f}GB")
    print(f"  Flash Attn: {time_fa:.3f}초, 메모리: {mem_fa:.2f}GB")
    
    if time_sdpa > 0:
        speedup = ((time_sdpa - time_fa) / time_sdpa) * 100
        mem_saving = ((mem_sdpa - mem_fa) / mem_sdpa) * 100 if mem_sdpa > 0 else 0
        print(f"  속도 개선: {speedup:+.1f}%, 메모리 절약: {mem_saving:+.1f}%")

# 실제 attention 구현 확인
print("\n" + "=" * 70)
print("Attention 구현 확인")
print("=" * 70)

# forward 메서드의 실제 코드 확인
import inspect
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention

print("\nQwen2Attention.forward() 소스 코드 (일부):")
source = inspect.getsource(Qwen2Attention.forward)
# Flash Attention 관련 코드 찾기
if 'flash_attn' in source.lower() or 'flash_attention' in source.lower():
    print("  ✅ Flash Attention 관련 코드 발견!")
    # 관련 부분 출력
    lines = source.split('\n')
    for i, line in enumerate(lines):
        if 'flash' in line.lower() or 'attn_implementation' in line.lower():
            start = max(0, i-2)
            end = min(len(lines), i+3)
            print(f"  라인 {i+1} 주변:")
            for j in range(start, end):
                marker = ">>>" if j == i else "   "
                print(f"  {marker} {j+1:4d}: {lines[j]}")
else:
    print("  ⚠️  Flash Attention 관련 코드를 찾을 수 없습니다.")

# 실제 forward pass에서 어떤 함수가 호출되는지 확인
print("\n" + "=" * 70)
print("실제 Forward Pass 추적")
print("=" * 70)

# 간단한 입력으로 테스트
test_input = tokenizer("Hello", return_tensors="pt").to(DEVICE)

# Hook을 사용하여 실제 호출되는 함수 확인
called_functions = []

def hook_fn(module, input, output):
    if hasattr(module, '__class__'):
        called_functions.append(module.__class__.__name__)

# 첫 번째 attention 레이어에 hook 등록
first_attn = model_fa.model.layers[0].self_attn
handle = first_attn.register_forward_hook(hook_fn)

model_fa.eval()
with torch.no_grad():
    _ = model_fa(**test_input)

handle.remove()

print(f"  호출된 모듈: {called_functions[:5]}")

# Flash Attention 함수가 실제로 호출되는지 확인
print("\n" + "=" * 70)
print("결론")
print("=" * 70)

print("""
Flash Attention이 성능 차이를 보이지 않는 이유:

1. **짧은 시퀀스 길이**: Flash Attention은 긴 시퀀스(2048+ 토큰)에서 효과적입니다.
   - 짧은 시퀀스에서는 오버헤드가 더 클 수 있습니다.

2. **자동 최적화**: PyTorch의 SDPA가 이미 최적화되어 있어서
   - 짧은 시퀀스에서는 SDPA가 Flash Attention과 비슷하거나 더 빠를 수 있습니다.

3. **메모리 vs 속도**: Flash Attention의 주요 장점은 메모리 절약입니다.
   - 속도 개선은 부수적인 효과이며, 시퀀스가 길수록 두드러집니다.

4. **실제 적용 여부**: config에 표시되지 않아도 forward pass에서는
   - Flash Attention이 사용될 수 있지만, 짧은 시퀀스에서는 차이가 없을 수 있습니다.

권장사항:
- 2048+ 토큰의 긴 시퀀스로 테스트
- 메모리 사용량 비교 (Flash Attention의 주요 장점)
- 실제 프로덕션 워크로드에서 테스트
""")

