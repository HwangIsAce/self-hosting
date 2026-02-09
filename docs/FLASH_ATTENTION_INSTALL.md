# Flash Attention 2 설치 및 적용 가이드

## 현재 환경
- Python: 3.11.10 ✅
- PyTorch: 2.4.1+cu124
- CUDA: 12.4 (시스템 및 PyTorch)
- Flash Attention: 2.8.3 ✅ (이미 설치됨)
- 패키지 관리: uv

## 🚀 빠른 설치 방법 (치트키 - 컴파일 피하기)

### ⚡ 방법 1: 미리 빌드된 wheel 사용 (가장 빠름, 5분 컷)

**조건:**
- NVIDIA GPU
- CUDA 11.8 또는 12.1/12.4
- Python 3.9 ~ 3.11
- PyTorch 공식 CUDA wheel 사용

```bash
# uv 환경 활성화
source $HOME/.local/bin/env

# 1. PyTorch 먼저 (CUDA 맞춰서)
# CUDA 12.1 예시:
uv pip install torch --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8:
# uv pip install torch --index-url https://download.pytorch.org/whl/cu118

# 2. flash-attn wheel만 받기 (컴파일 0초!)
pip install flash-attn --prefer-binary

# 또는 uv 환경에 직접 설치
uv pip install flash-attn
```

👉 **이게 성공하면 컴파일 0초!**  
👉 실패하면 다음 방법으로 넘어가면 됨

### ⚡ 방법 1-1: kingbri1의 Pre-built Wheel 직접 다운로드 (환경이 정확히 맞을 때)

**조건:**
- Python 3.10, 3.11, 3.12, 또는 3.13
- PyTorch 2.7.0, 2.8.0, 또는 2.9.0
- CUDA 12.8
- Linux x86_64 또는 Windows

**릴리즈 페이지:** https://github.com/kingbri1/flash-attention/releases

```bash
# 1. 현재 환경 확인
python3 --version
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}')"

# 2. 적절한 wheel 파일 다운로드
# 예시: Python 3.11, PyTorch 2.8.0, CUDA 12.8, Linux
wget https://github.com/kingbri1/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu128torch2.8.0cxx11abiFALSE-cp311-cp311-linux_x86_64.whl

# 3. 직접 설치
pip install flash_attn-2.8.3+cu128torch2.8.0cxx11abiFALSE-cp311-cp311-linux_x86_64.whl

# 또는 uv 사용
uv pip install flash_attn-2.8.3+cu128torch2.8.0cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
```

⚠️ **주의사항:**
- PyTorch 버전과 CUDA 버전이 정확히 일치해야 함
- 버전이 다르면 호환성 문제 발생 가능
- 일반적으로 `pip install flash-attn --prefer-binary`가 더 안전하고 편리함

💡 **언제 사용하나요?**
- 공식 wheel이 없을 때
- 특정 PyTorch/CUDA 버전 조합이 필요할 때
- 빌드 시간을 완전히 피하고 싶을 때

### ⚡ 방법 2: Conda wheel 경유 (실무에서 제일 안정적)

uv는 유지하되, wheel만 conda에서 빼오는 방법:

```bash
# 1. Conda 환경에서 flash-attn 설치
conda create -n fa python=3.10 -y
conda activate fa
conda install -c nvidia -c pytorch flash-attn -y

# 2. wheel 위치 찾기
python - <<EOF
import flash_attn, os
print(os.path.dirname(flash_attn.__file__))
EOF

# 3. uv 환경에 설치
source $HOME/.local/bin/env
uv pip install flash-attn --no-deps
```

💡 **대기업 실무에서 실제로 쓰는 방식 (빌드 지옥 회피용)**

### ⚡ 방법 3: Docker (가장 확실, 로컬 무관)

```bash
docker run --gpus all -it pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime bash
pip install flash-attn
```

→ **100% 성공**  
→ uv 필요 없음

---

## 기존 설치 방법 (소스 빌드 - 느림)

### 방법 4: uv를 통한 소스 빌드 (15-30분 소요)

```bash
# uv 환경 활성화
source $HOME/.local/bin/env

# CUDA 환경 변수 설정
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# 메모리 절약을 위해 병렬 작업 수 제한
export MAX_JOBS=2

# Flash Attention 설치 (빌드 시간이 오래 걸릴 수 있음)
uv pip install flash-attn --no-build-isolation
```

### 방법 5: 소스에서 직접 빌드 (최후의 수단)

```bash
# 필요한 빌드 도구 설치
apt-get update && apt-get install -y ninja-build

# Flash Attention 소스에서 빌드
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
MAX_JOBS=4 python setup.py install
```

## 설치 확인

```python
python -c "import flash_attn; print(f'Flash Attention 버전: {flash_attn.__version__}')"
```

## 적용 방법

### Transformers 라이브러리 사용 (권장)

Hugging Face Transformers에서 모델 로드 시 `attn_implementation="flash_attention_2"` 파라미터를 추가합니다.

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    device_map="auto",
    torch_dtype=torch.float16,
    attn_implementation="flash_attention_2"  # Flash Attention 2 활성화
)
```

### 코드 적용 위치

1. **LLM 모델**: `api/infrastructure/models/model_loader.py`의 `load_llm_model` 메서드
2. **VLM 모델**: `api/infrastructure/models/model_loader.py`의 `load_vlm_model` 메서드
3. **OCR 모델**: `api/infrastructure/models/ocr_engine.py`의 `_ensure_model_loaded` 메서드

## 성능 개선 효과

- **메모리 사용량**: 30-50% 감소
- **처리 속도**: 2-3배 향상 (긴 시퀀스에서 더 효과적)
- **최대 시퀀스 길이**: 메모리 제약 없이 더 긴 컨텍스트 처리 가능

## 주의사항

1. **빌드 시간**: 첫 설치 시 10-30분 소요 가능
2. **CUDA 버전**: PyTorch CUDA 버전과 시스템 CUDA 버전이 다를 수 있음
3. **모델 호환성**: 모든 모델이 Flash Attention을 지원하는 것은 아님
4. **메모리**: 설치 시 충분한 메모리 필요 (최소 8GB RAM 권장)

## 문제 해결

### 빌드 실패 시

```bash
# 빌드 캐시 정리
rm -rf ~/.cache/pip
rm -rf build/ dist/ *.egg-info

# 재시도
uv pip install flash-attn --no-build-isolation --verbose
```

### CUDA 버전 불일치

```bash
# PyTorch CUDA 버전 확인
python -c "import torch; print(torch.version.cuda)"

# 시스템 CUDA 버전 확인
nvcc --version

# 불일치 시 PyTorch 재설치 고려
```

### ImportError 발생 시

```bash
# Flash Attention이 제대로 설치되었는지 확인
python -c "import flash_attn; print(flash_attn.__file__)"

# 경로 확인 후 필요시 PYTHONPATH 설정
```

## 참고 자료

- [Flash Attention 공식 GitHub](https://github.com/Dao-AILab/flash-attention)
- [Hugging Face Transformers 문서](https://huggingface.co/docs/transformers/ko/llm_optims)
- [Flash Attention 논문](https://arxiv.org/abs/2205.14135)

